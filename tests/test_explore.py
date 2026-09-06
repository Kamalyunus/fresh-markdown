"""engine.explore: the spread ledger, the exploration budget and its tau
controller, and the budget sweep."""

import numpy as np
import pytest

from conftest import CFG
from engine import dp as dp_mod
from engine import explore
from engine.explore import SpreadLedger, tau_next


def test_solve_tau_lands_just_under_budget():
    rng = np.random.default_rng(0)
    led = SpreadLedger()
    for i in range(400):
        led.add(f"2026-08-{1 + i % 4:02d}", rng.lognormal(6, 1, rng.integers(2, 12)))
    budget = 4000.0
    tau = led.solve_tau(budget, n_days=4)
    spend = led.implied_daily_spend(tau, 4)
    assert spend <= budget
    assert spend > 0.95 * budget          # under, but not by much


def test_spend_is_monotone_in_tau():
    rng = np.random.default_rng(1)
    led = SpreadLedger()
    for i in range(200):
        led.add("d", rng.lognormal(5, 1, 6))
    seq = [led.implied_daily_spend(t, 1) for t in range(0, 2000, 100)]
    assert seq == sorted(seq)
    assert led.implied_daily_spend(0.0, 1) == 0.0


def test_spend_by_day_sums_to_the_total():
    rng = np.random.default_rng(2)
    led = SpreadLedger()
    for i in range(120):
        led.add(f"d{i % 3}", rng.lognormal(5, 1, 4))
    tau = 500.0
    by_day = led.spend_by_day(tau)
    assert len(by_day) == 3
    assert by_day.sum() == pytest.approx(led.implied_daily_spend(tau, 1))


def test_chunked_flush_does_not_lose_costs():
    rng = np.random.default_rng(3)
    a, b = SpreadLedger(), SpreadLedger()
    b._FLUSH = 64                       # force many chunks
    for _ in range(200):
        c = rng.lognormal(5, 1, 5)
        a.add("d", c)
        b.add("d", c)
    assert a.decisions == b.decisions == 200
    assert a.implied_daily_spend(1e12, 1) == pytest.approx(
        b.implied_daily_spend(1e12, 1))


def test_empty_ledger_is_not_a_crash():
    led = SpreadLedger()
    assert led.decisions == 0
    assert led.solve_tau(100.0) is None
    assert led.distribution() == {}
    assert led.quantile_of(10.0) is None


def test_entry_only_collection_understates_the_funded_tau():
    """The bug this class exists to prevent, reproduced."""
    rng = np.random.default_rng(4)
    entry_only, all_hours = SpreadLedger(), SpreadLedger()
    for ep in range(300):
        for hour in range(8):
            costs = rng.lognormal(6, 0.8, 6)
            all_hours.add("d", costs)
            if hour == 0:
                entry_only.add("d", costs)

    budget = 20000.0
    tau_entry = entry_only.solve_tau(budget, n_days=1)
    tau_all = all_hours.solve_tau(budget, n_days=1)
    assert tau_entry > tau_all
    # launching at the entry-only tau overspends on the real decision count
    assert all_hours.implied_daily_spend(tau_entry, 1) > budget


def test_budget_scales_down_as_posterior_narrows():
    wide = explore.budget_today(1e6, CFG["exploration"]["budget_scale_ref_std"], CFG)
    narrow = explore.budget_today(1e6, 0.0, CFG)
    assert wide == pytest.approx(CFG["exploration"]["budget_share_of_il"] * 1e6)
    assert narrow == pytest.approx(wide * CFG["exploration"]["budget_scale_floor"])


def test_tau_next_clipped():
    """(tau_after, clipped): the flag is the step's own verdict, so a
    reader never infers the bound from rounded taus."""
    lo, hi = CFG["exploration"]["tau_adjust_clip"]
    assert explore.tau_next(100.0, 1e9, 1.0, CFG) == (pytest.approx(100.0 * hi), True)
    assert explore.tau_next(100.0, 0.0, 1e9, CFG) == (pytest.approx(100.0 * lo), True)
    inside = (lo + hi) / 2
    assert explore.tau_next(100.0, inside * 500.0, 500.0, CFG) == \
        (pytest.approx(100.0 * inside), False)
    # and the walk carries it per row
    base = {"2026-07-26": 1e9}                 # a full window behind the day
    _, rows = explore.walk_tau(100.0, ["2026-08-02"], lambda d, t: 1.0,
                               base, 1.0, CFG)
    assert rows[0]["clipped"] is True and rows[0]["tau_after"] == pytest.approx(100.0 * hi)


def test_controller_cannot_correct_before_it_has_seen_a_day(cfg):
    """tau_next only ever reads the day just closed, so day 1 is spent at the
    launch tau whatever that is -- which is why the trace exists."""
    tau0 = 10_000.0
    budget, spend = 1_000.0, 8_700.0      # an 8.7x overspend, well past the clip
    lo = cfg["exploration"]["tau_adjust_clip"][0]
    stop_at = 2.0                         # pinned: the count below is sized to it
    assert tau_next(tau0, budget, spend, cfg)[0] == tau0 * lo    # clip floor
    # three halvings to get under a 2.0x stop, if spend fell proportionally
    tau, over = tau0, spend / budget
    days_over = 0
    for _ in range(5):
        if over > stop_at:
            days_over += 1
        tau, _ = tau_next(tau, budget, budget * over, cfg)
        over = over / 2
    assert days_over >= 3


def test_the_controller_holds_tau_on_a_zero_budget(cfg):
    """A zero budget from an EMPTY trailing history is an absence of signal,
    not an overspend: the controller must hold tau, not halve it. The moment
    history exists, calibration resumes."""
    from evaluate.shadow import _controller_trace

    led = SpreadLedger()
    led.add("2026-08-04", [50.0, 80.0])          # spend exists on day 1
    led.add("2026-08-05", [50.0, 80.0])
    # no IL history at all for day 1; day 2 sees day 1's realised IL
    il_by_day = {"2026-08-04": 1_000_000.0}
    trace = _controller_trace(led, il_by_day, tau0=100.0, widest_std=1.0,
                              cfg=cfg)
    d1, d2 = trace["by_day"][0], trace["by_day"][1]
    assert d1["budget"] == 0.0 and d1["over_budget"] is None
    assert d2["tau"] == pytest.approx(100.0), \
        "tau moved on a day whose budget was 0 -- empty history is not signal"
    assert d2["budget"] > 0


def test_budget_sweep_trades_count_for_depth_from_one_ledger():
    """The forced RATE is the budget's (tau); the delta_min multiple sets
    which tiers are drawn. From one ledger, a smaller share forces less at
    the same depth and a deeper multiple forces less but deeper -- so the
    owner reads the table instead of re-running shadow per candidate. A
    multiple below the one in force cannot be recovered and says so."""
    rng = np.random.default_rng(5)
    led = SpreadLedger()
    n_dec = 600
    for i in range(n_dec):
        moves = np.sort(rng.uniform(0.05, 0.6, 6))        # distance from ref
        costs = 40.0 * moves ** 2 * rng.lognormal(6, 0.3)  # deeper = dearer
        keep = moves >= 0.10                                # the floor in force
        led.add(f"d{i % 5}", costs[keep], moves[keep], delta_min=0.10)
    sw = led.sweep(daily_budget=20000.0, n_days=5, n_decisions=n_dec,
                   share_in_force=0.01, multiple_in_force=1.0,
                   shares=[0.0025, 0.005, 0.01], multiples=[0.5, 1.0, 2.0])
    rows = {(r["budget_share_of_il"], r["delta_min_bias_multiple"]): r
            for r in sw["rows"]}
    ref = rows[(0.01, 1.0)]
    assert ref["in_force"] and ref["information_rel"] == 1.0
    assert ref["implied_daily_spend"] <= ref["daily_budget"] == 20000.0
    # less budget: lower rate, same depth (within the draw's noise)
    half, quarter = rows[(0.005, 1.0)], rows[(0.0025, 1.0)]
    assert quarter["forced_rate"] < half["forced_rate"] < ref["forced_rate"]
    assert half["daily_budget"] == 10000.0
    assert abs(half["mean_log_move_forced"] - ref["mean_log_move_forced"]) < 0.05
    assert quarter["information_rel"] < half["information_rel"] < 1.0
    # deeper floor at the same budget: fewer, deeper forced moves
    deep = rows[(0.01, 2.0)]
    assert deep["forced_rate"] < ref["forced_rate"]
    assert deep["mean_log_move_forced"] > ref["mean_log_move_forced"] + 0.05
    # a shallower floor than recorded is not a number
    assert "forced_rate" not in rows[(0.01, 0.5)] and "never recorded" in rows[(0.01, 0.5)]["note"]
    # with no floor in force the multiples are inert, and the table says so
    flat = SpreadLedger()
    for i in range(100):
        flat.add("d", rng.lognormal(5, 1, 4))
    sw0 = flat.sweep(1000.0, 1, 100, 0.01, 1.0, [0.01], [1.0, 2.0])
    assert sw0["delta_min_in_force"] is False and "reads the same" in sw0["note"]
    a, b = (r for r in sw0["rows"])
    assert a["forced_rate"] == b["forced_rate"]


def _reference_per_decision(led, tau, weights=None, keep=None):
    """SpreadLedger._per_decision as it stood: a masked fancy-index gather
    (three copies) per call, kept as the reference for the one-pass bincount."""
    led._build()
    n_dec = len(led._lens)
    m = led._costs <= tau
    if keep is not None:
        m &= keep
    w = led._costs if weights is None else weights
    sums = np.bincount(led._dec_of[m], weights=w[m], minlength=n_dec)
    cnts = np.bincount(led._dec_of[m], minlength=n_dec)
    return np.divide(sums, cnts, out=np.zeros(n_dec), where=cnts > 0), cnts


def test_the_one_pass_ledger_reads_exactly_what_the_masked_gather_read():
    """Every bisection step is one bincount over the flat costs now, no
    per-step copies. A masked-out tier adds exactly 0.0 to its decision's
    sum in the same order, so the result is the same to the bit -- checked
    against the old gather at every tau the bisection visits, with and
    without a keep mask and with each weight vector the sweep uses."""
    rng = np.random.default_rng(6)
    led = SpreadLedger()
    for i in range(3000):
        n = int(rng.integers(1, 9))
        moves = np.sort(rng.uniform(0.02, 0.7, n))
        led.add(f"d{i % 7}", 40.0 * moves ** 2 * rng.lognormal(6, 0.4), moves,
                delta_min=0.02)
    led._build()
    keep = led._moves >= 0.2
    taus = [0.0, *np.quantile(led._costs, [0.001, 0.05, 0.3, 0.5, 0.9, 0.999]),
            float(led._costs.max()) * 2]
    for tau in taus:
        for weights in (None, led._moves, led._moves ** 2):
            for mask in (None, keep):
                got, got_n = led._per_decision(tau, weights=weights, keep=mask)
                want, want_n = _reference_per_decision(led, tau, weights=weights, keep=mask)
                assert np.array_equal(got, want) and np.array_equal(got_n, want_n), tau
    # and the bisection lands on the same tau
    budget = 0.4 * led.implied_daily_spend(led._costs.max(), 7)
    tau = led.solve_tau(budget, n_days=7)
    lo, hi = 0.0, float(led._costs.max())
    for _ in range(60):
        mid = (lo + hi) / 2
        per_dec, _ = _reference_per_decision(led, mid)
        if np.bincount(led._dec_day, weights=per_dec, minlength=7).sum() / 7 < budget:
            lo = mid
        else:
            hi = mid
    assert tau == lo


def test_add_query_add_keeps_every_index_aligned():
    """Querying between adds rebuilds the flat arrays over the FULL history:
    a ledger fed in two halves with a read between them answers exactly as
    one fed in a single pass."""
    rng = np.random.default_rng(7)
    rows = [(f"d{i % 3}", rng.lognormal(5, 1, int(rng.integers(1, 6)))) for i in range(300)]
    one, split = SpreadLedger(), SpreadLedger()
    split._FLUSH = 50
    for day, costs in rows:
        one.add(day, costs)
    for day, costs in rows[:150]:
        split.add(day, costs)
    mid_read = split.spend_by_day(200.0)              # forces a build mid-history
    assert len(mid_read) == 3
    for day, costs in rows[150:]:
        split.add(day, costs)
    for tau in (0.0, 100.0, 300.0, 1e9):
        assert np.array_equal(one.spend_by_day(tau), split.spend_by_day(tau))
    assert one.solve_tau(5000.0, 3) == split.solve_tau(5000.0, 3)
    assert one.decisions == split.decisions == 300


def test_a_decision_prices_its_admissible_tiers_once():
    """admissible_costs is the one table: the chooser, the ledger record and
    the assurance reconstruction read the same costs whether they build the
    table themselves or take the one decide() built before the draw."""
    res = dp_mod.solve(10000.0, 4000.0, 6, [0.8] * 6, 0.30, -1.0, 0.9, CFG,
                       anchor_discount=0.0, entry=False)      # the full grid
    dmin = 0.10
    costs = explore.admissible_costs(res, dmin)
    assert list(costs) == explore.admissible(res, dmin)
    assert res.optimal_index not in costs and len(costs) >= 3
    assert explore.affordable_set(res, 1e12, dmin) == explore.affordable_set(res, 1e12, costs=costs)
    assert explore.spread_table(res, dmin) == explore.spread_table(res, costs=costs)
    assert explore.spread_costs(res, dmin) == explore.spread_table(res, dmin)[0] == list(costs.values())
    tau = sorted(costs.values())[1]                   # two tiers affordable
    for seed in range(5):
        by_floor = explore.select(res, tau, np.random.default_rng(seed), delta_min=dmin)
        by_table = explore.select(res, tau, np.random.default_rng(seed), costs=costs)
        assert by_floor == by_table and by_floor["affordable_set_size"] == 2


def test_the_sweep_refuses_a_zero_share_multiple_or_decision_count():
    """A zero in-force share or multiple divides the grid; a zero decision
    count divides the forced rate. Each is a note, never a ZeroDivisionError
    or an inf in the report."""
    led = explore.SpreadLedger()
    led.add("2026-08-19", [10.0, 20.0], [0.1, 0.2], 0.05)
    good = led.sweep(100.0, 1, 2, 0.01, 1.0, [0.01], [1.0])
    assert "rows" in good
    for share, mult, n in ((0.0, 1.0, 2), (0.01, 0.0, 2), (0.01, 1.0, 0),
                           (None, 1.0, 2)):
        out = led.sweep(100.0, 1, n, share, mult, [0.01], [1.0])
        assert "note" in out and "rows" not in out, (share, mult, n)


def test_the_ledger_refuses_moves_that_do_not_align_with_costs():
    """`moves` are the same tiers' log distances, one per cost; a mismatch
    silently mis-aligned the sweep's keep mask against the costs."""
    led = SpreadLedger()
    with pytest.raises(ValueError, match="aligned"):
        led.add("d", [10.0, 20.0, 30.0], [0.1, 0.2])
    led.add("d", [10.0, 20.0], [0.1, 0.2])
    led.add("d", [10.0, 20.0])                     # no moves: zeros, allowed
    assert led.decisions == 2


def test_the_sweep_finds_the_in_force_cell_by_closeness_not_exact_key():
    """The in-force (share, multiple) came back rounded on its way into the
    grid, so an exact-key lookup found no reference and every row lost its
    information_rel. The same isclose test that labels `in_force` finds it."""
    rng = np.random.default_rng(8)
    led = SpreadLedger()
    for i in range(200):
        moves = np.sort(rng.uniform(0.05, 0.6, 5))
        led.add(f"d{i % 4}", 40.0 * moves ** 2 * rng.lognormal(6, 0.3), moves,
                delta_min=0.05)
    share = 0.01
    sw = led.sweep(daily_budget=5000.0, n_days=4, n_decisions=200,
                   share_in_force=share, multiple_in_force=1.0,
                   shares=[share * (1 + 1e-12), 0.005], multiples=[1.0])
    rows = {r["budget_share_of_il"]: r for r in sw["rows"]}
    assert rows[share * (1 + 1e-12)]["in_force"]
    assert rows[share * (1 + 1e-12)]["information_rel"] == 1.0
    assert 0 < rows[0.005]["information_rel"] < 1.0


def test_the_controller_holds_tau_until_the_il_base_spans_its_window(cfg):
    """A pilot's first mornings hold a handful of early closers, not the
    shop's IL: a budget priced from them is tiny, and on the owner's
    rehearsal the overspend stop fired on day three. Until the base reaches
    back a whole budget_il_window_days the day is held (an absence of
    signal, like a zero budget), and the row says why."""
    from engine.explore import budget_base_ready, walk_tau

    cfg["exploration"]["budget_il_window_days"] = 3
    il = {"2026-09-01": 1000.0, "2026-09-02": 1000.0, "2026-09-03": 1000.0,
          "2026-09-04": 1000.0}
    assert not budget_base_ready({}, "2026-09-02", cfg)
    assert not budget_base_ready(il, "2026-09-03", cfg)       # 2 days back
    assert budget_base_ready(il, "2026-09-04", cfg)           # 3 days back
    tau, rows = walk_tau(100.0, ["2026-09-02", "2026-09-03", "2026-09-04"],
                         lambda day, t: 1e6, il, 1.0, cfg)     # wildly over budget
    assert [r["held"] for r in rows] == [
        "IL base shorter than budget_il_window_days",
        "IL base shorter than budget_il_window_days", None]
    assert rows[0]["budget"] > 0 and rows[0]["tau_after"] == 100.0
    assert rows[2]["tau_after"] == pytest.approx(50.0) and not rows[2]["held"]
    assert tau == pytest.approx(50.0)
    # a zero budget still holds, and says so
    _, empty = walk_tau(100.0, ["2026-09-02"], lambda d, t: 5.0, {}, 1.0, cfg)
    assert empty[0]["held"] == "no trailing IL" and empty[0]["tau_after"] == 100.0
