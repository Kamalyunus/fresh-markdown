"""pricing.explore -- IL-budgeted exploration (design section 5.8).

The DP already computes Q(p) for every feasible price; exploration is a
constrained selection over those same values:

    p_star     = argmax_p Q(p)
    cost(p)    = Q(p_star) - Q(p)          expected IL loss, in currency
    affordable = { p : cost(p) <= tau, p != p_star }

If affordable is non-empty, select UNIFORMLY AT RANDOM from it. Uniform
selection is not a detail -- it is the randomisation that makes the outcome
clean evidence; any state-dependent choice of forced price reintroduces the
endogeneity that makes legacy history unusable.

tau is a CURRENCY amount, compared against Q(p_star) - Q(p) in won. There is
no exploration probability schedule, base rate, floor, ceiling, or cold-start
std.
"""

import math

import numpy as np
import pandas as pd

# float noise on a LOG price move (|log((1-d)/(1-d_ref))|). Same magnitude as
# pricing.dp.TIER_EPS, but a log move is not a tier: the two comparisons are
# kept apart so a change to the grid epsilon cannot silently move the
# admissibility floor.
LOG_EPS = 1e-9


def delta_min(cfg, eps, category=None):
    """The smallest INFORMATIVE log price move FROM THE REFERENCE, per cell
    (design 5.8).

    The learner reads every forced outcome against mu_ref at the reference
    discount: L = log((1-d)/(1-d_ref)) and the signal is eps*L, against a
    level bias in mu_ref of scale `delta_min_log_bias` (the largest of the
    three fidelity readings tune derives it from). Below k*bias/|eps| the
    signal sits inside the model's own level error and the outcome teaches
    nothing about eps. Cost is measured from p*; information from d_ref --
    a tier far from p* but at d_ref costs money and teaches nothing, which
    is why the floor is on the reference distance. 0 while the bias scale
    is null (nothing pasted yet).
    """
    ec = cfg["exploration"]
    bias = ec.get("delta_min_log_bias")
    if isinstance(bias, dict):
        # per category, `_default` for one the backtest never saw -- the
        # same key convention as reference_discount ('SIDE DISH' -> SIDE_DISH)
        key = str(category).replace(" ", "_") if category is not None else "_default"
        if key not in bias and "_default" not in bias:
            raise KeyError(
                f"exploration.delta_min_log_bias has no entry for {key!r} and "
                "no `_default`: a per-category floor mapping must name every "
                "priced category or carry `_default` (pipeline.tune writes "
                "both). A missing key is not 'no floor'.")
        bias = bias.get(key, bias.get("_default"))
    if not bias:
        return 0.0
    floor = abs(float(cfg["posterior"]["epsilon_max"]))       # the sign constraint
    return (float(ec["delta_min_bias_multiple"]) * float(bias)
            / max(abs(float(eps)), floor))


def log_move(d_from, d_to):
    """|log((1-d_to)/(1-d_from))|: the price move in log space."""
    return abs(math.log((1.0 - d_to) / (1.0 - d_from)))


def admissible(dp_result, delta_min=0.0):
    """Non-optimal tiers at least `delta_min` from the REFERENCE discount in
    log price -- the tiers a perturbation may land on before the budget is
    asked. ONE definition: the chooser, the spread ledger and the assurance
    check all read it, so tau is calibrated against exactly the set it is
    spent on."""
    star = dp_result.optimal_index
    d_ref = dp_result.d_ref
    return [j for j in dp_result.q_by_tier if j != star
            and (delta_min <= 0
                 or log_move(d_ref, dp_result.tiers[j]) >= delta_min - LOG_EPS)]


def tier_costs(dp_result):
    """{tier index: Q(p_star) - Q(p)} over the actions allowed now -- the
    expected IL each non-optimal action forgoes, in currency."""
    q = dp_result.q_by_tier
    q_star = q[dp_result.optimal_index]
    return {j: q_star - q[j] for j in q}


def affordable_set(dp_result, tau, delta_min=0.0):
    """Tier indices a perturbation may legally land on, and every tier's
    cost. Public: `pipeline.assurance` reconstructs this exact set to test
    that the draw was uniform."""
    costs = tier_costs(dp_result)
    return [j for j in admissible(dp_result, delta_min) if costs[j] <= tau], costs


def spread_table(dp_result, delta_min=0.0):
    """(costs, log moves from the reference) over the admissible tiers -- the
    Q-spreads the ledger prices tau against, so a tau solved here funds the
    draws production will make; the moves let the ledger re-judge
    admissibility at a deeper floor (SpreadLedger.sweep)."""
    costs = tier_costs(dp_result)
    kept = admissible(dp_result, delta_min)
    return ([costs[j] for j in kept],
            [log_move(dp_result.d_ref, dp_result.tiers[j]) for j in kept])


def spread_costs(dp_result, delta_min=0.0):
    return spread_table(dp_result, delta_min)[0]


def select(dp_result, tau, rng, explorable=True, delta_min=0.0):
    """Returns a dict describing the chosen action.

    explorable=False marks a structurally non-explorable episode (fewer than
    min_feasible_tiers): it is priced by the DP as normal but excluded from
    the exploration budget and never logged as a blocked attempt.
    """
    star = dp_result.optimal_index
    choice = {
        "optimal_index": star,
        "chosen_index": star,
        "is_exploration": False,
        "exploration_cost": 0.0,
        "affordable_set_size": 0,
    }
    if not explorable or tau is None:
        return choice

    affordable, costs = affordable_set(dp_result, tau, delta_min)
    choice["affordable_set_size"] = len(affordable)
    if affordable:
        j = affordable[int(rng.integers(0, len(affordable)))]
        choice.update(chosen_index=j, is_exploration=True,
                      exploration_cost=float(costs[j]))
    return choice


class SpreadLedger:
    """Q(p_star) - Q(p) spreads, bucketed by day, and the tau they imply.

    The ONE definition of "what would tau spend", shared by replay and
    shadow -- costs recorded for EVERY decision, independent of the tau in
    force (the entry-only version of this is in docs/learnings.md). Stored
    flat and chunked: ~30M costs at full population.
    """

    _FLUSH = 1 << 20

    def __init__(self):
        self._chunks, self._buf = [], []
        self._mchunks, self._mbuf = [], []      # log moves, aligned to costs
        self._lens, self._day_of, self._day_index = [], [], {}
        self._dmin = []                          # the floor in force, per decision

    def add(self, day, costs, moves=None, delta_min=0.0):
        """Record one decision's spreads. `costs` excludes the optimum;
        `moves` are the same tiers' log distances from the reference (zeros
        when the caller has none -- the sweep then reports multiples as
        inert) and `delta_min` the floor these tiers already cleared."""
        if not len(costs):
            return
        self._buf.extend(costs)
        self._mbuf.extend(moves if moves is not None else [0.0] * len(costs))
        self._lens.append(len(costs))
        self._dmin.append(float(delta_min))
        day = str(day)
        if day not in self._day_index:
            self._day_index[day] = len(self._day_index)
        self._day_of.append(self._day_index[day])
        if len(self._buf) >= self._FLUSH:
            self._chunks.append(np.asarray(self._buf, dtype=np.float64))
            self._mchunks.append(np.asarray(self._mbuf, dtype=np.float64))
            self._buf, self._mbuf = [], []

    def _build(self):
        if self._buf:
            self._chunks.append(np.asarray(self._buf, dtype=np.float64))
            self._mchunks.append(np.asarray(self._mbuf, dtype=np.float64))
            self._buf, self._mbuf = [], []
        if getattr(self, "_costs", None) is None or self._chunks:
            # _lens/_day_of span the FULL history, so _costs must too --
            # dropping prior chunks after add()-query-add() mis-aligns every
            # index against _dec_of
            have = getattr(self, "_costs", None) is not None and len(self._costs)
            prior = [self._costs] if have else []
            mprior = [self._moves] if have else []
            self._costs = (np.concatenate(prior + self._chunks)
                           if (prior or self._chunks) else np.zeros(0))
            self._moves = (np.concatenate(mprior + self._mchunks)
                           if (mprior or self._mchunks) else np.zeros(0))
            self._chunks, self._mchunks = [], []
            lens = np.asarray(self._lens, dtype=np.int64)
            self._dec_of = np.repeat(np.arange(len(lens)), lens)
            self._dec_day = np.asarray(self._day_of, dtype=np.int64)
            self._dec_dmin = np.asarray(self._dmin, dtype=np.float64)

    @property
    def decisions(self):
        return len(self._lens)

    @property
    def days(self):
        """Day labels in first-seen order; index i is row i of spend_by_day."""
        return [d for d, _ in sorted(self._day_index.items(), key=lambda kv: kv[1])]

    def _per_decision(self, tau, weights=None, keep=None):
        """Mean of `weights` (default: cost) over each decision's affordable
        tiers at `tau`, zero for an empty set; `keep` masks tiers out."""
        self._build()
        n_dec = len(self._lens)
        m = self._costs <= tau
        if keep is not None:
            m &= keep
        w = self._costs if weights is None else weights
        sums = np.bincount(self._dec_of[m], weights=w[m], minlength=n_dec)
        cnts = np.bincount(self._dec_of[m], minlength=n_dec)
        return np.divide(sums, cnts, out=np.zeros(n_dec), where=cnts > 0), cnts

    def spend_by_day(self, tau, keep=None):
        """EXPECTED spend per day at `tau`: mean affordable cost per
        decision (uniform draw), empty sets contribute nothing. Expected,
        not realised -- the trace walks counterfactual taus."""
        self._build()
        n_dec, n_day = len(self._lens), len(self._day_index)
        if not n_dec:
            return np.zeros(n_day)
        per_dec, _ = self._per_decision(tau, keep=keep)
        return np.bincount(self._dec_day, weights=per_dec, minlength=n_day)

    def implied_daily_spend(self, tau, n_days=None, keep=None):
        by_day = self.spend_by_day(tau, keep)
        return float(by_day.sum()) / max(n_days or len(by_day), 1)

    def solve_tau(self, budget_per_day, n_days=None, steps=60, keep=None):
        """Bisect for the tau whose implied daily spend equals the budget.
        Returns the LOW end of the bracket: spend steps as costs cross tau,
        and under-budget is the right side to miss on."""
        self._build()
        costs = self._costs if keep is None else self._costs[keep]
        if not len(costs) or budget_per_day <= 0:
            return None
        lo, hi = 0.0, float(costs.max())
        if self.implied_daily_spend(hi, n_days, keep) < budget_per_day:
            return hi                       # budget exceeds even unbounded tau
        for _ in range(steps):
            mid = (lo + hi) / 2
            if self.implied_daily_spend(mid, n_days, keep) < budget_per_day:
                lo = mid
            else:
                hi = mid
        return lo

    def sweep(self, daily_budget, n_days, n_decisions, share_in_force,
              multiple_in_force, shares, multiples):
        """What the budget share and the delta_min multiple each buy, from
        THIS ledger and no re-run: for every (share, multiple) the tau the
        budget solves to, the forced rate, spend, mean move and an
        information proxy (sum over forced decisions of E[move^2], relative
        to the in-force pair -- the NB Fisher information is quadratic in the
        move, so this is the count-and-depth trade in one number). A
        multiple below the in-force one cannot be recovered (those tiers
        were never recorded) and is reported as such; with no floor in force
        (delta_min null) every multiple is inert."""
        self._build()
        n_dec = len(self._lens)
        if not n_dec or daily_budget <= 0:
            return {"note": "no spreads or no budget -- nothing to sweep"}
        if not share_in_force or share_in_force <= 0 \
                or not multiple_in_force or multiple_in_force <= 0 \
                or not n_decisions or n_decisions <= 0:
            return {"note": ("nothing to sweep against: budget_share_of_il, "
                             "delta_min_bias_multiple and the decision count "
                             "must all be positive (got "
                             f"{share_in_force!r}, {multiple_in_force!r}, "
                             f"{n_decisions!r})")}
        floor_active = bool(len(self._dec_dmin)) and float(self._dec_dmin.max()) > 0
        move_sq = self._moves ** 2

        def same(a, b):
            return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)

        def cell(share, mult):
            if mult < multiple_in_force and not same(mult, multiple_in_force):
                return {"note": "below the multiple in force -- those tiers were "
                                "never recorded; re-run shadow at that multiple"}
            if floor_active and mult > multiple_in_force and not same(mult, multiple_in_force):
                rel = mult / multiple_in_force
                keep = self._moves >= rel * self._dec_dmin[self._dec_of] - LOG_EPS
            else:
                keep = None
            budget = daily_budget * share / share_in_force
            tau = self.solve_tau(budget, n_days, keep=keep)
            if tau is None:
                return {"note": "no positive tau at this budget"}
            _, cnts = self._per_decision(tau, keep=keep)
            forced = int((cnts > 0).sum())
            mean_move, _ = self._per_decision(tau, weights=self._moves, keep=keep)
            e_sq, _ = self._per_decision(tau, weights=move_sq, keep=keep)
            return {
                "tau": round(float(tau), 2),
                "forced_rate": round(forced / n_decisions, 4),
                "forced_per_day": round(forced / max(n_days, 1), 1),
                "implied_daily_spend": round(self.implied_daily_spend(tau, n_days, keep), 1),
                "daily_budget": round(budget, 1),
                "mean_log_move_forced": round(float(mean_move[cnts > 0].mean()), 4)
                    if forced else None,
                "_info": float(e_sq.sum()),
            }

        grid = {}
        for mult in multiples:
            for share in shares:
                grid[(share, mult)] = cell(share, mult)
        ref = grid.get((share_in_force, multiple_in_force), {}).get("_info")
        rows = []
        for (share, mult), c in grid.items():
            info = c.pop("_info", None)
            rows.append({"budget_share_of_il": share, "delta_min_bias_multiple": mult,
                         "in_force": (same(share, share_in_force)
                                      and same(mult, multiple_in_force)),
                         **c,
                         **({"information_rel": round(info / ref, 3)}
                            if info is not None and ref else {})})
        return {
            "basis": ("this run's spread ledger re-solved per cell: tau is the "
                      "bisection at share x (in-force budget / in-force share); "
                      "forced when any admissible tier is affordable; a deeper "
                      "multiple drops recorded tiers whose move is below it"),
            "delta_min_in_force": floor_active,
            "rows": rows,
            "note": ("the forced RATE is set by the budget (tau); the multiple "
                     "sets which tiers are drawn. Lower the share to force less "
                     "at the same depth; raise the multiple to force less but "
                     "deeper. information_rel is the count x depth trade against "
                     "the in-force pair (quadratic in the move, proxy only)."
                     + ("" if floor_active else
                        " No delta_min floor is in force, so every multiple "
                        "reads the same.")),
        }

    def quantile_of(self, tau):
        self._build()
        if not len(self._costs):
            return None
        return float((self._costs <= tau).mean())

    def distribution(self, percentiles=(10, 25, 50, 75, 90, 95, 99)):
        self._build()
        if not len(self._costs):
            return {}
        return {f"p{p}": round(float(np.percentile(self._costs, p)), 2)
                for p in percentiles}


def tau_provenance_error(cfg, backtest, shadow=None):
    """Why the pasted `exploration.tau_initial` cannot be trusted, or None.

    A PROVENANCE check, not a precision one: the paste must agree (within
    `tau_paste_tolerance_rel`) with shadow's own derivation, or -- only when
    no shadow derivation exists -- with a backtest derivation that carries
    `spread_decisions`. `backtest`/`shadow` are the loaded reports, or None.
    """
    tau = cfg["exploration"]["tau_initial"]
    if tau is None:
        return None                 # null is a separate, louder failure
    tol = float(cfg["exploration"]["tau_paste_tolerance_rel"])

    def agrees(derived):
        return abs(float(derived) - float(tau)) <= tol * abs(float(derived))

    sh = (shadow or {}).get("tau_initial_derivation") or {}
    if sh.get("tau_initial") is not None:
        if agrees(sh["tau_initial"]):
            return None             # sourced from the anchored-path derivation
        return (f"exploration.tau_initial is {tau} but the shadow run derived "
                f"{sh['tau_initial']} on its own anchored path. Re-paste from "
                "reports/shadow.json -> tau_initial_derivation.tau_initial, "
                "or re-run shadow if the paste is the newer of the two.")
    der = (backtest or {}).get("tau_initial_derivation") or {}
    if not der:
        return (f"exploration.tau_initial is {tau} but no backtest derivation "
                "or shadow derivation is on disk to source it from. Run "
                "`python3 -m pipeline.shadow` (preferred: it derives tau on "
                "the anchored path) and paste "
                "tau_initial_derivation.tau_initial.")
    if "spread_decisions" not in der:
        return (f"exploration.tau_initial ({tau}) came from a backtest that "
                "predates the entry-only scoping fix: it solved tau on ENTRY "
                "decisions only, funding ~1 exploration per episode against a "
                "system that explores every hour. Re-run `python3 -m backtest` "
                "and re-paste.")
    derived = float(der["tau_initial"])
    if not agrees(derived):
        return (f"exploration.tau_initial is {tau} but the backtest derived "
                f"{derived}. Re-paste, or re-run the backtest if the paste is "
                "the newer of the two.")
    return None


def trailing_daily_il(il_by_day, day, cfg):
    """Mean REALISED daily IL over the trailing `budget_il_window_days`
    calendar days ending yesterday -- trailing and realised, never same-day
    or a forecast (design 5.8). Zero-IL calendar days count as zero; no
    history at all means a zero budget, the conservative side to start on.
    """
    window = int(cfg["exploration"]["budget_il_window_days"])
    d0 = pd.Timestamp(str(day))
    days = [(d0 - pd.Timedelta(days=k)).strftime("%Y-%m-%d")
            for k in range(1, window + 1)]
    known = [k for k in days if k in il_by_day]
    if not known:
        return 0.0
    # zero-IL calendar days inside the known span count as zero, so the mean
    # is over the window, not over the days that happened to carry IL
    span = (d0 - pd.Timestamp(min(known))).days
    denom = min(max(span, 1), window)
    return float(sum(il_by_day.get(k, 0.0) for k in days) / denom)


def budget_scale(posterior_std, cfg):
    """The budget's posterior-width factor: 1 at the reference std, shrinking
    with the widest routed std, never below budget_scale_floor."""
    ec = cfg["exploration"]
    return min(max(posterior_std / ec["budget_scale_ref_std"],
                   ec["budget_scale_floor"]), 1.0)


def budget_today(trailing_il, posterior_std, cfg):
    """budget_share_of_il x trailing daily IL x budget_scale (design 5.8)."""
    return cfg["exploration"]["budget_share_of_il"] * budget_scale(posterior_std, cfg) * trailing_il


def walk_tau(tau, days, spend_for, il_by_day, widest_std, cfg):
    """The controller walk, day by day, in one place: production (every
    closed day since the last calibration -- a weekly batch is seven
    steps, never one) and shadow's trace (expected spend at the tau in
    force) both call it. `spend_for(day, tau)` returns the day's realised
    or expected exploration spend. A ZERO budget (no trailing IL yet) is an
    absence of signal, not an overspend: tau holds that day. Returns
    (tau_end, rows)."""
    rows = []
    for day in days:
        budget = budget_today(trailing_daily_il(il_by_day, day, cfg),
                              widest_std, cfg)
        spend = float(spend_for(day, tau))
        after = tau_next(tau, budget, spend, cfg) if budget > 0 else tau
        rows.append({"day": str(day), "tau": round(float(tau), 2),
                     "spend": round(spend, 1), "budget": round(budget, 1),
                     "tau_after": round(float(after), 2)})
        tau = after
    return tau, rows


def tau_next(tau, budget, realised_cost, cfg):
    """tau * clip(budget/spend, *tau_adjust_clip) -- asymmetric on purpose:
    cutting is the safety direction, raising is never urgent (config)."""
    ec = cfg["exploration"]
    lo, hi = ec["tau_adjust_clip"]
    ratio = budget / max(realised_cost, ec["tau_spend_guard"])
    return tau * min(max(ratio, lo), hi)
