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

import numpy as np
import pandas as pd


def affordable_set(dp_result, tau):
    """Tier indices a perturbation may legally land on, and their costs.
    Public: `pipeline.assurance` reconstructs this exact set to test that the
    draw was uniform -- one definition, never two.
    """
    q = dp_result.q_by_tier
    star = dp_result.optimal_index
    costs = {j: q[star] - q[j] for j in q}
    return [j for j in q if j != star and costs[j] <= tau], costs


def select(dp_result, tau, rng, explorable=True):
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

    affordable, costs = affordable_set(dp_result, tau)
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
        self._lens, self._day_of, self._day_index = [], [], {}

    def add(self, day, costs):
        """Record one decision's spreads. `costs` excludes the optimum."""
        if not len(costs):
            return
        self._buf.extend(costs)
        self._lens.append(len(costs))
        day = str(day)
        if day not in self._day_index:
            self._day_index[day] = len(self._day_index)
        self._day_of.append(self._day_index[day])
        if len(self._buf) >= self._FLUSH:
            self._chunks.append(np.asarray(self._buf, dtype=np.float64))
            self._buf = []

    def _build(self):
        if self._buf:
            self._chunks.append(np.asarray(self._buf, dtype=np.float64))
            self._buf = []
        if getattr(self, "_costs", None) is None or self._chunks:
            # _lens/_day_of span the FULL history, so _costs must too --
            # dropping prior chunks after add()-query-add() mis-aligns every
            # index against _dec_of
            prior = ([self._costs] if getattr(self, "_costs", None) is not None
                     and len(self._costs) else [])
            self._costs = (np.concatenate(prior + self._chunks)
                           if (prior or self._chunks) else np.zeros(0))
            self._chunks = []
            lens = np.asarray(self._lens, dtype=np.int64)
            self._dec_of = np.repeat(np.arange(len(lens)), lens)
            self._dec_day = np.asarray(self._day_of, dtype=np.int64)

    @property
    def decisions(self):
        return len(self._lens)

    @property
    def days(self):
        """Day labels in first-seen order; index i is row i of spend_by_day."""
        return [d for d, _ in sorted(self._day_index.items(), key=lambda kv: kv[1])]

    def spend_by_day(self, tau):
        """EXPECTED spend per day at `tau`: mean affordable cost per
        decision (uniform draw), empty sets contribute nothing. Expected,
        not realised -- the trace walks counterfactual taus."""
        self._build()
        n_dec, n_day = len(self._lens), len(self._day_index)
        if not n_dec:
            return np.zeros(n_day)
        m = self._costs <= tau
        sums = np.bincount(self._dec_of[m], weights=self._costs[m], minlength=n_dec)
        cnts = np.bincount(self._dec_of[m], minlength=n_dec)
        per_dec = np.divide(sums, cnts, out=np.zeros(n_dec), where=cnts > 0)
        return np.bincount(self._dec_day, weights=per_dec, minlength=n_day)

    def implied_daily_spend(self, tau, n_days=None):
        by_day = self.spend_by_day(tau)
        return float(by_day.sum()) / max(n_days or len(by_day), 1)

    def solve_tau(self, budget_per_day, n_days=None, steps=60):
        """Bisect for the tau whose implied daily spend equals the budget.
        Returns the LOW end of the bracket: spend steps as costs cross tau,
        and under-budget is the right side to miss on."""
        self._build()
        if not len(self._costs) or budget_per_day <= 0:
            return None
        lo, hi = 0.0, float(self._costs.max())
        if self.implied_daily_spend(hi, n_days) < budget_per_day:
            return hi                       # budget exceeds even unbounded tau
        for _ in range(steps):
            mid = (lo + hi) / 2
            if self.implied_daily_spend(mid, n_days) < budget_per_day:
                lo = mid
            else:
                hi = mid
        return lo

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

    The trusted source is SHADOW's own derivation (anchored path over the
    trailing pre-window week); the backtest derivation (exploit-only path)
    is accepted only when no shadow derivation exists, with the old three
    failure modes: none on disk; predating the entry-only scoping fix
    (marker: no `spread_decisions`); a value that disagrees with it.
    This is a PROVENANCE check -- did the paste come from the derivation --
    not a precision requirement on tau. `tau_initial` seeds day one only
    (`PosteriorStore.tau`); from the first calibration tau is production
    state and `tau_next` moves it up to 25% a day, so a few percent at launch
    is absorbed immediately. The tolerance is relative because tau is
    currency and scales with the population's IL. The pastes this catches --
    wrong run, wrong field, the pre-scoping-fix backtest -- are all off by
    tens of percent or multiples, never by fractions of one.

    `backtest`/`shadow` are the loaded reports, or None.
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


def budget_today(trailing_il, posterior_std, cfg):
    """budget_share_of_il x trailing daily IL, scaled down as the posterior
    narrows, never below budget_scale_floor (design 5.8)."""
    ec = cfg["exploration"]
    scale = min(max(posterior_std / ec["budget_scale_ref_std"],
                    ec["budget_scale_floor"]), 1.0)
    return ec["budget_share_of_il"] * scale * trailing_il


def tau_next(tau, budget, realised_cost, cfg):
    """tau * clip(budget/spend, *tau_adjust_clip) -- asymmetric on purpose:
    cutting is the safety direction, raising is never urgent (config)."""
    ec = cfg["exploration"]
    lo, hi = ec["tau_adjust_clip"]
    ratio = budget / max(realised_cost, ec["tau_spend_guard"])
    return tau * min(max(ratio, lo), hi)
