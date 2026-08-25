"""pricing.explore -- IL-budgeted exploration (PRD section 12).

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

    Public because `pipeline.assurance` has to reconstruct EXACTLY what this
    chose in order to test that the draw was uniform. Two copies of this rule
    that drift apart would leave the check quietly testing the wrong set, and
    the failure it exists to catch is already silent.
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

    ONE definition of "what would tau spend", shared by the backtest replay
    and the shadow harness, because two copies drifted apart once already:
    the replay collected spreads at the ENTRY decision only (`if t == 0`)
    while `select` is called at every decision hour, so tau_initial was
    solved to fund roughly one exploration per episode against a system that
    explores ~8 times per episode. That is most of the 8.7x the shadow report
    measured, and it was invisible because the replay's own bisection reports
    1.00x by construction.

    Costs are recorded for EVERY decision the caller sees, independently of
    the tau in force -- the spread is a property of the state, not of what
    was drawn from it. Stored flat and chunked rather than as one array per
    decision: at full population this holds ~30M costs, and 3M small numpy
    arrays cost more in object overhead than the numbers do.
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

    def sink(self, day):
        """A one-argument callable for `inference.decide(spread_sink=...)`."""
        return lambda costs: self.add(day, costs)

    def _build(self):
        if self._buf:
            self._chunks.append(np.asarray(self._buf, dtype=np.float64))
            self._buf = []
        if getattr(self, "_costs", None) is None or self._chunks:
            self._costs = (np.concatenate(self._chunks) if self._chunks
                           else np.zeros(0))
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
        """Expected exploration spend per day at `tau`.

        Uniform selection over the affordable set, so the expected cost of a
        forced decision is the MEAN affordable cost; a decision with an empty
        affordable set contributes nothing. Expected, not realised: the trace
        asks what a counterfactual tau would have spent, and the draws on
        record were made at a different one.
        """
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
        """The tau whose implied daily spend equals `budget_per_day`.

        Bisection rather than a quantile: spend is a non-linear function of
        tau (it moves both the size of the affordable set and the mean cost
        within it), so no fixed quantile of the cost distribution is the
        answer on a population it was not measured on.

        Returns the low end of the bracket, so the answer is always a tau
        whose implied spend is UNDER budget. Spend steps rather than sliding
        -- it jumps as each cost crosses tau -- so no tau lands exactly on
        the budget, and on a thin window the last step can be a percent wide.
        Under is the right side of a budget to miss on.
        """
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


def tau_provenance_error(cfg, backtest):
    """Why the pasted `exploration.tau_initial` cannot be trusted, or None.

    tau_initial is MEASURED: it is pasted by hand from a backtest derivation,
    and nothing until now checked that the paste still matched its source. A
    stale one is silent and expensive -- it sets how much exploration the
    pilot buys on day one, and the stop condition is evaluated on that day's
    spend before the controller has seen anything to correct from.

    Three ways a paste goes bad, all checked here:

      * no derivation to source it from at all;
      * a derivation that PREDATES the entry-only scoping fix. Those solved
        tau on entry decisions only, funding roughly one exploration per
        episode against a system that explores every hour, so the value is
        wrong by about that factor. `spread_decisions` is the marker: the old
        block did not emit it;
      * a value that simply disagrees with the derivation -- a hand-typed
        number, or a backtest re-run since the paste.

    `backtest` is the loaded reports/backtest.json, or None.
    """
    tau = cfg["exploration"]["tau_initial"]
    if tau is None:
        return None                 # null is a separate, louder failure
    der = (backtest or {}).get("tau_initial_derivation") or {}
    if not der:
        return (f"exploration.tau_initial is {tau} but no backtest derivation "
                "is on disk to source it from. Run `python3 -m backtest` and "
                "paste tau_initial_derivation.tau_initial.")
    if "spread_decisions" not in der:
        return (f"exploration.tau_initial ({tau}) came from a backtest that "
                "predates the entry-only scoping fix: it solved tau on ENTRY "
                "decisions only, funding ~1 exploration per episode against a "
                "system that explores every hour. Re-run `python3 -m backtest` "
                "and re-paste.")
    derived = float(der["tau_initial"])
    if abs(derived - float(tau)) > 0.01:
        return (f"exploration.tau_initial is {tau} but the backtest derived "
                f"{derived}. Re-paste, or re-run the backtest if the paste is "
                "the newer of the two.")
    return None


def trailing_daily_il(il_by_day, day, cfg):
    """The IL base for a day's budget: the mean of REALISED daily markdown IL
    over the trailing `budget_il_window_days` calendar days, ending yesterday.

    TRAILING AND REALISED, not same-day and not a forecast (section 12.3).
    Same-day IL is not observable when the day's budget is needed -- scrap
    lands only when an episode closes -- and it funds exploration hardest on
    exactly the days already losing most, since IL is discount plus scrap.
    A trailing mean is observable at the start of the day, smooths the
    weekday cycle once at 7 days, and follows real drift in markdown volume,
    which a whole-history mean would not. The fast dynamics belong to the
    posterior-std scale in `budget_today` and the daily `tau_next` controller,
    not to the base.

    Calendar days, not data days: a day with no closed IL counts as zero in
    the mean, because the budget is a share of the actual run-rate. Days with
    no history at all (the window's first day) fall back to whatever trailing
    days exist; with none, zero -- no IL history means no exploration budget,
    which is the conservative side to start a pilot on.
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
    """Section 12.3: a share of the trailing daily markdown IL
    (`trailing_daily_il`), scaled down as the posterior narrows, never below
    budget_scale_floor of the full budget."""
    ec = cfg["exploration"]
    scale = min(max(posterior_std / ec["budget_scale_ref_std"],
                    ec["budget_scale_floor"]), 1.0)
    return ec["budget_share_of_il"] * scale * trailing_il


def tau_next(tau, budget, realised_cost, cfg):
    """Daily multiplicative calibration of tau from realised spend."""
    ec = cfg["exploration"]
    lo, hi = ec["tau_adjust_clip"]
    ratio = budget / max(realised_cost, ec["tau_spend_guard"])
    return tau * min(max(ratio, lo), hi)
