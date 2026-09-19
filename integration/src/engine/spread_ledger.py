"""engine.spread_ledger -- the Q-spread ledger tau is priced against.

`SpreadLedger` records, for EVERY decision the backtest or shadow prices,
the Q(p_star) - Q(p) spreads over its admissible tiers, and answers what
a tau would spend and what budget a tau solves to (design 5.8, 5.13,
5.14). One definition for both harnesses; the chooser those spreads
describe is engine.explore, the controller that walks tau is
engine.budget.
"""

import math

import numpy as np


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
        # the built arrays: costs/moves flat over every decision, and per
        # tier its decision; per decision its day and floor
        self._costs = self._moves = np.zeros(0)
        self._dec_of = self._dec_day = self._dec_start = np.zeros(0, dtype=np.int64)
        self._dec_dmin = np.zeros(0)

    def add(self, day, costs, moves=None, delta_min=0.0):
        """Record one decision's spreads. `costs` excludes the optimum;
        `moves` are the same tiers' log distances from the reference (zeros
        when the caller has none -- the sweep then reports multiples as
        inert) and `delta_min` the floor these tiers already cleared."""
        if not len(costs):
            return
        if moves is not None and len(moves) != len(costs):
            raise ValueError(f"{len(moves)} moves for {len(costs)} costs: the "
                             "two are aligned per tier")
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
        if self._chunks:
            # _lens/_day_of span the FULL history, so _costs must too --
            # dropping prior chunks after add()-query-add() mis-aligns every
            # index against _dec_of
            self._costs = np.concatenate([self._costs] + self._chunks)
            self._moves = np.concatenate([self._moves] + self._mchunks)
            self._chunks, self._mchunks = [], []
            lens = np.asarray(self._lens, dtype=np.int64)
            self._dec_of = np.repeat(np.arange(len(lens)), lens)
            self._dec_start = np.cumsum(lens) - lens       # first tier of each decision
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
        if 2 * np.count_nonzero(m) <= len(m):
            # the affordable tiers are the minority (every step once the
            # bisection has closed in): index them once and take twice
            idx = np.flatnonzero(m)
            sums = np.bincount(self._dec_of[idx], weights=w[idx], minlength=n_dec)
            cnts = np.bincount(self._dec_of[idx], minlength=n_dec)
        else:
            # the majority: one full pass. A masked-out tier adds exactly 0.0
            # in the same index order, so both branches give the same sums
            # to the bit; the counts are segment sums (tiers of one decision
            # are contiguous in _dec_of)
            sums = np.bincount(self._dec_of, weights=np.where(m, w, 0.0),
                               minlength=n_dec)
            cnts = np.add.reduceat(m, self._dec_start, dtype=np.int64)
        return np.divide(sums, cnts, out=np.zeros(n_dec), where=cnts > 0), cnts

    def spend_by_day(self, tau, keep=None, per_dec=None):
        """EXPECTED spend per day at `tau`: mean affordable cost per
        decision (uniform draw), empty sets contribute nothing. Expected,
        not realised -- the trace walks counterfactual taus. `per_dec` is
        `_per_decision(tau, keep=keep)[0]` if the caller already has it."""
        self._build()
        n_dec, n_day = len(self._lens), len(self._day_index)
        if not n_dec:
            return np.zeros(n_day)
        if per_dec is None:
            per_dec, _ = self._per_decision(tau, keep=keep)
        return np.bincount(self._dec_day, weights=per_dec, minlength=n_day)

    def implied_daily_spend(self, tau, n_days=None, keep=None, per_dec=None):
        by_day = self.spend_by_day(tau, keep, per_dec)
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
        # the admissibility tolerance has one home, beside `admissible`
        from engine.explore import LOG_EPS          # sibling; no cycle
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
            # one cost pass serves the count AND the spend; the two weighted
            # passes are the move statistics
            per_dec, cnts = self._per_decision(tau, keep=keep)
            forced = int((cnts > 0).sum())
            mean_move, _ = self._per_decision(tau, weights=self._moves, keep=keep)
            e_sq, _ = self._per_decision(tau, weights=move_sq, keep=keep)
            return {
                "tau": round(float(tau), 2),
                "forced_rate": round(forced / n_decisions, 4),
                "forced_per_day": round(forced / max(n_days, 1), 1),
                "implied_daily_spend": round(
                    self.implied_daily_spend(tau, n_days, keep, per_dec), 1),
                "daily_budget": round(budget, 1),
                "mean_log_move_forced": round(float(mean_move[cnts > 0].mean()), 4)
                    if forced else None,
                "_info": float(e_sq.sum()),
            }

        grid = {}
        for mult in multiples:
            for share in shares:
                grid[(share, mult)] = cell(share, mult)
        # the in-force cell by the same closeness test the rows are labelled
        # with -- an exact-key lookup missed a share rounded on its way in
        ref = next((c.get("_info") for (share, mult), c in grid.items()
                    if same(share, share_in_force) and same(mult, multiple_in_force)),
                   None)
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
