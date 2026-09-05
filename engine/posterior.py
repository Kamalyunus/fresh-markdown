"""engine.posterior -- posterior read/write and the bounded-step projection.

One record per cell (design 5.9). The persisted posterior is a Normal
summary, never a stored grid -- the grid exists only inside the update
computation, which keeps storage trivial and makes the bounded step of
design 5.11 well-defined.

Cell assignment (design 5.9) happens once at launch from phase-0 volumes:
categories at or above min_episodes_per_week_for_cell get their own cell,
everything else reads and feeds the global cell. There is no fallback chain
and no _default key -- the global cell always exists.

The store file also carries processed_outcome_ids so that posterior revision
and processed-ID commit are a single atomic write (tmp + os.replace), giving
the exactly-once property of design 5.9. The same file carries the
exploration-suspension record (design 5.12): a fired stop condition suspends
forced exploration only, exploitation pricing continues, and only a human
(`daily.update --resume-exploration`) clears it.
"""

import json
import os
import tempfile

import pandas as pd

GLOBAL_CELL = "GLOBAL"


def bounded_step(mean_before, std_before, raw_mean, raw_std, cfg):
    """Design 5.11: clip the mean step, floor the std shrink. Returns
    (new_mean, new_std, clipped)."""
    lc = cfg["learning"]
    step = lc["max_mean_step"]
    new_mean = min(max(raw_mean, mean_before - step), mean_before + step)
    new_std = max(raw_std,
                  std_before * (1.0 - lc["max_std_shrink"]),
                  cfg["posterior"]["min_std"])
    clipped = (new_mean != raw_mean) or (new_std != raw_std)
    return new_mean, new_std, clipped


def launch_belief(mean, std, cfg):
    """The mean a cell LAUNCHES at: the prior mean pushed
    `posterior.cold_start_shift_std` prior stds toward more elastic and
    clipped to the epsilon range (design 5.9). The std is untouched, so the
    push is largest where the prior is widest and evidence weighs the same;
    the bounded step walks it back if outcomes disagree. 0 = the prior."""
    pc = cfg["posterior"]
    shifted = float(mean) - float(pc.get("cold_start_shift_std") or 0.0) * float(std)
    return float(min(max(shifted, pc["epsilon_min"]), pc["epsilon_max"]))


class PosteriorStore:
    def __init__(self, cfg, path=None):
        self.cfg = cfg
        self.path = path or cfg["posterior"]["path"]
        with open(self.path) as f:
            self.state = json.load(f)

    @staticmethod
    def launch_state(cfg, prior_by_category, episodes_per_week):
        """The state initialise() writes: every cell at its launch belief,
        categories routed by phase-0 weekly volume (assignment does not move
        during the MVP window). Pure, so launch_stale() can recompute it."""
        floor = cfg["posterior"]["min_episodes_per_week_for_cell"]
        cell_of = {c: (c if episodes_per_week.get(c, 0) >= floor else GLOBAL_CELL)
                   for c in prior_by_category}

        def record(mean, std):
            return {"mean": launch_belief(mean, std, cfg), "prior_mean": float(mean),
                    "std": float(std), "n_obs": 0,
                    "accumulated_information": 0.0,
                    "version": 0,
                    "updated_at": pd.Timestamp.now("UTC").isoformat()}

        cells = {c: record(p["mean"], p["std"])
                 for c, p in prior_by_category.items() if cell_of[c] == c}
        # global cell: precision-weighted average is over-confident for a
        # prior; use the simple mean of member means and the widest member std
        members = [p for c, p in prior_by_category.items()
                   if cell_of[c] == GLOBAL_CELL]
        if not members:
            members = list(prior_by_category.values())
        cells[GLOBAL_CELL] = record(
            sum(m["mean"] for m in members) / len(members),
            max(m["std"] for m in members))
        return {"cells": cells, "cell_of": cell_of,
                "processed_outcome_ids": [],
                "prior_source": "profile_density",
                "cold_start_shift_std": float(
                    cfg["posterior"].get("cold_start_shift_std") or 0.0)}

    @classmethod
    def initialise(cls, cfg, prior_by_category, episodes_per_week, path=None):
        """Create posterior.json at launch (design 5.9)."""
        path = path or cfg["posterior"]["path"]
        state = cls.launch_state(cfg, prior_by_category, episodes_per_week)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        cls._atomic_write(path, state)
        store = cls.__new__(cls)
        store.cfg, store.path, store.state = cfg, path, state
        return store

    @staticmethod
    def _atomic_write(path, state):
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".",
                                   prefix=".posterior-")
        # fsync file AND directory: without them a power failure can lose the
        # learning state and its exactly-once ledger
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(os.path.dirname(path) or ".", os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def launch_stale(self, prior_by_category, episodes_per_week):
        """True while the store holds NO learning and its cells differ from
        what initialise() would write now (the launch belief moved, or the
        prior did). Once an outcome has been consumed the learner owns the
        mean and the file is never re-initialised by the process."""
        if self.state.get("processed_outcome_ids"):
            return False
        fresh = self.launch_state(self.cfg, prior_by_category, episodes_per_week)
        mine = {c: (round(r["mean"], 9), round(r["std"], 9))
                for c, r in self.state["cells"].items()}
        want = {c: (round(r["mean"], 9), round(r["std"], 9))
                for c, r in fresh["cells"].items()}
        return mine != want or self.state["cell_of"] != fresh["cell_of"]

    def cell_name(self, category):
        return self.state["cell_of"].get(str(category), GLOBAL_CELL)

    @staticmethod
    def active_cells(cells, cell_of):
        """The cells that can still learn: routed to by some category now, or
        holding outcomes already (routing can move after a re-initialise).
        An unrouted GLOBAL takes no outcome and never narrows, so every
        per-cell reading -- the budget's widest std, the flat-std alert --
        is taken over THIS set, never over every cell in the file."""
        routed = set(cell_of.values())
        return [c for c, r in cells.items()
                if c in routed or r.get("n_obs", 0) > 0]

    @staticmethod
    def widest_active_std(cells, cell_of):
        """Widest std among `active_cells` -- the cell with the most still to
        learn, which the exploration budget is sized for (design 5.8)."""
        active = [cells[c]["std"] for c in
                  PosteriorStore.active_cells(cells, cell_of)]
        return max(active) if active else max(r["std"] for r in cells.values())

    def widest_std(self):
        """The std the exploration budget is sized for: the routed cell with
        the most still to learn."""
        return self.widest_active_std(self.state["cells"],
                                      self.state["cell_of"])

    def get(self, category):
        """The record priced with: the category's own cell or the global cell.
        It always exists -- initialised from the prior at launch."""
        return self.state["cells"][self.cell_name(category)]

    def is_processed(self, outcome_id):
        # cached: the ledger only grows, and rebuilding the set per lookup
        # made the daily batch quadratic in its own history
        if getattr(self, "_processed", None) is None:
            self._processed = set(self.state["processed_outcome_ids"])
        return outcome_id in self._processed

    def commit_update(self, cell, new_mean, new_std, n_new_obs,
                      effective_information, outcome_ids, applied):
        """Atomically persist a revision with the outcome IDs it consumed
        (both commit or neither). An outcome is marked processed ONLY when a
        revision consumes it -- a sub-threshold batch stays whole and is
        re-read tomorrow (design 5.11).
        """
        rec = self.state["cells"][cell]
        if not applied:
            return                       # nothing consumed, nothing persisted
        # no information_since_update counter -- the trigger reads the
        # unconsumed batch, never a running total (design 5.11)
        rec["mean"], rec["std"] = float(new_mean), float(new_std)
        rec["version"] += 1
        rec["accumulated_information"] += effective_information
        rec["updated_at"] = pd.Timestamp.now("UTC").isoformat()
        rec["n_obs"] += n_new_obs
        self.state["processed_outcome_ids"].extend(outcome_ids)
        if getattr(self, "_processed", None) is not None:
            self._processed.update(outcome_ids)
        self._atomic_write(self.path, self.state)

    # tau is production learning state: it lives here, not in hand-
    # maintained config.yaml (design 5.8).

    def tau(self, cfg):
        """The exploration budget in force, in currency.

        Falls back to `exploration.tau_initial` until the first calibration:
        a launch that has spent nothing has nothing to calibrate from.
        """
        stored = self.state.get("tau")
        return float(stored) if stored is not None \
            else cfg["exploration"]["tau_initial"]

    def tau_calibrated_through(self):
        """The last date tau was calibrated for, or None."""
        return self.state.get("tau_calibrated_through")

    def commit_tau(self, tau, through_date):
        """Persist a recalibrated tau, stamped with the date it consumed --
        the exactly-once-per-day guard."""
        self.state["tau"] = float(tau)
        self.state["tau_calibrated_through"] = str(through_date)
        self.state["tau_updated_at"] = pd.Timestamp.now("UTC").isoformat()
        self._atomic_write(self.path, self.state)

    # exploration suspension (design 5.12): a fired stop condition stops
    # FORCED exploration only -- decide() prices with no budget and records
    # tau_current None -- while exploitation continues. Set by
    # daily.monitor, cleared only by a human (update --resume-exploration).

    def exploration_suspended(self):
        """The suspension record {reasons, since, updated_at}, or None."""
        return self.state.get("exploration_suspended")

    def suspend_exploration(self, reasons, since):
        """Suspend forced exploration for `reasons` (stop-condition names)
        from `since` (the trading day that fired it). Idempotent: a second
        call keeps the FIRST `since` and unions the reasons, so a stop that
        keeps firing does not restart the clock. Same atomic write as the
        cells. Returns the record."""
        current = self.state.get("exploration_suspended")
        merged = sorted(set(reasons) | set(current["reasons"] if current else ()))
        if not merged:
            raise ValueError("suspend_exploration needs at least one reason")
        record = {"reasons": merged,
                  "since": current["since"] if current else str(since)}
        if current and current["reasons"] == merged \
                and current["since"] == record["since"]:
            return current                     # nothing changed, nothing written
        record["updated_at"] = pd.Timestamp.now("UTC").isoformat()
        self.state["exploration_suspended"] = record
        self._atomic_write(self.path, self.state)
        return record

    def resume_exploration(self):
        """Clear the suspension (the human gate). Returns the record cleared,
        or None when exploration was not suspended."""
        record = self.state.pop("exploration_suspended", None)
        if record is not None:
            self._atomic_write(self.path, self.state)
        return record
