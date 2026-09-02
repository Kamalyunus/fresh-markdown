"""pricing.posterior -- posterior read/write and the bounded-step projection.

One record per cell (design section 5.9). The persisted posterior is a Normal
summary, never a stored grid -- the grid exists only inside the update
computation, which keeps storage trivial and makes the bounded step of
section 13.4 well-defined.

Cell assignment (section 10.2) happens once at launch from phase-0 volumes:
categories at or above min_episodes_per_week_for_cell get their own cell,
everything else reads and feeds the global cell. There is no fallback chain
and no _default key -- the global cell always exists.

The store file also carries processed_outcome_ids so that posterior revision
and processed-ID commit are a single atomic write (tmp + os.replace), giving
the exactly-once property of section 13.5.
"""

import json
import os
import tempfile

import pandas as pd

GLOBAL_CELL = "GLOBAL"


def bounded_step(mean_before, std_before, raw_mean, raw_std, cfg):
    """Section 13.4: clip the mean step, floor the std shrink. Returns
    (new_mean, new_std, clipped)."""
    lc = cfg["learning"]
    step = lc["max_mean_step"]
    new_mean = min(max(raw_mean, mean_before - step), mean_before + step)
    new_std = max(raw_std,
                  std_before * (1.0 - lc["max_std_shrink"]),
                  cfg["posterior"]["min_std"])
    clipped = (new_mean != raw_mean) or (new_std != raw_std)
    return new_mean, new_std, clipped


class PosteriorStore:
    def __init__(self, cfg, path=None):
        self.cfg = cfg
        self.path = path or cfg["posterior"]["path"]
        with open(self.path) as f:
            self.state = json.load(f)

    @classmethod
    def initialise(cls, cfg, prior_by_category, episodes_per_week, path=None):
        """Create posterior.json at launch from the section 9.5 prior and
        phase-0 weekly episode volumes. Assignment does not move during the
        MVP window."""
        path = path or cfg["posterior"]["path"]
        floor = cfg["posterior"]["min_episodes_per_week_for_cell"]
        cell_of = {c: (c if episodes_per_week.get(c, 0) >= floor else GLOBAL_CELL)
                   for c in prior_by_category}

        def record(mean, std):
            return {"mean": float(mean), "std": float(std), "n_obs": 0,
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

        state = {"cells": cells, "cell_of": cell_of,
                 "processed_outcome_ids": [],
                 "prior_source": "profile_density"}
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

    def cell_name(self, category):
        return self.state["cell_of"].get(str(category), GLOBAL_CELL)

    @staticmethod
    def widest_active_std(cells, cell_of):
        """Widest std among cells a category actually ROUTES to.

        GLOBAL is always created -- `cell_name` falls back to it for a
        category the prior never saw -- but when every category clears
        `min_episodes_per_week_for_cell` nothing routes there, so it receives
        no outcome and never narrows. Taking the max over all cells then
        pinned this at the launch std forever: the exploration budget never
        scaled down as the learning cells converged (design 5.8's
        `budget_scale_floor` was unreachable) and the flat-std alert listed
        GLOBAL permanently.
        """
        routed = set(cell_of.values())
        # routed now, OR has taken outcomes (routing can move after a
        # re-initialise; a cell that learned something still has a std to
        # budget for)
        active = [r["std"] for c, r in cells.items()
                  if c in routed or r.get("n_obs", 0) > 0]
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
