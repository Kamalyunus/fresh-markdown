"""pricing.posterior -- posterior read/write and the bounded-step projection.

One record per cell (PRD section 10). The persisted posterior is a Normal
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
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, path)

    def cell_name(self, category):
        return self.state["cell_of"].get(str(category), GLOBAL_CELL)

    def get(self, category):
        """The record priced with: the category's own cell or the global cell.
        It always exists -- initialised from the prior at launch."""
        return self.state["cells"][self.cell_name(category)]

    def is_processed(self, outcome_id):
        return outcome_id in set(self.state["processed_outcome_ids"])

    def commit_update(self, cell, new_mean, new_std, n_new_obs,
                      effective_information, outcome_ids, applied):
        """Atomically persist a posterior revision together with the outcome
        IDs it consumed (section 13.5: both commit or neither).

        An outcome is marked processed ONLY when a revision actually consumes
        it. A sub-threshold batch is left entirely unconsumed so tomorrow's
        run re-collects it and evaluates the likelihood over the whole
        accumulated batch. Marking it processed while declining to update
        would bank a scalar and throw the data away -- the posterior would
        later step on the strength of an information count it no longer had
        the observations to justify, and on a pilot small enough that no
        single day clears the threshold it would discard every outcome
        forever while the mean never moved.
        """
        rec = self.state["cells"][cell]
        if not applied:
            return                       # nothing consumed, nothing persisted
        # No `information_since_update` counter. PRD 13.4 specified one and it
        # was carried here for a while, always reset and never incremented,
        # because the trigger is evaluated on the UNCONSUMED BATCH rather than
        # on a running total -- see `pipeline.update.run`. A field that is
        # permanently zero reads as "no evidence has accrued", which is the
        # opposite of what a growing batch means, so it is gone rather than
        # kept for the schema's sake. `accumulated_information` is the total.
        rec["mean"], rec["std"] = float(new_mean), float(new_std)
        rec["version"] += 1
        rec["accumulated_information"] += effective_information
        rec["updated_at"] = pd.Timestamp.now("UTC").isoformat()
        rec["n_obs"] += n_new_obs
        self.state["processed_outcome_ids"].extend(outcome_ids)
        self._atomic_write(self.path, self.state)

    # ------------------------------------------------------------------ tau
    # tau is production learning state in exactly the sense the posterior is:
    # it is set from config ONCE at launch and moves thereafter from realised
    # spend (12.3). It therefore lives here, in the file that already commits
    # atomically and is already the one thing production writes -- not back
    # into config.yaml, which is hand-maintained and would make a running
    # system edit its own source of truth.

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
        """Persist a recalibrated tau, stamped with the date it consumed.

        The stamp is the exactly-once guard: two `--apply` runs on the same
        day would otherwise apply the same budget-vs-spend ratio twice and
        move tau by its square.
        """
        self.state["tau"] = float(tau)
        self.state["tau_calibrated_through"] = str(through_date)
        self.state["tau_updated_at"] = pd.Timestamp.now("UTC").isoformat()
        self._atomic_write(self.path, self.state)
