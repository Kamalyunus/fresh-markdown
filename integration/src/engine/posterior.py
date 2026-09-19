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


GLOBAL_CELL = "GLOBAL"


class PosteriorStore:
    """The production learning state, read from posterior.json once at
    construction. Two writers share the file -- `daily.update` (cells, tau)
    and `daily.monitor` (the exploration suspension) -- so a long-lived
    reader (the hourly pricing service holding one store) sees neither
    until it calls `reload()`. The contract: the CALLER reloads once per
    decision batch, before its first `decide()`; `decide()` itself never
    re-reads the file (one read per hour, not one per SKU).

    Every write goes through ONE path, `_commit`: take the file lock,
    re-read the file, apply the mutation to what is on disk, write. A
    writer never writes the state it loaded earlier, so a suspension the
    monitor wrote between an `--apply`'s load and its write survives it,
    and neither writer can empty the other's processed-id ledger."""

    def __init__(self, cfg, path=None):
        self.cfg = cfg
        self.path = path or cfg["posterior"]["path"]
        self.reload()

    def reload(self):
        """Re-read the file, dropping every cached view of the old state.
        Returns self, so `store.reload().get(...)` chains; a fresh
        `PosteriorStore(cfg)` has already read the file once."""
        with open(self.path) as f:
            self.state = json.load(f)
        self._processed = None
        return self

    def cell_name(self, category):
        return self.state["cell_of"].get(str(category), GLOBAL_CELL)

    def get(self, category):
        """The record priced with: the category's own cell or the global cell.
        It always exists -- initialised from the prior at launch."""
        return self.state["cells"][self.cell_name(category)]

    # tau is production learning state: it lives here, not in hand-
    # maintained config.yaml (design 5.8).

    # exploration suspension (design 5.12): a fired stop condition stops
    # FORCED exploration only -- decide() prices with no budget and records
    # tau_current None -- while exploitation continues. Set by
    # daily.monitor, cleared only by a human (update --resume-exploration).

    def exploration_suspended(self):
        """The suspension record {reasons, since, updated_at}, or None."""
        return self.state.get("exploration_suspended")


    def tau(self):
        """The exploration budget in force, in currency.

        Falls back to `exploration.tau_initial` until the first calibration:
        a launch that has spent nothing has nothing to calibrate from. Read
        from the store's own config: the file and the config it was
        initialised under are one state.
        """
        stored = self.state.get("tau")
        return float(stored) if stored is not None \
            else self.cfg["exploration"]["tau_initial"]
