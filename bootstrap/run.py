"""bootstrap.run -- the pipeline, with its loop driven to convergence.

    python3 -m bootstrap.run --input data/flc_raw.parquet

The chain is NOT a line. The factor solve consumes `r`, while `r`, `rho` and
the prior are all fitted against calibrated `mu_ref`, so steps 3b-5 are one
turn of a fixed-point iteration that a bare chain needs MANY of -- the owner
measures 8-9 on the production extract; the repo fixture settles in 3-4, and
being small is why. Size the cap for the former, never the latter.
A linear script cannot express that: it ran the turn once, printed NOT
CONVERGED, and left a human or an agent to repeat 3b-4-5-5b by hand. Worse,
"re-run the script" then RETRAINED THE BASELINE at step 3, which moved every
artifact, reset the fixed point and broke hard rule 1 -- the loop this module
exists to close was the loop an agent could not escape.

So the order lives here, in one place, and the loop is the module's job:

    1   prepare_data
    3   train_baseline                     ONCE -- never inside the loop
    ->  3b calibration, 4 prior, 5 dispersion, 5b convergence   (iterate)
    6   backtest, 6b thresholds, 11 seal, status

`--check-only` re-runs the loop against artifacts already on disk without
retraining, which is what a config paste needs (see `pipeline.tune`); nothing
in normal operation should ever retrain to settle calibration.
"""

import argparse
import json
import os
import subprocess
import sys

from common.config import load_config

PREPARED = "data/prepared.parquet"


def step(label, args, fatal=True, quiet=False):
    """One pipeline step. Output streams through: the console lines are the
    evidence, and swallowing them to keep the log tidy is how a warning gets
    missed."""
    print(f"\n== {label} " + "=" * max(0, 62 - len(label)))
    r = subprocess.run([sys.executable, "-m", *args])
    if r.returncode and fatal:
        raise SystemExit(f"\n{label} FAILED (exit {r.returncode}) -- stopping "
                         "here rather than building on a broken artifact")
    return r.returncode


def convergence(cfg):
    """The verdict block the last `--check-convergence` wrote, or None."""
    path = cfg["baseline_model"]["calibration_factor_path"]
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return (json.load(f) or {}).get("convergence")


def settle(cfg, max_turns):
    """Turn the calibration <-> dispersion loop until it settles.

    Each turn re-fits the factors, then the prior and dispersion AGAINST those
    factors, then asks whether one more turn would reproduce them. The
    baseline is never retrained in here -- that is what makes this a loop
    rather than a restart.
    """
    seen = []
    for turn in range(1, max_turns + 1):
        # 3b ONLY on the first turn. Every later turn's factors were already
        # solved by the previous turn's --check-convergence, under exactly
        # this prior and r -- `--commit-convergence` keeps that solve instead
        # of recomputing it, which halves the calibration work in the loop.
        if turn == 1:
            step("loop turn 1: 3b calibration",
                 ["bootstrap.train_baseline", "--input", PREPARED,
                  "--fit-calibration"])
        # --fast drops the prior's fold_spread diagnostic (most of its cost).
        # It cannot move the fixed point: the loop compares FACTORS, which
        # depend on r, which is fitted at the prior MEAN, while fold_spread
        # only widens the std FLOOR. The final turn re-runs FULL, below,
        # because init_posterior reads that std.
        step(f"loop turn {turn}: 4 prior (fast)",
             ["bootstrap.estimate_prior", "--input", PREPARED, "--fast"])
        step(f"loop turn {turn}: 5 dispersion",
             ["bootstrap.fit_dispersion", "--input", PREPARED])
        step(f"loop turn {turn}: 5b convergence",
             ["bootstrap.train_baseline", "--input", PREPARED,
              "--check-convergence", "--commit-convergence"], fatal=False)

        block = convergence(cfg) or {}
        if block.get("converged"):
            print(f"\nloop SETTLED after {turn} turn(s).")
            # the loop ran on --fast priors; the ARTIFACT must carry the full
            # one, because init_posterior reads a std the fast path leaves at
            # its density/grid floor. Factors do not move: the mean is
            # unchanged, so this cannot unsettle what just settled.
            step("prior: full re-run for the artifact",
                 ["bootstrap.estimate_prior", "--input", PREPARED])
            # that re-run moved prior.json, so the verdict just written is
            # stamped against a chain that no longer exists and `status` says
            # so. Re-check (DRY -- nothing should move) to re-stamp the
            # digests. It confirms rather than re-settles: the full prior
            # changes only the std, and factors follow the MEAN.
            step("5b: re-check against the full prior",
                 ["bootstrap.train_baseline", "--input", PREPARED,
                  "--check-convergence"], fatal=False)
            return True, turn, (convergence(cfg) or block)

        # STALL test, not a bad-turn test: stop only after three turns with
        # no new best (a plateau inside a 9-turn settle is ordinary). Reads
        # THIS run's turns -- the artifact's `history` spans runs, and an old
        # chain's low reading would stall a fresh run on turn 1.
        if block.get("max_abs_dlog") is not None:
            seen.append(float(block["max_abs_dlog"]))
        if len(seen) >= 4 and min(seen[-3:]) >= min(seen[:-3]):
            print(f"\nloop STALLED: no new best in 3 turns over "
                  f"{[round(h, 5) for h in seen]} -- stopping at turn {turn}. "
                  f"Another turn will not help; read worst_cell / "
                  f"worst_cell_anchor_rows before continuing.")
            return False, turn, block

    return False, max_turns, (convergence(cfg) or {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="raw FLC parquet (omit with --check-only)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--max-turns", type=int, default=20,
                    help="cap on calibration<->dispersion turns (default 20). "
                         "The owner measures 8-9 on production; the fixture "
                         "settles in 3-4. The cap is a runaway guard, not a "
                         "budget -- the STALL test stops a loop that is not "
                         "improving long before this")
    ap.add_argument("--mde", type=float, default=0.075,
                    help="A/B target for derive_thresholds")
    ap.add_argument("--check-only", action="store_true",
                    help="settle the loop against the artifacts already on "
                         "disk and refresh the reports -- NO retrain. This is "
                         "what a config paste needs; see pipeline.tune")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if not args.check_only:
        if not args.input:
            raise SystemExit("--input is required unless --check-only")
        step("step 1: prepare_data",
             ["bootstrap.prepare_data", "--input", args.input,
              "--out", PREPARED])
        # ONCE, and outside the loop. Retraining to settle calibration moves
        # every artifact and invalidates every comparison (hard rule 1).
        step("step 3: train_baseline", ["bootstrap.train_baseline",
                                        "--input", PREPARED])

    ok, turns, block = settle(cfg, args.max_turns)

    step("step 6: backtest", ["backtest", "--input", PREPARED,
                              "--out", "reports/backtest.json"])
    step("step 6b: derive_thresholds",
         ["bootstrap.derive_thresholds", "--input", PREPARED,
          "--mde", str(args.mde)])
    step("step 11: seal", ["bootstrap.seal"], fatal=False)
    step("where this run stands: status", ["pipeline.status"], fatal=False)

    if not ok:
        print(f"\nLOOP DID NOT SETTLE in {turns} turn(s): "
              f"max |dlog f| {block.get('max_abs_dlog')} vs tol "
              f"{block.get('tol_log')} at {block.get('worst_cell')}"
              + (f" ({block['worst_cell_anchor_rows']:,} anchor rows)"
                 if block.get("worst_cell_anchor_rows") else "")
              + ".\nThe reports above were still written, and they are NOT "
                "decision-grade: every artifact under them depends on how "
                "many turns happened to run. Raise --max-turns if the "
                "trajectory is still contracting; investigate if it is not.")
        raise SystemExit(1)

    print(f"""
Bootstrap complete -- loop settled in {turns} turn(s). Next:
  1. Hold-out shadow (it derives its own launch tau):
       python3 -m bootstrap.init_posterior
       python3 -m pipeline.shadow --input {PREPARED} --out reports/shadow.json
  2. Let the reports decide the config, and record why:
       python3 -m pipeline.tune            # what to change, on what evidence
       python3 -m pipeline.tune --apply    # pastes the MEASURED values
     Then do EXACTLY what --apply prints. Most pastes need nothing re-run; a
     changed calibration_fit_trailing_weeks needs `--check-only` here, which
     settles the loop WITHOUT retraining. Never retrain for a config paste.
  3. Review artifacts/prior.json -- the prior-acceptance gate is BLOCKING and
     HUMAN; a pooled or uniform prior is a designed outcome.
  4. python3 -m pipeline.status -- "not run" is not a passing check.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
