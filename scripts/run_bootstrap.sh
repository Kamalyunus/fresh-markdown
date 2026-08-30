#!/usr/bin/env bash
# Perishable Markdown MVP -- bootstrap in AGENTS.md pipeline order.
# Usage: scripts/run_bootstrap.sh <flc_parquet>
#
# Runs steps 1-6 then seal (11). Step numbers below are the
# ones in the AGENTS.md / design.md Appendix A pipeline table -- the single
# numbering anyone reading both is meant to follow.
#
# The prior-acceptance gate (step 4's artifact) is a human review; this
# script produces its evidence and stops. Calibration is ALWAYS fitted and
# applied (owner, 2026-08-25) -- the anchor-level band is a reported
# diagnostic, not a gate.
set -euo pipefail
INPUT="${1:?usage: scripts/run_bootstrap.sh <flc_parquet>}"

echo "== step 1: bootstrap.prepare_data ==============================="
python3 -m bootstrap.prepare_data --input "$INPUT" --out data/prepared.parquet

echo "== step 3: bootstrap.train_baseline ============================="
python3 -m bootstrap.train_baseline --input data/prepared.parquet

echo "== step 3b: level calibration (always fitted and applied) ======="
# Factors are fit on anchor rows only, so a slope error cannot contaminate
# them, and every downstream fit and replay reads the calibrated mu_ref.
# The anchor-level band stays a DIAGNOSTIC in the backtest report.
python3 -m bootstrap.train_baseline --input data/prepared.parquet --fit-calibration

# ORDER: the prior comes FIRST, and it is not arbitrary. `fit_dispersion`
# forms residuals at a working elasticity -- each category's prior mean -- so
# the prior must exist before r and rho are fitted (the epsilon -> r
# direction costs 26% of the learning rate; the r -> epsilon direction is
# zero, since the prior's profile is censored Poisson and needs no r).
echo "== step 4: bootstrap.estimate_prior ============================="
python3 -m bootstrap.estimate_prior --input data/prepared.parquet

echo "== step 5: bootstrap.fit_dispersion ============================="
python3 -m bootstrap.fit_dispersion --input data/prepared.parquet

echo "== step 5b: calibration <-> dispersion convergence =============="
# The chain is CIRCULAR across runs: the factor solve consumes r, and r, rho
# and the prior are fitted against calibrated mu_ref. Steps 3b-5 are one
# iteration of that loop; this asserts one more turn would reproduce the
# factors (dry run -- the artifact is restored, so the chain on disk stays
# the one steps 4-5 were fitted against). NOT CONVERGED -> run 3b, 4, 5, 5b
# again. Non-fatal: the status line below carries the verdict.
python3 -m bootstrap.train_baseline --input data/prepared.parquet \
    --check-convergence || echo "  (convergence check errored -- see above)"

echo "== step 6: backtest -- level diagnostic + tau cross-check ======="
python3 -m backtest --input data/prepared.parquet --out reports/backtest.json

echo "== step 6b: bootstrap.derive_thresholds ========================="
# pipeline.tune reads this: the increment, both rails and the guardrail floors
# are all measured here, so the tuning loop cannot run without it.
python3 -m bootstrap.derive_thresholds --input data/prepared.parquet --mde 0.075

echo "== step 11: bootstrap.seal ======================================"
# Everything above wrote a fresh artifact against a fresh model, so this set
# is the bundle. Non-fatal on purpose: a refusal means the artifacts on disk
# are not internally consistent, and the `artifact bundle` line in the status
# view immediately below says so in the same place every other gate reports.
python3 -m bootstrap.seal || echo "  (seal REFUSED -- see the artifact bundle line below)"

echo "== where this run stands: pipeline.status ======================="
# Exits 1 while anything is red, which is the expected state right after a
# bootstrap (tau not pasted yet, shadow not run yet), so it must not abort the
# script. Read it as the summary, not the verdict.
python3 -m pipeline.status || true

cat <<'EOF'

Bootstrap complete. Next:
  1. Run the hold-out shadow phase (it derives its own launch tau):
       python3 -m bootstrap.init_posterior
       python3 -m pipeline.shadow --input data/prepared.parquet --out reports/shadow.json
  2. Let the reports decide the config, and record why:
       python3 -m pipeline.tune            # what to change, on what evidence
       python3 -m pipeline.tune --apply    # pastes the MEASURED values
     Then re-run this script: a changed increment or rail changes what the
     next run measures. Iterate until `tune` reports no PASTE and no BLOCK.
  3. Review artifacts/prior.json -- the prior-acceptance gate is BLOCKING and
     HUMAN; a pooled or uniform prior is a designed outcome.
  4. python3 -m pipeline.status -- every check green is the entry condition
     for the shadow gate review; "not run" is not a passing check.
EOF
