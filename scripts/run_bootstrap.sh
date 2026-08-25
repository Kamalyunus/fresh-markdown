#!/usr/bin/env bash
# Perishable Markdown MVP -- bootstrap in AGENTS.md pipeline order.
# Usage: scripts/run_bootstrap.sh <flc_parquet>
#
# Runs steps 1-6, then charts (10) and seal (11). Step numbers below are the
# ones in the AGENTS.md pipeline table -- the single numbering anyone reading
# both is meant to follow.
#
# The prior-acceptance gate (step 4's artifact) is a human review; this
# script produces its evidence and stops. Calibration is ALWAYS fitted and
# applied (owner, 2026-08-25) -- the anchor-level band is a reported
# diagnostic, not a gate.
set -euo pipefail
INPUT="${1:?usage: scripts/run_bootstrap.sh <flc_parquet>}"

echo "== step 1: bootstrap.prepare_data ==============================="
python3 -m bootstrap.prepare_data --input "$INPUT" --out data/prepared.parquet

echo "== step 1b: tools.eda ==========================================="
# Read BEFORE the fits, not after: it describes the population every later
# number is measured on, and it costs seconds. Non-fatal -- a broken panel
# must not stop the pipeline that produces the gates.
python3 -m tools.eda --input data/prepared.parquet || echo "  (eda skipped)"

echo "== step 2: bootstrap.measure (phase 0) =========================="
python3 -m bootstrap.measure --input "$INPUT" --out reports/phase0.json

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

echo "== step 6: backtest -- level diagnostic + tau_initial ==========="
python3 -m backtest --input data/prepared.parquet --out reports/backtest.json

echo "== step 10: tools.make_charts ==================================="
# generated from the reports above, so the pictures can never disagree with
# the numbers. Non-fatal: a missing report is skipped, not an error.
python3 -m tools.make_charts || echo "  (charts skipped)"

echo "== step 10b: tools.refresh_figures =============================="
# THE AGENT HOLDING THE DATA IS THE ONE WITH THE NUMBERS, so refreshing the
# documents belongs in the run rather than in someone's memory. Every anchored
# figure in the docs is rewritten from the artifacts just produced, and the
# run is stamped into each document that changed.
#
# It REFUSES to write from a dataset whose name says it is synthetic, because
# fixture numbers are plausible and silent -- they read as measurements. Run it
# by hand with --allow-fixture only to exercise the tool itself.
#
# Non-fatal: a stale document must not fail a bootstrap that succeeded.
python3 -m tools.refresh_figures --write --dataset "$INPUT" \
    || echo "  (figures not refreshed -- see above; run"\
            "'python3 -m tools.refresh_figures' to see what is stale)"

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

Bootstrap complete. Before any price is applied (design sections 9-10):
  1. Review reports/backtest.json: the level DIAGNOSTIC (calibration is
     always applied; fidelity read on baseline_model.calibration_gate_window).
     Out of band is a drift/staleness reading -- fidelity.by_week separates
     wobble from trend -- not a launch blocker.
  2. Review artifacts/prior.json: the prior-acceptance gate is BLOCKING and
     HUMAN; a pooled or uniform prior is a designed outcome.
  3. Paste MEASURED values into config.yaml (rho, forced hours, tau_initial,
     il_pct_ratio_se_clustered). For the SET BY OWNER keys, produce the
     evidence with:
       python3 -m bootstrap.derive_thresholds --input data/prepared.parquet --mde 0.075
  4. Initialise the posterior and run the shadow phase:
       python3 -m bootstrap.init_posterior
       python3 -m pipeline.shadow --input data/prepared.parquet --out reports/shadow.json
  5. Re-read the summary after each of the above:
       python3 -m pipeline.status
     Every check green is the entry condition for the shadow gate review; a
     check reading "not run" is not a passing check.
EOF
