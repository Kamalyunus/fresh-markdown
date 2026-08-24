#!/usr/bin/env bash
# Perishable Markdown MVP -- bootstrap in AGENTS.md pipeline order.
# Usage: scripts/run_bootstrap.sh <flc_parquet>
#
# Runs steps 1-6, then charts (10) and seal (11). Step numbers below are the
# ones in the AGENTS.md pipeline table -- the single numbering anyone reading
# both is meant to follow.
#
# The two blocking gates are NOT decided here: the calibration gate (step 6's
# report) and the prior-acceptance gate (step 5's artifact) are human reviews.
# This script produces their evidence and stops.
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

# ORDER: the prior comes FIRST, and it is not arbitrary. `fit_dispersion`
# forms residuals at a working elasticity, so running it first meant fitting r
# and rho at the fallback constant -- fine while the bracket was rejected for
# every category, wrong once brackets are accepted per category. The two steps
# are circular (the bracket's likelihood needs r), and the loop is cut on the
# prior's side because it is far weaker there: the bracket uses a reference r
# and drops its censored entry rows, costing ~0.099 against stds of 0.4-1.7,
# where dispersion at the wrong elasticity costs 26% of the learning rate.
echo "== step 4: bootstrap.estimate_prior ============================="
python3 -m bootstrap.estimate_prior --input data/prepared.parquet

echo "== step 5: bootstrap.fit_dispersion ============================="
python3 -m bootstrap.fit_dispersion --input data/prepared.parquet

echo "== step 6: backtest -- calibration gate + tau_initial ==========="
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

Bootstrap complete. Before any price is applied (PRD sections 1a, 19):
  1. Review reports/backtest.json: the calibration gate (fidelity, read on
     the window named by baseline_model.calibration_gate_window) is BLOCKING;
     resolve baseline_model.apply_level_calibration from the level/slope
     decomposition. The remedy, if needed:
       python3 -m bootstrap.train_baseline --input data/prepared.parquet --fit-calibration
       python3 -m bootstrap.seal      # calibration.json is new -- RE-SEAL
     --fit-calibration writes a seventh artifact into the bundle, so the seal
     taken above no longer describes the set. Re-run it or the bundle line
     stays stale.
  2. Review artifacts/prior.json: the prior-acceptance gate is BLOCKING;
     a fallback source is an acceptable outcome and is already recorded.
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
