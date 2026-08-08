#!/usr/bin/env bash
# Perishable Markdown MVP -- bootstrap in PRD section 1a order.
# Usage: scripts/run_bootstrap.sh <flc_parquet>
# Stops at each blocking gate exactly as specified: the calibration gate
# (step 5) and the prior-acceptance gate (step 8) must be reviewed by a
# human; this script surfaces their verdicts and continues only through the
# non-gated steps.
set -euo pipefail
INPUT="${1:?usage: scripts/run_bootstrap.sh <flc_parquet>}"

echo "== step 2: bootstrap.prepare_data ==============================="
python3 -m bootstrap.prepare_data --input "$INPUT" --out data/prepared.parquet

echo "== step 3: bootstrap.measure (phase 0) =========================="
python3 -m bootstrap.measure --input "$INPUT" --out reports/phase0.json

echo "== step 4: bootstrap.train_baseline ============================="
python3 -m bootstrap.train_baseline --input data/prepared.parquet

echo "== step 6: bootstrap.fit_dispersion ============================="
python3 -m bootstrap.fit_dispersion --input data/prepared.parquet

echo "== step 7: bootstrap.estimate_prior ============================="
python3 -m bootstrap.estimate_prior --input data/prepared.parquet

echo "== steps 4b/5: backtest -- calibration gate + tau_initial ======="
python3 -m backtest --input data/prepared.parquet --out reports/backtest.json

cat <<'EOF'

Bootstrap complete. Before any price is applied (PRD sections 1a, 19):
  1. Review reports/backtest.json: the calibration gate (fidelity) is
     BLOCKING; resolve baseline_model.apply_level_calibration from the
     level/slope decomposition.
  2. Review artifacts/prior.json: the prior-acceptance gate is BLOCKING;
     a fallback source is an acceptable outcome and is already recorded.
  3. Paste MEASURED values into config.yaml (rho, forced hours, tau_initial,
     il_pct_ratio_se_clustered) and have the owner set the SET BY OWNER keys.
  4. Initialise the posterior from the prior (pricing.posterior
     PosteriorStore.initialise) and proceed to the shadow phase.
EOF
