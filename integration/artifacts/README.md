# artifacts/

The five files the hourly command opens, copied from the owner's repository
(its own `artifacts/`), under the names `config.yaml` gives them:

| File | Config key | What |
| --- | --- | --- |
| `baseline_model.txt` | `baseline_model.model_path` | the frozen demand model (LightGBM) |
| `feature_schema.json` | `baseline_model.feature_schema_path` | its feature list and category levels |
| `calibration.json` | `baseline_model.calibration_factor_path` | the weekly level factors |
| `r_lookup.json` | `dispersion.r_lookup_path` | the negative-binomial dispersion by category |
| `posterior.json` | `posterior.path` | the elasticity belief per cell |

The owner puts them here, by `python3 -m ops.integration --sync` from the
repository or by hand after a retrain; `synced.json` records what arrived
and when. Nothing here is written by the folder's commands.
