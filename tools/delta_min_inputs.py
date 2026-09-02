"""Print the three bias-scale readings delta_min is derived from."""
import json, yaml, numpy as np

cfg = yaml.safe_load(open("config.yaml"))
fid = json.load(open("reports/backtest.json"))["fidelity"]
W = cfg["baseline_model"]["calibration_fit_trailing_weeks"]

sweep = fid["calibration_window_sweep"]
print(f"chosen W = {W}")
print(json.dumps(sweep, indent=2))

bc = fid["by_category"]
lg = np.log(np.array(list(bc.values()), dtype=float))
print("\nby_category, log scale (the cell-level systematic error):")
for k, v in sorted(bc.items(), key=lambda kv: np.log(kv[1])):
    print(f"  {k:26s} {v:7.4f}   log {np.log(v):+.4f}")
print(f"\n  n {len(lg)}   mean|log| {np.abs(lg).mean():.4f}   "
      f"rms {np.sqrt((lg**2).mean()):.4f}   std {lg.std(ddof=1):.4f}")

band = cfg["baseline_model"]["calibration_gate_band"]
print(f"\ngate: value {fid.get('calibration_gate_value')} "
      f"band {band} -> half-width in log {np.log(band[1]):.4f}")
print(f"week-aggregate MAE at W={W}: "
      f"{(sweep.get(f'trailing_{W}w') or {}).get('mean_abs_log_error')}")
