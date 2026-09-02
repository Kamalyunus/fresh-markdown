"""Print the three bias-scale readings delta_min is derived from.

Read them together: delta_min is set from the LARGEST, and each one alone
understates in a different way.
"""
import json

import numpy as np
import yaml

cfg = yaml.safe_load(open("config.yaml"))
fid = json.load(open("reports/backtest.json"))["fidelity"]
sweep = fid["calibration_window_sweep"]

# TWO DIFFERENT THINGS, and they are routinely not the same:
in_force = cfg["baseline_model"]["calibration_fit_trailing_weeks"]
recommended = sweep.get("recommended_fit_window")
print(f"W IN FORCE (config.yaml)      : trailing_{in_force}w")
print(f"W RECOMMENDED (this sweep)    : {recommended}")
if recommended and recommended != f"trailing_{in_force}w":
    print("  ^^ THEY DISAGREE. The bias scale below is read at the W IN "
          "FORCE, because that is the model actually deployed.\n"
          "     Run `python3 -m pipeline.tune` to see whether it wants the "
          "change (it holds near-ties, and refuses any W the split's\n"
          "     calib >= 2W rule cannot support).")
if sweep.get("uncalibrated_beats_all_windows"):
    print(f"  ^^ {sweep.get('verdict', 'uncalibrated beats every window')}")

row = sweep.get(f"trailing_{in_force}w") or {}
mae = row.get("mean_abs_log_error")
print(f"\n[1] week-aggregate MAE at the W in force : {mae}")
print("    optimistic: anchor rows, aggregated to category x week before the "
      "log, so most noise is averaged out")

bc = fid["by_category"]
lg = np.log(np.array(list(bc.values()), dtype=float))
print("\n[2] cell-level systematic error (by_category, log scale)")
for k, v in sorted(bc.items(), key=lambda kv: np.log(kv[1])):
    print(f"      {k:26s} {v:7.4f}   log {np.log(v):+.4f}")
print(f"    n {len(lg)}   mean|log| {np.abs(lg).mean():.4f}   "
      f"rms {np.sqrt((lg ** 2).mean()):.4f}   std {lg.std(ddof=1):.4f}")
print("    honest: these survive calibration and do NOT average away over "
      "hours -- this is the one that sets the floor")

band = cfg["baseline_model"]["calibration_gate_band"]
print(f"\n[3] accepted level tolerance : band {band} -> "
      f"half-width {np.log(band[1]):.4f} in log space")
print(f"    gate value now: {fid.get('calibration_gate_value')}")

scales = [x for x in (mae, float(np.sqrt((lg ** 2).mean())),
                      float(np.log(band[1]))) if x]
print(f"\nbias scale to use (the largest): {max(scales):.4f}")
print("delta_min = k x (bias scale) / |eps|, per category from [2]")
