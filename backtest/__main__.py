import argparse
import json
import os

import pandas as pd

from common.config import load_config
from bootstrap.prepare_data import population, pre_launch
from bootstrap.train_baseline import BaselineModel
from backtest.replay import fidelity, policy_replay, derive_tau_initial


def main():
    ap = argparse.ArgumentParser(prog="backtest")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="reports/backtest.json")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--policy-episodes", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=None,
                    help="processes for the episode replay. 0 = every core "
                         "but one. Episodes are independent and the replay is "
                         "deterministic, so this changes speed and nothing "
                         "else -- the report is identical either way.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = pd.read_parquet(args.input)
    # The backtest is a PRE-LAUNCH artifact and must see nothing past the gate
    # window. Without this, policy_replay and derive_tau_initial ran over the
    # hold-out too -- so tau_initial, a MEASURED launch value, was being fitted
    # on the one window reserved for grading it.
    before = d.episode_id.nunique()
    d = pre_launch(d, cfg)
    excluded = before - d.episode_id.nunique()
    # The DP cannot price an ineligible episode, and extend_to_window
    # refuses a counter above the cap -- so this filter is a
    # precondition, not a population choice. It must precede fidelity(),
    # which is where the extension happens.
    on_dp = d.episode_id.nunique()
    d = population(d, cfg, "dp_eligible")
    dp_excluded = on_dp - d.episode_id.nunique()
    if d.empty:
        raise SystemExit(
            f"no episodes opened on or before split.test_end "
            f"({cfg['data']['split']['test_end']})")
    model = BaselineModel(cfg)
    with open(cfg["posterior"]["prior"]["path"]) as f:
        prior = json.load(f)
    with open(cfg["dispersion"]["r_lookup_path"]) as f:
        r_lookup = json.load(f)

    fid, d_pred = fidelity(d, cfg, model, prior, r_lookup)
    pol, ep, ledger = policy_replay(d_pred, cfg,
                                    max_episodes=args.policy_episodes,
                                    seed=args.seed, workers=args.workers)
    tau = derive_tau_initial(ledger, ep, cfg)

    out = {
        "population": {
            "episodes": int(d.episode_id.nunique()),
            "episodes_excluded_after_test_end": int(excluded),
            "episodes_excluded_dp_ineligible": int(dp_excluded),
            "sees_up_to": cfg["data"]["split"]["test_end"],
            "note": ("The backtest is pre-launch and sees nothing past the "
                     "gate window, so by_week and by_window['all'] stop at "
                     "test_end. The hold-out is read once, by "
                     "`pipeline.shadow --holdout`."),
        },
        "artifact_versions": {
            "baseline_model_version": model.version,
            "train_population": cfg["baseline_model"]["train_population"],
            "prior_source": prior["source"],
            "config_version": cfg["meta"]["config_version"],
        },
        # two blocks, reported separately and never summed (design 5.14)
        "fidelity": fid,
        "policy_deltas": pol,
        "tau_initial_derivation": tau,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"fidelity_episode_sold_ratio : {fid['fidelity_episode_sold_ratio']}")
    print(f"gate ({fid['calibration_gate_metric']}) : "
          f"{fid['calibration_gate_value']} vs {fid['calibration_gate_band']}"
          f"  -> {fid['calibration_gate']}")
    gap = pol["policy_gap_like_for_like"]
    print(f"observed world  : legacy IL {pol['actual_il']:,.0f} "
          f"(IL% {pol['actual_il_pct']})")
    print(f"model world     : legacy IL {pol['legacy_model_il']:,.0f} "
          f"vs DP IL {pol['dp_il']:,.0f}  -> DP reduces IL by "
          f"{gap['dp_il_reduction_pct_of_legacy']:.1%} (like-for-like)"
          if gap["dp_il_reduction_pct_of_legacy"] is not None else
          "model world     : like-for-like gap unavailable")
    print(f"pct_dp_deepened             : {pol['pct_dp_deepened']:.1%}")
    if tau:
        print(f"tau_initial (currency)      : {tau['tau_initial']}  "
              f"(q{tau['cost_distribution_quantile']:.2f} of Q-spread; "
              f"daily spend {tau['implied_daily_spend']:,.0f} "
              f"vs budget {tau['daily_budget']:,.0f})")
        print("paste into config.yaml: exploration.tau_initial")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
