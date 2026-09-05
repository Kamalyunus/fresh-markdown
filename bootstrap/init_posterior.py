"""bootstrap.init_posterior -- create posterior.json from the prior artifact.

One-time launch step (design 5.9): initialises every learning cell from the
9.5 prior, assigns categories to cells from phase-0 weekly volumes; assignment
does not move during the MVP window. Refuses to overwrite without --force --
the posterior is production learning state.
Run: python3 -m bootstrap.init_posterior [--force]
"""

import argparse
import json
import os

from common.config import load_config
from pricing.posterior import PosteriorStore


def main():
    ap = argparse.ArgumentParser(prog="bootstrap.init_posterior")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing posterior (discards all "
                         "learning state)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    path = cfg["posterior"]["path"]
    if os.path.exists(path) and not args.force:
        raise SystemExit(f"{path} already exists -- this is production learning "
                         "state. Re-run with --force only if you mean to "
                         "discard it.")

    with open(cfg["posterior"]["prior"]["path"]) as f:
        prior = json.load(f)

    store = PosteriorStore.initialise(cfg, prior["per_category"],
                                      prior["episodes_per_week"])
    print(f"prior source: {prior['source']}  |  launch belief = prior mean - "
          f"{store.state['cold_start_shift_std']} std (posterior.cold_start_shift_std)")
    for cell, rec in store.state["cells"].items():
        members = [c for c, target in store.state["cell_of"].items()
                   if target == cell]
        print(f"  {cell:16s} prior {rec['prior_mean']:+.3f} -> launch "
              f"{rec['mean']:+.3f} std {rec['std']:.3f}"
              + (f"  <- {', '.join(sorted(members))}" if cell == "GLOBAL"
                 and members else ""))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
