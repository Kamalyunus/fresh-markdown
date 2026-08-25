"""bootstrap.seal -- declare the frozen artifacts a bundle, and hash them.

Stamps catch a MIXED bundle but not a post-hoc edit (an editor leaves the
stamp intact); sealing records per-file hashes beside the agreed bundle id so
both failures are detectable and distinguishable. Refuses an inconsistent set
-- a sealed mixed bundle looks decided. Run once per production bundle:
python3 -m bootstrap.seal
"""

import argparse
import json
import os

from common.config import load_config
from common import provenance


def seal(cfg):
    state = provenance.verify(cfg)
    if state["verdict"] != "PASS":
        raise SystemExit(
            "refusing to seal: " + ("; ".join(state["problems"])
                                    or "no stamped artifacts to seal"))
    return {
        "bundle": state["bundle"],
        "sealed_at": provenance.datetime.now(provenance.timezone.utc).isoformat(),
        "config_version": cfg["meta"]["config_version"],
        "sha256": {r["artifact"]: r["sha256"]
                   for r in state["artifacts"] if r["present"]},
        "missing": state["missing"],
    }


def main():
    ap = argparse.ArgumentParser(prog="bootstrap.seal", description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    payload = seal(cfg)
    path = cfg["artifacts"]["bundle_path"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"sealed bundle {payload['bundle']}")
    for name, digest in payload["sha256"].items():
        print(f"  {name:16s} {digest[:12]}")
    if payload["missing"]:
        print("  absent: " + ", ".join(payload["missing"]))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
