"""bootstrap.seal -- declare the frozen artifacts a bundle, and hash them.

Provenance stamps say which model each artifact was fitted against, which
catches a MIXED bundle. They cannot catch an artifact that was edited after the
fact -- an editor leaves the stamp intact. So sealing records a hash of every
file alongside the agreed bundle id, and from then on both failures are
detectable and distinguishable.

Run once, when a set of artifacts becomes the one production will use:

    python3 -m bootstrap.seal

It refuses to seal a set that is not internally consistent, because a sealed
mixed bundle is worse than an unsealed one: it looks decided.
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
