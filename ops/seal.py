"""ops.seal -- declare the frozen artifacts a bundle, and hash them.

Stamps catch a MIXED bundle but not a post-hoc edit (an editor leaves the
stamp intact); sealing records per-file hashes beside the agreed bundle id so
both failures are detectable and distinguishable. Refuses an inconsistent set
-- a sealed mixed bundle looks decided. The seal also records the ENVIRONMENT
a priced hour depends on beyond the artifacts -- the config (digest and
snapshot) and the library versions -- and the posterior as it stands
(recorded, never verified: it is learning state). verify() reads a moved
config or library as a problem exactly like an edited artifact; the remedy
is a deliberate re-seal. Every seal also copies the
bundle, config and posterior into artifacts/history/<bundle>/<sealed_at>/.
Run: python3 -m ops.seal [--reason bootstrap|retrain|check-only|weekly-refit|config|libraries]
"""

import argparse

from common.config import load_config
from common.io import write_json
from common import provenance


def seal(cfg):
    state = provenance.verify(cfg)          # no seal passed: the set as it stands
    if state["verdict"] != "PASS":
        raise SystemExit(
            "refusing to seal: " + ("; ".join(state["problems"])
                                    or "no stamped artifacts to seal"))
    fp = provenance.config_fingerprint(cfg)
    return {
        "bundle": state["bundle"],
        "sealed_at": provenance.datetime.now(provenance.timezone.utc).isoformat(),
        "config_version": cfg["meta"]["config_version"],
        "sha256": {r["artifact"]: r["sha256"]
                   for r in state["artifacts"] if r["present"]},
        "missing": state["missing"],
        "environment": provenance.environment(cfg),
        "config_snapshot": fp["snapshot"],
        "launch_posterior": provenance.launch_posterior(cfg),
    }


def main():
    ap = argparse.ArgumentParser(prog="ops.seal", description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--reason", default=None,
                    help="why this seal happened (bootstrap, retrain, "
                         "check-only, weekly-refit, config, "
                         "libraries); recorded in the history MANIFEST")
    args = ap.parse_args()

    cfg = load_config(args.config)
    payload = seal(cfg)
    path = cfg["artifacts"]["bundle_path"]
    write_json(path, payload)
    snap = provenance.archive(cfg, payload, config_path=args.config,
                              reason=args.reason)
    print(f"sealed bundle {payload['bundle']}  ->  audit copy {snap}")
    for name, digest in payload["sha256"].items():
        print(f"  {name:16s} {digest[:12]}")
    env = payload["environment"]
    print(f"  {'config':16s} {env['config_digest']}")
    print(f"  {'libraries':16s} " + ", ".join(f"{k} {v}" for k, v in env["libraries"].items()))
    lp = payload["launch_posterior"]
    print(f"  {'posterior':16s} " + (f"{lp['digest'][:12]}, {len(lp['cells'])} cells, "
                                     f"{lp['outcomes_consumed']} outcomes consumed"
                                     if lp else "absent"))
    if payload["missing"]:
        print("  absent: " + ", ".join(payload["missing"]))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
