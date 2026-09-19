"""ops.seal -- declare the frozen artifacts a bundle, and hash them.

Stamps catch a MIXED bundle but not a post-hoc edit (an editor leaves the
stamp intact); sealing records per-file hashes beside the agreed bundle id so
both failures are detectable and distinguishable. Refuses an inconsistent set
-- a sealed mixed bundle looks decided. The seal also records the ENVIRONMENT
a priced hour depends on beyond the artifacts -- the config (digest and
snapshot) and the library versions -- and the posterior as it stands
(recorded, never verified: it is learning state). verify() reads a moved
config or library as a problem exactly like an edited artifact; the remedy
is a deliberate re-seal. A re-seal carries its REASON, and the reason says
which artifacts may have moved since the previous seal: none under
`config` or `libraries` (the environment moved, the artifacts must not
have -- `advance` runs these on its own, and an automatic re-seal must
never bless a hand-edited or hand-refitted artifact as the record), the
calibration alone under `weekly-refit`; a fit reason (`bootstrap`,
`retrain`, `check-only`) or no reason seals the set as it stands. Every
seal also copies the bundle, config and posterior into
artifacts/history/<bundle>/<sealed_at>/.
Run: python3 -m ops.seal [--reason bootstrap|retrain|check-only|weekly-refit|config|libraries]
"""

from common.cli import make_parser
from common.config import load_config
from common.io import write_json
from common import history, provenance

# what each re-seal reason allows to have moved since the previous seal:
# a reason not listed here (a fit, or none given) allows everything
MAY_MOVE = {"config": frozenset(), "libraries": frozenset(),
            "weekly-refit": frozenset({"calibration"})}


def moved_artifacts(cfg, previous):
    """Artifacts whose hash differs from `previous` (the seal on disk), or
    that it sealed and are gone, or that were fitted after it: the names
    verify() reports, read from the problems it names."""
    if not previous:
        return []
    state = provenance.verify(cfg, previous)
    moved = []
    for p in state["problems"]:
        for head in ("changed since sealing: ", "sealed but no longer on disk: ",
                     "fitted after sealing (re-seal): "):
            if p.startswith(head):
                moved += [n.strip() for n in p[len(head):].split(",")]
    return sorted(set(moved))


def seal(cfg, reason=None):
    state = provenance.verify(cfg)          # no seal passed: the set as it stands
    if state["verdict"] != "PASS":
        raise SystemExit(
            "refusing to seal: " + ("; ".join(state["problems"])
                                    or "no stamped artifacts to seal"))
    if reason in MAY_MOVE:
        # the previous seal's hashes are the record this re-seal continues:
        # an artifact that moved under a reason that does not fit one is
        # refused, never blessed under `config`
        moved = [n for n in moved_artifacts(cfg, provenance.load_seal(cfg))
                 if n not in MAY_MOVE[reason]]
        if moved:
            raise SystemExit(
                f"refusing to re-seal under --reason {reason}: "
                + ", ".join(moved) + " moved since the previous seal, and a "
                f"{reason} re-seal may not bless a changed artifact. If the "
                "change was deliberate, seal it under the reason that made "
                "it (bootstrap, retrain, check-only, weekly-refit); if it was "
                "not, restore the artifact from the latest "
                "artifacts/history/<bundle>/<sealed_at>/ snapshot")
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
    ap = make_parser(prog="ops.seal", description=__doc__)
    ap.add_argument("--reason", default=None,
                    help="why this seal happened (bootstrap, retrain, "
                         "check-only, weekly-refit, config, "
                         "libraries); recorded in the history MANIFEST, and "
                         "config/libraries/weekly-refit refuse an artifact "
                         "that moved since the previous seal")
    args = ap.parse_args()

    cfg = load_config(args.config)
    payload = seal(cfg, reason=args.reason)
    path = cfg["artifacts"]["bundle_path"]
    write_json(path, payload)
    snap = history.archive(cfg, payload, config_path=args.config,
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
    # every seal is a new frozen state: the pricing folder engineering runs
    # carries it from here, never by a hand copy (ops.integration.sync)
    from ops import integration                                  # noqa: E402
    integration.sync_and_print(cfg, config_path=args.config)


if __name__ == "__main__":
    main()
