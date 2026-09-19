"""The pruned config, the category anchors, the digest a decision names."""
import hashlib
import json

import yaml


def load_config(path="config.yaml"):
    """The folder's config as synced: no gate, no learning-lane value; the
    keys the two commands read and nothing else."""
    with open(path) as f:
        return yaml.safe_load(f)


def reference_discount(cfg, category):
    """The category anchor d_ref; config keys spell 'SIDE DISH' as SIDE_DISH."""
    table = cfg["reference_discount"]
    return float(table.get(str(category).replace(" ", "_"), table["_default"]))


def config_digest(cfg):
    """The digest every decision records: which config priced it."""
    canon = json.dumps(cfg, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:16]
