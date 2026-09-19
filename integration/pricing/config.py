"""The pruned config, the launch gate, the category anchors, the digest a
decision names."""
import hashlib
import json

import yaml


class ConfigError(RuntimeError):
    pass


def load_config(path="config.yaml", strict=False):
    """`strict` refuses to price while `data.launch_date` is null: the one
    runtime gate this folder keeps (the learning lane's values are not read
    by an exploit-only hour and the config does not carry them)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if strict and cfg["data"].get("launch_date") is None:
        raise ConfigError("refusing to start: null SET BY OWNER values: data.launch_date")
    return cfg


def reference_discount(cfg, category):
    """The category anchor d_ref; config keys spell 'SIDE DISH' as SIDE_DISH."""
    table = cfg["reference_discount"]
    return float(table.get(str(category).replace(" ", "_"), table["_default"]))


def config_digest(cfg):
    """The digest every decision records: which config priced it."""
    canon = json.dumps(cfg, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:16]
