"""Config loader and validation. config.yaml is the single tuning surface
(design 5.1); modules never carry their own numeric literals.

load_config() permits nulls (bootstrap produces the MEASURED values);
load_config(strict=True) refuses to start while any runtime-required
MEASURED / SET BY OWNER value is null.
"""


import yaml


class ConfigError(RuntimeError):
    pass


# Keys that must be non-null before any price is applied in production.
RUNTIME_REQUIRED = [
    ("data", "launch_date"),          # the switch: set on launch day, null before
]


_MISSING = object()


def config_get(cfg, path, default=_MISSING):
    """The value at a dotted `path` (a tuple of keys). Without `default` a
    missing key raises, as a config read should; with one, a missing or
    non-mapping node returns it -- the one getter tune's findings and
    status's null check share."""
    node = cfg
    for key in path:
        if not isinstance(node, dict) or key not in node:
            if default is _MISSING:
                raise KeyError(".".join(map(str, path)))
            return default
        node = node[key]
    return node


def load_config(path="config.yaml", strict=False):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if strict:
        # THIS FOLDER: the one runtime gate is launch_date. The learning
        # lane's values (rho, tau, the stop thresholds) and its artifacts
        # (the prior) are not read by an exploit-only hour, so the trimmed
        # config carries none of them and nothing here asks for them.
        missing = [".".join(p) for p in RUNTIME_REQUIRED
                   if config_get(cfg, p) is None]
        if missing:
            raise ConfigError(
                "refusing to start: null SET BY OWNER values: " + ", ".join(missing))
    return cfg


def reference_discount(cfg, category):
    """Category anchor d_ref. Config keys use underscores ('SIDE_DISH');
    source data uses spaces ('SIDE DISH')."""
    table = cfg["reference_discount"]
    key = str(category).replace(" ", "_")
    return float(table.get(key, table["_default"]))


# a category's prior is "own data" when the pooled term carries nothing:
# estimate_prior labels it and status counts it on this one threshold
OWN_DATA_WEIGHT = 0.999
