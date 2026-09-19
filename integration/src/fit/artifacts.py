"""fit.artifacts -- the one way the frozen bundle is loaded for pricing or grading.

Every caller that prices or grades needs the same four things -- the frozen
model, the posterior store, the `r` lookup and the prior -- and each once
spelt the loads itself (seven copies, two wordings for a missing file).
`load_bundle(cfg)` returns them as one object whose parts load on first
use, so a caller that needs only the prior never opens the model.
"""

import functools

from common.io import read_json


class Bundle:
    """The frozen artifacts behind `cfg`, each read on first access and
    kept: `model` (fit.model.BaselineModel), `posterior`
    (engine.posterior.PosteriorStore, read once here -- a long-lived caller
    reloads it per batch), `r_lookup` and `prior` (the JSON artifacts). A
    JSON artifact that is not on disk raises FileNotFoundError naming its
    path, the wording ops.price_batch refuses a batch with."""

    def __init__(self, cfg):
        self.cfg = cfg

    @functools.cached_property
    def model(self):
        from fit.model import BaselineModel        # lightgbm: on demand
        return BaselineModel(self.cfg)

    @functools.cached_property
    def posterior(self):
        from engine.posterior import PosteriorStore
        return PosteriorStore(self.cfg)

    @functools.cached_property
    def r_lookup(self):
        return self._json(self.cfg["dispersion"]["r_lookup_path"])

    @staticmethod
    def _json(path):
        out = read_json(path)
        if out is None:
            raise FileNotFoundError(path)
        return out


def load_bundle(cfg):
    """The bundle `cfg` names, parts loaded lazily (see Bundle)."""
    return Bundle(cfg)


def lookup_r(r_lookup, subcategory, category):
    """r for one row of the loaded `r_lookup`, down its `fallback_order`
    (explicit None tests: 0.0 is a value, not a miss). The one scalar
    reader of the table; fit.fit_dispersion writes it and vectorises this
    for the fits."""
    keys = {"subcategory": str(subcategory), "category": str(category)}
    for level in r_lookup["fallback_order"]:
        if level == "global":
            return r_lookup["global"]
        r = r_lookup[level].get(keys[level])
        if r is not None:
            return r
    return r_lookup["global"]
