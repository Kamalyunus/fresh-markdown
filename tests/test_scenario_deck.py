"""tools/scenario_deck: the leadership deck is built from real solves and stays self-consistent."""
import json
import re

import numpy as np
import pytest

from tools import scenario_deck as sd


@pytest.fixture(scope="module")
def deck():
    from common.config import load_config
    return sd.build(load_config("config.yaml"), sd.QUICK, workers=0)


def test_twelve_scenarios_each_land_on_a_precomputed_state(deck):
    assert len(sd.SCENARIOS) == 12
    keys = {tuple(s["key"]) for s in deck["states"]}
    for s in sd.SCENARIOS:
        st = s["state"]
        # QUICK is a subset of GRID; every scenario's opening state must exist in the full grid
        assert st["q"] in sd.GRID["q"] and st["h"] in sd.GRID["h"]
        assert st["gamma"] in sd.GRID["gamma"] and st["mu"] in sd.GRID["mu"]
        assert st["belief"] in deck["beliefs"]
    assert keys  # quick grid produced states


def test_every_state_has_paths_scores_and_monotone_discounts(deck):
    for st in deck["states"]:
        assert st["star"] in st["q_by_tier"]
        for name in ("dp", "dp_world_half", "dp_world_double", "dp_restock", "flat_reference", "legacy_ramp"):
            p = st["paths"][name]
            assert set(p["score"]) == {"leftover", "scrap_cost", "discount_cost", "il", "sold"}
            assert p["score"]["il"] == pytest.approx(p["score"]["scrap_cost"] + p["score"]["discount_cost"], abs=1.0)
        ds = [r["d"] for r in st["paths"]["dp"]["path"] if r["d"] is not None]
        assert all(b >= a - 1e-9 for a, b in zip(ds, ds[1:])), "system path must never raise the price"
        # once sold out, no price is emitted
        rows = st["paths"]["dp"]["path"]
        assert all(r["d"] is None for r in rows if r["q"] == 0)


def test_page_embeds_valid_json_and_no_placeholders(deck, cfg, tmp_path):
    from common.provenance import config_fingerprint
    out = tmp_path / "deck.html"
    sd.write_page(deck, cfg, out)
    html = out.read_text()
    assert "__DATA__" not in html and "__CFGV__" not in html and "__CFGD__" not in html
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S)
    data = json.loads(m.group(1).replace("<\\/", "</"))
    assert len(data["scenarios"]) == 12
    assert data["config"]["config_version"] == cfg["meta"]["config_version"]
    # the version string rarely moves; the digest moves on every paste, and
    # the header shows both so a deck can be traced to its config
    digest = config_fingerprint(cfg)["digest"]
    assert data["config"]["config_digest"] == digest and digest in html
    assert "engine.dp.solve" in html


def test_the_deck_reads_its_inputs_strictly_and_derives_the_learned_belief(
        cfg, monkeypatch):
    """A default dispersion or delta_min multiple is an invented number on
    a page every figure of which claims to be the real solver's; the deck
    fails loudly instead. The learned belief is one cap-sized update
    (learning.max_mean_step) steeper than launch, floored at epsilon_min."""
    import copy
    from engine.posterior import launch_belief

    prior = {"per_category": {"A": {"mean": -1.0, "std": 0.4},
                              "B": {"mean": -2.0, "std": 0.4}}}
    monkeypatch.setattr(sd, "read_json", lambda path: prior)
    cfg = copy.deepcopy(cfg)
    cfg["learning"]["max_mean_step"] = 0.3
    cold = float(np.mean([launch_belief(v["mean"], v["std"], cfg)
                          for v in prior["per_category"].values()]))
    bel = sd.beliefs(cfg)
    assert bel["cold"] == pytest.approx(cold, abs=1e-3)
    assert bel["learned"] == pytest.approx(
        max(cold - 0.3, cfg["posterior"]["epsilon_min"]), abs=1e-3)
    cfg["posterior"]["epsilon_min"] = cold - 0.1                # floor binds
    assert sd.beliefs(cfg)["learned"] == pytest.approx(cold - 0.1, abs=1e-3)

    # no global r on disk -> refused by name, never 0.9
    monkeypatch.setattr(sd, "read_json", lambda path: {})
    with pytest.raises(SystemExit, match="global r"):
        sd.build(cfg, sd.QUICK, workers=0)
    # a config without the multiple is refused, never read as 1.0
    monkeypatch.setattr(sd, "read_json", lambda path: {"global": 1.0})
    del cfg["exploration"]["delta_min_bias_multiple"]
    with pytest.raises(KeyError, match="delta_min_bias_multiple"):
        sd.build(cfg, {"q": [2], "h": [2], "gamma": [0.5], "mu": [1.0]}, workers=0)


def test_no_fixed_schedule_prices_below_cost(deck):
    """The legacy ramp deepened to d_ref + 0.15 whatever d_max was, so at high
    COGS the comparison arm sold below cost -- a price the system itself can
    never emit. Every fixed schedule is capped at d_max (dp.feasible_tiers)."""
    assert any(g >= 0.7 for g in deck["grid"]["gamma"]), "no high-COGS state on the grid"
    for st in deck["states"]:
        gamma = st["key"][0]
        for name in ("flat_reference", "legacy_ramp"):
            for r in st["paths"][name]["path"]:
                if r["d"] is None:
                    continue
                assert r["d"] <= st["d_max"] + 1e-9, (name, st["key"], r)
                assert sd.P0 * (1 - r["d"]) >= sd.P0 * gamma - 1e-6
    # the ramp still ramps where it can
    assert sd.legacy_ramp(0.30, 12, 0.60) == pytest.approx(
        [0.15] * 4 + [0.30] * 4 + [0.45] * 4)
    assert sd.legacy_ramp(0.30, 12, 0.30) == pytest.approx([0.15] * 4 + [0.30] * 8)


def test_the_refusals_state_the_stop_rules_the_monitor_applies(deck, cfg, tmp_path):
    """daily.monitor: every stop -- overspend, scrap, margin -- needs
    persistence_days consecutive priced days over its threshold
    (evaluate_guardrail); the deck must say what the monitor does."""
    out = tmp_path / "deck.html"
    sd.write_page(deck, cfg, out)
    html = out.read_text()
    assert "Every stop needs ${D.config.persistence_days} consecutive days" in html
    assert "a single day is enough" not in html
