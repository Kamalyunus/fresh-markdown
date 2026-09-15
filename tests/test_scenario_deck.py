"""tools/scenario_deck: the leadership deck is built from real solves and stays self-consistent."""
import json
import re

import numpy as np
import pytest

from tools import scenario_deck as sd


@pytest.fixture(scope="module")
def deck_cfg(tmp_path_factory):
    """The shipped config pointed at throwaway artifacts -- a global r and a
    two-category prior written here, so the deck builds on a checkout with
    no artifacts/ at all (it once read whatever was on disk)."""
    import os
    from common.config import load_config
    from conftest import ROOT
    root = tmp_path_factory.mktemp("deck")
    (root / "r_lookup.json").write_text(json.dumps(
        {"fallback_order": ["subcategory", "category", "global"],
         "subcategory": {}, "category": {}, "global": 2.0}))
    (root / "prior.json").write_text(json.dumps(
        {"per_category": {"A": {"mean": -1.0, "std": 0.4},
                          "B": {"mean": -1.6, "std": 0.4}}}))
    cfg = load_config(os.path.join(ROOT, "config.yaml"))
    cfg["dispersion"]["r_lookup_path"] = str(root / "r_lookup.json")
    cfg["posterior"]["prior"]["path"] = str(root / "prior.json")
    return cfg


@pytest.fixture(scope="module")
def deck(deck_cfg):
    return sd.build(deck_cfg, sd.QUICK, workers=0)


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
        # the paths walk the replay's one forward simulation
        # (evaluate.backtest._simulate_arm, priced by _dp_price): its entry
        # solve is the state's own, so the system path opens at the chosen
        # tier and the restock path shares it until the delivery lands
        opening = st["paths"]["dp"]["path"][0]
        assert opening["d"] == st["tiers"][st["star"]] and opening["q"] == st["key"][3]
        h = st["key"][4]
        assert (st["paths"]["dp_restock"]["path"][:max(h // 2, 1)]
                == st["paths"]["dp"]["path"][:max(h // 2, 1)])
        for name in ("dp", "dp_world_half", "dp_world_double", "dp_restock", "flat_reference", "legacy_ramp"):
            p = st["paths"][name]
            assert set(p["score"]) == {"leftover", "scrap_cost", "discount_cost", "il", "sold"}
            assert p["score"]["il"] == pytest.approx(p["score"]["scrap_cost"] + p["score"]["discount_cost"], abs=1.0)
        ds = [r["d"] for r in st["paths"]["dp"]["path"] if r["d"] is not None]
        assert all(b >= a - 1e-9 for a, b in zip(ds, ds[1:])), "system path must never raise the price"
        # once sold out, no price is emitted
        rows = st["paths"]["dp"]["path"]
        assert all(r["d"] is None for r in rows if r["q"] == 0)


def test_page_embeds_valid_json_and_no_placeholders(deck, deck_cfg, tmp_path):
    from common.provenance import config_fingerprint
    cfg = deck_cfg
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
    # no prior on disk -> refused by name, never an invented elasticity
    # labelled as the launch belief
    monkeypatch.setattr(sd, "read_json", lambda path: {"global": 1.0})
    with pytest.raises(SystemExit, match="per_category"):
        sd.build(cfg, sd.QUICK, workers=0)
    # a config without the multiple is refused, never read as 1.0
    monkeypatch.setattr(sd, "read_json", lambda path: {"global": 1.0, **prior})
    del cfg["exploration"]["delta_min_bias_multiple"]
    with pytest.raises(KeyError, match="delta_min_bias_multiple"):
        sd.build(cfg, {"q": [2], "h": [2], "gamma": [0.5], "mu": [1.0]}, workers=0)


def test_the_exploration_table_is_solved_by_the_engine_not_the_browser(deck, deck_cfg):
    cfg = deck_cfg
    """delta_min, the admissible tiers and the affordable set on the page
    are engine.explore's own (delta_min, admissible_costs, affordable_set)
    for every slider position, embedded per state; the browser reads
    them. A JS copy of the floor once used `_default` for a per-category
    map and could not follow a change to the rule."""
    import copy
    from engine import dp as dp_mod
    from engine import explore

    st = next(s for s in deck["states"] if s["key"][2] == "cold")
    gamma, mu, _, q, h = st["key"]
    eps = deck["beliefs"]["cold"]
    res = dp_mod.solve(sd.P0, sd.P0 * gamma, q, [mu] * h, deck["d_ref"], eps,
                       deck["r"], cfg, anchor_discount=None, entry=True)
    x = st["explore"]
    assert set(x["by_dmin_scale"]) == {str(s) for s in sd.DMIN_SCALES}
    assert deck["taus"] == sd.TAUS and deck["dmin_scales"] == sd.DMIN_SCALES
    for scale in sd.DMIN_SCALES:
        c = cfg if scale == 0 else copy.deepcopy(cfg)
        if scale:
            c["exploration"]["delta_min_log_bias"] = scale
        dmin = explore.delta_min(c, eps, "_default")
        block = x["by_dmin_scale"][str(scale)]
        assert block["delta_min"] == pytest.approx(dmin, abs=1e-4)
        assert block["admissible"] == sorted(explore.admissible(res, dmin))
        for tau in sd.TAUS:
            assert block["affordable_by_tau"][str(tau)] == sorted(
                explore.affordable_set(res, tau, dmin)[0])
    for j, cost in x["cost"].items():
        assert cost == pytest.approx(res.q_by_tier[res.optimal_index]
                                     - res.q_by_tier[int(j)], abs=0.1)
        assert x["log_move"][j] == pytest.approx(
            explore.log_move(deck["d_ref"], res.tiers[int(j)]), abs=1e-4)
    # the config's own floor is what scale 0 shows, so a paste reaches the deck
    assert x["by_dmin_scale"]["0"]["delta_min"] == pytest.approx(
        explore.delta_min(cfg, eps, "_default"), abs=1e-4)


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


def test_the_refusals_state_the_stop_rules_the_monitor_applies(deck, deck_cfg, tmp_path):
    """daily.monitor: every stop -- overspend, scrap, margin -- needs
    persistence_days consecutive priced days over its threshold
    (evaluate_guardrail); the deck must say what the monitor does."""
    out = tmp_path / "deck.html"
    sd.write_page(deck, deck_cfg, out)
    html = out.read_text()
    assert "Every stop needs ${D.config.persistence_days} consecutive days" in html
    assert "a single day is enough" not in html
