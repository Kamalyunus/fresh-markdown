"""tools/scenario_deck: the leadership deck is built from real solves and stays self-consistent."""
import json
import re

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
    out = tmp_path / "deck.html"
    sd.write_page(deck, cfg, out)
    html = out.read_text()
    assert "__DATA__" not in html and "__CFGV__" not in html
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S)
    data = json.loads(m.group(1).replace("<\\/", "</"))
    assert len(data["scenarios"]) == 12
    assert data["config"]["config_version"] == cfg["meta"]["config_version"]
    assert "pricing.dp.solve" in html
