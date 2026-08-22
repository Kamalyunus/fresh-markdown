"""The EDA's own claims, checked.

The panels are prose plus pandas; most of what they say a human has to keep
true. Three things are machine-checkable and all three have gone wrong
elsewhere in this repo, so they are checked here:

  * a panel claims to inform a config key -- the key must exist;
  * a panel decides nothing -- no gate, no verdict, no MEASURED value, because
    two sources of truth for one of those is the failure `artifact_mirror_drift`
    exists to catch;
  * anything about stock is counted ONCE PER EPISODE, never summed over hours.
"""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from common.config import config_get, load_config
from tools import eda
from tools.eda_page import KINDS, render


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def frame():
    """A small but complete population: two categories, two FCs, windows that
    cross midnight, one sellout and one that ends holding stock."""
    rng = np.random.default_rng(0)
    rows = []
    for e in range(60):
        sku, fc = 1000 + e % 17, ["FC1", "FC2"][e % 2]
        cat = ["MEAT", "VEGETABLE"][e % 2]
        day = pd.Timestamp("2026-03-01") + pd.Timedelta(days=e % 30)
        start_h, hours = int(rng.integers(6, 22)), int(rng.integers(3, 10))
        q, price, cost = int(rng.integers(4, 40)), 10_000.0, 4_000.0
        for t in range(hours):
            ts = day + pd.Timedelta(hours=start_h + t)
            sold = int(min(q, rng.integers(0, 3)))
            rows.append({
                "episode_id": f"{sku}|{fc}|{e}", "sku_id": sku, "fc": fc,
                "category": cat, "subcategory": f"{cat}_A",
                "date": ts.strftime("%Y-%m-%d"), "hour_of_day": ts.hour,
                "hours_remaining": float(hours - t),
                "starting_inventory": q, "units_sold": sold,
                "ending_inventory": q - sold if t < hours - 1 else 0,
                "total_discount": round(0.25 + 0.025 * t, 4),
                "original_price": price, "cost": cost,
                "d_ref": 0.25, "d_max": 1 - cost / price,
                "offered_price": price * (1 - (0.25 + 0.025 * t)),
                "applied_price": price * (1 - (0.25 + 0.025 * t)) if sold else 0.0,
            })
            q -= sold
            if q <= 0:
                break
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def report(frame, cfg):
    return eda.build(frame, cfg)


# ------------------------------------------------------------- the guard

def _informs():
    return [(p["key"], key) for p in eda.PANELS for key in p["informs"]]


@pytest.mark.parametrize("panel,key", _informs(),
                         ids=[f"{p}:{k}" for p, k in _informs()])
def test_every_informs_key_exists_in_config(panel, key, cfg):
    """A panel that names a config key is making a claim about it. A rename
    must break the claim instead of leaving it quietly stale."""
    assert config_get(cfg, key.split(".")) is not None or True, key
    node = cfg
    for part in key.split("."):
        assert isinstance(node, dict) and part in node, \
            f"{panel} claims to inform {key}, which does not exist"
        node = node[part]


def test_every_panel_declares_what_it_informs():
    bare = [p["key"] for p in eda.PANELS if not p["informs"]]
    assert not bare, f"panels with no stated purpose: {bare}"


# ------------------------------------------------------- it decides nothing

def test_the_eda_produces_no_gate_and_no_measured_value(report):
    """`bootstrap.measure` owns the MEASURED values and the reassessment
    gates. A second source for any of them is the drift this repo already
    has a whole checker for."""
    blob = json.dumps(report).lower()
    for word in ('"verdict"', '"pass"', '"fail"', '"tau_initial"', '"rho"'):
        assert word not in blob, f"the EDA is deciding something: {word}"


# --------------------------------------------------------------- structure

def test_no_panel_emits_a_value_json_cannot_represent(report):
    """json.dump writes bare NaN and Infinity, which no strict parser accepts.
    A degenerate correlation (constant entry hour) produced one."""
    text = json.dumps(report)
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)          # strict by default: proves it round-trips


def test_every_panel_returns_a_titled_body(report):
    assert len(report["panels"]) == len(eda.PANELS)
    for key, body in report["panels"].items():
        assert body["title"] and isinstance(body["lede"], str)
        assert len(body) > 4, f"{key} carries no numbers"


def test_every_chart_declares_a_kind_the_page_can_draw(report):
    for key, body in report["panels"].items():
        c = body.get("chart")
        if c is None:
            continue
        assert c["kind"] in KINDS, f"{key}: no renderer for {c['kind']}"
        if c["kind"] == "line":
            assert all(len(v) == len(c["x"]) for v in c["series"].values())
        if c["kind"] == "bars":
            assert len(c["labels"]) == len(c["values"])
        if c["kind"] == "hist":
            assert len(c["edges"]) == len(c["counts"]) + 1
        if c["kind"] == "pareto":
            assert len(c["x"]) == len(c["y"])
            assert c["y"] == sorted(c["y"])          # cumulative, so monotone


def test_the_page_renders_every_panel(report):
    page = render(report)
    for key, body in report["panels"].items():
        assert f'id="{key}"' in page
        assert body["title"] in page
    assert page.count("<svg") >= 8
    assert "This page decides nothing" in page


# ------------------------------------------------------ episode-level counting

def test_exposure_is_counted_once_per_episode_not_summed_over_hours(frame, cfg):
    """Inventory persists hour to hour. A per-row sum multiplies the same
    stock by the window length -- the mistake the waterfall's cogs_at_risk
    was written to avoid, repeated here would be worse because nothing else
    cross-checks it."""
    got = eda.p_pareto(frame, cfg)["cogs_at_risk_total"]
    op = frame[~frame.episode_id.duplicated()]
    assert got == pytest.approx(float((op.cost * op.starting_inventory).sum()), abs=1)
    assert got < float((frame.cost * frame.starting_inventory).sum())


def test_pareto_shares_are_cumulative_and_ordered(frame, cfg):
    p = eda.p_pareto(frame, cfg)
    shares = [p["cogs_share_of_top_1pct"], p["cogs_share_of_top_5pct"],
              p["cogs_share_of_top_10pct"], p["cogs_share_of_top_25pct"]]
    assert shares == sorted(shares) and shares[-1] <= 1.0
    assert p["skus_covering_half_the_cogs"] <= p["skus_covering_80pct_of_cogs"]


def test_entry_arm_feasibility_is_monotone_in_depth(frame, cfg):
    """A shallower arm is feasible wherever a deeper one is: the cost floor
    only ever removes from the deep end."""
    shares = list(eda.p_entry_arms(frame, cfg)[
        "share_of_episodes_where_arm_is_feasible"].values())
    assert shares == sorted(shares, reverse=True)


def test_the_two_anchor_bands_are_not_conflated(frame, cfg):
    a = eda.p_anchors(frame, cfg)
    assert a["calibration_band_pp"] < a["velocity_band_pp"]
    assert a["rows_within_calibration_band"] <= a["rows_within_velocity_band"]


def test_it_runs_end_to_end_from_the_command_line(frame, tmp_path):
    frame.to_parquet(tmp_path / "prepared.parquet", index=False)
    r = subprocess.run(
        [sys.executable, "-m", "tools.eda", "--input",
         str(tmp_path / "prepared.parquet"), "--out", str(tmp_path / "eda.json"),
         "--html", str(tmp_path / "eda.html")],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads((tmp_path / "eda.json").read_text())["panels"]
    assert (tmp_path / "eda.html").read_text().startswith("<!doctype html>")
