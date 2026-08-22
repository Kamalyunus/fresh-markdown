"""The walkthrough's numbers are typed into prose. Keep them honest.

Two failures, and they are different. Editing a panel without updating the
ledger (or the reverse) leaves the page and its stated provenance
disagreeing, and that is always checkable. A re-run leaving every figure on
the replay tab stale is the other, and it is NOT a test failure -- a report
from a different model version is a different measurement of a different
thing, not proof the page is wrong. Comparing across runs is what hard rule 1
exists to prevent. That case surfaces in `pipeline.status`, which is what
anyone actually reads after a run.

This is the failure the deck already has: slides 2 and 42 still show 36.68%
where the report gives 38.68%, because nothing connected the two.
"""

import json
import pathlib

import pytest

from tools.walkthrough import figures
from tools.walkthrough.panels import PANEL_DIR, KEYS

ALL = figures.REPLAY_FIGURES + figures.SHADOW_FIGURES


@pytest.mark.parametrize("tab,literal,path,fmt", ALL,
                         ids=[f"{t}:{p}" for t, _, p, _ in ALL])
def test_every_registered_figure_is_on_its_page(tab, literal, path, fmt):
    if literal is figures.PENDING:
        pytest.skip("no measurement behind this slot yet")
    html = (PANEL_DIR / f"{tab}.html").read_text()
    assert literal in html, (
        f"{tab}.html no longer prints {literal} for {path} -- update "
        "tools/walkthrough/figures.py in the same commit as the panel")


def test_every_source_names_a_real_panel():
    for tab, src in figures.SOURCES.items():
        assert tab in KEYS
        assert (PANEL_DIR / f"{tab}.html").exists()
        assert all(f[0] == tab for f in src["figures"])


def test_pending_slots_are_visibly_empty_not_plausible_numbers():
    """A placeholder that looks like a measurement is worse than none."""
    html = (PANEL_DIR / "shadow.html").read_text()
    assert 'class="pend"' in html
    # the shadow tab must not carry a currency or percentage figure it cannot
    # source -- those read as measured whatever the caption says
    import re
    for m in re.finditer(r"₩[\d.]+M|\d+\.\d\d%", html):
        pytest.fail(f"shadow.html prints an unsourced figure: {m.group(0)}")


def test_the_shadow_tab_refuses_to_imply_a_loss_number():
    """The whole reason the replay tab stays: shadow has no IL to report."""
    html = (PANEL_DIR / "shadow.html").read_text()
    assert "no IL number here at all" in html
    assert "no price was applied" in html.lower()


def test_replay_points_at_the_rung_above_it():
    html = (PANEL_DIR / "replay.html").read_text()
    assert "Shadow" in html and "three rungs" in html.lower()


def test_figures_check_reports_rather_than_crashes(tmp_path):
    """Every branch of the drift check, on synthetic reports."""
    verdict, detail, problems = figures.check("replay", root=str(tmp_path))
    assert verdict == "no report" and not problems

    src = figures.SOURCES["replay"]
    out = tmp_path / "reports"
    out.mkdir()

    (out / "backtest.json").write_text(json.dumps(
        {"artifact_versions": {"baseline_model_version": "some-other-run"}}))
    verdict, detail, _ = figures.check("replay", root=str(tmp_path))
    assert verdict == "stale" and src["model_version"] in detail

    # same version, figures rebuilt from the ledger itself -> ok
    report = {"artifact_versions": {"baseline_model_version": src["model_version"]}}
    for _, literal, path, fmt in src["figures"]:
        node = report
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _invert(literal, fmt)
    (out / "backtest.json").write_text(json.dumps(report))
    verdict, detail, problems = figures.check("replay", root=str(tmp_path))
    assert verdict == "ok", problems

    # move one number -> drift, named
    report["policy_deltas"]["dp_il"] = 99e6
    (out / "backtest.json").write_text(json.dumps(report))
    verdict, _, problems = figures.check("replay", root=str(tmp_path))
    assert verdict == "drift"
    assert any("dp_il" in p for p in problems)


def _invert(literal, fmt):
    """The raw value a formatter would render as `literal`."""
    body = literal.replace("₩", "").replace(",", "")
    if body.endswith("M"):
        return float(body[:-1]) * 1e6
    if body.endswith("%"):
        return float(body[:-1]) / 100.0
    return float(body)


def test_the_shadow_source_is_unversioned_until_the_holdout_run():
    # bumping this to a real version is the commit that fills the tab in
    assert figures.SOURCES["shadow"]["model_version"] is None
    assert figures.check("shadow")[0] == "pending"


def test_the_built_page_carries_the_shadow_tab():
    page = pathlib.Path("docs/system_walkthrough.html")
    if not page.exists():
        pytest.skip("page not built")
    html = page.read_text()
    assert 'id="tab-shadow"' in html or ">Shadow<" in html
