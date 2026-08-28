"""The metrics index must not describe a system that no longer exists.

`tools/metrics_glossary.py` is mostly prose held true by hand -- there is no
way to derive "what does this number mean" from the code. But three of the
things it names DO have a machine-readable source of truth, and all three are
exactly the kind that drift silently:

  the two event schemas   fields get added and the docs do not follow
  the frozen artifacts    a file is renamed and every reference goes stale
  the status gate names   a check is renamed and the index points nowhere

Those are checked here. The figures the index quotes from config are checked
too, since a re-run moves them and a stale number in a reference document is
worse than no number at all.

What is deliberately NOT checked is coverage: the index does not list all 45
event fields, because `docs/event_contract.html` does, exhaustively and under
its own guard. Two exhaustive lists of the same schema is how they come to
disagree. The index names the handful worth explaining and points at the
contract for the rest.
"""

import re

import pytest

from common import provenance
from common.config import config_get, load_config
from events.store import DECISION_REQUIRED, OUTCOME_REQUIRED
from pipeline import status
from tools import metrics_glossary as gl

# fields that exist on an event but are conditional, so they are absent from
# the REQUIRED lists while still being real
CONDITIONAL = {"adjustment_reason", "execution_failure_reason"}


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _entries(section_title):
    for title, _source, _intro, entries in gl.CATALOGUE:
        if title == section_title:
            return entries
    raise AssertionError(f"no section titled {section_title!r}")


def _names(section_title):
    return [name for name, *_ in _entries(section_title)]


# --------------------------------------------------------------- event schema
def test_every_event_field_named_is_a_real_one():
    """The index may name a subset -- it may not invent one."""
    known = set(DECISION_REQUIRED) | set(OUTCOME_REQUIRED) | CONDITIONAL
    for section in ("Decision event", "Outcome event"):
        for name in _names(section):
            assert name in known, (
                f"{section} names {name!r}, which no event carries. Either the "
                "schema was renamed and the index was not, or it is a typo.")


def test_the_index_does_not_duplicate_the_event_contract():
    """If someone ever completes these two sections into full field lists, the
    contract and the index become two schemas that can disagree. The index is
    supposed to be a pointer here, not a second source."""
    listed = len(_names("Decision event")) + len(_names("Outcome event"))
    assert listed < len(DECISION_REQUIRED) + len(OUTCOME_REQUIRED), \
        ("the index now lists every event field -- fold it back to a subset and "
         "let docs/event_contract.html be the schema of record")


# ------------------------------------------------------------------ artifacts
def test_every_artifact_referenced_exists_in_the_bundle():
    """Source lines name the files a component writes. A renamed artifact must
    not leave the index pointing at a path nothing produces."""
    # posterior.json is deliberately absent from ARTIFACTS: it is production
    # learning state, the one file that MOVES, not a frozen artifact sealed
    # into the bundle. It is still a real path the index may name.
    known = {name for name, _key in provenance.ARTIFACTS} | {"posterior"}
    seen = set()
    for _title, source, _intro, _entries in gl.CATALOGUE:
        # [a-z0-9_] not [a-z]: a name carrying a digit -- manifest_v2 -- would
        # otherwise fail to match at all and be silently waved through, which
        # is the precise case this guard exists to catch
        for path in re.findall(r"artifacts/([a-z0-9_]+)\.(?:json|txt)", source):
            seen.add(path)
    assert seen, "no artifact paths found -- the source lines stopped naming files"
    unknown = sorted(seen - known)
    assert not unknown, (
        f"the index names {unknown}, which is not in provenance.ARTIFACTS")


# ---------------------------------------------------------------- status gates
def test_the_status_section_matches_the_real_check_names(cfg):
    """These are quoted verbatim so a reader can match a red line to an entry.
    A renamed check must break here rather than silently mismatch."""
    live = [row["check"] for row in status.collect(cfg)["checks"]]
    assert _names("Status board") == live, (
        "the Status board section has drifted from pipeline.status")


# ------------------------------------------------------- figures quoted inline
@pytest.mark.parametrize("figure, path", [
    ("0.2436", ("dispersion", "rho")),
    ("5.909", ("dispersion", "mean_forced_hours_per_episode")),
    ("12.0", ("learning", "information_increment")),
    ("0.15", ("learning", "max_mean_step")),
])
def test_quoted_config_figures_still_match_config(cfg, figure, path):
    """A re-run moves these. A stale number in a reference document is worse
    than no number, because it will be quoted in a meeting."""
    live = config_get(cfg, path)
    assert f"{live}".startswith(figure) or figure in f"{live}", \
        (f"the index quotes {figure} for {'.'.join(path)}, config says {live}")
    blob = " ".join(m for _t, _s, _i, es in gl.CATALOGUE
                    for _n, _u, m, _r in es)
    assert figure in blob, f"{figure} is no longer quoted anywhere in the index"


def test_deff_quoted_matches_the_computed_design_effect(cfg):
    from common.config import deff
    blob = " ".join(m for _t, _s, _i, es in gl.CATALOGUE for _n, _u, m, _r in es)
    assert f"{deff(cfg):.3f}" in blob, \
        f"the index should quote deff {deff(cfg):.3f}"


# ------------------------------------------------------------------- rendering
def test_every_entry_is_well_formed_and_the_page_builds(tmp_path):
    for title, source, intro, entries in gl.CATALOGUE:
        assert title and source and intro, title
        assert entries, f"{title} has no entries"
        for name, unit, meaning, read in entries:
            assert unit in gl.UNITS, f"{name}: unknown unit {unit!r}"
            assert len(meaning) > 20, f"{name}: meaning too thin to be useful"
            assert meaning.strip().rstrip("*`").endswith((".", "]")), \
                f"{name}: unpunctuated"
            # `read` is free text, but GATE must be spelled exactly, since the
            # renderer and the gates-only filter both key on it
            assert not read.upper().startswith("GATE") or read.startswith("GATE")

    # redirect the output, and PUT IT BACK -- leaving it pointed at tmp_path
    # would make any later test in the session render into a deleted directory
    real_out = gl.OUT
    try:
        gl.OUT = tmp_path / "metrics.html"
        total = gl.render()
        page = gl.OUT.read_text()
    finally:
        gl.OUT = real_out
    assert total == sum(len(e) for _t, _s, _i, e in gl.CATALOGUE)
    assert page.count('class="row"') == total
    # the filter is what makes this a tool rather than a document
    assert 'id="q"' in page and 'id="gateonly"' in page
    assert page.count("<section") == page.count("</section>")
