"""ops.bootstrap_loop: the settle driver, with the subprocess steps recorded
and the convergence verdicts scripted."""

import sys

from conftest import ROOT


def _scripted_settle(monkeypatch, dlogs, converge_at=None, max_turns=20):
    """Run ops.bootstrap_loop.settle with the subprocess steps recorded and the
    convergence verdicts scripted: turn k reads dlogs[k-1]; `converge_at`
    names the turn whose verdict says settled."""
    from ops import bootstrap_loop as br
    calls, state = [], {"turn": 0}

    def step(label, args, fatal=True, quiet=False):
        calls.append(list(args))
        if "--check-convergence" in args and "--commit-convergence" in args:
            state["turn"] += 1
        return 0

    def convergence(cfg):
        t = state["turn"]
        return {"converged": t == converge_at,
                "max_abs_dlog": dlogs[min(t, len(dlogs)) - 1]}

    monkeypatch.setattr(br, "step", step)
    monkeypatch.setattr(br, "convergence", convergence)
    ok, turns, _ = br.settle({}, max_turns)
    return ok, turns, calls


def test_the_loop_runs_3b_once_and_finishes_with_a_full_prior(monkeypatch):
    """Turn k's --check-convergence solves the factors turn k+1 would; the
    loop commits that solve instead of recomputing it, runs the prior --fast
    inside the loop, and re-runs it FULL (then re-checks) once settled --
    a production-shaped 9-turn trajectory with a plateau must survive."""
    dlogs = [2.29, .9, .9, .4, .35, .2, .09, .02, .005]
    ok, turns, calls = _scripted_settle(monkeypatch, dlogs, converge_at=9)
    assert ok and turns == 9
    fits = [c for c in calls if "--fit-calibration" in c]
    assert len(fits) == 1, "3b belongs to the first turn only"
    priors = [c for c in calls if c[0] == "fit.estimate_prior"]
    assert len(priors) == 10 and all("--fast" in c for c in priors[:9]) \
        and "--fast" not in priors[-1], "in-loop priors fast, the artifact's full"
    checks = [c for c in calls if "--check-convergence" in c]
    assert all("--commit-convergence" in c for c in checks[:9])
    assert "--commit-convergence" not in checks[-1], \
        "the confirm after the full prior is a DRY re-check"
    assert calls.index(priors[-1]) < calls.index(checks[-1])


def test_the_loop_stalls_on_three_turns_without_a_new_best(monkeypatch):
    """The STALL test reads THIS run's turns: stuck and oscillating both stop
    at turn 5; the cap is a runaway guard, not the budget."""
    for dlogs in ([.31, .30, .305, .30, .31, .30], [.5, .2, .5, .2, .5, .2]):
        ok, turns, _ = _scripted_settle(monkeypatch, dlogs)
        assert (ok, turns) == (False, 5)
    # a plain 20-turn cap never fires on a settling chain first
    ok, turns, _ = _scripted_settle(monkeypatch, [1 / (t + 1) for t in range(20)],
                                    converge_at=12)
    assert ok and turns == 12


def test_the_loop_cap_is_sized_for_production():
    """The owner measures 8-9 turns; the fixture 3-4. A cap under 20 would
    cut a real settle short (rule 19: size for the extract that matters)."""
    import subprocess
    r = subprocess.run([sys.executable, "-m", "ops.bootstrap_loop", "--help"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    import re
    m = re.search(r"--max-turns.*?default (\d+)", r.stdout, re.S)
    assert m and int(m.group(1)) >= 20
