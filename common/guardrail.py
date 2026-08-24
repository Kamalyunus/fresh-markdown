"""How a guardrail metric is compared against its baseline. ONE definition.

`bootstrap.derive_thresholds` measures the noise floor and `pipeline.monitor`
evaluates the trigger, and the two MUST compute the same quantity or the
threshold is graded against a yardstick nothing uses. That has to be a shared
function rather than two copies that look alike.

TWO BASES, because one does not fit both metrics.

  relative      `t / c - 1`. The natural scale for a strictly positive rate
                whose level varies: a 20% deterioration means the same thing
                whatever the baseline happens to be that week. This is what
                `scrap_rate` uses and it works -- measured on production, a
                7-day-smoothed 3-sigma floor of 0.4156 against a mean level of
                0.1814, not outlier-dominated, worst observed 0.3789.

  absolute_pp   `t - c`, in percentage points of the rate. Required when the
                metric can approach or CROSS ZERO, because a ratio to a
                near-zero denominator has no scale at all.

WHY `margin_rate` IS absolute_pp, measured rather than assumed. On the
production extract its daily series runs mean 0.0308, min -0.0464, and 36 of
134 days are AT OR BELOW ZERO. Under the relative basis that produced:

    raw 3-sigma            65.4497      (6,545%)
    robust 3-sigma          3.5853        (359%)
    worst observed        155.2302     (15,523%)

No threshold can sit above a floor larger than the series' own level, so the
margin stop condition could not be set at all -- it was structurally blocked,
not merely badly tuned. Smoothing does not fix it: averaging a series that
changes sign just averages across the crossing. On the absolute basis the same
series has a daily standard deviation of 0.0374, so a 3-sigma floor lands near
11 percentage points -- a number an owner can actually reason about, and close
to the 0.1363/0.15 recorded before the population changed.

The basis is per metric in `monitoring.stop_conditions.deterioration_basis`,
so the choice is config the owner can see rather than a branch buried in two
modules.
"""

RELATIVE = "relative"
ABSOLUTE_PP = "absolute_pp"
BASES = (RELATIVE, ABSOLUTE_PP)


def smooth(series, days):
    """Average `days` days before comparing.

    A low-base series swings by more than its own level day to day, so the
    floor is meaningless unless the trigger is smoothed the same way. Applied
    on BOTH sides or the guardrail is inert.
    """
    s = series.dropna()
    return (s.rolling(days, min_periods=days).mean().dropna() if days > 1
            else s)


def deviation(treatment, control, worse_when_higher, basis):
    """Deterioration of `treatment` against `control`. Positive = WORSE.

    `treatment` and `control` are already smoothed and index-aligned. The sign
    convention is the caller's: scrap is worse when higher, margin when lower.
    """
    if basis == ABSOLUTE_PP:
        return (treatment - control) if worse_when_higher else (control - treatment)
    if basis == RELATIVE:
        ratio = treatment / control
        return (ratio - 1) if worse_when_higher else (1 - ratio)
    raise ValueError(f"unknown deterioration basis {basis!r}, expected one of {BASES}")


def basis_for(cfg, metric_key):
    """The configured basis for `scrap` or `margin`.

    Defaults to RELATIVE when the key is absent so an older config still runs,
    but the key is written out in `config.yaml` and should stay there: which
    basis a guardrail uses changes what its threshold MEANS, and that is not
    something to leave implicit.
    """
    sc = cfg["monitoring"]["stop_conditions"]
    return sc.get("deterioration_basis", {}).get(metric_key, RELATIVE)


def units_of(basis):
    """Human-readable units, for the report to say what a number is."""
    return ("percentage points of the rate (t - c)" if basis == ABSOLUTE_PP
            else "relative deviation (t/c - 1); 0.15 means 15%")


def floor_is_unusable(floor, basis):
    """A floor a threshold cannot be set above, so the guardrail is BLOCKED.

    On the relative basis a floor at or above 1.0 means the series' ordinary
    daily swing exceeds its own level: any threshold clearing it would also
    clear the failure the condition exists to catch. That is not a tuning
    problem, it is the wrong basis for the metric -- and nothing said so out
    loud until a run produced 3.5853 and it read as "margin is volatile".

    Absolute-pp floors have no such bound: they are in the metric's own units
    and are only ever as large as the metric moves.
    """
    return bool(floor is not None and basis == RELATIVE and floor >= 1.0)
