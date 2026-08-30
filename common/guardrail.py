"""How a guardrail metric is compared against its baseline. ONE definition:
derive_thresholds measures the noise floor and pipeline.monitor evaluates the
trigger, and both MUST compute the same quantity. Two bases: `relative`
(t/c - 1) for strictly positive rates (scrap); `absolute_pp` (t - c) when a
metric can cross zero -- margin_rate does, so its relative floor exceeded the
series' own level and was structurally blocked (measured; docs/learnings.md).
"""

RELATIVE = "relative"
ABSOLUTE_PP = "absolute_pp"
BASES = (RELATIVE, ABSOLUTE_PP)


def smooth(series, days):
    """Average `days` days before comparing. Must be applied on BOTH sides
    (floor and trigger) or the guardrail is inert."""
    s = series.dropna()
    return (s.rolling(days, min_periods=days).mean().dropna() if days > 1
            else s)


def deviation(treatment, control, worse_when_higher, basis):
    """Deterioration of `treatment` against `control`, positive = WORSE.
    Inputs are already smoothed and index-aligned; the sign convention is the
    caller's (scrap worse when higher, margin when lower)."""
    if basis == ABSOLUTE_PP:
        return (treatment - control) if worse_when_higher else (control - treatment)
    if basis == RELATIVE:
        ratio = treatment / control
        return (ratio - 1) if worse_when_higher else (1 - ratio)
    raise ValueError(f"unknown deterioration basis {basis!r}, expected one of {BASES}")


BASIS = {"scrap": RELATIVE,        # strictly positive rate
         "margin": ABSOLUTE_PP}    # can cross zero (relative floor blocked)


def basis_for(cfg, metric_key):
    return BASIS.get(metric_key, RELATIVE)


def units_of(basis):
    """Human-readable units, for the report to say what a number is."""
    return ("percentage points of the rate (t - c)" if basis == ABSOLUTE_PP
            else "relative deviation (t/c - 1); 0.15 means 15%")


def floor_is_unusable(floor, basis):
    """A floor a threshold cannot be set above -- the guardrail is BLOCKED.
    On the RELATIVE basis a floor >= 1.0 means ordinary daily swing exceeds
    the series' own level: the wrong basis for the metric, not a tuning
    problem. Absolute-pp floors have no such bound."""
    return bool(floor is not None and basis == RELATIVE and floor >= 1.0)
