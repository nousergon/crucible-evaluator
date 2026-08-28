"""Statistical power beside every risk-adjusted metric.

alpha-engine-config-I8188 deliverable 8. The report card publishes
``sharpe_ratio = 0.673`` with a bootstrap CI of **[−2.36, +3.47]** and grades it
**RED** against a target of 1.0 — a target that sits comfortably inside its own
confidence interval. The same holds for the information ratio ([−5.09, +0.76])
and Sortino ([−2.99, +6.78]).

WHERE THE RED COMES FROM. ``krepis.metrics.derive_status`` has two RED clauses:

    if _at_or_worse(value, red_line):        return "RED"   # the point estimate
    bad_bound = ci_low if higher_is_better else ci_high
    if _at_or_worse(bad_bound, red_line):    return "RED"   # the CI's bad side

The second clause is the one firing here. With a CI this wide the bad-side bound
is below any red line that is not itself absurd, so **the RED is produced by the
width of the interval rather than by evidence about the system**. A status that
cannot come out any other way is not a status — it is a constant wearing a
measurement's clothes, and it is indistinguishable on the tile from a genuine
breakdown, which is the failure mode that makes an operator stop reading.

WHAT THIS MODULE DOES. It never upgrades anything and never touches a RED that
the POINT ESTIMATE earned. It acts on exactly one case:

    the status is RED, it was produced by the CI's bad-side bound, AND the
    TARGET also lies inside the CI

— i.e. the data cannot distinguish "system-breaking" from "on target". That is
downgraded to WATCH, with a status_reason that says so and names the N at which
the question becomes answerable.

REQUIRED-N. Distribution-free, so it applies to a bootstrap, Newey-West or
Wilson interval alike: a CI half-width shrinks as 1/√N, so reaching a
half-width ``h`` from an observed half-width ``h₀`` at ``N₀`` needs

    N_required = N₀ · (h₀ / h)²

The decision-relevant ``h`` is ``|target − red_line| / 2`` — the precision at
which the interval can no longer span both bars at once, which is precisely the
precision at which the status stops being predetermined. For the live Sharpe
(N₀=119, h₀=2.915, target 1.0, red_line 0.0) that is ≈4,000 sessions, ~16
years — the same order as the ~3,875 sessions computed by hand in I8188.

Publishing that number is the point. A metric that needs 16 years of data to
grade is not a failing metric; it is an unmeasured one, and those are different
facts that must not render identically.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

# Published beside the number, never applied silently.
POWER_SUPPRESSION_REASON_PREFIX = "Power-limited"


def observed_half_width(ci_low: float | None, ci_high: float | None) -> float | None:
    """Half-width of a two-sided interval, or None when it is not a real one."""
    if ci_low is None or ci_high is None:
        return None
    try:
        lo, hi = float(ci_low), float(ci_high)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi < lo:
        return None
    return (hi - lo) / 2.0


def required_n(
    *,
    ci_low: float | None,
    ci_high: float | None,
    n_samples: int | None,
    target: float | None,
    red_line: float | None,
) -> int | None:
    """Observations needed before this metric's status can be anything but forced.

    ``N_required = N₀ · (h₀ / h)²`` with ``h = |target − red_line| / 2``.
    Returns None when the inputs cannot support the calculation (no interval, no
    two distinct bars, N₀ absent or zero) — an ABSENT required-N, never a 0,
    because 0 reads as "no more data needed".
    """
    h0 = observed_half_width(ci_low, ci_high)
    if h0 is None or not n_samples or n_samples <= 0:
        return None
    if target is None or red_line is None:
        return None
    try:
        gap = abs(float(target) - float(red_line))
    except (TypeError, ValueError):
        return None
    if gap <= 0:
        return None
    h = gap / 2.0
    if h0 <= h:
        # Already precise enough to separate the bars — the status is earned.
        return int(n_samples)
    return int(math.ceil(n_samples * (h0 / h) ** 2))


def target_inside_ci(
    *, target: float | None, ci_low: float | None, ci_high: float | None
) -> bool:
    """True when the interval spans the target — the metric cannot rule it out."""
    if target is None or ci_low is None or ci_high is None:
        return False
    try:
        return float(ci_low) <= float(target) <= float(ci_high)
    except (TypeError, ValueError):
        return False


def _red_is_ci_driven(
    *, value: float | None, red_line: float | None, ci_low: float | None,
    ci_high: float | None, target: float | None,
) -> bool:
    """True when the RED came from the CI's bad-side bound, not the estimate.

    Mirrors ``krepis.metrics.derive_status``'s direction inference exactly: a
    reimplementation that disagreed with it would suppress the wrong REDs.
    """
    if value is None or red_line is None:
        return False
    higher_is_better = target is None or red_line is None or target >= red_line
    try:
        v, rl = float(value), float(red_line)
    except (TypeError, ValueError):
        return False
    point_estimate_is_red = (v <= rl) if higher_is_better else (v >= rl)
    if point_estimate_is_red:
        return False  # earned by the estimate — never suppressed
    bad_bound = ci_low if higher_is_better else ci_high
    if bad_bound is None:
        return False
    try:
        bb = float(bad_bound)
    except (TypeError, ValueError):
        return False
    return (bb <= rl) if higher_is_better else (bb >= rl)


def annotate_power(record: Any) -> Any:
    """Attach ``n_required`` / ``target_inside_ci`` and suppress a forced RED.

    Mutates and returns ``record`` (a ``krepis.MetricRecord``; ``extra="allow"``
    carries the new fields through to the artifact). Every record gets the two
    power fields — including ones whose status is untouched — so an operator can
    see the required-N beside a GREEN or WATCH too, and so a metric emitting
    nothing about its own power is distinguishable from one that is adequately
    powered.

    The suppression is deliberately narrow and never silent:

    * only RED → WATCH, never any other transition and never an upgrade;
    * only when the RED came from the CI's bad-side bound AND the target is
      also inside the CI;
    * the original status is preserved on ``status_before_power`` and the
      reason is rewritten to say what happened and what would settle it.
    """
    ci_low = getattr(record, "ci_low", None)
    ci_high = getattr(record, "ci_high", None)
    target = getattr(record, "target", None)
    red_line = getattr(record, "red_line", None)
    value = getattr(record, "value", None)
    n_samples = getattr(record, "n_samples", None)

    n_req = required_n(
        ci_low=ci_low, ci_high=ci_high, n_samples=n_samples,
        target=target, red_line=red_line,
    )
    inside = target_inside_ci(target=target, ci_low=ci_low, ci_high=ci_high)
    record.n_required = n_req
    record.target_inside_ci = inside

    if getattr(record, "status", None) != "RED":
        return record
    if not inside:
        return record
    if not _red_is_ci_driven(
        value=value, red_line=red_line, ci_low=ci_low, ci_high=ci_high,
        target=target,
    ):
        return record

    record.status_before_power = "RED"
    record.status = "WATCH"
    shortfall = ""
    if n_req is not None and n_samples:
        extra = max(0, n_req - int(n_samples))
        years = n_req / 252.0
        shortfall = (
            f" Separating target {target:g} from red-line {red_line:g} at this "
            f"interval width needs N≈{n_req:,} (~{years:.1f}y at 252 sessions/yr), "
            f"{extra:,} more than the {int(n_samples):,} in hand."
        )
    record.status_reason = (
        f"{POWER_SUPPRESSION_REASON_PREFIX}: {record.name} = "
        f"{value:g} with CI [{float(ci_low):g}, {float(ci_high):g}] — the "
        f"interval contains BOTH the target ({target:g}) and the red-line "
        f"({red_line:g}), so the RED was produced by the CI's width, not by "
        f"evidence about the system. Downgraded RED→WATCH.{shortfall}"
    )
    logger.info(
        "power: suppressed a CI-driven RED on %s (value=%s, CI=[%s, %s], "
        "target=%s, red_line=%s, n=%s, n_required=%s)",
        record.name, value, ci_low, ci_high, target, red_line, n_samples, n_req,
    )
    return record


def annotate_power_all(records: list[Any]) -> list[Any]:
    """Apply :func:`annotate_power` across a tile's components, in place."""
    for rec in records:
        annotate_power(rec)
    return records
