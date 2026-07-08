"""
module_agg.py — hierarchical aggregation for the System Report Card v2.

Pure functions over ``list[MetricRecord]`` implementing RC v2 Principles 3
(aggregation respects critical gates) + 4 (BH-FDR at the module layer):

  component statuses → module_status   (critical-gate rule, not weighted avg)
  module statuses    → overall_status  (worst-of, portfolio outcome leads)
  components         → numeric_grade   (legacy 0-100 compat)

The grade is NEVER a plain weighted average of letters: a single RED critical
component fails the module regardless of how green everything else is, and a
module cannot claim GREEN while a critical component is unimplemented. This is
the institutional rule the v1 surface lacked (which is how it floated a C+
overall while critical executor tiles were N/A).

Authoritative: ``system-report-card-revamp-260522.md`` §"Aggregation methodology".
"""

from __future__ import annotations

from krepis.metrics import MetricRecord, StatusLiteral
from nousergon_lib.quant.stats.multiple_testing import benjamini_hochberg

# Modules whose RED cascades to an overall RED (RC v2 module→overall rule).
_CASCADE_MODULES = ("research", "predictor", "executor", "substrate")


def bh_fdr_significant(p_values: list[float], alpha: float = 0.05) -> bool:
    """True if BH-FDR finds any significant test among ``p_values`` at ``alpha``.

    Empty / all-None input → False (no evidence of joint underperformance).
    """
    ps = [p for p in p_values if p is not None]
    if not ps:
        return False
    return any(benjamini_hochberg(ps, alpha=alpha))


def module_status(components: list[MetricRecord], *, alpha: float = 0.05) -> StatusLiteral:
    """Roll a tile's components up to a module status (RC v2 Principle 3).

    Order of precedence:
      RED   if any critical component is RED.
      RED   if ≥2 critical components are WATCH AND BH-FDR finds their joint
            underperformance significant.
      WATCH if any critical component is N/A-NOT-IMPL (can't claim GREEN with an
            unimplemented critical).
      WATCH if ≥2 critical WATCH (not BH-significant), or any critical WATCH, or
            any supporting RED.
      WATCH if any critical component is N/A-* (transparency); GREEN if only
            supporting/diagnostic are N/A.
      GREEN otherwise.
    """
    if not components:
        return "N/A-NOT-RUN"

    critical = [c for c in components if c.criticality == "critical"]
    supporting = [c for c in components if c.criticality == "supporting"]

    crit_red = [c for c in critical if c.status == "RED"]
    crit_watch = [c for c in critical if c.status == "WATCH"]
    crit_not_impl = [c for c in critical if c.status == "N/A-NOT-IMPL"]
    crit_na = [c for c in critical if c.is_na]
    sup_red = [c for c in supporting if c.status == "RED"]

    if crit_red:
        return "RED"
    if len(crit_watch) >= 2 and bh_fdr_significant(
        [c.bh_fdr_adjusted_p for c in crit_watch], alpha=alpha
    ):
        return "RED"
    if crit_not_impl:
        return "WATCH"
    if crit_watch or sup_red:
        return "WATCH"
    if crit_na:
        return "WATCH"
    if any(c.is_na for c in components):
        # Only supporting/diagnostic N/A remain — doesn't block GREEN.
        return "GREEN"
    return "GREEN"


def overall_status(tiles: dict[str, StatusLiteral]) -> StatusLiteral:
    """Roll module statuses to an overall status (RC v2 module→overall).

    Portfolio outcome leads (the system exists to produce alpha); a RED in any
    cascade module (research/predictor/executor/substrate) also fails overall.
    The lead tile being N/A holds the overall at WATCH — the same
    never-a-false-GREEN rule ``module_status`` applies to critical components
    (trust-battery fix, config#1958: previously an ungraded portfolio_outcome
    let the overall claim GREEN off the remaining tiles alone).
    """
    if not tiles:
        return "N/A-NOT-RUN"
    if all(s.startswith("N/A") for s in tiles.values()):
        return "N/A-NOT-RUN"
    if tiles.get("portfolio_outcome") == "RED":
        return "RED"
    if any(tiles.get(m) == "RED" for m in _CASCADE_MODULES):
        return "RED"
    n_watch = sum(1 for s in tiles.values() if s == "WATCH")
    if tiles.get("portfolio_outcome") == "WATCH" or n_watch >= 2:
        return "WATCH"
    if (tiles.get("portfolio_outcome") or "N/A").startswith("N/A"):
        return "WATCH"
    return "GREEN"


def _component_score(c: MetricRecord) -> float | None:
    """Map one component to a 0-100 score for the legacy numeric grade.

    Not a metric-specific calibration (that lived in v1's ``_*_to_grade``) — a
    uniform status+position mapping so the 0-100 stays comparable across tiles:
      - N/A-* or diagnostic            → excluded (None)
      - RED                            → 15 (capped at the red-line band)
      - position of value within [red_line, target] → [40, 90], clamped [0,100];
        GREEN beyond target can reach 100, WATCH below target floors at 40.
    Excluding N/A-NOT-IMPL (rather than averaging a neutral score) is the fix for
    the v1 inflation where unimplemented criticals propped the overall up.
    """
    if c.is_na or c.criticality == "diagnostic":
        return None
    if c.status == "RED":
        return 15.0
    if c.value is None:
        return None
    if c.target is None or c.red_line is None or c.target == c.red_line:
        return 90.0 if c.status == "GREEN" else 55.0

    higher_is_better = c.target >= c.red_line
    # Normalize value position from red_line(0.0) → target(1.0).
    span = c.target - c.red_line
    frac = (c.value - c.red_line) / span if higher_is_better else (c.red_line - c.value) / (-span)
    score = 40.0 + frac * 50.0
    return max(0.0, min(100.0, score))


def numeric_grade(components: list[MetricRecord]) -> float | None:
    """Legacy 0-100 grade: mean of per-component scores (RC v2 numeric-compat).

    N/A and diagnostic components are excluded; RED criticals drag via their
    capped 15. None when no component is scorable.
    """
    scores = [s for c in components if (s := _component_score(c)) is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)


def build_tile(module: str, components: list[MetricRecord], *, alpha: float = 0.05) -> dict:
    """Assemble a tile summary from its components."""
    from krepis.metrics import derive_letter

    status = module_status(components, alpha=alpha)
    return {
        "module": module,
        "status": status,
        "letter": derive_letter(status),
        "numeric_grade": numeric_grade(components),
        "n_components": len(components),
        "components": [c.model_dump(mode="json") for c in components],
    }
