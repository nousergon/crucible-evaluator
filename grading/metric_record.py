"""
metric_record.py — the evaluator-side factory for ``MetricRecord`` (RC v2).

The Pydantic contract + the pure status/letter/trend derivation live in
``alpha_engine_lib.metrics`` (the shared chokepoint so producer and consumers
agree). This module is the evaluator's *construction* convenience: one
``build_metric`` call that fills the full record from the raw measured value —
running the lib's ``derive_status`` / ``derive_letter`` /
``derive_trend_decoration`` and generating the operator-readable
``status_reason`` per the RC v2 N/A taxonomy (Principle 2 + the reason
templates, `system-report-card-revamp-260522.md`).

Every tile builder (Portfolio Outcome, Predictor, Executor, …) builds its
components through here so the status semantics + reason phrasing are uniform.
"""

from __future__ import annotations

from datetime import UTC, datetime

from alpha_engine_lib.metrics import (
    CriticalityLiteral,
    MetricRecord,
    MetricTypeLiteral,
    StatusLiteral,
    derive_letter,
    derive_status,
    derive_trend_decoration,
)


class MetricContractError(ValueError):
    """A critical MetricRecord violated the reliability/horizon contract.

    Raised at construction (the chokepoint) so a brittle metric can never reach
    the report card and the Director. See ROADMAP L4562 + ARCHITECTURE §18: the
    Director acts on critical tiles, so every critical metric must DECLARE a
    robust estimator + its measurement horizon, and must NOT use one of the
    proven-bad estimator classes that produced the 2026-06-07 false-positive
    trio (L4550 strict binary / L4551 sub-horizon proxy / L4554 unbounded-ratio
    mean).
    """


# Estimator classes that are STRUCTURALLY unreliable for a metric an autonomous
# agent acts on — naming any of these on a critical metric fails construction.
#   - strict_binary           : all-or-nothing flag that flips on one noisy
#                               bucket (L4550 composite monotonicity).
#   - sub_horizon_proxy       : measured on a window shorter than the strategy
#                               horizon (L4551 5d vs the 21d thesis).
#   - unbounded_ratio_mean    : mean of an unbounded per-item ratio, exploded by
#                               a few outliers (L4554 realized/MFE capture).
_FORBIDDEN_ESTIMATORS: frozenset[str] = frozenset(
    {"strict_binary", "sub_horizon_proxy", "unbounded_ratio_mean"}
)


def _fmt(v: float | None) -> str:
    """Compact human format for a metric value in a reason string."""
    if v is None:
        return "n/a"
    if abs(v) >= 1000 or (v != 0 and abs(v) < 0.001):
        return f"{v:.3g}"
    return f"{v:.4g}"


def _default_reason(
    *,
    status: StatusLiteral,
    name: str,
    value: float | None,
    n_samples: int | None,
    n_floor: int,
    target: float | None,
    red_line: float | None,
    ci_low: float | None,
    ci_high: float | None,
    na_detail: str | None,
) -> str:
    """Generate a specific, operator-readable status_reason.

    Never the generic "insufficient data" — each N/A code names *what* is
    missing (RC v2 N/A reason taxonomy).
    """
    if status == "N/A-NOT-IMPL":
        return na_detail or f"{name}: grader exists but the producer analysis is not yet implemented."
    if status == "N/A-NOT-RUN":
        return na_detail or f"{name}: producer implemented but did not run this cycle."
    if status == "N/A-MISSING-INPUT":
        return na_detail or f"{name}: a required upstream input was absent this cycle."
    if status == "N/A-LOW-N":
        return (
            f"{name}: N={n_samples if n_samples is not None else 0} below 0.5×floor "
            f"({n_floor}); CI too wide for a confident reading."
        )

    ci = ""
    if ci_low is not None and ci_high is not None:
        ci = f", CI [{_fmt(ci_low)}, {_fmt(ci_high)}]"
    bar = []
    if target is not None:
        bar.append(f"target {_fmt(target)}")
    if red_line is not None:
        bar.append(f"red-line {_fmt(red_line)}")
    bar_s = (" vs " + " / ".join(bar)) if bar else ""
    npart = f", N={n_samples}" if n_samples is not None else ""
    return f"{name} = {_fmt(value)}{ci}{npart}{bar_s} — {status}."


def build_metric(
    *,
    name: str,
    module: str,
    metric_type: MetricTypeLiteral,
    n_floor: int,
    value: float | None = None,
    n_samples: int | None = None,
    target: float | None = None,
    red_line: float | None = None,
    ci_low: float | None = None,
    ci_high: float | None = None,
    ci_method: str | None = None,
    criticality: CriticalityLiteral = "supporting",
    source_path: str,
    trend_4w: list[float] | None = None,
    trend_13w: list[float] | None = None,
    higher_is_better: bool | None = None,
    implemented: bool = True,
    ran: bool = True,
    input_present: bool = True,
    bh_fdr_adjusted_p: float | None = None,
    status: StatusLiteral | None = None,
    reason: str | None = None,
    na_detail: str | None = None,
    last_updated_utc: datetime | None = None,
    estimator: str | None = None,
    measurement_horizon: str | None = None,
    reliability: str | None = None,
) -> MetricRecord:
    """Construct a fully-populated ``MetricRecord``.

    ``status`` is derived via the lib (so producer/consumer agree) unless an
    explicit ``status`` is passed — used for band metrics (e.g. beta's two-sided
    target) that the single-direction ``derive_status`` can't express. Diagnostic
    band metrics pass their own status; everything else lets the lib derive it.

    ``higher_is_better`` only affects the trend glyph; when omitted it's inferred
    from the target/red-line ordering (matches ``derive_status``).

    ``estimator`` / ``measurement_horizon`` / ``reliability`` are the L4562
    metric-reliability contract (ARCHITECTURE §18). A ``criticality="critical"``
    metric MUST declare a non-empty ``estimator`` that is not one of the
    proven-bad classes (``_FORBIDDEN_ESTIMATORS``) — enforced here at
    construction so a brittle metric never reaches the Director. ``reliability``
    defaults to ``"high"`` for a declared critical estimator; pass ``"low"``
    explicitly for a metric with a known validity caveat (e.g. an in-sample-
    prone IC) so the digest can flag it and the Director can hedge.
    """
    # Contract applies to VALUE-BEARING criticals — the ones that can produce a
    # GREEN/WATCH/RED the Director acts on. An N/A-* critical (value is None:
    # not-impl / not-run / missing-input) carries no graded signal, so it is
    # exempt (it cannot launder a brittle estimate into a confident P0).
    if criticality == "critical" and value is not None:
        if not estimator:
            raise MetricContractError(
                f"critical metric '{name}' (value-bearing) must declare an estimator "
                f"(L4562 / ARCHITECTURE §18) — the Director acts on critical tiles."
            )
        if estimator in _FORBIDDEN_ESTIMATORS:
            raise MetricContractError(
                f"critical metric '{name}' uses forbidden estimator "
                f"'{estimator}' — brittle estimator class (L4562). Use a robust "
                f"construction (winsorized/median, continuous rank-corr + "
                f"significance) measured at the strategy horizon."
            )
        if reliability is None:
            reliability = "high"

    if status is None:
        status = derive_status(
            value=value,
            n_samples=n_samples,
            n_floor=n_floor,
            target=target,
            red_line=red_line,
            ci_low=ci_low,
            ci_high=ci_high,
            implemented=implemented,
            ran=ran,
            input_present=input_present,
        )

    if higher_is_better is None:
        higher_is_better = target is None or red_line is None or target >= red_line

    trend_source = trend_4w if trend_4w else trend_13w
    decoration = derive_trend_decoration(trend_source, higher_is_better=higher_is_better)

    status_reason = reason or _default_reason(
        status=status,
        name=name,
        value=value,
        n_samples=n_samples,
        n_floor=n_floor,
        target=target,
        red_line=red_line,
        ci_low=ci_low,
        ci_high=ci_high,
        na_detail=na_detail,
    )

    return MetricRecord(
        name=name,
        module=module,
        metric_type=metric_type,
        value=value,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_method=ci_method,
        n_samples=n_samples,
        n_floor=n_floor,
        target=target,
        red_line=red_line,
        trend_4w=trend_4w,
        trend_13w=trend_13w,
        trend_decoration=decoration,
        status=status,
        status_reason=status_reason,
        criticality=criticality,
        source_path=source_path,
        bh_fdr_adjusted_p=bh_fdr_adjusted_p,
        last_updated_utc=last_updated_utc or datetime.now(UTC),
        derived_letter=derive_letter(status),
        # L4562 reliability contract (MetricRecord allows extra fields).
        estimator=estimator,
        measurement_horizon=measurement_horizon,
        reliability=reliability,
    )
