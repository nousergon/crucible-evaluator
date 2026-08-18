"""
scorecard.py — Unified component scorecard (Layer B core).

Consumes the raw per-module analysis dicts and produces a single scorecard
with 0-100 grades and letter grades for every system component:

  Research: Scanner, 6 Sector Teams, Macro Agent, CIO, Composite Scoring
  Predictor: Meta Model, Veto Gate
  Executor: Entry Triggers, Risk Guard, Exit Rules, Position Sizing,
            Portfolio, Excursion, Action Entropy

Each grade combines precision, recall, and domain-specific metrics into a
weighted composite. Components with insufficient data receive a grade of None
and are excluded from module-level averages.

PROVENANCE — verbatim port of ``analysis/grading.py`` from
``nousergon/crucible-backtester`` @ commit f46e7e6 (2026-06-04). The
function is pure (no S3/disk reads, no backtester-internal imports), so the
port is a straight copy: the evaluator owns grading natively (Option B of
``director-implementation-plan-260604.md`` §2.4) by instantiating this pure
function against the analysis artifacts the backtester/predictor persist to S3
(see ``grading/artifacts.py``). The backtester drops its in-process grading
call once this grader is authoritative (Phase C cutover). Keep this file in
sync with the backtester source until that cutover lands; thereafter this is
the single home.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grade bands
# ---------------------------------------------------------------------------

GRADE_BANDS = [
    (90, "A"),
    (80, "A-"),
    (73, "B+"),
    (65, "B"),
    (58, "B-"),
    (50, "C+"),
    (42, "C"),
    (35, "C-"),
    (28, "D+"),
    (20, "D"),
    (0, "F"),
]


def _letter(score: float | None) -> str:
    """Map a 0-100 numeric grade to a letter grade."""
    if score is None:
        return "N/A"
    score = max(0.0, min(100.0, score))
    for threshold, letter in GRADE_BANDS:
        if score >= threshold:
            return letter
    return "F"


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _pct_to_grade(pct: float | None, baseline: float = 0.50,
                  ceiling: float = 0.80) -> float | None:
    """Map a 0-1 percentage to a 0-100 grade.

    baseline maps to 30 (D+), ceiling maps to 95 (A).
    Linear interpolation between them; clamped to [0, 100].
    """
    if pct is None:
        return None
    if ceiling == baseline:
        return 50.0
    raw = 30.0 + (pct - baseline) / (ceiling - baseline) * 65.0
    return _clamp(raw)


def _lift_to_grade(lift: float | None, floor: float = -2.0,
                   ceiling: float = 3.0, *, units: str) -> float | None:
    """Map a lift value to a 0-100 grade against pp-scale anchors.

    floor maps to 0, 0.0 maps to 40, ceiling maps to 100.

    ``units`` is REQUIRED and keyword-only. Every producer feeding this helper
    must declare the scale of the value it emits, because the anchors are
    calibrated in PERCENTAGE POINTS and most backtester producers emit raw
    return/probability FRACTIONS:

      - ``"pp"``       value is already in percentage points (e.g. 0.31 = +0.31%).
      - ``"fraction"`` value is a raw return/probability fraction (e.g. 0.0031
                       = +0.31%); scaled x100 here before grading.
      - ``"native"``   value is not a return at all and the anchors are
                       calibrated on its own scale (e.g. a Sharpe-ratio
                       difference). No conversion.

    Why this is mandatory rather than defaulted (alpha-engine-config-I2318):
    the parameter did not exist and the docstring's "percentage points" was a
    convention only. 7 of 19 live call sites passed fractions against pp
    anchors, so ``lift`` landed within ~0.5 of the zero-anchor for every
    realistic input and the term graded a CONSTANT ~40.0 (C-) regardless of the
    measured value. A default would have preserved exactly that silent failure
    for the next call site added. Mismatch is now unwritable, and
    ``tests/test_lift_units_declared.py`` fails the build if any call site
    omits the declaration.
    """
    if lift is None:
        return None
    if units == "fraction":
        lift = lift * 100.0
    elif units not in ("pp", "native"):
        raise ValueError(
            f"units must be one of 'pp' / 'fraction' / 'native', got {units!r}"
        )
    if lift <= 0:
        # Negative lift: 0 at floor, 40 at zero
        if floor == 0:
            return 40.0
        raw = 40.0 * (1.0 - lift / floor)
    else:
        # Positive lift: 40 at zero, 100 at ceiling
        if ceiling == 0:
            return 40.0
        raw = 40.0 + 60.0 * (lift / ceiling)
    return _clamp(raw)


def _fmt_lift(value: float | None, units: str, decimals: int = 2,
              suffix: str = "%") -> str | None:
    """Render a lift for a ``detail`` block, always in PERCENTAGE POINTS.

    Same ``units`` contract as :func:`_lift_to_grade`, and for the same reason:
    the display strings carried the identical defect. ``f"{lift:+.2f}%"`` on a
    raw fraction rendered a measured -0.0031 (-0.31pp) as ``"-0.00%"`` — a real
    non-zero edge shown as an exact zero, indistinguishable from "no signal".
    Formatting and grading now read the same declaration, so they cannot
    disagree about scale (alpha-engine-config-I2318).
    """
    if value is None:
        return None
    pp = value * 100.0 if units == "fraction" else value
    return f"{pp:+.{decimals}f}{suffix}"


def _ic_to_grade(ic: float | None) -> float | None:
    """Map an information coefficient to a 0-100 grade.

    IC 0.00 → 20, IC 0.05 → 55, IC 0.10 → 90.
    """
    if ic is None:
        return None
    raw = 20.0 + ic * 700.0
    return _clamp(raw)


def _ratio_to_grade(ratio: float | None, target: float = 0.75) -> float | None:
    """Map a 0-1 ratio (e.g. capture ratio) to a grade.

    0.0 → 0, target → 80, 1.0 → 100.
    """
    if ratio is None:
        return None
    if ratio <= 0:
        return 0.0
    if ratio >= 1.0:
        return 100.0
    if ratio <= target:
        raw = 80.0 * (ratio / target)
    else:
        raw = 80.0 + 20.0 * ((ratio - target) / (1.0 - target))
    return _clamp(raw)


def _band_to_grade(value: float | None, floor: float, mid: float, ceiling: float) -> float | None:
    """Linear map between three anchors: floor → 0, mid → 50, ceiling → 100.

    For metrics where the meaningful operating range isn't pegged to 0 or 1
    (Sortino, MFE/MAE ratio, normalized entropy), this gives a tunable
    three-anchor mapping. Out-of-range values are clamped to [0, 100].
    """
    if value is None:
        return None
    if floor >= mid or mid >= ceiling:
        raise ValueError(f"need floor < mid < ceiling, got {floor}/{mid}/{ceiling}")
    if value <= floor:
        return 0.0
    if value >= ceiling:
        return 100.0
    if value <= mid:
        raw = (value - floor) / (mid - floor) * 50.0
    else:
        raw = 50.0 + (value - mid) / (ceiling - mid) * 50.0
    return _clamp(raw)


def _cvar_to_grade(cvar_95: float | None, baseline: float = -0.04, ceiling: float = -0.01) -> float | None:
    """Map CVaR(95%) (negative = worse tail) to a 0-100 grade.

    Default anchors:
      - baseline (-4% mean worst-5%-day return) → 30 (D+)
      - ceiling  (-1% mean worst-5%-day return) → 95 (A)
      - 0 or positive (no tail loss) → 100
    """
    if cvar_95 is None:
        return None
    if cvar_95 >= 0.0:
        return 100.0
    if cvar_95 <= baseline:
        return 30.0
    raw = 30.0 + (cvar_95 - baseline) / (ceiling - baseline) * 65.0
    return _clamp(raw)


def _weighted_avg(components: list[tuple[float, float | None]]) -> float | None:
    """Weighted average of (weight, grade) pairs, skipping Nones.

    NOTE — this RENORMALIZES: a component whose grade is None is removed from
    the denominator and the remaining weights are rescaled to sum to 1. That is
    deliberate (a missing sub-metric must not drag a component to zero) but it
    is invisible in the returned scalar, which is why the module- and
    system-level composites additionally publish a coverage block
    (``_coverage``) naming what dropped out and how much declared weight it
    carried. See ``alpha-engine-config-I7202``.
    """
    total_w = 0.0
    total_v = 0.0
    for w, g in components:
        if g is not None:
            total_w += w
            total_v += w * g
    if total_w == 0:
        return None
    return total_v / total_w


# ---------------------------------------------------------------------------
# Declared weight tables — the SINGLE source of the module/system composites
# ---------------------------------------------------------------------------
#
# These were literals inside ``compute_scorecard`` until config-I7202. They are
# named here for two reasons: the composites below build their (weight, grade)
# pairs from them so there is exactly one declaration, and they are STAMPED ONTO
# THE ARTIFACT (``grading_weights``) so a reader can recompute the published
# grade from the card alone rather than having to read this file at the right
# commit. A published number whose weights live only in source is not a
# verifiable number.
#
# Each table must sum to 1.0 — asserted by tests/test_scorecard.py.

WEIGHT_TABLE_VERSION = "2026-08-13"

RESEARCH_WEIGHTS: dict[str, float] = {
    "scanner": 0.10,
    "sector_teams_avg": 0.25,
    "macro_agent": 0.10,
    "cio": 0.20,
    "composite_scoring": 0.20,
    "calibration_diagnostics": 0.15,
}

PREDICTOR_WEIGHTS: dict[str, float] = {
    "meta_model": 0.55,
    "veto_gate": 0.45,
}

EXECUTOR_WEIGHTS: dict[str, float] = {
    "entry_triggers": 0.10,
    "risk_guard": 0.15,
    "exit_rules": 0.15,
    "position_sizing": 0.10,
    "portfolio": 0.25,
    "excursion": 0.15,
    "action_entropy": 0.10,
}

OVERALL_WEIGHTS: dict[str, float] = {
    "research": 0.40,
    "predictor": 0.25,
    "executor": 0.35,
}

#: The renormalization rule, stamped onto the artifact in words. A reader
#: reproducing the grade needs the rule as much as the numbers.
RENORMALIZATION_RULE = (
    "A component whose grade is null is removed from the denominator; the "
    "remaining declared weights are rescaled to sum to 1. grade = "
    "sum(w_i * g_i for graded i) / sum(w_i for graded i). weight_present is "
    "sum(w_i for graded i) against a declared total of 1.0."
)

#: Coverage floor below which a grade would render PROVISIONAL rather than as a
#: letter. DELIBERATELY UNSET. Brian's 2026-08-11 ruling
#: (``ruling_detect_before_enforcing_when_the_floor_is_unmeasured``): a ceiling
#: with no measured distribution behind it gets a DETECTOR first. Measured
#: 2026-08-13 over all 15 historical cards in
#: ``s3://alpha-engine-research/evaluator/*/report_card.json``: effective
#: coverage has NEVER reached 1.0 on any card (observed range 0.39-0.91 over the
#: 12 gradable ones, three total-failure cards at 0.0), and the weight table
#: itself changed three times inside that window. There is no observation of the
#: healthy state, so any floor would be fitted entirely to degraded data. The
#: mechanism is built and parameterised; the number is not invented. When a
#: distribution exists, set this and the PROVISIONAL rendering activates.
DEFAULT_COVERAGE_FLOOR: float | None = None


def _na_reason(
    artifact: dict | None,
    *,
    label: str,
    ok_statuses: tuple[str, ...] = ("ok",),
) -> str:
    """Build a specific N/A reason naming *which* upstream artifact is missing.

    MIRRORED from ``crucible-backtester/analysis/grading.py::_na_reason`` rather
    than reinvented (``policy-shared-code``; this file is a port of that module
    and the helper landed there after the port). The generic
    ``"insufficient data"`` framed a producer-INPUT gap as a sample-size story
    when it almost always means the upstream analysis was not produced,
    persisted, or is deliberately retired — a plumbing or lifecycle fact, not a
    maturity one (config#859 Problem 1b). Two honest cases:

    * input ABSENT (``None``/empty, or present with no ``status``) → the
      analysis did not run this cycle: ``"no <label> this cycle"``;
    * input PRESENT carrying a non-ok ``status`` → a real producer signal,
      surfaced verbatim: ``"<label> status: <status>"``.

    Surfacing the producer's own status verbatim is what makes ``_skip_class``
    below able to tell a RETIRED component from a FAILED one — the generic
    string erased that distinction before any reader could see it.
    """
    if not artifact:
        return f"no {label} this cycle"
    status = artifact.get("status")
    if status is None:
        return f"no {label} this cycle"
    if status in ok_statuses:
        return f"{label} present"
    return f"{label} status: {status}"


#: Closed taxonomy for WHY a declared weight did not vote. Ordered by how a
#: reader must act on it. ``failed`` and ``failed_timeout`` are the two that
#: mean a grade was INFLATED by the drop, and they are counted separately from
#: the rest (``weight_failed``) precisely so they cannot hide inside a coverage
#: number that also absorbs legitimate absences.
SKIP_CLASSES = (
    "failed_timeout",   # the producer hit a wall — Brian ruling 2026-08-13:
                        # "anything that is timing out is considered failed"
    "failed",           # the producer ran and errored
    "retired",          # DECLARED end-of-life; the weight should be removed
    "not_implemented",  # the producer has never existed
    "input_absent",     # the artifact did not arrive this cycle
    "insufficient_data",  # ran, produced, sample below floor
    "unknown",
)

_FAILED_SKIP_CLASSES = frozenset({"failed_timeout", "failed"})

_TIMEOUT_TOKENS = ("timeout", "timed out", "timed_out", "deadline", "wall clock", "walltime")
_FAILED_TOKENS = ("failed", "failure", "error", "exception", "crash", "aborted", "killed")
_RETIRED_TOKENS = ("retired", "deprecated", "decommissioned", "sunset")
_NOT_IMPL_TOKENS = ("not implemented", "not_implemented", "never produced", "simulation-only", "deferred")


def _skip_class(entry: Any, reason: str) -> str:
    """Classify a skip into ``SKIP_CLASSES`` from the producer's own words.

    Deliberately keyed on the reason string the grader emitted, because that is
    the only channel the producer's status reaches this layer through. Ordered
    most-serious-first so a reason naming both a timeout and insufficient data
    classifies as the timeout.
    """
    if entry is None:
        return "input_absent"
    text = (reason or "").lower()
    if any(t in text for t in _TIMEOUT_TOKENS):
        return "failed_timeout"
    if any(t in text for t in _FAILED_TOKENS):
        return "failed"
    if any(t in text for t in _RETIRED_TOKENS):
        return "retired"
    if any(t in text for t in _NOT_IMPL_TOKENS):
        return "not_implemented"
    if "no " in text and "this cycle" in text:
        return "input_absent"
    if "insufficient" in text or "only " in text:
        return "insufficient_data"
    return "unknown"


def _skip_reason(entry: Any) -> str:
    """Why a declared component contributed nothing, from its own emitted block.

    Distinguishes the engineering states that a bare null conflates
    (``observability-policy.md`` §3.5's closed N/A taxonomy, applied here):
    the producer artifact never arrived, versus it arrived and the grade was
    not computable, versus the grader said why.
    """
    if entry is None:
        return "component absent from this card (input artifact not read)"
    if not isinstance(entry, dict):
        return f"component block is not a grade block (got {type(entry).__name__})"
    reason = entry.get("reason")
    if isinstance(reason, str) and reason:
        return reason
    return "grade null, no reason emitted by the grader"


def _coverage(
    weights: dict[str, float],
    graded: dict[str, Any],
    *,
    effective: dict[str, float] | None = None,
    effective_failed: dict[str, float] | None = None,
    effective_failed_members: dict[str, list[str]] | None = None,
    floor: float | None = None,
) -> dict:
    """Coverage of one composite level: how much declared weight actually voted.

    ``graded`` maps each declared component name to its emitted block (or None
    when the component was not emitted at all — the case a card cannot show
    today, because an absent producer removes the key entirely and the weight
    vanishes with it).

    ``effective`` optionally maps a child name to that child's own
    ``weight_present_effective``, which is what makes the system level honest:
    the overall grade can be 100% covered *at its own level* — all three module
    grades non-null — while two of those modules were themselves computed over
    little more than half their declared weight. Without the recursive number a
    reader sees 1.0 and concludes the card is complete.
    """
    declared = sum(weights.values())
    present = 0.0
    present_effective = 0.0
    failed_w = 0.0
    skips: list[dict] = []
    for name, w in weights.items():
        entry = graded.get(name)
        grade = entry.get("grade") if isinstance(entry, dict) else None
        if grade is None:
            reason = _skip_reason(entry)
            klass = _skip_class(entry, reason)
            if klass in _FAILED_SKIP_CLASSES:
                failed_w += w
            skips.append(
                {"component": name, "weight": w, "reason": reason, "skip_class": klass},
            )
            continue
        present += w
        present_effective += w * (effective or {}).get(name, 1.0)
        # A failure inside a CHILD is a failure this level's grade absorbed. It
        # must not become invisible one level up — that is how a masked failure
        # reaches the headline number looking like a clean C+.
        failed_w += w * (effective_failed or {}).get(name, 0.0)

    weight_present = present / declared if declared else 0.0
    weight_present_effective = present_effective / declared if declared else 0.0
    weight_failed = failed_w / declared if declared else 0.0
    failed = [s["component"] for s in skips if s["skip_class"] in _FAILED_SKIP_CLASSES]
    # Name the MEMBERS, not the container: a roll-up that reports "executor
    # failed" sends the reader down a level to find out what actually failed
    # (``observability-policy.md`` §7.2a — a group alert that omits its member
    # list has traded spam for uselessness).
    for name, fw in (effective_failed or {}).items():
        if not fw:
            continue
        members = (effective_failed_members or {}).get(name) or [name]
        failed += [m for m in members if m not in failed]

    # Brian ruling 2026-08-13: "anything that is timing out is considered failed
    # now". A FAILED component dropped from the denominator INFLATES the grade —
    # the same defect as the silent renormalization, in its most damaging form.
    # This build does not change how a failure scores (that moves the grade and
    # is Brian's call); it makes the masking impossible to miss, and the
    # qualifier is distinct so a surface cannot render it as ordinary partial
    # coverage.
    #
    # The qualifier reads the EFFECTIVE number, not this level's own. The
    # overall level is 100% covered at its own level whenever all three module
    # grades are non-null — which is exactly the reading that let 55.68 render
    # as an unqualified C+ over 73% of the declared leaf weight.
    if present == 0.0:
        qualifier = "UNGRADED"
    elif failed:
        qualifier = "PARTIAL-MASKED-FAILURE"
    elif skips or weight_present_effective < 1.0 - 1e-9:
        qualifier = "PARTIAL"
    else:
        qualifier = "COMPLETE"

    block = {
        "weight_declared": round(declared, 6),
        "weight_present": round(weight_present, 6),
        "weight_present_effective": round(weight_present_effective, 6),
        "components_declared": len(weights),
        "components_present": len(weights) - len(skips),
        "components_skipped": [s["component"] for s in skips],
        # Zero is a VALUE here, not an absence: a card reporting
        # weight_failed 0.0 is asserting no failure was dropped, which is what
        # makes a non-zero one readable as the finding it is.
        "weight_failed": round(weight_failed, 6),
        "components_failed": failed,
        "skip_classes": {
            s["component"]: s["skip_class"] for s in skips
        },
        "skips": skips,
        "weights": dict(weights),
        "renormalized": bool(skips) and present > 0.0,
        "renormalization_factor": (
            round(declared / present, 6) if present > 0.0 else None
        ),
        # The qualifier is a MEASURED fact (coverage < declared), not a
        # threshold judgement — it needs no invented number and is what stops a
        # renormalized grade rendering as a plain letter.
        "qualifier": qualifier,
        "floor": floor,
        "floor_status": (
            "unset-unmeasured (config-I7202: no card has ever reached full "
            "coverage, so there is no distribution to place a floor below)"
            if floor is None else "set"
        ),
        "provisional": bool(floor is not None and weight_present_effective < floor),
    }
    return block


def _display(letter: str, coverage: dict | None) -> str:
    """The string a surface should render instead of the bare letter.

    A renormalized grade never renders as a plain letter — deliverable 3 of
    config-I7202, satisfied without inventing a threshold.
    """
    if not isinstance(coverage, dict):
        return f"{letter} (coverage UNKNOWN)"
    q = coverage.get("qualifier")
    if q == "COMPLETE":
        return letter
    if q == "UNGRADED":
        return "N/A (no component contributed)"
    pct = coverage.get("weight_present_effective")
    pct_s = f"{pct:.0%}" if isinstance(pct, (int, float)) else "?"
    if q == "PARTIAL-MASKED-FAILURE":
        failed = coverage.get("components_failed") or []
        fw = coverage.get("weight_failed")
        fw_s = f"{fw:.0%}" if isinstance(fw, (int, float)) else "?"
        return (
            f"{letter} (PARTIAL — {pct_s} of declared weight; "
            f"{fw_s} DROPPED ON FAILURE: {', '.join(failed)} — grade is inflated)"
        )
    return f"{letter} (PARTIAL — {pct_s} of declared weight)"


#: Emitted in place of a real coverage block if the coverage computation itself
#: raises. Deviation from the fleet's fail-loud default is DELIBERATE and
#: narrow: (a) the failure mode swallowed is a defect in this reporting code,
#: which is additive to a grade already computed above it by the unchanged
#: arithmetic path; (b) the primary deliverable — the grade and every component
#: decomposition — is produced before this block runs and is unaffected; (c) the
#: recording surface is this literal marker on the artifact plus a
#: ``logger.exception`` carrying the greppable string below. Withholding the
#: coverage guarantee beats failing the report-card stage
#: (``sf-pipeline-policy.md`` §2.3a).
COVERAGE_UNKNOWN_MARKER = "scorecard_coverage_failed"
_COVERAGE_UNKNOWN = {
    "weight_declared": None,
    "weight_present": None,
    "weight_present_effective": None,
    "components_declared": None,
    "components_present": None,
    "components_skipped": None,
    "weight_failed": None,
    "components_failed": None,
    "skip_classes": None,
    "skips": None,
    "weights": None,
    "renormalized": None,
    "renormalization_factor": None,
    "qualifier": "UNKNOWN",
    "floor": None,
    "floor_status": "unknown",
    "provisional": None,
    "error": COVERAGE_UNKNOWN_MARKER,
}


def _safe_get(d: dict | None, *keys, default=None) -> Any:
    """Safely traverse nested dicts."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


# ---------------------------------------------------------------------------
# Component grading functions
# ---------------------------------------------------------------------------

def _selection_edge_pp(clf: dict | None) -> float | None:
    """Precision minus the cohort base rate, in percentage points.

    THE metric for a deliberately selective filter, and the one the composite
    was missing. Precision alone is unreadable without the base rate it is
    being compared against: 44.1% precision is a 10x lift against a 4% base
    rate and pure noise against a 43% one. Only the confusion matrix carries
    the base rate, so it is derived here rather than trusted from a sibling
    field.
    """
    if not isinstance(clf, dict):
        return None
    try:
        tp, fp, fn, tn = (float(clf[k]) for k in ("tp", "fp", "fn", "tn"))
    except (KeyError, TypeError, ValueError):
        return None
    n = tp + fp + fn + tn
    selected = tp + fp
    if n <= 0 or selected <= 0:
        return None
    precision = tp / selected
    base_rate = (tp + fn) / n
    return (precision - base_rate) * 100.0


def _max_achievable_recall(clf: dict | None) -> float | None:
    """Ceiling on recall given how many names this filter actually selects.

    A filter selecting S names out of P positives cannot exceed S/P recall even
    if every pick is correct. Used to detect the case where the recall band is
    unreachable by construction and grading against it is a category error.
    """
    if not isinstance(clf, dict):
        return None
    try:
        tp, fp, fn = (float(clf[k]) for k in ("tp", "fp", "fn"))
    except (KeyError, TypeError, ValueError):
        return None
    positives = tp + fn
    if positives <= 0:
        return None
    return min(1.0, (tp + fp) / positives)


# Recall band for the scanner. A scanner is a SELECTIVE instrument, so this
# band is only meaningful when the filter's pass rate makes it reachable —
# see the guard in _grade_scanner.
_SCANNER_RECALL_BASELINE = 0.10
_SCANNER_RECALL_CEILING = 0.40


def _grade_scanner(e2e: dict | None, scanner_opt: dict | None) -> dict:
    """Grade the quant scanner filter.

    ARM (alpha-engine-config-I2318): the producer stamps ``scanner_lift.arm``
    precisely so a consumer cannot present a retired gate's record as "the
    scanner" unlabeled (crucible-backtester ``analysis/end_to_end.py`` §36-42).
    This grader dropped that field, so the report card showed the retired
    ``tech_score_baseline`` gate — retired from the live feed 2026-06-29 — as an
    unqualified "scanner" grade. The arm is now carried into ``detail`` and
    ``arm``, alongside a pointer to the component that grades the LIVE feed
    (``attractiveness_ic``, config-I2994).

    HORIZON: grades ``classification_21d`` — the canonical horizon (L4551) —
    and falls back to the 5d ``classification`` only when the 21d block is
    absent. The 5d block remains in ``detail`` as a diagnostic.

    RECALL: graded only when the band is physically reachable. A filter passing
    653 of 15,274 name-days against 6,592 positives caps at 9.9% recall with
    PERFECT precision — below the 10% baseline — so the term contributed a
    near-floor constant that measured the filter's selectivity, not its skill.
    Where the band is unreachable, selection edge (precision - base rate)
    carries the weight instead.
    """
    sl = _safe_get(e2e, "scanner_lift")
    if not sl or _safe_get(sl, "n_passing") is None:
        return {"grade": None, "letter": "N/A", "reason": "insufficient data"}

    lift = _safe_get(sl, "lift")
    n_passing = _safe_get(sl, "n_passing", default=0)
    n_universe = _safe_get(sl, "n_universe", default=1)
    arm = _safe_get(sl, "arm")

    # Canonical 21d horizon first (L4551); 5d is the legacy diagnostic.
    clf_21d = _safe_get(sl, "classification_21d")
    clf_5d = _safe_get(sl, "classification")
    clf = clf_21d if isinstance(clf_21d, dict) and clf_21d.get("tp") is not None else clf_5d
    horizon = "21d" if clf is clf_21d and clf_21d else "5d"

    # Precision/recall from classification metrics (if available)
    precision = _safe_get(clf, "precision")
    recall = _safe_get(clf, "recall")
    f1 = _safe_get(clf, "f1")

    precision_g = _pct_to_grade(precision, baseline=0.40, ceiling=0.65) if precision is not None else None

    # Only grade recall where the band is reachable at this pass rate.
    max_recall = _max_achievable_recall(clf)
    recall_reachable = max_recall is None or max_recall >= _SCANNER_RECALL_BASELINE
    if recall is not None and recall_reachable:
        recall_g = _pct_to_grade(
            recall, baseline=_SCANNER_RECALL_BASELINE, ceiling=_SCANNER_RECALL_CEILING,
        )
    else:
        recall_g = None

    # Selection edge: precision over the cohort base rate, in pp. Anchors are
    # deliberately tight — a stock filter clearing its own base rate by 5pp on
    # a 21d horizon is a strong instrument, not a mediocre one.
    edge_pp = _selection_edge_pp(clf)
    edge_g = _lift_to_grade(edge_pp, floor=-5.0, ceiling=5.0, units="pp")

    # Fallback to lift if no classification data
    lift_g = _lift_to_grade(lift, floor=-1.5, ceiling=2.5, units="fraction")

    # Leakage from scanner_opt (lower is better)
    leakage = _safe_get(scanner_opt, "leakage_pct")
    if leakage is not None:
        leakage_g = _clamp(95.0 - leakage * 283.0)
    else:
        leakage_g = None

    if precision_g is not None:
        grade = _weighted_avg([
            (0.25, precision_g),
            (0.25, edge_g),
            (0.15, recall_g),
            (0.15, lift_g),
            (0.20, leakage_g),
        ])
    else:
        grade = _weighted_avg([
            (0.55, lift_g),
            (0.45, leakage_g),
        ])

    detail = {"horizon": horizon}
    if precision is not None:
        detail["precision"] = f"{precision:.1%}"
    if edge_pp is not None:
        detail["selection_edge"] = _fmt_lift(edge_pp, "pp", suffix="pp")
        detail["base_rate"] = f"{(precision - edge_pp / 100.0):.1%}" if precision is not None else None
    if recall is not None:
        detail["recall"] = f"{recall:.1%}"
    if max_recall is not None and not recall_reachable:
        # State the reason the term is absent. A missing component that says
        # nothing about why is indistinguishable from one that was never wired.
        detail["recall_graded"] = False
        detail["recall_ceiling"] = f"{max_recall:.1%}"
        detail["recall_not_graded_reason"] = (
            f"max achievable recall at this pass rate is {max_recall:.1%}, below the "
            f"{_SCANNER_RECALL_BASELINE:.0%} grading baseline — a perfect filter would "
            f"score at the floor, so the band measures selectivity, not skill"
        )
    if f1 is not None:
        detail["f1"] = f"{f1:.3f}"
    if lift is not None:
        detail["lift"] = _fmt_lift(lift, "fraction")
    if leakage is not None:
        detail["leakage"] = f"{leakage:.0%}"
    detail["n_passing"] = n_passing
    detail["n_universe"] = n_universe
    detail["n_universe_basis"] = "(ticker, eval_date) observations, not tickers"
    if arm:
        detail["arm"] = arm
        detail["live_arm_graded_by"] = (
            "attractiveness_ic (config-I2994) — the live champion feed's "
            "attractiveness_score IC. This component is NOT the live scanner."
        )
    # 5d stays visible as a diagnostic whenever 21d is what got graded.
    if horizon == "21d" and isinstance(clf_5d, dict):
        p5 = clf_5d.get("precision")
        if p5 is not None:
            detail["precision_5d_diagnostic"] = f"{p5:.1%}"
    detail = {k: v for k, v in detail.items() if v is not None}

    out = {"grade": grade, "letter": _letter(grade), "detail": detail}
    if arm:
        # Top-level too, not only inside detail — any consumer rendering the
        # letter without the detail block still sees which arm it describes.
        out["arm"] = arm
    return out


def _grade_sector_team(team: dict, team_metrics: dict | None = None) -> dict:
    """Grade a single sector team from e2e_lift team_lift entry.

    When ``team_metrics`` is provided for the team_id (the per-team
    skilled-risk-taking metric stack: IC + expectancy + MFE/MAE +
    risk-matched alpha vs both benchmarks), the composite uses those
    inputs per the evaluator-revamp spec:
      - 25% IC (rank correlation, conviction → forward return)
      - 20% expectancy_per_unit_loss (R-multiple form)
      - 15% MFE/MAE ratio (band 0.8 → 1.5 → 2.0)
      - 20% alpha vs EW-high-vol benchmark (lift in pp)
      - 20% alpha vs beta-matched SPY (lift in pp)

    When ``team_metrics`` is absent, falls back to the legacy
    precision/recall/lift composite — preserves backward compatibility
    for callers that haven't been wired through to the new metrics yet.
    """
    team_id = team.get("team_id", "unknown")
    n_picks = team.get("n_picks", 0)

    if n_picks < 3:
        return {
            "team_id": team_id, "grade": None, "letter": "N/A",
            "reason": f"only {n_picks} picks", "n_picks": n_picks,
        }

    # ── New metric-stack path: skilled-risk-taking composite ────────────
    metrics = (team_metrics or {}).get(team_id)
    if isinstance(metrics, dict) and metrics:
        return _grade_team_skill_composite(team_id, n_picks, team, metrics)

    # ── Legacy path: lift + classification ──────────────────────────────
    lift_vs_sector = team.get("lift")
    lift_vs_quant = team.get("lift_vs_quant")
    clf = team.get("classification")

    # Classification metrics (if available)
    precision = _safe_get(clf, "precision")
    recall = _safe_get(clf, "recall")
    f1 = _safe_get(clf, "f1")

    precision_g = _pct_to_grade(precision, baseline=0.40, ceiling=0.70) if precision is not None else None
    recall_g = _pct_to_grade(recall, baseline=0.10, ceiling=0.50) if recall is not None else None

    # Lift-based grades (always available)
    lift_sector_g = _lift_to_grade(lift_vs_sector, floor=-2.0, ceiling=3.0, units="fraction")
    lift_quant_g = _lift_to_grade(lift_vs_quant, floor=-2.0, ceiling=3.0, units="fraction")

    if precision_g is not None and recall_g is not None:
        grade = _weighted_avg([
            (0.30, precision_g),
            (0.20, recall_g),
            (0.25, lift_sector_g),
            (0.25, lift_quant_g),
        ])
    else:
        grade = _weighted_avg([
            (0.55, lift_sector_g),
            (0.45, lift_quant_g),
        ])

    detail = {}
    if precision is not None:
        detail["precision"] = f"{precision:.1%}"
    if recall is not None:
        detail["recall"] = f"{recall:.1%}"
    if f1 is not None:
        detail["f1"] = f"{f1:.3f}"
    if lift_vs_sector is not None:
        detail["lift_vs_sector"] = _fmt_lift(lift_vs_sector, "fraction")
    if lift_vs_quant is not None:
        detail["lift_vs_quant"] = _fmt_lift(lift_vs_quant, "fraction")
    detail["n_picks"] = n_picks

    return {
        "team_id": team_id, "grade": grade, "letter": _letter(grade),
        "detail": detail,
    }


def _grade_team_skill_composite(
    team_id: str, n_picks: int, team: dict, metrics: dict,
) -> dict:
    """Skilled-risk-taking composite per evaluator-revamp spec.

    Expects ``metrics`` to be a dict with sub-keys:
      - ic: ICResult from compute_ic
      - expectancy: ExpectancyResult from compute_expectancy
      - excursion: ExcursionSummary from summarize_excursions
      - alpha_vs_ew_high_vol: BenchmarkResult from compute_alpha_vs_benchmark
      - alpha_vs_beta_spy: BenchmarkResult from compute_alpha_vs_benchmark

    Any sub-metric absent or status != "ok" → that component drops out
    of the composite (weighted avg skips Nones).
    """
    ic = metrics.get("ic") or {}
    exp = metrics.get("expectancy") or {}
    exc = metrics.get("excursion") or {}
    ew = metrics.get("alpha_vs_ew_high_vol") or {}
    bm = metrics.get("alpha_vs_beta_spy") or {}

    ic_g = _ic_to_grade(ic.get("ic")) if ic.get("status") == "ok" else None
    expectancy_g = _ratio_to_grade(
        exp.get("expectancy_per_unit_loss"), target=0.4,
    ) if exp.get("status") == "ok" else None
    # MFE/MAE ratio: 0.8 = below floor (worse than YOLO), 1.5 = decent,
    # 2.0+ = strong skilled risk-taking.
    mfe_mae_g = _band_to_grade(
        exc.get("mean_mfe_mae_ratio"), floor=0.8, mid=1.5, ceiling=2.0,
    ) if exc.get("status") == "ok" else None
    # Excess returns from compute_alpha_vs_benchmark are decimal fractions
    # (e.g. 0.012 = +1.2%); the units="fraction" declaration does the pp
    # conversion inside _lift_to_grade / _fmt_lift.
    ew_lift = ew.get("excess_return")
    bm_lift = bm.get("excess_return")
    ew_g = _lift_to_grade(ew_lift, floor=-3.0, ceiling=4.0, units="fraction")
    bm_g = _lift_to_grade(bm_lift, floor=-3.0, ceiling=4.0, units="fraction")

    grade = _weighted_avg([
        (0.25, ic_g),
        (0.20, expectancy_g),
        (0.15, mfe_mae_g),
        (0.20, ew_g),
        (0.20, bm_g),
    ])

    # When all five sub-metrics fail, the legacy fallback would render
    # "insufficient data" with no n_picks context — Financials/Industrials/
    # Technology in 2026-05-07's report card hit this path with 6/3/3 picks
    # respectively, which read as a contradiction against the populated
    # team-lift table below. Surface n_picks + which sub-metrics dropped
    # so the operator can interpret the gap (usually IC needs ≥10 samples,
    # benchmarks need EW-high-vol overlap).
    if grade is None:
        passed = [
            label for label, g in (
                ("ic", ic_g), ("expectancy", expectancy_g),
                ("mfe_mae", mfe_mae_g),
                ("alpha_vs_ew", ew_g), ("alpha_vs_beta_spy", bm_g),
            ) if g is not None
        ]
        return {
            "team_id": team_id, "grade": None, "letter": "N/A",
            "reason": (
                f"{n_picks} picks but 0/5 sub-metrics computable"
                if not passed
                else f"{n_picks} picks, only {len(passed)}/5 sub-metrics computable ({', '.join(passed)})"
            ),
            "n_picks": n_picks,
        }

    detail: dict[str, str | float | int] = {"n_picks": n_picks}
    if ic.get("ic") is not None:
        detail["ic"] = round(ic["ic"], 3)
    if exp.get("expectancy") is not None:
        detail["expectancy"] = round(exp["expectancy"], 4)
    if exp.get("expectancy_per_unit_loss") is not None:
        detail["expectancy_per_unit_loss"] = round(exp["expectancy_per_unit_loss"], 3)
    if exp.get("hit_rate") is not None:
        detail["hit_rate"] = f"{exp['hit_rate']:.1%}"
    if exp.get("win_loss_ratio") is not None:
        detail["win_loss_ratio"] = round(exp["win_loss_ratio"], 2)
    if exc.get("mean_mfe_mae_ratio") is not None:
        detail["mfe_mae_ratio"] = round(exc["mean_mfe_mae_ratio"], 2)
    if exc.get("pct_high_quality") is not None:
        detail["pct_high_quality"] = f"{exc['pct_high_quality']:.1%}"
    if ew_lift is not None:
        detail["alpha_vs_ew_high_vol"] = _fmt_lift(ew_lift, "fraction")
    if bm_lift is not None:
        detail["alpha_vs_beta_spy"] = _fmt_lift(bm_lift, "fraction")

    return {
        "team_id": team_id, "grade": grade, "letter": _letter(grade),
        "detail": detail,
    }


def _grade_macro(macro_eval: dict | None) -> dict:
    """Grade the macro agent's contribution."""
    if not macro_eval or macro_eval.get("status") != "ok":
        return {"grade": None, "letter": "N/A",
                "reason": _na_reason(macro_eval, label="macro evaluation")}

    acc_lift = macro_eval.get("accuracy_lift")
    alpha_lift = macro_eval.get("alpha_lift")
    assessment = macro_eval.get("assessment", "neutral")

    acc_g = _lift_to_grade(acc_lift, floor=-5.0, ceiling=5.0, units="fraction") if acc_lift is not None else None
    alpha_g = _lift_to_grade(alpha_lift, floor=-1.0, ceiling=2.0, units="fraction") if alpha_lift is not None else None

    grade = _weighted_avg([
        (0.50, acc_g),
        (0.50, alpha_g),
    ])

    detail = {}
    if acc_lift is not None:
        detail["accuracy_lift"] = _fmt_lift(acc_lift, "fraction", decimals=2, suffix="pp")
    if alpha_lift is not None:
        detail["alpha_lift"] = _fmt_lift(alpha_lift, "fraction")
    detail["assessment"] = assessment

    return {"grade": grade, "letter": _letter(grade), "detail": detail}


def _grade_cio(e2e: dict | None, cio_opt: dict | None) -> dict:
    """Grade the CIO's selection decisions."""
    cio_lift = _safe_get(e2e, "cio_lift")
    cio_vs = _safe_get(e2e, "cio_vs_ranking")

    if not cio_lift or _safe_get(cio_lift, "n_advance", default=0) < 3:
        # The producer has emitted ``{"status": "retired", ...}`` here on every
        # card since 2026-07-17 (six-team + CIO research graph retired
        # 2026-07-12, config#1580 / config-I2993) and this branch reported it as
        # "insufficient data" — a DECISION rendered as a defect, and 20% of the
        # declared research weight silently renormalized away with it
        # (config-I7202). Surface the producer's own status verbatim.
        return {
            "grade": None, "letter": "N/A",
            "reason": _na_reason(cio_lift, label="CIO lift"),
        }

    clf = _safe_get(cio_lift, "classification")
    precision = _safe_get(clf, "precision")
    recall = _safe_get(clf, "recall")
    f1 = _safe_get(clf, "f1")

    precision_g = _pct_to_grade(precision, baseline=0.40, ceiling=0.75) if precision is not None else None
    recall_g = _pct_to_grade(recall, baseline=0.30, ceiling=0.70) if recall is not None else None

    # Lift-based grades (fallback/complement)
    adv_lift = cio_lift.get("lift")
    lift_g = _lift_to_grade(adv_lift, floor=-3.0, ceiling=3.0, units="fraction")

    # CIO vs mechanical ranking baseline
    ranking_lift = _safe_get(cio_vs, "lift")
    ranking_g = _lift_to_grade(ranking_lift, floor=-2.0, ceiling=2.0, units="fraction") if ranking_lift is not None else None

    if precision_g is not None and recall_g is not None:
        grade = _weighted_avg([
            (0.30, precision_g),
            (0.20, recall_g),
            (0.25, lift_g),
            (0.25, ranking_g),
        ])
    else:
        # Fallback: rejection spread as recall proxy
        reject_avg = cio_lift.get("reject_avg")
        advance_avg = cio_lift.get("advance_avg")
        if reject_avg is not None and advance_avg is not None:
            rejection_spread = advance_avg - reject_avg
            rejection_g = _lift_to_grade(rejection_spread, floor=-2.0, ceiling=4.0, units="fraction")
        else:
            rejection_g = None
        grade = _weighted_avg([
            (0.40, lift_g),
            (0.30, rejection_g),
            (0.30, ranking_g),
        ])

    detail = {}
    if precision is not None:
        detail["precision"] = f"{precision:.1%}"
    if recall is not None:
        detail["recall"] = f"{recall:.1%}"
    if f1 is not None:
        detail["f1"] = f"{f1:.3f}"
    if adv_lift is not None:
        detail["selection_lift"] = _fmt_lift(adv_lift, "fraction")
    if ranking_lift is not None:
        detail["vs_ranking"] = _fmt_lift(ranking_lift, "fraction")
    detail["n_advance"] = cio_lift.get("n_advance", 0)
    detail["n_reject"] = cio_lift.get("n_reject", 0)

    return {"grade": grade, "letter": _letter(grade), "detail": detail}


def _grade_composite_scoring(signal_quality: dict | None,
                             score_cal: dict | None) -> dict:
    """Grade the composite scoring system (monotonicity + bucket accuracy)."""
    if not signal_quality or signal_quality.get("status") != "ok":
        return {"grade": None, "letter": "N/A",
                "reason": _na_reason(signal_quality, label="signal quality")}

    overall = signal_quality.get("overall", {})
    buckets = signal_quality.get("by_score_bucket", [])

    # Overall accuracy at the canonical 21d horizon (config-I7208; the
    # producer retired accuracy_10d — mirrors
    # crucible-backtester/analysis/grading.py::_grade_composite_scoring,
    # config#1456).
    acc_21d = overall.get("accuracy_21d")
    acc_g = _pct_to_grade(acc_21d, baseline=0.45, ceiling=0.70)

    # High-score bucket accuracy (90+ should be highest)
    high_bucket = next((b for b in buckets if b.get("bucket") == "90+"), None)
    high_acc = _safe_get(high_bucket, "accuracy_21d") if high_bucket else None
    high_g = _pct_to_grade(high_acc, baseline=0.50, ceiling=0.80)

    # Monotonicity from calibration
    monotonic = _safe_get(score_cal, "monotonic")
    mono_g = 90.0 if monotonic else (40.0 if monotonic is not None else None)

    grade = _weighted_avg([
        (0.40, acc_g),
        (0.30, high_g),
        (0.30, mono_g),
    ])

    detail = {}
    if acc_21d is not None:
        detail["accuracy_21d"] = f"{acc_21d:.1%}"
    elif overall:
        # Fail loud: the artifact is present but the canonical-horizon key is
        # absent — say why instead of letting acc_g silently drop out of the
        # weighted average with no trace in the detail block.
        detail["accuracy_21d_reason"] = "no accuracy_21d in signal_quality.overall this cycle"
    if high_acc is not None:
        detail["90+_accuracy"] = f"{high_acc:.1%}"
    if monotonic is not None:
        detail["monotonic"] = "YES" if monotonic else "NO"

    return {"grade": grade, "letter": _letter(grade), "detail": detail}


def _grade_meta_model(predictor_sizing: dict | None,
                      veto_result: dict | None) -> dict:
    """Grade the predictor meta-model quality (rank IC, stability, sizing lift).

    Named for the v3 predictor architecture (4 specialized models + ridge
    meta-learner, deployed 2026-04-01). Prior to v3 this graded a single
    LightGBM; the signals consumed here (overall_rank_ic, sizing_lift,
    weekly_ic) are identical across architectures, so the function kept
    working through the cutover but the name was stale.
    """
    ic = _safe_get(predictor_sizing, "overall_rank_ic")
    hit_rate = None

    # Try to get hit rate from predictor_sizing weekly data
    recent_weeks = _safe_get(predictor_sizing, "weekly_ic") or []
    n_positive = _safe_get(predictor_sizing, "recent_positive_weeks", default=0)
    n_total = _safe_get(predictor_sizing, "recent_total_weeks", default=0)

    if not predictor_sizing or predictor_sizing.get("status") != "ok":
        # Fall back to veto result for any signal of model quality
        if not veto_result or veto_result.get("status") not in ("ok", "insufficient_lift"):
            return {"grade": None, "letter": "N/A",
                    "reason": _na_reason(predictor_sizing, label="predictor sizing")}

    ic_g = _ic_to_grade(ic)

    # Stability: fraction of recent weeks with positive IC
    if n_total > 0:
        stability = n_positive / n_total
        stability_g = _pct_to_grade(stability, baseline=0.40, ceiling=0.85)
    else:
        stability_g = None

    # Sizing lift (does p_up signal correlate with returns?)
    sizing_lift = _safe_get(predictor_sizing, "sizing_lift")
    sizing_g = _lift_to_grade(sizing_lift, floor=-1.0, ceiling=2.0, units="fraction") if sizing_lift is not None else None

    grade = _weighted_avg([
        (0.45, ic_g),
        (0.30, stability_g),
        (0.25, sizing_g),
    ])

    detail = {}
    if ic is not None:
        detail["rank_ic"] = f"{ic:.4f}"
    if n_total > 0:
        detail["stability"] = f"{n_positive}/{n_total} weeks positive"
    if sizing_lift is not None:
        detail["sizing_lift"] = _fmt_lift(sizing_lift, "fraction")

    return {"grade": grade, "letter": _letter(grade), "detail": detail}


def _grade_veto_gate(veto_result: dict | None,
                     veto_value: dict | None) -> dict:
    """Grade the predictor's veto system."""
    if not veto_result or veto_result.get("status") not in ("ok", "insufficient_lift"):
        return {"grade": None, "letter": "N/A",
                "reason": _na_reason(veto_result, label="veto analysis",
                                     ok_statuses=("ok", "insufficient_lift"))}

    # Find the recommended threshold's metrics
    thresholds = veto_result.get("thresholds", [])
    rec_thresh = veto_result.get("recommended_threshold")
    rec_row = next((t for t in thresholds if t.get("confidence") == rec_thresh), None)

    precision = _safe_get(rec_row, "precision")
    recall = _safe_get(rec_row, "recall")
    f1 = _safe_get(rec_row, "f1")
    lift = _safe_get(rec_row, "lift")

    precision_g = _pct_to_grade(precision, baseline=0.45, ceiling=0.80)
    recall_g = _pct_to_grade(recall, baseline=0.10, ceiling=0.50) if recall is not None else None

    # Net dollar value (positive = veto system saves money)
    net_value = _safe_get(veto_value, "net_value")
    if net_value is not None:
        value_g = _clamp(50.0 + net_value / 20.0)
    else:
        value_g = None

    if recall_g is not None:
        grade = _weighted_avg([
            (0.30, precision_g),
            (0.20, recall_g),
            (0.20, _lift_to_grade(lift, floor=-5.0, ceiling=20.0, units="fraction") if lift is not None else None),
            (0.30, value_g),
        ])
    else:
        grade = _weighted_avg([
            (0.40, precision_g),
            (0.30, _lift_to_grade(lift, floor=-5.0, ceiling=20.0, units="fraction") if lift is not None else None),
            (0.30, value_g),
        ])

    detail = {}
    if precision is not None:
        detail["precision"] = f"{precision:.1%}"
    if recall is not None:
        detail["recall"] = f"{recall:.1%}"
    if f1 is not None:
        detail["f1"] = f"{f1:.3f}"
    if lift is not None:
        detail["lift"] = _fmt_lift(lift, "fraction", decimals=2, suffix="pp")
    if net_value is not None:
        detail["net_value"] = f"${net_value:+,.0f}"
    detail["threshold"] = rec_thresh

    return {"grade": grade, "letter": _letter(grade), "detail": detail}


def _grade_entry_triggers(trigger_scorecard: dict | None) -> dict:
    """Grade entry trigger effectiveness."""
    if not trigger_scorecard or trigger_scorecard.get("status") != "ok":
        return {"grade": None, "letter": "N/A",
                "reason": _na_reason(trigger_scorecard, label="trigger scorecard")}

    summary = trigger_scorecard.get("summary", {})
    triggers = trigger_scorecard.get("triggers", [])

    # Overall slippage vs signal (negative = bought below signal = good)
    slip = summary.get("avg_slippage_vs_signal")
    if slip is not None:
        # -1% → 90, 0% → 55, +1% → 20
        slip_g = _clamp(55.0 - slip * 35.0)
    else:
        slip_g = None

    # Overall win rate
    win_rate = summary.get("win_rate_vs_spy")
    win_g = _pct_to_grade(win_rate, baseline=0.40, ceiling=0.65)

    # Overall avg alpha
    avg_alpha = summary.get("avg_realized_alpha")
    alpha_g = _lift_to_grade(avg_alpha, floor=-3.0, ceiling=5.0, units="pp")

    grade = _weighted_avg([
        (0.35, slip_g),
        (0.35, win_g),
        (0.30, alpha_g),
    ])

    detail = {}
    if slip is not None:
        detail["avg_slippage"] = f"{slip:+.2f}%"
    if win_rate is not None:
        detail["win_rate"] = f"{win_rate:.1%}"
    if avg_alpha is not None:
        detail["avg_alpha"] = _fmt_lift(avg_alpha, "pp")
    detail["n_triggers"] = len(triggers)
    detail["total_entries"] = summary.get("total_entries", 0)

    # Per-trigger mini-grades
    trigger_grades = []
    for t in triggers:
        t_slip = t.get("avg_slippage_vs_signal")
        t_win = t.get("win_rate_vs_spy")
        t_slip_g = _clamp(55.0 - t_slip * 35.0) if t_slip is not None else None
        t_win_g = _pct_to_grade(t_win, baseline=0.40, ceiling=0.65)
        t_grade = _weighted_avg([(0.5, t_slip_g), (0.5, t_win_g)])
        trigger_grades.append({
            "trigger": t.get("trigger"),
            "grade": t_grade,
            "letter": _letter(t_grade),
            "n_trades": t.get("n_trades", 0),
        })
    detail["per_trigger"] = trigger_grades

    return {"grade": grade, "letter": _letter(grade), "detail": detail}


def _grade_risk_guard(shadow_book: dict | None) -> dict:
    """Grade the risk guard's blocking decisions."""
    if not shadow_book or shadow_book.get("status") != "ok":
        return {"grade": None, "letter": "N/A",
                "reason": _na_reason(shadow_book, label="shadow book")}

    assessment = shadow_book.get("assessment", "neutral")
    guard_lift = shadow_book.get("guard_lift")
    n_blocked = shadow_book.get("n_blocked", 0)
    clf = shadow_book.get("classification")

    # Classification metrics: precision = % blocked that were actual losers
    precision = _safe_get(clf, "precision")
    recall = _safe_get(clf, "recall")
    f1 = _safe_get(clf, "f1")

    precision_g = _pct_to_grade(precision, baseline=0.40, ceiling=0.75) if precision is not None else None
    recall_g = _pct_to_grade(recall, baseline=0.05, ceiling=0.30) if recall is not None else None

    # Guard lift: positive = blocked entries were worse than traded (good)
    lift_g = _lift_to_grade(guard_lift, floor=-3.0, ceiling=3.0, units="fraction") if guard_lift is not None else None

    # Assessment mapping
    assessment_scores = {
        "appropriate": 80.0,
        "too_tight": 45.0,
        "too_loose": 35.0,
        "neutral": 55.0,
        "insufficient_return_data": None,
    }
    assess_g = assessment_scores.get(assessment)

    # Fallback: blocked_beat_spy_pct if no classification
    if precision_g is None:
        blocked_beat = shadow_book.get("blocked_beat_spy_pct")
        if blocked_beat is not None:
            precision_g = _clamp(95.0 - blocked_beat * 95.0)

    if recall_g is not None:
        grade = _weighted_avg([
            (0.30, precision_g),
            (0.20, recall_g),
            (0.25, lift_g),
            (0.25, assess_g),
        ])
    else:
        grade = _weighted_avg([
            (0.35, precision_g),
            (0.35, lift_g),
            (0.30, assess_g),
        ])

    detail = {
        "assessment": assessment,
        "n_blocked": n_blocked,
    }
    if precision is not None:
        detail["precision"] = f"{precision:.1%}"
    if recall is not None:
        detail["recall"] = f"{recall:.1%}"
    if f1 is not None:
        detail["f1"] = f"{f1:.3f}"
    if guard_lift is not None:
        detail["guard_lift"] = _fmt_lift(guard_lift, "fraction")

    return {"grade": grade, "letter": _letter(grade), "detail": detail}


def _grade_exit_rules(exit_timing: dict | None) -> dict:
    """Grade exit rule effectiveness."""
    if not exit_timing or exit_timing.get("status") != "ok":
        return {"grade": None, "letter": "N/A",
                "reason": _na_reason(exit_timing, label="exit timing")}

    summary = exit_timing.get("summary", {})
    diagnosis = exit_timing.get("diagnosis", "unknown")

    capture = summary.get("avg_capture_ratio")
    capture_g = _ratio_to_grade(capture, target=0.70)

    avg_return = summary.get("avg_realized_return")
    return_g = _lift_to_grade(avg_return, floor=-5.0, ceiling=5.0, units="pp") if avg_return is not None else None

    # Diagnosis bonus/penalty
    diag_scores = {
        "exits_well_timed": 85.0,
        "exits_could_improve": 55.0,
        "exits_too_early": 35.0,
    }
    diag_g = diag_scores.get(diagnosis)

    grade = _weighted_avg([
        (0.40, capture_g),
        (0.30, return_g),
        (0.30, diag_g),
    ])

    detail = {"diagnosis": diagnosis}
    if capture is not None:
        detail["capture_ratio"] = f"{capture:.2f}"
    if avg_return is not None:
        detail["avg_return"] = _fmt_lift(avg_return, "pp")
    detail["n_roundtrips"] = exit_timing.get("n_roundtrips", 0)

    return {"grade": grade, "letter": _letter(grade), "detail": detail}


def _grade_position_sizing(sizing_ab: dict | None) -> dict:
    """Grade position sizing vs equal-weight baseline."""
    if not sizing_ab or sizing_ab.get("status") != "ok":
        # ``backtest/{date}/sizing_ab.json`` has NEVER been written for any date
        # (verified 2026-08-13 across every card in
        # s3://alpha-engine-research/evaluator/); the backtester's weekly path
        # passes ``sizing_ab=None  # simulation-only``. This is a producer that
        # does not exist, not a thin sample — 10% of the declared executor
        # weight that has never once applied (config-I7202).
        return {
            "grade": None, "letter": "N/A",
            "reason": _na_reason(sizing_ab, label="sizing A/B"),
        }

    sharpe_diff = sizing_ab.get("sharpe_diff")
    alpha_diff = sizing_ab.get("alpha_diff")
    assessment = sizing_ab.get("assessment", "no_difference")

    # Sharpe improvement: 0 → 50, +0.3 → 80, +0.5 → 95
    sharpe_g = _lift_to_grade(sharpe_diff, floor=-0.3, ceiling=0.5, units="native") if sharpe_diff is not None else None

    # Alpha improvement
    alpha_g = _lift_to_grade(alpha_diff, floor=-2.0, ceiling=3.0, units="fraction") if alpha_diff is not None else None

    grade = _weighted_avg([
        (0.55, sharpe_g),
        (0.45, alpha_g),
    ])

    detail = {"assessment": assessment}
    if sharpe_diff is not None:
        detail["sharpe_diff"] = f"{sharpe_diff:+.3f}"
    if alpha_diff is not None:
        detail["alpha_diff"] = _fmt_lift(alpha_diff, "fraction")

    return {"grade": grade, "letter": _letter(grade), "detail": detail}


def _grade_portfolio(signal_quality: dict | None,
                     portfolio_stats: dict | None) -> dict:
    """Grade overall portfolio construction and performance.

    When ``portfolio_stats`` includes the evaluator-revamp downside-aware
    fields (``sortino_ratio``, ``cvar_95``, plus optionally an
    ``information_ratio_spy`` populated upstream), the composite uses:
      - 25% accuracy_21d  (selection accuracy, kept)
      - 25% Sortino       (replaces Sharpe — penalises downside vol only)
      - 15% Calmar        (annualised return / max drawdown)
      - 15% CVaR(95%)     (tail-risk metric)
      - 10% IR vs SPY     (only when supplied)
      - 10% max_drawdown
    Sharpe is still emitted in ``detail`` as a side-channel diagnostic
    but is intentionally dropped from the composite — it penalises the
    upside vol that a long-only risk-seeking strategy is *trying* to
    capture, which is the wrong shape for grading.

    Falls back to the legacy accuracy/alpha/Sharpe/DD weights when the
    new fields are absent — preserves backward compatibility for older
    portfolio_stats producers.
    """
    overall = _safe_get(signal_quality, "overall") or {}

    # config-I7208: the producer (crucible-backtester) never emitted a 10d
    # horizon on `overall` — live keys are accuracy_21d/avg_alpha_21d (and
    # the _5d pair). accuracy_10d/avg_alpha_10d read here always resolved to
    # None, so this component's alpha term silently dropped out of the
    # weighted average on every card ever produced. The fleet objective is
    # per-cycle net-of-cost 21d log-alpha vs SPY (crucible-evaluator-PR198),
    # so 21d is also the correct horizon, not just the only one available.
    acc_21d = overall.get("accuracy_21d")
    avg_alpha = overall.get("avg_alpha_21d")
    acc_g = _pct_to_grade(acc_21d, baseline=0.45, ceiling=0.70)
    # Anchor re-derivation (config-I7208): floor=-2.0/ceiling=4.0pp is left
    # UNCHANGED from the value this call site already carried. That is not
    # laziness — crucible-backtester/analysis/grading.py::_grade_portfolio is
    # the SOTA reference this module is a port of (policy-shared-code), it
    # grades the SAME avg_alpha_21d field, and it uses these exact anchors
    # (-2.0 floor / 4.0 ceiling) already. Live metrics.json history
    # (2026-07-10..2026-08-14, s3://alpha-engine-research/backtest/) shows
    # avg_alpha_21d ranging -0.75pp to +0.02pp, comfortably inside the band,
    # so no rescaling is warranted by the data either. What WAS wrong is the
    # ``units`` declaration: avg_alpha_21d is emitted already in percentage
    # points (confirmed by crucible-backtester's own `f"{avg_alpha:+.2f}%"`
    # formatting and by the live magnitudes above — a raw-fraction reading
    # would imply -75% 21d alpha, which is not a real number), not a raw
    # fraction. ``units="fraction"`` here was silently multiplying an
    # already-pp value by 100 the one time this call site is exercised with
    # real 21d data below the anchors' upper reach; ``units="pp"`` is correct.
    alpha_g = _lift_to_grade(avg_alpha, floor=-2.0, ceiling=4.0, units="pp") if avg_alpha is not None else None

    sharpe = _safe_get(portfolio_stats, "sharpe_ratio")
    sortino = _safe_get(portfolio_stats, "sortino_ratio")
    calmar = _safe_get(portfolio_stats, "calmar_ratio")
    cvar = _safe_get(portfolio_stats, "cvar_95")
    ir_spy = _safe_get(portfolio_stats, "information_ratio_spy")
    max_dd = _safe_get(portfolio_stats, "max_drawdown")

    # Legacy Sharpe → grade map kept for the fallback path + side-channel
    # display. Sharpe 0 → 30, 1.0 → 65, 2.0 → 95.
    sharpe_g = _clamp(30.0 + sharpe * 32.5) if sharpe is not None else None
    # Sortino: 0 → 30, 1.5 → 65, 3.0 → 95 (Sortino runs higher than Sharpe
    # because the denominator is smaller; calibrate the band accordingly).
    sortino_g = _clamp(30.0 + sortino * 21.67) if sortino is not None else None
    # Calmar: 0 → 30, 1.0 → 65, 3.0 → 95.
    calmar_g = _clamp(30.0 + calmar * 21.67) if calmar is not None else None
    cvar_g = _cvar_to_grade(cvar)
    ir_g = (
        _band_to_grade(ir_spy, floor=-1.0, mid=0.5, ceiling=2.0)
        if ir_spy is not None else None
    )
    # max_dd: -5% → 85, -10% → 65, -20% → 30, -30% → 10.
    dd_g = _clamp(95.0 + max_dd * 2.83) if max_dd is not None else None

    use_new_stack = sortino is not None and cvar is not None
    if use_new_stack:
        grade = _weighted_avg([
            (0.25, acc_g),
            (0.25, sortino_g),
            (0.15, calmar_g),
            (0.15, cvar_g),
            (0.10, ir_g),
            (0.10, dd_g),
        ])
    else:
        grade = _weighted_avg([
            (0.30, acc_g),
            (0.25, alpha_g),
            (0.25, sharpe_g),
            (0.20, dd_g),
        ])

    detail = {}
    if acc_21d is not None:
        detail["accuracy_21d"] = f"{acc_21d:.1%}"
    if avg_alpha is not None:
        detail["avg_alpha_21d"] = _fmt_lift(avg_alpha, "pp")
    elif overall:
        # Fail loud: signal_quality is present but avg_alpha_21d is absent —
        # name the reason instead of letting alpha_g vanish into the
        # weighted-average renormalization with no trace in the detail block.
        detail["avg_alpha_21d_reason"] = "no avg_alpha_21d in signal_quality.overall this cycle"
    if sharpe is not None:
        detail["sharpe"] = f"{sharpe:.2f}"
    if sortino is not None:
        detail["sortino"] = f"{sortino:.2f}"
    if calmar is not None:
        detail["calmar"] = f"{calmar:.2f}"
    if cvar is not None:
        detail["cvar_95"] = f"{cvar:.2%}"
    if ir_spy is not None:
        detail["information_ratio_spy"] = f"{ir_spy:.2f}"
    if max_dd is not None:
        detail["max_drawdown"] = f"{max_dd:.1%}"

    return {"grade": grade, "letter": _letter(grade), "detail": detail}


# ---------------------------------------------------------------------------
# New evaluator-revamp graders (calibration / action entropy / excursion)
# ---------------------------------------------------------------------------


def _grade_calibration_diagnostics(calibration: dict | None) -> dict:
    """Grade conviction-vs-realized calibration (reliability diagram quality).

    Consumes the output of ``analysis.calibration_diagnostics.compute_calibration``.
    Grade is driven by ECE: lower = better calibration. Bands match
    the existing production_health.compute_calibration_validation labels:
      - ECE < 0.05 → "good" → 90
      - ECE < 0.10 → "acceptable" → 65
      - ECE < 0.20 → "poor" → 35
      - ECE ≥ 0.20 → 10
    """
    if not calibration or calibration.get("status") not in ("ok",):
        return {
            "grade": None, "letter": "N/A",
            "reason": calibration.get("reason") if calibration else "no data",
        }

    ece = calibration.get("ece")
    if ece is None:
        return {"grade": None, "letter": "N/A", "reason": "ece missing"}

    if ece < 0.05:
        grade = 90.0
    elif ece < 0.10:
        grade = 65.0
    elif ece < 0.20:
        grade = 35.0
    else:
        grade = 10.0

    detail: dict[str, Any] = {
        "ece": round(ece, 4),
        "n": calibration.get("n"),
        "quality": calibration.get("quality"),
    }
    if calibration.get("brier_score") is not None:
        detail["brier_score"] = calibration["brier_score"]
    if calibration.get("bins"):
        detail["n_bins"] = len(calibration["bins"])

    return {"grade": grade, "letter": _letter(grade), "detail": detail}


def _grade_action_entropy(action_entropy: dict | None) -> dict:
    """Grade action-stream Shannon entropy (BUY/HOLD/SELL distribution).

    Consumes the output of ``analysis.action_entropy.compute_action_entropy``.
    Catches degenerate-LLM-behavior failure modes (always-hold,
    always-trade) that risk-adjusted return metrics don't see. Grade
    is driven by ``entropy_normalized`` (in [0, 1]):
      - 1.0 → 100 (perfectly uniform)
      - alarm threshold (0.3 default) → 40 (concerning)
      - 0.0 → 0 (single-action collapse)
    The function honours the alarm flag emitted by the producer.
    """
    if not action_entropy or action_entropy.get("status") != "ok":
        return {
            "grade": None, "letter": "N/A",
            "reason": _na_reason(action_entropy, label="action entropy"),
        }

    h_norm = action_entropy.get("entropy_normalized")
    grade = _band_to_grade(h_norm, floor=0.0, mid=0.3, ceiling=1.0)
    if grade is not None and h_norm is not None and h_norm < 0.3:
        # Pull below 40 explicitly when the alarm floor is breached
        # (band floor=0.0/mid=0.3 already does this; this is a
        # belt-and-suspenders check).
        grade = min(grade, 40.0)

    detail: dict[str, Any] = {}
    if h_norm is not None:
        detail["entropy_normalized"] = round(float(h_norm), 3)
    if action_entropy.get("most_common") is not None:
        detail["most_common"] = action_entropy["most_common"]
    if action_entropy.get("most_common_fraction") is not None:
        detail["most_common_fraction"] = f"{action_entropy['most_common_fraction']:.1%}"
    if action_entropy.get("alarm") is not None:
        detail["alarm"] = bool(action_entropy["alarm"])
    if action_entropy.get("n") is not None:
        detail["n"] = action_entropy["n"]

    return {"grade": grade, "letter": _letter(grade), "detail": detail}


def _grade_excursion(excursion_summary: dict | None) -> dict:
    """Grade per-trade MFE/MAE process quality.

    Consumes the output of ``analysis.excursion.summarize_excursions``.
    Composite over two indicators:
      - mean_mfe_mae_ratio (60%) — band 0.8 → 1.5 → 2.0
      - pct_high_quality (40%) — fraction of trades with ratio > 1.5;
        banded 0 → 0.3 → 0.6 → 80
    """
    if not excursion_summary or excursion_summary.get("status") != "ok":
        # The FOURTH chronically-null component, which config-I7202 did not name
        # because the card never showed it: when ``excursion_summary`` is None
        # the key is omitted from ``executor.components`` entirely, so its 15%
        # declared weight vanished without leaving even a null behind.
        # ``backtest/{date}/portfolio_excursion.json`` is absent on every date
        # checked (2026-05-29 .. 2026-08-07).
        return {
            "grade": None, "letter": "N/A",
            "reason": _na_reason(excursion_summary, label="portfolio excursion"),
        }

    ratio = excursion_summary.get("mean_mfe_mae_ratio")
    pct = excursion_summary.get("pct_high_quality")

    ratio_g = _band_to_grade(ratio, floor=0.8, mid=1.5, ceiling=2.0)
    pct_g = _pct_to_grade(pct, baseline=0.30, ceiling=0.60)

    grade = _weighted_avg([(0.60, ratio_g), (0.40, pct_g)])

    detail: dict[str, Any] = {}
    if ratio is not None:
        detail["mean_mfe_mae_ratio"] = round(ratio, 3)
    if excursion_summary.get("median_mfe_mae_ratio") is not None:
        detail["median_mfe_mae_ratio"] = round(
            excursion_summary["median_mfe_mae_ratio"], 3,
        )
    if pct is not None:
        detail["pct_high_quality"] = f"{pct:.1%}"
    if excursion_summary.get("pct_mfe_gt_mae") is not None:
        detail["pct_mfe_gt_mae"] = f"{excursion_summary['pct_mfe_gt_mae']:.1%}"
    if excursion_summary.get("n") is not None:
        detail["n"] = excursion_summary["n"]

    return {"grade": grade, "letter": _letter(grade), "detail": detail}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_scorecard(
    signal_quality: dict | None = None,
    e2e_lift: dict | None = None,
    macro_eval: dict | None = None,
    score_calibration: dict | None = None,
    veto_result: dict | None = None,
    veto_value: dict | None = None,
    trigger_scorecard: dict | None = None,
    shadow_book: dict | None = None,
    exit_timing: dict | None = None,
    sizing_ab: dict | None = None,
    predictor_sizing: dict | None = None,
    portfolio_stats: dict | None = None,
    scanner_opt: dict | None = None,
    cio_opt: dict | None = None,
    *,
    team_metrics: dict | None = None,
    calibration_diagnostics: dict | None = None,
    action_entropy: dict | None = None,
    excursion_summary: dict | None = None,
) -> dict:
    """Compute the unified system scorecard.

    Returns a dict with:
        status: "ok" | "partial" | "insufficient_data"
        overall: {grade, letter}
        research: {grade, letter, components: {...}}
        predictor: {grade, letter, components: {...}}
        executor: {grade, letter, components: {...}}
    """
    # -----------------------------------------------------------------------
    # Research components
    # -----------------------------------------------------------------------
    scanner = _grade_scanner(e2e_lift, scanner_opt)
    macro = _grade_macro(macro_eval)
    cio = _grade_cio(e2e_lift, cio_opt)
    composite = _grade_composite_scoring(signal_quality, score_calibration)

    # Sector teams
    # team_lift is contractually a list[dict] (see end_to_end._team_lift).
    # Defensive isinstance check here guards against producer regressions
    # where a status dict leaks through — iterating a dict yields its keys
    # (strings), which crashes _grade_sector_team.get() with AttributeError.
    # That's exactly what happened on 2026-04-11.
    team_lift_list = _safe_get(e2e_lift, "team_lift") or []
    if not isinstance(team_lift_list, list):
        team_lift_list = []
    teams = [_grade_sector_team(t, team_metrics=team_metrics) for t in team_lift_list]

    # Average team grade
    team_grades = [t["grade"] for t in teams if t.get("grade") is not None]
    avg_team_grade = sum(team_grades) / len(team_grades) if team_grades else None

    # New: decision-quality grade — calibration of agent conviction vs realized.
    calibration_grade = _grade_calibration_diagnostics(calibration_diagnostics)

    # Recompose research with calibration when available; preserves
    # existing weights when calibration is absent (calibration_grade.grade
    # is None → _weighted_avg drops it from the average).
    research_grade = _weighted_avg([
        (RESEARCH_WEIGHTS["scanner"], scanner.get("grade")),
        (RESEARCH_WEIGHTS["sector_teams_avg"], avg_team_grade),
        (RESEARCH_WEIGHTS["macro_agent"], macro.get("grade")),
        (RESEARCH_WEIGHTS["cio"], cio.get("grade")),
        (RESEARCH_WEIGHTS["composite_scoring"], composite.get("grade")),
        (RESEARCH_WEIGHTS["calibration_diagnostics"], calibration_grade.get("grade")),
    ])

    sector_teams_avg = {"grade": avg_team_grade, "letter": _letter(avg_team_grade)}
    if avg_team_grade is None:
        # An empty/ungradable team list previously emitted a bare null with no
        # reason, which reads identically to "the graders all failed". The six-
        # team research graph was RETIRED 2026-07-12 (config#1580 /
        # config-I2993) and e2e_lift.team_lift has been `[]` on every card
        # since — a permanently-absent component carrying 25% of the declared
        # research weight (config-I7202).
        retired = _safe_get(e2e_lift, "research_graph_retired")
        if isinstance(retired, dict) and retired:
            sector_teams_avg["reason"] = (
                "sector teams status: retired"
                + (f" ({retired.get('retired_date')})" if retired.get("retired_date") else "")
                + (f" — {retired['reason']}" if retired.get("reason") else "")
            )
        else:
            sector_teams_avg["reason"] = (
                f"no gradable sector team ({len(teams)} team blocks, "
                f"{len(team_grades)} with a grade)"
            )

    research_components = {
        "scanner": scanner,
        "sector_teams": teams,
        "sector_teams_avg": sector_teams_avg,
        "macro_agent": macro,
        "cio": cio,
        "composite_scoring": composite,
    }
    if calibration_diagnostics is not None:
        research_components["calibration_diagnostics"] = calibration_grade

    research = {
        "grade": research_grade,
        "letter": _letter(research_grade),
        "components": research_components,
    }

    # -----------------------------------------------------------------------
    # Predictor components
    # -----------------------------------------------------------------------
    meta = _grade_meta_model(predictor_sizing, veto_result)
    veto = _grade_veto_gate(veto_result, veto_value)

    predictor_grade = _weighted_avg([
        (PREDICTOR_WEIGHTS["meta_model"], meta.get("grade")),
        (PREDICTOR_WEIGHTS["veto_gate"], veto.get("grade")),
    ])

    predictor = {
        "grade": predictor_grade,
        "letter": _letter(predictor_grade),
        "components": {
            "meta_model": meta,
            "veto_gate": veto,
        },
    }

    # -----------------------------------------------------------------------
    # Executor components
    # -----------------------------------------------------------------------
    triggers = _grade_entry_triggers(trigger_scorecard)
    guard = _grade_risk_guard(shadow_book)
    exits = _grade_exit_rules(exit_timing)
    sizing = _grade_position_sizing(sizing_ab)
    portfolio = _grade_portfolio(signal_quality, portfolio_stats)
    # New: process-quality graders.
    excursion_grade = _grade_excursion(excursion_summary)
    entropy_grade = _grade_action_entropy(action_entropy)

    executor_grade = _weighted_avg([
        (EXECUTOR_WEIGHTS["entry_triggers"], triggers.get("grade")),
        (EXECUTOR_WEIGHTS["risk_guard"], guard.get("grade")),
        (EXECUTOR_WEIGHTS["exit_rules"], exits.get("grade")),
        (EXECUTOR_WEIGHTS["position_sizing"], sizing.get("grade")),
        (EXECUTOR_WEIGHTS["portfolio"], portfolio.get("grade")),
        (EXECUTOR_WEIGHTS["excursion"], excursion_grade.get("grade")),
        (EXECUTOR_WEIGHTS["action_entropy"], entropy_grade.get("grade")),
    ])

    executor_components = {
        "entry_triggers": triggers,
        "risk_guard": guard,
        "exit_rules": exits,
        "position_sizing": sizing,
        "portfolio": portfolio,
    }
    if excursion_summary is not None:
        executor_components["excursion"] = excursion_grade
    if action_entropy is not None:
        executor_components["action_entropy"] = entropy_grade

    executor = {
        "grade": executor_grade,
        "letter": _letter(executor_grade),
        "components": executor_components,
    }

    # -----------------------------------------------------------------------
    # Overall
    # -----------------------------------------------------------------------
    overall_grade = _weighted_avg([
        (OVERALL_WEIGHTS["research"], research_grade),
        (OVERALL_WEIGHTS["predictor"], predictor_grade),
        (OVERALL_WEIGHTS["executor"], executor_grade),
    ])

    # Determine status
    graded_count = sum(1 for g in [research_grade, predictor_grade, executor_grade] if g is not None)
    if graded_count == 0:
        status = "insufficient_data"
    elif graded_count < 3:
        status = "partial"
    else:
        status = "ok"

    card = {
        "status": status,
        "overall": {"grade": overall_grade, "letter": _letter(overall_grade)},
        "research": research,
        "predictor": predictor,
        "executor": executor,
    }

    # -----------------------------------------------------------------------
    # Coverage (config-I7202) — additive, and it cannot fail the run.
    #
    # Everything above this line is the arithmetic that has always produced the
    # grade; nothing below changes a single grade value. The coverage block is
    # computed separately and guarded, so a defect in this reporting code
    # degrades the guarantee rather than failing the report-card stage three
    # days before a scheduled weekly run.
    # -----------------------------------------------------------------------
    try:
        research_cov = _coverage(
            RESEARCH_WEIGHTS,
            {
                "scanner": scanner,
                "sector_teams_avg": sector_teams_avg,
                "macro_agent": macro,
                "cio": cio,
                "composite_scoring": composite,
                "calibration_diagnostics": calibration_grade,
            },
            floor=DEFAULT_COVERAGE_FLOOR,
        )
        predictor_cov = _coverage(
            PREDICTOR_WEIGHTS,
            {"meta_model": meta, "veto_gate": veto},
            floor=DEFAULT_COVERAGE_FLOOR,
        )
        executor_cov = _coverage(
            EXECUTOR_WEIGHTS,
            {
                "entry_triggers": triggers,
                "risk_guard": guard,
                "exit_rules": exits,
                "position_sizing": sizing,
                "portfolio": portfolio,
                "excursion": excursion_grade,
                "action_entropy": entropy_grade,
            },
            floor=DEFAULT_COVERAGE_FLOOR,
        )
        overall_cov = _coverage(
            OVERALL_WEIGHTS,
            {
                "research": {"grade": research_grade},
                "predictor": {"grade": predictor_grade},
                "executor": {"grade": executor_grade},
            },
            effective={
                "research": research_cov["weight_present_effective"],
                "predictor": predictor_cov["weight_present_effective"],
                "executor": executor_cov["weight_present_effective"],
            },
            effective_failed={
                "research": research_cov["weight_failed"],
                "predictor": predictor_cov["weight_failed"],
                "executor": executor_cov["weight_failed"],
            },
            effective_failed_members={
                "research": research_cov["components_failed"],
                "predictor": predictor_cov["components_failed"],
                "executor": executor_cov["components_failed"],
            },
            floor=DEFAULT_COVERAGE_FLOOR,
        )
        for block, cov in (
            (card["overall"], overall_cov),
            (research, research_cov),
            (predictor, predictor_cov),
            (executor, executor_cov),
        ):
            block["coverage"] = cov
            block["display"] = _display(block["letter"], cov)
        card["grading_weights"] = {
            "version": WEIGHT_TABLE_VERSION,
            "rule": RENORMALIZATION_RULE,
            "overall": dict(OVERALL_WEIGHTS),
            "research": dict(RESEARCH_WEIGHTS),
            "predictor": dict(PREDICTOR_WEIGHTS),
            "executor": dict(EXECUTOR_WEIGHTS),
        }
    except Exception:  # noqa: BLE001 — see COVERAGE_UNKNOWN_MARKER rationale
        logger.exception(
            "%s: coverage computation raised; the grade is unaffected and is "
            "emitted with coverage UNKNOWN", COVERAGE_UNKNOWN_MARKER,
        )
        for block in (card["overall"], research, predictor, executor):
            block["coverage"] = dict(_COVERAGE_UNKNOWN)
            block["display"] = _display(block.get("letter", "N/A"), None)
        card["grading_weights"] = {
            "version": WEIGHT_TABLE_VERSION,
            "rule": RENORMALIZATION_RULE,
            "error": COVERAGE_UNKNOWN_MARKER,
        }

    return card
