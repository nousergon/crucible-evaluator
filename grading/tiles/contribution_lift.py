"""
contribution_lift.py — Tile 10: per-component marginal contribution to the
single T2 objective (RC v3 T5, ``alpha-engine-config-I7473``).

Reads ``backtest/{run_date}/contribution_lift.json`` — a single artifact
persisted by crucible-backtester's replay harness (``analysis/contribution_lift/``,
registered as ``tracker.run_module("contribution_lift", ...)`` in
``evaluate.py``) that carries a ``components[]`` list, one entry per
counterfactual replay: the mean, over paired cycles, of
``baseline_alpha − ablated_alpha`` (net-of-cost 21d log-alpha vs SPY,
``nousergon_lib.quant.horizons.DEFAULT_POLICY.primary_horizon``), where the
"ablated" arm has that ONE component's contribution removed (substitution
baseline or null arm — the artifact's ``pattern`` field says which).

Full contract: ``alpha-engine-docs/private/report-card-v3-objective-and-
attribution-260816.md`` §3 and the T5 artifact contract these tiles were
built against (contribution_lift_contract.md, RC v3 T5).

DESIGN — one new tile file, not four edited ones (config-I7473 T5 consumer
task named both options; this is "the former"): each emitted
``MetricRecord`` carries ``module=<owning tile>`` (research / predictor /
executor / behavioral — the component's REAL home, taken from the artifact's
own ``module`` field) so a rendering surface that groups by a record's own
``.module`` places it beside the component it measures. The tile-dict KEY
this module registers under in ``grading/aggregate.py`` is a 10th, separate
``"contribution_lift"`` entry (``build_tile("contribution_lift", ...)``) —
editing ``build_research_tile`` / ``build_predictor_tile`` /
``build_executor_tile`` / ``build_behavioral_tile`` directly was rejected:
those four files are mid-flight in a concurrent PR (rc-v3/i7485-units) this
session must not collide with, and — more durably — a component's raw metric
and its contribution-lift replay are graded on different cadences (weekly
report card vs a Saturday-only, currently-unbuilt T5 replay harness) and by
different producers, so folding them into the same builder function would
tie two independently-evolving artifacts to one read path. ``aggregate.py``
did not make the alternative impossible; this is a legibility/blast-radius
choice, documented per the task's ``state why``.

Every component named across the seven T5 replay-producer issues
(alpha-engine-config-I7478–I7484) has a ``registry.yaml`` row already, keyed
``(module, f"{name}_contribution_lift")`` — see the block comment in
``grading/thresholds/registry.yaml``. None of those seven producers has
shipped yet (they are the OTHER halves of this epic), so today every real-
world run reads a MISSING artifact and every registered component grades
N/A-MISSING-INPUT, naming the missing key — this module's job is the
consumer contract + the wiring, not the replay math.
"""

from __future__ import annotations

from grading.artifacts import BACKTESTER_ARTIFACT_MAX_AGE_DAYS, artifact_is_stale, get_json_windowed
from grading.metric_record import build_metric
from grading.module_agg import build_tile

TILE_KEY = "contribution_lift"

# unit is fixed by the T2 objective (spec §3): every contribution_lift record
# measures net-of-cost 21d log-alpha vs SPY, regardless of which component it
# measures.
_UNIT = "log_alpha_21d"
_ESTIMATOR = "paired_cycle_bootstrap"
_MEASUREMENT_HORIZON = "21d"
_N_FLOOR = 60  # spec §3 — fixed, not read from the artifact.

# Every component named across alpha-engine-config-I7478..I7484 (the seven T5
# replay-producer issues), and the tile it belongs to. This is the fallback
# enumeration used ONLY when the artifact itself is absent/unusable — when the
# artifact is present, the components it actually carries drive the loop
# (whatever a producer emits, graded or fail-loud on a missing registry row);
# this dict never gates what a live artifact may report.
#
# Landing a new producer (one of I7478-I7484) needs exactly two edits: add its
# component(s) here (so a missing artifact still names it), and add its
# registry row(s) in grading/thresholds/registry.yaml under the matching
# module section. Nothing else in this file changes.
KNOWN_COMPONENTS: dict[str, str] = {
    # I7478 — research composite/selection group.
    "research_composite_ic": "research",
    "sector_teams_avg": "research",
    "cio_selection_skill": "research",
    "neutralization_live_efficacy": "research",
    "scanner_feed_counterfactual": "research",
    # I7483 — research diagnostics group.
    "thinktank_coverage_ic": "research",
    "macro_agent": "research",
    "calibration_diagnostics": "research",
    "momentum_regime_ic": "research",
    "attractiveness_ic": "research",
    "attractiveness_trajectory_ic": "research",
    "judge_outcome_ic": "research",
    "judge_rubric_pass_rate": "research",
    # I7479 — predictor L1/L2 group.
    "meta_l2_ic": "predictor",
    "momentum_l1_ic": "predictor",
    "volatility_l1_ic": "predictor",
    "research_calibrator_l1_ic": "predictor",
    "ensemble_lift_over_best_l1": "predictor",
    # I7480 — predictor gates group.
    "veto_gate_precision": "predictor",
    "output_distribution_gate": "predictor",
    "direction_accuracy_vs_majority_baseline": "predictor",
    # I7481 — executor rules group.
    "exit_rules": "executor",
    "position_sizing": "executor",
    "entry_triggers": "executor",
    # I7482 — executor risk_guard.
    "risk_guard": "executor",
    # I7484 — behavioral cost_adjusted_quality.
    "cost_adjusted_quality": "behavioral",
}

# Statuses the producer may report verbatim in its ``components[].status``
# field (contribution_lift_contract.md) that the shared ``StatusLiteral``
# (krepis.metrics) does not itself carry — RC v2's taxonomy has no "retired"
# or "not lift-shaped" code, and "gap" is this artifact's own escape hatch for
# a cycle that ran but produced no usable paired measurement (e.g. a width
# mismatch between arms). Mapped to the closest existing StatusLiteral, the
# producer's ``status_reason`` carried through VERBATIM (prefixed with the
# producer's own status string so the remap is never silently lossy).
_STATUS_REMAP: dict[str, str] = {
    "gap": "N/A-MISSING-INPUT",
}

# Producer statuses that are DECLARED TERMINAL STATES, not gaps
# (alpha-engine-config-I8177).
#
# These two were previously in ``_STATUS_REMAP`` above, collapsed onto a bare
# ``N/A-NOT-IMPL`` — "declared and never built". That inverted their meaning.
# ``N/A-NOT-IMPL`` is an open gap someone is expected to close; these are the
# producer stating, with a reason, that the component will never carry a lift
# number. Nine components on the 2026-08-22 card sat in that state: two
# retired with the six-team/CIO research graph, seven structurally not
# lift-shaped (an in-tile lift, an observe-only shadow score, a pass/fail gate
# over a whole output distribution, or observability OF the judge rather than a
# steering input). Each was an assertion that could never pass, permanently
# depressing ``evaluator_coverage``.
#
# Routing them through ``permanent_na_reason`` keeps the letter at N/A and the
# producer's own reason verbatim, but stamps ``permanent_na: true`` on the
# record — which is what ``grading/coverage.py`` reads to take a DECLARED
# component out of the coverage denominator. The declaration stays visible on
# the card; only its miscounting as an open gap goes away.
#
# ``observability-policy.md`` §8.3: RETIRED is DECLARED, never inferred — and,
# symmetrically, a declared terminal state must not render as an undeclared gap.
_STATUS_DECLARED_TERMINAL = frozenset({"N/A-RETIRED", "N/A-NOT-LIFT-SHAPED"})
# Passed straight through — already valid StatusLiteral values.
_STATUS_PASSTHROUGH = {"N/A-MISSING-INPUT", "N/A-LOW-N"}

# EVALUATOR-declared permanent N/A (alpha-engine-config-I12584, option 1
# "Retire", taken at the 2026-10-25 decision-queue triage).
#
# Everything above maps a state the PRODUCER declared. These two are declared
# here instead, on the consumer side, and that is deliberate: the established
# pattern (I8177) is that ``contribution_lift.json`` declares a terminal state
# itself, but Sat 2026-10-24 was the last v1 weekly, so no v1 backtester run
# will ever write one. A producer-side declaration would never take effect.
#
# This is a DECLARATION, not an inference. Both lifts read
# ``N/A-MISSING-INPUT`` on the 10-09, 10-16 and 10-23 cards with the same shape,
# because #12243's support assurance (crucible-backtester#810) refuses to
# compare arms whose support differs, and these arms differ BY CONSTRUCTION:
#
# * ``entry_triggers`` decides WHEN to enter, so the ablated arm enters on
#   different dates (10-23: 25 baseline-only and 11 ablated-only dates of 60).
# * ``risk_guard``'s main effect is the orders it blocks (204 baseline vs 902
#   ablated), so the ablated arm is wider on purpose.
#
# Matching the cohorts would grade a residual that excludes what each
# component does, which is a new estimand (reserved by #12243 to estimator
# authority), not a repair. So the reason says "by construction", never
# "missing input".
#
# What the card loses: it no longer states a net-P&L contribution for either
# component. What remains: the entry_triggers trigger scorecard (#12387) and
# the risk_guard shadow-book precision/recall in the executor tile.
#
# Revisit triggers, either one:
# * EVENT: the producer emits a VALUE for one of these components (the arms
#   shared support, so the by-construction reason no longer holds). The
#   declaration then does not apply and the record grades normally; see
#   ``_evaluator_declared_reason``.
# * EVENT: v2 ships an ablation harness, or a matched-cohort estimand is ruled
#   under estimator authority. Delete the entry here when it does.
# * DATED BACKSTOP: ``_EVALUATOR_DECLARED_REVISIT_BY``. From that run_date on,
#   the declaration LAPSES on its own: both records return to the producer's
#   state, re-enter the coverage denominator, and the digest flags them again.
#   Renewing it is an edit here with a fresh date and a reason, so a
#   retirement nobody re-examined cannot outlive its backstop silently.
#
# ``position_sizing`` is NOT here: it is a few dates from matchable, and it was
# not part of the ruling.
_EVALUATOR_DECLARED_REVISIT_BY = "2026-11-30"
_KNOWN_PRODUCER_STATUSES = frozenset(
    {None, "ok"} | _STATUS_PASSTHROUGH | _STATUS_DECLARED_TERMINAL | set(_STATUS_REMAP)
)
_EVALUATOR_DECLARED_PERMANENT_NA: dict[str, str] = {
    "entry_triggers": (
        "N/A-NOT-MEASURABLE-BY-CONSTRUCTION: the ablated arm (no entry triggers) "
        "enters on different dates, so the arms never share support and the "
        "#12243 support assurance refuses the paired comparison. Matching the "
        "cohorts by date would remove the timing effect the component exists "
        "to have. The card no longer states a net-P&L contribution for "
        "entry_triggers; its signal is the trigger scorecard (#12387). Declared "
        "evaluator-side (alpha-engine-config-I12584, Retire). Revisit when v2 "
        "ships an ablation harness, or by "
    ),
    "risk_guard": (
        "N/A-NOT-MEASURABLE-BY-CONSTRUCTION: removing the guard admits about "
        "4.4x the orders (204 vs 902), so the ablated arm is wider on purpose "
        "and the #12243 support assurance refuses the paired comparison. "
        "Capping both arms to a common width would drop the orders the guard "
        "blocks, which is what it does. The card no longer states a net-P&L "
        "contribution for risk_guard; its signal is the shadow-book "
        "precision/recall. Declared evaluator-side (alpha-engine-config-I12584, "
        "Retire). Revisit when v2 ships an ablation harness, or by "
    ),
}


def _evaluator_declared_reason(name: str, run_date: str, value=None) -> str | None:
    """The written retirement reason for ``name`` on ``run_date``, or None.

    None when ``name`` is not declared, when the producer measured a value
    (the arms shared support: the revisit event fired), or when ``run_date``
    has reached the dated backstop (the declaration lapsed).
    """
    reason = _EVALUATOR_DECLARED_PERMANENT_NA.get(name)
    if reason is None or value is not None:
        return None
    if run_date >= _EVALUATOR_DECLARED_REVISIT_BY:
        return None
    return f"{reason}{_EVALUATOR_DECLARED_REVISIT_BY}."


def _missing_input_detail(name: str, key: str, reason: str | None = None) -> str:
    detail = f"{name}_contribution_lift: contribution_lift.json absent or unusable this cycle (key: {key})."
    if reason:
        detail += f" Producer reason: {reason}"
    return detail


def build_contribution_lift_tile(bucket: str, run_date: str, s3_client=None) -> dict:
    """Build the Contribution-Lift tile from ``backtest/{run_date}/contribution_lift.json``.

    Every ``components[]`` entry in the artifact becomes ONE
    ``f"{name}_contribution_lift"`` MetricRecord, ``metric_type="contribution_lift"``,
    ``unit="log_alpha_21d"``, ``red_line=0.0`` and ``target`` resolved from
    ``registry.yaml`` under ``(module, f"{name}_contribution_lift")`` — same
    chokepoint every other tile uses (``build_metric``'s default
    ``band="champion"`` path), so red_line/target never need to be threaded
    explicitly here.
    """
    import boto3

    s3 = s3_client or boto3.client("s3")
    key_template = "backtest/{date}/contribution_lift.json"
    doc, _src_date, age_days, key = get_json_windowed(s3, bucket, key_template, run_date)
    src = f"s3://{bucket}/{key}" if key else f"s3://{bucket}/backtest/{run_date}/contribution_lift.json"
    stale = artifact_is_stale(age_days, BACKTESTER_ARTIFACT_MAX_AGE_DAYS)

    components = []

    if doc is None or doc.get("status") != "ok":
        top_reason = (doc or {}).get("reason")
        for name, module in KNOWN_COMPONENTS.items():
            declared = _evaluator_declared_reason(name, run_date)
            if declared is not None:
                # Retired by construction (I12584): an absent artifact does
                # not make it measurable, so it stays declared out.
                components.append(build_metric(
                    name=f"{name}_contribution_lift", module=module, metric_type="contribution_lift",
                    unit=_UNIT, n_floor=_N_FLOOR, source_path=src,
                    criticality="diagnostic", permanent_na_reason=declared,
                ))
                continue
            components.append(build_metric(
                name=f"{name}_contribution_lift", module=module, metric_type="contribution_lift",
                unit=_UNIT, n_floor=_N_FLOOR, source_path=src,
                criticality="diagnostic",  # no producer yet ships any of these (I7478-I7484 open).
                input_present=False,
                na_detail=_missing_input_detail(
                    name, key or f"backtest/{run_date}/contribution_lift.json", top_reason,
                ),
            ))
        return build_tile(TILE_KEY, components, staleness={
            "stale_artifact_count": 1 if stale else 0,
            "max_artifact_age_days": age_days,
            "any_stale": stale,
        })

    for comp in doc.get("components") or []:
        name = comp["name"]
        module = comp["module"]
        criticality = comp["criticality"]
        producer_status = comp.get("status")
        status_reason = comp.get("status_reason")
        comp_src = comp.get("source_path") or src

        kwargs = dict(
            name=f"{name}_contribution_lift", module=module, metric_type="contribution_lift",
            unit=_UNIT, n_floor=_N_FLOOR, criticality=criticality, source_path=comp_src,
            value=comp.get("value"), ci_low=comp.get("ci_low"), ci_high=comp.get("ci_high"),
            ci_method=comp.get("ci_method"), n_samples=comp.get("n_samples"),
        )
        if criticality == "critical" and comp.get("value") is not None:
            kwargs["estimator"] = _ESTIMATOR
            kwargs["measurement_horizon"] = _MEASUREMENT_HORIZON

        declared = (
            _evaluator_declared_reason(name, run_date, comp.get("value"))
            if producer_status in _KNOWN_PRODUCER_STATUSES else None
        )  # an unrecognized status still fails loud below, declared or not.
        if declared is not None:
            # Evaluator-declared retirement (I12584) wins over the producer's
            # non-value state. The producer's own state rides along verbatim
            # so the override is never silently lossy.
            if producer_status not in (None, "ok"):
                declared += f" Producer reported {producer_status}: {status_reason}"
            kwargs["permanent_na_reason"] = declared
        elif producer_status is None or producer_status == "ok":
            pass  # let build_metric/derive_status compute GREEN/WATCH/RED/N/A-LOW-N.
        elif producer_status in _STATUS_PASSTHROUGH:
            kwargs["status"] = producer_status
            kwargs["reason"] = status_reason
        elif producer_status in _STATUS_DECLARED_TERMINAL:
            # Declared terminal state — carried as an accepted permanent N/A so
            # it leaves the coverage denominator instead of reading as an open
            # gap. build_metric derives N/A-NOT-IMPL + permanent_na=True from
            # permanent_na_reason; the producer's own reason rides through
            # verbatim, prefixed with its status so the mapping is not lossy.
            kwargs["permanent_na_reason"] = f"{producer_status}: {status_reason}"
        elif producer_status in _STATUS_REMAP:
            kwargs["status"] = _STATUS_REMAP[producer_status]
            kwargs["reason"] = f"{producer_status}: {status_reason}"
        else:
            raise ValueError(
                f"contribution_lift artifact reports unrecognized status "
                f"{producer_status!r} for component {name!r} — contract violation "
                f"(contribution_lift_contract.md), fail loud rather than grade on an "
                f"unmapped state."
            )

        components.append(build_metric(**kwargs))

    return build_tile(TILE_KEY, components, staleness={
        "stale_artifact_count": 1 if stale else 0,
        "max_artifact_age_days": age_days,
        "any_stale": stale,
    })
