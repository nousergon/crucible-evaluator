"""
director_quality.py — Tile 9: Director Retro Quality (RC v2, config#1674).

Grades the Director's own weekly Phase-G retro self-assessment — the
grounding / calibration / actionability scores the retro judge assigns to the
PRIOR week's action plan once THIS week's realized outcome (the fresh Report
Card) is available. See ``director/retro.py`` (the judge call) and
``director/handler.py::_persist_retro`` (the persistence). This tile exists so
that self-grade, which today is computed and written to
``director/retro_trend.json`` but consumed by nothing (not the report card,
not the dashboard), is finally surfaced.

Source artifact: ``director/retro_trend.json`` —

    {"updated": <run_date>, "grades": [
        {"prior_run_date": ..., "retro_run_date": ..., "grounding": <0-100>,
         "calibration": <0-100>, "actionability": <0-100>, "notes": ...,
         "judge_model": ..., "resolved_model": ...},   # judge_model/resolved_model
        ...                                             # land with config#1673;
    ]}                                                  # may be absent on older rows.

upserted + sorted ascending by ``prior_run_date`` on every Director run
(``director/handler.py`` ``_persist_retro``, ~lines 214-244). This tile reads
``grades[-1]`` — the most recently upserted (= most recent ``prior_run_date``)
entry.

STALENESS IS CORRECT BY CONSTRUCTION — do not "fix" it later: the ReportCard
Step-Functions state runs BEFORE the Director state in the Saturday chain (the
Director's own weekly retro grades what happened after ITS prior plan, using
the report card the Director state itself just consumed as input earlier in
the same chain). So on any given Saturday this tile necessarily shows the
grade of the *previous* completed retro cycle, one cycle behind the freshest
report card it's embedded in — the same cross-state lag every S3-handoff tile
in this repo lives with. This is the documented cross-repo invariant (see
alpha-engine-config's ``private-docs/system_state/cross_repo_invariants.md``,
"ReportCard-before-Director" ordering) — not a bug to chase.

A fourth component, ``director_route_degraded``, is not a retro grade at all:
it reads ``director/latest/action_plan.json`` and reports whether a FALLBACK
model produced the Director's most recent plan, naming the model that served
(alpha-engine-config-I8165). It lives in this tile because it is the same
question the retro grades answer — how much to trust that plan — asked about
the substrate rather than the content. See ``_route_degraded_component``.

All four components are ``criticality="supporting"``: a single LLM-judge call
grading a single prior plan is exactly the kind of low-N, single-rater
estimate the L4562 critical-metric contract (``metric_record.py`` ~lines
163-175) exists to keep off the critical path — it must never force a red
overall via ``module_status``'s critical-gate rule, and (per ``module_agg.py``
``_CASCADE_MODULES``) this tile is deliberately NOT wired into the
tiles→overall cascade: it contributes to WATCH only, the same class as the
Agent and Behavioral tiles.

Bands (registry: ``director_quality.director_*``) are provisional ratified starting
values — revisit once several cycles of real retro grades accumulate.
``director_route_degraded`` is the exception: its band is not provisional, it is
definitional — 0 is "the champion served" and 1 is "it did not".

Spec: config#1674.
"""

from __future__ import annotations

import json
import logging

import boto3
from botocore.exceptions import ClientError

from grading.metric_record import build_metric
from grading.thresholds.registry import resolve as resolve_band
from grading.module_agg import build_tile
from grading.units import FRACTION, SCORE_0_100

logger = logging.getLogger(__name__)

MODULE = "director_quality"

RETRO_TREND_KEY = "director/retro_trend.json"

# The Director's own plan artifact, standing pointer (config-I7157). Read for
# the served-model / degradation stamp, NOT for the retro grades above.
LATEST_ACTION_PLAN_KEY = "director/latest/action_plan.json"

# Keys stamped onto that artifact by `director/agent.py::_stamp_route_degradation`
# — literals rather than an import, because the Report Card Lambda does not
# package `director/`. `director/agent.py` names the same four as
# `PLAN_KEY_*` constants and
# `tests/test_director_route_degradation.py::TestArtifactContract` asserts the
# two lists are identical, so a rename cannot land on one side only.
PLAN_KEY_ROUTE_DEGRADED = "route_degraded"
PLAN_KEY_SERVED_MODEL = "served_model"
PLAN_KEY_ROUTE_PRIMARY_MODEL = "route_primary_model"
PLAN_KEY_DEGRADED_REASON = "route_degraded_reason"



def _get_json(s3, bucket: str, key: str) -> dict | None:
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        logger.error("S3 read failed for s3://%s/%s: %s", bucket, key, e)
        raise
    return json.loads(resp["Body"].read())


def _latest_grade(trend: dict | None) -> dict | None:
    """The most recent retro grade row, or None if absent/empty.

    ``grades`` is upserted + sorted ascending by ``prior_run_date``
    (``director/handler.py::_persist_retro``), so the most recent entry is
    always the last element.
    """
    if not trend:
        return None
    grades = trend.get("grades")
    if not isinstance(grades, list) or not grades:
        return None
    return grades[-1]


def _component(name: str, grade_key: str, grade: dict | None, src: str):
    """Build one 0-100 supporting component (grounding/calibration/actionability).

    ``name`` is the MetricRecord name (``director_{grade_key}``); ``grade_key``
    is the field name on the persisted retro row (``grounding`` /
    ``calibration`` / ``actionability`` — unprefixed, see ``director/schema.py``
    ``RetroGrade``). ``grade`` is the latest retro row, or None when the
    artifact/list is absent — degrades to a precise N/A-MISSING-INPUT, never an
    exception.
    """
    _band = resolve_band(MODULE, name)
    if grade is not None and grade.get(grade_key) is not None:
        value = grade[grade_key]
        # judge_model is safe-.get() — config#1673 adds it; older persisted
        # rows (or a repo where #1673 hasn't landed yet) simply lack it, and
        # this must never crash reading them.
        judge_model = grade.get("judge_model")
        judge_part = f", judge_model={judge_model}" if judge_model else ""
        return build_metric(
            name=name, module=MODULE, metric_type="ratio", criticality="supporting",
            value=value, unit=SCORE_0_100, n_samples=1, n_floor=1,
            higher_is_better=True, source_path=src,
            reason=(f"{name} = {value}/100 (prior_run_date={grade.get('prior_run_date')}"
                    f"{judge_part}) vs target {_band.target:g} / red-line {_band.red_line:g}."),
        )
    return build_metric(
        name=name, module=MODULE, metric_type="ratio", criticality="supporting",
        n_floor=1, higher_is_better=True,
        source_path=src, input_present=False,
        na_detail=(f"{name}: director/retro_trend.json absent or empty this cycle "
                    "(Director disabled, first cycle, or retro skipped — config#1674)."),
    )


def _route_degraded_component(plan: dict | None, src: str):
    """Did a FALLBACK model produce the Director's most recent plan?

    ``1.0`` = a fallback served, ``0.0`` = the group's declared primary served,
    N/A-MISSING-INPUT = unknowable. The ``reason`` names the model that actually
    served, which is the whole point of the component: the number says a weaker
    model wrote that week's plan, and the string says which one, so a reader
    comparing this week's plan quality against last week's can see the seam.

    **Why the Report Card carries this at all** (alpha-engine-config-I8165).
    The ``ultra`` group gained a second arm on 2026-08-22 — ``deepseek-v4-pro``,
    behind a different provider from the primary, and deliberately WEAKER for
    this call. It exists for availability (model-router-policy R33) and nothing
    else, so a fallback-produced plan is a DEGRADED plan. Brian's ruling
    admitting the second arm was conditioned on that degradation being visible
    rather than inferred: a weaker model's plan silently entering the record as
    if a champion wrote it is the failure the arm would otherwise have
    introduced. This component is that visibility.

    **The lag is the same one the rest of this tile lives with, and is correct.**
    The ReportCard state runs BEFORE the Director state in the Saturday chain,
    so ``director/latest/action_plan.json`` at this moment is the PRIOR cycle's
    plan — the same one-cycle offset documented in the module docstring for the
    retro grades. Reading the dated key for ``run_date`` instead would render
    absent every single week, which is a detector that cries wolf rather than
    one that measures.

    **``None`` is not collapsed into ``0.0``.** The producer stamps
    ``route_degraded: None`` when it could not tell a champion-served plan from
    an unmeasured one, and an older artifact predating I8165 carries no stamp at
    all. Both render N/A with the reason attached — `principles.md` §2.7: *no
    data* is never rendered as green, and "the champion served" is exactly what
    a green 0.0 would assert.
    """
    name = "director_route_degraded"
    _band = resolve_band(MODULE, name)
    degraded = (plan or {}).get(PLAN_KEY_ROUTE_DEGRADED)
    served = (plan or {}).get(PLAN_KEY_SERVED_MODEL)
    primary = (plan or {}).get(PLAN_KEY_ROUTE_PRIMARY_MODEL)

    if isinstance(degraded, bool):
        stamped_reason = (plan or {}).get(PLAN_KEY_DEGRADED_REASON)
        detail = (
            f"served={served!r} vs group primary={primary!r}"
            + (f" — {stamped_reason}" if degraded and stamped_reason else "")
        )
        return build_metric(
            name=name, module=MODULE, metric_type="ratio", criticality="supporting",
            value=1.0 if degraded else 0.0, unit=FRACTION, n_samples=1, n_floor=1,
            higher_is_better=False, source_path=src,
            reason=(
                f"Director plan produced by a FALLBACK model: {detail}."
                if degraded else
                f"Director plan produced by the group's primary: {detail}."
            ),
        )

    return build_metric(
        name=name, module=MODULE, metric_type="ratio", criticality="supporting",
        n_floor=1, higher_is_better=False, source_path=src, input_present=False,
        na_detail=(
            f"{name}: {LATEST_ACTION_PLAN_KEY} carries no boolean "
            f"{PLAN_KEY_ROUTE_DEGRADED!r} stamp (artifact absent, Director "
            "disabled this cycle, a plan written before "
            "alpha-engine-config-I8165, or the producer could not determine the "
            f"served model: {PLAN_KEY_DEGRADED_REASON}="
            f"{(plan or {}).get(PLAN_KEY_DEGRADED_REASON)!r}). Not rendered 0.0 "
            "— that would assert the champion served."
        ),
    )


def build_director_quality_tile(bucket: str, run_date: str, s3_client=None) -> dict:
    """Build the Director Retro-Quality tile.

    ``run_date`` is accepted for signature parity with the other tile
    builders (``grading/aggregate.py`` calls every tile builder the same way)
    but is not used to key the source artifact — ``director/retro_trend.json``
    is a single running ledger, not a per-date artifact.
    """
    s3 = s3_client or boto3.client("s3")
    trend = _get_json(s3, bucket, RETRO_TREND_KEY)
    src = f"s3://{bucket}/{RETRO_TREND_KEY}"
    grade = _latest_grade(trend)

    plan_src = f"s3://{bucket}/{LATEST_ACTION_PLAN_KEY}"
    plan = _get_json(s3, bucket, LATEST_ACTION_PLAN_KEY)

    components = [
        _component("director_grounding", "grounding", grade, src),
        _component("director_calibration", "calibration", grade, src),
        _component("director_actionability", "actionability", grade, src),
        _route_degraded_component(plan, plan_src),
    ]

    return build_tile(MODULE, components)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Build the Director Retro-Quality tile.")
    parser.add_argument("--bucket", default="alpha-engine-research")
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(json.dumps(build_director_quality_tile(args.bucket, args.date), indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
