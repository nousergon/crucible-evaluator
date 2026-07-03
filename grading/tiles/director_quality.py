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

All three components are ``criticality="supporting"``: a single LLM-judge call
grading a single prior plan is exactly the kind of low-N, single-rater
estimate the L4562 critical-metric contract (``metric_record.py`` ~lines
163-175) exists to keep off the critical path — it must never force a red
overall via ``module_status``'s critical-gate rule, and (per ``module_agg.py``
``_CASCADE_MODULES``) this tile is deliberately NOT wired into the
tiles→overall cascade: it contributes to WATCH only, the same class as the
Agent and Behavioral tiles.

Bands (``target=75`` / ``red_line=40``) are provisional ratified starting
values — revisit once several cycles of real retro grades accumulate.

Spec: config#1674.
"""

from __future__ import annotations

import json
import logging

import boto3
from botocore.exceptions import ClientError

from grading.metric_record import build_metric
from grading.module_agg import build_tile

logger = logging.getLogger(__name__)

MODULE = "director_quality"

RETRO_TREND_KEY = "director/retro_trend.json"

# Provisional ratified starting bands (config#1674) — 0-100 raw scores,
# higher is better, revisit once real retro history accumulates.
_TARGET = 75
_RED_LINE = 40


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
    if grade is not None and grade.get(grade_key) is not None:
        value = grade[grade_key]
        # judge_model is safe-.get() — config#1673 adds it; older persisted
        # rows (or a repo where #1673 hasn't landed yet) simply lack it, and
        # this must never crash reading them.
        judge_model = grade.get("judge_model")
        judge_part = f", judge_model={judge_model}" if judge_model else ""
        return build_metric(
            name=name, module=MODULE, metric_type="ratio", criticality="supporting",
            value=value, n_samples=1, n_floor=1,
            target=_TARGET, red_line=_RED_LINE, higher_is_better=True, source_path=src,
            reason=(f"{name} = {value}/100 (prior_run_date={grade.get('prior_run_date')}"
                    f"{judge_part}) vs target {_TARGET} / red-line {_RED_LINE}."),
        )
    return build_metric(
        name=name, module=MODULE, metric_type="ratio", criticality="supporting",
        n_floor=1, target=_TARGET, red_line=_RED_LINE, higher_is_better=True,
        source_path=src, input_present=False,
        na_detail=(f"{name}: director/retro_trend.json absent or empty this cycle "
                    "(Director disabled, first cycle, or retro skipped — config#1674)."),
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

    components = [
        _component("director_grounding", "grounding", grade, src),
        _component("director_calibration", "calibration", grade, src),
        _component("director_actionability", "actionability", grade, src),
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
