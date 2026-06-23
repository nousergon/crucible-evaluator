"""
substrate.py — Tile 5: Substrate Reliability (RC v2).

Operational substrate — a flaky substrate invalidates everything above it
(RC v2 Principle 8). One component is sourceable from S3 today
(``price_cache_freshness`` via the price-cache objects' LastModified); the rest
need producers the evaluator can't yet reach (SF/CW execution history, the
data-quality substrate inventory, GitHub Actions, CFN drift), so they grade a
**transparent N/A-NOT-IMPL whose reason names the producer to build** — the
report card says "the substrate is mostly unmeasured" out loud rather than
hiding it.

``sf_success_rate_4w`` is the headline substrate metric and the highest-value
follow-up: wire ``alpha_engine_lib.pipeline_status.list_recent_pipeline_runs``
over the 3 Step Function ARNs (Saturday / Weekday / EOD).

Spec: ``system-report-card-revamp-260522.md`` Tile 5.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from grading.metric_record import build_metric
from grading.module_agg import build_tile

logger = logging.getLogger(__name__)

MODULE = "substrate"
PRICE_CACHE_PREFIX = "predictor/price_cache/"

# The 3 orchestration Step Functions whose rolling success rate IS the substrate
# headline. ARNs are *discovered* at runtime (list_state_machines) rather than
# hardcoded, so no AWS account id lives in this public repo.
_SF_NAMES = (
    "alpha-engine-saturday-pipeline",
    "alpha-engine-weekday-pipeline",
    "alpha-engine-eod-pipeline",
)
_SF_TERMINAL = {"SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"}
_SF_WINDOW_DAYS = 28

# data_quality_incidents — the data collector's per-run quality gate emits
# ``AlphaEngine/Data/daily_append_quality_blocked_count`` (rows EXCLUDED from the
# feature store on a quality failure — the load-bearing incident) +
# ``_warned_count`` (flagged-but-kept) per run (alpha-engine-data
# builders/daily_append.py). We grade the trailing-4w SUM of BLOCKED as the
# incident count; warned rides in the reason. Thresholds calibrated off the
# 2026-06 baseline (~95 blocked / 4w ≈ 0.5%/day of ~900 tickers → GREEN); they
# are a REGRESSION detector (WATCH ≈ 2x, RED ≈ 5x baseline), re-tune once a
# trend establishes. (config#1150 Batch B.)
_DQ_NAMESPACE = "AlphaEngine/Data"
_DQ_BLOCKED_METRIC = "daily_append_quality_blocked_count"
_DQ_WARNED_METRIC = "daily_append_quality_warned_count"

# schema_drift_incidents — the data collector counts every ArcticDB
# StreamDescriptorMismatch / DataError raised on a universe write path
# (``update_batch`` / ``write_batch`` / ``_write_row_backfill_safe``) per run
# and emits ``AlphaEngine/Data/daily_append_schema_drift_count`` (alpha-engine-data
# builders/daily_append.py). UNLIKE data_quality_incidents (a routine row-level
# gate with a ~95/4w steady-state baseline), a schema-drift incident is a HARD
# data-integrity failure: the persisted ArcticDB descriptor no longer matches the
# row being written, the daily_append run FAILS LOUD (counted then re-raised),
# and downstream features are starved until an operator repairs the descriptor.
# It is meant to be ZERO in steady state, so the thresholds are an absolute
# incident count, NOT a regression band: target 0 (any incident is a real event
# worth a WATCH) / red-line 3 (a cluster of ≥3 schema failures in 4w is a
# systemic descriptor regression → RED). (config#1150 Batch B.)
_SCHEMA_DRIFT_METRIC = "daily_append_schema_drift_count"


def _cw_metric_sum(cw, namespace: str, metric: str, as_of: datetime, window_days: int) -> float | None:
    """Trailing-``window_days`` SUM of a CloudWatch metric, or None if no data."""
    resp = cw.get_metric_statistics(
        Namespace=namespace,
        MetricName=metric,
        StartTime=as_of - timedelta(days=window_days),
        EndTime=as_of,
        Period=window_days * 86400,
        Statistics=["Sum"],
    )
    points = resp.get("Datapoints") or []
    if not points:
        return None
    return float(sum(p.get("Sum", 0.0) for p in points))

# SCHEDULED-cadence roles: an EventBridge-triggered run that is SUPPOSED to
# complete on its own (the "unattended" target). Everything else role-carrying
# (recovery / operator / operator-replay / backfill / shell-run) is an operator
# intervention — it still completes the cycle, but its presence means the
# scheduled run did NOT succeed unattended. (config#1059 / #970 / L4552d.)
_SCHEDULED_ROLES = {"weekly", "saturday", "daily", "eod"}


def _discover_sf_arns(sfn) -> list[str]:
    """Resolve the 3 pipeline SF ARNs by name (no hardcoded account id).

    Env override ``EVALUATOR_SF_ARNS`` (comma-separated) wins — lets the Lambda
    skip the ListStateMachines call / IAM grant if the ARNs are configured.
    """
    env = os.environ.get("EVALUATOR_SF_ARNS")
    if env:
        return [a.strip() for a in env.split(",") if a.strip()]
    arns: list[str] = []
    paginator = sfn.get_paginator("list_state_machines")
    for page in paginator.paginate():
        for sm in page.get("stateMachines", []):
            if sm.get("name") in _SF_NAMES:
                arns.append(sm["stateMachineArn"])
    return arns


def _sf_success_rate(sfn, as_of: datetime, window_days: int) -> dict | None:
    """Cycle-level success across the 3 SFs over the trailing window.

    Returns two distinct, complementary metrics (config#1059 / #970 / L4552d) —
    the OLD per-execution rate conflated operator-recovered cycles AND scheduled
    failures into one false-RED "everything is broken" number:

    - ``cycle_rate`` (distinct-cycle outcome): a TRADING CYCLE that ultimately
      completed clean = success, REGARDLESS of how many recovery runs it took.
      This is the honest "did the work get done?" axis — a Saturday that failed
      its scheduled run but was recovered-to-green still produced the retrains/
      backtests, so the downstream tiles are NOT measured on a starved system.
    - ``unattended_rate`` (first-pass / no-operator): the SCHEDULED run
      (pipeline_role ∈ scheduled-cadence) succeeded with NO recovery run in the
      same cycle. This surfaces the genuine target — full automation — honestly
      (e.g. ~0 for Saturday) instead of hiding it inside the conflated number.

    **Cycle key (principled approximation + its limitation):** the lightweight
    ``PipelineExecutionSummary`` carries ``pipeline_role`` + ``start_utc`` but
    NOT the artifact ``trading_day``, so a cycle cannot be keyed on the true
    trading day. We approximate one cycle = ``(sf_name, start_utc UTC-date)``:
    all executions of a given SF that START on the same UTC calendar date are
    treated as one cycle, with recovery reruns landing same-day as the scheduled
    run. This holds for the live cadence (scheduled run + same-day recoveries);
    it would mis-split only if a recovery slipped past UTC midnight (rare) — in
    which case the cycle is counted as two, slightly UNDER-counting recovery
    linkage (conservative: never inflates the unattended rate). When the SF
    summary gains a ``trading_day`` field, re-key on it directly.

    Returns ``{cycle_rate, n_cycles, n_cycles_clean, unattended_rate,
    n_unattended, per_sf, per_sf_unattended}`` or ``None`` when no SF ARNs are
    discoverable. Terminal = SUCCEEDED/FAILED/TIMED_OUT/ABORTED; RUNNING /
    NOT_RUN excluded.
    """
    from alpha_engine_lib.pipeline_status import list_recent_pipeline_runs

    arns = _discover_sf_arns(sfn)
    if not arns:
        return None
    cutoff = as_of - timedelta(days=window_days)
    n_cycles = n_cycles_clean = 0
    n_unattended_cycles = n_unattended_ok = 0
    per_sf: dict[str, str] = {}
    per_sf_unattended: dict[str, str] = {}
    for arn in arns:
        runs = list_recent_pipeline_runs(arn, limit=50, client=sfn)
        # Bucket this SF's terminal, role-carrying, in-window executions into
        # cycles keyed on the UTC date of start_utc.
        cycles: dict[object, list[tuple[str, str | None]]] = {}
        for r in runs:
            start = getattr(r, "start_utc", None)
            if start is None or start < cutoff:
                continue
            # PRODUCTION runs only: EventBridge-triggered + operator-tracked
            # executions carry a pipeline_role; ad-hoc smoke / legacy runs have
            # no role and would misleadingly tank the rate (cry-wolf).
            role = getattr(r, "pipeline_role", None)
            if role is None:
                continue
            # RunStatus is a (str, Enum) — str() yields "RunStatus.SUCCEEDED",
            # so read .value to get the bare AWS status vocabulary.
            raw = getattr(r, "status", None)
            status = getattr(raw, "value", raw)
            if status not in _SF_TERMINAL:
                continue
            cycles.setdefault(start.date(), []).append((status, role))

        sf_cycles = sf_cycles_clean = 0
        sf_unatt = sf_unatt_ok = 0
        for _day, execs in cycles.items():
            sf_cycles += 1
            # Distinct-cycle outcome: clean iff ANY execution in the cycle (the
            # scheduled run OR a recovery rerun) ultimately SUCCEEDED.
            if any(st == "SUCCEEDED" for st, _role in execs):
                sf_cycles_clean += 1
            # Unattended first-pass: only cycles that HAD a scheduled run count
            # toward the unattended denominator (an operator-only ad-hoc day is
            # not an unattended-cadence opportunity). It succeeded unattended iff
            # the scheduled run itself SUCCEEDED *and* no recovery role appears.
            scheduled = [(st, role) for st, role in execs if role in _SCHEDULED_ROLES]
            if scheduled:
                sf_unatt += 1
                had_recovery = any(role not in _SCHEDULED_ROLES for _st, role in execs)
                scheduled_ok = any(st == "SUCCEEDED" for st, _role in scheduled)
                if scheduled_ok and not had_recovery:
                    sf_unatt_ok += 1

        n_cycles += sf_cycles
        n_cycles_clean += sf_cycles_clean
        n_unattended_cycles += sf_unatt
        n_unattended_ok += sf_unatt_ok
        name = arn.rsplit(":", 1)[-1]
        per_sf[name] = f"{sf_cycles_clean}/{sf_cycles}"
        per_sf_unattended[name] = f"{sf_unatt_ok}/{sf_unatt}"

    return {
        "cycle_rate": (n_cycles_clean / n_cycles) if n_cycles else None,
        "n_cycles": n_cycles,
        "n_cycles_clean": n_cycles_clean,
        "unattended_rate": (n_unattended_ok / n_unattended_cycles) if n_unattended_cycles else None,
        "n_unattended": n_unattended_cycles,
        "n_unattended_ok": n_unattended_ok,
        "per_sf": per_sf,
        "per_sf_unattended": per_sf_unattended,
    }


def _latest_mtime(s3, bucket: str, prefix: str) -> datetime | None:
    """Max LastModified across objects under ``prefix`` (None if empty)."""
    latest = None
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                lm = obj["LastModified"]
                if latest is None or lm > latest:
                    latest = lm
    except ClientError as e:
        logger.error("S3 list failed for s3://%s/%s: %s", bucket, prefix, e)
        raise
    return latest


def build_substrate_tile(
    bucket: str,
    s3_client=None,
    *,
    as_of: datetime | None = None,
    sfn_client=None,
    cloudwatch_client=None,
) -> dict:
    """Build the Substrate Reliability tile."""
    s3 = s3_client or boto3.client("s3")
    as_of = as_of or datetime.now(UTC)
    components = []

    # 1. price_cache_freshness (critical) — days since the price cache last wrote.
    pc_src = f"s3://{bucket}/{PRICE_CACHE_PREFIX}"
    latest = _latest_mtime(s3, bucket, PRICE_CACHE_PREFIX)
    if latest is not None:
        age_d = (as_of - latest).total_seconds() / 86400.0
        components.append(build_metric(
            name="price_cache_freshness", module=MODULE, metric_type="duration", criticality="critical",
            estimator="freshness_age",
            value=age_d, n_samples=1, n_floor=1, target=7.0, red_line=14.0, higher_is_better=False,
            source_path=pc_src,
            reason=f"price_cache_freshness = {age_d:.1f}d since the price cache last refreshed vs target 7d / red-line 14d.",
        ))
    else:
        components.append(build_metric(
            name="price_cache_freshness", module=MODULE, metric_type="duration", criticality="critical",
            estimator="freshness_age",
            n_floor=1, target=7.0, red_line=14.0, higher_is_better=False, source_path=pc_src,
            input_present=False,
            na_detail="price_cache_freshness: no objects under predictor/price_cache/ to date-stamp.",
        ))

    # 2. sf_success_rate_4w (critical) + unattended_first_pass_rate (supporting) —
    #    the substrate headline, re-keyed (config#1059 / #970 / L4552d). The OLD
    #    per-EXECUTION rate counted operator-recovered cycles AND scheduled-run
    #    failures as failures, producing a false P0 RED (0.4918) even on a week
    #    where every cycle ultimately completed clean. We now grade two distinct
    #    axes off the SF execution history (alpha_engine_lib.pipeline_status):
    #      - sf_success_rate_4w   = DISTINCT-CYCLE outcome (clean = recovered or
    #                               not). The honest "did the work get done?" axis.
    #      - unattended_first_pass_rate = scheduled run succeeded w/ NO recovery.
    #                               Surfaces the genuine full-automation target.
    #    Graceful N/A on an SF access error (a secondary read must not fail the
    #    whole report card — WARN-logged, not swallowed).
    sf_src = "stepfunctions:alpha-engine-{saturday,weekday,eod}-pipeline"

    def _na_pair(*, ran=True, input_present=True, cycle_detail, unatt_detail):
        components.append(build_metric(
            name="sf_success_rate_4w", module=MODULE, metric_type="pct", criticality="critical",
            estimator="distinct_cycle_success_4w", measurement_horizon="trailing_4w",
            n_floor=3, target=0.95, red_line=0.80, source_path=sf_src,
            ran=ran, input_present=input_present, na_detail=cycle_detail,
        ))
        components.append(build_metric(
            name="unattended_first_pass_rate", module=MODULE, metric_type="pct", criticality="supporting",
            estimator="unattended_first_pass_4w", measurement_horizon="trailing_4w",
            n_floor=3, target=0.95, red_line=0.50, source_path=sf_src,
            ran=ran, input_present=input_present, na_detail=unatt_detail,
        ))

    try:
        sfn = sfn_client or boto3.client(
            "stepfunctions", region_name=os.environ.get("AWS_REGION", "us-east-1")
        )
        sf = _sf_success_rate(sfn, as_of, _SF_WINDOW_DAYS)
        if sf is None:
            _na_pair(
                input_present=False,
                cycle_detail="sf_success_rate_4w: no pipeline SF ARNs discoverable (set EVALUATOR_SF_ARNS or grant states:ListStateMachines).",
                unatt_detail="unattended_first_pass_rate: no pipeline SF ARNs discoverable (set EVALUATOR_SF_ARNS or grant states:ListStateMachines).",
            )
        elif sf["cycle_rate"] is None:
            _na_pair(
                ran=False,
                cycle_detail=f"sf_success_rate_4w: no terminal production-role SF cycles in the last {_SF_WINDOW_DAYS}d.",
                unatt_detail=f"unattended_first_pass_rate: no scheduled-cadence SF cycles in the last {_SF_WINDOW_DAYS}d.",
            )
        else:
            # Distinct-cycle outcome (critical headline).
            components.append(build_metric(
                name="sf_success_rate_4w", module=MODULE, metric_type="pct", criticality="critical",
                estimator="distinct_cycle_success_4w", measurement_horizon="trailing_4w",
                value=sf["cycle_rate"], n_samples=sf["n_cycles"], n_floor=3, target=0.95, red_line=0.80,
                source_path=sf_src,
                reason=(f"sf_success_rate_4w = {sf['cycle_rate']:.0%} ({sf['n_cycles_clean']}/{sf['n_cycles']} "
                        f"DISTINCT production-role cycles completed clean — recovery counts as clean — in "
                        f"{_SF_WINDOW_DAYS}d: {sf['per_sf']}) vs target 95% / red-line 80%. "
                        f"Re-keyed off per-execution (config#1059): a recovered cycle still produced its "
                        f"artifacts, so downstream tiles are NOT measured on a starved system."),
            ))
            # Unattended first-pass (supporting — the true automation target).
            if sf["unattended_rate"] is None:
                components.append(build_metric(
                    name="unattended_first_pass_rate", module=MODULE, metric_type="pct", criticality="supporting",
                    estimator="unattended_first_pass_4w", measurement_horizon="trailing_4w",
                    n_floor=3, target=0.95, red_line=0.50, source_path=sf_src, ran=False,
                    na_detail=f"unattended_first_pass_rate: no scheduled-cadence cycles in the last {_SF_WINDOW_DAYS}d.",
                ))
            else:
                components.append(build_metric(
                    name="unattended_first_pass_rate", module=MODULE, metric_type="pct", criticality="supporting",
                    estimator="unattended_first_pass_4w", measurement_horizon="trailing_4w",
                    value=sf["unattended_rate"], n_samples=sf["n_unattended"], n_floor=3,
                    target=0.95, red_line=0.50, source_path=sf_src,
                    reason=(f"unattended_first_pass_rate = {sf['unattended_rate']:.0%} ({sf['n_unattended_ok']}/"
                            f"{sf['n_unattended']} scheduled cycles succeeded with NO operator recovery in "
                            f"{_SF_WINDOW_DAYS}d: {sf['per_sf_unattended']}) vs target 95% / red-line 50%. "
                            f"The genuine full-automation target (config#970/L4552d) — distinct from the "
                            f"did-the-work-get-done cycle rate above."),
                ))
    except (ClientError, BotoCoreError) as e:
        code = e.response.get("Error", {}).get("Code") if isinstance(e, ClientError) else type(e).__name__
        logger.warning("sf_success_rate_4w: SF API read failed (%s) — grading N/A", e)
        _na_pair(
            ran=False,
            cycle_detail=f"sf_success_rate_4w: Step Functions read failed this cycle ({code}).",
            unatt_detail=f"unattended_first_pass_rate: Step Functions read failed this cycle ({code}).",
        )

    # 3. data_quality_incidents (critical) — trailing-4w SUM of feature-store
    #    rows BLOCKED by the data-quality gate, read from CloudWatch (mirrors the
    #    SF-history read above — the substrate tile is the one tile that reaches
    #    AWS APIs, not just S3). Graceful N/A on a CW access error / missing perm
    #    (a secondary read must not fail the whole card — WARN-logged, not
    #    swallowed). config#1150 Batch B.
    dq_src = f"cloudwatch:{_DQ_NAMESPACE}/{_DQ_BLOCKED_METRIC}"
    try:
        cw = cloudwatch_client or boto3.client(
            "cloudwatch", region_name=os.environ.get("AWS_REGION", "us-east-1")
        )
        blocked = _cw_metric_sum(cw, _DQ_NAMESPACE, _DQ_BLOCKED_METRIC, as_of, _SF_WINDOW_DAYS)
        warned = _cw_metric_sum(cw, _DQ_NAMESPACE, _DQ_WARNED_METRIC, as_of, _SF_WINDOW_DAYS)
        if blocked is None:
            components.append(build_metric(
                name="data_quality_incidents", module=MODULE, metric_type="count", criticality="critical",
                estimator="incident_count_4w", measurement_horizon="trailing_4w",
                n_floor=1, target=200.0, red_line=500.0, higher_is_better=False, source_path=dq_src,
                ran=False,
                na_detail=f"data_quality_incidents: no {_DQ_BLOCKED_METRIC} datapoints in CloudWatch over {_SF_WINDOW_DAYS}d.",
            ))
        else:
            warned_s = f"{warned:.0f}" if warned is not None else "n/a"
            components.append(build_metric(
                name="data_quality_incidents", module=MODULE, metric_type="count", criticality="critical",
                estimator="incident_count_4w", measurement_horizon="trailing_4w",
                value=blocked, n_samples=1, n_floor=1, target=200.0, red_line=500.0,
                higher_is_better=False, source_path=dq_src,
                reason=(f"data_quality_incidents = {blocked:.0f} feature-store rows BLOCKED by the "
                        f"quality gate over {_SF_WINDOW_DAYS}d (warned: {warned_s}) vs target 200 / "
                        f"red-line 500 — a regression detector calibrated off the 2026-06 baseline."),
            ))
    except (ClientError, BotoCoreError) as e:
        code = e.response.get("Error", {}).get("Code") if isinstance(e, ClientError) else type(e).__name__
        logger.warning("data_quality_incidents: CloudWatch read failed (%s) — grading N/A", e)
        components.append(build_metric(
            name="data_quality_incidents", module=MODULE, metric_type="count", criticality="critical",
            estimator="incident_count_4w", measurement_horizon="trailing_4w",
            n_floor=1, target=200.0, red_line=500.0, higher_is_better=False, source_path=dq_src,
            input_present=False,
            na_detail=f"data_quality_incidents: CloudWatch read failed this cycle ({code}) — grant cloudwatch:GetMetricStatistics to the evaluator role.",
        ))

    # 3b. schema_drift_incidents (critical) — trailing-4w SUM of ArcticDB
    #    StreamDescriptorMismatch / DataError write failures, counted-then-
    #    re-raised by the data collector and emitted to CloudWatch (mirrors the
    #    data_quality read above — the substrate tile is the one tile that reaches
    #    AWS APIs). A schema-drift incident is a HARD data-integrity failure (the
    #    daily_append run fails loud), so the band is an absolute incident count
    #    (target 0 / red-line 3) rather than a regression detector. Graceful N/A
    #    on a CW access error / missing perm — WARN-logged, not swallowed.
    #    config#1150 Batch B.
    sd_src = f"cloudwatch:{_DQ_NAMESPACE}/{_SCHEMA_DRIFT_METRIC}"
    try:
        cw = cloudwatch_client or boto3.client(
            "cloudwatch", region_name=os.environ.get("AWS_REGION", "us-east-1")
        )
        drift = _cw_metric_sum(cw, _DQ_NAMESPACE, _SCHEMA_DRIFT_METRIC, as_of, _SF_WINDOW_DAYS)
        if drift is None:
            components.append(build_metric(
                name="schema_drift_incidents", module=MODULE, metric_type="count", criticality="critical",
                estimator="incident_count_4w", measurement_horizon="trailing_4w",
                n_floor=1, target=0.0, red_line=3.0, higher_is_better=False, source_path=sd_src,
                ran=False,
                na_detail=(f"schema_drift_incidents: no {_SCHEMA_DRIFT_METRIC} datapoints in CloudWatch "
                           f"over {_SF_WINDOW_DAYS}d (the metric self-activates once a daily_append run "
                           f"emits it — zero is emitted on every clean run, so N/A means the instrumented "
                           f"producer has not yet deployed/run)."),
            ))
        else:
            components.append(build_metric(
                name="schema_drift_incidents", module=MODULE, metric_type="count", criticality="critical",
                estimator="incident_count_4w", measurement_horizon="trailing_4w",
                value=drift, n_samples=1, n_floor=1, target=0.0, red_line=3.0,
                higher_is_better=False, source_path=sd_src,
                reason=(f"schema_drift_incidents = {drift:.0f} ArcticDB StreamDescriptorMismatch / "
                        f"DataError write failures over {_SF_WINDOW_DAYS}d vs target 0 / red-line 3. "
                        f"A schema-drift incident is a HARD data-integrity failure (the daily_append "
                        f"write fails loud) — meant to be zero in steady state; any incident is a WATCH, "
                        f"a cluster (≥3) is a systemic descriptor regression (RED)."),
            ))
    except (ClientError, BotoCoreError) as e:
        code = e.response.get("Error", {}).get("Code") if isinstance(e, ClientError) else type(e).__name__
        logger.warning("schema_drift_incidents: CloudWatch read failed (%s) — grading N/A", e)
        components.append(build_metric(
            name="schema_drift_incidents", module=MODULE, metric_type="count", criticality="critical",
            estimator="incident_count_4w", measurement_horizon="trailing_4w",
            n_floor=1, target=0.0, red_line=3.0, higher_is_better=False, source_path=sd_src,
            input_present=False,
            na_detail=f"schema_drift_incidents: CloudWatch read failed this cycle ({code}) — grant cloudwatch:GetMetricStatistics to the evaluator role.",
        ))

    # 4-9. Producers not yet reachable by the evaluator — transparent N/A-NOT-IMPL,
    #      each reason naming the producer to wire.
    not_impl = [
        ("deploy_success_rate", "supporting",
         "deploy_success_rate: needs GitHub Actions run history across the 8 repos (GH API) — outside the evaluator's S3 reach today."),
        ("alert_noise_ratio", "supporting",
         "alert_noise_ratio: needs the alerts log + a manual actionable/total tag — not yet sourced."),
        ("watchdog_firings", "supporting",
         "watchdog_firings: needs the backtester PhaseTimeoutError / silent-phase-tripwire firing count — not yet aggregated."),
        ("changelog_coverage", "diagnostic",
         "changelog_coverage: needs an expected-event-source set to compute % writing to the changelog — not yet defined."),
        ("iam_drift", "diagnostic",
         "iam_drift: needs CFN detect-drift delta — not yet exposed to the evaluator."),
    ]
    for name, crit, detail in not_impl:
        components.append(build_metric(
            name=name, module=MODULE, metric_type="pct", criticality=crit, n_floor=1,
            source_path=f"s3://{bucket}/", implemented=False, na_detail=detail,
        ))

    return build_tile(MODULE, components)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Build the Substrate Reliability tile.")
    parser.add_argument("--bucket", default="alpha-engine-research")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(json.dumps(build_substrate_tile(args.bucket), indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
