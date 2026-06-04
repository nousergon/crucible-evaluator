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
    """Rolling success rate across the 3 SFs over the trailing window.

    Returns ``{rate, n_terminal, n_succeeded, per_sf}`` or ``None`` when no SF
    ARNs are discoverable. Terminal = SUCCEEDED/FAILED/TIMED_OUT/ABORTED;
    RUNNING / NOT_RUN excluded from the denominator.
    """
    from alpha_engine_lib.pipeline_status import list_recent_pipeline_runs

    arns = _discover_sf_arns(sfn)
    if not arns:
        return None
    cutoff = as_of - timedelta(days=window_days)
    n_terminal = n_succeeded = 0
    per_sf: dict[str, str] = {}
    for arn in arns:
        runs = list_recent_pipeline_runs(arn, limit=50, client=sfn)
        succ = term = 0
        for r in runs:
            start = getattr(r, "start_utc", None)
            if start is not None and start < cutoff:
                continue
            # Count PRODUCTION runs only: EventBridge-triggered executions carry a
            # pipeline_role (weekly/saturday/daily/eod/recovery); ad-hoc smoke /
            # manual / legacy runs have no role and would misleadingly tank the
            # rate (cry-wolf). Role-carrying ⇒ a tracked, intended run.
            if getattr(r, "pipeline_role", None) is None:
                continue
            # RunStatus is a (str, Enum) — str() yields "RunStatus.SUCCEEDED",
            # so read .value to get the bare AWS status vocabulary.
            raw = getattr(r, "status", None)
            status = getattr(raw, "value", raw)
            if status in _SF_TERMINAL:
                term += 1
                if status == "SUCCEEDED":
                    succ += 1
        n_terminal += term
        n_succeeded += succ
        name = arn.rsplit(":", 1)[-1]
        per_sf[name] = f"{succ}/{term}"
    if n_terminal == 0:
        return {"rate": None, "n_terminal": 0, "n_succeeded": 0, "per_sf": per_sf}
    return {
        "rate": n_succeeded / n_terminal,
        "n_terminal": n_terminal,
        "n_succeeded": n_succeeded,
        "per_sf": per_sf,
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
            value=age_d, n_samples=1, n_floor=1, target=7.0, red_line=14.0, higher_is_better=False,
            source_path=pc_src,
            reason=f"price_cache_freshness = {age_d:.1f}d since the price cache last refreshed vs target 7d / red-line 14d.",
        ))
    else:
        components.append(build_metric(
            name="price_cache_freshness", module=MODULE, metric_type="duration", criticality="critical",
            n_floor=1, target=7.0, red_line=14.0, higher_is_better=False, source_path=pc_src,
            input_present=False,
            na_detail="price_cache_freshness: no objects under predictor/price_cache/ to date-stamp.",
        ))

    # 2. sf_success_rate_4w (critical) — the substrate headline: rolling success
    #    rate across the 3 orchestration SFs (Saturday/Weekday/EOD). Reads the SF
    #    API via alpha_engine_lib.pipeline_status. Graceful N/A-NOT-RUN on an SF
    #    access error (a secondary read must not fail the whole report card —
    #    WARN-logged, not swallowed).
    sf_src = "stepfunctions:alpha-engine-{saturday,weekday,eod}-pipeline"
    try:
        sfn = sfn_client or boto3.client(
            "stepfunctions", region_name=os.environ.get("AWS_REGION", "us-east-1")
        )
        sf = _sf_success_rate(sfn, as_of, _SF_WINDOW_DAYS)
        if sf is None:
            components.append(build_metric(
                name="sf_success_rate_4w", module=MODULE, metric_type="pct", criticality="critical",
                n_floor=3, target=0.95, red_line=0.80, source_path=sf_src, input_present=False,
                na_detail="sf_success_rate_4w: no pipeline SF ARNs discoverable (set EVALUATOR_SF_ARNS or grant states:ListStateMachines).",
            ))
        elif sf["rate"] is None:
            components.append(build_metric(
                name="sf_success_rate_4w", module=MODULE, metric_type="pct", criticality="critical",
                n_floor=3, target=0.95, red_line=0.80, source_path=sf_src, ran=False,
                na_detail=f"sf_success_rate_4w: no terminal SF executions in the last {_SF_WINDOW_DAYS}d.",
            ))
        else:
            components.append(build_metric(
                name="sf_success_rate_4w", module=MODULE, metric_type="pct", criticality="critical",
                value=sf["rate"], n_samples=sf["n_terminal"], n_floor=3, target=0.95, red_line=0.80,
                source_path=sf_src,
                reason=(f"sf_success_rate_4w = {sf['rate']:.0%} ({sf['n_succeeded']}/{sf['n_terminal']} "
                        f"production-role terminal SF executions in {_SF_WINDOW_DAYS}d: {sf['per_sf']}) "
                        f"vs target 95% / red-line 80%."),
            ))
    except (ClientError, BotoCoreError) as e:
        code = e.response.get("Error", {}).get("Code") if isinstance(e, ClientError) else type(e).__name__
        logger.warning("sf_success_rate_4w: SF API read failed (%s) — grading N/A", e)
        components.append(build_metric(
            name="sf_success_rate_4w", module=MODULE, metric_type="pct", criticality="critical",
            n_floor=3, target=0.95, red_line=0.80, source_path=sf_src, ran=False,
            na_detail=f"sf_success_rate_4w: Step Functions read failed this cycle ({code}).",
        ))

    # 3-9. Producers not yet reachable by the evaluator — transparent N/A-NOT-IMPL,
    #      each reason naming the producer to wire.
    not_impl = [
        ("data_quality_incidents", "critical",
         "data_quality_incidents: needs the data-quality substrate inventory (count of failing rows, last 4w) — not yet exposed to the evaluator."),
        ("schema_drift_incidents", "critical",
         "schema_drift_incidents: needs the ArcticDB StreamDescriptorMismatch / schema-failure error log — not yet aggregated."),
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
