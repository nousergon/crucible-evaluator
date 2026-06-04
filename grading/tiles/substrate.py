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
from datetime import UTC, datetime

import boto3
from botocore.exceptions import ClientError

from grading.metric_record import build_metric
from grading.module_agg import build_tile

logger = logging.getLogger(__name__)

MODULE = "substrate"
PRICE_CACHE_PREFIX = "predictor/price_cache/"


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


def build_substrate_tile(bucket: str, s3_client=None, *, as_of: datetime | None = None) -> dict:
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

    # 2-9. Producers not yet reachable by the evaluator — transparent N/A-NOT-IMPL,
    #      each reason naming the producer to wire.
    not_impl = [
        ("sf_success_rate_4w", "critical",
         "sf_success_rate_4w: wire alpha_engine_lib.pipeline_status.list_recent_pipeline_runs over the 3 SF ARNs (Saturday/Weekday/EOD), rolling 4w success rate. HIGHEST-VALUE substrate follow-up."),
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
