"""
backtester.py — Tile 4: Backtester (RC v2).

The backtester self-grades on observable feedback-loop integrity — bounded to
artifacts it emits, NOT meta-judgement of whether its recommendations are good
(that's what the auto-apply guardrails + holdout validation cover). Avoids the
evaluator-grading-itself foot-gun (RC v2 Tile-4 note).

Gradable from S3 today:
  - evaluator_coverage : audit grading.json — % of leaf components with a real
    (non-N/A) grade this cycle. This is the meta-metric: it measures the very
    "insufficient data" cliff RC v2 exists to close.
  - grading_freshness  : hours since grading.json last written vs the Sat cadence
  - vectorized_vs_consolidated_parity : parity_report.json divergence
  - fdr_surface_health : count of BH-FDR-significant correlations in attribution
  - auto_apply_rollback_count : objects under config/rollback_audit/

optimizer_churn / sample_size_adequacy / backtest_vs_live_parity /
walk_forward_stability need param-history diffing or sweep-fold data not cleanly
persisted yet → transparent N/A-NOT-IMPL.

Spec: ``system-report-card-revamp-260522.md`` Tile 4.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

import boto3
from botocore.exceptions import ClientError

from grading.metric_record import build_metric
from grading.module_agg import build_tile

logger = logging.getLogger(__name__)

MODULE = "backtester"
_CADENCE_H = 168  # weekly Saturday cadence


def _get_json(s3, bucket: str, key: str) -> dict | None:
    """Read JSON, tolerant of an absent / empty / corrupt artifact.

    A missing or unparseable diagnostic is a legitimate N/A here (secondary
    observability — recorded as N/A on the tile, WARN-logged), never a hard
    failure of the whole report card; a real S3 access error still raises.
    """
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        logger.error("S3 read failed for s3://%s/%s: %s", bucket, key, e)
        raise
    body = resp["Body"].read()
    if not body or not body.strip():
        logger.warning("Artifact empty: s3://%s/%s", bucket, key)
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        logger.warning("Artifact unparseable (corrupt JSON): s3://%s/%s", bucket, key)
        return None


def _coverage(grading: dict) -> tuple[float | None, int, int]:
    """Fraction of leaf components in grading.json with a non-N/A letter."""
    total = graded = 0
    for module in ("research", "predictor", "executor"):
        comps = (grading.get(module) or {}).get("components") or {}
        for key, v in comps.items():
            if key.endswith("_avg"):
                continue  # derived rollup, not a leaf
            if isinstance(v, list):  # sector_teams expands to per-team leaves
                for item in v:
                    total += 1
                    if isinstance(item, dict) and item.get("letter") not in (None, "N/A"):
                        graded += 1
            elif isinstance(v, dict):
                total += 1
                if v.get("letter") not in (None, "N/A"):
                    graded += 1
    return (graded / total if total else None), graded, total


def build_backtester_tile(bucket: str, run_date: str, s3_client=None, *, as_of: datetime | None = None) -> dict:
    """Build the Backtester self-grade tile."""
    s3 = s3_client or boto3.client("s3")
    as_of = as_of or datetime.now(UTC)
    prefix = f"backtest/{run_date}"
    components = []

    def src(name):
        return f"s3://{bucket}/{prefix}/{name}"

    # 1. evaluator_coverage (critical) — the anti-"insufficient data" meta-metric.
    grading = _get_json(s3, bucket, f"{prefix}/grading.json")
    g_src = src("grading.json")
    if grading:
        cov, graded, total = _coverage(grading)
        components.append(build_metric(
            name="evaluator_coverage", module=MODULE, metric_type="pct", criticality="critical",
            value=cov, n_samples=total, n_floor=1, target=0.95, red_line=0.80, source_path=g_src,
            reason=(f"evaluator_coverage = {cov:.0%} ({graded}/{total} leaf components graded, "
                    f"non-N/A) vs target 95% / red-line 80%." if cov is not None else None),
            na_detail="evaluator_coverage: grading.json has no gradable components this cycle.",
        ))
    else:
        components.append(build_metric(
            name="evaluator_coverage", module=MODULE, metric_type="pct", criticality="critical",
            n_floor=1, target=0.95, red_line=0.80, source_path=g_src, input_present=False,
            na_detail="evaluator_coverage: grading.json absent this cycle.",
        ))

    # 2. grading_freshness (supporting) — hours since grading.json last written.
    try:
        head = s3.head_object(Bucket=bucket, Key=f"{prefix}/grading.json")
        last_mod = head["LastModified"]
        age_h = (as_of - last_mod).total_seconds() / 3600.0
        components.append(build_metric(
            name="grading_freshness", module=MODULE, metric_type="duration", criticality="supporting",
            value=age_h, n_samples=1, n_floor=1, target=float(_CADENCE_H), red_line=float(_CADENCE_H + 24),
            higher_is_better=False, source_path=g_src,
            reason=f"grading_freshness = {age_h:.0f}h since grading.json written vs {_CADENCE_H}h cadence.",
        ))
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            components.append(build_metric(
                name="grading_freshness", module=MODULE, metric_type="duration", criticality="supporting",
                n_floor=1, target=float(_CADENCE_H), red_line=float(_CADENCE_H + 24), higher_is_better=False,
                source_path=g_src, input_present=False,
                na_detail="grading_freshness: no grading.json for this cycle to date-stamp.",
            ))
        else:
            raise

    # 3. vectorized_vs_consolidated_parity (supporting) — sim-path agreement.
    parity = _get_json(s3, bucket, f"{prefix}/parity_report.json")
    p_src = src("parity_report.json")
    ok_states = ("ok", "parity_ok", "clean", "match")
    if parity and (parity.get("data_state") in ok_states):
        diverged = bool(parity.get("trade_count_divergence")) or bool(parity.get("ticker_set_divergence"))
        components.append(build_metric(
            name="vectorized_vs_consolidated_parity", module=MODULE, metric_type="pct", criticality="supporting",
            value=0.0 if diverged else 1.0, n_samples=1, n_floor=1, source_path=p_src,
            status="WATCH" if diverged else "GREEN",
            reason=(f"parity data_state={parity.get('data_state')}; "
                    f"{'divergence present' if diverged else 'no trade/ticker divergence'}."),
        ))
    else:
        ds = parity.get("data_state") if parity else None
        components.append(build_metric(
            name="vectorized_vs_consolidated_parity", module=MODULE, metric_type="pct", criticality="supporting",
            n_floor=1, source_path=p_src, input_present=False,
            na_detail=(f"vectorized_vs_consolidated_parity: parity_report data_state={ds!r} "
                       "(not a clean comparison this cycle)." if ds
                       else "vectorized_vs_consolidated_parity: parity_report.json absent this cycle."),
        ))

    # 4. fdr_surface_health (supporting) — count of BH-FDR-significant correlations.
    attr = _get_json(s3, bucket, f"{prefix}/attribution.json")
    a_src = src("attribution.json")
    if attr and attr.get("status") == "ok":
        corr = attr.get("correlations") or {}
        n_sig = sum(
            1 for label in corr.values() if isinstance(label, dict)
            for k, v in label.items() if k.endswith("_fdr_significant") and v
        )
        # Healthy band 3–15 rejections (too few = no signal; too many = overfit).
        status = "GREEN" if 3 <= n_sig <= 15 else ("WATCH" if (1 <= n_sig < 3 or 15 < n_sig <= 30) else "RED")
        components.append(build_metric(
            name="fdr_surface_health", module=MODULE, metric_type="count", criticality="supporting",
            value=float(n_sig), n_samples=attr.get("rows_analyzed"), n_floor=1, source_path=a_src,
            status=status,
            reason=f"fdr_surface_health = {n_sig} BH-FDR-significant correlations (α=0.05) vs healthy band 3–15.",
        ))
    else:
        components.append(build_metric(
            name="fdr_surface_health", module=MODULE, metric_type="count", criticality="supporting",
            n_floor=1, source_path=a_src, input_present=False,
            na_detail="fdr_surface_health: attribution.json absent/empty/not-ok this cycle.",
        ))

    # 5. auto_apply_rollback_count (diagnostic) — config rollbacks in the last
    #    ~4 cycles (28d by LastModified), per the Tile-4 spec (C6-fu); all-time
    #    over-counted long-resolved rollbacks.
    try:
        cutoff = as_of - timedelta(days=28)
        resp = s3.list_objects_v2(Bucket=bucket, Prefix="config/rollback_audit/")
        n_rb = sum(
            1 for o in resp.get("Contents", [])
            if not o["Key"].endswith("/") and o.get("LastModified") and o["LastModified"] >= cutoff
        )
        components.append(build_metric(
            name="auto_apply_rollback_count", module=MODULE, metric_type="count", criticality="diagnostic",
            value=float(n_rb), n_samples=1, n_floor=1, target=0.0, red_line=2.0, higher_is_better=False,
            source_path=f"s3://{bucket}/config/rollback_audit/",
            reason=f"auto_apply_rollback_count = {n_rb} rollback-audit objects in the last 28d vs target 0 / red-line 2.",
        ))
    except ClientError:
        components.append(build_metric(
            name="auto_apply_rollback_count", module=MODULE, metric_type="count", criticality="diagnostic",
            n_floor=1, target=0.0, red_line=2.0, higher_is_better=False,
            source_path=f"s3://{bucket}/config/rollback_audit/", input_present=False,
            na_detail="auto_apply_rollback_count: could not list config/rollback_audit/.",
        ))

    # 6-9. Need param-history diffing / sweep-fold data not cleanly persisted yet.
    for name, crit, detail in (
        ("optimizer_churn", "critical",
         "optimizer_churn: needs per-cycle param-delta history (config/*_history) diffed across cycles — not yet computed."),
        ("sample_size_adequacy", "critical",
         "sample_size_adequacy: needs per-analysis sample-count vs documented floors — not yet aggregated for the report card."),
        ("backtest_vs_live_parity", "critical",
         "backtest_vs_live_parity: needs the predictor synthetic-backtest IC vs live L2 IC drift series — not yet persisted."),
        ("walk_forward_stability", "supporting",
         "walk_forward_stability: needs param-recommendation rank-corr across walk-forward folds — not yet persisted."),
    ):
        components.append(build_metric(
            name=name, module=MODULE, metric_type="ratio", criticality=crit, n_floor=1,
            source_path=f"s3://{bucket}/{prefix}/", implemented=False, na_detail=detail,
        ))

    return build_tile(MODULE, components)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Build the Backtester self-grade tile.")
    parser.add_argument("--bucket", default="alpha-engine-research")
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(json.dumps(build_backtester_tile(args.bucket, args.date), indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
