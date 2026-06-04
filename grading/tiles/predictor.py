"""
predictor.py — Tile 2: Predictor (RC v2).

Grades the stacked meta-ensemble on its **leak-free** skill. The load-bearing
choice here is the IC source: the report card MUST read the leak-free
walk-forward / CPCV IC, NOT the in-sample fit IC. ``predictor/metrics/latest.json``
exposes ``l2_ic`` ≈ 0.52 — that is the IN-SAMPLE BayesianRidge fit IC (the
inflated number L4469 W1 exists to stop trusting). The honest leak-free read is
``predictor/weights/meta/manifest.json::meta_model_oos_ic_cpcv`` (combinatorial
purged CV, a distribution of OOS ICs). Grading the in-sample number would make
the Director confidently wrong about the model — exactly what measurement-first
ordering prevents. So this tile grades the CPCV mean with a bootstrap CI over
its per-path ICs, and the per-L1 ICs from the leak-free walk-forward medians.

Sources: ``predictor/metrics/latest.json`` + ``predictor/weights/meta/manifest.json``.
Spec: ``system-report-card-revamp-260522.md`` Tile 2.
"""

from __future__ import annotations

import logging

import boto3
from botocore.exceptions import ClientError

from alpha_engine_lib.quant.stats.intervals import bootstrap_ci

from grading.metric_record import build_metric
from grading.module_agg import build_tile

logger = logging.getLogger(__name__)

MODULE = "predictor"
LATEST_KEY = "predictor/metrics/latest.json"
MANIFEST_KEY = "predictor/weights/meta/manifest.json"


def _get_json(s3, bucket: str, key: str) -> dict | None:
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            logger.warning("Predictor artifact absent: s3://%s/%s", bucket, key)
            return None
        logger.error("S3 read failed for s3://%s/%s: %s", bucket, key, e)
        raise
    import json
    return json.loads(resp["Body"].read())


def build_predictor_tile(bucket: str, s3_client=None) -> dict:
    """Build the Predictor tile from the predictor's metrics + weights manifest."""
    s3 = s3_client or boto3.client("s3")
    latest = _get_json(s3, bucket, LATEST_KEY)
    manifest = _get_json(s3, bucket, MANIFEST_KEY)
    latest_src = f"s3://{bucket}/{LATEST_KEY}"
    manifest_src = f"s3://{bucket}/{MANIFEST_KEY}"

    if latest is None and manifest is None:
        miss = build_metric(
            name="meta_l2_ic", module=MODULE, metric_type="ic", criticality="critical",
            n_floor=10, target=0.05, red_line=0.0, source_path=manifest_src, input_present=False,
            na_detail="predictor metrics + weights manifest both absent this cycle.",
        )
        return build_tile(MODULE, [miss])

    latest = latest or {}
    manifest = manifest or {}
    wf = manifest.get("walk_forward") or {}
    cpcv = manifest.get("meta_model_oos_ic_cpcv") or {}
    components = []

    # 1. meta_l2_ic (critical) — LEAK-FREE CPCV mean, NOT the in-sample l2_ic.
    cpcv_ok = cpcv.get("status") == "ok"
    cpcv_mean = cpcv.get("mean_ic") if cpcv_ok else None
    ics = cpcv.get("ics") if cpcv_ok else None
    n_combos = cpcv.get("n_combos")
    ci_low = ci_high = ci_method = None
    if ics:
        boot = bootstrap_ci(ics)  # CI of the MEAN across CPCV paths (not the path dispersion)
        if boot.get("status") == "ok":
            ci_low, ci_high, ci_method = boot["ci_low"], boot["ci_high"], "bootstrap"
    frac_pos = cpcv.get("frac_positive")
    meta_reason = None
    if cpcv_mean is not None:
        meta_reason = (
            f"meta_l2_ic (leak-free CPCV) = {cpcv_mean:.3g} over {n_combos} purged-CV paths "
            f"({frac_pos:.0%} positive) vs target 0.05 / red-line 0.0 — NOT the in-sample "
            f"fit IC ({latest.get('l2_ic')}). "
        )
    components.append(build_metric(
        name="meta_l2_ic", module=MODULE, metric_type="ic", criticality="critical",
        value=cpcv_mean, n_samples=n_combos, n_floor=10, target=0.05, red_line=0.0,
        ci_low=ci_low, ci_high=ci_high, ci_method=ci_method, source_path=manifest_src,
        input_present=cpcv_ok, reason=meta_reason,
        na_detail="meta_l2_ic: CPCV leak-free read insufficient this cycle (single-path WF is canonical-coverage-starved — L4480).",
    ))

    # 2-4. Per-L1 leak-free walk-forward median ICs.
    mom_ic = wf.get("momentum_median_ic")
    vol_ic = wf.get("volatility_median_ic")
    n_folds = wf.get("n_folds")
    components.append(build_metric(
        name="momentum_l1_ic", module=MODULE, metric_type="ic", criticality="critical",
        value=mom_ic, n_samples=n_folds, n_floor=8, target=0.03, red_line=0.0,
        source_path=manifest_src, input_present=mom_ic is not None,
    ))
    components.append(build_metric(
        name="volatility_l1_ic", module=MODULE, metric_type="ic", criticality="supporting",
        value=vol_ic, n_samples=n_folds, n_floor=8, target=0.03, red_line=0.0,
        source_path=manifest_src, input_present=vol_ic is not None,
    ))
    rescal_ic = (latest.get("l1_ic") or {}).get("research_calibrator")
    rescal_n = latest.get("research_calibrator_n_samples")
    components.append(build_metric(
        name="research_calibrator_l1_ic", module=MODULE, metric_type="ic", criticality="supporting",
        value=rescal_ic, n_samples=rescal_n, n_floor=50, target=0.03, red_line=0.0,
        source_path=latest_src,
    ))

    # 5. ensemble_lift_over_best_l1 (critical) — does stacking beat the best L1?
    l1_ics = [v for v in (mom_ic, vol_ic, rescal_ic) if v is not None]
    lift = None
    lift_reason = None
    if cpcv_mean is not None and l1_ics:
        best_l1 = max(l1_ics)
        lift = cpcv_mean - best_l1
        lift_reason = (
            f"ensemble_lift_over_best_l1 = {lift:+.3g} (leak-free meta {cpcv_mean:.3g} − best L1 "
            f"{best_l1:.3g}) vs target 0.01 / red-line -0.01 — directional (CPCV meta vs WF-median L1)."
        )
    components.append(build_metric(
        name="ensemble_lift_over_best_l1", module=MODULE, metric_type="ic", criticality="critical",
        value=lift, n_samples=n_combos, n_floor=10, target=0.01, red_line=-0.01,
        source_path=manifest_src, input_present=lift is not None, reason=lift_reason,
        na_detail="ensemble_lift: needs both the leak-free meta IC and at least one L1 IC.",
    ))

    # 6. confidence_calibration_ece (critical) — lower is better.
    cc = latest.get("confidence_calibration") or {}
    ece = cc.get("ece_after")
    components.append(build_metric(
        name="confidence_calibration_ece", module=MODULE, metric_type="calibration", criticality="critical",
        value=ece, n_samples=cc.get("n_samples"), n_floor=100, target=0.05, red_line=0.15,
        higher_is_better=False, source_path=latest_src, input_present=ece is not None,
    ))

    # 7. output_distribution_gate (critical) — the predictor's own pass/fail gate.
    odg = latest.get("output_distribution_gate") or {}
    if "passed" in odg:
        passed = bool(odg["passed"])
        components.append(build_metric(
            name="output_distribution_gate", module=MODULE, metric_type="pct", criticality="critical",
            value=1.0 if passed else 0.0, n_samples=1, n_floor=1, source_path=latest_src,
            status="GREEN" if passed else "RED",
            reason=f"output_distribution_gate {'PASS' if passed else 'FAIL'}: {odg.get('reason', '')}".strip(),
        ))
    else:
        components.append(build_metric(
            name="output_distribution_gate", module=MODULE, metric_type="pct", criticality="critical",
            n_floor=1, source_path=latest_src, input_present=False,
            na_detail="output_distribution_gate: no gate result in latest.json this cycle.",
        ))

    # 8-10. Cross-tile / unresolved inputs — precise N/A (not silent omission).
    components.append(build_metric(
        name="veto_gate_precision", module=MODULE, metric_type="pct", criticality="supporting",
        n_floor=30, target=0.60, red_line=0.40, source_path=latest_src, input_present=False,
        na_detail="veto_gate_precision: needs the backtester shadow-book + realized-outcome join (backtest/{date}/veto_analysis.json); not sourced from predictor metrics. Cross-tile follow-up.",
    ))
    components.append(build_metric(
        name="inference_coverage", module=MODULE, metric_type="pct", criticality="critical",
        value=None, n_floor=1, target=0.95, red_line=0.80, source_path=latest_src, input_present=False,
        na_detail=(f"inference_coverage: n_predictions_today={latest.get('n_predictions_today')} observed, "
                   "but the tradable-universe denominator (signals.json universe count) is not joined yet to compute %. Follow-up."),
    ))
    components.append(build_metric(
        name="slim_cache_freshness", module=MODULE, metric_type="duration", criticality="supporting",
        n_floor=1, target=7.0, red_line=14.0, higher_is_better=False, source_path=latest_src,
        input_present=False,
        na_detail="slim_cache_freshness: slim-cache S3 prefix not resolved (predictor/price_cache_slim/ empty); needs the live cache path to compute age. Follow-up.",
    ))

    # 11. feature_drift_ks (diagnostic) — not yet produced.
    components.append(build_metric(
        name="feature_drift_ks", module=MODULE, metric_type="ratio", criticality="diagnostic",
        n_floor=30, target=0.10, red_line=0.25, higher_is_better=False, source_path=latest_src,
        implemented=False,
        na_detail="feature_drift_ks: inference-vs-training feature KS not yet computed/persisted by the predictor.",
    ))

    return build_tile(MODULE, components)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Build the Predictor tile from predictor metrics.")
    parser.add_argument("--bucket", default="alpha-engine-research")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(json.dumps(build_predictor_tile(args.bucket), indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
