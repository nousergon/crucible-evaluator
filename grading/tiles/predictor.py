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
from datetime import UTC, datetime

import boto3
from botocore.exceptions import ClientError

from alpha_engine_lib.quant.stats.intervals import bootstrap_ci

from grading.artifacts import get_json_windowed
from grading.metric_record import build_metric
from grading.module_agg import build_tile

logger = logging.getLogger(__name__)

MODULE = "predictor"
LATEST_KEY = "predictor/metrics/latest.json"
MANIFEST_KEY = "predictor/weights/meta/manifest.json"
SLIM_CACHE_PREFIX = "predictor/price_cache_slim/"

# The DIRECTIONAL L1 model outputs — the three columns in META_FEATURES that are
# themselves OOS L1 predictions trained against the SAME signed alpha label the
# L2 targets. These are the apples-to-apples comparison set for ensemble lift.
# Excluded from this set: the walk-forward `volatility_median_ic`, because the
# volatility L1 is trained/scored on abs(return) MAGNITUDE (not signed alpha) —
# its standalone surface here is `expected_move`'s directional alpha-IC, which
# IS directional and IS included. (config#1062 false-RED fix.)
_DIRECTIONAL_L1_FEATURES = ("expected_move", "research_calibrator_prob", "momentum_score", "momentum")


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


def build_predictor_tile(
    bucket: str, run_date: str | None = None, s3_client=None, *, as_of: datetime | None = None
) -> dict:
    """Build the Predictor tile from the predictor's metrics + weights manifest.

    ``run_date`` (YYYY-MM-DD) enables the cross-tile veto_gate_precision read
    from ``backtest/{run_date}/veto_analysis.json``; when omitted that one
    component grades a transparent N/A.
    """
    s3 = s3_client or boto3.client("s3")
    as_of = as_of or datetime.now(UTC)
    latest = _get_json(s3, bucket, LATEST_KEY)
    manifest = _get_json(s3, bucket, MANIFEST_KEY)
    latest_src = f"s3://{bucket}/{LATEST_KEY}"
    manifest_src = f"s3://{bucket}/{MANIFEST_KEY}"

    if latest is None and manifest is None:
        miss = build_metric(
            name="meta_l2_ic", module=MODULE, metric_type="ic", criticality="critical",
            estimator="rank_ic", measurement_horizon="21d",
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
        estimator="rank_ic", measurement_horizon="21d",
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
        estimator="rank_ic_oos", measurement_horizon="21d",
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
    #    config#1062: the OLD code compared the directional meta CPCV IC against
    #    max(momentum_wf, volatility_wf, research_calibrator) — but
    #    `volatility_median_ic` is a MAGNITUDE IC (the volatility L1 trains/scores
    #    on abs(return), predictor meta_trainer.py:1521-1530), not signed alpha.
    #    Subtracting a magnitude-IC from a directional-IC is meaningless and
    #    produced a false −0.296 "destroys signal" RED. The apples-to-apples read
    #    is the manifest's `meta_l1_standalone_alpha_ic` — each L1 output's IC vs
    #    the SAME signed alpha label the L2 targets. We compare the meta CPCV IC
    #    against the best DIRECTIONAL standalone L1 alpha-IC.
    standalone = manifest.get("meta_l1_standalone_alpha_ic")
    directional_l1_ics: dict[str, float] = {}
    if isinstance(standalone, dict) and standalone.get("status") not in ("not_run", "error"):
        for feat in _DIRECTIONAL_L1_FEATURES:
            entry = standalone.get(feat)
            if isinstance(entry, dict):
                xic = entry.get("xsec_ic")
                if isinstance(xic, (int, float)):
                    directional_l1_ics[feat] = float(xic)
    lift = None
    lift_reason = None
    lift_present = False
    lift_na = None
    if cpcv_mean is None:
        lift_na = "ensemble_lift: needs the leak-free meta CPCV IC (absent this cycle)."
    elif directional_l1_ics:
        best_feat = max(directional_l1_ics, key=lambda k: directional_l1_ics[k])
        best_l1 = directional_l1_ics[best_feat]
        lift = cpcv_mean - best_l1
        lift_present = True
        lift_reason = (
            f"ensemble_lift_over_best_l1 = {lift:+.3g} (leak-free meta {cpcv_mean:.3g} − best "
            f"DIRECTIONAL standalone L1 {best_feat}={best_l1:+.3g}) vs target 0.01 / red-line "
            f"-0.01. Compares signed-alpha-IC vs signed-alpha-IC; the volatility L1's "
            f"MAGNITUDE walk-forward IC ({vol_ic if vol_ic is not None else 'n/a'}) is "
            f"deliberately EXCLUDED — it scores abs(return), not directional alpha (config#1062)."
        )
    else:
        # Manifest lacks the standalone-alpha-IC field (older training run, or it
        # errored/not-run). Do NOT fall back to the magnitude WF IC — that is the
        # exact false-RED bug. Surface honest N/A instead.
        lift_na = (
            "ensemble_lift: directional `meta_l1_standalone_alpha_ic` absent/not-run in the "
            "manifest this cycle. Refusing to fall back to the volatility MAGNITUDE walk-forward "
            "IC (config#1062 false-RED). Re-runs once the standalone alpha-IC diagnostic is present."
        )
    components.append(build_metric(
        name="ensemble_lift_over_best_l1", module=MODULE, metric_type="ic", criticality="critical",
        estimator="ic_delta", measurement_horizon="21d",
        value=lift, n_samples=n_combos, n_floor=10, target=0.01, red_line=-0.01,
        source_path=manifest_src, input_present=lift_present, reason=lift_reason,
        na_detail=lift_na or "ensemble_lift: needs the leak-free meta IC and a directional standalone L1 alpha-IC.",
    ))

    # 6. confidence_calibration_ece (critical) — lower is better.
    cc = latest.get("confidence_calibration") or {}
    ece = cc.get("ece_after")
    components.append(build_metric(
        name="confidence_calibration_ece", module=MODULE, metric_type="calibration", criticality="critical",
        estimator="expected_calibration_error",
        value=ece, n_samples=cc.get("n_samples"), n_floor=100, target=0.05, red_line=0.15,
        higher_is_better=False, source_path=latest_src, input_present=ece is not None,
    ))

    # 7. output_distribution_gate (critical) — the predictor's own pass/fail gate.
    odg = latest.get("output_distribution_gate") or {}
    if "passed" in odg:
        passed = bool(odg["passed"])
        components.append(build_metric(
            name="output_distribution_gate", module=MODULE, metric_type="pct", criticality="critical",
            estimator="distribution_gate",
            value=1.0 if passed else 0.0, n_samples=1, n_floor=1, source_path=latest_src,
            status="GREEN" if passed else "RED",
            reason=f"output_distribution_gate {'PASS' if passed else 'FAIL'}: {odg.get('reason', '')}".strip(),
        ))
    else:
        components.append(build_metric(
            name="output_distribution_gate", module=MODULE, metric_type="pct", criticality="critical",
            estimator="distribution_gate",
            n_floor=1, source_path=latest_src, input_present=False,
            na_detail="output_distribution_gate: no gate result in latest.json this cycle.",
        ))

    # veto_gate_precision (supporting) — precision of the veto gate AT THE LIVE
    # threshold: of the names the gate vetoed, the fraction that actually
    # underperformed (did not beat SPY). Cross-tile read from the backtester's
    # backtest/{run_date}/veto_analysis.json (config#859 — was an unwired N/A).
    # 10d-horizon measurement (beat_spy_10d), hence supporting not critical.
    # Windowed resolution (config#1190): freshest within the trailing window.
    va, _, _, _veto_key = (
        get_json_windowed(s3, bucket, "backtest/{date}/veto_analysis.json", run_date)
        if run_date else (None, None, None, None)
    )
    veto_src = f"s3://{bucket}/{_veto_key}" if _veto_key else latest_src
    va_match = None
    if va:
        cur = va.get("current_threshold")
        entries = [e for e in (va.get("thresholds") or []) if e.get("precision") is not None]
        if cur is not None and entries:
            va_match = min(entries, key=lambda e: abs((e.get("confidence") or 0) - cur))
    if va_match is not None:
        ci = va_match.get("precision_ci_95")
        ci_low = ci[0] if isinstance(ci, (list, tuple)) and len(ci) >= 2 else None
        ci_high = ci[1] if isinstance(ci, (list, tuple)) and len(ci) >= 2 else None
        prec = va_match["precision"]
        n_v = va_match.get("n_vetoes")
        conf = va_match.get("confidence") or 0.0
        components.append(build_metric(
            name="veto_gate_precision", module=MODULE, metric_type="pct", criticality="supporting",
            estimator="wilson_precision", measurement_horizon="10d",
            value=prec, n_samples=n_v, n_floor=30, target=0.60, red_line=0.40,
            ci_low=ci_low, ci_high=ci_high, ci_method="wilson" if ci_low is not None else None,
            source_path=veto_src,
            reason=(f"veto_gate_precision [10d] = {prec:.1%} at the live veto threshold {conf:.2f} "
                    f"({va_match.get('true_negatives')}/{n_v} vetoed names underperformed) "
                    f"vs target 60% / red-line 40%."),
        ))
    else:
        if run_date is None:
            na = "veto_gate_precision: run_date not provided to the predictor tile; cross-tile veto_analysis.json read skipped."
        elif va is None:
            na = f"veto_gate_precision: veto_analysis.json absent in the trailing window ending {run_date}."
        else:
            na = f"veto_gate_precision: veto_analysis.json has no usable precision at the live threshold (status={va.get('status')}) this cycle."
        components.append(build_metric(
            name="veto_gate_precision", module=MODULE, metric_type="pct", criticality="supporting",
            estimator="wilson_precision", measurement_horizon="10d",
            n_floor=30, target=0.60, red_line=0.40, source_path=veto_src, input_present=False,
            na_detail=na,
        ))
    # inference_coverage (critical) — fraction of the intended tradable universe
    # that got a prediction. The producer persists the denominator + covered
    # count (config#1075); we grade covered/universe ∈ [0,1]. Honest N/A until
    # the producer field lands (or when the universe is empty this cycle).
    n_universe = latest.get("n_universe")
    n_covered = latest.get("n_universe_covered")
    n_preds_today = latest.get("n_predictions_today")
    if isinstance(n_universe, int) and n_universe > 0 and isinstance(n_covered, int):
        coverage = n_covered / n_universe
        components.append(build_metric(
            name="inference_coverage", module=MODULE, metric_type="pct", criticality="critical",
            estimator="coverage_proportion",
            value=coverage, n_samples=n_universe, n_floor=1, target=0.95, red_line=0.80,
            source_path=latest_src,
            reason=(f"inference_coverage = {coverage:.1%} ({n_covered}/{n_universe} tradable-universe "
                    f"tickers predicted) vs target 95% / red-line 80%."),
        ))
    else:
        components.append(build_metric(
            name="inference_coverage", module=MODULE, metric_type="pct", criticality="critical",
            estimator="coverage_proportion",
            value=None, n_floor=1, target=0.95, red_line=0.80, source_path=latest_src, input_present=False,
            na_detail=(f"inference_coverage: n_predictions_today={n_preds_today} observed, but the "
                       "tradable-universe denominator (n_universe) is absent/zero in latest.json this "
                       "cycle (predictor config#1075 producer field not present yet)."),
        ))
    # slim_cache_freshness (supporting) — days since the 2y inference slim-cache
    # last refreshed (weekly cadence). Sourced directly from the slim-cache
    # objects' LastModified (config#859 — was an unwired N/A; mirrors the
    # substrate tile's price_cache_freshness).
    slim_src = f"s3://{bucket}/{SLIM_CACHE_PREFIX}"
    slim_mtime = _latest_mtime(s3, bucket, SLIM_CACHE_PREFIX)
    if slim_mtime is not None:
        slim_age_d = (as_of - slim_mtime).total_seconds() / 86400.0
        components.append(build_metric(
            name="slim_cache_freshness", module=MODULE, metric_type="duration", criticality="supporting",
            estimator="freshness_age",
            value=slim_age_d, n_samples=1, n_floor=1, target=7.0, red_line=14.0,
            higher_is_better=False, source_path=slim_src,
            reason=f"slim_cache_freshness = {slim_age_d:.1f}d since the inference slim-cache last refreshed vs target 7d / red-line 14d.",
        ))
    else:
        components.append(build_metric(
            name="slim_cache_freshness", module=MODULE, metric_type="duration", criticality="supporting",
            estimator="freshness_age",
            n_floor=1, target=7.0, red_line=14.0, higher_is_better=False, source_path=slim_src,
            input_present=False,
            na_detail="slim_cache_freshness: no objects under predictor/price_cache_slim/ to date-stamp.",
        ))

    # 11. feature_drift_ks (diagnostic) — inference-vs-training KS (config#859).
    #     Headline = max_ks (worst-feature drift); lower is better.
    #     target 0.10 / red-line 0.25. Block emitted by the predictor inference
    #     stage once a training reference exists.
    fdk = latest.get("feature_drift_ks")
    if fdk and fdk.get("max_ks") is not None:
        max_ks = float(fdk["max_ks"])
        per_feat = fdk.get("per_feature") or {}
        worst = next(iter(per_feat.items()), None)
        components.append(build_metric(
            name="feature_drift_ks", module=MODULE, metric_type="ratio", criticality="diagnostic",
            estimator="ks_2samp_max", value=max_ks, n_samples=int(fdk.get("n_samples") or 0),
            n_floor=30, target=0.10, red_line=0.25, higher_is_better=False, source_path=latest_src,
            reason=(
                f"feature_drift_ks: worst-feature inference-vs-training KS = {max_ks:.3f} "
                f"across {fdk.get('n_features')} cross-sectional features"
                + (f" (worst: {worst[0]}={worst[1]:.3f})" if worst else "")
                + f", mean {fdk.get('mean_ks')}, vs target 0.10 / red-line 0.25."
            ),
        ))
    else:
        components.append(build_metric(
            name="feature_drift_ks", module=MODULE, metric_type="ratio", criticality="diagnostic",
            n_floor=30, target=0.10, red_line=0.25, higher_is_better=False, source_path=latest_src,
            input_present=False,
            na_detail=(
                "feature_drift_ks: feature_drift_ks block absent in metrics/latest.json "
                "(no training reference yet, or inference compute degraded — config#859)."
            ),
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
