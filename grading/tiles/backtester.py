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
  - apply_loop_health : per-loop outcomes from config/apply_audit/{date}.json —
    silent non-promotion of the four auto-apply loops (config#1841)

sample_size_adequacy, optimizer_churn and walk_forward_stability are WIRED
(config#1151 Batch C) — they grade from their backtester producer artifacts
(backtest/{date}/{sample_size,optimizer_churn,walk_forward_stability}.json) and
fall to a transparent N/A only when the artifact is absent. backtest_vs_live_parity
still needs a synthetic-backtest-IC-vs-live-IC drift series not yet persisted →
transparent N/A-NOT-IMPL.

Spec: ``system-report-card-revamp-260522.md`` Tile 4.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

import boto3
from botocore.exceptions import ClientError

from grading.artifacts import get_json_windowed
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
    # Windowed resolution (config#1190): freshest within the trailing window.
    grading, _, _, _grading_key = get_json_windowed(s3, bucket, "backtest/{date}/grading.json", run_date)
    g_src = f"s3://{bucket}/{_grading_key}" if _grading_key else src("grading.json")
    if grading:
        cov, graded, total = _coverage(grading)
        components.append(build_metric(
            name="evaluator_coverage", module=MODULE, metric_type="pct", criticality="critical",
            estimator="coverage_proportion", measurement_horizon="trailing_4w",
            value=cov, n_samples=total, n_floor=1, target=0.95, red_line=0.80, source_path=g_src,
            reason=(f"evaluator_coverage = {cov:.0%} ({graded}/{total} leaf components graded, "
                    f"non-N/A) vs target 95% / red-line 80%." if cov is not None else None),
            na_detail="evaluator_coverage: grading.json has no gradable components this cycle.",
        ))
    else:
        components.append(build_metric(
            name="evaluator_coverage", module=MODULE, metric_type="pct", criticality="critical",
            estimator="coverage_proportion", measurement_horizon="trailing_4w",
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
    parity, _, _, _parity_key = get_json_windowed(s3, bucket, "backtest/{date}/parity_report.json", run_date)
    p_src = f"s3://{bucket}/{_parity_key}" if _parity_key else src("parity_report.json")
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
    attr, _, _, _attr_key = get_json_windowed(s3, bucket, "backtest/{date}/attribution.json", run_date)
    a_src = f"s3://{bucket}/{_attr_key}" if _attr_key else src("attribution.json")
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

    # 6. apply_loop_health (critical, config#1841) — did the four backtester
    #    auto-apply loops (scoring_weights / executor_params / predictor_params /
    #    research_params) actually apply anything, or are they silently stuck?
    #    Consumes the apply-audit producer artifact
    #    (config/apply_audit/{date}.json, schema_version 1 — FROZEN contract,
    #    canonical fixture tests/fixtures/apply_audit_v1.json). Live motivation:
    #    scoring_weights + predictor_params never promoted for ~2 months and
    #    nothing surfaced it (config#1841) — this component makes that silence
    #    impossible. Per-loop state machine:
    #      error                                   → RED
    #      blocked, consecutive_blocked_weeks >= 4 → RED
    #      blocked, weeks in [2, 3]                → WATCH
    #      promoted / insufficient_data / disabled
    #        / blocked < 2 weeks                   → healthy
    #      unrecognized outcome                    → RED (producer contract
    #        drift — fail loud, never silently green)
    #    value = number of unhealthy (RED+WATCH) loops, so the card sorts on it;
    #    the reason names every non-green loop with outcome + blocked_by slugs
    #    + consecutive weeks so the Director digest carries the specifics.
    aa, _, _, _aa_key = get_json_windowed(s3, bucket, "config/apply_audit/{date}.json", run_date)
    aa_src = f"s3://{bucket}/{_aa_key}" if _aa_key else f"s3://{bucket}/config/apply_audit/{run_date}.json"
    _HEALTHY_OUTCOMES = ("promoted", "insufficient_data", "disabled")
    loops = (aa or {}).get("loops")
    if aa is not None and not (isinstance(loops, dict) and loops):
        # Present-but-malformed artifact (no usable "loops" block) — producer
        # contract drift on secondary observability: WARN + specific N/A, never
        # a crash and never a silent GREEN (the critical N/A forces tile WATCH).
        logger.warning("apply_audit artifact at %s has no usable 'loops' block", aa_src)
    if aa is not None and isinstance(loops, dict) and loops:
        red: list[str] = []
        watch: list[str] = []
        healthy: list[str] = []
        for loop_name in sorted(loops):
            entry = loops[loop_name] or {}
            outcome = entry.get("outcome")
            weeks = int(entry.get("consecutive_blocked_weeks") or 0)
            if outcome == "blocked":
                slugs = ",".join(entry.get("blocked_by") or []) or "unspecified"
                desc = f"{loop_name} blocked {weeks}w [{slugs}]"
                if weeks >= 4:
                    red.append(desc)
                elif weeks >= 2:
                    watch.append(desc)
                else:
                    healthy.append(f"{loop_name}=blocked({weeks}w)")
            elif outcome == "error":
                detail = str(entry.get("detail") or "").strip()
                red.append(f"{loop_name} error" + (f" ({detail[:120]})" if detail else ""))
            elif outcome in _HEALTHY_OUTCOMES:
                healthy.append(f"{loop_name}={outcome}")
            else:
                red.append(f"{loop_name} unrecognized outcome {outcome!r} (producer contract drift)")
        unhealthy = red + watch
        status = "RED" if red else ("WATCH" if watch else "GREEN")
        if unhealthy:
            reason = (f"apply_loop_health: {len(unhealthy)}/{len(loops)} auto-apply loops unhealthy — "
                      f"{'; '.join(unhealthy)} (as_of {aa.get('as_of')}).")
        else:
            reason = (f"apply_loop_health: all {len(loops)} auto-apply loops healthy "
                      f"({', '.join(healthy)}) as_of {aa.get('as_of')}.")
        components.append(build_metric(
            name="apply_loop_health", module=MODULE, metric_type="count", criticality="critical",
            estimator="per_loop_outcome_state_machine", measurement_horizon="per_cycle",
            value=float(len(unhealthy)), n_samples=len(loops), n_floor=1,
            higher_is_better=False, source_path=aa_src, status=status, reason=reason,
        ))
    else:
        components.append(build_metric(
            name="apply_loop_health", module=MODULE, metric_type="count", criticality="critical",
            n_floor=1, higher_is_better=False, source_path=aa_src, input_present=False,
            na_detail=(
                f"apply_loop_health: no usable config/apply_audit/{{date}}.json in the trailing "
                f"window ending {run_date}"
                + (" (artifact present but its 'loops' block is absent/malformed)" if aa is not None else "")
                + " — needs the backtester apply-audit producer (config#1841); "
                "self-activates on first emission."
            ),
        ))

    # sample_size_adequacy (critical, config#1151 Batch C) — are the per-cycle
    # finalized-signal counts feeding the accuracy/attribution grades ABOVE the
    # floor each needs, or are those grades computed on too few samples to mean
    # anything? Graded on the WEAKEST-LINK adequacy ratio (min n/floor across
    # analyses) from the backtester producer. ratio>=1.0 → GREEN (well-powered),
    # 0.5<=ratio<1.0 → WATCH (building), <0.5 → RED (grades are under-powered noise).
    ss, _, _, _ss_key = get_json_windowed(s3, bucket, "backtest/{date}/sample_size.json", run_date)
    ss_src = f"s3://{bucket}/{_ss_key}" if _ss_key else f"s3://{bucket}/{prefix}/sample_size.json"
    if ss and ss.get("status") == "ok" and ss.get("adequacy_ratio") is not None:
        ratio = ss["adequacy_ratio"]
        weakest = ss.get("weakest_analysis")
        per = ss.get("per_analysis") or {}
        wk = per.get(weakest) or {}
        breakdown = "; ".join(f"{k} {v.get('n')}/{v.get('floor')}" for k, v in per.items())
        verdict = ("well-powered" if ratio >= 1.0
                   else "building — grades under-powered, WATCH" if ratio >= 0.5
                   else "severely under-powered — accuracy/attribution grades are noise")
        components.append(build_metric(
            name="sample_size_adequacy", module=MODULE, metric_type="ratio", criticality="critical",
            estimator="weakest_link_finalized_signal_count_vs_floor", measurement_horizon="per_cycle",
            value=ratio, n_samples=wk.get("n"), n_floor=1, target=1.0, red_line=0.5,
            higher_is_better=True, source_path=ss_src,
            reason=(f"sample_size_adequacy: weakest-link adequacy = {ratio:.2f} "
                    f"({weakest}: n={wk.get('n')} vs floor {wk.get('floor')}; {breakdown}). {verdict}"),
        ))
    else:
        components.append(build_metric(
            name="sample_size_adequacy", module=MODULE, metric_type="ratio", criticality="critical",
            n_floor=1, target=1.0, red_line=0.5, higher_is_better=True, source_path=ss_src,
            input_present=False,
            na_detail=(f"sample_size_adequacy: no ok sample_size.json in the trailing window ending {run_date} "
                       f"(status={(ss or {}).get('status')!r}); needs the backtester producer (config#1151)."),
        ))

    # optimizer_churn (critical, config#1151 Batch C) — how hard did the weight
    # optimizer push against its single-change guardrail this cycle? The producer
    # emits churn_ratio = max|Δweight| / guardrail_cap; <1.0 = within guardrails.
    # Lower is better: target 0.8 (well inside the cap → GREEN), red-line 1.0
    # (at/over the cap → the tuner is fighting its own guardrail → RED).
    oc, _, _, _oc_key = get_json_windowed(s3, bucket, "backtest/{date}/optimizer_churn.json", run_date)
    oc_src = f"s3://{bucket}/{_oc_key}" if _oc_key else f"s3://{bucket}/{prefix}/optimizer_churn.json"
    if oc and oc.get("status") == "ok" and oc.get("churn_ratio") is not None:
        cr = float(oc["churn_ratio"])
        within = bool(oc.get("within_guardrails"))
        components.append(build_metric(
            name="optimizer_churn", module=MODULE, metric_type="ratio", criticality="critical",
            estimator="max_abs_weight_change_over_guardrail_cap", measurement_horizon="per_cycle",
            value=cr, n_samples=oc.get("n_params_changed"), n_floor=1,
            target=0.8, red_line=1.0, higher_is_better=False, source_path=oc_src,
            reason=(f"optimizer_churn = {cr:.2f} (max |Δ|={oc.get('max_abs_change')} on "
                    f"{oc.get('max_change_param')!r} vs cap {oc.get('guardrail_cap')}; "
                    f"{oc.get('n_params_changed')} params moved) — "
                    f"{'within guardrails' if within else 'AT/OVER the guardrail cap'} "
                    f"vs target 0.8 / red-line 1.0."),
        ))
    else:
        components.append(build_metric(
            name="optimizer_churn", module=MODULE, metric_type="ratio", criticality="critical",
            n_floor=1, target=0.8, red_line=1.0, higher_is_better=False, source_path=oc_src,
            input_present=False,
            na_detail=(f"optimizer_churn: no ok optimizer_churn.json in the trailing window ending {run_date} "
                       f"(status={(oc or {}).get('status')!r}); the optimizer had no usable recommendation "
                       f"this cycle (config#1151)."),
        ))

    # backtest_vs_live_parity — still genuinely unwired (no producer artifact).
    components.append(build_metric(
        name="backtest_vs_live_parity", module=MODULE, metric_type="ratio", criticality="critical",
        n_floor=1, source_path=f"s3://{bucket}/{prefix}/", implemented=False,
        na_detail="backtest_vs_live_parity: needs the predictor synthetic-backtest IC vs live L2 IC drift series — not yet persisted.",
    ))

    # walk_forward_stability (supporting, config#1151 Batch C) — do the weekly
    # weight recommendations converge or oscillate? The producer emits
    # stability_ratio = 1 - reversals/max_possible_reversals over the loaded
    # weekly window; 1.0 = monotone/converging, 0.0 = fully oscillating. Higher
    # is better: target 0.8 (stable → GREEN), red-line 0.5 (half the steps
    # reverse → drifting → RED).
    wf, _, _, _wf_key = get_json_windowed(s3, bucket, "backtest/{date}/walk_forward_stability.json", run_date)
    wf_src = f"s3://{bucket}/{_wf_key}" if _wf_key else f"s3://{bucket}/{prefix}/walk_forward_stability.json"
    if wf and wf.get("status") == "ok" and wf.get("stability_ratio") is not None:
        sr = float(wf["stability_ratio"])
        components.append(build_metric(
            name="walk_forward_stability", module=MODULE, metric_type="ratio", criticality="supporting",
            estimator="one_minus_reversal_fraction_over_weekly_window", measurement_horizon="multi_week",
            value=sr, n_samples=wf.get("weeks_loaded"), n_floor=1,
            target=0.8, red_line=0.5, higher_is_better=True, source_path=wf_src,
            reason=(f"walk_forward_stability = {sr:.2f} ({wf.get('n_reversals')} reversals of "
                    f"{wf.get('max_possible_reversals')} possible over {wf.get('weeks_loaded')} prior weeks; "
                    f"stable={wf.get('stable')}) vs target 0.8 / red-line 0.5."),
        ))
    else:
        components.append(build_metric(
            name="walk_forward_stability", module=MODULE, metric_type="ratio", criticality="supporting",
            n_floor=1, target=0.8, red_line=0.5, higher_is_better=True, source_path=wf_src,
            input_present=False,
            na_detail=(f"walk_forward_stability: no ok walk_forward_stability.json in the trailing window ending "
                       f"{run_date} (status={(wf or {}).get('status')!r}); need >=2 prior weeks of weight history "
                       f"to judge drift (config#1151)."),
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
