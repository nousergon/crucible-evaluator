"""
research.py — Tile 1: Research (RC v2).

Grades the signal generator (scanner → sector teams → CIO → composite scoring +
macro + calibration). The gradable signal in the e2e artifact is each stage's
**classification precision** (did the selected names beat their baseline) — it
carries tp/fp counts, so we grade precision with a Wilson interval (proper CI +
N), which is more institutional than grading the noisy point-estimate return
*lift* that has no per-pick CI in the artifact. Lift is surfaced in the reason.

Sources: ``backtest/{date}/e2e_lift.json`` (scanner / team / cio classification),
``score_calibration.json`` (composite score→alpha Spearman calibration), ``macro_eval.json`` (macro
accuracy lift), ``portfolio_calibration.json`` (judge-vs-realized ECE). The last
three persist only from a post-2026-06-04 Saturday run (B1a/#279) — until then
they grade a precise N/A-MISSING-INPUT, never silently omitted.

Spec: ``system-report-card-revamp-260522.md`` Tile 1.
"""

from __future__ import annotations

import json
import logging

import boto3
from botocore.exceptions import ClientError

from alpha_engine_lib.quant.stats.intervals import wilson_score_interval

from grading.metric_record import build_metric
from grading.module_agg import build_tile

logger = logging.getLogger(__name__)

MODULE = "research"
_PRECISION_FLOOR = 30  # selected names needed for a confident precision read


def _get_json(s3, bucket: str, key: str) -> dict | None:
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        logger.error("S3 read failed for s3://%s/%s: %s", bucket, key, e)
        raise
    return json.loads(resp["Body"].read())


def _pick_clf(block: dict) -> tuple[dict | None, str]:
    """Prefer the canonical 21d classification; fall back to the legacy 5d.

    Returns ``(classification_block, horizon_label)``. The research picks are
    21-day theses, so selection skill is graded on ``beat_spy_21d`` precision
    (ROADMAP L4551); pre-2026-06-07 artifacts without ``classification_21d``
    grade on the legacy 5d block.
    """
    c21 = block.get("classification_21d")
    if c21:
        return c21, "21d"
    return block.get("classification"), "5d"


def _precision_metric(
    name, clf, *, criticality, source, target=0.05, red_line=-0.02,
    lift=None, lift_label=None, missing_detail=None, horizon="5d",
) -> dict:
    """A selection-precision MetricRecord graded as the EDGE over the base rate.

    A selector's precision near the cross-sectional base rate means *no edge*
    regardless of its absolute level (C4-fu). So we grade ``precision −
    base_rate`` (pp lift over the population positive rate), with a Wilson CI on
    precision shifted by the base rate. Falls back to raw precision (no edge
    framing) only when the classification block lacks the fn/tn needed to
    compute the base rate. N/A-MISSING-INPUT when the whole block is absent.

    ``horizon`` ("21d"|"5d") is the realized-outcome window the precision was
    measured over — woven into the reason so a 5d-fallback grade is never
    mistaken for the canonical 21d read.

    base_rate = (tp+fn)/(tp+fp+fn+tn); edge = precision − base_rate.
    """
    hz = f"[{horizon}] "
    if not clf or clf.get("precision") is None:
        return build_metric(
            name=name, module=MODULE, metric_type="pct", criticality=criticality,
            estimator="wilson_precision_edge", measurement_horizon=horizon,
            n_floor=_PRECISION_FLOOR, target=target, red_line=red_line, source_path=source,
            input_present=False, na_detail=missing_detail or f"{name}: no classification block in e2e_lift this cycle.",
        )
    tp, fp = int(clf.get("tp", 0)), int(clf.get("fp", 0))
    fn, tn = int(clf.get("fn", 0)), int(clf.get("tn", 0))
    n_sel = tp + fp
    n_pop = tp + fp + fn + tn
    precision = clf.get("precision")
    w = wilson_score_interval(tp, n_sel) if n_sel > 0 else {"status": "insufficient_data"}
    lift_s = f"; {lift_label} {lift:+.2%}" if (lift is not None and lift_label) else ""

    base_rate = (tp + fn) / n_pop if n_pop > 0 else None
    if base_rate is None:
        # No fn/tn → can't compute the base rate; grade raw precision (legacy).
        return build_metric(
            name=name, module=MODULE, metric_type="pct", criticality=criticality,
            estimator="wilson_precision_edge", measurement_horizon=horizon,
            value=precision, n_samples=n_sel, n_floor=_PRECISION_FLOOR, target=0.45, red_line=0.35,
            ci_low=w.get("ci_low"), ci_high=w.get("ci_high"),
            ci_method="wilson" if w.get("status") == "ok" else None, source_path=source,
            reason=(f"{hz}{name} precision = {precision:.1%} (raw — base rate unavailable; "
                    f"N={n_sel}){lift_s}." if w.get("status") == "ok" else None),
        )

    edge = precision - base_rate
    ci_low = w["ci_low"] - base_rate if w.get("status") == "ok" else None
    ci_high = w["ci_high"] - base_rate if w.get("status") == "ok" else None
    return build_metric(
        name=name, module=MODULE, metric_type="pct", criticality=criticality,
        estimator="wilson_precision_edge", measurement_horizon=horizon,
        value=edge, n_samples=n_sel, n_floor=_PRECISION_FLOOR, target=target, red_line=red_line,
        ci_low=ci_low, ci_high=ci_high,
        ci_method="wilson" if w.get("status") == "ok" else None, source_path=source,
        reason=(f"{hz}{name} edge = {edge:+.1%} (precision {precision:.1%} − base-rate {base_rate:.1%}; "
                f"Wilson CI [{ci_low:+.2f}, {ci_high:+.2f}], N={n_sel} selected) "
                f"vs target +{target:.0%} / red-line {red_line:+.0%}{lift_s}.")
        if w.get("status") == "ok" else None,
    )


def build_research_tile(bucket: str, run_date: str, s3_client=None) -> dict:
    """Build the Research tile from the e2e + research diagnostic artifacts."""
    s3 = s3_client or boto3.client("s3")
    prefix = f"backtest/{run_date}"
    e2e = _get_json(s3, bucket, f"{prefix}/e2e_lift.json")
    score_cal = _get_json(s3, bucket, f"{prefix}/score_calibration.json")
    macro = _get_json(s3, bucket, f"{prefix}/macro_eval.json")
    pcal = _get_json(s3, bucket, f"{prefix}/portfolio_calibration.json")
    e2e_src = f"s3://{bucket}/{prefix}/e2e_lift.json"
    components = []

    e2e = e2e or {}

    # All three selectors are graded on the CANONICAL 21d horizon (beat_spy_21d
    # precision) when the producer emits it — the picks are 21-day theses, and
    # the legacy 5d window collapsed precision toward the base rate (ROADMAP
    # L4551). Pre-2026-06-07 artifacts fall back to the 5d block via _pick_clf.

    # 1. scanner (supporting) — precision of quant-filter passers vs baseline.
    sl = e2e.get("scanner_lift") or {}
    sl_clf, sl_hz = _pick_clf(sl)
    sl_lift = ((sl.get("lift_21d_log") or {}).get("lift") if sl_hz == "21d"
               else sl.get("lift"))
    components.append(_precision_metric(
        "scanner", sl_clf, criticality="supporting", source=e2e_src, horizon=sl_hz,
        lift=sl_lift, lift_label="21d-alpha-lift" if sl_hz == "21d" else "return-lift",
        missing_detail="scanner: e2e_lift.json absent or has no scanner classification this cycle.",
    ))

    # 2. sector_teams_avg (critical) — pooled precision across the 6 sector teams.
    team_lift = e2e.get("team_lift")
    if isinstance(team_lift, list) and team_lift:
        # Pool whichever horizon each team carries; 21d when present (the modern
        # producer emits it for every team), else legacy 5d. A homogeneous slate
        # is the norm, so derive the tile horizon from the pooled blocks chosen.
        chosen = [_pick_clf(t) for t in team_lift]
        team_hz = "21d" if any(hz == "21d" for _, hz in chosen) else "5d"
        tp = sum(int((c or {}).get("tp", 0)) for c, _ in chosen)
        fp = sum(int((c or {}).get("fp", 0)) for c, _ in chosen)
        fn = sum(int((c or {}).get("fn", 0)) for c, _ in chosen)
        tn = sum(int((c or {}).get("tn", 0)) for c, _ in chosen)
        pooled = {"precision": (tp / (tp + fp)) if (tp + fp) else None,
                  "tp": tp, "fp": fp, "fn": fn, "tn": tn}
        components.append(_precision_metric(
            "sector_teams_avg", pooled, criticality="critical", source=e2e_src, horizon=team_hz,
        ))
    else:
        components.append(build_metric(
            name="sector_teams_avg", module=MODULE, metric_type="pct", criticality="critical",
            estimator="wilson_precision_edge", measurement_horizon="21d",
            n_floor=_PRECISION_FLOOR, target=0.45, red_line=0.35, source_path=e2e_src,
            input_present=False, na_detail="sector_teams_avg: no team_lift list in e2e_lift this cycle.",
        ))

    # 3. cio (critical) — entrant-gate precision (CIO-advanced names that won).
    cl = e2e.get("cio_lift") or {}
    cl_clf, cl_hz = _pick_clf(cl)
    cl_lift = ((cl.get("lift_21d_log") or {}).get("lift") if cl_hz == "21d"
               else (e2e.get("cio_vs_ranking") or {}).get("lift"))
    components.append(_precision_metric(
        "cio", cl_clf, criticality="critical", source=e2e_src, horizon=cl_hz,
        lift=cl_lift, lift_label="21d-alpha-lift" if cl_hz == "21d" else "vs-ranking-lift",
        missing_detail="cio: e2e_lift.json absent or has no cio classification this cycle.",
    ))

    # 3b. cio_selection_skill (critical, L4561) — does the CIO entrant gate ADVANCE
    #     the names that realize higher 21d alpha than the ones it REJECTs? The
    #     gate's whole job is selection; this measures it at the canonical horizon.
    #     Dogfoods the L4562 reliability contract: a statistically-insignificant
    #     gap (Mann-Whitney p >= 0.10) grades WATCH + reliability=low — we make the
    #     gate's skill VISIBLE and accumulate to significance, we do NOT rewrite the
    #     rubric on noise.
    sel = cl.get("selection_skill_21d") or {}
    gap = sel.get("selection_gap_21d")
    if gap is not None:
        gp = sel.get("selection_gap_p")
        n_sel = (sel.get("n_advance") or 0) + (sel.get("n_reject") or 0)
        cic = sel.get("conviction_ic_21d")
        insignificant = gp is None or gp >= 0.10
        ic_txt = f", conviction-IC {cic:+.3f}" if cic is not None else ""
        p_txt = f", MW p={gp:.3f}" if gp is not None else ""
        components.append(build_metric(
            name="cio_selection_skill", module=MODULE, metric_type="log_return", criticality="critical",
            estimator="advance_minus_reject_alpha_21d", measurement_horizon="21d",
            reliability="low" if insignificant else "high",
            value=gap, n_samples=n_sel, n_floor=60, target=0.005, red_line=0.0, source_path=e2e_src,
            status="WATCH" if insignificant else None,
            reason=(f"cio_selection_skill: ADVANCE−REJECT 21d log-alpha gap = {gap:+.4f} "
                    f"(ADVANCE {sel.get('advance_alpha_21d')} vs REJECT {sel.get('reject_alpha_21d')}, "
                    f"N={n_sel}{p_txt}{ic_txt}). "
                    + ("Not yet significant — WATCH, accumulating." if insignificant
                       else ("anti-selecting (gate advances worse names)" if gap < 0 else "adds selection value"))),
        ))
    else:
        components.append(build_metric(
            name="cio_selection_skill", module=MODULE, metric_type="log_return", criticality="critical",
            estimator="advance_minus_reject_alpha_21d", measurement_horizon="21d",
            n_floor=60, target=0.005, red_line=0.0, source_path=e2e_src, input_present=False,
            na_detail="cio_selection_skill: no selection_skill_21d block in e2e_lift this cycle "
                      "(needs cio_evaluations joined to closed-21d universe_returns).",
        ))

    # 3c. research_composite_ic (critical, L4561) — the fundamental harness question:
    #     does the blended research score the system ACTS ON predict realized 21d
    #     alpha at all? Graded on the final_score rank-IC from the layer-attribution
    #     block. Under the L4562 contract: insignificant (p >= 0.10) → WATCH +
    #     reliability=low. The per-layer ICs (combined/macro/conviction) ride in the
    #     reason so the orchestration leak is visible (e.g. macro tilt degrading the
    #     stock score), motivating the de-blending arc.
    attr = cl.get("layer_attribution_21d") or {}
    fic = attr.get("final_score_ic")
    if fic is not None:
        fp = attr.get("final_score_ic_p")
        n_at = attr.get("n")
        insig = fp is None or fp >= 0.10
        layers = ", ".join(
            f"{k}={attr.get(k + '_ic'):+.3f}" for k in ("combined_score", "macro_shift", "cio_conviction")
            if attr.get(k + "_ic") is not None
        )
        components.append(build_metric(
            name="research_composite_ic", module=MODULE, metric_type="ic", criticality="critical",
            estimator="rank_ic_vs_21d_alpha", measurement_horizon="21d",
            reliability="low" if insig else "high",
            value=fic, n_samples=n_at, n_floor=60, target=0.03, red_line=0.0, source_path=e2e_src,
            status="WATCH" if insig else None,
            reason=(f"research_composite_ic: final_score→21d-alpha rank-IC = {fic:+.3f} "
                    f"(p={fp if fp is None else round(fp,3)}, N={n_at}); per-layer [{layers}]. "
                    + ("Not yet significant — WATCH, accumulating." if insig
                       else ("no forward signal — composite does not predict 21d alpha" if fic <= 0
                             else "composite carries forward signal"))),
        ))
    else:
        components.append(build_metric(
            name="research_composite_ic", module=MODULE, metric_type="ic", criticality="critical",
            estimator="rank_ic_vs_21d_alpha", measurement_horizon="21d",
            n_floor=60, target=0.03, red_line=0.0, source_path=e2e_src, input_present=False,
            na_detail="research_composite_ic: no layer_attribution_21d block in e2e_lift this cycle.",
        ))

    # 4. composite_scoring (supporting) — does higher composite score → higher
    #    realized return? Graded on the ROBUST row-level Spearman rank
    #    correlation of score vs realized alpha (target +0.10, red-line 0.0), not
    #    the legacy binary bucket-monotonicity flag — that flag flipped RED on a
    #    single noisy quantile bucket. A statistically flat calibration
    #    (p >= 0.10) grades WATCH, not RED: no measurable signal is not the same
    #    as inverted calibration. The composite formula itself is provably
    #    monotonic in its inputs (ROADMAP L4550 — metric-quality fix, not a
    #    scoring-formula bug; the negative-edge substance lives in L4551).
    #    SUPPORTING (config#1063): this is computed at the score_performance
    #    horizon (10d — that table has no 21d column), BELOW the 21d strategy
    #    horizon, so it is a leading diagnostic rather than a critical gate. The
    #    canonical-21d composite signal is graded critically as
    #    research_composite_ic (final_score rank-IC vs 21d alpha) above; keeping
    #    BOTH critical would double-count the construct and let a sub-horizon
    #    proxy drive a Director P0 (the L4551/L4562 sub-horizon-proxy concern).
    sc_src = f"s3://{bucket}/{prefix}/score_calibration.json"
    # The score_calibration artifact is computed from the score_performance table,
    # which carries only 5d/10d/30d outcomes (NO 21d column) — the producer's
    # default horizon is 10d. Report the TRUE horizon from the artifact rather
    # than asserting the canonical 21d (config#1063: the metric was *computed* at
    # 10d but *framed* as 21d). The canonical-21d composite signal IS graded — as
    # research_composite_ic (final_score rank-IC vs 21d alpha) above — so this is
    # an honest shorter-horizon calibration diagnostic, not a mislabeled 21d gate.
    cal_horizon = (score_cal or {}).get("horizon") or "10d"
    if score_cal and score_cal.get("status") == "ok" and score_cal.get("spearman_rho") is not None:
        rho = float(score_cal["spearman_rho"])
        pval = score_cal.get("spearman_p")
        n_cal = score_cal.get("spearman_n") or score_cal.get("n")
        assessment = score_cal.get("calibration_assessment")
        beat = score_cal.get("beat_spy_pct")
        # Flat (insignificant) calibration → WATCH override; otherwise let the
        # lib derive GREEN/WATCH/RED from rho vs target/red-line.
        flat = assessment == "flat" or (pval is not None and pval >= 0.10)
        p_txt = f", p={pval:.3f}" if pval is not None else ""
        beat_txt = f", beat_spy_pct={beat:.1%}" if beat is not None else ""
        components.append(build_metric(
            name="composite_scoring", module=MODULE, metric_type="calibration", criticality="supporting",
            estimator="spearman_calibration", measurement_horizon=cal_horizon,
            value=rho, n_samples=n_cal, n_floor=30, target=0.10, red_line=0.0, source_path=sc_src,
            status="WATCH" if flat else None,
            reason=(f"composite_scoring [{cal_horizon}] Spearman rho={rho:+.3f} (score→realized-alpha rank{p_txt}); "
                    f"assessment={assessment}{beat_txt} over N={n_cal}. "
                    f"Canonical-21d composite signal graded separately as research_composite_ic."),
        ))
    elif score_cal and score_cal.get("status") == "ok" and "monotonic" in score_cal:
        # Legacy fallback: pre-2026-06-07 artifacts without the Spearman fields.
        # The legacy `monotonic` flag IS the brittle strict-binary the L4562
        # contract forbids — so we do NOT let it drive a confident GREEN/RED.
        # It grades WATCH + reliability=low (a deprecated, neutralized path that
        # only fires for stale artifacts), surfacing the binary as context only.
        mono = bool(score_cal["monotonic"])
        beat = score_cal.get("beat_spy_pct")
        components.append(build_metric(
            name="composite_scoring", module=MODULE, metric_type="calibration", criticality="supporting",
            estimator="legacy_monotonic_binary_deprecated", measurement_horizon=cal_horizon, reliability="low",
            value=1.0 if mono else 0.0, n_samples=score_cal.get("n"), n_floor=1, source_path=sc_src,
            status="WATCH",
            reason=(f"composite_scoring legacy monotonic={mono} — DEPRECATED brittle bucket binary, "
                    f"neutralized to WATCH per L4562 (awaiting a Spearman-bearing score_calibration.json); "
                    f"beat_spy_pct={beat:.1%} over N={score_cal.get('n')}." if beat is not None
                    else f"composite_scoring legacy monotonic={mono} — DEPRECATED brittle bucket binary, "
                         f"neutralized to WATCH per L4562."),
        ))
    else:
        components.append(build_metric(
            name="composite_scoring", module=MODULE, metric_type="calibration", criticality="supporting",
            estimator="spearman_calibration", measurement_horizon=cal_horizon,
            n_floor=30, source_path=sc_src, input_present=False,
            na_detail="composite_scoring: score_calibration.json absent this cycle (persists from a post-2026-06-04 Saturday run, B1a #279).",
        ))

    # 5. macro_agent (supporting) — macro accuracy lift vs realized regime (pp).
    mac_src = f"s3://{bucket}/{prefix}/macro_eval.json"
    if macro and macro.get("status") == "ok" and macro.get("accuracy_lift") is not None:
        components.append(build_metric(
            name="macro_agent", module=MODULE, metric_type="lift", criticality="supporting",
            value=macro.get("accuracy_lift"), n_samples=macro.get("n_evaluated") or macro.get("n"),
            n_floor=20, target=0.0, red_line=-3.0, source_path=mac_src,
            reason=f"macro_agent accuracy_lift={macro['accuracy_lift']:+.1f}pp, assessment={macro.get('assessment')}.",
        ))
    else:
        components.append(build_metric(
            name="macro_agent", module=MODULE, metric_type="lift", criticality="supporting",
            n_floor=20, target=0.0, red_line=-3.0, source_path=mac_src, input_present=False,
            na_detail="macro_agent: macro_eval.json absent this cycle (persists from a post-2026-06-04 Saturday run, B1a #279).",
        ))

    # 6. calibration_diagnostics (supporting) — judge-vs-realized ECE (lower better).
    pc_src = f"s3://{bucket}/{prefix}/portfolio_calibration.json"
    if pcal and pcal.get("status") == "ok" and pcal.get("ece") is not None:
        components.append(build_metric(
            name="calibration_diagnostics", module=MODULE, metric_type="calibration", criticality="supporting",
            value=pcal.get("ece"), n_samples=pcal.get("n"), n_floor=100, target=0.05, red_line=0.15,
            higher_is_better=False, source_path=pc_src,
        ))
    else:
        components.append(build_metric(
            name="calibration_diagnostics", module=MODULE, metric_type="calibration", criticality="supporting",
            n_floor=100, target=0.05, red_line=0.15, higher_is_better=False, source_path=pc_src,
            input_present=False,
            na_detail="calibration_diagnostics: portfolio_calibration.json absent this cycle (persists from a post-2026-06-04 Saturday run, B1a #279).",
        ))

    # Breadth-conditioned momentum IC (config#1140) — DIAGNOSTIC surfacing the
    # regime mechanism behind the negative funnel edge (config#1060). Grade on
    # low_breadth_ic (the actionable harm: short-momentum should not anti-predict
    # even in narrow breadth); the breadth<->IC correlation + high-breadth IC
    # ride in the reason. Diagnostic criticality — informs, does not gate.
    mri = e2e.get("momentum_regime_ic") or {}
    if mri.get("status") == "ok" and mri.get("low_breadth_ic") is not None:
        lo = mri.get("low_breadth_ic")
        hi = mri.get("high_breadth_ic")
        corr = mri.get("breadth_ic_corr")
        components.append(build_metric(
            name="momentum_regime_ic", module=MODULE, metric_type="ic", criticality="diagnostic",
            value=lo, n_samples=mri.get("n_weeks"), n_floor=6,
            target=0.0, red_line=-0.05, higher_is_better=True, source_path=e2e_src,
            measurement_horizon="21d",
            reason=(
                f"momentum_regime_ic: tech_score momentum IC = {lo:+.3f} in low-breadth weeks vs "
                f"{('%+.3f' % hi) if hi is not None else 'n/a'} in high-breadth "
                f"(breadth<->IC corr {('%+.3f' % corr) if corr is not None else 'n/a'}, "
                f"{mri.get('n_weeks')} weeks). Negative low-breadth IC = short-momentum mean-reverts "
                f"in narrow tape — the regime mechanism behind the negative funnel edge "
                f"(config#1060); validation target for the Phase-2 neutralization (config#1142)."
            ),
        ))
    else:
        components.append(build_metric(
            name="momentum_regime_ic", module=MODULE, metric_type="ic", criticality="diagnostic",
            n_floor=6, target=0.0, red_line=-0.05, higher_is_better=True, source_path=e2e_src,
            input_present=False,
            na_detail=(
                f"momentum_regime_ic: no ok block in e2e_lift this cycle "
                f"(status={mri.get('status')!r}); needs the backtester producer (config#1140) "
                f"+ >=4 realized weekly cohorts."
            ),
        ))

    # 7-9. Agent-quality producer components (config Batch A #1149). These read
    #      backtest/{date}/agent_quality.json — the same artifact the Agent tile
    #      reads — but grade the research-output-quality axis (judge pass-rate,
    #      pillar coverage, signal volume). Absent → precise N/A-MISSING-INPUT
    #      naming the producer, self-activating on the first agent_quality.json.
    aq_key = f"{prefix}/agent_quality.json"
    aq_src = f"s3://{bucket}/{aq_key}"
    aq = _get_json(s3, bucket, aq_key)

    def _aq_block(key: str) -> dict | None:
        if not aq or aq.get("status") != "ok":
            return None
        blk = aq.get(key)
        return blk if isinstance(blk, dict) and blk.get("value") is not None else None

    # 7. judge_rubric_pass_rate (supporting) — % of judge evals clearing the
    #    rubric pass threshold (higher better).
    blk = _aq_block("judge_rubric_pass_rate")
    if blk is not None:
        components.append(build_metric(
            name="judge_rubric_pass_rate", module=MODULE, metric_type="pct", criticality="supporting",
            value=blk["value"], n_samples=blk.get("n"), n_floor=10, target=0.85, red_line=0.60,
            source_path=aq_src,
            reason=f"judge_rubric_pass_rate = {blk['value']:.1%} (N={blk.get('n')} evals) vs target 85% / red-line 60%.",
        ))
    else:
        components.append(build_metric(
            name="judge_rubric_pass_rate", module=MODULE, metric_type="pct", criticality="supporting",
            n_floor=10, target=0.85, red_line=0.60, source_path=aq_src, input_present=False,
            na_detail="judge_rubric_pass_rate: agent_quality.json absent or no value this cycle (research agent-quality producer, config#1149).",
        ))

    # 8. pillar_emit_coverage (supporting) — % of universe entries carrying a
    #    pillar_assessment (higher better).
    blk = _aq_block("pillar_emit_coverage")
    if blk is not None:
        components.append(build_metric(
            name="pillar_emit_coverage", module=MODULE, metric_type="pct", criticality="supporting",
            value=blk["value"], n_samples=blk.get("n"), n_floor=10, target=0.90, red_line=0.50,
            source_path=aq_src,
            reason=f"pillar_emit_coverage = {blk['value']:.1%} (N={blk.get('n')} universe entries) vs target 90% / red-line 50%.",
        ))
    else:
        components.append(build_metric(
            name="pillar_emit_coverage", module=MODULE, metric_type="pct", criticality="supporting",
            n_floor=10, target=0.90, red_line=0.50, source_path=aq_src, input_present=False,
            na_detail="pillar_emit_coverage: agent_quality.json absent or no value this cycle (research agent-quality producer, config#1149).",
        ))

    # 9. signal_volume_adequacy (diagnostic) — count of finalized signals vs the
    #    adequacy floor (higher better).
    blk = _aq_block("signal_volume_adequacy")
    if blk is not None:
        components.append(build_metric(
            name="signal_volume_adequacy", module=MODULE, metric_type="count", criticality="diagnostic",
            value=blk["value"], n_samples=blk.get("n"), n_floor=1, target=20.0, red_line=8.0,
            source_path=aq_src,
            reason=f"signal_volume_adequacy = {blk['value']:.0f} finalized signals vs target 20 / red-line 8.",
        ))
    else:
        components.append(build_metric(
            name="signal_volume_adequacy", module=MODULE, metric_type="count", criticality="diagnostic",
            n_floor=1, target=20.0, red_line=8.0, source_path=aq_src, input_present=False,
            na_detail="signal_volume_adequacy: agent_quality.json absent or no value this cycle (research agent-quality producer, config#1149).",
        ))

    return build_tile(MODULE, components)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Build the Research tile from e2e + diagnostic artifacts.")
    parser.add_argument("--bucket", default="alpha-engine-research")
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(json.dumps(build_research_tile(args.bucket, args.date), indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
