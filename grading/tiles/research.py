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

from nousergon_lib import contracts
from nousergon_lib.quant.stats.intervals import wilson_score_interval

from grading.artifacts import RESEARCH_ARTIFACT_MAX_AGE_DAYS, artifact_is_stale, get_json_windowed
from grading.history import CardHistory
from grading.metric_record import build_metric
from grading.thresholds.registry import resolve as resolve_band
from grading.module_agg import build_tile
from grading.units import COUNT_SIGNALS, ECE, FRACTION, LOG_RETURN_21D, PCT, RANK_IC, SCORE_0_1, SPEARMAN_RHO

logger = logging.getLogger(__name__)

MODULE = "research"
_PRECISION_FLOOR = 30  # selected names needed for a confident precision read

# config#2318: since the 2026-06-29 attractiveness champion-feed cutover, the
# scanner block in e2e_lift.json (``scanner_lift`` / ``scanner_evaluations.
# quant_filter_pass``) reflects only the retired tech_score baseline gate — it
# no longer feeds live candidate generation. Label the scanner tile component
# with this arm explicitly so it is never mistaken for a live-scanner read
# (matches the label crucible-backtester's ``analysis/end_to_end.py`` now
# stamps onto ``scanner_lift`` / ``actual_scanner_pass``).
_SCANNER_METRIC_ARM = "tech_score_baseline (retired from live feed 2026-06-29)"

# config-I2994: the live champion feed's ranking score IS the scanner universe-
# board attractiveness_score (copied verbatim into signals.json by crucible-
# research scoring/signals_envelope.py). ``attractiveness_ic`` (attractiveness_
# eval.json ``composite_ic`` vs realized 21d alpha) is therefore the honestly-
# named live-arm score-IC that REPLACES the retired, CIO-sourced
# ``research_composite_ic`` — promoted supporting→critical (Brian ruling: it
# measures the live feed's ranking power; a critical WATCH while it accumulates
# to the 8-week floor is the honest read, not pollution).
_LIVE_ARM_SCANNER = (
    "scanner_attractiveness (live champion feed — universe-board "
    "attractiveness_score, config-I2994)"
)


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
    name, clf, *, criticality, source,
    lift=None, lift_label=None, missing_detail=None, horizon="5d",
    history: CardHistory | None = None, arm=None,
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

    ``history`` (config#1836) threads cross-cycle ``trend_4w``/``trend_13w``
    from prior report cards onto whichever value branch grades.

    ``arm`` (config#2318) optionally labels which measurement arm this metric
    was computed from (e.g. the retired tech_score baseline scanner, post the
    2026-06-29 champion-feed cutover). Passed through to the ``MetricRecord``
    (extra field) and woven into the reason string so the Director's evidence
    walker — which reads ``status_reason``/``reason`` generically — inherits
    the label with no director/-side changes needed.
    """

    def _tr(value):
        return history.trends_for(MODULE, name, value) if history is not None else {}

    _band = resolve_band(MODULE, name)
    hz = f"[{horizon}] "
    arm_s = f" [arm: {arm}]" if arm else ""
    if not clf or clf.get("precision") is None:
        return build_metric(
            name=name, module=MODULE, metric_type="pct", criticality=criticality,
            estimator="wilson_precision_edge", measurement_horizon=horizon,
            n_floor=_PRECISION_FLOOR, source_path=source,
            input_present=False, na_detail=missing_detail or f"{name}: no classification block in e2e_lift this cycle.",
            arm=arm,
            **_tr(None),
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
            value=precision, n_samples=n_sel, n_floor=_PRECISION_FLOOR, band="raw_precision",
            unit=FRACTION,
            ci_low=w.get("ci_low"), ci_high=w.get("ci_high"),
            ci_method="wilson" if w.get("status") == "ok" else None, source_path=source,
            reason=(f"{hz}{name} precision = {precision:.1%} (raw — base rate unavailable; "
                    f"N={n_sel}){lift_s}{arm_s}." if w.get("status") == "ok" else None),
            arm=arm,
            **_tr(precision),
        )

    edge = precision - base_rate
    ci_low = w["ci_low"] - base_rate if w.get("status") == "ok" else None
    ci_high = w["ci_high"] - base_rate if w.get("status") == "ok" else None
    return build_metric(
        name=name, module=MODULE, metric_type="pct", criticality=criticality,
        estimator="wilson_precision_edge", measurement_horizon=horizon,
        value=edge, n_samples=n_sel, n_floor=_PRECISION_FLOOR, 
        unit=FRACTION,
        ci_low=ci_low, ci_high=ci_high,
        ci_method="wilson" if w.get("status") == "ok" else None, source_path=source,
        reason=(f"{hz}{name} edge = {edge:+.1%} (precision {precision:.1%} − base-rate {base_rate:.1%}; "
                f"Wilson CI [{ci_low:+.2f}, {ci_high:+.2f}], N={n_sel} selected) "
                f"vs target +{_band.target:.0%} / red-line {_band.red_line:+.0%}{lift_s}{arm_s}.")
        if w.get("status") == "ok" else None,
        arm=arm,
        **_tr(edge),
    )


def build_research_tile(
    bucket: str, run_date: str, s3_client=None, *, history: CardHistory | None = None,
) -> dict:
    """Build the Research tile from the e2e + research diagnostic artifacts.

    ``history`` (config#1836) supplies prior-card values so the critical
    score-vs-return components (research_composite_ic / cio_selection_skill /
    sector_teams_avg / cio) carry cross-cycle ``trend_4w``/``trend_13w``;
    omitted (standalone CLI / tests) → trends stay unpopulated.
    """

    def _tr(name: str, value: float | None) -> dict:
        return history.trends_for(MODULE, name, value) if history is not None else {}

    s3 = s3_client or boto3.client("s3")
    prefix = f"backtest/{run_date}"
    # Windowed resolution (config#1190): grade off the freshest artifact within the
    # trailing window so a partial/retried/off-cycle Saturday run still grades.
    e2e, _e2e_date, e2e_age, _e2e_key = get_json_windowed(s3, bucket, "backtest/{date}/e2e_lift.json", run_date)
    score_cal, _sc_date, sc_age, _score_key = get_json_windowed(s3, bucket, "backtest/{date}/score_calibration.json", run_date)
    macro, _macro_date, macro_age, _macro_key = get_json_windowed(s3, bucket, "backtest/{date}/macro_eval.json", run_date)
    pcal, _pcal_date, pcal_age, _pcal_key = get_json_windowed(s3, bucket, "backtest/{date}/portfolio_calibration.json", run_date)
    e2e_src = f"s3://{bucket}/{_e2e_key}" if _e2e_key else f"s3://{bucket}/{prefix}/e2e_lift.json"
    components = []

    e2e_stale = artifact_is_stale(e2e_age, RESEARCH_ARTIFACT_MAX_AGE_DAYS)
    e2e = e2e or {}

    # config#1580 / config-I2993: the six-team+CIO research orchestration was
    # RETIRED (last producing cycle 2026-07-11). When the backtester emits the
    # ``research_graph_retired`` marker, the four graph-sourced components
    # (sector_teams_avg / cio / cio_selection_skill / research_composite_ic)
    # render as RETIRED (accepted-permanent-N/A, DIAGNOSTIC so they never drag
    # the tile to WATCH) with the retired_date surfaced — NOT scored, NOT
    # critical-failing. Both-format tolerance: an OLD e2e_lift.json without the
    # marker still scores them exactly as before (graceful rollout).
    graph_retired = e2e.get("research_graph_retired")
    _RETIRED_SUPERSEDED = (
        "the live-arm scanner attractiveness IC (attractiveness_ic, config-I2994) "
        "+ the thinktank_coverage challenger arm"
    )

    def _retired_component(name: str, metric_type: str):
        rd = (graph_retired or {}).get("retired_date")
        return build_metric(
            name=name, module=MODULE, metric_type=metric_type, criticality="diagnostic",
            n_floor=1, band="unbanded", source_path=e2e_src, arm=f"six_team_cio (retired {rd})",
            permanent_na_reason=(
                f"retired {rd}: six-team+CIO research orchestration removed "
                f"(config#1580 / config-I2993) — measured a graph that no longer "
                f"produces. Superseded by {_RETIRED_SUPERSEDED}."
            ),
            **_tr(name, None),
        )

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
        history=history, arm=_SCANNER_METRIC_ARM,
    ))

    # 2. sector_teams_avg — RETIRED with the six-team graph (config-I2993) when
    #    the marker is present; otherwise pooled precision across the 6 teams.
    team_lift = e2e.get("team_lift")
    if graph_retired:
        components.append(_retired_component("sector_teams_avg", "pct"))
    elif isinstance(team_lift, list) and team_lift:
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
            history=history,
        ))
    else:
        components.append(build_metric(
            name="sector_teams_avg", module=MODULE, metric_type="pct", criticality="critical",
            estimator="wilson_precision_edge", measurement_horizon="21d",
            n_floor=_PRECISION_FLOOR, band="raw_precision", source_path=e2e_src,
            input_present=False, na_detail="sector_teams_avg: no team_lift list in e2e_lift this cycle.",
            **_tr("sector_teams_avg", None),
        ))

    # 3. cio — RETIRED with the CIO gate (config-I2993) when the marker is
    #    present; otherwise entrant-gate precision (CIO-advanced names that won).
    cl = e2e.get("cio_lift") or {}
    if graph_retired:
        components.append(_retired_component("cio", "pct"))
    else:
        cl_clf, cl_hz = _pick_clf(cl)
        cl_lift = ((cl.get("lift_21d_log") or {}).get("lift") if cl_hz == "21d"
                   else (e2e.get("cio_vs_ranking") or {}).get("lift"))
        components.append(_precision_metric(
            "cio", cl_clf, criticality="critical", source=e2e_src, horizon=cl_hz,
            lift=cl_lift, lift_label="21d-alpha-lift" if cl_hz == "21d" else "vs-ranking-lift",
            missing_detail="cio: e2e_lift.json absent or has no cio classification this cycle.",
            history=history,
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
    if graph_retired:
        components.append(_retired_component("cio_selection_skill", "log_return"))
    elif gap is not None:
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
            value=gap, n_samples=n_sel, n_floor=60, source_path=e2e_src,
            unit=LOG_RETURN_21D,
            status="WATCH" if insignificant else None,
            reason=(f"cio_selection_skill: ADVANCE−REJECT 21d log-alpha gap = {gap:+.4f} "
                    f"(ADVANCE {sel.get('advance_alpha_21d')} vs REJECT {sel.get('reject_alpha_21d')}, "
                    f"N={n_sel}{p_txt}{ic_txt}). "
                    + ("Not yet significant — WATCH, accumulating." if insignificant
                       else ("anti-selecting (gate advances worse names)" if gap < 0 else "adds selection value"))),
            **_tr("cio_selection_skill", gap),
        ))
    else:
        components.append(build_metric(
            name="cio_selection_skill", module=MODULE, metric_type="log_return", criticality="critical",
            estimator="advance_minus_reject_alpha_21d", measurement_horizon="21d",
            n_floor=60, source_path=e2e_src, input_present=False,
            na_detail="cio_selection_skill: no selection_skill_21d block in e2e_lift this cycle "
                      "(needs cio_evaluations joined to closed-21d universe_returns).",
            **_tr("cio_selection_skill", None),
        ))

    # 3c. research_composite_ic (critical, L4561) — the fundamental harness question:
    #     does the blended research score the system ACTS ON predict realized 21d
    #     alpha at all? Graded on the final_score rank-IC from the layer-attribution
    #     block. Under the L4562 contract: insignificant (p >= 0.10) → WATCH +
    #     reliability=low. The per-layer ICs (combined/macro/conviction) ride in the
    #     reason so the orchestration leak is visible (e.g. macro tilt degrading the
    #     stock score), motivating the de-blending arc.
    #
    #     SIGNIFICANCE is read off the DATE-CLUSTERED (Grinold-Kahn) IC t-stat when
    #     the producer emits it (config#1164): the pooled ``final_score_ic`` p counts
    #     every CIO-evaluated name as an independent draw, but the signal lives at the
    #     eval_date level — pooling ~K weeks of names as N≈K·25 manufactured a
    #     "significant negative composite IC" flag (pooled p≈0.02) that DISSOLVES under
    #     the honest weeks-as-N estimator (date-clustered p≈0.18). We grade on
    #     ``final_score_date_ic`` / ``_date_ic_p`` with n_samples = n_eval_dates, and
    #     fall back to the pooled IC only for pre-config#1164 e2e_lift artifacts.
    attr = cl.get("layer_attribution_21d") or {}
    date_ic = attr.get("final_score_date_ic")
    pooled_ic = attr.get("final_score_ic")
    use_clustered = date_ic is not None
    fic = date_ic if use_clustered else pooled_ic
    if graph_retired:
        # RETIRED (config-I2994): research_composite_ic was entirely CIO-sourced
        # (final_score rank-IC from the retired layer_attribution_21d block). It
        # is NOT resurrected — the live champion feed's ranking power is measured
        # honestly by the promoted, critical ``attractiveness_ic`` component
        # (scanner attractiveness_score→21d-alpha IC), with the challenger arm as
        # ``thinktank_coverage_ic``. Rendered retired, not scored.
        components.append(_retired_component("research_composite_ic", "ic"))
    elif fic is not None:
        if use_clustered:
            fp = attr.get("final_score_date_ic_p")
            n_at = attr.get("n_eval_dates")
            estimator = "date_clustered_rank_ic_vs_21d_alpha"
            n_floor = 8  # weeks: ~8 cohorts for a stable IC t-stat
        else:
            fp = attr.get("final_score_ic_p")
            n_at = attr.get("n")
            estimator = "rank_ic_vs_21d_alpha"
            n_floor = 60
        insig = fp is None or fp >= 0.10
        # Per-layer ride-along: prefer each layer's date-clustered IC, else pooled.
        layers = ", ".join(
            f"{k}={(attr.get(k + '_date_ic') if attr.get(k + '_date_ic') is not None else attr.get(k + '_ic')):+.3f}"
            for k in ("combined_score", "macro_shift", "cio_conviction")
            if (attr.get(k + "_date_ic") is not None or attr.get(k + "_ic") is not None)
        )
        # Surface the pseudo-replication gap when the pooled p would have over-flagged.
        pooled_p = attr.get("final_score_ic_p")
        pseudo_note = ""
        if (use_clustered and pooled_p is not None and pooled_p < 0.10
                and (fp is None or fp >= 0.10)):
            pseudo_note = (f" (pooled p={round(pooled_p, 3)} over N={attr.get('n')} names is "
                           "pseudo-replication-inflated; weeks-as-N is the honest test)")
        components.append(build_metric(
            name="research_composite_ic", module=MODULE, metric_type="ic", criticality="critical",
            estimator=estimator, measurement_horizon="21d",
            reliability="low" if insig else "high",
            value=fic, n_samples=n_at, n_floor=n_floor, source_path=e2e_src,
            unit=RANK_IC,
            status="WATCH" if insig else None,
            reason=(f"research_composite_ic: final_score→21d-alpha "
                    f"{'date-clustered ' if use_clustered else ''}rank-IC = {fic:+.3f} "
                    f"(p={fp if fp is None else round(fp,3)}, N={n_at}"
                    f"{' weeks' if use_clustered else ' names'}); per-layer [{layers}].{pseudo_note} "
                    + ("Not yet significant — WATCH, accumulating." if insig
                       else ("no forward signal — composite does not predict 21d alpha" if fic <= 0
                             else "composite carries forward signal"))),
            **_tr("research_composite_ic", fic),
        ))
    else:
        components.append(build_metric(
            name="research_composite_ic", module=MODULE, metric_type="ic", criticality="critical",
            estimator="rank_ic_vs_21d_alpha", measurement_horizon="21d",
            n_floor=60, source_path=e2e_src, input_present=False,
            na_detail="research_composite_ic: no layer_attribution_21d block in e2e_lift this cycle.",
            **_tr("research_composite_ic", None),
        ))

    # 3d. thinktank_coverage_ic (config-I2994, INFORMATIONAL/diagnostic) — the
    #     observe-only Think Tank challenger's shadow score vs realized 21d alpha,
    #     the ONLY research-authored composite score left in the system. Same
    #     date-clustered rank-IC estimator as the scanner arm (methodological
    #     parity). Diagnostic: labeled, never gates. Warms up on the 21d
    #     realization lag — an insufficient_data block grades a precise N/A and
    #     self-activates once a shadow cycle has realized 21d alpha.
    tt = (e2e.get("live_arm_score_ic") or {}).get("thinktank_coverage") or {}
    tt_hz = f"{tt.get('horizon_days', 21)}d"
    tt_arm = tt.get("arm") or "thinktank_coverage (observe-only challenger shadow)"
    tt_ic = tt.get("date_ic_mean") if tt.get("status") == "ok" else None
    if tt_ic is not None:
        tt_p = tt.get("date_ic_p")
        tt_n = tt.get("n_eval_dates")
        tt_insig = tt_p is None or tt_p >= 0.10 or (tt_n or 0) < 8
        components.append(build_metric(
            name="thinktank_coverage_ic", module=MODULE, metric_type="ic", criticality="diagnostic",
            estimator="date_clustered_rank_ic_vs_21d_alpha", measurement_horizon=tt_hz,
            reliability="low" if tt_insig else "high",
            value=tt_ic, n_samples=tt_n, n_floor=8, 
            unit=RANK_IC,
            source_path=e2e_src, status="WATCH" if tt_insig else None, arm=tt_arm,
            reason=(f"thinktank_coverage_ic: challenger shadow score→{tt_hz}-alpha "
                    f"date-clustered rank-IC = {tt_ic:+.3f} "
                    f"(p={tt_p if tt_p is None else round(tt_p, 3)}, N={tt_n} eval dates; "
                    f"series_start {tt.get('series_start')}, pooled {tt.get('pooled_ic')}). "
                    f"Observe-only challenger arm [arm: {tt_arm}] — informational, feeds no gate. "
                    + ("Not yet significant — WATCH, accumulating." if tt_insig
                       else ("challenger score carries forward signal" if tt_ic > 0
                             else "challenger score does not predict 21d alpha"))),
        ))
    else:
        tt_na = tt.get("reason") or "no thinktank_coverage block in live_arm_score_ic this cycle"
        components.append(build_metric(
            name="thinktank_coverage_ic", module=MODULE, metric_type="ic", criticality="diagnostic",
            estimator="date_clustered_rank_ic_vs_21d_alpha", measurement_horizon=tt_hz,
            n_floor=8, source_path=e2e_src, input_present=False, arm=tt_arm,
            na_detail=(f"thinktank_coverage_ic: {tt_na} (observe-only challenger shadow "
                       f"scores, config-I2994; warms up on the 21d realization lag). "
                       f"Feeds no gate."),
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
    #    SUPPORTING (config#1063): historically computed at a sub-strategy
    #    horizon, so it stays a leading diagnostic rather than a critical gate
    #    (promotion is a human call). The canonical-21d composite signal is
    #    graded critically as research_composite_ic (final_score rank-IC vs 21d
    #    alpha) above; keeping BOTH critical would double-count the construct
    #    and let a sub-horizon proxy drive a Director P0 (L4551/L4562).
    sc_src = f"s3://{bucket}/{_score_key}" if _score_key else f"s3://{bucket}/{prefix}/score_calibration.json"
    # HORIZON IS DECLARED BY THE ARTIFACT, never assumed here: we grade + label
    # at whatever `horizon` score_calibration.json carries (config#1063 — the
    # metric was once *computed* at 10d but *framed* as 21d; the fix is to echo
    # the producer's declaration). A parallel backtester PR moves this artifact
    # to the canonical primary horizon — this consumer needs no change when it
    # lands. The "10d" fallback applies ONLY to legacy artifacts predating the
    # `horizon` field (whose producer default was 10d).
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
            value=rho, n_samples=n_cal, n_floor=30, source_path=sc_src,
            unit=SPEARMAN_RHO,
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
            value=1.0 if mono else 0.0, n_samples=score_cal.get("n"), n_floor=1, band="unbanded", source_path=sc_src,
            unit=SCORE_0_1,
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
            n_floor=30, band="unbanded", source_path=sc_src, input_present=False,
            na_detail="composite_scoring: score_calibration.json absent this cycle (persists from a post-2026-06-04 Saturday run, B1a #279).",
        ))

    # 5. macro_agent (supporting) — macro accuracy lift vs realized regime (pp).
    mac_src = f"s3://{bucket}/{_macro_key}" if _macro_key else f"s3://{bucket}/{prefix}/macro_eval.json"
    if macro and macro.get("status") == "ok" and macro.get("accuracy_lift") is not None:
        components.append(build_metric(
            name="macro_agent", module=MODULE, metric_type="lift", criticality="supporting",
            value=macro.get("accuracy_lift"), n_samples=macro.get("n_evaluated") or macro.get("n"),
            unit=PCT,
            n_floor=20, source_path=mac_src,
            reason=f"macro_agent accuracy_lift={macro['accuracy_lift']:+.1f}pp, assessment={macro.get('assessment')}.",
        ))
    else:
        components.append(build_metric(
            name="macro_agent", module=MODULE, metric_type="lift", criticality="supporting",
            n_floor=20, source_path=mac_src, input_present=False,
            na_detail="macro_agent: macro_eval.json absent this cycle (persists from a post-2026-06-04 Saturday run, B1a #279).",
        ))

    # 6. calibration_diagnostics (supporting) — judge-vs-realized ECE (lower better).
    pc_src = f"s3://{bucket}/{_pcal_key}" if _pcal_key else f"s3://{bucket}/{prefix}/portfolio_calibration.json"
    if pcal and pcal.get("status") == "ok" and pcal.get("ece") is not None:
        components.append(build_metric(
            name="calibration_diagnostics", module=MODULE, metric_type="calibration", criticality="supporting",
            value=pcal.get("ece"), n_samples=pcal.get("n"), n_floor=100, 
            unit=ECE,
            higher_is_better=False, source_path=pc_src,
        ))
    else:
        components.append(build_metric(
            name="calibration_diagnostics", module=MODULE, metric_type="calibration", criticality="supporting",
            n_floor=100, higher_is_better=False, source_path=pc_src,
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
        # AMBIGUOUS(config#7485): the backtester producer's estimator for
        # `momentum_regime_ic` isn't declared in this artifact (no `estimator`
        # kwarg is passed here, unlike the rank-IC components above) — assumed
        # rank-IC by convention with the file's other IC-typed metrics.
        components.append(build_metric(
            name="momentum_regime_ic", module=MODULE, metric_type="ic", criticality="diagnostic",
            value=lo, n_samples=mri.get("n_weeks"), n_floor=6,
            unit=RANK_IC,
            higher_is_better=True, source_path=e2e_src,
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
            n_floor=6, higher_is_better=True, source_path=e2e_src,
            input_present=False,
            na_detail=(
                f"momentum_regime_ic: no ok block in e2e_lift this cycle "
                f"(status={mri.get('status')!r}); needs the backtester producer (config#1140) "
                f"+ >=4 realized weekly cohorts."
            ),
        ))

    # neutralization_live_efficacy (critical, config#1187) — does the LIVE #1142
    # score-neutralization (cut over 2026-06-22) actually recover forward edge?
    # The live cutover rewrites only signals.json score, which no IC producer
    # reads back, so research_composite_ic / composite_scoring / momentum_regime_ic
    # all measure the RAW research.db score and are BLIND to the live fix. The
    # backtester closes that with a post-cutover per-week (neutralized − raw) IC
    # delta + a date-clustered Grinold-Kahn t-stat (config#1164). WATCH while
    # accumulating / underpowered; GREEN once it's significantly positive; RED
    # only if the live fix is SIGNIFICANTLY HURTING (a revert signal), not noisy.
    #
    # SOURCE PREFERENCE (config#1187 root-cause fix): the DIRECT producer
    # `neutralization_live_forward_ic` joins the score the live system ACTUALLY
    # ranked on — crucible-research now persists it as the DUAL field
    # cio_evaluations.neutralized_final_score — to realized 21d alpha. That is the
    # true live efficacy, so prefer it. Fall back to neutralized_composite_ic's
    # live_forward segment (which RE-DERIVES the neutralized score with the same
    # residualizer the live path applies) until the persisted source has a
    # post-cutover cohort. The grading contract is identical for both.
    nlf = e2e.get("neutralization_live_forward_ic") or {}
    nci = e2e.get("neutralized_composite_ic") or {}

    # alpha-engine-config-I3000 (I2993 pattern extension, ruling: diagnostic
    # demotion): the neutralization pair already carried a ``cutover_date``
    # label, but the label alone did not stop them sourcing the RETIRED
    # six-team+CIO graph (``team_candidates`` / ``cio_evaluations``) — the
    # backtester now emits ``{"status": "retired", ...}`` for both (mirroring
    # the ``cio_lift`` shape). This still-critical component reads that
    # retired pair, so a silent frozen-sourced read is no longer possible; per
    # the deliberate ruling (option b — no live-path neutralization efficacy
    # series exists yet, config-I1187 tracks that follow-up), demote to
    # DIAGNOSTIC with a permanent_na_reason surfacing the retired_date, using
    # the exact same ``_retired_component`` convention PR128 used for the
    # four six-team+CIO components above.
    neu_retired = nlf if nlf.get("status") == "retired" else (
        nci if nci.get("status") == "retired" else None)
    if neu_retired is not None:
        _neu_rd = neu_retired.get("retired_date")
        components.append(build_metric(
            name="neutralization_live_efficacy", module=MODULE, metric_type="ic",
            criticality="diagnostic", n_floor=1, band="unbanded", source_path=e2e_src,
            arm=f"six_team_cio (retired {_neu_rd})",
            permanent_na_reason=(
                f"retired {_neu_rd}: six-team+CIO research orchestration removed "
                f"(config#1580 / config-I2993) — neutralization_live_efficacy read "
                f"the neutralized_composite_ic / neutralization_live_forward_ic pair, "
                f"which sourced that retired graph (team_candidates / "
                f"cio_evaluations); the cutover_date label alone did not stop it "
                f"aggregating frozen history at live weight (alpha-engine-config-"
                f"I3000). No live-path neutralization efficacy series exists yet "
                f"(config-I1187 tracks that follow-up) — demoted diagnostic rather "
                f"than left silently reading frozen CIO-era data."
            ),
            **_tr("neutralization_live_efficacy", None),
        ))
    elif nlf.get("status") == "ok" and (nlf.get("live_forward") or {}).get("n_weeks"):
        lf = nlf.get("live_forward")
        cutover_date = nlf.get("cutover_date")
    elif nci.get("status") == "ok" and (nci.get("live_forward") or {}).get("n_weeks"):
        lf = nci.get("live_forward")
        cutover_date = nci.get("cutover_date")
    else:
        lf = None
        cutover_date = nlf.get("cutover_date") or nci.get("cutover_date")

    if neu_retired is not None:
        pass  # already appended as a diagnostic retired component above.
    elif lf is not None and lf.get("n_weeks"):
        delta = lf.get("mean_weekly_delta")
        p = lf.get("delta_t_p")
        nwk = lf.get("n_weeks")
        sig = bool(lf.get("significant"))
        if not sig:
            status = "WATCH"
        elif delta is not None and delta < 0:
            status = "RED"      # live neutralization significantly degrades edge → revert
        else:
            status = None       # significantly positive → GREEN via thresholds
        components.append(build_metric(
            name="neutralization_live_efficacy", module=MODULE, metric_type="ic", criticality="critical",
            estimator="date_clustered_neutralized_minus_raw_ic_21d", measurement_horizon="21d",
            reliability="high" if sig else "low",
            value=delta, n_samples=nwk, n_floor=4, 
            unit=RANK_IC,
            higher_is_better=True, source_path=e2e_src, status=status,
            reason=(
                f"neutralization_live_efficacy: post-cutover ({cutover_date}) "
                f"weekly (neutralized−raw) 21d-alpha IC delta = "
                f"{('%+.3f' % delta) if delta is not None else 'n/a'} "
                f"(date-clustered p={p if p is None else round(p, 3)}, N={nwk} weeks; "
                f"neu {lf.get('neutralized_mean_weekly_ic')} vs raw {lf.get('raw_mean_weekly_ic')}). "
                + ("Live cutover SIGNIFICANTLY DEGRADES edge — revert candidate." if status == "RED"
                   else "Live cutover recovers edge (significant)." if sig
                   else "Accumulating post-cutover cohorts — WATCH, not yet significant.")
            ),
        ))
    else:
        components.append(build_metric(
            name="neutralization_live_efficacy", module=MODULE, metric_type="ic", criticality="critical",
            estimator="date_clustered_neutralized_minus_raw_ic_21d", measurement_horizon="21d",
            n_floor=4, higher_is_better=True, source_path=e2e_src,
            input_present=False,
            na_detail=(
                f"neutralization_live_efficacy: no live_forward block from either the "
                f"persisted-field producer neutralization_live_forward_ic "
                f"(status={nlf.get('status')!r}) or the re-derived "
                f"neutralized_composite_ic (status={nci.get('status')!r}) this cycle; needs the "
                f"backtester producer (config#1187) + >=1 post-cutover weekly cohort with realized 21d alpha."
            ),
        ))

    # 7-9. Agent-quality producer components (config Batch A #1149). These read
    #      backtest/{date}/agent_quality.json — the same artifact the Agent tile
    #      reads — but grade the research-output-quality axis (judge pass-rate,
    #      pillar coverage, signal volume). Absent → precise N/A-MISSING-INPUT
    #      naming the producer, self-activating on the first agent_quality.json.
    aq, _aq_date, aq_age, _aq_key = get_json_windowed(s3, bucket, "backtest/{date}/agent_quality.json", run_date)
    aq_src = f"s3://{bucket}/{_aq_key}" if _aq_key else f"s3://{bucket}/{prefix}/agent_quality.json"

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
            value=blk["value"], n_samples=blk.get("n"), n_floor=10, 
            unit=FRACTION,
            source_path=aq_src,
            reason=f"judge_rubric_pass_rate = {blk['value']:.1%} (N={blk.get('n')} evals) vs target 85% / red-line 60%.",
        ))
    else:
        components.append(build_metric(
            name="judge_rubric_pass_rate", module=MODULE, metric_type="pct", criticality="supporting",
            n_floor=10, source_path=aq_src, input_present=False,
            na_detail="judge_rubric_pass_rate: agent_quality.json absent or no value this cycle (research agent-quality producer, config#1149).",
        ))

    # 8. pillar_emit_coverage (supporting) — % of universe entries carrying a
    #    pillar_assessment (higher better).
    blk = _aq_block("pillar_emit_coverage")
    if blk is not None:
        components.append(build_metric(
            name="pillar_emit_coverage", module=MODULE, metric_type="pct", criticality="supporting",
            value=blk["value"], n_samples=blk.get("n"), n_floor=10, 
            unit=FRACTION,
            source_path=aq_src,
            reason=f"pillar_emit_coverage = {blk['value']:.1%} (N={blk.get('n')} universe entries) vs target 90% / red-line 50%.",
        ))
    else:
        components.append(build_metric(
            name="pillar_emit_coverage", module=MODULE, metric_type="pct", criticality="supporting",
            n_floor=10, source_path=aq_src, input_present=False,
            na_detail="pillar_emit_coverage: agent_quality.json absent or no value this cycle (research agent-quality producer, config#1149).",
        ))

    # 9. signal_volume_adequacy (diagnostic) — count of finalized signals vs the
    #    adequacy floor (higher better).
    blk = _aq_block("signal_volume_adequacy")
    if blk is not None:
        components.append(build_metric(
            name="signal_volume_adequacy", module=MODULE, metric_type="count", criticality="diagnostic",
            value=blk["value"], n_samples=blk.get("n"), n_floor=1, 
            unit=COUNT_SIGNALS,
            source_path=aq_src,
            reason=f"signal_volume_adequacy = {blk['value']:.0f} finalized signals vs target 20 / red-line 8.",
        ))
    else:
        components.append(build_metric(
            name="signal_volume_adequacy", module=MODULE, metric_type="count", criticality="diagnostic",
            n_floor=1, source_path=aq_src, input_present=False,
            na_detail="signal_volume_adequacy: agent_quality.json absent or no value this cycle (research agent-quality producer, config#1149).",
        ))

    # 10-12. Attractiveness-evaluation components (config#1389/#1392/#1398).
    #        Source: backtest/{date}/attractiveness_eval.json (schema_version 1,
    #        FROZEN — see tests/fixtures/attractiveness_eval_v1.json), produced
    #        by the backtester's attractiveness evaluator. The producer PR lands
    #        in parallel with this consumer: until it ships, every component
    #        below grades a precise N/A-MISSING-INPUT and self-activates on the
    #        first artifact (same forward-compat contract as agent_quality).
    #        All alpha values in the artifact are decimal 21d log-alpha.
    att, _att_date, att_age, _att_key = get_json_windowed(
        s3, bucket, "backtest/{date}/attractiveness_eval.json", run_date
    )
    att_src = f"s3://{bucket}/{_att_key}" if _att_key else f"s3://{bucket}/{prefix}/attractiveness_eval.json"
    att_ok = bool(att) and att.get("status") == "ok"
    att_hz = f"{(att or {}).get('horizon_days', 21)}d"
    # config#1861: validate against the lib contract, but only once the producer
    # emits v2 — v1 artifacts still sitting in S3 during the S3 cutover window
    # (~2026-07-15) predate the schema move and must not fail the v2 schema.
    if att and (att.get("schema_version") or 0) >= 2:
        contracts.validate("attractiveness_eval", att)
    _ATT_MISSING = (
        "attractiveness_eval.json absent this cycle (backtester attractiveness-eval "
        "producer, config#1389 — deploys in parallel with this consumer)."
        if not att else
        f"attractiveness_eval.json status={att.get('status')!r} this cycle "
        f"(insufficient realized-21d history to evaluate — accumulating)."
    )

    def _fmt_p(p):
        return "n/a" if p is None else f"{p:.3g}"

    def _fmt_pct(v):
        return "n/a" if v is None else f"{v:.0%}"

    def _alpha(d):
        # dual-tolerance for the config#1861 v1→v2 field rename; drop the
        # `mean_alpha_21d` fallback after the 1-week S3 window (~2026-07-15).
        return d.get("mean_alpha", d.get("mean_alpha_21d"))

    # 10. attractiveness_ic (CRITICAL, config-I2994 — the live-arm scanner
    #     attractiveness score-IC that REPLACES the retired research_composite_ic;
    #     promoted supporting→critical per Brian's ruling since the universe-board
    #     attractiveness_score IS the live champion feed's ranking signal).
    #     Date-clustered (weeks-as-N) rank-IC vs realized 21d alpha, WATCH +
    #     reliability=low while insignificant (p >= 0.10) or under-sampled —
    #     accumulate to significance, never a confident grade on noise.
    comp_ic = (att or {}).get("composite_ic") or {}
    a_ic = comp_ic.get("date_ic_mean") if att_ok else None
    if a_ic is not None:
        a_p = comp_ic.get("date_ic_p")
        a_n = comp_ic.get("n_eval_dates")
        a_floor = 8  # weeks: ~8 cohorts for a stable IC t-stat (mirrors research_composite_ic)
        a_insig = a_p is None or a_p >= 0.10 or (a_n or 0) < a_floor
        pillar_ic = att.get("pillar_ic") or {}
        weights = att.get("suggested_pillar_weights") or {}
        shrink = att.get("shrinkage") or {}
        pillars_s = ", ".join(
            f"{k}={v.get('date_ic_mean'):+.3f}(p={_fmt_p(v.get('date_ic_p'))})"
            for k, v in sorted(pillar_ic.items())
            if isinstance(v, dict) and v.get("date_ic_mean") is not None
        ) or "n/a"
        weights_s = ", ".join(f"{k}={v:.2f}" for k, v in sorted(weights.items())) or "n/a"
        shrink_s = (f"{shrink.get('method')} λ={shrink.get('lambda')}"
                    if shrink.get("method") else "n/a")
        components.append(build_metric(
            name="attractiveness_ic", module=MODULE, metric_type="ic", criticality="critical",
            estimator="date_clustered_rank_ic_vs_21d_alpha", measurement_horizon=att_hz,
            reliability="low" if a_insig else "high", arm=_LIVE_ARM_SCANNER,
            value=a_ic, n_samples=a_n, n_floor=a_floor, 
            unit=RANK_IC,
            source_path=att_src, status="WATCH" if a_insig else None,
            reason=(f"attractiveness_ic: attractiveness→{att_hz}-alpha date-clustered rank-IC = "
                    f"{a_ic:+.3f} (p={_fmt_p(a_p)}, N={a_n} eval dates; pooled "
                    f"{comp_ic.get('pooled_ic')} p={_fmt_p(comp_ic.get('pooled_ic_p'))}, "
                    f"n={comp_ic.get('n')}). Per-pillar IC [{pillars_s}]; suggested pillar "
                    f"weights ({shrink_s}) [{weights_s}]. Live champion-feed ranking metric "
                    f"[arm: {_LIVE_ARM_SCANNER}] — replaces the retired research_composite_ic "
                    f"(config-I2994). "
                    + ("Not yet significant — WATCH, accumulating." if a_insig
                       else ("attractiveness carries forward signal" if a_ic > 0
                             else "no forward signal — attractiveness does not predict 21d alpha"))),
        ))
    else:
        components.append(build_metric(
            name="attractiveness_ic", module=MODULE, metric_type="ic", criticality="critical",
            estimator="date_clustered_rank_ic_vs_21d_alpha", measurement_horizon=att_hz,
            n_floor=8, source_path=att_src, input_present=False,
            arm=_LIVE_ARM_SCANNER,
            na_detail=(f"attractiveness_ic: {_ATT_MISSING} Live champion-feed ranking metric "
                       f"[arm: {_LIVE_ARM_SCANNER}] — replaces the retired research_composite_ic "
                       f"(config-I2994)."),
        ))

    # 11. attractiveness_trajectory_ic (DIAGNOSTIC) — does the PRE-repricing
    #     attractiveness score predict realized 21d alpha? This is the
    #     observe→cutover evidence for config#1392 (repricing-trajectory
    #     substrate): a significant pre-repricing IC is the gate criterion for
    #     cutting the live feed over to the trajectory-aware score. Informs the
    #     gate; the flag flip itself stays a human/registry decision.
    traj = ((att or {}).get("trajectory_ic") or {}).get("pre_repricing_score") or {}
    t_ic = traj.get("date_ic_mean") if att_ok else None
    if t_ic is not None:
        t_p = traj.get("date_ic_p")
        t_n = traj.get("n_eval_dates")
        t_insig = t_p is None or t_p >= 0.10
        slope = ((att or {}).get("trajectory_ic") or {}).get("attr_slope_z") or {}
        slope_s = (f"; attr_slope_z IC {slope.get('date_ic_mean'):+.3f} "
                   f"(p={_fmt_p(slope.get('date_ic_p'))})"
                   if slope.get("date_ic_mean") is not None else "")
        components.append(build_metric(
            name="attractiveness_trajectory_ic", module=MODULE, metric_type="ic",
            criticality="diagnostic",
            estimator="date_clustered_rank_ic_vs_21d_alpha", measurement_horizon=att_hz,
            reliability="low" if t_insig else "high",
            value=t_ic, n_samples=t_n, n_floor=8, 
            unit=RANK_IC,
            source_path=att_src, status="WATCH" if t_insig else None,
            reason=(f"attractiveness_trajectory_ic: pre_repricing_score→{att_hz}-alpha "
                    f"date-clustered rank-IC = {t_ic:+.3f} (p={_fmt_p(t_p)}, N={t_n} eval "
                    f"dates){slope_s}. This IC is the observe→cutover evidence for the "
                    f"repricing-trajectory gate (config#1392). "
                    + ("Not yet significant — WATCH, accumulating." if t_insig
                       else "significant — cutover evidence accruing")),
        ))
    else:
        components.append(build_metric(
            name="attractiveness_trajectory_ic", module=MODULE, metric_type="ic",
            criticality="diagnostic",
            estimator="date_clustered_rank_ic_vs_21d_alpha", measurement_horizon=att_hz,
            n_floor=8, source_path=att_src, input_present=False,
            na_detail=(f"attractiveness_trajectory_ic: no pre_repricing_score IC block — "
                       f"{_ATT_MISSING} Drives the config#1392 observe→cutover gate once present."),
        ))

    # 12. scanner_feed_counterfactual (SUPPORTING) — config#1398 "size the
    #     prize", surfaced weekly: how much realized 21d log-alpha would a
    #     top-N attractiveness feed have captured vs the LIVE scanner gate?
    #     Headline value = mean_alpha_21d(top-N) − mean_alpha_21d(live_gate),
    #     comparing at matched breadth (the top_n row whose N is closest to the
    #     live gate's survivor count). No red-line: this is an opportunity
    #     metric (positive prize → GREEN, else WATCH), never a failure RED.
    cf = (att or {}).get("counterfactual") or {}
    top_n = [e for e in (cf.get("top_n") or [])
             if isinstance(e, dict) and _alpha(e) is not None]
    live_gate = cf.get("live_gate") or {}
    lg_alpha = _alpha(live_gate)
    if att_ok and top_n and lg_alpha is not None:
        n_surv = live_gate.get("n_survivors")
        chosen = (min(top_n, key=lambda e: abs((e.get("n") or 0) - n_surv))
                  if n_surv is not None else top_n[0])
        prize = _alpha(chosen) - lg_alpha
        # alpha-engine-config-I7213 (symmetry): render each cohort's excess
        # over the PIT population beside its raw alpha, and do the same for
        # the live gate. Raw mean_alpha alone is not comparable across arms —
        # every cohort here is measured on its own cycle set, and the tape of
        # those cycles is common to selector and population alike. On the
        # 2026-08-13 card the live gate's +0.0275 stood beside top-20 cohorts
        # near +0.001 with no denominator on the live leg, so the entire gap
        # read as selection skill. Absent on pre-I7213 artifacts (e.g.
        # backtest/2026-08-07) — omitted silently there rather than shown as
        # a fabricated zero.
        def _excess_s(e):
            x = e.get("excess_vs_population")
            if x is None:
                return ""
            p = e.get("excess_p")
            return f", vs-population {x:+.4f}" + (f" (p={_fmt_p(p)})" if p is not None else "")

        rows_s = "; ".join(
            f"top-{e.get('n')}{'(sector-bal)' if e.get('sector_balanced') else ''}: "
            f"capture {_fmt_pct(e.get('capture_rate'))}, alpha {_alpha(e):+.4f}"
            f"{_excess_s(e)}"
            for e in top_n
        )
        lg_excess_s = _excess_s(live_gate)
        pop_s = (f" Population (PIT, same cycles) mean {att_hz} alpha "
                 f"{live_gate['population_mean_alpha']:+.4f} — every arm above is "
                 f"stated against it."
                 if live_gate.get("population_mean_alpha") is not None else
                 " Population leg absent on this artifact (pre-alpha-engine-config-I7213 "
                 "producer) — the arms below are raw means with no shared denominator.")
        components.append(build_metric(
            name="scanner_feed_counterfactual", module=MODULE, metric_type="log_return",
            criticality="supporting",
            estimator="matched_breadth_alpha_delta", measurement_horizon=att_hz,
            value=prize, n_samples=n_surv, n_floor=10, 
            unit=LOG_RETURN_21D,
            higher_is_better=True, source_path=att_src,
            reason=(f"scanner_feed_counterfactual: top-{chosen.get('n')} attractiveness feed "
                    f"− live gate = {prize:+.4f} mean {att_hz} log-alpha (live gate: capture "
                    f"{_fmt_pct(live_gate.get('capture_rate'))}, alpha {lg_alpha:+.4f}"
                    f"{lg_excess_s}, N={n_surv} survivors).{pop_s} All cohorts [{rows_s}]. "
                    f"Sizes the prize of an "
                    f"attractiveness-ranked scanner feed (config#1398) — opportunity surface, "
                    f"not a failure gate."),
        ))
    else:
        components.append(build_metric(
            name="scanner_feed_counterfactual", module=MODULE, metric_type="log_return",
            criticality="supporting",
            estimator="matched_breadth_alpha_delta", measurement_horizon=att_hz,
            n_floor=10, higher_is_better=True,
            source_path=att_src, input_present=False,
            na_detail=(f"scanner_feed_counterfactual: no usable counterfactual block "
                       f"(needs top_n rows AND a live_gate baseline for a like-for-like "
                       f"delta) — {_ATT_MISSING}"),
        ))

    # 13. judge_outcome_ic (DIAGNOSTIC) — does the LLM-judge quality-score rank
    #     correlate with realized 21d alpha? Reads the judge_outcome_ic block a
    #     parallel backtester PR adds to backtest/{date}/agent_quality.json
    #     (schema_version 1, FROZEN). This is observability OF THE JUDGE
    #     ITSELF: it VALIDATES (does not steer) the rubric layer — diagnostic
    #     criticality by design, it must never feed a gate. Producers deploy
    #     independently: an agent_quality.json without the block (or with
    #     status "insufficient") grades a precise N/A and self-activates once
    #     the producer ships (graceful-forward-compat, like every unwired
    #     producer today).
    joi = (aq or {}).get("judge_outcome_ic") or {}
    joi_overall = joi.get("overall") or {}
    j_ic = joi_overall.get("date_ic_mean") if joi.get("status") == "ok" else None
    j_hz = f"{joi.get('horizon_days', 21)}d"
    if j_ic is not None:
        j_p = joi_overall.get("date_ic_p")
        j_n = joi_overall.get("n_eval_dates")
        j_insig = j_p is None or j_p >= 0.10
        dims = joi.get("by_dimension") or {}
        dims_s = ", ".join(
            f"{k}={v.get('date_ic_mean'):+.3f}(p={_fmt_p(v.get('date_ic_p'))})"
            for k, v in sorted(dims.items())
            if isinstance(v, dict) and v.get("date_ic_mean") is not None
        ) or "n/a"
        components.append(build_metric(
            name="judge_outcome_ic", module=MODULE, metric_type="ic", criticality="diagnostic",
            estimator="date_clustered_rank_ic_vs_21d_alpha", measurement_horizon=j_hz,
            reliability="low" if j_insig else "high",
            value=j_ic, n_samples=j_n, n_floor=8, 
            unit=RANK_IC,
            source_path=aq_src, status="WATCH" if j_insig else None,
            reason=(f"judge_outcome_ic: judge quality-score→{j_hz}-alpha date-clustered "
                    f"rank-IC = {j_ic:+.3f} (p={_fmt_p(j_p)}, N={j_n} eval dates; pooled "
                    f"{joi_overall.get('pooled_ic')} p={_fmt_p(joi_overall.get('pooled_ic_p'))}, "
                    f"n={joi_overall.get('n')}; {joi.get('n_unattributable')} unattributable). "
                    f"Per-dimension [{dims_s}]. VALIDATES (does not steer) the judge rubric "
                    f"layer — observability of the judge itself; feeds no gate. "
                    + ("Not yet significant — WATCH, accumulating." if j_insig
                       else ("judge scores carry outcome signal" if j_ic > 0
                             else "judge scores anti-correlate with outcomes"))),
        ))
    else:
        joi_status = joi.get("status")
        components.append(build_metric(
            name="judge_outcome_ic", module=MODULE, metric_type="ic", criticality="diagnostic",
            estimator="date_clustered_rank_ic_vs_21d_alpha", measurement_horizon=j_hz,
            n_floor=8, source_path=aq_src, input_present=False,
            na_detail=("judge_outcome_ic: "
                       + (f"judge_outcome_ic block status={joi_status!r} this cycle "
                          f"(insufficient judge↔realized-alpha joins — accumulating)."
                          if joi_status == "insufficient" else
                          "no judge_outcome_ic block in agent_quality.json this cycle "
                          "(backtester judge-outcome producer deploys in parallel with this "
                          "consumer — self-activates on first emission). ")
                       + " Validates (does not steer) the judge rubric layer; feeds no gate."),
        ))

    # 14. scanner_basket_return (CRITICAL, alpha-engine-config-I7213) — the
    #     number Brian asked directly for: "population (~900 stocks) vs the
    #     scanner attractiveness top 20 over a specific window". Replaces the
    #     classification-metric framing (precision/recall on a beat_spy
    #     binary) — a retrieval metric that discards magnitude — which fit the
    #     retired pass/fail tech_score GATE but does not fit a RANKED top-N
    #     feed. Value = excess_vs_population for the top-20, sector_balanced=
    #     True cohort (I7213's named variant); the raw (unbalanced) cohort is
    #     reported alongside in status_reason because a 20-name basket carries
    #     several times the idiosyncratic variance of the ~900-name
    #     population and a sector-concentrated selector can look skilled in
    #     any favourable tape — measured precedent: the retired arm's
    #     scanner_lift.lift_21d_log was +0.402pp raw vs -0.079pp sector-
    #     neutral, i.e. the entire apparent edge there was a sector bet.
    #
    #     Thresholds (set here, not copied): target=0.0025 (25bp mean 21d
    #     excess-over-population). Chosen conservatively below the measured
    #     top-60 counterfactual mean alpha (+1.155%, market-relative) and well
    #     below live-gate's measured +2.731% headline, because
    #     excess_vs_population is a NEW, unmeasured, strictly harder bar — it
    #     is alpha relative to the PIT population mean on top of the alpha
    #     frame already being SPY-relative, so a smaller edge than either
    #     prior number is expected even from a working selector. red_line=0.0:
    #     a top-20 pick that does not beat the population it was drawn from
    #     adds nothing over holding the universe — a floor derived directly
    #     from what the metric measures, not borrowed from another component.
    #
    #     Producer (crucible-backtester attractiveness_eval.py, paired PR)
    #     adds n=20/40 cohorts plus excess_vs_population / excess_t /
    #     excess_p / excess_ci95 / population_mean_alpha / holding_rule to
    #     counterfactual.top_n. Until it lands (or if any of those fields is
    #     absent from an n=20 row), this grades a precise N/A-MISSING-INPUT
    #     naming the missing field and self-activates on first emission —
    #     same forward-compat contract as the other attractiveness_eval-
    #     sourced components above.
    def _cf_row(n, sector_balanced):
        for e in top_n_all:
            if isinstance(e, dict) and e.get("n") == n and bool(e.get("sector_balanced")) is sector_balanced:
                return e
        return None

    top_n_all = cf.get("top_n") or []
    sb20 = _cf_row(20, True)
    raw20 = _cf_row(20, False)
    holding_rule = cf.get("holding_rule")
    sb_excess = sb20.get("excess_vs_population") if sb20 else None
    sb_pop = sb20.get("population_mean_alpha") if sb20 else None

    _missing_field = None
    if att_ok:
        if sb20 is None:
            _missing_field = "top_n row for n=20, sector_balanced=True"
        elif sb_excess is None:
            _missing_field = "excess_vs_population"
        elif sb_pop is None:
            _missing_field = "population_mean_alpha"
        elif holding_rule is None:
            _missing_field = "holding_rule"

    if att_ok and _missing_field is None:
        raw_s = (
            f"raw (unbalanced) top-20: excess_vs_population={raw20.get('excess_vs_population'):+.4f} "
            f"(t={_fmt_p(raw20.get('excess_t'))}, p={_fmt_p(raw20.get('excess_p'))})"
            if raw20 and raw20.get("excess_vs_population") is not None
            else "raw (unbalanced) top-20: not available this cycle"
        )
        n_cyc = sb20.get("n_cycles")
        sb_p = sb20.get("excess_p")
        insig = sb_p is None or sb_p >= 0.10 or (n_cyc or 0) < 8
        ci = sb20.get("excess_ci95") or [None, None]
        ci_low = ci[0] if isinstance(ci, list) and len(ci) == 2 else None
        ci_high = ci[1] if isinstance(ci, list) and len(ci) == 2 else None
        components.append(build_metric(
            name="scanner_basket_return", module=MODULE, metric_type="log_return",
            criticality="critical",
            estimator="date_clustered_mean_excess_vs_pit_population", measurement_horizon=att_hz,
            reliability="low" if insig else "high", arm=_LIVE_ARM_SCANNER,
            value=sb_excess, n_samples=n_cyc, n_floor=8, 
            unit=LOG_RETURN_21D,
            ci_low=ci_low, ci_high=ci_high, source_path=att_src,
            status="WATCH" if insig else None,
            reason=(f"scanner_basket_return: top-20 sector-balanced attractiveness basket "
                    f"vs PIT population ({sb_pop:+.4f} mean {att_hz} alpha) = {sb_excess:+.4f} "
                    f"excess (t={_fmt_p(sb20.get('excess_t'))}, p={_fmt_p(sb_p)}, N={n_cyc} "
                    f"cycles). {raw_s}. Holding rule: {holding_rule}. Direct top-20-basket-"
                    f"vs-population read Brian asked for (alpha-engine-config-I7213); "
                    f"replaces the classification-metric framing for a ranked feed. "
                    f"[arm: {_LIVE_ARM_SCANNER}] "
                    + ("Not yet significant — WATCH, accumulating." if insig
                       else ("basket beats the population" if sb_excess > 0
                             else "basket does not beat the population"))),
        ))
    else:
        components.append(build_metric(
            name="scanner_basket_return", module=MODULE, metric_type="log_return",
            criticality="critical",
            estimator="date_clustered_mean_excess_vs_pit_population", measurement_horizon=att_hz,
            n_floor=8, source_path=att_src, input_present=False,
            arm=_LIVE_ARM_SCANNER,
            na_detail=(f"scanner_basket_return: "
                       + (_ATT_MISSING if not att_ok else
                          f"attractiveness_eval.json present but missing "
                          f"{_missing_field!r} — the top-20 basket-vs-population cohort "
                          f"and its excess-vs-population fields land via the "
                          f"alpha-engine-config-I7213 backtester producer PR, deploying in "
                          f"parallel with this consumer, self-activates on first emission.")
                       + f" Direct top-20-basket-vs-population read Brian asked for. "
                       f"[arm: {_LIVE_ARM_SCANNER}]"),
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
