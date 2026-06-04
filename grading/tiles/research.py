"""
research.py — Tile 1: Research (RC v2).

Grades the signal generator (scanner → sector teams → CIO → composite scoring +
macro + calibration). The gradable signal in the e2e artifact is each stage's
**classification precision** (did the selected names beat their baseline) — it
carries tp/fp counts, so we grade precision with a Wilson interval (proper CI +
N), which is more institutional than grading the noisy point-estimate return
*lift* that has no per-pick CI in the artifact. Lift is surfaced in the reason.

Sources: ``backtest/{date}/e2e_lift.json`` (scanner / team / cio classification),
``score_calibration.json`` (composite monotonicity), ``macro_eval.json`` (macro
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


def _precision_metric(
    name, clf, *, criticality, source, target=0.45, red_line=0.35,
    lift=None, lift_label=None, missing_detail=None,
) -> dict:
    """A precision MetricRecord (Wilson CI) from an e2e classification block.

    precision = tp/(tp+fp); N = tp+fp (the count of *selected* names). When the
    classification block is absent the component grades N/A-MISSING-INPUT with a
    specific reason.
    """
    if not clf or clf.get("precision") is None:
        return build_metric(
            name=name, module=MODULE, metric_type="pct", criticality=criticality,
            n_floor=_PRECISION_FLOOR, target=target, red_line=red_line, source_path=source,
            input_present=False, na_detail=missing_detail or f"{name}: no classification block in e2e_lift this cycle.",
        )
    tp = int(clf.get("tp", 0))
    fp = int(clf.get("fp", 0))
    n_sel = tp + fp
    w = wilson_score_interval(tp, n_sel) if n_sel > 0 else {"status": "insufficient_data"}
    precision = clf.get("precision")
    lift_s = f"; {lift_label} {lift:+.2%}" if (lift is not None and lift_label) else ""
    return build_metric(
        name=name, module=MODULE, metric_type="pct", criticality=criticality,
        value=precision, n_samples=n_sel, n_floor=_PRECISION_FLOOR, target=target, red_line=red_line,
        ci_low=w.get("ci_low"), ci_high=w.get("ci_high"),
        ci_method="wilson" if w.get("status") == "ok" else None, source_path=source,
        reason=(f"{name} precision = {precision:.1%} (Wilson CI "
                f"[{w.get('ci_low', 0):.2f}, {w.get('ci_high', 0):.2f}], N={n_sel} selected) "
                f"vs target {target:.0%} / red-line {red_line:.0%}{lift_s}.")
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

    # 1. scanner (supporting) — precision of quant-filter passers vs baseline.
    sl = e2e.get("scanner_lift") or {}
    components.append(_precision_metric(
        "scanner", sl.get("classification"), criticality="supporting", source=e2e_src,
        lift=sl.get("lift"), lift_label="return-lift",
        missing_detail="scanner: e2e_lift.json absent or has no scanner classification this cycle.",
    ))

    # 2. sector_teams_avg (critical) — pooled precision across the 6 sector teams.
    team_lift = e2e.get("team_lift")
    if isinstance(team_lift, list) and team_lift:
        tp = sum(int((t.get("classification") or {}).get("tp", 0)) for t in team_lift)
        fp = sum(int((t.get("classification") or {}).get("fp", 0)) for t in team_lift)
        pooled = {"precision": (tp / (tp + fp)) if (tp + fp) else None, "tp": tp, "fp": fp}
        components.append(_precision_metric(
            "sector_teams_avg", pooled, criticality="critical", source=e2e_src,
        ))
    else:
        components.append(build_metric(
            name="sector_teams_avg", module=MODULE, metric_type="pct", criticality="critical",
            n_floor=_PRECISION_FLOOR, target=0.45, red_line=0.35, source_path=e2e_src,
            input_present=False, na_detail="sector_teams_avg: no team_lift list in e2e_lift this cycle.",
        ))

    # 3. cio (critical) — entrant-gate precision (CIO-advanced names that won).
    cl = e2e.get("cio_lift") or {}
    components.append(_precision_metric(
        "cio", cl.get("classification"), criticality="critical", source=e2e_src,
        lift=(e2e.get("cio_vs_ranking") or {}).get("lift"), lift_label="vs-ranking-lift",
        missing_detail="cio: e2e_lift.json absent or has no cio classification this cycle.",
    ))

    # 4. composite_scoring (critical) — does higher composite score → higher
    #    realized return? Graded on the score_calibration monotonicity flag.
    sc_src = f"s3://{bucket}/{prefix}/score_calibration.json"
    if score_cal and score_cal.get("status") == "ok" and "monotonic" in score_cal:
        mono = bool(score_cal["monotonic"])
        beat = score_cal.get("beat_spy_pct")
        components.append(build_metric(
            name="composite_scoring", module=MODULE, metric_type="pct", criticality="critical",
            value=1.0 if mono else 0.0, n_samples=score_cal.get("n"), n_floor=1, source_path=sc_src,
            status="GREEN" if mono else "RED",
            reason=(f"composite_scoring monotonic={mono} (score→return rank); "
                    f"beat_spy_pct={beat:.1%} over N={score_cal.get('n')}." if beat is not None
                    else f"composite_scoring monotonic={mono}."),
        ))
    else:
        components.append(build_metric(
            name="composite_scoring", module=MODULE, metric_type="pct", criticality="critical",
            n_floor=1, source_path=sc_src, input_present=False,
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

    # 7-9. Aspirational components not yet produced — transparent N/A-NOT-IMPL.
    for name, crit in (("judge_rubric_pass_rate", "supporting"),
                       ("pillar_emit_coverage", "supporting"),
                       ("signal_volume_adequacy", "diagnostic")):
        components.append(build_metric(
            name=name, module=MODULE, metric_type="pct", criticality=crit, n_floor=1,
            source_path=f"s3://{bucket}/{prefix}/", implemented=False,
            na_detail=f"{name}: producer-side analysis not yet implemented/persisted for the report card.",
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
