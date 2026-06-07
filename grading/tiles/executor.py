"""
executor.py — Tile 3: Executor (RC v2).

Grades the execution layer (entry triggers / risk guard / exit rules / process
quality). The gradable components read the backtester's executor diagnostics
over S3:
  - entry_triggers : trigger_scorecard.summary (win-rate vs SPY, Wilson CI)
  - risk_guard     : shadow_book.classification (block precision, Wilson CI)
  - exit_rules     : exit_timing.summary (capture ratio)
  - excursion      : portfolio_excursion (MFE/MAE process quality)

These are OK-only-persisted by the backtester and only land on a Saturday run,
so they grade a precise N/A-MISSING-INPUT until present. position_sizing
(sizing_ab) is B1c-deferred (genuinely unwired); action_entropy /
reconciliation_integrity are not yet produced → transparent N/A. Portfolio
construction / implementation-shortfall / trigger-hit-rate components from the
Tile-3 spec are deferred to a later increment (kept out to avoid N/A noise; the
Portfolio Outcome tile already covers P&L).

Spec: ``system-report-card-revamp-260522.md`` Tile 3.
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

MODULE = "executor"


def _get_json(s3, bucket: str, key: str) -> dict | None:
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        logger.error("S3 read failed for s3://%s/%s: %s", bucket, key, e)
        raise
    return json.loads(resp["Body"].read())


def build_executor_tile(bucket: str, run_date: str, s3_client=None) -> dict:
    """Build the Executor tile from the backtester executor diagnostics."""
    s3 = s3_client or boto3.client("s3")
    prefix = f"backtest/{run_date}"
    trig = _get_json(s3, bucket, f"{prefix}/trigger_scorecard.json")
    shadow = _get_json(s3, bucket, f"{prefix}/shadow_book.json")
    exits = _get_json(s3, bucket, f"{prefix}/exit_timing.json")
    exc = _get_json(s3, bucket, f"{prefix}/portfolio_excursion.json")
    components = []

    def src(name):
        return f"s3://{bucket}/{prefix}/{name}"

    # 1. entry_triggers (critical) — win-rate vs SPY across timed entries (Wilson).
    ts_src = src("trigger_scorecard.json")
    if trig and trig.get("status") == "ok":
        summ = trig.get("summary") or {}
        wr = summ.get("win_rate_vs_spy")
        n = int(summ.get("total_entries") or 0)
        if wr is not None and n > 0:
            successes = round(wr * n)
            w = wilson_score_interval(successes, n)
            slip = summ.get("avg_slippage_vs_signal")
            components.append(build_metric(
                name="entry_triggers", module=MODULE, metric_type="pct", criticality="critical",
                estimator="wilson_winrate", measurement_horizon="intraday_to_exit",
                value=wr, n_samples=n, n_floor=30, target=0.55, red_line=0.45,
                ci_low=w.get("ci_low"), ci_high=w.get("ci_high"),
                ci_method="wilson" if w.get("status") == "ok" else None, source_path=ts_src,
                reason=(f"entry_triggers win-rate vs SPY = {wr:.1%} (Wilson CI "
                        f"[{w.get('ci_low', 0):.2f}, {w.get('ci_high', 0):.2f}], N={n} entries)"
                        + (f"; avg slippage {slip:+.2%}" if slip is not None else "")
                        + " vs target 55% / red-line 45%."),
            ))
        else:
            components.append(build_metric(
                name="entry_triggers", module=MODULE, metric_type="pct", criticality="critical",
                estimator="wilson_winrate", measurement_horizon="intraday_to_exit",
                n_floor=30, target=0.55, red_line=0.45, source_path=ts_src, input_present=False,
                na_detail="entry_triggers: trigger_scorecard has no win-rate/entries this cycle.",
            ))
    else:
        components.append(build_metric(
            name="entry_triggers", module=MODULE, metric_type="pct", criticality="critical",
            estimator="wilson_winrate", measurement_horizon="intraday_to_exit",
            n_floor=30, target=0.55, red_line=0.45, source_path=ts_src, input_present=False,
            na_detail="entry_triggers: trigger_scorecard.json absent this cycle (OK-only persisted; lands on a Saturday run).",
        ))

    # 2. risk_guard (critical) — precision of blocks (% blocked that were losers).
    sb_src = src("shadow_book.json")
    clf = (shadow or {}).get("classification") if shadow and shadow.get("status") == "ok" else None
    if clf and clf.get("precision") is not None:
        tp, fp = int(clf.get("tp", 0)), int(clf.get("fp", 0))
        n_blk = tp + fp
        w = wilson_score_interval(tp, n_blk) if n_blk > 0 else {"status": "insufficient_data"}
        components.append(build_metric(
            name="risk_guard", module=MODULE, metric_type="pct", criticality="critical",
            estimator="wilson_precision",
            value=clf.get("precision"), n_samples=n_blk, n_floor=20, target=0.55, red_line=0.40,
            ci_low=w.get("ci_low"), ci_high=w.get("ci_high"),
            ci_method="wilson" if w.get("status") == "ok" else None, source_path=sb_src,
            reason=(f"risk_guard block-precision = {clf['precision']:.1%} (Wilson CI "
                    f"[{w.get('ci_low', 0):.2f}, {w.get('ci_high', 0):.2f}], N={n_blk} blocked); "
                    f"assessment={(shadow or {}).get('assessment')}."),
        ))
    else:
        components.append(build_metric(
            name="risk_guard", module=MODULE, metric_type="pct", criticality="critical",
            estimator="wilson_precision",
            n_floor=20, target=0.55, red_line=0.40, source_path=sb_src, input_present=False,
            na_detail="risk_guard: shadow_book.json absent or has no classification this cycle (OK-only persisted).",
        ))

    # 3. exit_rules (critical) — WINNER capture ratio (realized / max-favorable
    #    on trades that had a favorable move). Graded on the robust median over
    #    winners, NOT the legacy all-trade avg_capture_ratio — that mean is an
    #    unbounded ratio polluted by stopped-out losers (tiny MFE denominator)
    #    and read RED even when exits captured their winners well (ROADMAP
    #    L4554: it was a metric artifact, not an exit-timing defect — the real
    #    lever is entry quality, L4560). Pre-2026-06-07 artifacts without the
    #    winner-capture field fall back to the legacy mean.
    et_src = src("exit_timing.json")
    if exits and exits.get("status") == "ok":
        summ = exits.get("summary") or {}
        cap = summ.get("capture_winners_median")
        cap_label = "winner-capture-median"
        if cap is None:
            cap = summ.get("avg_capture_ratio")  # legacy fallback
            cap_label = "capture-ratio (legacy all-trade mean)"
        n_cap = summ.get("n_winners") or exits.get("n_roundtrips")
        components.append(build_metric(
            name="exit_rules", module=MODULE, metric_type="ratio", criticality="critical",
            estimator="winner_capture_median", measurement_horizon="per_hold",
            value=cap, n_samples=n_cap, n_floor=15, target=0.70, red_line=0.40,
            source_path=et_src, input_present=cap is not None,
            reason=(f"exit_rules {cap_label} = {cap:.2f} (N={n_cap} winners of "
                    f"{exits.get('n_roundtrips')} roundtrips, win_rate={summ.get('win_rate')}); "
                    f"diagnosis={exits.get('diagnosis')}, stop_eff_median={summ.get('stop_efficiency_median')}."
                    if cap is not None else None),
            na_detail="exit_rules: exit_timing has no capture ratio this cycle.",
        ))
    else:
        components.append(build_metric(
            name="exit_rules", module=MODULE, metric_type="ratio", criticality="critical",
            estimator="winner_capture_median", measurement_horizon="per_hold",
            n_floor=20, target=0.70, red_line=0.40, source_path=et_src, input_present=False,
            na_detail="exit_rules: exit_timing.json absent this cycle (OK-only persisted; lands on a Saturday run).",
        ))

    # 4. excursion (supporting) — MFE/MAE process quality.
    pe_src = src("portfolio_excursion.json")
    if exc and exc.get("status") == "ok" and exc.get("mean_mfe_mae_ratio") is not None:
        components.append(build_metric(
            name="excursion", module=MODULE, metric_type="ratio", criticality="supporting",
            value=exc.get("mean_mfe_mae_ratio"), n_samples=exc.get("n"), n_floor=20,
            target=1.5, red_line=0.8, source_path=pe_src,
            reason=f"excursion mean MFE/MAE = {exc['mean_mfe_mae_ratio']:.2f}, pct_high_quality={exc.get('pct_high_quality')}.",
        ))
    else:
        components.append(build_metric(
            name="excursion", module=MODULE, metric_type="ratio", criticality="supporting",
            n_floor=20, target=1.5, red_line=0.8, source_path=pe_src, input_present=False,
            na_detail="excursion: portfolio_excursion.json absent this cycle (persists from a post-2026-06-04 Saturday run, B1a #279).",
        ))

    # 5. position_sizing (supporting) — B1c-deferred (sizing_ab genuinely unwired).
    components.append(build_metric(
        name="position_sizing", module=MODULE, metric_type="ratio", criticality="supporting",
        n_floor=20, target=0.0, red_line=-0.3, source_path=src("sizing_ab.json"), implemented=False,
        na_detail="position_sizing: sizing_ab analysis genuinely unwired (evaluate.py hardcodes None) — ROADMAP B1c.",
    ))

    # 6. action_entropy (diagnostic) — not persisted.
    components.append(build_metric(
        name="action_entropy", module=MODULE, metric_type="ratio", criticality="diagnostic",
        n_floor=20, target=0.7, red_line=0.3, source_path=src("action_entropy.json"), input_present=False,
        na_detail="action_entropy: decision-stream entropy not persisted this cycle.",
    ))

    # 7. reconciliation_integrity (critical) — EOD NAV parity audit, not yet produced.
    components.append(build_metric(
        name="reconciliation_integrity", module=MODULE, metric_type="pct", criticality="critical",
        estimator="reconciliation_match_rate",
        n_floor=1, target=1.0, red_line=0.0, source_path=f"s3://{bucket}/trades/", implemented=False,
        na_detail="reconciliation_integrity: daemon-vs-IB NAV parity audit not yet produced/persisted for the report card.",
    ))

    return build_tile(MODULE, components)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Build the Executor tile from executor diagnostics.")
    parser.add_argument("--bucket", default="alpha-engine-research")
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(json.dumps(build_executor_tile(args.bucket, args.date), indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
