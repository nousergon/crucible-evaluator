"""
behavioral.py — Tile 7: Behavioral anomaly (RC v2, L4514 / config#698).

Grades the system's decision BEHAVIOR (is the book churning, flip-flopping,
paying away its alpha, or drifting?) — orthogonal to the outcome tiles. Five
components:

  - turnover            : executor L4515 tripwire result, read from the most
                          recent ``predictor/optimizer_shadow/{date}.json``
                          (daily artifact — walked back <= 7 calendar days
                          from run_date, which is a Saturday on report cards)
  - decision_reversal   : per-ticker EXIT -> re-ENTER churn rate
  - conviction_stability: rolling std of composite score per ticker
  - cost_adjusted_quality: median realized alpha net of entry slippage
  - portfolio_state_drift: day-over-day L1 weight drift

Components 2-5 read ``backtest/{run_date}/behavioral_anomaly.json``
(ALWAYS-EMIT by the backtester since the L4514 arc — an absent artifact
means the Saturday chain predates the arc or didn't run; per-component
``insufficient_data`` is graded distinctly).

SOAK POSTURE: every component is supporting/diagnostic — never critical —
until baselines accumulate (institutional observe-first; the bands below
are provisional and revisited once ~4 weeks of artifacts exist).
``reliability="medium"`` marks the provisional bands per the L4562 contract.

Spec: config#698 design comment (2026-06-11).
"""

from __future__ import annotations

import datetime as _dt
import json
import logging

import boto3
from botocore.exceptions import ClientError

from grading.artifacts import get_json_windowed
from grading.metric_record import build_metric
from grading.module_agg import build_tile

logger = logging.getLogger(__name__)

MODULE = "behavioral"

_SHADOW_PREFIX = "predictor/optimizer_shadow"
_SHADOW_LOOKBACK_DAYS = 7


def _get_json(s3, bucket: str, key: str) -> dict | None:
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        logger.error("S3 read failed for s3://%s/%s: %s", bucket, key, e)
        raise
    return json.loads(resp["Body"].read())


def _latest_shadow_tripwire(s3, bucket: str, run_date: str) -> tuple[dict | None, str | None]:
    """Most recent optimizer_shadow turnover_tripwire block at/before run_date.

    The shadow artifact is written on weekdays by the morning planner; the
    report card runs on Saturdays, so walk back up to a week.
    """
    try:
        day = _dt.date.fromisoformat(run_date)
    except ValueError:
        return None, None
    for delta in range(_SHADOW_LOOKBACK_DAYS + 1):
        d = (day - _dt.timedelta(days=delta)).isoformat()
        key = f"{_SHADOW_PREFIX}/{d}.json"
        doc = _get_json(s3, bucket, key)
        if doc is not None:
            return (doc.get("turnover_tripwire") or None), f"s3://{bucket}/{key}"
    return None, None


def build_behavioral_tile(bucket: str, run_date: str, s3_client=None) -> dict:
    """Build the Behavioral tile from behavioral_anomaly.json + the tripwire."""
    s3 = s3_client or boto3.client("s3")
    # Windowed resolution (config#1190): freshest within the trailing window.
    ba, _, _, _ba_key = get_json_windowed(s3, bucket, "backtest/{date}/behavioral_anomaly.json", run_date)
    ba_src = f"s3://{bucket}/{_ba_key}" if _ba_key else f"s3://{bucket}/backtest/{run_date}/behavioral_anomaly.json"
    components = []

    def _sub(name: str) -> dict | None:
        if not ba:
            return None
        comp = ba.get(name)
        return comp if comp and comp.get("status") == "ok" else None

    def _na_detail(name: str) -> str:
        if ba is None:
            return f"{name}: behavioral_anomaly.json absent this cycle (lands on a Saturday run post-L4514)."
        sub = (ba.get(name) or {})
        return f"{name}: component status={sub.get('status', 'missing')} this cycle."

    # 1. turnover (diagnostic) — L4515 tripwire, surfaced not recomputed.
    trip, trip_src = _latest_shadow_tripwire(s3, bucket, run_date)
    if trip and trip.get("status") == "ok" and trip.get("rolling_sum") is not None:
        band = trip.get("rolling_band")
        components.append(build_metric(
            name="turnover", module=MODULE, metric_type="ratio", criticality="diagnostic",
            estimator="l4515_tripwire_rolling_sum", measurement_horizon=f"{trip.get('rolling_days', 5)}d_rolling",
            reliability="high",  # the tripwire itself is production-paged
            value=trip["rolling_sum"], n_samples=int(trip.get("n_days_used") or 0), n_floor=1,
            target=(band / 2 if band else None), red_line=band, source_path=trip_src or "",
            reason=(f"one-way turnover rolling sum {trip['rolling_sum']:.1%} over "
                    f"{trip.get('n_days_used')} session(s) vs band {band:.0%}; "
                    f"daily_breach={trip.get('daily_breach')}, rolling_breach={trip.get('rolling_breach')} "
                    f"(executor L4515 tripwire pages on breach independently)."),
        ))
    else:
        components.append(build_metric(
            name="turnover", module=MODULE, metric_type="ratio", criticality="diagnostic",
            estimator="l4515_tripwire_rolling_sum", measurement_horizon="5d_rolling",
            n_floor=1, source_path=trip_src or f"s3://{bucket}/{_SHADOW_PREFIX}/", input_present=False,
            na_detail=("turnover: no optimizer_shadow artifact with an ok tripwire block in the "
                       f"trailing {_SHADOW_LOOKBACK_DAYS}d (status="
                       f"{(trip or {}).get('status', 'absent')})."),
        ))

    # 2. decision_reversal (supporting) — lower is better (target < red_line).
    rev = _sub("decision_reversal")
    if rev:
        components.append(build_metric(
            name="decision_reversal", module=MODULE, metric_type="pct", criticality="supporting",
            estimator="reversal_rate_rolling_window", measurement_horizon=f"{rev.get('window_days', 10)}d_window",
            reliability="medium",
            value=rev["reversal_rate"], n_samples=int(rev.get("n_exits") or 0), n_floor=10,
            target=0.10, red_line=0.30, source_path=ba_src,
            reason=(f"{rev.get('n_reversals')}/{rev.get('n_exits')} exits re-entered within "
                    f"{rev.get('window_days')}td (rate {rev['reversal_rate']:.1%}) vs provisional "
                    f"target 10% / red-line 30%."),
        ))
    else:
        components.append(build_metric(
            name="decision_reversal", module=MODULE, metric_type="pct", criticality="supporting",
            estimator="reversal_rate_rolling_window", measurement_horizon="10d_window",
            n_floor=10, target=0.10, red_line=0.30, source_path=ba_src, input_present=False,
            na_detail=_na_detail("decision_reversal"),
        ))

    # 3. conviction_stability (diagnostic) — lower is better.
    conv = _sub("conviction_stability")
    if conv:
        components.append(build_metric(
            name="conviction_stability", module=MODULE, metric_type="count", criticality="diagnostic",
            estimator="median_rolling_score_std", measurement_horizon=f"{conv.get('window_days', 90)}d_window",
            reliability="medium",
            value=conv["median_score_std"], n_samples=int(conv.get("n_tickers") or 0), n_floor=5,
            target=5.0, red_line=15.0, source_path=ba_src,
            reason=(f"median per-ticker score std {conv['median_score_std']:.2f} "
                    f"(p90 {conv.get('p90_score_std')}, N={conv.get('n_tickers')} tickers) on the "
                    f"0-100 composite scale vs provisional target 5 / red-line 15."),
        ))
    else:
        components.append(build_metric(
            name="conviction_stability", module=MODULE, metric_type="count", criticality="diagnostic",
            estimator="median_rolling_score_std", measurement_horizon="90d_window",
            n_floor=5, target=5.0, red_line=15.0, source_path=ba_src, input_present=False,
            na_detail=_na_detail("conviction_stability"),
        ))

    # 4. cost_adjusted_quality (supporting) — higher is better.
    cost = _sub("cost_adjusted_quality")
    if cost:
        components.append(build_metric(
            name="cost_adjusted_quality", module=MODULE, metric_type="pct", criticality="supporting",
            estimator="median_net_alpha_after_slippage", measurement_horizon="per_roundtrip",
            reliability="medium",
            value=cost["median_net_alpha_pct"], n_samples=int(cost.get("n_roundtrips") or 0), n_floor=15,
            target=0.5, red_line=0.0, source_path=ba_src,
            reason=(f"median net alpha after entry slippage {cost['median_net_alpha_pct']:+.2f}% "
                    f"(gross {cost.get('median_gross_alpha_pct'):+.2f}%, median slippage "
                    f"{cost.get('median_slippage_pct'):.3f}%, N={cost.get('n_roundtrips')}); "
                    f"cost-drag fraction {cost.get('cost_drag_fraction')} of winners."),
        ))
    else:
        components.append(build_metric(
            name="cost_adjusted_quality", module=MODULE, metric_type="pct", criticality="supporting",
            estimator="median_net_alpha_after_slippage", measurement_horizon="per_roundtrip",
            n_floor=15, target=0.5, red_line=0.0, source_path=ba_src, input_present=False,
            na_detail=_na_detail("cost_adjusted_quality"),
        ))

    # 5. portfolio_state_drift (diagnostic) — lower is better.
    drift = _sub("portfolio_state_drift")
    if drift:
        components.append(build_metric(
            name="portfolio_state_drift", module=MODULE, metric_type="ratio", criticality="diagnostic",
            estimator="median_daily_l1_weight_drift", measurement_horizon="day_over_day",
            reliability="medium",
            value=drift["median_daily_drift"], n_samples=int(drift.get("n_days") or 0), n_floor=10,
            target=0.05, red_line=0.20, source_path=ba_src,
            reason=(f"median daily one-way L1 weight drift {drift['median_daily_drift']:.1%} "
                    f"(max {drift.get('max_daily_drift'):.1%}, {drift.get('n_spike_days')} spike day(s) "
                    f"> {drift.get('spike_threshold'):.0%}, N={drift.get('n_days')} days)."),
        ))
    else:
        components.append(build_metric(
            name="portfolio_state_drift", module=MODULE, metric_type="ratio", criticality="diagnostic",
            estimator="median_daily_l1_weight_drift", measurement_horizon="day_over_day",
            n_floor=10, target=0.05, red_line=0.20, source_path=ba_src, input_present=False,
            na_detail=_na_detail("portfolio_state_drift"),
        ))

    return build_tile(MODULE, components)
