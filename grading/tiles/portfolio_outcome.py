"""
portfolio_outcome.py — Tile 0: Portfolio Outcome (RC v2, outcome-leads).

Grades the system's *product*: paper-portfolio P&L vs SPY. This is the
institutional headline (RC v2 Principle 7) — read first, components decompose
the why. Source: ``s3://{bucket}/trades/eod_pnl.csv`` (the executor's EOD
reconciliation export; columns `date,portfolio_nav,daily_return_pct,
spy_return_pct,daily_alpha_pct,...` — the `_pct` columns are in PERCENT, so we
divide by 100 to get the daily fractions the lib quant battery expects).

Each component is a ``MetricRecord`` with value + CI + N vs floor + target /
red-line + status, per the Tile-0 spec in
``system-report-card-revamp-260522.md``. Metrics not supportable from this CSV
alone (DSR's trial count, regime-weighted alpha's macro tags) emit a *specific*
N/A — never a silent omission.
"""

from __future__ import annotations

import csv
import io
import logging
import math

import boto3
import numpy as np
from botocore.exceptions import ClientError

from alpha_engine_lib.quant.risk_measures import historical_cvar
from alpha_engine_lib.quant.riskstats import max_drawdown, sharpe_ratio, sortino_ratio
from alpha_engine_lib.quant.stats.dsr import compute_psr
from alpha_engine_lib.quant.stats.intervals import bootstrap_ci, wilson_score_interval

from grading.metric_record import build_metric
from grading.module_agg import build_tile

logger = logging.getLogger(__name__)

MODULE = "portfolio_outcome"
EOD_PNL_KEY = "trades/eod_pnl.csv"
_TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# eod_pnl.csv reader
# ---------------------------------------------------------------------------

class EodPnlSeries:
    """Parsed daily series from eod_pnl.csv (returns as fractions, not pct)."""

    def __init__(self, dates, nav, port, spy, alpha):
        self.dates = dates
        self.nav = nav            # portfolio NAV level series
        self.port = port          # daily portfolio returns (fraction)
        self.spy = spy            # daily SPY returns (fraction)
        self.alpha = alpha        # daily active return port-spy (fraction)

    @property
    def n(self) -> int:
        return len(self.port)


def read_eod_pnl(bucket: str, s3_client=None) -> EodPnlSeries | None:
    """Read + parse eod_pnl.csv from S3. None if absent (NoSuchKey); raises on
    any other S3 error (fail-loud per no_silent_fails)."""
    s3 = s3_client or boto3.client("s3")
    try:
        resp = s3.get_object(Bucket=bucket, Key=EOD_PNL_KEY)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            logger.warning("eod_pnl.csv absent at s3://%s/%s — Portfolio Outcome tile N/A", bucket, EOD_PNL_KEY)
            return None
        logger.error("S3 read failed for s3://%s/%s: %s", bucket, EOD_PNL_KEY, e)
        raise

    text = resp["Body"].read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    rows = [r for r in rows if r.get("date")]
    rows.sort(key=lambda r: r["date"])

    dates, nav, port, spy, alpha = [], [], [], [], []
    for r in rows:
        try:
            nav.append(float(r["portfolio_nav"]))
            p = float(r["daily_return_pct"]) / 100.0
            s = float(r["spy_return_pct"]) / 100.0
        except (TypeError, ValueError):
            continue
        dates.append(r["date"])
        port.append(p)
        spy.append(s)
        a = r.get("daily_alpha_pct")
        alpha.append(float(a) / 100.0 if a not in (None, "") else p - s)
    return EodPnlSeries(dates, nav, port, spy, alpha)


# ---------------------------------------------------------------------------
# statistic helpers (consistent point-estimate + bootstrap statistic)
# ---------------------------------------------------------------------------

def _ann_sharpe(arr: np.ndarray) -> float:
    v = sharpe_ratio(list(arr))
    return float("nan") if v is None else v


def _ann_sortino(arr: np.ndarray) -> float:
    v = sortino_ratio(list(arr))
    return float("nan") if v is None else v


def _ann_ir(arr: np.ndarray) -> float:
    """Annualized information ratio: mean active / stdev active × √252."""
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return float("nan")
    sd = a.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(a.mean() / sd * math.sqrt(_TRADING_DAYS))


def _neg_cvar(arr: np.ndarray) -> float:
    """CVaR(95) expressed as a (negative) daily return — lib returns the loss
    magnitude as a positive fraction, so negate to put it on the return scale."""
    v = historical_cvar(list(arr), confidence=0.95)
    return float("nan") if v is None else -v


def _ci(stat, arr: list[float]) -> tuple[float | None, float | None, str | None]:
    res = bootstrap_ci(arr, statistic=stat)
    if res.get("status") != "ok":
        return None, None, None
    return res["ci_low"], res["ci_high"], "bootstrap"


def _max_dd_duration_days(nav: list[float]) -> int:
    """Longest run (in rows ≈ trading days) the NAV spends below its prior peak."""
    peak = -math.inf
    run = best = 0
    for v in nav:
        if v >= peak:
            peak = v
            run = 0
        else:
            run += 1
            best = max(best, run)
    return best


# ---------------------------------------------------------------------------
# tile builder
# ---------------------------------------------------------------------------

def build_portfolio_outcome_tile(bucket: str, s3_client=None, *, n_trials: int | None = None) -> dict:
    """Build the Portfolio Outcome tile from eod_pnl.csv.

    ``n_trials`` (DSR selection-bias count) is unset for the live book today —
    DSR therefore emits N/A-NOT-IMPL until a documented cumulative trial count
    lands (mirrors L4469 W1.3b's deferred trial-count tracking).
    """
    src = f"s3://{bucket}/{EOD_PNL_KEY}"
    series = read_eod_pnl(bucket, s3_client=s3_client)

    if series is None:
        # Whole tile N/A-MISSING-INPUT — one record per component, specific reason.
        def miss(name, mt, crit, floor, tgt, rl):
            return build_metric(
                name=name, module=MODULE, metric_type=mt, criticality=crit, n_floor=floor,
                target=tgt, red_line=rl, source_path=src, input_present=False,
                na_detail=f"{name}: trades/eod_pnl.csv not present this cycle — no EOD reconciliation export to grade.",
            )

        components = [
            miss("sharpe_ratio", "sharpe", "critical", 60, 1.0, 0.0),
            miss("information_ratio", "ratio", "critical", 60, 0.5, 0.0),
            miss("psr", "pct", "critical", 60, 0.95, 0.50),
            miss("alpha_vs_spy", "log_return", "critical", 60, 0.0, -0.05),
            miss("max_drawdown", "ratio", "critical", 2, -0.15, -0.25),
            miss("sortino_ratio", "sharpe", "supporting", 60, 1.5, 0.5),
            miss("calmar_ratio", "ratio", "supporting", 90, 1.0, 0.0),
            miss("cvar_95_daily", "ratio", "supporting", 60, -0.01, -0.04),
        ]
        return build_tile(MODULE, components)

    n = series.n
    port = series.port
    nav = series.nav
    active = series.alpha
    components = []

    # 1. Sharpe (critical)
    sharpe = sharpe_ratio(port)
    s_lo, s_hi, s_m = _ci(_ann_sharpe, port) if sharpe is not None else (None, None, None)
    components.append(build_metric(
        name="sharpe_ratio", module=MODULE, metric_type="sharpe", criticality="critical",
        value=sharpe, n_samples=n, n_floor=60, target=1.0, red_line=0.0,
        ci_low=s_lo, ci_high=s_hi, ci_method=s_m, source_path=src,
    ))

    # 2. Information ratio (critical)
    ir = _ann_ir(np.asarray(active))
    ir = None if (ir is None or math.isnan(ir)) else ir
    i_lo, i_hi, i_m = _ci(_ann_ir, active) if ir is not None else (None, None, None)
    components.append(build_metric(
        name="information_ratio", module=MODULE, metric_type="ratio", criticality="critical",
        value=ir, n_samples=n, n_floor=60, target=0.5, red_line=0.0,
        ci_low=i_lo, ci_high=i_hi, ci_method=i_m, source_path=src,
    ))

    # 3. PSR (critical) — P(true Sharpe > 0). Probability, no CI.
    psr_res = compute_psr(np.asarray(port), sharpe_benchmark=0.0)
    psr_val = psr_res.get("psr") if psr_res.get("status") == "ok" else None
    components.append(build_metric(
        name="psr", module=MODULE, metric_type="pct", criticality="critical",
        value=psr_val, n_samples=psr_res.get("n", n), n_floor=60, target=0.95, red_line=0.50,
        source_path=src,
    ))

    # 4. Alpha vs SPY (critical) — cumulative log-alpha since inception.
    log_alpha = sum(
        math.log1p(p) - math.log1p(s)
        for p, s in zip(port, series.spy)
        if (1 + p) > 0 and (1 + s) > 0
    )
    components.append(build_metric(
        name="alpha_vs_spy", module=MODULE, metric_type="log_return", criticality="critical",
        value=log_alpha, n_samples=n, n_floor=60, target=0.0, red_line=-0.05, source_path=src,
    ))

    # 5. Max drawdown (critical) — point obs from the NAV series.
    mdd = max_drawdown(nav)
    components.append(build_metric(
        name="max_drawdown", module=MODULE, metric_type="ratio", criticality="critical",
        value=mdd, n_samples=len(nav), n_floor=2, target=-0.15, red_line=-0.25, source_path=src,
    ))

    # 6. Sortino (supporting)
    sortino = sortino_ratio(port)
    so_lo, so_hi, so_m = _ci(_ann_sortino, port) if sortino is not None else (None, None, None)
    components.append(build_metric(
        name="sortino_ratio", module=MODULE, metric_type="sharpe", criticality="supporting",
        value=sortino, n_samples=n, n_floor=60, target=1.5, red_line=0.5,
        ci_low=so_lo, ci_high=so_hi, ci_method=so_m, source_path=src,
    ))

    # 7. Calmar (supporting) — annualized return / |max drawdown|.
    calmar = None
    if mdd is not None and mdd < 0 and len(nav) >= 2 and nav[0] > 0:
        total_ret = nav[-1] / nav[0] - 1.0
        ann_ret = (1 + total_ret) ** (_TRADING_DAYS / n) - 1.0 if n > 0 else None
        if ann_ret is not None:
            calmar = ann_ret / abs(mdd)
    components.append(build_metric(
        name="calmar_ratio", module=MODULE, metric_type="ratio", criticality="supporting",
        value=calmar, n_samples=n, n_floor=90, target=1.0, red_line=0.0, source_path=src,
    ))

    # 8. CVaR(95) daily (supporting) — mean worst-5% daily return.
    cvar = _neg_cvar(np.asarray(port))
    cvar = None if math.isnan(cvar) else cvar
    cv_lo, cv_hi, cv_m = _ci(_neg_cvar, port) if cvar is not None else (None, None, None)
    components.append(build_metric(
        name="cvar_95_daily", module=MODULE, metric_type="ratio", criticality="supporting",
        value=cvar, n_samples=n, n_floor=60, target=-0.01, red_line=-0.04,
        ci_low=cv_lo, ci_high=cv_hi, ci_method=cv_m, source_path=src,
    ))

    # 9. Hit rate daily (diagnostic) — % days portfolio beats SPY. Wilson CI.
    wins = sum(1 for a in active if a > 0)
    w = wilson_score_interval(wins, n) if n > 0 else {"status": "insufficient_data"}
    hit = w["rate"] if w.get("status") == "ok" else None
    components.append(build_metric(
        name="hit_rate_daily", module=MODULE, metric_type="pct", criticality="diagnostic",
        value=hit, n_samples=n, n_floor=60, target=0.55, red_line=0.45,
        ci_low=w.get("ci_low"), ci_high=w.get("ci_high"),
        ci_method="wilson" if w.get("status") == "ok" else None, source_path=src,
    ))

    # 10. Beta vs SPY (diagnostic, two-sided band 0.7–1.1) — explicit status.
    beta = None
    if n >= 2:
        p_arr, s_arr = np.asarray(port), np.asarray(series.spy)
        var_s = s_arr.var(ddof=1)
        if var_s > 0:
            beta = float(np.cov(p_arr, s_arr, ddof=1)[0, 1] / var_s)
    if beta is None:
        beta_status = "N/A-LOW-N"
    elif 0.7 <= beta <= 1.1:
        beta_status = "GREEN"
    elif 0.3 < beta < 1.5:
        beta_status = "WATCH"
    else:
        beta_status = "RED"
    components.append(build_metric(
        name="beta_vs_spy", module=MODULE, metric_type="ratio", criticality="diagnostic",
        value=beta, n_samples=n, n_floor=60, source_path=src, status=beta_status,
        reason=(f"beta_vs_spy = {beta:.3g} (target band 0.7–1.1)." if beta is not None
                else "beta_vs_spy: too few observations to estimate."),
    ))

    # 11. Max-DD duration (diagnostic) — days underwater since prior peak.
    components.append(build_metric(
        name="max_dd_duration_days", module=MODULE, metric_type="duration", criticality="diagnostic",
        value=float(_max_dd_duration_days(nav)), n_samples=len(nav), n_floor=2,
        target=30.0, red_line=90.0, higher_is_better=False, source_path=src,
    ))

    # 12. DSR (supporting) — needs a documented trial count; not yet tracked.
    if n_trials is not None and n_trials >= 1:
        from alpha_engine_lib.quant.stats.dsr import compute_dsr
        dsr_res = compute_dsr(np.asarray(port), n_trials=n_trials)
        dsr_val = dsr_res.get("dsr") if dsr_res.get("status") == "ok" else None
        components.append(build_metric(
            name="dsr", module=MODULE, metric_type="pct", criticality="supporting",
            value=dsr_val, n_samples=dsr_res.get("n", n), n_floor=60, target=0.95, red_line=0.50,
            source_path=src,
        ))
    else:
        components.append(build_metric(
            name="dsr", module=MODULE, metric_type="pct", criticality="supporting",
            n_floor=60, target=0.95, red_line=0.50, source_path=src, implemented=False,
            na_detail="dsr: needs a documented cumulative strategy-trial count for the live book; tracked under L4469 W1.3b.",
        ))

    # 13. Regime-weighted alpha (critical) — needs macro regime tags per date.
    components.append(build_metric(
        name="regime_weighted_alpha", module=MODULE, metric_type="log_return", criticality="critical",
        n_floor=30, target=0.0, red_line=0.0, source_path=src, input_present=False,
        na_detail=("regime_weighted_alpha: requires per-date macro regime tags "
                   "(bull/bear/neutral/caution) to decompose; eod_pnl.csv carries only "
                   "date/nav/returns. Needs a regime join — ROADMAP follow-up."),
    ))

    return build_tile(MODULE, components)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Build the Portfolio Outcome tile from eod_pnl.csv.")
    parser.add_argument("--bucket", default="alpha-engine-research")
    parser.add_argument("--n-trials", type=int, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    tile = build_portfolio_outcome_tile(args.bucket, n_trials=args.n_trials)
    print(json.dumps(tile, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
