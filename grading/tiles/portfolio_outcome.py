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
alone (DSR's trial count) emit a *specific* N/A — never a silent omission.
``regime_weighted_alpha`` decomposes daily alpha by macro regime — it joins
``eod_pnl.csv`` dates against the per-date ``market_regime`` tag persisted at
``s3://{bucket}/signals/{date}/signals.json`` (config#857 C2-fu).
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
from statistics import NormalDist

import boto3
import numpy as np
from botocore.exceptions import ClientError

from nousergon_lib.quant.risk_measures import historical_cvar
from nousergon_lib.quant.riskstats import max_drawdown, sharpe_ratio, sortino_ratio
from nousergon_lib.quant.stats.dsr import compute_psr
from nousergon_lib.quant.stats.intervals import bootstrap_ci, newey_west_se, wilson_score_interval

from grading.history import CardHistory
from grading.metric_record import build_metric
from grading.thresholds.registry import DEFAULT_BAND, resolve as resolve_band
from grading.module_agg import build_tile
from grading.attribution import ATTRIBUTION_LATEST_KEY, build_attribution
from grading.power import annotate_power_all
from grading.units import (
    ANNUALIZED_RATIO,
    DAYS,
    FRACTION,
    LOG_RETURN,
    PCT,
    PROBABILITY,
    RATIO,
)

logger = logging.getLogger(__name__)

MODULE = "portfolio_outcome"
EOD_PNL_KEY = "trades/eod_pnl.csv"
SIGNALS_KEY_TEMPLATE = "signals/{date}/signals.json"
# alpha-engine-config-I9684 — the model-zoo rotation leaderboard, which is the
# ONLY artifact in the fleet that persists a CSCV/PBO verdict. crucible-predictor
# writes it from `training/model_zoo.py::_selection_pbo` on every rotation, over
# the aligned (n_splits x n_specs) per-spec IC matrix of the specs that actually
# raced. The evaluator reads the verdict; it never recomputes it and it never
# reconstructs a trial matrix of its own.
MODEL_ZOO_LEADERBOARD_KEY = "predictor/model_zoo/leaderboard/latest.json"
# The floor is NOT authored here. `nousergon_lib.quant.stats.pbo.cscv_pbo`
# declares `min_splits: int = 4` as the point below which it refuses to return a
# number at all; that is the floor this component reports N against, so the card
# and the engine cannot disagree about when PBO is underpowered.
_PBO_MIN_SPLITS = 4
_TRADING_DAYS = 252
_TRADING_DAYS_PER_MONTH = 21
_ALPHA_TREND_N_FLOOR = 60
_REGIME_ALPHA_N_FLOOR = 30
_REGIME_MIN_BUCKET_N = 5     # min daily samples for one regime bucket to qualify
_REGIME_MIN_BUCKETS = 2      # min qualifying regime buckets to decompose at all


# ---------------------------------------------------------------------------
# eod_pnl.csv reader
# ---------------------------------------------------------------------------

class EodPnlSeries:
    """Parsed daily series from eod_pnl.csv (returns as fractions, not pct)."""

    def __init__(self, dates, nav, port, spy, alpha, rows_dropped: int = 0):
        self.dates = dates
        self.nav = nav            # portfolio NAV level series
        self.port = port          # daily portfolio returns (fraction)
        self.spy = spy            # daily SPY returns (fraction)
        self.alpha = alpha        # daily active return port-spy (fraction)
        self.rows_dropped = rows_dropped  # config#2885: silently-unparseable rows

    @property
    def n(self) -> int:
        return len(self.port)


def cumulative_log_alpha(port, spy) -> float:
    """Cumulative log-domain alpha vs SPY: Σ (log(1+r_p) − log(1+r_spy)).

    Extracted from ``build_portfolio_outcome_tile`` so the runtime attestation
    (``grading/attestation.py``) can pin the ``alpha_vs_spy`` headline to a
    closed-form expectation — a metric computed only inside a 400-line tile
    builder cannot be attested without standing up its whole input surface.

    Days where either leg would take the log of a non-positive quantity (a −100%
    or worse daily return) are skipped rather than producing ``-inf``; that guard
    is part of the definition and is attested with it.
    """
    return sum(
        math.log1p(p) - math.log1p(s)
        for p, s in zip(port, spy)
        if (1 + p) > 0 and (1 + s) > 0
    )


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
    rows_dropped = 0
    for r in rows:
        try:
            nav.append(float(r["portfolio_nav"]))
            p = float(r["daily_return_pct"]) / 100.0
            s = float(r["spy_return_pct"]) / 100.0
        except (TypeError, ValueError):
            rows_dropped += 1
            continue
        dates.append(r["date"])
        port.append(p)
        spy.append(s)
        a = r.get("daily_alpha_pct")
        alpha.append(float(a) / 100.0 if a not in (None, "") else p - s)
    if rows_dropped:
        logger.warning(
            "read_eod_pnl: %d row(s) dropped from eod_pnl.csv "
            "(unparseable portfolio_nav or daily_return_pct)",
            rows_dropped,
        )
    return EodPnlSeries(dates, nav, port, spy, alpha, rows_dropped=rows_dropped)


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
    """Annualized information ratio: mean active / stdev active × √252.

    IR is Sharpe computed on the active-return series (risk-free = 0), so this
    is the same `riskstats.sharpe_ratio` the two adapters above call rather
    than a fifth copy of `mean / std * sqrt(252)` (config-I7597). Non-finite
    inputs are dropped first — the library takes a clean series — and the
    library's `None` for undefined is mapped to this tile's `nan` sentinel,
    exactly as `_ann_sharpe`/`_ann_sortino` already do.
    """
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return float("nan")
    v = sharpe_ratio([float(x) for x in a], periods_per_year=_TRADING_DAYS)
    return float("nan") if v is None else float(v)


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
# alpha_trend — OLS slope of daily alpha with HAC (Newey-West) SE (config#1962)
# ---------------------------------------------------------------------------

def _ols_slope_hac(y: list[float]) -> dict | None:
    """OLS slope of ``y`` on a trading-day index, with a HAC (Newey-West) SE.

    The OLS normal equations make ``sum(x_t * e_t) == 0`` exactly (``x_t`` the
    demeaned day index, ``e_t`` the fit residual), so that product series is
    zero-mean by construction — which means ``newey_west_se``'s "HAC variance of
    a series mean" machinery gives the HAC variance of the slope estimator
    directly: ``Var(b) = n * LRV(x*e) / S_xx**2``. This reuses the lib's Bartlett-
    kernel estimator rather than adding a statsmodels/scipy dependency for a
    single coefficient SE.
    """
    n = len(y)
    if n < 2:
        return None
    t = np.arange(n, dtype=float)
    x = t - t.mean()
    s_xx = float(np.dot(x, x))
    if s_xx <= 0:
        return None
    y_arr = np.asarray(y, dtype=float)
    b = float(np.dot(x, y_arr) / s_xx)
    a = float(y_arr.mean() - b * t.mean())
    resid = y_arr - (a + b * t)
    nw = newey_west_se(x * resid)
    if nw.get("status") != "ok":
        return None
    se_b = n * nw["se"] / s_xx
    return {"slope": b, "se": se_b, "resid": resid}


def _effective_n(resid: np.ndarray) -> int:
    """Autocorrelation-adjusted effective N: ``n * Var_iid / Var_HAC``, measured
    on the DETRENDED residuals (not the raw series).

    Daily alpha is autocorrelated (overlapping positions), so the raw row count
    overstates how many *independent* observations back the trend read. Using
    the residuals (rather than the raw series) isolates the autocorrelation of
    the noise around the trend line from the trend itself — a strong genuine
    drift must not get penalized as "low information" just because the level
    series it produces is highly autocorrelated.
    """
    n = len(resid)
    if n < 2:
        return n
    centered = resid - resid.mean()
    gamma0 = float(np.dot(centered, centered) / n)
    nw = newey_west_se(resid)
    if nw.get("status") != "ok" or gamma0 <= 0:
        return n
    inflation = (nw["se"] ** 2) * n / gamma0
    if inflation <= 0:
        return n
    return max(1, min(n, int(round(n / inflation))))


def _build_alpha_trend(alpha_pct: list[float], src: str) -> object:
    """``alpha_trend`` component: is daily alpha statistically improving?

    Descriptive dashboard (crucible-dashboard PR350) shows the monthly-bucket
    alpha; per derive-don't-transcribe this is the statistical answer the
    dashboard must not adjudicate itself — an OLS trend on the full
    ``daily_alpha_pct`` history, HAC SE, honestly gated on an
    autocorrelation-adjusted effective N (not the raw row count).
    """
    name = "alpha_trend"
    n = len(alpha_pct)
    fit = _ols_slope_hac(alpha_pct)
    n_eff = _effective_n(fit["resid"]) if fit is not None else n

    if fit is None or n_eff < 0.5 * _ALPHA_TREND_N_FLOOR:
        return build_metric(
            name=name, module=MODULE, metric_type="pct", criticality="diagnostic",
            estimator="ols_slope_newey_west_hac", measurement_horizon="since_inception",
            n_floor=_ALPHA_TREND_N_FLOOR, n_samples=n_eff, band="unbanded", source_path=src,
            status="N/A-LOW-N",
            reason=(
                f"{name}: N_eff={n_eff} (raw N={n}) below the autocorrelation-adjusted "
                f"floor ({0.5 * _ALPHA_TREND_N_FLOOR:.0f}); daily alpha is autocorrelated via "
                f"overlapping positions, so too few independent observations exist yet for a "
                f"trend read."
            ),
        )

    monthly = fit["slope"] * _TRADING_DAYS_PER_MONTH
    se_monthly = fit["se"] * _TRADING_DAYS_PER_MONTH
    z = fit["slope"] / fit["se"] if fit["se"] > 0 else 0.0
    # Two-sided p on the raw slope test — NOT BH-FDR-adjusted (this is a single
    # diagnostic metric, not a family of critical tests); reusing the schema's
    # only p-value slot since MetricRecord has no separate raw-p field.
    p = 2.0 * (1.0 - NormalDist().cdf(abs(z))) if fit["se"] > 0 else 1.0
    ci_low = monthly - 1.96 * se_monthly
    ci_high = monthly + 1.96 * se_monthly

    if ci_low > 0:
        status, verdict = "GREEN", "statistically significant positive drift"
    elif ci_high < 0:
        status, verdict = "RED", "statistically significant negative drift"
    else:
        status, verdict = "WATCH", "not distinguishable from zero"
    direction = "positive" if monthly >= 0 else "negative"

    reason = (
        f"{name}: {direction} drift ({monthly:+.2f}%/mo), {verdict} at n={n} "
        f"(n_eff={n_eff}, 95% CI [{ci_low:+.2f}, {ci_high:+.2f}]%/mo, p={p:.2f})."
    )

    return build_metric(
        name=name, module=MODULE, metric_type="pct", criticality="diagnostic",
        estimator="ols_slope_newey_west_hac", measurement_horizon="since_inception",
        value=monthly, unit=PCT, n_samples=n_eff, n_floor=_ALPHA_TREND_N_FLOOR,
        ci_low=ci_low, ci_high=ci_high, ci_method="newey-west",
        bh_fdr_adjusted_p=p, source_path=src, status=status, reason=reason,
    )


# ---------------------------------------------------------------------------
# regime_weighted_alpha — market-regime decomposition of daily alpha (config#857 C2-fu)
# ---------------------------------------------------------------------------

def _read_regime_for_dates(bucket: str, dates: list[str], s3_client=None) -> dict[str, str]:
    """Join each ``eod_pnl.csv`` date to its ``market_regime`` tag.

    Reads ``s3://{bucket}/signals/{date}/signals.json`` for every date and pulls
    the TOP-LEVEL ``market_regime`` field (``bull``/``neutral``/``caution``/
    ``bear`` — matches ``crucible-research/scripts/backfill_calibrator_v1_context.py``'s
    ``payload.get("market_regime")``). A date with no ``signals.json``
    (``NoSuchKey`` — weekends/holidays already shouldn't reach ``eod_pnl.csv``,
    but be defensive) or no ``market_regime`` key is SKIPPED, never fabricated —
    the caller sees it simply absent from the returned mapping. Any other S3
    error is fail-loud (mirrors ``read_eod_pnl``'s posture: a real S3 problem
    must not be silently read as "this date has no regime").

    One ``get_object`` per date — ``eod_pnl.csv``'s date range is bounded to the
    still-young live book's since-inception history (tens to low hundreds of
    rows today), so N sequential GETs is fine; revisit (parallelize / batch) if
    the live book's history grows materially.
    """
    s3 = s3_client or boto3.client("s3")
    regimes: dict[str, str] = {}
    for d in dates:
        key = SIGNALS_KEY_TEMPLATE.format(date=d)
        try:
            resp = s3.get_object(Bucket=bucket, Key=key)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                continue
            logger.error("S3 read failed for s3://%s/%s: %s", bucket, key, e)
            raise
        try:
            payload = json.loads(resp["Body"].read())
        except (json.JSONDecodeError, ValueError):
            logger.warning("Skipping corrupt signals.json at s3://%s/%s", bucket, key)
            continue
        regime = payload.get("market_regime") if isinstance(payload, dict) else None
        if regime:
            regimes[d] = regime
    return regimes


def _read_selection_pbo(bucket: str, s3_client=None) -> tuple[dict | None, str]:
    """Read the model-zoo rotation leaderboard's ``selection_pbo`` block.

    Returns ``(block_or_None, na_detail)``. ``block`` is ``None`` with a
    specific, operator-readable reason whenever the verdict is not available;
    the reason NAMES the producer and the key, never "insufficient data".

    Fail-loud on a real S3 fault (mirrors ``read_eod_pnl`` / ``_read_regime_for_dates``):
    a permissions or network error must never be read as "the rotation didn't
    compute PBO".
    """
    s3 = s3_client or boto3.client("s3")
    uri = f"s3://{bucket}/{MODEL_ZOO_LEADERBOARD_KEY}"
    try:
        resp = s3.get_object(Bucket=bucket, Key=MODEL_ZOO_LEADERBOARD_KEY)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None, (
                f"pbo: no model-zoo rotation leaderboard at {uri} — "
                f"crucible-predictor's ModelZooSelect stage has not written one. "
                f"PBO requires the rotation's (n_splits x n_specs) per-spec "
                f"performance matrix; the evaluator does not reconstruct it."
            )
        logger.error("S3 read failed for %s: %s", uri, e)
        raise
    try:
        payload = json.loads(resp["Body"].read())
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"model-zoo leaderboard at {uri} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"model-zoo leaderboard at {uri} is not a JSON object.")
    block = payload.get("selection_pbo")
    if not isinstance(block, dict):
        return None, (
            f"pbo: {uri} carries no `selection_pbo` block — the rotation ran but "
            f"crucible-predictor did not emit a CSCV verdict for it "
            f"(training/model_zoo.py::_selection_pbo)."
        )
    block = dict(block)
    block["_leaderboard_date"] = payload.get("date")
    return block, ""


def _build_pbo_component(bucket: str, s3_client=None):
    """Component: CSCV Probability of Backtest Overfitting (alpha-engine-config-I9684).

    The second of the two leak-free terms in the fleet's own written graduation
    bar (``alpha-engine-config/private-docs/SYSTEM_OPTIMIZED.md`` s12: *"leak-free
    validated (DSR significant, **PBO < 0.2**)"*, restated in s2 Tier 1 as
    "PBO ... target < 0.2"). ``dsr`` has been on this tile since config#2454;
    ``pbo`` was computed nowhere the card could see it, so any predicate written
    against s12 was unevaluable.

    The bar is DECLARED, not authored here: 0.2 lives in the threshold registry
    row, sourced from s12. The status is passed explicitly (the ``beta_vs_spy``
    precedent) because the generic deriver orients direction from the
    target/red-line ORDERING, and s12 declares a target with no red line -- with
    ``red_line=None`` the deriver would silently read PBO as higher-is-better and
    grade a healthy 0.09 as WATCH. Nothing goes RED: a red line for PBO is not
    written down anywhere, and this component does not invent one.

    ``n_samples`` is the CSCV split count and ``n_floor`` is the engine's own
    declared ``min_splits``. The TRIAL axis (``n_specs``) carries its own
    power question that no document answers -- it is reported in full in the
    reason (spec count, dropped specs, selection concentration) rather than
    silently folded into a letter. See alpha-engine-config-I9685.
    """
    src = f"s3://{bucket}/{MODEL_ZOO_LEADERBOARD_KEY}"
    band = resolve_band(MODULE, "pbo", DEFAULT_BAND)
    target = band.target

    block, na_detail = _read_selection_pbo(bucket, s3_client=s3_client)
    if block is None:
        return build_metric(
            name="pbo", module=MODULE, metric_type="pct", criticality="supporting",
            n_floor=_PBO_MIN_SPLITS, source_path=src, implemented=False,
            na_detail=na_detail,
        )

    status_raw = block.get("status")
    n_splits = block.get("n_splits")
    n_specs = block.get("n_specs")
    n_splits = int(n_splits) if isinstance(n_splits, (int, float)) else None
    n_specs = int(n_specs) if isinstance(n_specs, (int, float)) else None
    pbo = block.get("pbo")
    pbo = float(pbo) if isinstance(pbo, (int, float)) and math.isfinite(float(pbo)) else None

    if status_raw != "ok" or pbo is None:
        reason = block.get("reason") or f"engine status={status_raw!r}"
        return build_metric(
            name="pbo", module=MODULE, metric_type="pct", criticality="supporting",
            n_samples=n_splits, n_floor=_PBO_MIN_SPLITS, source_path=src,
            status="N/A-LOW-N",
            reason=(
                f"pbo: the rotation's CSCV returned no number ({reason}); "
                f"n_splits={n_splits}, n_specs={n_specs}. The engine's declared "
                f"floor is min_splits={_PBO_MIN_SPLITS} "
                f"(nousergon_lib.quant.stats.pbo.cscv_pbo)."
            ),
        )

    # Everything the reader needs to judge how much this number is worth --
    # the trial axis is where a CSCV estimate degenerates, so it is stated,
    # never inferred away.
    dropped = block.get("dropped_misaligned_specs") or []
    counts = block.get("selected_counts") or {}
    caveats = [f"n_specs={n_specs}"]
    if dropped:
        caveats.append(f"{len(dropped)} spec(s) dropped as misaligned ({', '.join(map(str, dropped))})")
    if isinstance(counts, dict) and len(counts) == 1 and n_splits:
        only = next(iter(counts))
        caveats.append(
            f"selection is DEGENERATE — '{only}' was the in-sample winner in all "
            f"{n_splits} splits, so this PBO measures a strictly dominant "
            f"configuration, not a contested sweep"
        )
    lb_date = block.get("_leaderboard_date")
    if lb_date:
        caveats.append(f"rotation {lb_date}")

    if n_splits is None or n_splits < _PBO_MIN_SPLITS:
        status = "N/A-LOW-N"
        headline = (
            f"pbo = {pbo:.4g} over n_splits={n_splits}, below the engine's "
            f"declared floor of {_PBO_MIN_SPLITS} — UNDERPOWERED"
        )
    elif target is not None and pbo < target:
        status = "GREEN"
        headline = f"pbo = {pbo:.4g} < {target:g} (the declared bar, SYSTEM_OPTIMIZED.md s12), N={n_splits} splits"
    else:
        # WATCH, never RED: s12 declares a target and no red line, and this
        # component does not author one.
        status = "WATCH"
        headline = (
            f"pbo = {pbo:.4g} does NOT clear the declared bar of {target:g} "
            f"(SYSTEM_OPTIMIZED.md s12), N={n_splits} splits"
        )

    return build_metric(
        name="pbo", module=MODULE, metric_type="pct", criticality="supporting",
        value=pbo, unit=PROBABILITY, n_samples=n_splits, n_floor=_PBO_MIN_SPLITS,
        source_path=src, higher_is_better=False, status=status,
        reason=f"{headline}. " + "; ".join(caveats) + ".",
    )


def _regime_na_detail(
    bucket_counts: dict[str, int], qualifying: dict[str, float], total_joined: int,
    *, n_floor: int, min_bucket_n: int, min_buckets: int,
) -> str:
    """Specific N/A reason naming exactly which regimes/samples were available."""
    if not bucket_counts:
        return (
            "regime_weighted_alpha: no eod_pnl.csv date could be joined to a "
            "signals.json market_regime tag (signals.json absent, or missing the "
            "top-level market_regime field, for every date since inception) — "
            "cannot decompose regime-weighted alpha."
        )
    bucket_summary = ", ".join(
        f"{r} (n={c})" for r, c in sorted(bucket_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    if len(qualifying) < min_buckets:
        return (
            f"regime_weighted_alpha: only {len(qualifying)} of {len(bucket_counts)} observed "
            f"regime(s) meet the ≥{min_bucket_n}-sample floor ({bucket_summary}) since "
            f"inception — cannot decompose regime-weighted alpha without ≥{min_buckets} "
            f"qualifying regimes."
        )
    return (
        f"regime_weighted_alpha: only {total_joined} total regime-joined daily samples since "
        f"inception ({bucket_summary}), below the n_floor={n_floor} needed for a confident "
        f"regime-decomposed reading."
    )


def _compute_regime_weighted_alpha(
    dates: list[str], alpha: list[float], regime_by_date: dict[str, str],
    *, n_floor: int = _REGIME_ALPHA_N_FLOOR, min_bucket_n: int = _REGIME_MIN_BUCKET_N,
    min_buckets: int = _REGIME_MIN_BUCKETS,
) -> dict:
    """Regime-decompose daily alpha into the equal-weight mean log-alpha across
    qualifying regime buckets (config#857 C2-fu design spec).

    Each date's daily active return (fraction) is converted to log-alpha
    (``ln(1 + alpha)``) and grouped by its joined regime. A regime bucket
    "qualifies" at ``>= min_bucket_n`` daily samples. ``regime_weighted_alpha``
    is the SIMPLE (equal-weight) mean of the qualifying buckets' mean log-alpha
    — deliberately NOT a time-weighted mean of all daily observations, so a
    strategy that has only traded through one dominant regime can't get a
    headline number that looks validated across regimes it has never seen.

    Returns a dict: ``value`` (None if not decomposable), ``n_samples`` (total
    joined samples across qualifying buckets only), ``na_detail`` (None when
    populated), ``bucket_counts`` (all observed regimes, for diagnostics).
    """
    buckets: dict[str, list[float]] = {}
    for d, a in zip(dates, alpha):
        regime = regime_by_date.get(d)
        if regime is None or (1 + a) <= 0:
            continue
        buckets.setdefault(regime, []).append(math.log1p(a))

    bucket_counts = {r: len(vals) for r, vals in buckets.items()}
    total_joined = sum(bucket_counts.values())
    qualifying_means = {r: sum(vals) / len(vals) for r, vals in buckets.items() if len(vals) >= min_bucket_n}

    if len(qualifying_means) < min_buckets or total_joined < n_floor:
        return {
            "value": None,
            "n_samples": total_joined,
            "bucket_counts": bucket_counts,
            "na_detail": _regime_na_detail(
                bucket_counts, qualifying_means, total_joined,
                n_floor=n_floor, min_bucket_n=min_bucket_n, min_buckets=min_buckets,
            ),
        }

    qualifying_n = sum(bucket_counts[r] for r in qualifying_means)
    value = sum(qualifying_means.values()) / len(qualifying_means)
    return {
        "value": value,
        "n_samples": qualifying_n,
        "bucket_counts": bucket_counts,
        "na_detail": None,
    }


# ---------------------------------------------------------------------------
# tile builder
# ---------------------------------------------------------------------------

_ATTR_NA = (
    "attribution not built this cycle: {reason}. The Brinson-Fachler "
    "decomposition, the SPY carve-out and the beta-adjusted benchmark all read "
    "the same inputs, so all four are absent together."
)


def _build_attribution_components(bucket: str, s3_client=None) -> tuple[list, dict]:
    """Deliverables 6-7 as graded components.

    A failure to BUILD the attribution renders as N/A-MISSING-INPUT with the
    producer's own reason — never as a zero effect, which would read as "no
    allocation bet and no selection skill" on exactly the cycles where nothing
    was measured at all.
    """
    src = f"s3://{bucket}/{ATTRIBUTION_LATEST_KEY}"
    try:
        attr = build_attribution(bucket, s3_client=s3_client)
    except Exception as e:  # noqa: BLE001
        # (a) swallowed: any attribution build error. (b) the tile's other 14
        # components are the primary deliverable and must still render.
        # (c) recording surface: the four N/A components below carry the
        # exception text, and it is logged at ERROR.
        logger.error("attribution build failed: %s", e)
        attr = {"status": "missing_input", "reason": str(e)}

    ok = attr.get("status") == "ok"
    reason = None if ok else _ATTR_NA.format(reason=attr.get("reason", "unknown"))
    n = attr.get("sessions_used") or 0
    bf = attr.get("brinson_fachler") or {}
    act = attr.get("active_return") or {}

    def _m(name, value, metric_type, unit, criticality, estimator):
        return build_metric(
            name=name, module=MODULE, metric_type=metric_type,
            criticality=criticality, estimator=estimator,
            measurement_horizon="since_inception",
            value=value if ok else None, unit=unit,
            n_samples=n if ok else None, n_floor=20, source_path=src,
            input_present=ok, na_detail=reason,
        )

    resid = bf.get("residual_pct_of_active")
    return ([
        # Deliverable 6's own closes-when: the decomposition must reconcile to
        # active return with a residual under 1% of it.
        _m("attribution_residual_pct_of_active", resid, "pct", PCT,
           "diagnostic", "brinson_fachler_carino_linked"),
        _m("bf_selection_effect", bf.get("selection"), "ratio", FRACTION,
           "supporting", "brinson_fachler_carino_linked"),
        # Deliverable 7: the book holds SPY and is benchmarked against SPY, so
        # the headline active return is diluted by a sleeve whose active return
        # is ~0 by construction. This is the active sleeve alone.
        _m("active_return_ex_index_sleeve", act.get("active_return_ex_index_sleeve"),
           "ratio", FRACTION, "supporting", "cumulative_active_return_ex_benchmark_sleeve"),
        _m("beta_adjusted_alpha", act.get("beta_adjusted_alpha"),
           "ratio", FRACTION, "supporting", "ols_beta_adjusted_cumulative_alpha"),
    ], attr)


def build_portfolio_outcome_tile(
    bucket: str, s3_client=None, *, n_trials: int | None = None,
    history: CardHistory | None = None,
) -> dict:
    """Build the Portfolio Outcome tile from eod_pnl.csv.

    ``n_trials`` (DSR selection-bias count, config#2454) is read by the
    caller (``grading/aggregate.py``) from the shared
    ``nousergon_lib.quant.stats.trial_accumulator`` cumulative counter —
    incremented by crucible-backtester's 4 sweep producers on every cycle
    that ran real (non-skipped) trials — and passed in here. DSR emits
    N/A-NOT-IMPL only when the caller passes ``None`` (e.g. the counter
    artifact doesn't exist yet / hasn't been backfilled) or 0. Standalone
    CLI / tests that don't pass ``n_trials`` explicitly get the same
    pre-#2454 N/A behavior.

    ``history`` (config#1836) supplies prior-card values so the critical
    score-vs-return components (sharpe_ratio / alpha_vs_spy / hit_rate_daily)
    carry cross-cycle ``trend_4w``/``trend_13w``; omitted (standalone CLI /
    tests) → trends stay unpopulated, exactly the pre-#1836 behavior.
    """

    def _tr(name: str, value: float | None) -> dict:
        return history.trends_for(MODULE, name, value) if history is not None else {}

    src = f"s3://{bucket}/{EOD_PNL_KEY}"
    series = read_eod_pnl(bucket, s3_client=s3_client)

    if series is None:
        # Whole tile N/A-MISSING-INPUT — one record per component, specific reason.
        def miss(name, mt, crit, floor, tgt, rl, est=None):
            return build_metric(
                name=name, module=MODULE, metric_type=mt, criticality=crit, n_floor=floor,
                source_path=src, input_present=False,
                estimator=est, measurement_horizon="since_inception",
                na_detail=f"{name}: trades/eod_pnl.csv not present this cycle — no EOD reconciliation export to grade.",
                **_tr(name, None),
            )

        components = [
            miss("sharpe_ratio", "sharpe", "critical", 60, 1.0, 0.0, "sharpe_with_bootstrap_ci"),
            miss("information_ratio", "ratio", "critical", 60, 0.5, 0.0, "info_ratio_bootstrap_ci"),
            miss("psr", "pct", "critical", 60, 0.95, 0.50, "probabilistic_sharpe"),
            miss("alpha_vs_spy", "log_return", "critical", 60, 0.0, -0.05, "cumulative_log_alpha"),
            miss("max_drawdown", "ratio", "critical", 2, -0.15, -0.25, "peak_to_trough_nav"),
            miss("sortino_ratio", "sharpe", "supporting", 60, 1.5, 0.5),
            miss("calmar_ratio", "ratio", "supporting", 90, 1.0, 0.0),
            miss("cvar_95_daily", "ratio", "supporting", 60, -0.01, -0.04),
            miss("alpha_trend", "pct", "diagnostic", _ALPHA_TREND_N_FLOOR, 0.0, None, "ols_slope_newey_west_hac"),
        ]
        # pbo reads the model-zoo leaderboard, NOT eod_pnl.csv — a missing EOD
        # export says nothing about whether the rotation computed a CSCV verdict,
        # so it is built here too rather than blanket-N/A'd (I9622).
        components.append(_build_pbo_component(bucket, s3_client=s3_client))
        annotate_power_all(components)
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
        estimator="sharpe_with_bootstrap_ci", measurement_horizon="since_inception",
        value=sharpe, unit=ANNUALIZED_RATIO, n_samples=n, n_floor=60,
        ci_low=s_lo, ci_high=s_hi, ci_method=s_m, source_path=src,
        **_tr("sharpe_ratio", sharpe),
    ))

    # 2. Information ratio (critical)
    ir = _ann_ir(np.asarray(active))
    ir = None if (ir is None or math.isnan(ir)) else ir
    i_lo, i_hi, i_m = _ci(_ann_ir, active) if ir is not None else (None, None, None)
    components.append(build_metric(
        name="information_ratio", module=MODULE, metric_type="ratio", criticality="critical",
        estimator="info_ratio_bootstrap_ci", measurement_horizon="since_inception",
        value=ir, unit=ANNUALIZED_RATIO, n_samples=n, n_floor=60,
        ci_low=i_lo, ci_high=i_hi, ci_method=i_m, source_path=src,
    ))

    # 3. PSR (critical) — P(true Sharpe > 0). Probability, no CI.
    psr_res = compute_psr(np.asarray(port), sharpe_benchmark=0.0)
    psr_val = psr_res.get("psr") if psr_res.get("status") == "ok" else None
    components.append(build_metric(
        name="psr", module=MODULE, metric_type="pct", criticality="critical",
        estimator="probabilistic_sharpe", measurement_horizon="since_inception",
        value=psr_val, unit=PROBABILITY, n_samples=psr_res.get("n", n), n_floor=60,
        source_path=src,
    ))

    # 4. Alpha vs SPY (critical) — cumulative log-alpha since inception.
    log_alpha = cumulative_log_alpha(port, series.spy)
    components.append(build_metric(
        name="alpha_vs_spy", module=MODULE, metric_type="log_return", criticality="critical",
        estimator="cumulative_log_alpha", measurement_horizon="since_inception",
        value=log_alpha, unit=LOG_RETURN, n_samples=n, n_floor=60, source_path=src,
        **_tr("alpha_vs_spy", log_alpha),
    ))

    # 5. Max drawdown (critical) — point obs from the NAV series.
    mdd = max_drawdown(nav)
    components.append(build_metric(
        name="max_drawdown", module=MODULE, metric_type="ratio", criticality="critical",
        estimator="peak_to_trough_nav", measurement_horizon="since_inception",
        value=mdd, unit=FRACTION, n_samples=len(nav), n_floor=2, source_path=src,
    ))

    # 6. Sortino (supporting)
    sortino = sortino_ratio(port)
    so_lo, so_hi, so_m = _ci(_ann_sortino, port) if sortino is not None else (None, None, None)
    components.append(build_metric(
        name="sortino_ratio", module=MODULE, metric_type="sharpe", criticality="supporting",
        value=sortino, unit=ANNUALIZED_RATIO, n_samples=n, n_floor=60,
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
        value=calmar, unit=RATIO, n_samples=n, n_floor=90, source_path=src,
    ))

    # 8. CVaR(95) daily (supporting) — mean worst-5% daily return.
    cvar = _neg_cvar(np.asarray(port))
    cvar = None if math.isnan(cvar) else cvar
    cv_lo, cv_hi, cv_m = _ci(_neg_cvar, port) if cvar is not None else (None, None, None)
    components.append(build_metric(
        name="cvar_95_daily", module=MODULE, metric_type="ratio", criticality="supporting",
        value=cvar, unit=FRACTION, n_samples=n, n_floor=60,
        ci_low=cv_lo, ci_high=cv_hi, ci_method=cv_m, source_path=src,
    ))

    # 9. Hit rate daily (diagnostic) — % days portfolio beats SPY. Wilson CI.
    wins = sum(1 for a in active if a > 0)
    w = wilson_score_interval(wins, n) if n > 0 else {"status": "insufficient_data"}
    hit = w["rate"] if w.get("status") == "ok" else None
    components.append(build_metric(
        name="hit_rate_daily", module=MODULE, metric_type="pct", criticality="diagnostic",
        value=hit, unit=FRACTION, n_samples=n, n_floor=60,
        ci_low=w.get("ci_low"), ci_high=w.get("ci_high"),
        ci_method="wilson" if w.get("status") == "ok" else None, source_path=src,
        **_tr("hit_rate_daily", hit),
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
        value=beta, unit=RATIO, n_samples=n, n_floor=60, source_path=src, status=beta_status,
        reason=(f"beta_vs_spy = {beta:.3g} (target band 0.7–1.1)." if beta is not None
                else "beta_vs_spy: too few observations to estimate."),
    ))

    # 11. Max-DD duration (diagnostic) — days underwater since prior peak.
    components.append(build_metric(
        name="max_dd_duration_days", module=MODULE, metric_type="duration", criticality="diagnostic",
        value=float(_max_dd_duration_days(nav)), unit=DAYS, n_samples=len(nav), n_floor=2,
        higher_is_better=False, source_path=src,
    ))

    # 12. DSR (supporting) — deflated Sharpe, corrected for the
    #     multiple-testing selection bias of picking the deployed config out
    #     of n_trials candidates (config#2454). n_trials is the cumulative
    #     trial count the caller reads from the shared
    #     nousergon_lib.quant.stats.trial_accumulator counter.
    if n_trials is not None and n_trials >= 1:
        from nousergon_lib.quant.stats.dsr import compute_dsr
        dsr_res = compute_dsr(np.asarray(port), n_trials=n_trials)
        dsr_val = dsr_res.get("dsr") if dsr_res.get("status") == "ok" else None
        components.append(build_metric(
            name="dsr", module=MODULE, metric_type="pct", criticality="supporting",
            value=dsr_val, unit=PROBABILITY, n_samples=dsr_res.get("n", n), n_floor=60,
            source_path=src,
        ))
    else:
        components.append(build_metric(
            name="dsr", module=MODULE, metric_type="pct", criticality="supporting",
            n_floor=60, source_path=src, implemented=False,
            na_detail=(
                "dsr: cumulative trial count unavailable this cycle "
                "(s3://{bucket}/backtest/cumulative_trial_count.json missing, "
                "empty, or not yet backfilled — see config#2454)."
            ).format(bucket=bucket),
        ))

    # 12b. PBO (supporting) — CSCV Probability of Backtest Overfitting, the
    #      selection-overfitting complement to dsr and the second of the two
    #      leak-free terms in SYSTEM_OPTIMIZED.md §12's graduation bar
    #      (alpha-engine-config-I9684).
    components.append(_build_pbo_component(bucket, s3_client=s3_client))

    # 13. Regime-weighted alpha (critical) — market-regime decomposition of daily
    #     alpha via a join against signals/{date}/signals.json's market_regime
    #     tag (config#857 C2-fu). Equal-weight mean log-alpha across regime
    #     buckets meeting the sample floor; N/A (not a guess) if the book hasn't
    #     yet traded through >=2 regimes with enough days in each.
    regime_by_date = _read_regime_for_dates(bucket, series.dates, s3_client=s3_client)
    rwa = _compute_regime_weighted_alpha(series.dates, active, regime_by_date)
    components.append(build_metric(
        name="regime_weighted_alpha", module=MODULE, metric_type="log_return", criticality="critical",
        estimator="regime_weighted_log_alpha", measurement_horizon="since_inception",
        value=rwa["value"], unit=LOG_RETURN, n_samples=rwa["n_samples"], n_floor=_REGIME_ALPHA_N_FLOOR,
        source_path=src,
        input_present=rwa["na_detail"] is None,
        na_detail=rwa["na_detail"],
    ))

    # 14. Alpha trend (diagnostic) — is daily alpha statistically improving?
    components.append(_build_alpha_trend([a * 100.0 for a in active], src))

    # 15-18. Benchmark-relative attribution (I8188 deliverables 6-7).
    #        Built from the same eod_pnl.csv plus SPY's published sector weights
    #        and the eleven SPDR sector ETFs' dividend-adjusted closes — no new
    #        producer. `grading/attribution.py` carries the math, the exact
    #        Cariño-linked reconciliation, and the stated approximations.
    _attr_components, _attr_payload = _build_attribution_components(
        bucket, s3_client=s3_client,
    )
    components.extend(_attr_components)

    # ── Statistical power (alpha-engine-config-I8188 deliverable 8) ─────────
    # Publish n_required + target_inside_ci beside EVERY component, and
    # downgrade a RED that the CI's width produced rather than the evidence.
    # Applied here, once, after every component is built and BEFORE the tile
    # rolls them up — module_agg's critical-gate + BH-FDR pass must see the
    # corrected statuses, or a power-limited RED still cascades to the card.
    annotate_power_all(components)

    tile = build_tile(MODULE, components)
    # The full decomposition rides on the tile so the report-card writer can
    # persist it as its own artifact (evaluator/{run_date}/attribution.json)
    # without recomputing it — one build per cycle, one number everywhere.
    tile["attribution"] = _attr_payload
    # config#2885: surface silently-dropped CSV rows so the report card
    # shows data-integrity issues (unparseable portfolio_nav/daily_return_pct).
    if series is not None:
        tile["rows_dropped"] = series.rows_dropped
    return tile


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
