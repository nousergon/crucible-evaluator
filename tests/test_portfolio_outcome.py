"""Tests for grading/tiles/portfolio_outcome.py — Tile 0 from eod_pnl.csv."""

import json
import math

import boto3
import pytest
from moto import mock_aws

from grading.tiles.portfolio_outcome import (
    EOD_PNL_KEY,
    SIGNALS_KEY_TEMPLATE,
    build_portfolio_outcome_tile,
    read_eod_pnl,
)

BUCKET = "alpha-engine-research"

# A strong-positive synthetic book: ~80 trading days, portfolio beats SPY most
# days. daily_return_pct / spy_return_pct are in PERCENT (matching live CSV).
_HEADER = "date,portfolio_nav,daily_return_pct,spy_return_pct,daily_alpha_pct,positions_snapshot,created_at"


def _synth_csv(n: int = 80) -> str:
    rows = [_HEADER]
    nav = 1_000_000.0
    for i in range(n):
        port_pct = 0.12 if i % 5 else -0.05   # mostly up, occasional down day
        spy_pct = 0.04
        alpha_pct = port_pct - spy_pct
        nav *= 1 + port_pct / 100.0
        d = f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}"
        rows.append(f"{d},{nav:.2f},{port_pct},{spy_pct},{alpha_pct},{{}},2026-01-01T00:00:00+00:00")
    return "\n".join(rows) + "\n"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _put_eod(s3, body):
    s3.put_object(Bucket=BUCKET, Key=EOD_PNL_KEY, Body=body.encode("utf-8"))


def _put_regime(s3, date, regime=None, extra=None):
    """Write a synthetic signals.json for ``date``. ``regime=None`` writes a
    payload with no top-level market_regime key (malformed-but-present case).
    """
    payload = dict(extra or {})
    if regime is not None:
        payload["market_regime"] = regime
    key = SIGNALS_KEY_TEMPLATE.format(date=date)
    s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(payload).encode("utf-8"))


def _regime_csv(n, alpha_by_index):
    """Build an eod_pnl.csv where ``alpha_by_index(i)`` (percent) drives both
    daily_return_pct and daily_alpha_pct (spy pinned at 0%, so port == alpha).
    Returns (csv_body, dates)."""
    rows = [_HEADER]
    nav = 1_000_000.0
    dates = []
    for i in range(n):
        alpha_pct = alpha_by_index(i)
        nav *= 1 + alpha_pct / 100.0
        d = f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}"
        dates.append(d)
        rows.append(f"{d},{nav:.2f},{alpha_pct},0.0,{alpha_pct},{{}},2026-01-01T00:00:00+00:00")
    return "\n".join(rows) + "\n", dates


class TestReadEodPnl:
    def test_absent_returns_none(self, s3):
        assert read_eod_pnl(BUCKET, s3_client=s3) is None

    def test_percent_to_fraction(self, s3):
        _put_eod(s3, _HEADER + "\n2026-03-10,1000260.0,1.0,0.5,0.5,{},2026-03-10T00:00:00+00:00\n")
        series = read_eod_pnl(BUCKET, s3_client=s3)
        assert series.n == 1
        # 1.0 percent → 0.01 fraction.
        assert series.port[0] == pytest.approx(0.01)
        assert series.spy[0] == pytest.approx(0.005)
        assert series.alpha[0] == pytest.approx(0.005)

    def test_rows_sorted_by_date(self, s3):
        body = (_HEADER + "\n"
                "2026-03-11,1001.0,0.2,0.1,0.1,{},x\n"
                "2026-03-10,1000.0,0.1,0.1,0.0,{},x\n")
        _put_eod(s3, body)
        series = read_eod_pnl(BUCKET, s3_client=s3)
        assert series.dates == ["2026-03-10", "2026-03-11"]


class TestBuildTileMissingInput:
    def test_all_components_na_missing_input(self, s3):
        tile = build_portfolio_outcome_tile(BUCKET, s3_client=s3)
        assert tile["module"] == "portfolio_outcome"
        # No eod_pnl → every component N/A-MISSING-INPUT.
        statuses = {c["status"] for c in tile["components"]}
        assert statuses == {"N/A-MISSING-INPUT"}
        # Critical N/A → tile WATCH (transparency), never a false GREEN.
        assert tile["status"] == "WATCH"
        assert tile["numeric_grade"] is None


class TestBuildTileFull:
    def test_components_present_and_typed(self, s3):
        _put_eod(s3, _synth_csv(80))
        tile = build_portfolio_outcome_tile(BUCKET, s3_client=s3)
        names = {c["name"] for c in tile["components"]}
        expected = {
            "sharpe_ratio", "information_ratio", "psr", "alpha_vs_spy", "max_drawdown",
            "sortino_ratio", "calmar_ratio", "cvar_95_daily", "hit_rate_daily",
            "beta_vs_spy", "max_dd_duration_days", "dsr", "regime_weighted_alpha",
            "alpha_trend",
        }
        assert expected <= names

    def test_strong_book_sharpe_is_real_and_positive(self, s3):
        _put_eod(s3, _synth_csv(80))
        tile = build_portfolio_outcome_tile(BUCKET, s3_client=s3)
        sharpe = next(c for c in tile["components"] if c["name"] == "sharpe_ratio")
        assert sharpe["value"] is not None
        assert sharpe["value"] > 0
        assert sharpe["n_samples"] == 80
        # CI populated via bootstrap.
        assert sharpe["ci_method"] == "bootstrap"
        assert sharpe["ci_low"] is not None and sharpe["ci_high"] is not None

    def test_alpha_vs_spy_positive_for_outperforming_book(self, s3):
        _put_eod(s3, _synth_csv(80))
        tile = build_portfolio_outcome_tile(BUCKET, s3_client=s3)
        alpha = next(c for c in tile["components"] if c["name"] == "alpha_vs_spy")
        assert alpha["value"] > 0  # portfolio beats SPY cumulatively

    def test_dsr_not_impl_without_trial_count(self, s3):
        _put_eod(s3, _synth_csv(80))
        tile = build_portfolio_outcome_tile(BUCKET, s3_client=s3)
        dsr = next(c for c in tile["components"] if c["name"] == "dsr")
        assert dsr["status"] == "N/A-NOT-IMPL"
        assert "trial count" in dsr["status_reason"].lower()

    def test_dsr_computed_with_trial_count(self, s3):
        _put_eod(s3, _synth_csv(80))
        tile = build_portfolio_outcome_tile(BUCKET, s3_client=s3, n_trials=20)
        dsr = next(c for c in tile["components"] if c["name"] == "dsr")
        assert dsr["status"] != "N/A-NOT-IMPL"
        assert dsr["value"] is not None

    def test_regime_alpha_missing_input(self, s3):
        _put_eod(s3, _synth_csv(80))
        tile = build_portfolio_outcome_tile(BUCKET, s3_client=s3)
        rwa = next(c for c in tile["components"] if c["name"] == "regime_weighted_alpha")
        assert rwa["status"] == "N/A-MISSING-INPUT"
        assert "regime" in rwa["status_reason"].lower()

    def test_psr_is_probability_in_unit_interval(self, s3):
        _put_eod(s3, _synth_csv(80))
        tile = build_portfolio_outcome_tile(BUCKET, s3_client=s3)
        psr = next(c for c in tile["components"] if c["name"] == "psr")
        assert psr["value"] is not None
        assert 0.0 <= psr["value"] <= 1.0


class TestAlphaTrend:
    """config#1962 — statistical answer to 'is alpha improving?'."""

    def test_missing_input_is_na(self, s3):
        tile = build_portfolio_outcome_tile(BUCKET, s3_client=s3)
        trend = next(c for c in tile["components"] if c["name"] == "alpha_trend")
        assert trend["status"] == "N/A-MISSING-INPUT"
        assert trend["criticality"] == "diagnostic"

    def test_short_series_is_low_n(self, s3):
        _put_eod(s3, _synth_csv(10))
        tile = build_portfolio_outcome_tile(BUCKET, s3_client=s3)
        trend = next(c for c in tile["components"] if c["name"] == "alpha_trend")
        assert trend["status"] == "N/A-LOW-N"
        assert trend["value"] is None

    def test_flat_oscillating_alpha_has_ci_and_estimator(self, s3):
        _put_eod(s3, _synth_csv(80))
        tile = build_portfolio_outcome_tile(BUCKET, s3_client=s3)
        trend = next(c for c in tile["components"] if c["name"] == "alpha_trend")
        assert trend["status"] in {"WATCH", "GREEN", "RED"}
        assert trend["ci_low"] is not None and trend["ci_high"] is not None
        assert trend["ci_method"] == "newey-west"
        assert trend.get("estimator") == "ols_slope_newey_west_hac"
        assert trend["bh_fdr_adjusted_p"] is not None

    def test_clear_upward_drift_is_green(self, s3):
        rows = [_HEADER]
        nav = 1_000_000.0
        for i in range(120):
            alpha_pct = 0.001 * i  # steadily widening daily alpha
            port_pct = 0.05 + alpha_pct
            spy_pct = 0.05
            nav *= 1 + port_pct / 100.0
            d = f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}"
            rows.append(f"{d},{nav:.2f},{port_pct},{spy_pct},{alpha_pct},{{}},2026-01-01T00:00:00+00:00")
        _put_eod(s3, "\n".join(rows) + "\n")
        tile = build_portfolio_outcome_tile(BUCKET, s3_client=s3)
        trend = next(c for c in tile["components"] if c["name"] == "alpha_trend")
        assert trend["status"] == "GREEN"
        assert trend["value"] > 0

    def test_clear_downward_drift_is_red(self, s3):
        rows = [_HEADER]
        nav = 1_000_000.0
        for i in range(120):
            alpha_pct = -0.001 * i  # steadily worsening daily alpha
            port_pct = 0.05 + alpha_pct
            spy_pct = 0.05
            nav *= 1 + port_pct / 100.0
            d = f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}"
            rows.append(f"{d},{nav:.2f},{port_pct},{spy_pct},{alpha_pct},{{}},2026-01-01T00:00:00+00:00")
        _put_eod(s3, "\n".join(rows) + "\n")
        tile = build_portfolio_outcome_tile(BUCKET, s3_client=s3)
        trend = next(c for c in tile["components"] if c["name"] == "alpha_trend")
        assert trend["status"] == "RED"
        assert trend["value"] < 0


class TestRegimeWeightedAlpha:
    """config#857 C2-fu — regime join against signals/{date}/signals.json's
    top-level market_regime field."""

    def _rwa(self, tile):
        return next(c for c in tile["components"] if c["name"] == "regime_weighted_alpha")

    def test_populated_with_two_qualifying_regimes(self, s3):
        # 40 days tagged "bull" at +1.0%/day, 40 days tagged "bear" at +2.0%/day
        # — two regimes, each well above the 5-sample bucket floor and 30 total.
        body, dates = _regime_csv(80, lambda i: 1.0 if i < 40 else 2.0)
        _put_eod(s3, body)
        for d in dates[:40]:
            _put_regime(s3, d, "bull")
        for d in dates[40:]:
            _put_regime(s3, d, "bear")

        tile = build_portfolio_outcome_tile(BUCKET, s3_client=s3)
        rwa = self._rwa(tile)

        expected = (math.log1p(0.01) + math.log1p(0.02)) / 2.0
        assert rwa["value"] == pytest.approx(expected, rel=1e-9)
        assert rwa["n_samples"] == 80
        assert rwa["criticality"] == "critical"
        assert rwa["estimator"] == "regime_weighted_log_alpha"
        assert rwa["status"] in {"GREEN", "WATCH", "RED"}
        assert not rwa["status"].startswith("N/A")

    def test_na_with_fewer_than_two_qualifying_regimes(self, s3):
        # All 80 days tagged the single regime "bull" — decomposition needs >=2.
        body, dates = _regime_csv(80, lambda i: 1.0)
        _put_eod(s3, body)
        for d in dates:
            _put_regime(s3, d, "bull")

        tile = build_portfolio_outcome_tile(BUCKET, s3_client=s3)
        rwa = self._rwa(tile)

        assert rwa["status"] == "N/A-MISSING-INPUT"
        assert rwa["value"] is None
        reason = rwa["status_reason"].lower()
        assert "bull" in reason
        assert "80" in rwa["status_reason"]
        assert "≥2" in rwa["status_reason"] or ">=2" in rwa["status_reason"] or "2 qualifying" in reason

    def test_na_with_insufficient_total_samples(self, s3):
        # Two qualifying regimes (5 samples each, right at the per-bucket floor)
        # but only 10 total joined samples — well below the n_floor=30 gate.
        body, dates = _regime_csv(80, lambda i: 1.0 if i % 2 == 0 else 2.0)
        _put_eod(s3, body)
        for d in dates[:5]:
            _put_regime(s3, d, "bull")
        for d in dates[5:10]:
            _put_regime(s3, d, "bear")
        # Remaining 70 dates deliberately left with no signals.json at all.

        tile = build_portfolio_outcome_tile(BUCKET, s3_client=s3)
        rwa = self._rwa(tile)

        assert rwa["status"] == "N/A-MISSING-INPUT"
        assert rwa["value"] is None
        assert rwa["n_samples"] == 10
        reason = rwa["status_reason"]
        assert "bull" in reason and "bear" in reason
        assert "30" in reason

    def test_missing_or_malformed_signals_json_skipped_without_error(self, s3):
        # Only every-other date has a signals.json; one present-but-malformed
        # (no market_regime key) date is thrown in too. Nothing should raise,
        # and the join should simply exclude the untagged dates.
        body, dates = _regime_csv(80, lambda i: 1.0 if i % 2 == 0 else 2.0)
        _put_eod(s3, body)
        tagged_dates = dates[::2]  # 40 dates get a signals.json
        for i, d in enumerate(tagged_dates):
            if i == 0:
                _put_regime(s3, d, regime=None)  # present but no market_regime key
            elif i < 20:
                _put_regime(s3, d, "bull")
            else:
                _put_regime(s3, d, "bear")
        # The other 40 dates get no signals.json object at all (NoSuchKey).

        tile = build_portfolio_outcome_tile(BUCKET, s3_client=s3)
        rwa = self._rwa(tile)

        # 19 bull + 20 bear = 39 joined samples, two qualifying regimes, >=30 total.
        assert rwa["value"] is not None
        assert rwa["n_samples"] == 39
        assert not rwa["status"].startswith("N/A")
