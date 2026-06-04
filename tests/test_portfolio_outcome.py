"""Tests for grading/tiles/portfolio_outcome.py — Tile 0 from eod_pnl.csv."""

import boto3
import pytest
from moto import mock_aws

from grading.tiles.portfolio_outcome import (
    EOD_PNL_KEY,
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
