"""Tests for grading/tiles/executor.py — Tile 3."""

import json

import boto3
import pytest
from moto import mock_aws

from grading.tiles.executor import build_executor_tile

BUCKET = "alpha-engine-research"
RUN_DATE = "2026-06-07"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _put(s3, name, data):
    s3.put_object(Bucket=BUCKET, Key=f"backtest/{RUN_DATE}/{name}", Body=json.dumps(data).encode())


def _comp(tile, name):
    return next(c for c in tile["components"] if c["name"] == name)


class TestMissing:
    def test_all_absent_watch(self, s3):
        tile = build_executor_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert tile["module"] == "executor"
        assert _comp(tile, "entry_triggers")["status"] == "N/A-MISSING-INPUT"
        # position_sizing is genuinely unwired → NOT-IMPL, not MISSING-INPUT.
        assert _comp(tile, "position_sizing")["status"] == "N/A-NOT-IMPL"
        assert _comp(tile, "reconciliation_integrity")["status"] == "N/A-NOT-IMPL"
        # Critical components N/A → WATCH (never false GREEN).
        assert tile["status"] == "WATCH"


class TestPopulated:
    def test_entry_triggers_winrate_wilson(self, s3):
        _put(s3, "trigger_scorecard.json", {
            "status": "ok",
            "summary": {"win_rate_vs_spy": 0.60, "total_entries": 50, "avg_slippage_vs_signal": -0.003},
            "triggers": [],
        })
        tile = build_executor_tile(BUCKET, RUN_DATE, s3_client=s3)
        et = _comp(tile, "entry_triggers")
        assert et["value"] == pytest.approx(0.60)
        assert et["n_samples"] == 50
        assert et["ci_method"] == "wilson"
        assert et["status"] == "GREEN"  # 0.60 > target 0.55, N>floor

    def test_risk_guard_precision(self, s3):
        _put(s3, "shadow_book.json", {
            "status": "ok", "assessment": "appropriate", "guard_lift": 1.2,
            "classification": {"precision": 0.62, "tp": 31, "fp": 19},
        })
        tile = build_executor_tile(BUCKET, RUN_DATE, s3_client=s3)
        rg = _comp(tile, "risk_guard")
        assert rg["value"] == pytest.approx(0.62)
        assert rg["n_samples"] == 50
        assert rg["ci_method"] == "wilson"

    def test_exit_rules_capture_ratio(self, s3):
        _put(s3, "exit_timing.json", {
            "status": "ok", "n_roundtrips": 40, "diagnosis": "exits_well_timed",
            "summary": {"avg_capture_ratio": 0.75, "avg_realized_return": 0.02},
        })
        tile = build_executor_tile(BUCKET, RUN_DATE, s3_client=s3)
        ex = _comp(tile, "exit_rules")
        assert ex["value"] == pytest.approx(0.75)
        assert ex["status"] == "GREEN"  # 0.75 > target 0.70

    def test_excursion_mfe_mae(self, s3):
        _put(s3, "portfolio_excursion.json", {
            "status": "ok", "mean_mfe_mae_ratio": 1.6, "pct_high_quality": 0.5, "n": 40,
        })
        tile = build_executor_tile(BUCKET, RUN_DATE, s3_client=s3)
        exc = _comp(tile, "excursion")
        assert exc["value"] == pytest.approx(1.6)
        assert exc["status"] == "GREEN"  # 1.6 > target 1.5

    def test_full_executor_green_when_all_strong(self, s3):
        _put(s3, "trigger_scorecard.json", {"status": "ok", "summary": {"win_rate_vs_spy": 0.62, "total_entries": 60}, "triggers": []})
        _put(s3, "shadow_book.json", {"status": "ok", "classification": {"precision": 0.65, "tp": 33, "fp": 17}})
        _put(s3, "exit_timing.json", {"status": "ok", "n_roundtrips": 40, "summary": {"avg_capture_ratio": 0.78}})
        _put(s3, "portfolio_excursion.json", {"status": "ok", "mean_mfe_mae_ratio": 1.7, "pct_high_quality": 0.55, "n": 40})
        tile = build_executor_tile(BUCKET, RUN_DATE, s3_client=s3)
        # 3 critical GREEN (triggers/guard/exits); reconciliation is critical N/A-NOT-IMPL → WATCH.
        assert _comp(tile, "entry_triggers")["status"] == "GREEN"
        assert _comp(tile, "risk_guard")["status"] == "GREEN"
        assert _comp(tile, "exit_rules")["status"] == "GREEN"
        # reconciliation_integrity (critical, N/A-NOT-IMPL) keeps the tile at WATCH.
        assert tile["status"] == "WATCH"
