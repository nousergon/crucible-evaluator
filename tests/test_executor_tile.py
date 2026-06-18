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


def _put_trades(s3, name, data):
    s3.put_object(Bucket=BUCKET, Key=f"trades/{RUN_DATE}/{name}", Body=json.dumps(data).encode())


def _comp(tile, name):
    return next(c for c in tile["components"] if c["name"] == name)


class TestMissing:
    def test_all_absent_watch(self, s3):
        tile = build_executor_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert tile["module"] == "executor"
        assert _comp(tile, "entry_triggers")["status"] == "N/A-MISSING-INPUT"
        # position_sizing is genuinely unwired → NOT-IMPL, not MISSING-INPUT.
        assert _comp(tile, "position_sizing")["status"] == "N/A-NOT-IMPL"
        # reconciliation_integrity is now IMPLEMENTED (config#859) — absent
        # artifact → MISSING-INPUT, not NOT-IMPL.
        assert _comp(tile, "reconciliation_integrity")["status"] == "N/A-MISSING-INPUT"
        # Critical components N/A → WATCH (never false GREEN).
        assert tile["status"] == "WATCH"


class TestReconciliationIntegrity:
    def test_perfect_parity_green(self, s3):
        _put_trades(s3, "reconciliation_audit.json", {
            "reconciliation_match_rate": 1.0, "status": "OK",
            "n_positions": 12, "n_mismatched": 0,
            "daily_delta": {"computed": True, "match_rate": 1.0},
        })
        c = _comp(build_executor_tile(BUCKET, RUN_DATE, s3_client=s3), "reconciliation_integrity")
        assert c["value"] == 1.0
        assert c["status"] == "GREEN"
        assert "12/12 positions match" in c["status_reason"]

    def test_drift_below_red_line_is_red(self, s3):
        _put_trades(s3, "reconciliation_audit.json", {
            "reconciliation_match_rate": 0.80, "status": "DRIFT",
            "n_positions": 10, "n_mismatched": 2,
            "daily_delta": {"computed": True, "match_rate": 0.9},
        })
        c = _comp(build_executor_tile(BUCKET, RUN_DATE, s3_client=s3), "reconciliation_integrity")
        assert c["value"] == 0.80
        assert c["status"] == "RED"

    def test_minor_drift_is_watch(self, s3):
        _put_trades(s3, "reconciliation_audit.json", {
            "reconciliation_match_rate": 0.95, "status": "DRIFT",
            "n_positions": 20, "n_mismatched": 1, "daily_delta": {"computed": False},
        })
        c = _comp(build_executor_tile(BUCKET, RUN_DATE, s3_client=s3), "reconciliation_integrity")
        assert c["status"] == "WATCH"


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

    def test_exit_rules_grades_winner_capture(self, s3):
        # Robust winner-capture median is graded; legacy mean is ignored even
        # when it is negative (the L4554 artifact).
        _put(s3, "exit_timing.json", {
            "status": "ok", "n_roundtrips": 40, "diagnosis": "exits_well_timed",
            "summary": {"avg_capture_ratio": -0.27, "capture_winners_median": 0.75,
                        "n_winners": 30, "win_rate": 0.75, "stop_efficiency_median": 0.6,
                        "avg_realized_return": 0.02},
        })
        ex = _comp(build_executor_tile(BUCKET, RUN_DATE, s3_client=s3), "exit_rules")
        assert ex["value"] == pytest.approx(0.75)  # the robust winner-capture, NOT -0.27
        assert ex["status"] == "GREEN"  # 0.75 > target 0.70
        assert ex["n_samples"] == 30

    def test_exit_rules_legacy_fallback(self, s3):
        # Pre-2026-06-07 artifact without winner-capture → legacy mean graded.
        _put(s3, "exit_timing.json", {
            "status": "ok", "n_roundtrips": 40, "diagnosis": "exits_well_timed",
            "summary": {"avg_capture_ratio": 0.75, "avg_realized_return": 0.02},
        })
        ex = _comp(build_executor_tile(BUCKET, RUN_DATE, s3_client=s3), "exit_rules")
        assert ex["value"] == pytest.approx(0.75)
        assert ex["status"] == "GREEN"

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
        _put(s3, "exit_timing.json", {"status": "ok", "n_roundtrips": 40, "summary": {"capture_winners_median": 0.78, "n_winners": 30}})
        _put(s3, "portfolio_excursion.json", {"status": "ok", "mean_mfe_mae_ratio": 1.7, "pct_high_quality": 0.55, "n": 40})
        _put_trades(s3, "reconciliation_audit.json", {
            "reconciliation_match_rate": 1.0, "status": "OK", "n_positions": 8, "n_mismatched": 0,
            "daily_delta": {"computed": True, "match_rate": 1.0},
        })
        tile = build_executor_tile(BUCKET, RUN_DATE, s3_client=s3)
        # 4 critical GREEN (triggers/guard/exits/reconciliation) → tile GREEN.
        assert _comp(tile, "entry_triggers")["status"] == "GREEN"
        assert _comp(tile, "risk_guard")["status"] == "GREEN"
        assert _comp(tile, "exit_rules")["status"] == "GREEN"
        assert _comp(tile, "reconciliation_integrity")["status"] == "GREEN"
        assert tile["status"] == "GREEN"
