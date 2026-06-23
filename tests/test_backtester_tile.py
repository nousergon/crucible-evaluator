"""Tests for grading/tiles/backtester.py — Tile 4 (self-grade)."""

import json
from datetime import UTC, datetime, timedelta

import boto3
import pytest
from moto import mock_aws

from grading.tiles.backtester import _coverage, build_backtester_tile

BUCKET = "alpha-engine-research"
RUN_DATE = "2026-06-07"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _put(s3, key, data, raw=False):
    body = data.encode() if raw else json.dumps(data).encode()
    s3.put_object(Bucket=BUCKET, Key=key, Body=body)


def _comp(tile, name):
    return next(c for c in tile["components"] if c["name"] == name)


# A grading.json mirroring the live v1 shape: lots of N/A in executor/predictor.
_GRADING = {
    "status": "partial",
    "research": {"components": {
        "scanner": {"letter": "D"},
        "sector_teams": [{"letter": "C"}, {"letter": "N/A"}, {"letter": "B"}],
        "sector_teams_avg": {"letter": "F"},  # rollup — excluded
        "macro_agent": {"letter": "C-"},
    }},
    "predictor": {"components": {"meta_model": {"letter": "C"}, "veto_gate": {"letter": "N/A"}}},
    "executor": {"components": {
        "entry_triggers": {"letter": "N/A"}, "risk_guard": {"letter": "N/A"},
        "exit_rules": {"letter": "N/A"}, "portfolio": {"letter": "B+"},
    }},
}


class TestCoverageHelper:
    def test_counts_leaves_excludes_avg(self):
        cov, graded, total = _coverage(_GRADING)
        # leaves: scanner, 3 sector_teams items, macro_agent (5 research, avg excluded)
        #         + meta_model, veto_gate (2) + entry/risk/exit/portfolio (4) = 11 total
        # graded (non-N/A): scanner, 2 teams, macro_agent (4) + meta_model (1)
        #         + portfolio (1) = 6
        assert total == 11
        assert graded == 6
        assert cov == pytest.approx(6 / 11)


class TestEvaluatorCoverage:
    def test_low_coverage_red(self, s3):
        _put(s3, f"backtest/{RUN_DATE}/grading.json", _GRADING)
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        cov = _comp(tile, "evaluator_coverage")
        assert cov["value"] == pytest.approx(6 / 11)
        assert cov["status"] == "RED"  # 0.55 < red-line 0.80

    def test_absent_grading_missing_input(self, s3):
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert _comp(tile, "evaluator_coverage")["status"] == "N/A-MISSING-INPUT"


class TestFreshness:
    def test_fresh_grading_green(self, s3):
        _put(s3, f"backtest/{RUN_DATE}/grading.json", _GRADING)
        # moto stamps LastModified ≈ now → age ~0h → GREEN.
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3, as_of=datetime.now(UTC))
        fr = _comp(tile, "grading_freshness")
        assert fr["status"] == "GREEN"

    def test_stale_grading_red(self, s3):
        _put(s3, f"backtest/{RUN_DATE}/grading.json", _GRADING)
        future = datetime.now(UTC) + timedelta(hours=200)  # > 192h red-line
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3, as_of=future)
        assert _comp(tile, "grading_freshness")["status"] == "RED"


class TestParity:
    def test_clean_parity_green(self, s3):
        _put(s3, f"backtest/{RUN_DATE}/parity_report.json",
             {"data_state": "ok", "trade_count_divergence": {}, "ticker_set_divergence": {}})
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert _comp(tile, "vectorized_vs_consolidated_parity")["status"] == "GREEN"

    def test_replay_error_missing_input(self, s3):
        _put(s3, f"backtest/{RUN_DATE}/parity_report.json", {"data_state": "backtester_replay_error"})
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        p = _comp(tile, "vectorized_vs_consolidated_parity")
        assert p["status"] == "N/A-MISSING-INPUT"
        assert "backtester_replay_error" in p["status_reason"]


class TestFdrSurface:
    def test_healthy_band_green(self, s3):
        corr = {f"f{i}": {"r10_fdr_significant": True} for i in range(5)}
        _put(s3, f"backtest/{RUN_DATE}/attribution.json",
             {"status": "ok", "rows_analyzed": 500, "correlations": corr})
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        fdr = _comp(tile, "fdr_surface_health")
        assert fdr["value"] == 5.0
        assert fdr["status"] == "GREEN"  # 3 ≤ 5 ≤ 15

    def test_zero_significant_red(self, s3):
        _put(s3, f"backtest/{RUN_DATE}/attribution.json",
             {"status": "ok", "rows_analyzed": 500, "correlations": {"f1": {"r10_fdr_significant": False}}})
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert _comp(tile, "fdr_surface_health")["status"] == "RED"

    def test_empty_attribution_tolerated(self, s3):
        # corrupt/empty artifact → N/A, not a crash.
        _put(s3, f"backtest/{RUN_DATE}/attribution.json", "", raw=True)
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert _comp(tile, "fdr_surface_health")["status"] == "N/A-MISSING-INPUT"


class TestRollbackAndNA:
    def test_rollback_count(self, s3):
        _put(s3, "config/rollback_audit/2026-05-01.json", {"x": 1})
        _put(s3, "config/rollback_audit/2026-05-08.json", {"x": 2})
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        rb = _comp(tile, "auto_apply_rollback_count")
        assert rb["value"] == 2.0

    def test_not_impl_components(self, s3):
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        # sample_size_adequacy is now WIRED (config#1151) — no longer a stub here.
        for name in ("optimizer_churn", "backtest_vs_live_parity", "walk_forward_stability"):
            assert _comp(tile, name)["status"] == "N/A-NOT-IMPL"


class TestSampleSizeAdequacy:
    """sample_size_adequacy reads backtest/{date}/sample_size.json (config#1151)."""

    def _put_ss(self, s3, ratio, weakest="signal_quality", per=None):
        per = per or {"signal_quality": {"n": int(ratio * 60), "floor": 60, "adequacy_ratio": ratio}}
        _put(s3, f"backtest/{RUN_DATE}/sample_size.json", {
            "status": "ok", "adequacy_ratio": ratio, "adequate": ratio >= 1.0,
            "weakest_analysis": weakest, "per_analysis": per})

    def test_missing_input(self, s3):
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "sample_size_adequacy")
        assert m["status"] == "N/A-MISSING-INPUT"
        assert "config#1151" in m["status_reason"]
        assert m["criticality"] == "critical"

    def test_well_powered_green(self, s3):
        self._put_ss(s3, 1.5)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "sample_size_adequacy")
        assert m["status"] == "GREEN"
        assert m["value"] == 1.5
        assert "well-powered" in m["status_reason"]

    def test_building_watch(self, s3):
        self._put_ss(s3, 0.7)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "sample_size_adequacy")
        assert m["status"] == "WATCH"

    def test_severely_underpowered_red(self, s3):
        self._put_ss(s3, 0.3, weakest="attribution",
                     per={"attribution": {"n": 30, "floor": 100, "adequacy_ratio": 0.3}})
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "sample_size_adequacy")
        assert m["status"] == "RED"
        assert "under-powered" in m["status_reason"]
        assert m["n_samples"] == 30

    def test_windowed_grades_off_earlier_date(self, s3):
        # Producer ran 2 days before run_date (partial/off-cycle) → still grades.
        _put(s3, "backtest/2026-06-05/sample_size.json", {
            "status": "ok", "adequacy_ratio": 1.2, "adequate": True,
            "weakest_analysis": "signal_quality",
            "per_analysis": {"signal_quality": {"n": 72, "floor": 60, "adequacy_ratio": 1.2}}})
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "sample_size_adequacy")
        assert m["status"] == "GREEN" and m["value"] == 1.2
