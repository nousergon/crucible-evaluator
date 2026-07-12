"""Tests for grading/tiles/backtester.py — Tile 4 (self-grade)."""

import json
from datetime import UTC, datetime, timedelta

import boto3
import pytest
from moto import mock_aws

from grading.tiles.backtester import _coverage, _fdr_surface_status, build_backtester_tile

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
    """config#2305: the band scales with n_fdr_tests (the actual size of the
    shared BH-FDR correction pool), not a fixed absolute count — the fixed
    3-15 band was calibrated for the pre-config#1456 ~8-9-test surface and is
    unreachable/misleading on the post-cutover 4-5-test surface. Legacy
    artifacts without n_fdr_tests still get the old fixed-band behavior."""

    def test_healthy_band_green_with_n_fdr_tests(self, s3):
        # 3 of 10 tests significant = 0.30 fraction, within [0.15, 0.60].
        corr = {f"f{i}": {"r10_fdr_significant": i < 3} for i in range(10)}
        _put(s3, f"backtest/{RUN_DATE}/attribution.json",
             {"status": "ok", "rows_analyzed": 500, "n_fdr_tests": 10, "correlations": corr})
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        fdr = _comp(tile, "fdr_surface_health")
        assert fdr["value"] == 3.0
        assert fdr["status"] == "GREEN"

    def test_zero_significant_on_small_surface_is_watch_not_red(self, s3):
        """The live 2026-07-10 scenario: 4 tests (2 sub-scores x 2 canonical
        horizons post-config#1456), 0 survive BH correction — this is the
        statistically EXPECTED outcome at this sample size (best raw p=0.034
        still fails the n=4 rank-1 threshold of 0.0125), not evidence the
        surface went flat. Must be WATCH, not RED (the config#2305 bug)."""
        corr = {
            "quant": {"beat_spy_21d_fdr_significant": False, "return_21d_fdr_significant": False},
            "qual": {"beat_spy_21d_fdr_significant": False, "return_21d_fdr_significant": False},
        }
        _put(s3, f"backtest/{RUN_DATE}/attribution.json",
             {"status": "ok", "rows_analyzed": 402, "n_fdr_tests": 4, "correlations": corr})
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        fdr = _comp(tile, "fdr_surface_health")
        assert fdr["value"] == 0.0
        assert fdr["status"] == "WATCH"

    def test_zero_significant_on_large_surface_is_red(self, s3):
        """0 survivors IS informative once the correction pool is large
        enough for the test to have real discriminating power."""
        corr = {f"f{i}": {"r_fdr_significant": False} for i in range(12)}
        _put(s3, f"backtest/{RUN_DATE}/attribution.json",
             {"status": "ok", "rows_analyzed": 500, "n_fdr_tests": 12, "correlations": corr})
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert _comp(tile, "fdr_surface_health")["status"] == "RED"

    def test_all_significant_is_watch_not_green(self, s3):
        """Every test surviving correction reads as overfit-suspicious, not
        healthy — fraction 1.0 is outside the [0.15, 0.60] GREEN band."""
        corr = {f"f{i}": {"r_fdr_significant": True} for i in range(4)}
        _put(s3, f"backtest/{RUN_DATE}/attribution.json",
             {"status": "ok", "rows_analyzed": 500, "n_fdr_tests": 4, "correlations": corr})
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert _comp(tile, "fdr_surface_health")["status"] == "WATCH"

    def test_legacy_artifact_without_n_fdr_tests_uses_fixed_band(self, s3):
        """Older attribution.json artifacts (written before config#2305)
        lack n_fdr_tests entirely — must fall back to the original fixed
        3-15 band rather than erroring or silently reinterpreting."""
        corr = {f"f{i}": {"r10_fdr_significant": True} for i in range(5)}
        _put(s3, f"backtest/{RUN_DATE}/attribution.json",
             {"status": "ok", "rows_analyzed": 500, "correlations": corr})  # no n_fdr_tests key
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        fdr = _comp(tile, "fdr_surface_health")
        assert fdr["value"] == 5.0
        assert fdr["status"] == "GREEN"  # 3 <= 5 <= 15, legacy band

    def test_legacy_zero_significant_red(self, s3):
        _put(s3, f"backtest/{RUN_DATE}/attribution.json",
             {"status": "ok", "rows_analyzed": 500, "correlations": {"f1": {"r10_fdr_significant": False}}})
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert _comp(tile, "fdr_surface_health")["status"] == "RED"

    def test_empty_attribution_tolerated(self, s3):
        # corrupt/empty artifact → N/A, not a crash.
        _put(s3, f"backtest/{RUN_DATE}/attribution.json", "", raw=True)
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert _comp(tile, "fdr_surface_health")["status"] == "N/A-MISSING-INPUT"


class TestFdrSurfaceStatusHelper:
    """Direct unit tests for `_fdr_surface_status` — pins the calibration
    contract independent of the S3/tile plumbing."""

    def test_none_n_fdr_tests_uses_legacy_band(self):
        assert _fdr_surface_status(5, None)[0] == "GREEN"
        assert _fdr_surface_status(0, None)[0] == "RED"
        assert _fdr_surface_status(2, None)[0] == "WATCH"
        assert _fdr_surface_status(20, None)[0] == "WATCH"
        assert _fdr_surface_status(40, None)[0] == "RED"

    def test_zero_tests_edge_case_falls_back_to_legacy(self):
        # n_fdr_tests=0 shouldn't happen when status=="ok", but must not crash.
        status, _ = _fdr_surface_status(0, 0)
        assert status == "RED"  # legacy rule: n_sig=0 -> RED

    def test_small_surface_zero_sig_is_watch(self):
        for n in (1, 2, 3, 4, 5, 9):
            assert _fdr_surface_status(0, n)[0] == "WATCH", f"n_fdr_tests={n}"

    def test_large_surface_zero_sig_is_red(self):
        for n in (10, 15, 50):
            assert _fdr_surface_status(0, n)[0] == "RED", f"n_fdr_tests={n}"

    def test_proportional_green_band(self):
        # 0.15 <= frac <= 0.60
        assert _fdr_surface_status(2, 10)[0] == "GREEN"   # 0.20
        assert _fdr_surface_status(6, 10)[0] == "GREEN"   # 0.60 (inclusive)
        assert _fdr_surface_status(1, 10)[0] == "WATCH"   # 0.10, below floor
        assert _fdr_surface_status(7, 10)[0] == "WATCH"   # 0.70, above ceiling


class TestRollbackAndNA:
    def test_rollback_count(self, s3):
        _put(s3, "config/rollback_audit/2026-05-01.json", {"x": 1})
        _put(s3, "config/rollback_audit/2026-05-08.json", {"x": 2})
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        rb = _comp(tile, "auto_apply_rollback_count")
        assert rb["value"] == 2.0

    def test_not_impl_components(self, s3):
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        # sample_size_adequacy, optimizer_churn, walk_forward_stability are now
        # WIRED (config#1151) — they read producer artifacts, so absent ⇒
        # N/A-MISSING-INPUT, not N/A-NOT-IMPL. Only backtest_vs_live_parity is
        # still a genuine stub.
        assert _comp(tile, "backtest_vs_live_parity")["status"] == "N/A-NOT-IMPL"
        for name in ("optimizer_churn", "walk_forward_stability"):
            assert _comp(tile, name)["status"] == "N/A-MISSING-INPUT"


class TestOptimizerChurn:
    """optimizer_churn reads backtest/{date}/optimizer_churn.json (config#1151)."""

    def _put(self, s3, churn_ratio, within):
        _put(s3, f"backtest/{RUN_DATE}/optimizer_churn.json", {
            "status": "ok", "churn_ratio": churn_ratio, "max_abs_change": 0.06,
            "max_change_param": "momentum", "guardrail_cap": 0.10,
            "within_guardrails": within, "n_params_changed": 3})

    def test_missing_input(self, s3):
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "optimizer_churn")
        assert m["status"] == "N/A-MISSING-INPUT"
        assert "config#1151" in m["status_reason"]
        assert m["criticality"] == "critical"

    def test_within_guardrails_green(self, s3):
        self._put(s3, 0.6, True)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "optimizer_churn")
        assert m["status"] == "GREEN"
        assert m["value"] == 0.6
        assert "within guardrails" in m["status_reason"]

    def test_approaching_cap_watch(self, s3):
        self._put(s3, 0.9, True)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "optimizer_churn")
        assert m["status"] == "WATCH"

    def test_over_cap_red(self, s3):
        self._put(s3, 1.1, False)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "optimizer_churn")
        assert m["status"] == "RED"
        assert "AT/OVER" in m["status_reason"]

    def test_insufficient_data_na(self, s3):
        _put(s3, f"backtest/{RUN_DATE}/optimizer_churn.json",
             {"status": "insufficient_data", "reason": "no usable recommendation"})
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "optimizer_churn")
        assert m["status"] == "N/A-MISSING-INPUT"


class TestWalkForwardStability:
    """walk_forward_stability reads backtest/{date}/walk_forward_stability.json (config#1151)."""

    def _put(self, s3, stability_ratio, n_reversals, weeks=4, stable=True):
        _put(s3, f"backtest/{RUN_DATE}/walk_forward_stability.json", {
            "status": "ok", "stability_ratio": stability_ratio, "n_reversals": n_reversals,
            "max_possible_reversals": 10, "weeks_loaded": weeks, "stable": stable, "reversals": []})

    def test_missing_input(self, s3):
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "walk_forward_stability")
        assert m["status"] == "N/A-MISSING-INPUT"
        assert m["criticality"] == "supporting"

    def test_stable_green(self, s3):
        self._put(s3, 0.9, 1)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "walk_forward_stability")
        assert m["status"] == "GREEN"
        assert m["value"] == 0.9
        assert m["n_samples"] == 4

    def test_drifting_watch(self, s3):
        self._put(s3, 0.6, 4)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "walk_forward_stability")
        assert m["status"] == "WATCH"

    def test_oscillating_red(self, s3):
        self._put(s3, 0.3, 7, stable=False)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "walk_forward_stability")
        assert m["status"] == "RED"

    def test_insufficient_history_na(self, s3):
        _put(s3, f"backtest/{RUN_DATE}/walk_forward_stability.json",
             {"status": "insufficient_data", "reason": "only 1 prior week", "weeks_loaded": 1})
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "walk_forward_stability")
        assert m["status"] == "N/A-MISSING-INPUT"


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


class TestRiskRatioCI:
    """risk_ratio_ci reads backtest/{date}/risk_ratio_ci.json (config#976,
    Director L4558's magnitude-certainty no-action monitor)."""

    def _put_rr(self, s3, all_certain, n_samples=200, sample_floor=126, ratios=None):
        ratios = ratios if ratios is not None else {
            "sharpe_ratio": {"point": 1.1, "ci_95": [0.4, 1.8], "straddles_zero": False,
                              "magnitude_certain": all_certain},
            "sortino_ratio": {"point": 1.4, "ci_95": [0.6, 2.2], "straddles_zero": False,
                               "magnitude_certain": all_certain},
            "information_ratio": {"point": 0.3, "ci_95": [-0.1, 0.7], "straddles_zero": not all_certain,
                                   "magnitude_certain": all_certain},
        }
        _put(s3, f"backtest/{RUN_DATE}/risk_ratio_ci.json", {
            "status": "ok", "n_samples": n_samples, "sample_floor": sample_floor,
            "n_adequate": n_samples >= sample_floor, "ratios": ratios,
            "all_magnitude_certain": all_certain,
            "note": "Director L4558: size remediation by direction, not magnitude, until certain.",
        })

    def test_missing_input(self, s3):
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "risk_ratio_ci")
        assert m["status"] == "N/A-MISSING-INPUT"
        assert "config#976" in m["status_reason"]
        assert m["criticality"] == "critical"

    def test_all_certain_green(self, s3):
        self._put_rr(s3, all_certain=True, n_samples=200)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "risk_ratio_ci")
        assert m["status"] == "GREEN"
        assert m["value"] == 1.0
        assert m["n_samples"] == 200

    def test_magnitude_uncertain_watch_not_red(self, s3):
        # A no-action monitor: CI-straddles-zero is a data state to wait out,
        # not a regression — must grade WATCH, never RED.
        self._put_rr(s3, all_certain=False, n_samples=63)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "risk_ratio_ci")
        assert m["status"] == "WATCH"
        assert m["value"] == 0.0
        assert "information_ratio" in m["status_reason"]
        assert "DIRECTION" in m["status_reason"]

    def test_insufficient_data_status_grades_na(self, s3):
        _put(s3, f"backtest/{RUN_DATE}/risk_ratio_ci.json", {
            "status": "insufficient_data", "n_samples": 1, "sample_floor": 126,
            "n_adequate": False, "ratios": {}, "all_magnitude_certain": False,
            "note": "Fewer than 2 aligned daily returns — no ratio estimable.",
        })
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "risk_ratio_ci")
        assert m["status"].startswith("N/A")

    def test_windowed_grades_off_earlier_date(self, s3):
        _put(s3, "backtest/2026-06-05/risk_ratio_ci.json", {
            "status": "ok", "n_samples": 300, "sample_floor": 126, "n_adequate": True,
            "ratios": {
                "sharpe_ratio": {"point": 0.9, "ci_95": [0.2, 1.6], "straddles_zero": False,
                                  "magnitude_certain": True},
            },
            "all_magnitude_certain": True,
            "note": "x",
        })
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "risk_ratio_ci")
        assert m["status"] == "GREEN" and m["value"] == 1.0
