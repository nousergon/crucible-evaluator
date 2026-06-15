"""Tests for grading/tiles/predictor.py — Tile 2, leak-free IC sourcing."""

import json
from datetime import timedelta

import boto3
import pytest
from moto import mock_aws

from grading.tiles.predictor import (
    LATEST_KEY,
    MANIFEST_KEY,
    SLIM_CACHE_PREFIX,
    build_predictor_tile,
)

BUCKET = "alpha-engine-research"

# CPCV per-path ICs whose mean ≈ 0.18, mostly positive (mirrors live manifest).
_CPCV_ICS = [0.32, 0.27, 0.05, 0.07, 0.18, 0.47, 0.27, 0.28, 0.13, 0.09, 0.19, -0.14]

_MANIFEST = {
    "walk_forward": {
        "momentum_median_ic": -0.0015,   # dead
        "volatility_median_ic": 0.322,   # strong — but MAGNITUDE IC (abs-return)
        "n_folds": 16,
    },
    "meta_model_oos_ic_cpcv": {
        "status": "ok", "n_combos": 12, "mean_ic": 0.18,
        "frac_positive": 0.917, "ics": _CPCV_ICS,
    },
    # config#1062: the directional standalone alpha-ICs (each L1 output's IC vs
    # the SAME signed-alpha label the L2 targets). These are the apples-to-apples
    # comparison set for ensemble lift — NOT the magnitude volatility_median_ic.
    "meta_l1_standalone_alpha_ic": {
        "expected_move": {"xsec_ic": 0.21, "n_dates": 400},
        "research_calibrator_prob": {"xsec_ic": 0.17, "n_dates": 400},
        "momentum_score": {"xsec_ic": -0.03, "n_dates": 400},
        # raw context features carry an xsec_ic too but must NOT enter best_l1
        "macro_spy_20d_return": {"xsec_ic": 0.40, "n_dates": 400},
    },
}

_LATEST = {
    "l2_ic": 0.525,  # IN-SAMPLE — must NOT be used as the graded meta IC
    "l1_ic": {"momentum": 0.002, "volatility": 0.328, "research_calibrator": None},
    "research_calibrator_n_samples": 42,
    "confidence_calibration": {"ece_after": 0.0001, "n_samples": 929},
    "output_distribution_gate": {"passed": True, "reason": "all checks passed"},
    "n_predictions_today": 29,
}


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _seed(s3, latest=_LATEST, manifest=_MANIFEST):
    if latest is not None:
        s3.put_object(Bucket=BUCKET, Key=LATEST_KEY, Body=json.dumps(latest).encode())
    if manifest is not None:
        s3.put_object(Bucket=BUCKET, Key=MANIFEST_KEY, Body=json.dumps(manifest).encode())


def _comp(tile, name):
    return next(c for c in tile["components"] if c["name"] == name)


class TestMissing:
    def test_both_absent_watch(self, s3):
        tile = build_predictor_tile(BUCKET, s3_client=s3)
        assert tile["module"] == "predictor"
        # Single critical N/A-MISSING-INPUT component → tile WATCH (not false GREEN).
        assert tile["status"] == "WATCH"


class TestLeakFreeSourcing:
    def test_meta_ic_uses_cpcv_not_in_sample(self, s3):
        _seed(s3)
        tile = build_predictor_tile(BUCKET, s3_client=s3)
        meta = _comp(tile, "meta_l2_ic")
        # Graded value is the leak-free CPCV mean (0.18), NOT in-sample l2_ic (0.525).
        assert meta["value"] == pytest.approx(0.18)
        assert meta["value"] != pytest.approx(0.525)
        assert meta["ci_method"] == "bootstrap"
        assert meta["ci_low"] is not None
        # 0.18 > target 0.05 with CI clear of 0 → GREEN.
        assert meta["status"] == "GREEN"
        assert "in-sample" in meta["status_reason"].lower()

    def test_momentum_l1_dead_is_red(self, s3):
        _seed(s3)
        tile = build_predictor_tile(BUCKET, s3_client=s3)
        mom = _comp(tile, "momentum_l1_ic")
        assert mom["value"] == pytest.approx(-0.0015)
        assert mom["status"] == "RED"

    def test_volatility_l1_strong_is_green(self, s3):
        _seed(s3)
        tile = build_predictor_tile(BUCKET, s3_client=s3)
        vol = _comp(tile, "volatility_l1_ic")
        assert vol["status"] == "GREEN"

    def test_ensemble_lift_vs_directional_l1_not_magnitude(self, s3):
        # config#1062: lift compares meta CPCV (0.18) vs the BEST DIRECTIONAL
        # standalone L1 alpha-IC (expected_move 0.21), NOT the magnitude
        # volatility_median_ic (0.322) and NOT the raw macro context feature
        # (macro_spy_20d_return 0.40). 0.18 − 0.21 = −0.03.
        _seed(s3)
        tile = build_predictor_tile(BUCKET, s3_client=s3)
        lift = _comp(tile, "ensemble_lift_over_best_l1")
        assert lift["value"] == pytest.approx(0.18 - 0.21)
        # best L1 is the directional expected_move, not the 0.322 magnitude IC
        # nor the 0.40 macro context feature.
        assert "expected_move" in lift["status_reason"]
        assert lift["status"] == "RED"  # −0.03 below red-line −0.01

    def test_ensemble_lift_na_when_standalone_absent(self, s3):
        # config#1062: with no directional standalone alpha-IC in the manifest we
        # must NOT fall back to the magnitude WF IC (the old false-RED bug) —
        # honest N/A instead.
        manifest = {k: v for k, v in _MANIFEST.items() if k != "meta_l1_standalone_alpha_ic"}
        _seed(s3, manifest=manifest)
        tile = build_predictor_tile(BUCKET, s3_client=s3)
        lift = _comp(tile, "ensemble_lift_over_best_l1")
        assert lift["value"] is None
        assert lift["status"] == "N/A-MISSING-INPUT"
        assert "magnitude" in lift["status_reason"].lower()

    def test_ensemble_lift_na_when_standalone_not_run(self, s3):
        # status not_run/error → treated as absent, no magnitude fallback.
        manifest = {**_MANIFEST, "meta_l1_standalone_alpha_ic": {"status": "not_run"}}
        _seed(s3, manifest=manifest)
        tile = build_predictor_tile(BUCKET, s3_client=s3)
        assert _comp(tile, "ensemble_lift_over_best_l1")["status"] == "N/A-MISSING-INPUT"

    def test_volatility_magnitude_ic_still_reported(self, s3):
        # The magnitude volatility IC is excluded from LIFT but MUST remain
        # reported in its own component (a valid magnitude metric — config#1062).
        _seed(s3)
        tile = build_predictor_tile(BUCKET, s3_client=s3)
        vol = _comp(tile, "volatility_l1_ic")
        assert vol["value"] == pytest.approx(0.322)
        assert vol["status"] == "GREEN"

    def test_research_calibrator_null_low_n(self, s3):
        _seed(s3)
        tile = build_predictor_tile(BUCKET, s3_client=s3)
        rc = _comp(tile, "research_calibrator_l1_ic")
        assert rc["status"] == "N/A-LOW-N"

    def test_ece_green(self, s3):
        _seed(s3)
        tile = build_predictor_tile(BUCKET, s3_client=s3)
        ece = _comp(tile, "confidence_calibration_ece")
        assert ece["value"] == pytest.approx(0.0001)
        assert ece["status"] == "GREEN"

    def test_output_gate_pass_green(self, s3):
        _seed(s3)
        tile = build_predictor_tile(BUCKET, s3_client=s3)
        odg = _comp(tile, "output_distribution_gate")
        assert odg["status"] == "GREEN"

    def test_output_gate_fail_red(self, s3):
        latest = {**_LATEST, "output_distribution_gate": {"passed": False, "reason": "flat p_up"}}
        _seed(s3, latest=latest)
        tile = build_predictor_tile(BUCKET, s3_client=s3)
        assert _comp(tile, "output_distribution_gate")["status"] == "RED"


class TestNAComponents:
    def test_cross_tile_and_unresolved_na(self, s3):
        _seed(s3)
        tile = build_predictor_tile(BUCKET, s3_client=s3)
        assert _comp(tile, "veto_gate_precision")["status"] == "N/A-MISSING-INPUT"
        assert _comp(tile, "inference_coverage")["status"] == "N/A-MISSING-INPUT"
        assert _comp(tile, "slim_cache_freshness")["status"] == "N/A-MISSING-INPUT"
        assert _comp(tile, "feature_drift_ks")["status"] == "N/A-NOT-IMPL"
        # coverage reason carries the observed prediction count for context.
        assert "29" in _comp(tile, "inference_coverage")["status_reason"]


class TestInferenceCoverage:
    """config#1075: graded from the producer-persisted universe denominator."""

    def test_high_coverage_green(self, s3):
        latest = {**_LATEST, "n_universe": 30, "n_universe_covered": 29}
        _seed(s3, latest=latest)
        ic = _comp(build_predictor_tile(BUCKET, s3_client=s3), "inference_coverage")
        assert ic["value"] == pytest.approx(29 / 30)
        assert ic["status"] == "GREEN"  # 96.7% > target 95%
        assert "29/30" in ic["status_reason"]

    def test_low_coverage_red(self, s3):
        latest = {**_LATEST, "n_universe": 30, "n_universe_covered": 20}
        _seed(s3, latest=latest)
        ic = _comp(build_predictor_tile(BUCKET, s3_client=s3), "inference_coverage")
        assert ic["value"] == pytest.approx(20 / 30)
        assert ic["status"] == "RED"  # 66.7% < red-line 80%

    def test_absent_denominator_is_missing_input(self, s3):
        _seed(s3)  # _LATEST has no n_universe
        ic = _comp(build_predictor_tile(BUCKET, s3_client=s3), "inference_coverage")
        assert ic["status"] == "N/A-MISSING-INPUT"

    def test_zero_universe_is_missing_input(self, s3):
        latest = {**_LATEST, "n_universe": 0, "n_universe_covered": 0}
        _seed(s3, latest=latest)
        ic = _comp(build_predictor_tile(BUCKET, s3_client=s3), "inference_coverage")
        assert ic["status"] == "N/A-MISSING-INPUT"  # no /0


_RUN_DATE = "2026-06-14"


def _put_veto(s3, payload):
    s3.put_object(
        Bucket=BUCKET, Key=f"backtest/{_RUN_DATE}/veto_analysis.json",
        Body=json.dumps(payload).encode(),
    )


class TestVetoGatePrecision:
    """config#859: precision of the veto gate at the LIVE threshold, read
    cross-tile from backtest/{run_date}/veto_analysis.json."""

    _OK = {
        "status": "ok", "current_threshold": 0.65, "base_rate": 0.5,
        "thresholds": [
            {"confidence": 0.55, "n_vetoes": 40, "true_negatives": 20,
             "precision": 0.50, "precision_ci_95": [0.35, 0.65]},
            {"confidence": 0.65, "n_vetoes": 50, "true_negatives": 35,
             "precision": 0.70, "precision_ci_95": [0.56, 0.81]},
        ],
    }

    def test_grades_precision_at_live_threshold(self, s3):
        _seed(s3)
        _put_veto(s3, self._OK)
        vg = _comp(build_predictor_tile(BUCKET, _RUN_DATE, s3_client=s3), "veto_gate_precision")
        assert vg["value"] == pytest.approx(0.70)  # the 0.65 entry, not the 0.55 one
        assert vg["status"] == "GREEN"  # 70% > target 60%
        assert vg["ci_method"] == "wilson"
        assert "0.65" in vg["status_reason"]

    def test_low_precision_red(self, s3):
        _seed(s3)
        payload = {**self._OK, "thresholds": [
            {"confidence": 0.65, "n_vetoes": 50, "true_negatives": 15,
             "precision": 0.30, "precision_ci_95": [0.18, 0.44]},
        ]}
        _put_veto(s3, payload)
        vg = _comp(build_predictor_tile(BUCKET, _RUN_DATE, s3_client=s3), "veto_gate_precision")
        assert vg["status"] == "RED"  # 30% < red-line 40%

    def test_absent_artifact_missing_input(self, s3):
        _seed(s3)  # no veto_analysis.json
        vg = _comp(build_predictor_tile(BUCKET, _RUN_DATE, s3_client=s3), "veto_gate_precision")
        assert vg["status"] == "N/A-MISSING-INPUT"
        assert "absent" in vg["status_reason"]

    def test_no_run_date_missing_input(self, s3):
        _seed(s3)
        _put_veto(s3, self._OK)  # present, but run_date not passed
        vg = _comp(build_predictor_tile(BUCKET, s3_client=s3), "veto_gate_precision")
        assert vg["status"] == "N/A-MISSING-INPUT"
        assert "run_date not provided" in vg["status_reason"]

    def test_insufficient_status_no_precision_na(self, s3):
        _seed(s3)
        _put_veto(s3, {"status": "insufficient_data", "current_threshold": 0.65, "thresholds": []})
        vg = _comp(build_predictor_tile(BUCKET, _RUN_DATE, s3_client=s3), "veto_gate_precision")
        assert vg["status"] == "N/A-MISSING-INPUT"
        assert "insufficient_data" in vg["status_reason"]


class TestSlimCacheFreshness:
    def test_wired_from_s3_lastmodified(self, s3):
        # config#859: slim_cache_freshness is now sourced from the slim-cache
        # objects' LastModified instead of grading a permanent N/A.
        _seed(s3)
        s3.put_object(Bucket=BUCKET, Key=f"{SLIM_CACHE_PREFIX}AAPL.parquet", Body=b"x")
        # Read back the moto-assigned mtime so the age assertion is deterministic.
        mtime = s3.list_objects_v2(Bucket=BUCKET, Prefix=SLIM_CACHE_PREFIX)["Contents"][0]["LastModified"]
        tile = build_predictor_tile(BUCKET, s3_client=s3, as_of=mtime + timedelta(days=3))
        sc = _comp(tile, "slim_cache_freshness")
        assert sc["value"] == pytest.approx(3.0, abs=0.01)
        assert sc["status"] == "GREEN"  # 3d < target 7d
        assert "slim_cache_freshness = 3.0d" in sc["status_reason"]

    def test_stale_slim_cache_grades_red(self, s3):
        _seed(s3)
        s3.put_object(Bucket=BUCKET, Key=f"{SLIM_CACHE_PREFIX}AAPL.parquet", Body=b"x")
        mtime = s3.list_objects_v2(Bucket=BUCKET, Prefix=SLIM_CACHE_PREFIX)["Contents"][0]["LastModified"]
        tile = build_predictor_tile(BUCKET, s3_client=s3, as_of=mtime + timedelta(days=20))
        assert _comp(tile, "slim_cache_freshness")["status"] == "RED"  # 20d > red-line 14d

    def test_absent_slim_cache_is_missing_input(self, s3):
        _seed(s3)  # no slim-cache objects
        sc = _comp(build_predictor_tile(BUCKET, s3_client=s3), "slim_cache_freshness")
        assert sc["status"] == "N/A-MISSING-INPUT"
        assert "price_cache_slim" in sc["status_reason"]


class TestTileStatus:
    def test_dead_momentum_makes_tile_red(self, s3):
        _seed(s3)
        tile = build_predictor_tile(BUCKET, s3_client=s3)
        # momentum_l1_ic is critical + RED → module RED (honest: a dead critical L1).
        assert tile["status"] == "RED"
        assert tile["letter"] == "F"
