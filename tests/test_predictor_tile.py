"""Tests for grading/tiles/predictor.py — Tile 2, leak-free IC sourcing."""

import json

import boto3
import pytest
from moto import mock_aws

from grading.tiles.predictor import (
    LATEST_KEY,
    MANIFEST_KEY,
    build_predictor_tile,
)

BUCKET = "alpha-engine-research"

# CPCV per-path ICs whose mean ≈ 0.18, mostly positive (mirrors live manifest).
_CPCV_ICS = [0.32, 0.27, 0.05, 0.07, 0.18, 0.47, 0.27, 0.28, 0.13, 0.09, 0.19, -0.14]

_MANIFEST = {
    "walk_forward": {
        "momentum_median_ic": -0.0015,   # dead
        "volatility_median_ic": 0.322,   # strong
        "n_folds": 16,
    },
    "meta_model_oos_ic_cpcv": {
        "status": "ok", "n_combos": 12, "mean_ic": 0.18,
        "frac_positive": 0.917, "ics": _CPCV_ICS,
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

    def test_ensemble_lift_negative_red(self, s3):
        _seed(s3)
        tile = build_predictor_tile(BUCKET, s3_client=s3)
        lift = _comp(tile, "ensemble_lift_over_best_l1")
        # meta 0.18 − best L1 (vol 0.322) = −0.142 → stacking adds no value → RED.
        assert lift["value"] < 0
        assert lift["status"] == "RED"

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


class TestTileStatus:
    def test_dead_momentum_makes_tile_red(self, s3):
        _seed(s3)
        tile = build_predictor_tile(BUCKET, s3_client=s3)
        # momentum_l1_ic is critical + RED → module RED (honest: a dead critical L1).
        assert tile["status"] == "RED"
        assert tile["letter"] == "F"
