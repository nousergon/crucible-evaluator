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


class TestFeatureDriftKS:
    def _fdk(self, max_ks):
        return {"feature_drift_ks": {
            "max_ks": max_ks, "mean_ks": round(max_ks * 0.7, 4), "n_features": 6,
            "n_samples": 120, "per_feature": {"expected_move": max_ks},
        }}

    def test_low_drift_green(self, s3):
        _seed(s3, latest=self._fdk(0.06))
        c = _comp(build_predictor_tile(BUCKET, s3_client=s3), "feature_drift_ks")
        assert c["value"] == 0.06
        assert c["status"] == "GREEN"  # 0.06 <= target 0.10 (lower is better)

    def test_high_drift_red(self, s3):
        _seed(s3, latest=self._fdk(0.40))
        c = _comp(build_predictor_tile(BUCKET, s3_client=s3), "feature_drift_ks")
        assert c["status"] == "RED"  # 0.40 >= red-line 0.25

    def test_mid_drift_watch(self, s3):
        _seed(s3, latest=self._fdk(0.18))
        c = _comp(build_predictor_tile(BUCKET, s3_client=s3), "feature_drift_ks")
        assert c["status"] == "WATCH"

    def test_absent_block_missing_input(self, s3):
        _seed(s3, latest={"status": "ok"})
        c = _comp(build_predictor_tile(BUCKET, s3_client=s3), "feature_drift_ks")
        assert c["status"] == "N/A-MISSING-INPUT"


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
        # feature_drift_ks is now IMPLEMENTED (config#859) — absent block →
        # MISSING-INPUT, not NOT-IMPL.
        assert _comp(tile, "feature_drift_ks")["status"] == "N/A-MISSING-INPUT"
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


def _put_confusion(s3, payload, run_date=_RUN_DATE):
    s3.put_object(
        Bucket=BUCKET, Key=f"backtest/{run_date}/confusion_matrix.json",
        Body=json.dumps(payload).encode(),
    )


class TestDirectionAccuracyVsBaseline:
    """config#2298: direction-classification accuracy vs the majority-class
    baseline, plus per-class (UP/DOWN) precision vs base rate — reproducing
    the original incident's manual read (39.96% accuracy vs 60.6% always-DOWN
    baseline; UP precision 31.3% vs 36.0% base rate; DOWN precision 55.2% vs
    60.6% base rate) as a standing report-card tile."""

    # Mirrors the original incident's numbers (confusion_matrix.json n=1379).
    _CM = {
        "status": "ok", "n": 1379, "accuracy": 0.3996,
        "matrix": {
            "UP": {"UP": 156, "FLAT": 100, "DOWN": 200},
            "FLAT": {"UP": 50, "FLAT": 40, "DOWN": 60},
            "DOWN": {"UP": 150, "FLAT": 120, "DOWN": 503},
        },
        "per_class": {
            "UP": {"precision": 0.313, "recall": 0.4, "f1": 0.35, "n_predicted": 456, "n_actual": 356},
            "FLAT": {"precision": 0.267, "recall": 0.2, "f1": 0.23, "n_predicted": 150, "n_actual": 260},
            "DOWN": {"precision": 0.552, "recall": 0.65, "f1": 0.6, "n_predicted": 773, "n_actual": 763},
        },
        "up_threshold": 0.005, "horizons_days": [21],
    }

    def test_accuracy_lift_matches_incident_and_is_red(self, s3):
        _seed(s3)
        _put_confusion(s3, self._CM)
        tile = build_predictor_tile(BUCKET, _RUN_DATE, s3_client=s3)
        c = _comp(tile, "direction_accuracy_vs_majority_baseline")
        baseline = 763 / 1379  # DOWN is the majority actual class
        assert c["value"] == pytest.approx(0.3996 - baseline, abs=1e-4)
        assert c["status"] == "RED"  # deeply sub-baseline, mirrors the 2026-06 incident
        assert "DOWN" in c["status_reason"]
        assert c["n_samples"] == 1379

    def test_up_precision_lift_matches_incident(self, s3):
        _seed(s3)
        _put_confusion(s3, self._CM)
        tile = build_predictor_tile(BUCKET, _RUN_DATE, s3_client=s3)
        c = _comp(tile, "up_precision_vs_base_rate")
        base_rate = 356 / 1379
        assert c["value"] == pytest.approx(0.313 - base_rate, abs=1e-3)
        assert c["n_samples"] == 456

    def test_down_precision_lift_matches_incident(self, s3):
        _seed(s3)
        _put_confusion(s3, self._CM)
        tile = build_predictor_tile(BUCKET, _RUN_DATE, s3_client=s3)
        c = _comp(tile, "down_precision_vs_base_rate")
        base_rate = 763 / 1379
        assert c["value"] == pytest.approx(0.552 - base_rate, abs=1e-3)
        assert c["n_samples"] == 773

    def test_beats_baseline_is_green(self, s3):
        _seed(s3)
        cm = {
            "status": "ok", "n": 500, "accuracy": 0.70,
            "per_class": {
                "UP": {"precision": 0.65, "recall": 0.6, "f1": 0.62, "n_predicted": 200, "n_actual": 190},
                "FLAT": {"precision": 0.10, "recall": 0.1, "f1": 0.10, "n_predicted": 20, "n_actual": 20},
                "DOWN": {"precision": 0.75, "recall": 0.7, "f1": 0.72, "n_predicted": 280, "n_actual": 290},
            },
        }
        _put_confusion(s3, cm)
        tile = build_predictor_tile(BUCKET, _RUN_DATE, s3_client=s3)
        c = _comp(tile, "direction_accuracy_vs_majority_baseline")
        # majority baseline = 290/500 = 0.58; accuracy 0.70 → lift +0.12, well above +3pp target.
        assert c["status"] == "GREEN"

    def test_absent_artifact_missing_input(self, s3):
        _seed(s3)  # no confusion_matrix.json
        tile = build_predictor_tile(BUCKET, _RUN_DATE, s3_client=s3)
        for name in (
            "direction_accuracy_vs_majority_baseline",
            "up_precision_vs_base_rate", "down_precision_vs_base_rate",
        ):
            c = _comp(tile, name)
            assert c["status"] == "N/A-MISSING-INPUT"
            assert "absent" in c["status_reason"]

    def test_no_run_date_missing_input(self, s3):
        _seed(s3)
        _put_confusion(s3, self._CM)
        tile = build_predictor_tile(BUCKET, s3_client=s3)  # no run_date
        c = _comp(tile, "direction_accuracy_vs_majority_baseline")
        assert c["status"] == "N/A-MISSING-INPUT"
        assert "run_date not provided" in c["status_reason"]

    def test_insufficient_data_status_is_missing_input(self, s3):
        _seed(s3)
        _put_confusion(s3, {"status": "insufficient_data", "n": 10, "min_required": 30})
        tile = build_predictor_tile(BUCKET, _RUN_DATE, s3_client=s3)
        c = _comp(tile, "direction_accuracy_vs_majority_baseline")
        assert c["status"] == "N/A-MISSING-INPUT"
        assert "insufficient_data" in c["status_reason"]

    def test_zero_predicted_class_precision_undefined_na(self, s3):
        _seed(s3)
        cm = {
            "status": "ok", "n": 100, "accuracy": 0.5,
            "per_class": {
                "UP": {"precision": None, "recall": None, "f1": None, "n_predicted": 0, "n_actual": 30},
                "FLAT": {"precision": 0.4, "recall": 0.3, "f1": 0.34, "n_predicted": 30, "n_actual": 20},
                "DOWN": {"precision": 0.6, "recall": 0.7, "f1": 0.65, "n_predicted": 70, "n_actual": 50},
            },
        }
        _put_confusion(s3, cm)
        tile = build_predictor_tile(BUCKET, _RUN_DATE, s3_client=s3)
        c = _comp(tile, "up_precision_vs_base_rate")
        assert c["status"] == "N/A-MISSING-INPUT"
        assert "n_predicted=0" in c["status_reason"]


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


class TestICReliability:
    """alpha-engine-config#969: the producer (crucible-predictor#340) tags each
    scalar IC field with methodological reliability in the manifest's
    ``ic_reliability`` map; the tile passes it into build_metric so a
    low-reliability IC flags "⚠ reliability LOW" in the Director digest.
    Forward/backward compatible: a manifest WITHOUT the map preserves today's
    behavior (value-bearing criticals default to "high")."""

    # A manifest whose CPCV meta IC is tagged LOW and momentum WF IC tagged HIGH.
    _RELIABILITY_MAP = {
        "meta_model_oos_ic_cpcv": "low",
        "momentum_median_ic": "high",
        "volatility_median_ic": "high",
        "meta_l1_standalone_alpha_ic": "high",
    }

    def test_low_reliability_flows_into_metric(self, s3):
        manifest = {**_MANIFEST, "ic_reliability": self._RELIABILITY_MAP}
        _seed(s3, manifest=manifest)
        tile = build_predictor_tile(BUCKET, s3_client=s3)
        meta = _comp(tile, "meta_l2_ic")
        assert meta["reliability"] == "low"

    def test_high_reliability_flows_into_metric(self, s3):
        manifest = {**_MANIFEST, "ic_reliability": self._RELIABILITY_MAP}
        _seed(s3, manifest=manifest)
        tile = build_predictor_tile(BUCKET, s3_client=s3)
        assert _comp(tile, "momentum_l1_ic")["reliability"] == "high"

    def test_low_reliability_reaches_digest_warning(self, s3):
        # End-to-end: a LOW-reliability critical metric surfaces the digest's
        # "⚠ reliability LOW" hedge line (director/report_card_digest.py).
        from director.report_card_digest import _component_line

        manifest = {**_MANIFEST, "ic_reliability": self._RELIABILITY_MAP}
        _seed(s3, manifest=manifest)
        tile = build_predictor_tile(BUCKET, s3_client=s3)
        meta = _comp(tile, "meta_l2_ic")
        line = _component_line(meta)
        assert "reliability LOW" in line

    def test_missing_map_preserves_current_high_default(self, s3):
        # No ic_reliability key at all (older predictor manifest). The
        # value-bearing critical meta_l2_ic must keep TODAY's behavior:
        # build_metric defaults it to "high" (we do NOT force it low, and we do
        # NOT drop the field).
        assert "ic_reliability" not in _MANIFEST
        _seed(s3)  # default _MANIFEST, no map
        tile = build_predictor_tile(BUCKET, s3_client=s3)
        meta = _comp(tile, "meta_l2_ic")
        assert meta["reliability"] == "high"  # unchanged current semantics

    def test_field_absent_from_map_falls_back(self, s3):
        # Map present but does NOT cover the CPCV field → fall back to current
        # behavior (default "high" for the value-bearing critical), don't error.
        manifest = {**_MANIFEST, "ic_reliability": {"momentum_median_ic": "high"}}
        _seed(s3, manifest=manifest)
        tile = build_predictor_tile(BUCKET, s3_client=s3)
        assert _comp(tile, "meta_l2_ic")["reliability"] == "high"
        assert _comp(tile, "momentum_l1_ic")["reliability"] == "high"

    def test_malformed_map_value_ignored(self, s3):
        # A garbage value in the map must be ignored (fall back), never passed
        # through as a bogus reliability.
        manifest = {**_MANIFEST, "ic_reliability": {"meta_model_oos_ic_cpcv": "maybe"}}
        _seed(s3, manifest=manifest)
        tile = build_predictor_tile(BUCKET, s3_client=s3)
        assert _comp(tile, "meta_l2_ic")["reliability"] == "high"


class TestTileStatus:
    def test_dead_momentum_makes_tile_red(self, s3):
        _seed(s3)
        tile = build_predictor_tile(BUCKET, s3_client=s3)
        # momentum_l1_ic is critical + RED → module RED (honest: a dead critical L1).
        assert tile["status"] == "RED"
        assert tile["letter"] == "F"


class TestCrossCycleTrends:
    """config#1836/#1958 — the predictor tile threads prior-card trends into
    its headline critical (meta_l2_ic) when a CardHistory is supplied, same
    contract as the research/portfolio_outcome tiles (grading/history.py:
    N/A weeks skipped, never zero-filled; no history → default trends)."""

    def test_history_threads_trends_onto_meta_l2_ic(self, s3):
        from grading.history import CardHistory

        _seed(s3)  # default _MANIFEST → cpcv_mean == 0.18 (mean of _CPCV_ICS)
        history = CardHistory({("predictor", "meta_l2_ic"): [0.10, 0.12, 0.14]}, 3)
        tile = build_predictor_tile(BUCKET, s3_client=s3, history=history)

        meta = _comp(tile, "meta_l2_ic")
        assert meta["trend_4w"] == [0.10, 0.12, 0.14, pytest.approx(0.18)]
        assert meta["trend_13w"] == meta["trend_4w"]
        # Monotonic improvement across all 4 points → an up glyph, not the
        # perpetual default "→".
        assert meta["trend_decoration"] in ("↑", "↑↑")

    def test_gap_week_skipped_not_zero_filled(self, s3):
        # A prior N/A week for meta_l2_ic must contribute no point to the
        # series (CardHistory itself enforces the skip at extraction time;
        # this pins that the tile carries the already-skipped series through
        # untouched rather than re-introducing a zero-fill at the call site).
        from grading.history import CardHistory

        _seed(s3)
        history = CardHistory({("predictor", "meta_l2_ic"): [0.20, 0.24]}, 3)
        tile = build_predictor_tile(BUCKET, s3_client=s3, history=history)
        meta = _comp(tile, "meta_l2_ic")
        # Prior series has only 2 points (the middle N/A week already absent,
        # never a 0.0) + this cycle's value appended.
        assert meta["trend_4w"] == [0.20, 0.24, pytest.approx(0.18)]
        assert 0.0 not in meta["trend_4w"]

    def test_no_history_keeps_default_trends(self, s3):
        _seed(s3)
        tile = build_predictor_tile(BUCKET, s3_client=s3)
        meta = _comp(tile, "meta_l2_ic")
        assert meta["trend_4w"] is None
        assert meta["trend_13w"] is None
        assert meta["trend_decoration"] == "→"

    def test_na_current_value_uses_prior_history_only(self, s3):
        # meta_l2_ic grades N/A this cycle (both artifacts absent) — the
        # tile must still ride the prior series without appending a
        # nonexistent current value.
        from grading.history import CardHistory

        history = CardHistory({("predictor", "meta_l2_ic"): [0.10, 0.12]}, 2)
        tile = build_predictor_tile(BUCKET, s3_client=s3, history=history)
        meta = _comp(tile, "meta_l2_ic")
        assert meta["trend_4w"] == [0.10, 0.12]
