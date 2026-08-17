"""Tests for grading/tiles/contribution_lift.py — Tile 10 (RC v3 T5, config-I7473).

Fixture shape follows contribution_lift_contract.md (RC v3 T5 artifact
contract agreed with the crucible-backtester producer harness).
"""

import json

import boto3
import pytest
from moto import mock_aws

from grading.tiles.contribution_lift import KNOWN_COMPONENTS, build_contribution_lift_tile

BUCKET = "alpha-engine-research"
RUN_DATE = "2026-08-15"  # a Saturday


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _put(s3, data):
    s3.put_object(
        Bucket=BUCKET, Key=f"backtest/{RUN_DATE}/contribution_lift.json",
        Body=json.dumps(data).encode(),
    )


def _comp(tile, name):
    return next(c for c in tile["components"] if c["name"] == name)


FULL_DOC = {
    "schema_version": 1,
    "status": "ok",
    "run_date": RUN_DATE,
    "objective": {
        "name": "log_alpha_21d_net_of_cost_vs_spy", "horizon_days": 21,
        "fees": 0.001, "slippage_bps": 0.0, "init_cash": 1000000.0,
    },
    "window": {"start": "2026-05-01", "end": RUN_DATE, "n_cycles": 64, "n_floor": 60},
    "inputs": {"price_matrix_shape": [64, 900], "n_signal_dates": 64, "source_paths": []},
    "n_trials_cumulative": 1234,
    "components": [
        {
            "name": "risk_guard", "module": "executor", "criticality": "critical",
            "pattern": "null_arm", "issue": "alpha-engine-config-I7482",
            "status": "ok", "status_reason": "risk_guard graded ok this cycle.",
            "value": 0.0123, "unit": "log_alpha_21d",
            "ci_low": -0.001, "ci_high": 0.03, "ci_method": "bootstrap",
            "n_samples": 64, "n_floor": 60, "count_matched": True,
            "source_path": f"s3://{BUCKET}/backtest/{RUN_DATE}/contribution_lift.json#components/risk_guard",
        },
        {
            "name": "cost_adjusted_quality", "module": "behavioral", "criticality": "supporting",
            "pattern": "substitution", "issue": "alpha-engine-config-I7484",
            "status": "ok", "status_reason": "cost_adjusted_quality graded ok this cycle.",
            "value": 0.02, "unit": "log_alpha_21d",
            "ci_low": 0.001, "ci_high": 0.04, "ci_method": "bootstrap",
            "n_samples": 64, "n_floor": 60, "count_matched": True,
            "source_path": f"s3://{BUCKET}/backtest/{RUN_DATE}/contribution_lift.json#components/cost_adjusted_quality",
        },
        {
            "name": "sector_teams_avg", "module": "research", "criticality": "critical",
            "pattern": "substitution", "issue": "alpha-engine-config-I7478",
            "status": "gap",
            "status_reason": "arm width mismatch: baseline had 5 picks/cycle, ablated arm produced 3 — not count-matched.",
            "value": None, "unit": None, "ci_low": None, "ci_high": None, "ci_method": None,
            "n_samples": None, "n_floor": 60, "count_matched": False,
            "source_path": f"s3://{BUCKET}/backtest/{RUN_DATE}/contribution_lift.json#components/sector_teams_avg",
        },
        {
            "name": "veto_gate_precision", "module": "predictor", "criticality": "supporting",
            "pattern": "null_arm", "issue": "alpha-engine-config-I7480",
            "status": "N/A-MISSING-INPUT",
            "status_reason": "veto_gate_precision: predictor_sizing.json absent this cycle.",
            "value": None, "unit": None, "ci_low": None, "ci_high": None, "ci_method": None,
            "n_samples": None, "n_floor": 60, "count_matched": None,
            "source_path": f"s3://{BUCKET}/backtest/{RUN_DATE}/contribution_lift.json#components/veto_gate_precision",
        },
    ],
}


class TestMissingArtifact:
    def test_no_artifact_grades_every_known_component_na(self, s3):
        tile = build_contribution_lift_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert tile["n_components"] == len(KNOWN_COMPONENTS)
        for name, module in KNOWN_COMPONENTS.items():
            c = _comp(tile, f"{name}_contribution_lift")
            assert c["status"] == "N/A-MISSING-INPUT"
            assert c["module"] == module
            assert c["metric_type"] == "contribution_lift"
            assert "contribution_lift.json" in c["status_reason"] or "contribution_lift.json" in (c.get("source_path") or "")

    def test_producer_error_status_also_grades_na(self, s3):
        _put(s3, {"schema_version": 1, "status": "error", "reason": "replay harness crashed mid-cycle", "run_date": RUN_DATE})
        tile = build_contribution_lift_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert tile["n_components"] == len(KNOWN_COMPONENTS)
        c = _comp(tile, "risk_guard_contribution_lift")
        assert c["status"] == "N/A-MISSING-INPUT"
        assert "replay harness crashed mid-cycle" in c["status_reason"]


class TestFullArtifact:
    def test_ok_component_shape(self, s3):
        _put(s3, FULL_DOC)
        tile = build_contribution_lift_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert tile["n_components"] == 4
        c = _comp(tile, "risk_guard_contribution_lift")
        assert c["module"] == "executor"
        assert c["metric_type"] == "contribution_lift"
        assert c["unit"] == "log_alpha_21d"
        assert c["red_line"] == 0.0
        assert c["value"] == 0.0123
        assert c["n_floor"] == 60
        assert c["status"] in ("GREEN", "WATCH", "RED")  # derived, not passed through
        assert c["criticality"] == "critical"
        assert c["estimator"] == "paired_cycle_bootstrap"

    def test_renders_beside_owning_module(self, s3):
        _put(s3, FULL_DOC)
        tile = build_contribution_lift_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert _comp(tile, "cost_adjusted_quality_contribution_lift")["module"] == "behavioral"
        assert _comp(tile, "sector_teams_avg_contribution_lift")["module"] == "research"
        assert _comp(tile, "veto_gate_precision_contribution_lift")["module"] == "predictor"

    def test_gap_status_maps_to_na_missing_input_with_verbatim_reason(self, s3):
        _put(s3, FULL_DOC)
        tile = build_contribution_lift_tile(BUCKET, RUN_DATE, s3_client=s3)
        c = _comp(tile, "sector_teams_avg_contribution_lift")
        assert c["status"] == "N/A-MISSING-INPUT"
        assert "arm width mismatch" in c["status_reason"]
        assert c["status_reason"].startswith("gap:")

    def test_na_missing_input_passthrough(self, s3):
        _put(s3, FULL_DOC)
        tile = build_contribution_lift_tile(BUCKET, RUN_DATE, s3_client=s3)
        c = _comp(tile, "veto_gate_precision_contribution_lift")
        assert c["status"] == "N/A-MISSING-INPUT"
        assert c["status_reason"] == "veto_gate_precision: predictor_sizing.json absent this cycle."

    def test_unrecognized_producer_status_fails_loud(self, s3):
        doc = json.loads(json.dumps(FULL_DOC))
        doc["components"][0]["status"] = "N/A-BOGUS-STATUS"
        _put(s3, doc)
        with pytest.raises(ValueError, match="unrecognized status"):
            build_contribution_lift_tile(BUCKET, RUN_DATE, s3_client=s3)


class TestContractShape:
    """Every emitted component honours the contract's fixed fields."""

    def test_every_value_bearing_component_carries_unit_and_zero_red_line(self, s3):
        _put(s3, FULL_DOC)
        tile = build_contribution_lift_tile(BUCKET, RUN_DATE, s3_client=s3)
        for c in tile["components"]:
            if c["value"] is not None:
                assert c["unit"] == "log_alpha_21d"
            assert c["red_line"] == 0.0
            assert c["n_floor"] == 60
            assert c["name"].endswith("_contribution_lift")
