"""A converged optimizer is not a blocked one — alpha-engine-config-I7654.

`apply_loop_health` graded every loop `blocked` for >=4 weeks as RED without
reading why, and the producer stamps `blocked` for two structurally different
states. Measured on the live `config/apply_audit/2026-08-14.json`: two of the
three "unhealthy" loops were proposing EXACTLY what was already live — an
optimizer agreeing with the champion, which is the designed outcome of a
champion/challenger loop, not a stuck one.

The cost: the 2026-08-14 Director plan escalated "3/4 auto-apply loops blocked
— the optimization loop is broken" as a P1, and buried the sentence that
actually mattered (`executor_params`: all 60 combos alpha-negative) inside a
count of three.
"""
from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws

from grading.tiles.backtester import _loop_converged, build_backtester_tile

BUCKET = "alpha-engine-research"
RUN_DATE = "2026-08-14"

#: The live artifact, verbatim in the fields that decide the grade.
LIVE_LOOPS = {
    "scoring_weights": {
        "outcome": "blocked", "blocked_by": ["min_meaningful_change"],
        "consecutive_blocked_weeks": 9,
        "detail": "all changes < 2% — not worth updating",
        "proposed": {"quant": 0.5, "qual": 0.5},
        "current": {"quant": 0.5, "qual": 0.5},
    },
    "executor_params": {
        "outcome": "blocked", "blocked_by": ["alpha_floor"],
        "consecutive_blocked_weeks": 7,
        "detail": "All 60 valid combos backtested with total_alpha < 0.0",
        "proposed": None, "current": None,
    },
    "predictor_params": {
        "outcome": "blocked", "blocked_by": ["significance_floor"],
        "consecutive_blocked_weeks": 9,
        "proposed": {"veto_confidence": 0.65},
        "current": {"veto_confidence": 0.65},
    },
    "research_params": {
        "outcome": "disabled", "blocked_by": None,
        "consecutive_blocked_weeks": 0, "proposed": None, "current": None,
    },
}


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _health(s3, loops: dict) -> dict:
    s3.put_object(
        Bucket=BUCKET, Key=f"config/apply_audit/{RUN_DATE}.json",
        Body=json.dumps({
            "schema_version": 1, "as_of": RUN_DATE, "loops": loops,
        }).encode(),
    )
    tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
    return next(c for c in tile["components"] if c["name"] == "apply_loop_health")


# --------------------------------------------------------------------------
# _loop_converged
# --------------------------------------------------------------------------

def test_a_proposal_equal_to_live_is_converged():
    assert _loop_converged(LIVE_LOOPS["scoring_weights"]) is True
    assert _loop_converged(LIVE_LOOPS["predictor_params"]) is True


def test_no_proposal_at_all_is_NOT_converged():
    """The load-bearing guard. Without `proposed is not None`, `None == None`
    grades the one genuinely-blocked loop as converged — turning this metric
    from over-firing into SILENT, which is strictly worse than the defect."""
    assert _loop_converged(LIVE_LOOPS["executor_params"]) is False
    assert _loop_converged({"proposed": None, "current": {"a": 1}}) is False
    assert _loop_converged({}) is False


def test_a_genuinely_different_proposal_is_not_converged():
    assert _loop_converged({"proposed": {"a": 2}, "current": {"a": 1}}) is False


# --------------------------------------------------------------------------
# The graded metric
# --------------------------------------------------------------------------

def test_the_live_artifact_grades_one_unhealthy_not_three(s3):
    c = _health(s3, LIVE_LOOPS)
    assert c["value"] == 1.0
    assert "executor_params" in c["status_reason"]
    assert c["status"] == "RED"  # executor_params is real, and must still fire


def test_the_converged_loops_are_named_not_dropped(s3):
    """Nine weeks of agreeing with the champion is worth SEEING — a reason
    listing only the unhealthy ones cannot be told from one whose producer
    went quiet."""
    reason = _health(s3, LIVE_LOOPS)["status_reason"]
    assert "Converged (proposal == live)" in reason
    assert "scoring_weights=converged(9w)" in reason
    assert "predictor_params=converged(9w)" in reason


def test_a_real_block_on_a_real_proposal_still_escalates(s3):
    """The fix must not disarm the metric. A loop proposing something genuinely
    different, blocked 5 weeks, is exactly what config#1841 built this for."""
    loops = {
        "scoring_weights": {
            "outcome": "blocked", "blocked_by": ["min_meaningful_change"],
            "consecutive_blocked_weeks": 5,
            "proposed": {"quant": 0.7, "qual": 0.3},
            "current": {"quant": 0.5, "qual": 0.5},
        },
    }
    c = _health(s3, loops)
    assert c["value"] == 1.0
    assert c["status"] == "RED"
    assert "blocked 5w" in c["status_reason"]


def test_all_loops_converged_grades_green(s3):
    loops = {
        k: v for k, v in LIVE_LOOPS.items() if k != "executor_params"
    }
    c = _health(s3, loops)
    assert c["value"] == 0.0
    assert c["status"] == "GREEN"


def test_an_error_outcome_is_untouched_by_this_change(s3):
    c = _health(s3, {"x": {"outcome": "error", "detail": "boom",
                           "proposed": {"a": 1}, "current": {"a": 1}}})
    assert c["status"] == "RED"
