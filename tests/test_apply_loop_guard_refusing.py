"""A guard refusing on a complete sample grades WATCH, not RED.

alpha-engine-config-I10064, Brian's ruling #18 (2026-10-14): keep
``alpha_floor`` at 0.0 and accept the 10-12 health rule with card colour WATCH.
Live ``config/apply_audit/2026-10-09.json``: ``executor_params`` blocked 16
audits on ``alpha_floor`` with 60 of 60 combos measured, coverage 1.0, and best
``total_alpha`` -0.10766. That is the guard refusing as designed. It graded
``apply_loop_health`` RED and kept the Director escalating a malfunction.

Every fault shape (unmeasured, insufficient sample, low coverage, an
unmeasured combo, a missing or malformed predicate) stays on the week rule.
"""
from __future__ import annotations

import copy
import json

import boto3
import pytest
from moto import mock_aws

from grading.tiles.backtester import _guard_refusing, build_backtester_tile

BUCKET = "alpha-engine-research"
RUN_DATE = "2026-10-09"

#: The live executor_params record, verbatim in the fields that decide the grade.
LIVE_EXECUTOR = {
    "outcome": "blocked",
    "blocked_by": ["alpha_floor"],
    "consecutive_blocked_weeks": 16,
    "detail": "All 60 MEASURED combos (of 60 valid) backtested with total_alpha < 0.0",
    "proposed": None,
    "current": None,
    "unblock_predicate": {
        "predicate_id": "executor_params.alpha_floor.unblock",
        "predicate_version": 1,
        "state": "blocked",
        "met": False,
        "alpha_floor": 0.0,
        "sample": {"n_valid_combos": 60, "n_measured": 60, "n_unmeasured": 0, "min_valid_combos": 3},
        "window": {"dates_simulated_min": 58, "dates_simulated_max": 58,
                   "dates_expected_max": 58, "coverage_min": 1.0},
        "n_at_or_above_floor": 0,
        "best_alpha": -0.10766,
    },
}

CONVERGED = {
    "outcome": "blocked", "blocked_by": ["min_meaningful_change"],
    "consecutive_blocked_weeks": 12,
    "proposed": {"quant": 0.5, "qual": 0.5}, "current": {"quant": 0.5, "qual": 0.5},
}


def _loops(executor: dict) -> dict:
    return {
        "executor_params": executor,
        "scoring_weights": CONVERGED,
        "predictor_params": {**CONVERGED, "proposed": {"v": 0.65}, "current": {"v": 0.65}},
        "research_params": {"outcome": "disabled", "blocked_by": None,
                            "consecutive_blocked_weeks": 0, "proposed": None, "current": None},
    }


def _with(**pred_changes) -> dict:
    e = copy.deepcopy(LIVE_EXECUTOR)
    for k, v in pred_changes.items():
        if k in ("n_unmeasured", "n_measured"):
            e["unblock_predicate"]["sample"][k] = v
        elif k == "coverage_min":
            e["unblock_predicate"]["window"][k] = v
        else:
            e["unblock_predicate"][k] = v
    return e


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _health(s3, loops: dict) -> dict:
    s3.put_object(
        Bucket=BUCKET, Key=f"config/apply_audit/{RUN_DATE}.json",
        Body=json.dumps({"schema_version": 1, "as_of": RUN_DATE, "loops": loops}).encode(),
    )
    tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
    return next(c for c in tile["components"] if c["name"] == "apply_loop_health")


def test_the_live_record_is_guard_refusing():
    assert _guard_refusing(LIVE_EXECUTOR) is True


def test_the_live_artifact_grades_watch_not_red(s3):
    c = _health(s3, _loops(LIVE_EXECUTOR))
    assert c["status"] == "WATCH"
    assert c["value"] == 1.0  # still counted and named, never dropped
    assert "executor_params blocked 16w [alpha_floor] guard_refusing" in c["status_reason"]
    assert "-0.10766" in c["status_reason"]


@pytest.mark.parametrize("change", [
    {"state": "unmeasured"},
    {"state": "insufficient_sample"},
    {"coverage_min": 0.94},
    {"n_unmeasured": 1},
    {"coverage_min": None},
    {"n_unmeasured": "0"},
    {"coverage_min": True},
])
def test_every_fault_shape_stays_on_the_week_rule_and_goes_red(s3, change):
    entry = _with(**change)
    assert _guard_refusing(entry) is False
    assert _health(s3, _loops(entry))["status"] == "RED"


def test_a_missing_or_malformed_predicate_is_never_a_refusal(s3):
    no_pred = {k: v for k, v in LIVE_EXECUTOR.items() if k != "unblock_predicate"}
    assert _guard_refusing(no_pred) is False
    assert _guard_refusing({**LIVE_EXECUTOR, "unblock_predicate": "blocked"}) is False
    assert _guard_refusing({**LIVE_EXECUTOR, "unblock_predicate": {"state": "blocked"}}) is False
    assert _health(s3, _loops(no_pred))["status"] == "RED"


def test_another_red_loop_still_makes_the_metric_red(s3):
    loops = _loops(LIVE_EXECUTOR)
    loops["research_params"] = {"outcome": "error", "detail": "boom"}
    c = _health(s3, loops)
    assert c["status"] == "RED"
    assert "guard_refusing" in c["status_reason"]


def test_a_one_week_refusal_is_unchanged_and_healthy(s3):
    entry = copy.deepcopy(LIVE_EXECUTOR)
    entry["consecutive_blocked_weeks"] = 1
    assert _health(s3, _loops(entry))["status"] == "GREEN"
