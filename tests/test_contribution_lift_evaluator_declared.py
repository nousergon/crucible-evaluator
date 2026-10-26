"""Evaluator-declared retirement of two ablation lifts (alpha-engine-config-I12584).

Option 1, "Retire", was taken at the 2026-10-25 decision-queue triage:
``entry_triggers_contribution_lift`` and ``risk_guard_contribution_lift`` are
declared permanent N/A because their arms differ BY CONSTRUCTION, which the
#12243 support assurance correctly refuses to pair. The declaration lives in the
evaluator because the v1 producer runs no weekly after 2026-10-24.

These tests pin the four properties the ruling asked for: the reason says "by
construction" (never "missing input") and names what the card loses; both
records leave the coverage denominator; the declaration yields to a measured
value (the event revisit trigger); and it lapses on its dated backstop.
"""

from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws

from grading.coverage import card_component_census
from grading.tiles.contribution_lift import (
    _EVALUATOR_DECLARED_PERMANENT_NA,
    _EVALUATOR_DECLARED_REVISIT_BY,
    _evaluator_declared_reason,
    build_contribution_lift_tile,
)

_BUCKET = "alpha-engine-research"
_DATE = "2026-10-23"  # the last v1 weekly card
_DECLARED = ("entry_triggers", "risk_guard")


def _gap(name, reason="width mismatch"):
    return {
        "name": name, "module": "executor", "criticality": "critical",
        "status": "gap", "status_reason": reason, "value": None,
    }


@pytest.fixture
def s3_with():
    ctxs = []

    def _make(components, run_date=_DATE):
        ctx = mock_aws()
        ctx.start()
        ctxs.append(ctx)
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=_BUCKET)
        client.put_object(
            Bucket=_BUCKET,
            Key=f"backtest/{run_date}/contribution_lift.json",
            Body=json.dumps({
                "schema_version": 1, "run_date": run_date, "status": "ok",
                "components": components,
            }).encode(),
        )
        return client

    yield _make
    for ctx in ctxs:
        ctx.stop()


def _component(tile, name):
    return next(c for c in tile["components"] if c["name"] == name)


def test_exactly_the_two_ruled_components_are_declared() -> None:
    # position_sizing is close to matchable and was not part of the ruling.
    assert set(_EVALUATOR_DECLARED_PERMANENT_NA) == set(_DECLARED)


@pytest.mark.parametrize("name", _DECLARED)
def test_reason_is_by_construction_and_names_the_loss(name) -> None:
    reason = _evaluator_declared_reason(name, _DATE)
    assert reason is not None
    assert "BY-CONSTRUCTION" in reason and "#12243" in reason
    assert "missing input" not in reason.lower()
    assert "no longer states a net-P&L contribution" in reason
    assert "I12584" in reason
    assert _EVALUATOR_DECLARED_REVISIT_BY in reason


@pytest.mark.parametrize("name", _DECLARED)
def test_gap_is_declared_out_with_producer_state_kept(s3_with, name) -> None:
    s3 = s3_with([_gap(name, reason="25 baseline-only dates")])
    tile = build_contribution_lift_tile(_BUCKET, _DATE, s3_client=s3)
    record = _component(tile, f"{name}_contribution_lift")
    assert record["permanent_na"] is True
    assert record["status"] == "N/A-NOT-IMPL"
    assert "BY-CONSTRUCTION" in record["permanent_na_reason"]
    # The producer's own state rides along, so the override is not lossy.
    assert "gap: 25 baseline-only dates" in record["permanent_na_reason"]


def test_both_leave_the_coverage_denominator(s3_with) -> None:
    s3 = s3_with([_gap("entry_triggers"), _gap("risk_guard"), _gap("position_sizing")])
    tile = build_contribution_lift_tile(_BUCKET, _DATE, s3_client=s3)
    census = card_component_census({"contribution_lift": tile})
    out = " ".join(census["declared_out"])
    assert "entry_triggers_contribution_lift" in out
    assert "risk_guard_contribution_lift" in out
    assert "position_sizing_contribution_lift" not in out


def test_a_measured_value_wins_over_the_declaration(s3_with) -> None:
    """EVENT revisit trigger: the arms shared support, so the reason no longer holds."""
    comp = {
        "name": "risk_guard", "module": "executor", "criticality": "critical",
        "status": "ok", "value": 0.0123, "ci_low": 0.001, "ci_high": 0.02,
        "ci_method": "bootstrap", "n_samples": 80,
    }
    s3 = s3_with([comp])
    tile = build_contribution_lift_tile(_BUCKET, _DATE, s3_client=s3)
    record = _component(tile, "risk_guard_contribution_lift")
    assert record["permanent_na"] is False
    assert record["status"] in ("GREEN", "WATCH", "RED")


def test_declaration_lapses_on_its_dated_backstop(s3_with) -> None:
    """DATED BACKSTOP: unrenewed, the retirement cannot outlive its date silently."""
    run_date = _EVALUATOR_DECLARED_REVISIT_BY
    assert _evaluator_declared_reason("entry_triggers", run_date) is None
    s3 = s3_with([_gap("entry_triggers")], run_date=run_date)
    tile = build_contribution_lift_tile(_BUCKET, run_date, s3_client=s3)
    record = _component(tile, "entry_triggers_contribution_lift")
    assert record["permanent_na"] is False
    assert record["status"] == "N/A-MISSING-INPUT"


def test_unrecognized_status_still_fails_loud_for_a_declared_component(s3_with) -> None:
    comp = _gap("entry_triggers")
    comp["status"] = "N/A-WHO-KNOWS"
    s3 = s3_with([comp])
    with pytest.raises(ValueError, match="unrecognized status"):
        build_contribution_lift_tile(_BUCKET, _DATE, s3_client=s3)
