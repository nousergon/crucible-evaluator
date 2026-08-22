"""A producer-declared terminal state must not render as an open gap.

alpha-engine-config-I8177. ``grading/tiles/contribution_lift.py`` remapped the
producer's ``N/A-RETIRED`` and ``N/A-NOT-LIFT-SHAPED`` onto a bare
``N/A-NOT-IMPL`` — "declared and never built". That inverts the meaning:
``N/A-NOT-IMPL`` is an open gap someone is expected to close, whereas these are
the producer stating, with a written reason, that the component will never
carry a lift number.

The ``contribution_lift.json`` producer has shipped (146KB, 26 components,
written 2026-08-22). Nine of its components carried a declared terminal state
on the 2026-08-22 card:

* ``N/A-RETIRED`` (2) — ``sector_teams_avg``, ``cio_selection_skill``, whose
  only source is the six-team/CIO research graph retired 2026-07-12
  (``config#1580`` / ``alpha-engine-config-I2993``).
* ``N/A-NOT-LIFT-SHAPED`` (7) — ``output_distribution_gate``,
  ``direction_accuracy_vs_majority_baseline``, ``thinktank_coverage_ic``,
  ``calibration_diagnostics``, ``attractiveness_trajectory_ic``,
  ``judge_outcome_ic``, ``judge_rubric_pass_rate``.

All nine were assertions that could never pass, permanently holding
``evaluator_coverage`` below 1.0. They now carry ``permanent_na=True`` with the
producer's reason verbatim, which is what ``grading/coverage.py`` reads to take
a declared component out of the coverage denominator.

``observability-policy.md`` §8.3 — RETIRED is DECLARED, never inferred; and
symmetrically a declared terminal state must not be rendered as an undeclared
gap.
"""

from __future__ import annotations

import pytest

from grading.tiles.contribution_lift import (
    _STATUS_DECLARED_TERMINAL,
    _STATUS_PASSTHROUGH,
    _STATUS_REMAP,
    build_contribution_lift_tile,
)

_BUCKET = "alpha-engine-research"
_DATE = "2026-08-21"


def _artifact(components):
    return {
        "schema_version": 1,
        "run_date": _DATE,
        "status": "ok",
        "objective": "net_of_cost_21d_log_alpha_vs_spy",
        "window": {"start": "2026-04-01", "end": _DATE},
        "inputs": {},
        "components": components,
    }


def _comp(name, status, *, module="research", reason="because"):
    return {
        "name": name, "module": module, "criticality": "supporting",
        "status": status, "status_reason": reason,
    }


@pytest.fixture
def s3_with(monkeypatch):
    def _make(components):
        import json

        import boto3
        from moto import mock_aws

        ctx = mock_aws()
        ctx.start()
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=_BUCKET)
        client.put_object(
            Bucket=_BUCKET,
            Key=f"backtest/{_DATE}/contribution_lift.json",
            Body=json.dumps(_artifact(components)).encode(),
        )
        return client, ctx
    made = []
    yield lambda c: (made.append(_make(c)) or made[-1])[0]
    for _, ctx in made:
        ctx.stop()


def _component(tile, name):
    return next(c for c in tile["components"] if c["name"] == name)


# --------------------------------------------------------------------------

def test_declared_terminal_set_is_exactly_the_two_producer_states() -> None:
    assert _STATUS_DECLARED_TERMINAL == {"N/A-RETIRED", "N/A-NOT-LIFT-SHAPED"}


def test_declared_states_are_no_longer_in_the_plain_remap() -> None:
    """The regression this file exists for."""
    assert "N/A-RETIRED" not in _STATUS_REMAP
    assert "N/A-NOT-LIFT-SHAPED" not in _STATUS_REMAP
    assert _STATUS_REMAP == {"gap": "N/A-MISSING-INPUT"}


@pytest.mark.parametrize("declared", ["N/A-RETIRED", "N/A-NOT-LIFT-SHAPED"])
def test_declared_state_sets_permanent_na(s3_with, declared) -> None:
    s3 = s3_with([_comp("sector_teams_avg", declared, reason="graph retired 2026-07-12")])
    tile = build_contribution_lift_tile(_BUCKET, _DATE, s3_client=s3)
    record = _component(tile, "sector_teams_avg_contribution_lift")
    assert record["permanent_na"] is True, (
        "a declared terminal state must leave the coverage denominator"
    )


@pytest.mark.parametrize("declared", ["N/A-RETIRED", "N/A-NOT-LIFT-SHAPED"])
def test_producer_reason_survives_verbatim(s3_with, declared) -> None:
    """The remap must never be lossy — a reader sees WHICH state and WHY."""
    s3 = s3_with([_comp("judge_outcome_ic", declared, reason="validates, does not steer")])
    tile = build_contribution_lift_tile(_BUCKET, _DATE, s3_client=s3)
    record = _component(tile, "judge_outcome_ic_contribution_lift")
    assert declared in record["permanent_na_reason"]
    assert "validates, does not steer" in record["permanent_na_reason"]
    assert declared in record["status_reason"]


def test_gap_still_counts_against_coverage(s3_with) -> None:
    """A width mismatch is a real producer defect, NOT a declared retirement."""
    s3 = s3_with([_comp("risk_guard", "gap", module="executor",
                        reason="width mismatch on 14/39 shared cycles")])
    tile = build_contribution_lift_tile(_BUCKET, _DATE, s3_client=s3)
    record = _component(tile, "risk_guard_contribution_lift")
    assert record["status"] == "N/A-MISSING-INPUT"
    assert record["permanent_na"] is False, (
        "a producer defect must stay in the denominator until it is fixed"
    )


def test_missing_input_passthrough_is_not_declared_out(s3_with) -> None:
    s3 = s3_with([_comp("macro_agent", "N/A-MISSING-INPUT")])
    tile = build_contribution_lift_tile(_BUCKET, _DATE, s3_client=s3)
    assert _component(tile, "macro_agent_contribution_lift")["permanent_na"] is False
    assert "N/A-MISSING-INPUT" in _STATUS_PASSTHROUGH


def test_unrecognized_producer_status_still_fails_loud(s3_with) -> None:
    """Widening the taxonomy must not become a fall-through."""
    s3 = s3_with([_comp("macro_agent", "N/A-WHO-KNOWS")])
    with pytest.raises(ValueError, match="unrecognized status"):
        build_contribution_lift_tile(_BUCKET, _DATE, s3_client=s3)


def test_the_nine_live_declared_components_all_leave_the_denominator(s3_with) -> None:
    """The exact set measured on the 2026-08-21 artifact."""
    live = [
        ("sector_teams_avg", "N/A-RETIRED", "research"),
        ("cio_selection_skill", "N/A-RETIRED", "research"),
        ("output_distribution_gate", "N/A-NOT-LIFT-SHAPED", "predictor"),
        ("direction_accuracy_vs_majority_baseline", "N/A-NOT-LIFT-SHAPED", "predictor"),
        ("thinktank_coverage_ic", "N/A-NOT-LIFT-SHAPED", "research"),
        ("calibration_diagnostics", "N/A-NOT-LIFT-SHAPED", "research"),
        ("attractiveness_trajectory_ic", "N/A-NOT-LIFT-SHAPED", "research"),
        ("judge_outcome_ic", "N/A-NOT-LIFT-SHAPED", "research"),
        ("judge_rubric_pass_rate", "N/A-NOT-LIFT-SHAPED", "research"),
    ]
    s3 = s3_with([_comp(n, s, module=m) for n, s, m in live])
    tile = build_contribution_lift_tile(_BUCKET, _DATE, s3_client=s3)

    from grading.coverage import card_component_census

    census = card_component_census({"contribution_lift": tile})
    assert census["total"] == 0, (
        f"all nine must be declared out, still counted: {census['ungraded']}"
    )
    assert len(census["declared_out"]) == 9
