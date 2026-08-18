"""The card's consumer half for the run's own scope (config-I7620).

One rule dominates every test here: **an absent, degraded or unrecognised scope
block means "grade nothing", never "everything ran".** A card that grades the
full stage list against a run that dispatched three of them is confidently
wrong; a card that says it does not know is merely uninformative, and the
difference is the whole reason the block exists.

Payloads cross a real `json.loads(json.dumps(...))` round-trip. The producer is
in another repo and the block arrives from S3, so the interned-literal trap that
made the pipeline-gate verdict permanently UNKNOWN (config-I7614) is reachable
here too — and a test that builds the dict in Python cannot see it.
"""
from __future__ import annotations

import json

import pytest

from grading.run_scope import (
    DISPOSITIONS,
    GRADED_DISPOSITIONS,
    MEASURED,
    RUN_SCOPE_KEY,
    UNKNOWN,
    read_run_scope,
    scope_unknown,
)


def _wire(payload: dict) -> dict:
    """The block as the card actually receives it — off S3, never interned."""
    return json.loads(json.dumps(payload))


def _scope(**stages) -> dict:
    return _wire({
        "schema": "run_scope-1.0.0",
        "schema_version": 1,
        "run_date": "2026-08-14",
        "stages": {
            name: ({"disposition": v} if isinstance(v, str) else v)
            for name, v in stages.items()
        },
    })


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("degenerate", [
    None, {}, [], "not a block", 7,
    {"stages": {}},
    {"stages": "not a mapping"},
    {"degraded": True, "degraded_reason": "RuntimeError: no such execution",
     "stages": {"Scanner": {"disposition": "ENABLED_COMPLETED"}}},
])
def test_no_degenerate_block_ever_grades_anything(degenerate):
    block = read_run_scope(degenerate)
    assert block["verdict"] == UNKNOWN
    assert block["graded_stages"] == []
    assert block["graded_count"] == 0
    assert scope_unknown(block) is True
    assert block["statement"].startswith("SCOPE UNKNOWN")


def test_a_degraded_producer_is_unmeasured_not_narrow():
    """The Lambda's own fail-open writes `degraded: true` WITH stage rows.

    Those rows are all NOT_REACHED, but the card must not treat the block as a
    legitimately narrow run — the difference between "we ran three stages" and
    "we could not establish what ran" is the difference between a real result
    and no result.
    """
    block = read_run_scope(_wire({
        "degraded": True,
        "degraded_reason": "ClientError: AccessDenied on GetExecutionHistory",
        "stages": {"Scanner": {"disposition": "NOT_REACHED"}},
    }))
    assert block["verdict"] == UNKNOWN
    assert "AccessDenied" in block["statement"]
    assert "unmeasured, not narrow" in block["statement"]


# ---------------------------------------------------------------------------
# The measured path
# ---------------------------------------------------------------------------


def test_only_dispatched_stages_are_graded():
    """Grading follows DISPATCH, not success.

    `ENABLED_FAILED` is graded — as a failure. If grading followed success, a
    stage could silently drop out of the denominator by crashing, which is the
    defect class this whole mechanism exists to close.
    """
    block = read_run_scope(_scope(
        Scanner="ENABLED_COMPLETED",
        Backtester="ENABLED_FAILED",
        Parity={"disposition": "DISABLED", "disabled_by": "skip_parity"},
        PitParityLookahead="NOT_REACHED",
    ))
    assert block["verdict"] == MEASURED
    assert block["graded_stages"] == ["Backtester", "Scanner"]
    assert block["graded_count"] == 2
    assert block["stage_count"] == 4
    assert scope_unknown(block) is False


def test_disabled_and_not_reached_are_never_merged():
    """Excluded from grading for OPPOSITE reasons — one is a decision, the
    other is an absence of evidence. Collapsing them would let a run that died
    at stage 3 render as a deliberately narrow, fully green cycle."""
    block = read_run_scope(_scope(
        Parity={"disposition": "DISABLED", "disabled_by": "skip_parity"},
        PitParityLookahead="NOT_REACHED",
    ))
    assert block["disabled_stages"] == ["Parity"]
    assert block["not_reached_stages"] == ["PitParityLookahead"]
    assert block["disabled_by"] == ["skip_parity"]


def test_the_statement_names_the_flag_and_the_denominator():
    """The sentence rendered beside the grade.

    "GREEN" over an unstated denominator is not a falsifiable claim, and an
    operator reading "disabled" needs the flag to flip, not just the count.
    """
    block = read_run_scope(_scope(
        Scanner="ENABLED_COMPLETED",
        Parity={"disposition": "DISABLED", "disabled_by": "skip_parity"},
        Backtester={"disposition": "DISABLED", "disabled_by": "skip_backtester"},
    ))
    assert "1 of 3 gated stages dispatched and graded" in block["statement"]
    assert "skip_backtester, skip_parity" in block["statement"]


def test_a_failed_stage_is_named_not_just_counted():
    block = read_run_scope(_scope(
        Backtester="ENABLED_FAILED", Scanner="ENABLED_COMPLETED",
    ))
    assert "did NOT complete (Backtester)" in block["statement"]


# ---------------------------------------------------------------------------
# The boundary
# ---------------------------------------------------------------------------


def test_an_unrecognised_disposition_is_recorded_and_not_graded():
    """A producer that grows a fifth value must WITHHOLD, not slip through.

    And it must not vanish either: a stage the card cannot classify is a stage
    the card cannot claim to have covered, so it is named in the statement.
    """
    block = read_run_scope(_scope(
        Scanner="ENABLED_COMPLETED", Mystery="PROBABLY_FINE",
    ))
    assert block["graded_stages"] == ["Scanner"]
    assert block["unrecognised_stages"] == ["Mystery"]
    assert "does not recognise" in block["statement"]


def test_the_vocabulary_matches_the_producer():
    """Restated at the boundary, so a producer-side rename fails here rather
    than degrading every card to an empty denominator in silence."""
    assert DISPOSITIONS == {
        "DISABLED", "ENABLED_COMPLETED", "ENABLED_FAILED", "NOT_REACHED",
    }
    assert GRADED_DISPOSITIONS == {"ENABLED_COMPLETED", "ENABLED_FAILED"}
    assert RUN_SCOPE_KEY == "run_scope"


def test_scope_unknown_withholds_on_an_unfamiliar_verdict():
    """Expressed as != MEASURED, not == UNKNOWN, and by VALUE not identity —
    the block crosses a JSON boundary, which is exactly what made the
    pipeline-gate verdict permanently UNKNOWN (config-I7614)."""
    assert scope_unknown({"verdict": "SOMETHING_NEW"}) is True
    assert scope_unknown(_wire({"verdict": MEASURED})) is False
    assert scope_unknown(None) is True


# ---------------------------------------------------------------------------
# The Director's surface — both polarities, always
# ---------------------------------------------------------------------------


def _digest(card: dict) -> str:
    from director.report_card_digest import summarize_report_card  # noqa: PLC0415
    return summarize_report_card(card)


def _card(**over) -> dict:
    card = {"run_date": "2026-08-14", "overall": {"grade": 55.0}, "tiles": {}}
    card.update(over)
    return card


def test_the_digest_states_the_denominator_on_a_measured_run():
    """Rendered on the GOOD week too. A scope line that appears only when the
    run was narrow is indistinguishable from a producer that stopped emitting
    one — the same absence-is-not-evidence rule the two verdicts above it
    already follow."""
    card = _card(**{RUN_SCOPE_KEY: read_run_scope(_scope(
        Scanner="ENABLED_COMPLETED", Evaluator="ENABLED_COMPLETED",
    ))})
    text = _digest(card)
    assert "RUN SCOPE: 2 of 2 gated stages dispatched and graded" in text
    assert "⚠ RUN SCOPE" not in text


def test_the_digest_flags_an_absent_scope_and_says_what_it_hides():
    """The pre-I7620 state of the world, and it must read as a warning rather
    than as silence — that silence is what let the 2026-08-14 plan call a
    deliberately disabled producer one that "never ran"."""
    text = _digest(_card())
    assert "⚠ RUN SCOPE: UNKNOWN" in text
    assert "switched off by an operator flag" in text


def test_the_digest_names_the_flag_so_the_director_stops_guessing():
    card = _card(**{RUN_SCOPE_KEY: read_run_scope(_scope(
        Scanner="ENABLED_COMPLETED",
        Parity={"disposition": "DISABLED", "disabled_by": "skip_parity"},
    ))})
    text = _digest(card)
    assert "disabled by skip_parity" in text
