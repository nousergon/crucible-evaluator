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


# ════════════════════════════════════════════════════════════════════════════
# Delivery: in-band first, S3 as the fallback (alpha-engine-config-I7392)
#
# The S3 artifact was the ONLY delivery path, and it is the one path a
# REHEARSAL is forbidden to use: nousergon-data's RunScope Lambda derives the
# scope and skips its put_object when dry_run is true
# (infrastructure/lambdas/weekly-run-scope/index.py). So on the
# 2026-08-29T00:47Z Friday shell run the RunScope stage derived exactly the
# right answer — Parity: DISABLED, "CheckSkipParity was entered and took its
# skip branch" — the card could not read it, contamination_scope() hit
# scope_unknown, "absence is never out-of-scope" left the half IN scope, and it
# resolved to UNKNOWN. The whole NOT_IN_SCOPE mechanism built for exactly that
# case (config-I7620) was defeated by the DELIVERY PATH, not by the logic: the
# correct answer existed in memory and was discarded on the way to its consumer.
#
# The run's own scope now travels IN-BAND with the run — the ReportCard Task's
# `run_scope` payload key, threaded from $.run_scope_result.Payload
# (nousergon-data infrastructure/step_function.json). S3 is the FALLBACK, not
# the removal: a manual/CLI card build or a snapshot rebuild has no SF payload
# behind it.
# ════════════════════════════════════════════════════════════════════════════

import logging as _logging
import pytest
from botocore.exceptions import ClientError

import grading.artifacts as _artifacts
from grading.artifacts import (
    RUN_SCOPE_SOURCE_ABSENT,
    RUN_SCOPE_SOURCE_PAYLOAD,
    RUN_SCOPE_SOURCE_S3,
    _read_run_scope,
)

_IN_BAND = {"schema": "run_scope-1.0.0", "run_date": "2026-08-28",
            "stages": {"Parity": {"disposition": "DISABLED",
                                  "disabled_by": "skip_parity"}}}
_FROM_S3 = {"schema": "run_scope-1.0.0", "run_date": "2026-08-28", "stages": {}}


def _no_s3(monkeypatch, body=None):
    monkeypatch.setattr(_artifacts, "get_json", lambda *a, **k: body)


def test_the_in_band_payload_is_preferred(monkeypatch):
    """S3 is never consulted when the run carried its own scope."""
    calls = []
    monkeypatch.setattr(_artifacts, "get_json",
                        lambda *a, **k: calls.append(a) or _FROM_S3)
    block, source = _read_run_scope(None, "b", "backtest/2026-08-28",
                                    payload=_IN_BAND)
    assert block == _IN_BAND
    assert source == RUN_SCOPE_SOURCE_PAYLOAD
    assert calls == []


def test_s3_remains_the_fallback_not_the_removal(monkeypatch):
    """A manual/CLI build or a snapshot rebuild has no SF payload behind it."""
    _no_s3(monkeypatch, _FROM_S3)
    for payload in (None, {}):
        block, source = _read_run_scope(None, "b", "backtest/2026-08-28",
                                        payload=payload)
        assert block == _FROM_S3
        assert source == RUN_SCOPE_SOURCE_S3


def test_absent_from_both_paths_is_absent(monkeypatch):
    """And absence stays meaningful — read_run_scope resolves None to UNKNOWN
    with an empty graded set, never to 'everything ran'."""
    _no_s3(monkeypatch, None)
    block, source = _read_run_scope(None, "b", "backtest/2026-08-28", payload=None)
    assert block is None
    assert source == RUN_SCOPE_SOURCE_ABSENT
    assert read_run_scope(block)["verdict"] == "UNKNOWN"
    assert read_run_scope(block)["graded_stages"] == []


def test_a_payload_of_the_wrong_shape_is_loud(monkeypatch, caplog):
    """No silent swallow: a payload of the wrong SHAPE is a cross-repo contract
    breach, and falling through to S3 without saying so makes it invisible."""
    _no_s3(monkeypatch, _FROM_S3)
    with caplog.at_level(_logging.DEBUG, logger="grading.artifacts"):
        block, source = _read_run_scope(None, "b", "backtest/2026-08-28",
                                        payload="run_scope-1.0.0")
    assert block == _FROM_S3
    assert source == RUN_SCOPE_SOURCE_S3
    errors = [r.getMessage() for r in caplog.records if r.levelno >= _logging.ERROR]
    assert len(errors) == 1
    assert "run_scope payload is str" in errors[0]


def test_build_report_card_hands_the_in_band_scope_to_both_consumers(monkeypatch):
    """The two hops the fix depends on, asserted at the seam rather than
    through a full card build: ``build_report_card`` must pass the event's
    ``run_scope`` down to the artifact reader, and must pass the NORMALIZED
    block plus ``dry_run`` on to the attestation. Break either and the fix is
    inert while every other test still passes."""
    import grading.aggregate as aggregate
    from grading.attestation import NOT_IN_SCOPE

    seen = {}

    def _fake_reader(bucket, run_date, s3_client=None, run_scope_payload=None):
        seen["payload"] = run_scope_payload
        return {}, _StubReport(run_scope=run_scope_payload)

    def _fake_attestation(bucket, run_date, s3_client=None, *, run_scope=None,
                          dry_run=False):
        seen["run_scope"] = run_scope
        seen["dry_run"] = dry_run
        raise _Stop

    monkeypatch.setattr(aggregate, "assert_input_freshness", lambda *a, **k: {})
    monkeypatch.setattr(aggregate, "read_scorecard_inputs", _fake_reader)
    monkeypatch.setattr(aggregate, "build_run_attestation", _fake_attestation)
    monkeypatch.setattr(aggregate, "compute_scorecard", lambda **k: {"tiles": {}})
    monkeypatch.setattr(aggregate, "load_card_history", lambda *a, **k: None)
    monkeypatch.setattr(aggregate, "build_leaderboard", lambda *a, **k: None)
    # The ten tiles sit BETWEEN the two seams under test and each reads its own
    # artifacts (one reaches Step Functions through nousergon_lib). They are
    # graded by their own tests; stubbed here so this test fails for exactly one
    # reason.
    for _tile in ("portfolio_outcome", "predictor", "research", "executor",
                  "backtester", "substrate", "agent", "behavioral",
                  "director_quality", "contribution_lift"):
        monkeypatch.setattr(aggregate, f"build_{_tile}_tile",
                            lambda *a, **k: {"status": "GREEN", "components": []})

    with pytest.raises(_Stop):
        aggregate.build_report_card("b", "2026-08-28", s3_client=_AbsentS3(),
                                    dry_run=True, run_scope=_IN_BAND)

    assert seen["payload"] == _IN_BAND
    assert seen["dry_run"] is True
    assert seen["run_scope"]["verdict"] == "MEASURED"
    assert seen["run_scope"]["disabled_stages"] == ["Parity"]
    assert seen["run_scope"]["disabled_by"] == ["skip_parity"]
    assert NOT_IN_SCOPE  # the vocabulary the normalized block feeds


def test_the_handler_forwards_the_events_run_scope(monkeypatch):
    """The SF hands it in on the ReportCard Task's Payload; the handler must
    not drop it on the floor between the event and the build."""
    import grading.handler as handler

    seen = {}

    def _fake_build(bucket, run_date, **kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(handler, "build_report_card", _fake_build)
    monkeypatch.setattr(handler, "run_self_test", lambda **k: {"verdict": "PASS"})
    with pytest.raises(_Stop):
        handler.handler({"date": "2026-08-28", "dry_run": True, "write": False,
                         "run_scope": _IN_BAND})
    assert seen["run_scope"] == _IN_BAND
    assert seen["dry_run"] is True


class _AbsentS3:
    """Everything absent — the rehearsal's actual S3 state. Enough surface for
    the tile reads that sit between the two seams under test."""

    def get_object(self, **_kw):
        raise ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Not Found"}}, "GetObject",
        )

    def head_object(self, **_kw):
        raise ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject",
        )

    def list_objects_v2(self, **_kw):
        return {"KeyCount": 0, "Contents": []}

    def get_paginator(self, _name):
        return self

    def paginate(self, **_kw):
        return iter(())


class _Stop(Exception):
    """Stops the build at the seam under test — everything after it needs a
    real S3 and is covered by the card-build tests."""


class _StubReport:
    def __init__(self, run_scope=None):
        self.run_scope = run_scope
        self.run_scope_source = RUN_SCOPE_SOURCE_PAYLOAD
        self.read: list[str] = []
        self.missing: list[str] = []
        self.signal_quality_source = None

    def as_dict(self) -> dict:
        return {"artifacts_read": [], "artifacts_missing": []}
