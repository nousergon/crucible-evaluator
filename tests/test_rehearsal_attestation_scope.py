"""A declared rehearsal does not page, and does not gain a guarantee either.

``alpha-engine-config-I7392``, second half.

WHAT HAPPENED
-------------
A production ERROR page fired at 2026-08-29T01:44Z:

    report card attestation UNKNOWN for 2026-08-28: Correctness guarantee
    WITHHELD: backtester=UNKNOWN, evaluator_stage=UNKNOWN, contamination=UNKNOWN

each half naming an absent artifact under
``s3://alpha-engine-research/backtest/2026-08-28/`` and asserting "the producer
never ran this cycle." **No producer failed.** It came from inside a deliberate
Friday-evening shell (dry) run — execution ``offcycle-shell-20260829-004717``,
input ``{"shell_run": true, "pipeline_role": "shell-run", "skip_parity": true}``,
SUCCEEDED — in which ``Backtester`` dispatched ``spot_backtester.sh
--preflight-only`` (which writes no ``attestation.json`` BY DESIGN) and
``ReportCard`` was invoked with ``dry_run: true``. A rehearsal was grading
itself against artifacts it GUARANTEES do not exist, and paging every Friday
for it.

``I7392`` had already threaded ``dry_run`` to ``assert_input_freshness`` so the
card stopped hard-failing on the shell run. It stopped there:
``build_run_attestation`` had no ``dry_run`` awareness at all, and
``attestation.py``'s ``logger.error`` ran unconditionally on any non-PASS.

WHAT THESE TESTS PIN, and the second one is the one that matters
----------------------------------------------------------------
1. A rehearsal whose producers wrote nothing resolves those halves to
   ``NOT_IN_SCOPE`` and does not page.
2. **Both polarities keep rendering.** A genuinely dead producer on a REAL
   Saturday still pages; a real FAILURE reached on a rehearsal still pages; a
   half that is UNKNOWN for any reason OTHER than the object being absent still
   pages, on a rehearsal too.
3. ``NOT_IN_SCOPE`` is never a pass. No surface gains a guarantee it did not
   earn — the flag buys honesty about the DENOMINATOR, never about the numbers.
"""
from __future__ import annotations

import logging

import pytest
from botocore.exceptions import ClientError

import grading.attestation as att
from grading.attestation import (
    FAIL,
    NOT_IN_SCOPE,
    PASS,
    PARTIAL,
    UNKNOWN,
    build_run_attestation,
    verdict_is_pass,
)

RUN_DATE = "2026-08-28"
BUCKET = "alpha-engine-research"


class _S3:
    """Absent by default — the rehearsal's actual S3 state. The live prefix
    ``backtest/2026-08-28/`` held only ``.phases/preflight.json`` and
    ``.phases/runtime_smoke.json``."""

    def __init__(self, objects: dict | None = None):
        self._objects = objects or {}

    def get_object(self, Bucket=None, Key=None):  # noqa: N803 — boto3 casing
        if Key not in self._objects:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "Not Found"}}, "GetObject",
            )
        body = self._objects[Key]
        return {"Body": _Body(body), "LastModified": None}


class _Body:
    def __init__(self, raw: bytes):
        self._raw = raw

    def read(self) -> bytes:
        return self._raw


@pytest.fixture(autouse=True)
def _pin_the_in_process_half(monkeypatch):
    """The ``evaluator`` half is the in-process known-answer battery. It runs
    identically on a rehearsal and is NEVER out of scope, so it is pinned to
    PASS here and asserted separately in
    ``test_the_in_process_battery_is_never_out_of_scope``."""
    monkeypatch.setattr(att, "run_evaluator_attestation",
                        lambda: {"verdict": PASS, "reason": "battery PASS", "as_of": None})


def _build(dry_run: bool, objects: dict | None = None, run_scope=None) -> dict:
    return build_run_attestation(
        BUCKET, RUN_DATE, s3_client=_S3(objects), run_scope=run_scope,
        dry_run=dry_run,
    )


def _att(verdict: str = PASS) -> bytes:
    import json
    return json.dumps({"run_date": RUN_DATE, "verdict": verdict,
                       "n_checks": 3, "n_failed": 0, "n_errored": 0}).encode()


# ════════════════════════════════════════════════════════════════════════════
# 1. The rehearsal
# ════════════════════════════════════════════════════════════════════════════

def test_a_rehearsal_marks_the_preflight_only_halves_not_in_scope():
    """The exact 2026-08-29T01:44Z page, reproduced and then not fired."""
    block = _build(dry_run=True)
    assert block["backtester"]["verdict"] == NOT_IN_SCOPE
    assert block["evaluator_stage"]["verdict"] == NOT_IN_SCOPE
    assert block["contamination_verdict"] == NOT_IN_SCOPE
    assert block["evaluator"]["verdict"] == PASS
    assert block["verdict"] == PASS
    assert block["dry_run"] is True


def test_a_rehearsal_does_not_page(caplog):
    with caplog.at_level(logging.DEBUG, logger="grading.attestation"):
        block = _build(dry_run=True)
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors == [], [r.getMessage() for r in errors]
    assert block["verdict"] == PASS


def test_the_rehearsal_still_says_what_it_did_not_measure(caplog):
    """sf-pipeline-policy.md §2.3a rule 4, second obligation: BOTH sides.

    Not paging is not the same as going quiet. A PASS with three halves never
    measured is precisely the state that must not slip past a reader — so it
    renders at WARNING, names every out-of-scope half, and carries the
    withheld set beside the ran-regardless set."""
    with caplog.at_level(logging.DEBUG, logger="grading.attestation"):
        block = _build(dry_run=True)
    warnings = [r.getMessage() for r in caplog.records
                if r.levelno == logging.WARNING and "report card attestation" in r.getMessage()]
    assert len(warnings) == 1
    for half in ("BACKTESTER", "EVALUATOR_STAGE", "CONTAMINATION"):
        assert half + " NOT MEASURED" in warnings[0]
    assert block["not_in_scope_halves"] == [
        "backtester", "evaluator_stage", "contamination",
    ]
    assert block["withheld_halves"] == []
    assert "NOT MEASURED" in block["reason"]
    assert "--preflight-only" in block["backtester"]["reason"]


def test_out_of_scope_is_never_a_pass():
    """The invariant the whole vocabulary rests on. ``NOT_IN_SCOPE`` buys
    honesty about the DENOMINATOR, never a guarantee about the numbers."""
    assert verdict_is_pass(NOT_IN_SCOPE) is False
    assert verdict_is_pass(UNKNOWN) is False
    assert verdict_is_pass(PARTIAL) is False
    assert verdict_is_pass(FAIL) is False
    assert verdict_is_pass(None) is False
    assert verdict_is_pass("ok") is False
    assert verdict_is_pass(PASS) is True

    block = _build(dry_run=True)
    for name in block["not_in_scope_halves"]:
        assert verdict_is_pass(block[name]["verdict"]) is False
        assert "NOT a pass" in block[name]["reason"]


def test_not_in_scope_is_outside_the_producer_vocabulary():
    """No producer may ever write it, and normalization can never yield it —
    it is assigned by this module alone, on a fact about DISPATCH."""
    assert NOT_IN_SCOPE not in att._VALID_VERDICTS
    assert att._normalize_verdict(NOT_IN_SCOPE) == UNKNOWN


# ════════════════════════════════════════════════════════════════════════════
# 2. Both polarities — the half that keeps this from being a mute button
# ════════════════════════════════════════════════════════════════════════════

def test_a_real_saturday_with_a_dead_producer_still_pages(caplog):
    """THE regression guard. Same absent artifacts, ``dry_run=False`` — a
    genuinely dead producer on a real Saturday is exactly what this ERROR
    exists for, and it must survive the rehearsal fix untouched."""
    with caplog.at_level(logging.DEBUG, logger="grading.attestation"):
        block = _build(dry_run=False)
    assert block["verdict"] == UNKNOWN
    assert block["backtester"]["verdict"] == UNKNOWN
    assert block["not_in_scope_halves"] == []
    assert "backtester=UNKNOWN" in block["reason"]
    errors = [r.getMessage() for r in caplog.records
              if r.levelno >= logging.ERROR and "report card attestation" in r.getMessage()]
    assert len(errors) == 1
    assert "UNKNOWN" in errors[0]


def test_a_rehearsal_still_pages_on_a_real_failure(caplog):
    """A dry run over a date whose artifact EXISTS and says FAIL keeps that
    FAIL. Re-classifying a half that read real evidence would let the rehearsal
    flag erase a genuine finding."""
    objects = {att.backtester_attestation_key(RUN_DATE): _att(FAIL)}
    with caplog.at_level(logging.DEBUG, logger="grading.attestation"):
        block = _build(dry_run=True, objects=objects)
    assert block["backtester"]["verdict"] == FAIL
    assert block["verdict"] == FAIL
    assert "backtester=FAIL" in block["reason"]
    assert any(r.levelno >= logging.ERROR and "report card attestation" in r.getMessage()
               for r in caplog.records)


def test_a_rehearsal_still_pages_when_the_half_is_unknown_for_any_other_reason(caplog):
    """Narrow on purpose: ONLY an absent object is re-classified.

    A body that is unparseable, not an object, or stamped with another cycle's
    run_date is UNKNOWN because the evidence is BAD, not because nobody was
    asked for it — and a rehearsal is exactly when a corrupt artifact should be
    found."""
    import json
    cases = {
        "unparseable": b"{not json",
        "not an object": b"[]",
        "another cycle": json.dumps({"run_date": "2026-08-21", "verdict": PASS}).encode(),
    }
    for label, body in cases.items():
        objects = {att.backtester_attestation_key(RUN_DATE): body}
        block = _build(dry_run=True, objects=objects)
        assert block["backtester"]["verdict"] == UNKNOWN, label
        assert block["verdict"] == UNKNOWN, label
        assert "backtester=UNKNOWN" in block["reason"], label


def test_the_in_process_battery_is_never_out_of_scope(monkeypatch, caplog):
    """A rehearsal that could not fail on the deployed image's own arithmetic
    would be a rehearsal of nothing. The ``evaluator`` half runs identically on
    a dry run and is absent from ``REHEARSAL_OUT_OF_SCOPE_HALVES``."""
    assert "evaluator" not in att.REHEARSAL_OUT_OF_SCOPE_HALVES
    monkeypatch.setattr(att, "run_evaluator_attestation",
                        lambda: {"verdict": FAIL, "reason": "battery FAILED", "as_of": None})
    with caplog.at_level(logging.DEBUG, logger="grading.attestation"):
        block = _build(dry_run=True)
    assert block["verdict"] == FAIL
    assert block["arithmetic_verdict"] == FAIL
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


def test_a_clean_real_run_is_unchanged(caplog):
    """No new noise on the path that was already right: all four halves
    attested, one INFO line, no warning and no error."""
    objects = {
        att.backtester_attestation_key(RUN_DATE): _att(PASS),
        att.evaluator_stage_attestation_key(RUN_DATE): _att(PASS),
        att.contamination_key(RUN_DATE): _att(PASS),
    }
    with caplog.at_level(logging.DEBUG, logger="grading.attestation"):
        block = _build(dry_run=False, objects=objects)
    assert block["verdict"] == PASS
    assert block["not_in_scope_halves"] == []
    assert "All four halves attested" in block["reason"]
    assert not [r for r in caplog.records
                if r.levelno >= logging.WARNING and "report card attestation" in r.getMessage()]


# ════════════════════════════════════════════════════════════════════════════
# 3. The two out-of-scope grounds compose
# ════════════════════════════════════════════════════════════════════════════

def test_the_run_scope_ground_still_works_on_a_real_run():
    """``skip_parity`` (config-I7309/I7620) is untouched by the dry-run ground:
    a REAL Saturday with the parity producer switched off still resolves
    contamination — and only contamination — to NOT_IN_SCOPE."""
    scope = {
        "schema": "run_scope-1.0.0",
        "run_date": RUN_DATE,
        "stages": {
            **{s: {"disposition": "ENABLED_COMPLETED"}
               for s in att.CONTAMINATION_PRODUCER_STAGES},
            "Parity": {"disposition": "DISABLED", "disabled_by": "skip_parity"},
        },
    }
    objects = {
        att.backtester_attestation_key(RUN_DATE): _att(PASS),
        att.evaluator_stage_attestation_key(RUN_DATE): _att(PASS),
    }
    block = _build(dry_run=False, objects=objects, run_scope=scope)
    assert block["contamination_verdict"] == NOT_IN_SCOPE
    assert block["contamination_in_scope"] is False
    assert block["contamination"]["out_of_scope_because"] == "run_scope"
    assert block["verdict"] == PASS
    assert "skip_parity" in block["reason"]


def test_the_two_grounds_are_recorded_distinctly():
    """A reader asking WHY a half was not measured gets a different answer for
    'switched off by an operator' than for 'this was a rehearsal', and the
    diagnosis and remedy differ."""
    block = _build(dry_run=True)
    assert block["contamination"]["out_of_scope_because"] == "dry_run"
    assert block["backtester"]["out_of_scope_because"] == "dry_run"
