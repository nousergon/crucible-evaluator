"""alpha-engine-config-I8811 — a run-scope claim never overwrites read evidence.

WHAT THIS PINS
--------------
``NOT_IN_SCOPE`` is granted on two grounds (see ``grading/attestation.py``'s
``NOT_IN_SCOPE`` docstring): a declared dry run, and a run-scope block saying the
producer took its skip branch. The dry-run grant has always required the object
to be genuinely ABSENT. The run-scope grant required nothing, and overwrote
whatever verdict had already been read.

That asymmetry is a **fail-open**, and it is reachable with artifacts that exist
on S3 today. ``backtest/{date}/run_scope.json`` is written by whichever execution
ran the ``RunScope`` stage LAST for that ``run_date`` — not by the execution that
produced the numbers being attested. Measured 2026-08-31: both live scope
artifacts were authored by skip-flagged recovery reruns
(``watch-rerun-2026-08-22-3`` and ``watch-rerun-2026-08-28-13``, the latter
written 2026-08-30T18:47Z, a day and a half after the scheduled run wrote
``backtest/2026-08-28/attestation.json``), and both report ``Backtester:
DISABLED`` for a cycle whose backtester artifacts demonstrably exist.

Compose one of those with a scheduled run whose parity pass reported ``FAIL``
and the pre-fix path published a combined ``PASS`` over a week the system had
itself found contaminated.

The rule these tests pin: **a claim about DISPATCH never overwrites EVIDENCE**,
and the disagreement is emitted rather than resolved silently in either
direction.
"""

from __future__ import annotations

import io
import json
import logging

import pytest

from grading import attestation as att

_KEY = "backtest/2026-08-15/pit_parity.json"
_RUN_DATE = "2026-08-15"


class _S3:
    def __init__(self, objects: dict):
        self.objects = objects

    def get_object(self, Bucket=None, Key=None):  # noqa: N803 — boto3 casing
        val = self.objects.get(Key)
        if val is None:
            from botocore.exceptions import ClientError

            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject",
            )
        if isinstance(val, Exception):
            raise val
        return {"Body": io.BytesIO(val), "LastModified": _FakeDT()}


class _FakeDT:
    def strftime(self, fmt):  # noqa: D102
        return "2026-08-15T09:00:00Z"


def _parity(verdict: str, run_date: str = _RUN_DATE) -> bytes:
    return json.dumps({
        "schema": "pit_parity-1.0.0",
        "run_date": run_date,
        "status": "ok",
        "verdict": verdict,
        "verdict_reason": f"{verdict}: from the producer.",
        "coverage": {"coverage_fraction": 1.0, "budget_stopped": False,
                     "complete": True, "measured": True},
    }).encode()


def _clobbering_scope() -> dict:
    """The shape a skip-flagged recovery rerun actually writes.

    Modelled on the live ``backtest/2026-08-28/run_scope.json``: the umbrella
    Parity gate reports DISABLED because the rerun's own input carried
    ``skip_parity: true``, while the artifacts being attested were written by a
    different, earlier execution.
    """
    return {
        "schema": "run_scope-1.0.0",
        "run_date": _RUN_DATE,
        "execution_arn": (
            "arn:aws:states:us-east-1:711398986525:execution:"
            "ne-weekly-freshness-pipeline:watch-rerun-2026-08-15-13"
        ),
        "stages": {
            "Parity": {"disposition": "DISABLED", "disabled_by": "skip_parity",
                       "flag": "skip_parity", "gate": "CheckSkipParity",
                       "reason": "CheckSkipParity was entered and took its skip branch."},
            "LibPinDriftCheck": {"disposition": "ENABLED_COMPLETED"},
        },
    }


def _honest_scope() -> dict:
    """A scheduled run that genuinely skipped parity and wrote nothing for it."""
    return _clobbering_scope()


def _build(monkeypatch, objects, run_scope, arithmetic=att.PASS):
    monkeypatch.setattr(att, "run_evaluator_attestation",
                        lambda: {"verdict": arithmetic})
    monkeypatch.setattr(att, "read_backtester_attestation",
                        lambda *a, **k: {"verdict": arithmetic, "as_of": None})
    monkeypatch.setattr(att, "read_evaluator_stage_attestation",
                        lambda *a, **k: {"verdict": arithmetic, "as_of": None})
    return att.build_run_attestation(
        "b", _RUN_DATE, s3_client=_S3(objects), run_scope=run_scope,
    )


# ════════════════════════════════════════════════════════════════════════════
# THE FAIL-OPEN — the test that fails without the fix
# ════════════════════════════════════════════════════════════════════════════

def test_a_clobbering_scope_cannot_erase_a_measured_contamination_FAIL(monkeypatch):
    """THE regression. A rerun's scope says parity never ran; the scheduled
    run's parity artifact says FAIL. Pre-fix the FAIL was overwritten with
    NOT_IN_SCOPE, dropped from the worst-of, and the combined verdict published
    as PASS."""
    block = _build(monkeypatch, {_KEY: _parity(att.FAIL)}, _clobbering_scope())

    assert block["contamination_verdict"] == att.FAIL, (
        "a measured contamination FAILURE was overwritten by a claim that the "
        "producer never ran — the exact fail-open I8811 describes"
    )
    assert block["verdict"] == att.FAIL, (
        "the combined verdict must carry the FAIL; excluding it from the "
        "worst-of is how a contaminated week publishes as PASS"
    )
    assert not att.verdict_is_pass(block["verdict"])


def test_a_clobbering_scope_cannot_erase_a_measured_PARTIAL(monkeypatch):
    """PARTIAL is not a pass either, and it is likewise evidence."""
    block = _build(monkeypatch, {_KEY: _parity(att.PARTIAL)}, _clobbering_scope())
    assert block["contamination_verdict"] == att.PARTIAL
    assert block["verdict"] == att.PARTIAL


def test_a_clobbering_scope_cannot_erase_a_measured_PASS(monkeypatch):
    """Symmetry matters: the rule is 'evidence wins', not 'the worse value
    wins'. A real PASS that was actually measured is kept as a PASS, not
    downgraded to the weaker NOT_IN_SCOPE claim."""
    block = _build(monkeypatch, {_KEY: _parity(att.PASS)}, _clobbering_scope())
    assert block["contamination_verdict"] == att.PASS
    assert block["verdict"] == att.PASS


# ════════════════════════════════════════════════════════════════════════════
# The conflict is EMITTED, not swallowed (principles.md §2.7)
# ════════════════════════════════════════════════════════════════════════════

def test_the_disagreement_is_recorded_on_the_block(monkeypatch):
    block = _build(monkeypatch, {_KEY: _parity(att.FAIL)}, _clobbering_scope())
    conflict = block["contamination"]["scope_conflict"]
    assert conflict["scope_says"] == "not dispatched"
    assert conflict["evidence_verdict"] == att.FAIL
    assert conflict["evidence_absent"] is False
    assert "Parity" in conflict["disabled_stages"]


def test_the_disagreement_is_logged_loud(monkeypatch, caplog):
    with caplog.at_level(logging.ERROR):
        _build(monkeypatch, {_KEY: _parity(att.FAIL)}, _clobbering_scope())
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("I8811" in m for m in msgs), (
        "a silently-resolved conflict is indistinguishable from no conflict"
    )
    assert any("different execution" in m for m in msgs)


def test_no_conflict_is_recorded_when_scope_and_evidence_agree(monkeypatch):
    """Both polarities on the same surface. A field that only ever appears when
    something is wrong cannot be told apart from a producer that stopped
    emitting it."""
    block = _build(monkeypatch, {}, _clobbering_scope())
    assert "scope_conflict" not in block["contamination"]


# ════════════════════════════════════════════════════════════════════════════
# The legitimate grant still works — this fix must not re-break I7620
# ════════════════════════════════════════════════════════════════════════════

def test_an_absent_artifact_under_an_honest_skip_is_still_NOT_IN_SCOPE(monkeypatch):
    """The case NOT_IN_SCOPE exists for (config-I7309 / I7620): parity was
    switched off on purpose and wrote nothing. Excluded from the worst-of, so
    the combined verdict is not dragged to UNKNOWN every week."""
    block = _build(monkeypatch, {}, _honest_scope())
    assert block["contamination_verdict"] == att.NOT_IN_SCOPE
    assert block["contamination"]["out_of_scope_because"] == "run_scope"
    assert block["verdict"] == att.PASS
    assert att.verdict_is_pass(block["verdict"])


def test_NOT_IN_SCOPE_is_still_not_a_pass_for_the_half(monkeypatch):
    block = _build(monkeypatch, {}, _honest_scope())
    assert not att.verdict_is_pass(block["contamination_verdict"])


def test_an_unreadable_body_under_a_skip_stays_UNKNOWN_and_still_pages(monkeypatch):
    """Narrowness clause 2, the half that is easy to lose. The object EXISTS
    and could not be parsed — that is a broken producer, not a skipped one, and
    a scope claim must not launder it into NOT_IN_SCOPE."""
    block = _build(monkeypatch, {_KEY: b"{not json"}, _clobbering_scope())
    assert block["contamination_verdict"] == att.UNKNOWN
    assert block["contamination"].get("out_of_scope_because") is None
    assert block["verdict"] == att.UNKNOWN


def test_a_body_stamped_with_another_cycle_stays_UNKNOWN(monkeypatch):
    """A verdict from another run_date is never inherited — and a scope claim
    does not convert that refusal into an excuse."""
    block = _build(
        monkeypatch, {_KEY: _parity(att.PASS, run_date="2026-08-08")},
        _clobbering_scope(),
    )
    assert block["contamination_verdict"] == att.UNKNOWN
    assert block["contamination"].get("out_of_scope_because") is None


def test_an_absent_scope_leaves_the_half_in_scope_and_UNKNOWN(monkeypatch):
    """Unchanged behaviour, restated here because it is the property the whole
    module protects: an unmeasured denominator never excuses a missing half."""
    block = _build(monkeypatch, {}, None)
    assert block["contamination_verdict"] == att.UNKNOWN
    assert block["verdict"] == att.UNKNOWN
