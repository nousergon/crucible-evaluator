"""The contamination half is scope-aware — config-I7620 follow-up.

`skip_parity: true` has stood on the live weekly EventBridge target since
2026-08-13 by a recorded ruling (config-I7309). Before this, the attestation
read the resulting absent `pit_parity.json` as `UNKNOWN`, the combined verdict
went UNKNOWN with it, and the Director withheld `issue_filing` and
`loop_verification` under sf-pipeline-policy §2.3a — every week, indefinitely.

These tests pin the distinction that fixes it, and the two ways a lazy version
of that fix would be wrong: silence must not excuse a half, and out-of-scope
must never render as a pass.
"""
from __future__ import annotations

import grading.attestation as att
from grading.attestation import (
    CONTAMINATION_PRODUCER_STAGES,
    NOT_IN_SCOPE,
    PASS,
    UNKNOWN,
    build_run_attestation,
    contamination_scope,
    verdict_is_pass,
)


def _scope(stages: dict) -> dict:
    return {
        "schema": "run_scope-1.0.0",
        "run_date": "2026-08-22",
        "stages": stages,
    }


def _row(disposition: str, disabled_by: str | None = None) -> dict:
    row = {"disposition": disposition}
    if disabled_by:
        row["disabled_by"] = disabled_by
    return row


def _all_dispatched() -> dict:
    return {s: _row("ENABLED_COMPLETED") for s in CONTAMINATION_PRODUCER_STAGES}


# --------------------------------------------------------------------------
# contamination_scope
# --------------------------------------------------------------------------

def test_parity_disabled_puts_contamination_out_of_scope():
    block = contamination_scope(_scope({
        **_all_dispatched(),
        "Parity": _row("DISABLED", "skip_parity"),
    }))
    assert block["in_scope"] is False
    assert block["disabled_stages"] == ["Parity"]
    assert block["disabled_by"] == ["skip_parity"]
    assert "skip_parity" in block["reason"]


def test_any_one_producer_stage_disabled_is_enough():
    """pit_parity.json needs all of them; one skipped means no comparison."""
    for stage in CONTAMINATION_PRODUCER_STAGES:
        block = contamination_scope(_scope({
            **_all_dispatched(),
            stage: _row("DISABLED", "skip_x"),
        }))
        assert block["in_scope"] is False, stage


def test_dispatched_producer_stays_in_scope():
    assert contamination_scope(_scope(_all_dispatched()))["in_scope"] is True


def test_a_failed_producer_is_in_scope_not_out_of_it():
    """A stage that ran and died is graded as a failure — never excused.

    This is the whole reason the run-scope vocabulary separates DISABLED from
    ENABLED_FAILED. Collapsing them would let a crashing producer switch off
    its own attestation.
    """
    block = contamination_scope(_scope({
        **_all_dispatched(),
        "PitParityCompare": _row("ENABLED_FAILED"),
    }))
    assert block["in_scope"] is True


def test_absent_or_unmeasured_scope_leaves_the_half_in_scope():
    """Silence never excuses a half — otherwise a dead scope reader would
    quietly grant the guarantee to every run it can no longer see."""
    for degenerate in (None, {}, {"stages": {}}, {"degraded": True}, "nonsense", 7):
        assert contamination_scope(degenerate)["in_scope"] is True, degenerate


def test_an_already_normalized_block_is_accepted():
    from grading.run_scope import read_run_scope
    normalized = read_run_scope(_scope({
        **_all_dispatched(),
        "Parity": _row("DISABLED", "skip_parity"),
    }))
    assert contamination_scope(normalized)["in_scope"] is False


# --------------------------------------------------------------------------
# build_run_attestation
# --------------------------------------------------------------------------

def _stub_halves(monkeypatch, contamination_verdict=UNKNOWN):
    ok = {"verdict": PASS, "as_of": "2026-08-22T00:00:00Z", "reason": "ok"}
    monkeypatch.setattr(att, "run_evaluator_attestation", lambda: dict(ok))
    monkeypatch.setattr(att, "read_backtester_attestation", lambda *a, **k: dict(ok))
    monkeypatch.setattr(att, "read_evaluator_stage_attestation", lambda *a, **k: dict(ok))
    monkeypatch.setattr(att, "read_contamination_verdict", lambda *a, **k: {
        "verdict": contamination_verdict,
        "as_of": None,
        "reason": "the producer never ran this cycle.",
    })


def test_out_of_scope_contamination_no_longer_drags_the_combined_verdict(monkeypatch):
    """The regression this whole change exists for."""
    _stub_halves(monkeypatch, contamination_verdict=UNKNOWN)
    block = build_run_attestation("b", "2026-08-22", run_scope=_scope({
        **_all_dispatched(),
        "Parity": _row("DISABLED", "skip_parity"),
    }))
    assert block["verdict"] == PASS
    assert block["arithmetic_verdict"] == PASS
    assert block["contamination_in_scope"] is False
    assert block["contamination_verdict"] == NOT_IN_SCOPE
    assert "CONTAMINATION NOT MEASURED" in block["reason"]
    # ...and it does NOT claim the four-halves sentence.
    assert "found no material look-ahead contamination" not in block["reason"]


def test_out_of_scope_is_never_a_pass():
    assert verdict_is_pass(NOT_IN_SCOPE) is False


def test_not_in_scope_is_outside_the_producer_vocabulary():
    """No producer can ever write it; only this module assigns it."""
    assert NOT_IN_SCOPE not in att._VALID_VERDICTS
    assert att._normalize_verdict(NOT_IN_SCOPE) == UNKNOWN


def test_in_scope_unknown_still_withholds(monkeypatch):
    """Without the scope artifact, behaviour is exactly what it was."""
    _stub_halves(monkeypatch, contamination_verdict=UNKNOWN)
    block = build_run_attestation("b", "2026-08-22", run_scope=None)
    assert block["verdict"] == UNKNOWN
    assert block["contamination_verdict"] == UNKNOWN
    assert block["contamination_in_scope"] is True
    assert "WITHHELD" in block["reason"]


def test_arithmetic_failure_still_fails_when_contamination_is_out_of_scope(monkeypatch):
    """Excluding one half must not excuse the other three."""
    _stub_halves(monkeypatch)
    monkeypatch.setattr(att, "read_backtester_attestation", lambda *a, **k: {
        "verdict": att.FAIL, "as_of": None, "reason": "arithmetic wrong",
    })
    block = build_run_attestation("b", "2026-08-22", run_scope=_scope({
        **_all_dispatched(),
        "Parity": _row("DISABLED", "skip_parity"),
    }))
    assert block["verdict"] == att.FAIL


def test_all_four_halves_pass_in_scope_keeps_the_original_sentence(monkeypatch):
    _stub_halves(monkeypatch, contamination_verdict=PASS)
    block = build_run_attestation("b", "2026-08-22", run_scope=_scope(_all_dispatched()))
    assert block["verdict"] == PASS
    assert block["contamination_in_scope"] is True
    assert "found no material look-ahead contamination" in block["reason"]


# --------------------------------------------------------------------------
# The Director half — the consumer whose authority this restores
# --------------------------------------------------------------------------

def test_director_acts_again_when_contamination_is_merely_out_of_scope(monkeypatch):
    from director.verdict import actions_withheld, read_card_verdict

    _stub_halves(monkeypatch, contamination_verdict=UNKNOWN)
    card = {"attestation": build_run_attestation("b", "2026-08-22", run_scope=_scope({
        **_all_dispatched(),
        "Parity": _row("DISABLED", "skip_parity"),
    }))}
    assert read_card_verdict(card)["verdict"] == PASS
    assert actions_withheld(read_card_verdict(card)) is False


def test_director_digest_states_that_contamination_was_not_measured(monkeypatch):
    from director.report_card_digest import summarize_report_card

    _stub_halves(monkeypatch, contamination_verdict=UNKNOWN)
    card = {
        "run_date": "2026-08-22",
        "attestation": build_run_attestation("b", "2026-08-22", run_scope=_scope({
            **_all_dispatched(),
            "Parity": _row("DISABLED", "skip_parity"),
        })),
    }
    digest = summarize_report_card(card)
    assert "CONTAMINATION NOT MEASURED" in digest
    assert "skip_parity" in digest
    # The instruction that stops the plan re-proposing a fix for a stage nobody
    # started — the 2026-08-14 plan's P0.
    assert "do not propose fixing" in digest
