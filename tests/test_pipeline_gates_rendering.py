"""The Report Card, the Director digest and the digest email STATE the
pre-spend gate verdict — in both polarities (``alpha-engine-config-I7282``,
``sf-pipeline-policy.md`` §2.3a rule 3).

The clause these cover is not "the data is available somewhere on the artifact".
It is that a human reading the surface can tell whether the correctness check
ran. So every assertion here is against rendered text or against the field a
renderer reads, never against the raw payload.
"""
from __future__ import annotations

import pytest

from director.emailer import _verdict_banner, _verdict_footer
from director.report_card_digest import summarize_report_card
from director.verdict import PIPELINE_GATES_KEY, read_pipeline_gates
from grading.pipeline_gates import MEASURED, UNKNOWN, read_gate_state


def _payload(**overrides) -> dict:
    p = {
        "schema_version": 1,
        "gate_degraded": False,
        "health_check_degraded": False,
        "parity_degraded": False,
        "research_predictor_degraded": False,
        "lib_pin_drift": {"status": "MEASURED", "has_drift": False},
        "pipeline_contract": {"status": "MEASURED", "has_violation": False},
    }
    p.update(overrides)
    return p


_UNMEASURED = _payload(
    gate_degraded=True,
    pipeline_contract={"status": "UNKNOWN", "reason": "fetch_failed"},
)


def _card(gate_state=None, attestation_verdict="PASS") -> dict:
    return {
        "status": "ok",
        "overall": {"letter": "B", "numeric": 71.0},
        "tiles_overall_status": "GREEN",
        "tiles": {},
        "attestation": {"verdict": attestation_verdict, "as_of": {},
                        "reason": "all halves agreed"},
        PIPELINE_GATES_KEY: read_gate_state(gate_state),
    }


# ---------------------------------------------------------------------------
# The Director's prompt digest — the text the LLM and any human reader sees
# ---------------------------------------------------------------------------


def test_digest_names_the_unmeasured_gate():
    text = summarize_report_card(_card(_UNMEASURED))
    assert "⚠ PIPELINE GATES" in text
    assert "PipelineContractCheck" in text
    assert "fetch_failed" in text
    assert "UNATTESTED" in text


def test_digest_states_the_clean_case_too():
    text = summarize_report_card(_card(_payload()))
    assert "PIPELINE GATES: VERIFIED" in text
    assert "⚠ PIPELINE GATES" not in text


def test_digest_on_a_card_with_no_block_says_so_rather_than_nothing():
    card = _card(_payload())
    card.pop(PIPELINE_GATES_KEY)
    text = summarize_report_card(card)
    assert "⚠ PIPELINE GATES: UNKNOWN" in text
    assert "carries no pipeline_gates block" in text


# ---------------------------------------------------------------------------
# The digest EMAIL — the surface Brian actually opens
# ---------------------------------------------------------------------------


def test_amber_banner_fires_when_only_the_gates_are_unmeasured():
    vb = {"verdict": "PASS", "as_of": {},
          PIPELINE_GATES_KEY: read_gate_state(_UNMEASURED)}
    prefix, plain, html = _verdict_banner(vb)
    assert prefix == "[GATES UNVERIFIED] "
    # calibrated: distinct from the attestation banner in prefix, wording, colour
    assert "[UNVERIFIED] " != prefix
    assert "#b58900" in html and "#b00" not in html
    assert "nothing was withheld" in plain
    assert "pipeline_contract" in plain


def test_no_banner_on_a_fully_clean_run():
    vb = {"verdict": "PASS", "as_of": {},
          PIPELINE_GATES_KEY: read_gate_state(_payload())}
    assert _verdict_banner(vb) is None


def test_attestation_banner_still_wins_and_carries_the_gate_line():
    vb = {"verdict": "UNKNOWN", "as_of": {}, "reason": "backtester half absent",
          PIPELINE_GATES_KEY: read_gate_state(_UNMEASURED)}
    prefix, plain, html = _verdict_banner(vb)
    assert prefix == "[UNVERIFIED] "
    assert "#b00" in html
    assert "Pipeline gates —" in plain


@pytest.mark.parametrize("payload,expected", [
    (_payload(), f"Pipeline gates: {MEASURED}"),
    (_UNMEASURED, f"Pipeline gates: {UNKNOWN} (unmeasured: pipeline_contract)"),
])
def test_footer_states_the_gate_verdict_in_both_polarities(payload, expected):
    vb = {"verdict": "PASS", "as_of": {},
          PIPELINE_GATES_KEY: read_gate_state(payload)}
    assert expected in _verdict_footer(vb)


def test_footer_with_no_verdict_block_says_not_read():
    assert "Pipeline gates: NOT READ" in _verdict_footer({})


# ---------------------------------------------------------------------------
# The Director's source precedence
# ---------------------------------------------------------------------------


def test_sf_payload_wins_over_the_card():
    block = read_pipeline_gates(_payload(), _card(_UNMEASURED))
    assert block["verdict"] == MEASURED
    assert block.get("source") != "report_card"


def test_card_is_the_fallback_when_the_sf_sends_nothing():
    block = read_pipeline_gates(None, _card(_UNMEASURED))
    assert block["verdict"] == UNKNOWN
    assert block["source"] == "report_card"
    assert block["unmeasured"] == ["pipeline_contract"]


def test_neither_source_resolves_to_unknown_never_to_a_pass():
    block = read_pipeline_gates(None, None)
    assert block["verdict"] == UNKNOWN
    assert block["present"] is False


def test_pipeline_gates_do_not_withhold_director_actions():
    """Deliberate, and documented on ``read_pipeline_gates``: the pre-spend gates
    answer a different question from the attestation, and gating on them would —
    today — stop Director issue filing permanently, because PipelineContractGate
    has never measured anything (``alpha-engine-config-I7281``)."""
    from director.verdict import actions_withheld

    vb = {"verdict": "PASS", "as_of": {},
          PIPELINE_GATES_KEY: read_gate_state(_UNMEASURED)}
    assert actions_withheld(vb) is False
