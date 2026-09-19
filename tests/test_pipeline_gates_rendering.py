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


# ---------------------------------------------------------------------------
# A fail-open OUTSIDE the pre-spend gates must not be reported as one INSIDE
# them (alpha-engine-config-I10534 / -I10062).
#
# Measured origin: on run_date 2026-09-04 and 2026-09-11 the weekly SF fired
# exactly one degradation route — ChallengerShadow raising
# ChallengerShadowGapError into MarkChallengerShadowDegraded — while BOTH
# pre-spend gates reported MEASURED. All three surfaces said "the run's
# pre-spend protection was incomplete", and the email named its (empty)
# unmeasured-gate list as the literal placeholder "unnamed". The Director read
# that sentence and filed it P0 twice.
# ---------------------------------------------------------------------------

_RP_DEGRADED = _payload(research_predictor_degraded=True)
_GATE_DEGRADED = _payload(gate_degraded=True)


def test_a_non_pre_spend_fail_open_still_withholds_the_attestation():
    """The verdict is NOT widened. This is the load-bearing assertion: the whole
    fix is a truer REASON, never a broader success condition."""
    block = read_gate_state(_RP_DEGRADED)
    assert block["verdict"] == UNKNOWN
    assert block["degraded_families"] == ["research_predictor_degraded"]


def test_a_non_pre_spend_fail_open_does_not_claim_the_spend_was_unprotected():
    block = read_gate_state(_RP_DEGRADED)
    assert "pre-spend protection was incomplete" not in block["statement"]
    assert "none of them in the pre-spend gate family" in block["statement"]
    assert "spend WAS gated" in block["statement"]
    # and it still names what DID fail open, so the finding is actionable
    assert "ResearchPredictorParallel" in block["statement"]


def test_a_pre_spend_fail_open_still_makes_the_strong_claim():
    block = read_gate_state(_GATE_DEGRADED)
    assert block["verdict"] == UNKNOWN
    assert "pre-spend protection was incomplete" in block["statement"]
    assert block["degraded_pre_spend"] == ["gate_degraded"]


@pytest.mark.parametrize("payload,pre,other", [
    (_payload(), [], []),
    (_RP_DEGRADED, [], ["research_predictor_degraded"]),
    (_GATE_DEGRADED, ["gate_degraded"], []),
    (_payload(gate_degraded=True, parity_degraded=True),
     ["gate_degraded"], ["parity_degraded"]),
])
def test_the_split_is_carried_as_data_in_both_polarities(payload, pre, other):
    block = read_gate_state(payload)
    assert block["degraded_pre_spend"] == pre
    assert block["degraded_other"] == other


def test_pre_spend_degraded_falls_back_for_a_card_written_before_the_split():
    from grading.pipeline_gates import pre_spend_degraded

    legacy = {"degraded_families": ["gate_degraded"]}
    assert pre_spend_degraded(legacy) is True
    assert pre_spend_degraded({"degraded_families": ["parity_degraded"]}) is False
    assert pre_spend_degraded({}) is False
    assert pre_spend_degraded(None) is False


def test_the_email_banner_never_renders_the_unnamed_placeholder():
    vb = {"verdict": "PASS", "as_of": {},
          PIPELINE_GATES_KEY: read_gate_state(_RP_DEGRADED)}
    prefix, plain, html = _verdict_banner(vb)
    assert prefix == "[GATES UNVERIFIED] "
    assert "unnamed" not in plain
    assert "gates did not all run this cycle" not in plain
    assert "OUTSIDE the pre-spend gates" in plain
    assert "spend WAS gated" in plain
    assert "ResearchPredictorParallel" in plain


def test_the_email_banner_keeps_the_strong_wording_for_a_pre_spend_fail_open():
    vb = {"verdict": "PASS", "as_of": {},
          PIPELINE_GATES_KEY: read_gate_state(_GATE_DEGRADED)}
    _, plain, _ = _verdict_banner(vb)
    assert "pre-spend gate family fail-opened" in plain
    assert "BEFORE it spent" in plain


def test_the_email_banner_on_a_card_the_sf_never_reported():
    """No gate_state at all drives BOTH gates UNKNOWN, so this lands on the
    unmeasured branch and names them — the one thing it must never do is render
    an empty list as a placeholder."""
    vb = {"verdict": "PASS", "as_of": {},
          PIPELINE_GATES_KEY: read_gate_state(None)}
    _, plain, _ = _verdict_banner(vb)
    assert "lib_pin_drift" in plain and "pipeline_contract" in plain
    assert "unnamed" not in plain


def test_the_email_banner_default_when_nothing_explains_the_withholding():
    """A hand-built or future block that is UNKNOWN with no unmeasured gate and
    no fired family still gets a sentence rather than an empty parenthetical."""
    vb = {"verdict": "PASS", "as_of": {},
          PIPELINE_GATES_KEY: {"verdict": UNKNOWN, "unmeasured": [],
                               "degraded_families": [], "statement": ""}}
    _, plain, _ = _verdict_banner(vb)
    assert "did not report this cycle's gate state" in plain
    assert "unnamed" not in plain


def test_the_digest_carries_the_corrected_sentence_too():
    text = summarize_report_card(_card(_RP_DEGRADED))
    assert "⚠ PIPELINE GATES" in text
    assert "pre-spend protection was incomplete" not in text
    assert "spend WAS gated" in text


# ---------------------------------------------------------------------------
# alpha-engine-config-I11073 — the route reaches the surface Brian opens
#
# The whole failure mode this closes is a reader acting on the banner's first
# sentence. "an internal ResearchPredictorParallel fail-open" names one of ten
# possibilities; MarkChallengerShadowDegraded names the one that happened.
# ---------------------------------------------------------------------------

_RP_ROUTED = _payload(
    research_predictor_degraded=True,
    research_predictor_degraded_routes=["MarkChallengerShadowDegraded"],
)
_RP_TWO_ROUTES = _payload(
    research_predictor_degraded=True,
    research_predictor_degraded_routes=["MarkChallengerShadowDegraded",
                                        "MarkModelZooDegraded"],
)


def test_the_banner_headline_names_the_single_route_that_fired():
    vb = {"verdict": "PASS", "as_of": {},
          PIPELINE_GATES_KEY: read_gate_state(_RP_ROUTED)}
    prefix, plain, html = _verdict_banner(vb)
    assert prefix == "[GATES UNVERIFIED] "
    headline = plain.splitlines()[0]
    assert "MarkChallengerShadowDegraded" in headline, (
        "the route is in the body but not the headline — the failure mode "
        "being designed against (alpha-engine-config-I10062, -I10534) is a "
        "reader who acts on the first sentence alone"
    )
    assert "MarkChallengerShadowDegraded" in html


def test_two_routes_are_both_named_in_the_banner_body():
    vb = {"verdict": "PASS", "as_of": {},
          PIPELINE_GATES_KEY: read_gate_state(_RP_TWO_ROUTES)}
    _, plain, html = _verdict_banner(vb)
    for route in ("MarkChallengerShadowDegraded", "MarkModelZooDegraded"):
        assert route in plain and route in html
    # The headline counts rather than listing: two SF state names do not fit a
    # subject-line-length sentence, and a truncated one names the wrong route.
    assert "2 fail-open routes fired" in plain.splitlines()[0]


def test_an_unnamed_route_says_so_on_the_surface_rather_than_reading_clean():
    """The pre-1.1.0 producer. The banner must still fire, and must say the
    route is unknown — silence here is what made the old boolean unreadable."""
    vb = {"verdict": "PASS", "as_of": {},
          PIPELINE_GATES_KEY: read_gate_state(_RP_DEGRADED)}
    prefix, plain, _ = _verdict_banner(vb)
    assert prefix == "[GATES UNVERIFIED] "
    assert "did not name the route" in plain
    assert "research_predictor_degraded" in plain


def test_the_digest_carries_the_route_too():
    """The Director's prompt digest is what the LLM reads; a route named only in
    the email leaves the automated reader exactly as blind as before."""
    card = _card(_RP_ROUTED)
    text = summarize_report_card(card)
    assert "MarkChallengerShadowDegraded" in text
