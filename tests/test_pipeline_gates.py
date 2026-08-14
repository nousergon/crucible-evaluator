"""The consumer half of the SF -> evaluator ``gate_state`` contract
(``alpha-engine-config-I7282``, ``sf-pipeline-policy.md`` §2.3a).

Three things are asserted here, and the second is the one that matters:

1. a conforming payload is validated against the versioned schema, in both
   polarities (the producer's copy of that schema lives in ``nousergon-data``
   and is pinned to the same sha256 there);
2. **no degenerate input ever resolves to MEASURED** — absent block, empty
   block, a probe missing ``status``, an unrecognised status string, a boolean
   family the SF did not send. Every one of them is indistinguishable from a
   clean run to any consumer that tests truthiness, and every one of them is
   what the world actually looked like before I7282;
3. the rendered statement is present in BOTH polarities and names the specific
   gate that did not run.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import jsonschema
import pytest

from grading.pipeline_gates import (
    GATE_LABELS,
    MEASURED,
    PIPELINE_GATES_KEY,
    SCHEMA,
    SCHEMA_VERSION,
    UNKNOWN,
    gates_unmeasured,
    read_gate_state,
)

_SCHEMA_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "grading" / "contracts" / "sf_gate_state.v1.schema.json"
)

#: The producer repo (``nousergon-data``) carries a byte-identical copy at
#: ``infrastructure/contracts/sf_gate_state.v1.schema.json`` and pins the SAME
#: digest in ``tests/test_sf_gate_state_wiring.py``. Neither repo's CI can read
#: the other, so this pin is what makes a one-sided edit fail loudly instead of
#: silently forking the contract. Changing the schema means changing this digest
#: in BOTH repos, in the same cross-repo change.
_SCHEMA_SHA256 = "5f4c4a7736238103aa64d9cf989eddfd87840612a6872b266ce1e5578c2439b6"


def _clean_payload(**overrides) -> dict:
    payload = {
        "schema_version": 1,
        "gate_degraded": False,
        "health_check_degraded": False,
        "parity_degraded": False,
        "research_predictor_degraded": False,
        "lib_pin_drift": {"status": "MEASURED", "has_drift": False},
        "pipeline_contract": {"status": "MEASURED", "has_violation": False,
                              "violations": [], "boundary_count": 12},
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# 1. Schema — the versioned contract at the cross-repo boundary
# ---------------------------------------------------------------------------


def test_schema_digest_is_pinned_against_the_producer_copy():
    digest = hashlib.sha256(_SCHEMA_PATH.read_bytes()).hexdigest()
    assert digest == _SCHEMA_SHA256, (
        "grading/contracts/sf_gate_state.v1.schema.json changed. The producer "
        "(nousergon-data/infrastructure/contracts/sf_gate_state.v1.schema.json) "
        "carries a byte-identical copy pinned to the same digest — update BOTH "
        "repos in one cross-repo change, or the contract has silently forked."
    )


def test_schema_declares_the_version_the_module_emits():
    schema = json.loads(_SCHEMA_PATH.read_text())
    assert schema["properties"]["schema_version"]["const"] == SCHEMA_VERSION
    assert schema["title"] == SCHEMA


@pytest.mark.parametrize("payload", [
    _clean_payload(),
    _clean_payload(
        gate_degraded=True,
        pipeline_contract={"status": "UNKNOWN", "boundary_count": None,
                           "reason": "fetch_failed"},
    ),
    _clean_payload(
        lib_pin_drift={"status": "UNKNOWN", "reason": "gate_did_not_run"},
        pipeline_contract={"status": "UNKNOWN", "reason": "gate_did_not_run"},
        gate_degraded=True, parity_degraded=True,
    ),
])
def test_conforming_payloads_validate(payload):
    jsonschema.validate(payload, json.loads(_SCHEMA_PATH.read_text()))


@pytest.mark.parametrize("payload", [
    # the pre-I7277 shape: an empty evidence list where a status should be
    _clean_payload(pipeline_contract={"violations": [], "boundary_count": None,
                                      "reason": "fetch_failed"}),
    # a status outside the producer's closed vocabulary
    _clean_payload(lib_pin_drift={"status": "OK"}),
    # a degradation family sent as something other than a boolean
    _clean_payload(gate_degraded="false"),
    # an extra top-level key — the contract is closed, so a producer that starts
    # sending a field nobody agreed to fails at the boundary, not silently
    _clean_payload(some_new_field=1),
])
def test_nonconforming_payloads_are_rejected(payload):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, json.loads(_SCHEMA_PATH.read_text()))


# ---------------------------------------------------------------------------
# 2. No degenerate input resolves to MEASURED
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("degenerate", [
    None,
    {},
    "MEASURED",
    [],
    0,
    # every field but one
    _clean_payload(pipeline_contract=None),
    # a probe present but carrying no status at all
    _clean_payload(lib_pin_drift={"has_drift": False}),
    # a status string outside the vocabulary — a vocabulary that accepts new
    # truthy strings is not a vocabulary
    _clean_payload(lib_pin_drift={"status": "ok"}),
    _clean_payload(pipeline_contract={"status": True}),
    # the SF did not send a degradation family: unreported is not false
    {k: v for k, v in _clean_payload().items() if k != "parity_degraded"},
], ids=lambda d: str(d)[:48])
def test_no_degenerate_input_reads_as_measured(degenerate):
    block = read_gate_state(degenerate)
    assert block["verdict"] == UNKNOWN
    assert gates_unmeasured(block) is True
    assert block["statement"].startswith("NOT VERIFIED")


def test_absent_block_records_the_specific_cause():
    block = read_gate_state(None)
    assert block["present"] is False
    assert "supplied no gate_state" in block["reason"]
    # the two gates are still enumerated — a block that lists nothing cannot be
    # told apart from a block that found nothing wrong
    assert set(block["gates"]) == set(GATE_LABELS)
    assert block["unmeasured"] == list(GATE_LABELS)


def test_unknown_probe_is_named_in_the_statement():
    block = read_gate_state(_clean_payload(
        gate_degraded=True,
        pipeline_contract={"status": "UNKNOWN", "reason": "fetch_failed"},
    ))
    assert block["verdict"] == UNKNOWN
    assert block["unmeasured"] == ["pipeline_contract"]
    assert "lib_pin_drift" not in block["unmeasured"]
    assert "PipelineContractCheck" in block["statement"]
    assert "fetch_failed" in block["statement"]
    # calibration: it says what is NOT wrong, so a usable card is not discarded
    assert "UNATTESTED, not as wrong" in block["statement"]
    assert "FAILED" not in block["statement"]
    # the fail-open family that fired is named too
    assert "pre-spend gates" in block["statement"]


# ---------------------------------------------------------------------------
# 3. Both polarities
# ---------------------------------------------------------------------------


def test_clean_run_renders_the_positive_statement():
    block = read_gate_state(_clean_payload())
    assert block["verdict"] == MEASURED
    assert gates_unmeasured(block) is False
    assert block["unmeasured"] == []
    assert block["degraded_families"] == []
    assert block["statement"].startswith("VERIFIED")
    assert "2 pre-spend correctness gates" in block["statement"]


def test_every_gate_is_present_in_both_polarities():
    for payload in (_clean_payload(),
                    _clean_payload(lib_pin_drift={"status": "UNKNOWN"})):
        block = read_gate_state(payload)
        assert set(block["gates"]) == set(GATE_LABELS)
        for entry in block["gates"].values():
            assert entry["status"] in (MEASURED, UNKNOWN)


def test_key_name_is_stable():
    """The block sits under one name on the card, the verdict block and the
    stamped plan — a rename is a cross-surface break, so it is pinned."""
    assert PIPELINE_GATES_KEY == "pipeline_gates"


# ---------------------------------------------------------------------------
# 4. alpha-engine-config-I7312 — a fired degradation family withholds the
#    attestation, and the VERIFIED sentence can never assert otherwise
#
# The gap this closes: every pre-existing `gate_degraded=True` case above ALSO
# sets a probe to UNKNOWN, so the verdict came out UNKNOWN for the other
# reason and the degradation's own effect on it was never exercised. The
# combination below — both probes MEASURED, every family reported, one fired —
# is reachable without any probe going UNKNOWN, because three SF states write
# `$.gate_degraded=true` that GATE_LABELS does not cover: EvaluatorGateDegraded,
# EvaluatorDirectorGateDegraded and SetMutexAcquireDegradedFlag.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", [
    "gate_degraded",
    "health_check_degraded",
    "parity_degraded",
    "research_predictor_degraded",
])
def test_a_fired_family_withholds_the_verdict_even_with_every_probe_measured(family):
    block = read_gate_state(_clean_payload(**{family: True}))

    assert block["unmeasured"] == [], "precondition: both probes MEASURED"
    assert block["families_unreported"] == [], "precondition: every family reported"
    assert block["degraded_families"] == [family]

    assert block["verdict"] == UNKNOWN, (
        f"{family} fired and every probe reported MEASURED — the verdict was "
        "MEASURED before I7312, and the statement it drives asserts BOTH that "
        "the gates ran AND that nothing degraded."
    )
    assert gates_unmeasured(block) is True


@pytest.mark.parametrize("family", [
    "gate_degraded",
    "health_check_degraded",
    "parity_degraded",
    "research_predictor_degraded",
])
def test_the_statement_never_claims_no_degradation_when_one_fired(family):
    block = read_gate_state(_clean_payload(**{family: True}))
    statement = block["statement"]

    assert "recorded no fail-open degradation" not in statement, (
        f"the card claimed no fail-open degradation while degraded_families "
        f"carried {block['degraded_families']} — the contradiction I7312 fixed"
    )
    assert statement.startswith("NOT VERIFIED — "), (
        "the affirmative VERIFIED head is the one that carries the false claim; "
        f"got: {statement[:40]!r}"
    )
    # It must still say what is NOT wrong: a fired family is not evidence the
    # numbers are bad, and a reader who cannot tell the two apart stops reading.
    assert "UNATTESTED, not as wrong" in statement
    assert "FAILED" not in statement
    # ...and it must name which family, or the operator has nowhere to go.
    assert "Fail-open degradation recorded on this run" in statement


def test_the_reason_is_never_empty_on_the_degradation_only_path():
    """The one path where the verdict is UNKNOWN and NEITHER an unmeasured gate
    nor an unreported family explains why. Without its own top_reason the head
    renders the bare string 'NOT VERIFIED — '."""
    block = read_gate_state(_clean_payload(gate_degraded=True))
    assert block["reason"].strip(), "UNKNOWN verdict with no reason"
    assert "fail-open" in block["reason"]
    head, _, rest = block["statement"].partition("NOT VERIFIED — ")
    assert head == "" and rest.strip(), "rendered a bare 'NOT VERIFIED — '"


def test_a_genuinely_clean_run_still_verifies():
    """The fix must not make MEASURED unreachable — a verdict that is always
    UNKNOWN carries exactly as little information as one that is always
    MEASURED."""
    block = read_gate_state(_clean_payload())
    assert block["verdict"] == MEASURED
    assert block["degraded_families"] == []
    assert gates_unmeasured(block) is False
    assert block["statement"].startswith("VERIFIED — ")
    assert "recorded no fail-open degradation" in block["statement"]


def test_multiple_fired_families_are_all_named():
    block = read_gate_state(_clean_payload(gate_degraded=True, parity_degraded=True))
    assert block["verdict"] == UNKNOWN
    assert set(block["degraded_families"]) == {"gate_degraded", "parity_degraded"}
    assert "pre-spend gates" in block["statement"]
    assert "parity verdict" in block["statement"]
