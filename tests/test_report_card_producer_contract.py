"""Producer contract: the emitted Report Card against its FROZEN schema.

The M0 discipline (`~/Development/AGENTS.md`) requires every cross-repo artifact
to carry a versioned schema **and a producer/consumer contract test at birth**.
``evaluator/{date}/report_card.json`` has had the schema since config#2343 —
``nousergon_lib/contracts/report_card.schema.json`` — and had never had the
producer half. The consumer half exists in three repos (crucible-dashboard's
``report_card_v2``, the console's ``document-fields`` binding, and the Director);
nothing validated what this repo actually writes.

That gap was not theoretical. It was found by writing this file: the contract has
declared ``schema_version`` REQUIRED since it was frozen, and
``build_report_card`` had never stamped it — so every card this repo has ever
written was invalid against its own write-contract, invisibly, because all three
consumers are tolerant on read (config-I7039).

The §2.3a correctness block is asserted here too. It is not yet declared in the
lib schema (``additionalProperties`` is true, so it validates as an unknown
field) — this test pins the producer side while that declaration is tracked
separately, so the block cannot silently stop being emitted in the meantime.
"""
from __future__ import annotations

import json
from importlib.resources import files

import boto3
import jsonschema
import pytest
from moto import mock_aws

from grading.aggregate import build_report_card
# The freshness preflight hard-fails a bucket with none of its required inputs
# (by design), so the minimum viable card needs them seeded. Reusing the
# existing helper rather than copying it: a second copy of the required-input
# list is a second thing to update when the preflight grows a check, and the
# copy that is missed is the one that stops testing anything.
from test_aggregate import BUCKET, RUN_DATE, _put, _seed_freshness_inputs


@pytest.fixture(scope="module")
def schema() -> dict:
    """The FROZEN contract, read from the installed lib — never a local copy.

    Reading the shipped resource rather than a checked-in duplicate is what makes
    this a contract test: a lib pin bump that changed the contract would fail
    here, which is the whole point of pinning it.
    """
    return json.loads(
        (files("nousergon_lib.contracts") / "report_card.schema.json").read_text()
    )


@pytest.fixture
def s3():
    with mock_aws():
        c = boto3.client("s3", region_name="us-east-1")
        c.create_bucket(Bucket=BUCKET)
        yield c


@pytest.fixture
def card(s3):
    """The MINIMUM viable card: only the freshness preflight's hard inputs.

    Deliberately the sparse case — every optional artifact absent, every tile
    N/A, and both upstream §2.3a verdicts unreadable. If the producer emits a
    contract-valid body only when its inputs are all present, the contract is
    not being met on exactly the runs where a consumer most needs to parse it.
    """
    _seed_freshness_inputs(s3)
    _put(s3, "metrics.json", {"run_date": RUN_DATE, "status": "ok"})
    _put(s3, "e2e_lift.json", {"status": "ok"})
    return build_report_card(BUCKET, RUN_DATE, s3_client=s3)


def test_emitted_card_validates_against_the_frozen_contract(card, schema):
    jsonschema.validate(instance=card, schema=schema)


def test_schema_version_is_stamped(card):
    # The regression guard for the defect this file found. `schema_version` is
    # `const: 1` on the contract; a producer that omits it writes a body that
    # only a tolerant reader accepts, and every reader being tolerant is how it
    # stayed invisible.
    assert card["schema_version"] == 1


def test_every_required_field_is_present(card, schema):
    missing = [k for k in schema["required"] if k not in card]
    assert not missing, f"producer omits required contract fields: {missing}"


class TestCorrectnessBlock:
    """§2.3a — the verdict is part of what this producer promises to emit."""

    def test_attestation_is_always_emitted(self, card):
        # ALWAYS-EMIT is what makes absence unambiguous downstream: a consumer
        # that finds no block knows the producer never ran, rather than having
        # to guess between that and "ran, nothing to say".
        assert "attestation" in card
        assert card["attestation"]["schema"] == "report_card_attestation-1.0.0"

    def test_verdict_is_in_the_closed_vocabulary(self, card):
        assert card["attestation"]["verdict"] in {"PASS", "FAIL", "PARTIAL", "UNKNOWN"}

    def test_absent_upstream_verdicts_yield_unknown_not_pass(self, card):
        # Neither `backtest/{date}/attestation.json` nor its evaluator-stage
        # sibling exists on this bucket. That is an absence of evidence and the
        # producer must say so — the single most important assertion in this file.
        assert card["attestation"]["verdict"] == "UNKNOWN"
        assert card["degraded_attestation"] is True

    def test_as_of_is_present_for_each_upstream_half(self, card):
        # Declared even when null: a verdict with no timestamp cannot read as
        # stale, and the key's absence would be indistinguishable from a
        # consumer that forgot to look.
        as_of = card["attestation"]["as_of"]
        assert set(as_of) == {"backtester", "evaluator_stage", "contamination"}


class TestCoverageBlock:
    """config-I7202 — coverage is part of what this producer promises to emit.

    Not yet declared in the lib schema (``additionalProperties`` is true, so it
    validates as unknown fields); this pins the producer side while the schema
    declaration is tracked separately, exactly as the §2.3a block above does.
    The card under test is the SPARSE one — the case where coverage matters
    most, because that is when the renormalization is largest.
    """

    LEVELS = ("overall", "research", "predictor", "executor")

    def test_declared_weights_are_stamped_on_the_artifact(self, card):
        # The whole point: a reader recomputes from the card, not from source at
        # the right commit.
        gw = card["grading_weights"]
        assert set(gw) >= {"version", "rule", "overall", "research", "predictor", "executor"}
        for section in ("overall", "research", "predictor", "executor"):
            assert abs(sum(gw[section].values()) - 1.0) < 1e-9

    @pytest.mark.parametrize("level", LEVELS)
    def test_every_level_carries_coverage(self, card, level):
        cov = card[level]["coverage"]
        assert set(cov) >= {
            "weight_present", "weight_present_effective", "components_skipped",
            "skips", "weights", "qualifier", "weight_failed", "components_failed",
            "skip_classes", "floor", "provisional",
        }

    @pytest.mark.parametrize("level", LEVELS)
    def test_a_partial_grade_never_renders_as_a_bare_letter(self, card, level):
        block = card[level]
        if block["coverage"]["qualifier"] != "COMPLETE":
            assert block["display"] != block["letter"]

    def test_the_sparse_card_reports_its_own_emptiness(self, card):
        # ALWAYS-EMIT, same reasoning as the attestation block: a consumer that
        # finds no coverage knows the producer never ran, rather than guessing.
        assert card["overall"]["coverage"]["qualifier"] in {
            "COMPLETE", "PARTIAL", "PARTIAL-MASKED-FAILURE", "UNGRADED", "UNKNOWN",
        }

    def test_coverage_survives_the_schema(self, card, schema):
        jsonschema.validate(instance=card, schema=schema)
