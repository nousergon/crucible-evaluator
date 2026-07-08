"""Consumer-side contract test: backtester tile vs apply_audit.json v1.

M0 contract discipline: ``config/apply_audit/{date}.json`` (+ a ``latest.json``
convenience copy the grader does not consume) is a FROZEN schema_version-1
producer/consumer contract (producer: the backtester's auto-apply audit writer,
landing in a parallel PR; consumer: the backtester tile's ``apply_loop_health``
component, config#1841). This test grades the tile against the canonical
fixture (tests/fixtures/apply_audit_v1.json) so any consumer drift from the
frozen shape fails HERE, in this repo's CI — mirroring
test_attractiveness_consumer_contract.py.

Forward-compat is part of the contract: the producer deploys independently, so
an ABSENT artifact (pre-first-Saturday / producer PR unmerged) must grade an
honest, specific N/A — never a crash, never a silent omission — and the
component must self-activate on the first emission with no consumer change.

Status semantics under test (the config#1841 silence-killer):
  RED   — any loop ``error``, or ``blocked`` with consecutive_blocked_weeks>=4
  WATCH — any loop ``blocked`` with consecutive_blocked_weeks in [2, 3]
  GREEN — every loop promoted / insufficient_data / disabled / blocked <2w
  RED   — unrecognized outcome value (producer contract drift; fail loud)
"""

import copy
import json
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from grading.tiles.backtester import build_backtester_tile

BUCKET = "alpha-engine-research"
RUN_DATE = "2026-07-04"
FIXTURE = Path(__file__).parent / "fixtures" / "apply_audit_v1.json"

LOOP_NAMES = {"scoring_weights", "executor_params", "predictor_params", "research_params"}
OUTCOMES = {"promoted", "blocked", "insufficient_data", "error", "disabled"}
LOOP_KEYS = {"outcome", "blocked_by", "consecutive_blocked_weeks", "detail", "proposed", "current"}


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def _put(s3, doc, date=RUN_DATE):
    s3.put_object(Bucket=BUCKET, Key=f"config/apply_audit/{date}.json",
                  Body=json.dumps(doc).encode())


def _health(s3):
    tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
    return next(c for c in tile["components"] if c["name"] == "apply_loop_health")


def _all_green_doc() -> dict:
    doc = _fixture()
    for entry in doc["loops"].values():
        entry.update(outcome="promoted", blocked_by=None, consecutive_blocked_weeks=0)
    return doc


class TestFixtureShape:
    """The fixture IS the frozen v1 producer contract — drift fails here."""

    def test_fixture_carries_the_frozen_v1_shape(self):
        doc = _fixture()
        assert doc["schema_version"] == 1
        assert doc["as_of"] == "2026-07-04"
        assert set(doc["loops"]) == LOOP_NAMES
        for name, entry in doc["loops"].items():
            assert set(entry) == LOOP_KEYS, f"loops[{name}] keys drifted"
            assert entry["outcome"] in OUTCOMES, f"loops[{name}] bad outcome"
            assert isinstance(entry["consecutive_blocked_weeks"], int)
            assert entry["blocked_by"] is None or isinstance(entry["blocked_by"], list)
            assert isinstance(entry["detail"], str)


class TestRed:
    def test_fixture_long_blocked_loops_grade_red(self, s3):
        # The live config#1841 state: two loops blocked >=4 consecutive weeks.
        _put(s3, _fixture())
        m = _health(s3)
        assert m["status"] == "RED"
        assert m["criticality"] == "critical"
        assert m["value"] == 2.0  # scoring_weights + predictor_params unhealthy
        assert m["n_samples"] == 4
        # Director-digest specificity: loop + outcome + blocked_by slug + weeks.
        assert "scoring_weights blocked 9w [significance_floor]" in m["status_reason"]
        assert "predictor_params blocked 8w [oos_degradation]" in m["status_reason"]

    def test_error_outcome_red(self, s3):
        doc = _all_green_doc()
        doc["loops"]["executor_params"].update(outcome="error", detail="optimizer raised KeyError")
        _put(s3, doc)
        m = _health(s3)
        assert m["status"] == "RED"
        assert m["value"] == 1.0
        assert "executor_params error (optimizer raised KeyError)" in m["status_reason"]

    def test_unrecognized_outcome_red_not_silently_green(self, s3):
        # Producer contract drift (a new/renamed outcome) must fail loud.
        doc = _all_green_doc()
        doc["loops"]["research_params"]["outcome"] = "deferred"
        _put(s3, doc)
        m = _health(s3)
        assert m["status"] == "RED"
        assert "unrecognized outcome 'deferred'" in m["status_reason"]

    def test_blocked_exactly_4_weeks_red(self, s3):
        doc = _all_green_doc()
        doc["loops"]["scoring_weights"].update(
            outcome="blocked", blocked_by=["significance_floor"], consecutive_blocked_weeks=4)
        _put(s3, doc)
        assert _health(s3)["status"] == "RED"


class TestWatch:
    @pytest.mark.parametrize("weeks", [2, 3])
    def test_blocked_2_or_3_weeks_watch(self, s3, weeks):
        doc = _all_green_doc()
        doc["loops"]["predictor_params"].update(
            outcome="blocked", blocked_by=["oos_degradation"], consecutive_blocked_weeks=weeks)
        _put(s3, doc)
        m = _health(s3)
        assert m["status"] == "WATCH"
        assert m["value"] == 1.0
        assert f"predictor_params blocked {weeks}w [oos_degradation]" in m["status_reason"]


class TestGreen:
    def test_all_healthy_outcomes_green(self, s3):
        doc = _all_green_doc()
        doc["loops"]["research_params"].update(outcome="insufficient_data")
        doc["loops"]["predictor_params"].update(outcome="disabled")
        # blocked <2 consecutive weeks is still healthy (not yet a stall).
        doc["loops"]["scoring_weights"].update(
            outcome="blocked", blocked_by=["significance_floor"], consecutive_blocked_weeks=1)
        _put(s3, doc)
        m = _health(s3)
        assert m["status"] == "GREEN"
        assert m["value"] == 0.0
        assert m["n_samples"] == 4


class TestForwardCompat:
    def test_absent_artifact_grades_honest_na(self, s3):
        # Producer PR unmerged / pre-first-Saturday: specific N/A, no crash.
        m = _health(s3)
        assert m["status"] == "N/A-MISSING-INPUT"
        assert m["criticality"] == "critical"
        assert m["value"] is None
        assert "config#1841" in m["status_reason"]
        assert "self-activates" in m["status_reason"]

    def test_self_activates_on_first_emission_via_windowed_read(self, s3):
        # Artifact written on an earlier date inside the 10-day window (the
        # weekly Saturday cadence + retry slack) grades — no consumer change.
        _put(s3, _all_green_doc(), date="2026-06-27")
        m = _health(s3)
        assert m["status"] == "GREEN"
        assert "config/apply_audit/2026-06-27.json" in m["source_path"]

    def test_malformed_loops_block_grades_na_not_crash(self, s3):
        _put(s3, {"schema_version": 1, "as_of": RUN_DATE, "loops": None})
        m = _health(s3)
        assert m["status"] == "N/A-MISSING-INPUT"
        assert "malformed" in m["status_reason"]

    def test_additive_producer_fields_tolerated(self, s3):
        # S3 contract safety: producer may ADD fields; consumer must not care.
        doc = copy.deepcopy(_all_green_doc())
        doc["producer_version"] = "1.4.0"
        for entry in doc["loops"].values():
            entry["new_diag_field"] = {"anything": True}
        _put(s3, doc)
        assert _health(s3)["status"] == "GREEN"
