"""Producer-side contract test for the reference/champion experiment's
``experiment_record.v1`` boundary (alpha-engine-config#3077 Phase C).

crucible-evaluator is the PRODUCER of
``experiments/reference/records/{run_date}.json`` (+ .../latest.json), the
per-run index the results renderer consumes instead of a directory listing.
This test builds a real payload via the local ``build_experiment_record``
builder (mirrors ``tests/test_attractiveness_consumer_contract.py``'s
``contracts.validate`` usage and crucible-research's
``tests/test_research_intel_producer_contract.py`` producer-contract
pattern) and asserts it validates against the shared
``nousergon_lib.contracts`` schema — the single cross-repo source of truth —
plus pins the partial-run / absent-artifact honesty the record exists to
guarantee.
"""

from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws
from nousergon_lib import contracts

from grading.experiment_record import (
    REFERENCE_EXPERIMENT_ID,
    build_experiment_record,
    experiment_record_key,
    latest_experiment_record_key,
    write_experiment_record,
)

BUCKET = "alpha-engine-research"
RUN_DATE = "2026-07-18"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _card(*, missing=None, read=None):
    return {
        "_provenance": {
            "run_date": RUN_DATE,
            "freshness_preflight": {
                "run_date": RUN_DATE,
                "checks": [
                    {"artifact_id": "research_signals", "content_date": RUN_DATE, "window": 7},
                    {"artifact_id": "predictor_meta_weights_manifest", "content_date": RUN_DATE, "window": 7},
                    {"artifact_id": "metrics_json", "content_date": RUN_DATE, "window": 7},
                    {"artifact_id": "e2e_lift_json", "content_date": RUN_DATE, "window": 7},
                    {"artifact_id": "eod_reconcile_pnl", "content_date": RUN_DATE, "window": 7},
                ],
            },
            "artifacts": {
                "artifacts_read": read if read is not None else ["metrics.json", "e2e_lift.json"],
                "artifacts_missing": missing if missing is not None else [],
                "n_read": len(read) if read is not None else 2,
                "n_missing": len(missing) if missing is not None else 0,
            },
        },
    }


def test_complete_run_validates_against_lib_contract():
    card = _card()
    record = build_experiment_record(
        BUCKET, RUN_DATE, card,
        report_card_key="evaluator/2026-07-18/report_card.json",
    )
    errors = contracts.conformance_errors("experiment_record", record)
    assert errors == [], (
        "experiment_record payload violates the nousergon_lib contract:\n  "
        + "\n  ".join(errors)
    )
    assert record["schema_version"] == 1
    assert record["experiment_id"] == REFERENCE_EXPERIMENT_ID
    assert record["run_date"] == RUN_DATE
    assert record["status"] == "complete"


def test_required_top_level_fields_present():
    record = build_experiment_record(
        BUCKET, RUN_DATE, _card(),
        report_card_key="evaluator/2026-07-18/report_card.json",
    )
    required = {"schema_version", "experiment_id", "run_date", "status", "manifest", "slots", "artifacts"}
    missing = required - record.keys()
    assert not missing, f"experiment_record producer dropped required field(s): {sorted(missing)}"


def test_slots_carry_one_entry_per_product_slot():
    record = build_experiment_record(
        BUCKET, RUN_DATE, _card(),
        report_card_key="evaluator/2026-07-18/report_card.json",
    )
    slot_names = {s["slot"] for s in record["slots"]}
    assert slot_names == {"research", "model", "strategy"}
    for s in record["slots"]:
        assert s["impl"] in ("stock", "artifact", "command", "entry_point")
        assert "@" in s["fingerprint"]


def test_manifest_hash_is_deterministic():
    r1 = build_experiment_record(
        BUCKET, RUN_DATE, _card(), report_card_key="k",
    )
    r2 = build_experiment_record(
        BUCKET, RUN_DATE, _card(), report_card_key="a-different-key",
    )
    # The manifest hash covers the SLOTS (what ran), not the report-card key
    # (where the output landed) — two runs with identical upstream artifact
    # provenance must hash identically regardless of where the card itself
    # was written.
    assert r1["manifest"]["hash"] == r2["manifest"]["hash"]


def test_missing_artifact_is_a_link_table_row_never_an_omission():
    # Partial run: one upstream artifact absent. The record must carry it as
    # an explicit status="absent" row with a reason — never silently drop it.
    card = _card(missing=["behavioral_anomaly.json"])
    record = build_experiment_record(
        BUCKET, RUN_DATE, card,
        report_card_key="evaluator/2026-07-18/report_card.json",
    )
    assert record["status"] == "partial"
    absent_rows = [a for a in record["artifacts"] if a["status"] == "absent"]
    names = {a["name"] for a in absent_rows}
    assert "behavioral_anomaly.json" in names
    for row in absent_rows:
        assert row.get("reason"), f"absent row {row['name']!r} missing a reason"
    errors = contracts.conformance_errors("experiment_record", record)
    assert errors == []


def test_no_report_card_key_grades_status_failed():
    # The report card itself was never persisted this cycle (e.g. dry run) —
    # the record must say so honestly, not claim "complete".
    record = build_experiment_record(
        BUCKET, RUN_DATE, _card(), report_card_key=None,
    )
    assert record["status"] == "failed"
    report_card_row = next(a for a in record["artifacts"] if a["name"] == "report_card")
    assert report_card_row["status"] == "absent"
    assert report_card_row["reason"]
    errors = contracts.conformance_errors("experiment_record", record)
    assert errors == []


class TestWriteExperimentRecord:
    def test_writes_dated_and_latest_keys(self, s3):
        record = build_experiment_record(
            BUCKET, RUN_DATE, _card(),
            report_card_key="evaluator/2026-07-18/report_card.json",
        )
        written = write_experiment_record(BUCKET, RUN_DATE, record, s3_client=s3)
        assert written["dated_key"] == experiment_record_key(RUN_DATE)
        assert written["latest_key"] == latest_experiment_record_key()
        assert written["dated_key"] == f"experiments/{REFERENCE_EXPERIMENT_ID}/records/{RUN_DATE}.json"
        assert written["latest_key"] == f"experiments/{REFERENCE_EXPERIMENT_ID}/records/latest.json"

        dated_body = json.loads(s3.get_object(Bucket=BUCKET, Key=written["dated_key"])["Body"].read())
        latest_body = json.loads(s3.get_object(Bucket=BUCKET, Key=written["latest_key"])["Body"].read())
        assert dated_body == record
        assert latest_body == record
