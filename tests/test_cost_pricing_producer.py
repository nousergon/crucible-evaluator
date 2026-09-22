"""Tests for grading/producers/cost_pricing.py — the degraded-cost_source scan.

alpha-engine-config-I11113. The invariant under test throughout: **"zero
degraded rows" and "I could not read the window" never render identically.**
A test that only proves zero renders as zero does not close this issue.
"""

import json

import boto3
import pytest
from moto import mock_aws

from grading.producers.cost_pricing import (
    COST_RAW_PREFIX,
    DEGRADED_COST_SOURCES,
    CostPricingUnmeasured,
    format_by_callsite,
    scan_degraded_cost_rows,
    window_partitions,
)

BUCKET = "alpha-engine-research"
RUN_DATE = "2026-09-19"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _row(cost_source="price_card", callsite_id="director-plan", cost_usd=0.5):
    row = {
        "ts": f"{RUN_DATE}T12:00:00+00:00",
        "provider": "anthropic",
        "model": "claude-opus-5",
        "input_tokens": 100,
        "output_tokens": 20,
        "cost_usd": cost_usd,
        "cost_source": cost_source,
    }
    if callsite_id is not None:
        row["callsite_id"] = callsite_id
        row["agent_id"] = callsite_id
    if cost_source == "usage_unreported":
        row["usage_unknown"] = True
        row["cost_usd"] = None
    if cost_source == "unpriced":
        row["cost_usd"] = None
    return row


def _put_rows(s3, date, callsite, rows, seq=0):
    key = f"{COST_RAW_PREFIX}/{date}/krepis-abc123/{callsite}.{seq}.jsonl"
    body = "\n".join(json.dumps(r) for r in rows).encode("utf-8")
    s3.put_object(Bucket=BUCKET, Key=key, Body=body)
    return key


class TestWindow:
    def test_window_is_the_trailing_week_plus_the_spill_day(self):
        parts = window_partitions(RUN_DATE)
        assert parts[0] == "2026-09-13"
        assert parts[-1] == "2026-09-20"
        assert len(parts) == 8
        assert RUN_DATE in parts

    def test_unparseable_run_date_is_unmeasured_not_an_empty_window(self, s3):
        with pytest.raises(CostPricingUnmeasured):
            scan_degraded_cost_rows(s3, BUCKET, "not-a-date")


class TestCounting:
    def test_empty_window_is_a_measured_zero(self, s3):
        out = scan_degraded_cost_rows(s3, BUCKET, RUN_DATE)
        assert out["degraded_total"] == 0
        assert out["by_callsite"] == {}
        assert out["rows_scanned"] == 0

    def test_priced_rows_do_not_count(self, s3):
        _put_rows(s3, RUN_DATE, "director-plan", [_row(), _row(), _row(cost_source="provider_reported")])
        out = scan_degraded_cost_rows(s3, BUCKET, RUN_DATE)
        assert out["degraded_total"] == 0
        assert out["rows_scanned"] == 3

    def test_both_degrade_paths_are_counted_not_just_the_newer_one(self, s3):
        """I11113 is explicit: registering only `unpriced` would leave the older
        `usage_unreported` gap — unmonitored since it was added — reading as
        covered."""
        _put_rows(s3, RUN_DATE, "director-plan", [
            _row(cost_source="unpriced"),
            _row(cost_source="usage_unreported"),
            _row(),
        ])
        out = scan_degraded_cost_rows(s3, BUCKET, RUN_DATE)
        assert out["degraded_total"] == 2
        assert out["by_cost_source"] == {"unpriced": 1, "usage_unreported": 1}
        assert set(DEGRADED_COST_SOURCES) == {"unpriced", "usage_unreported"}

    def test_breakdown_is_per_callsite(self, s3):
        _put_rows(s3, RUN_DATE, "director-plan", [_row(cost_source="unpriced")] * 3)
        _put_rows(s3, RUN_DATE, "evaljudge-sync",
                  [_row(cost_source="usage_unreported", callsite_id="evaljudge-sync")])
        out = scan_degraded_cost_rows(s3, BUCKET, RUN_DATE)
        assert out["by_callsite"]["director-plan"]["unpriced"] == 3
        assert out["by_callsite"]["evaljudge-sync"]["usage_unreported"] == 1
        assert out["by_callsite"]["evaljudge-sync"]["total"] == 1

    def test_callsite_falls_back_to_the_object_key_when_the_row_carries_none(self, s3):
        _put_rows(s3, RUN_DATE, "replay-concordance",
                  [_row(cost_source="unpriced", callsite_id=None)])
        out = scan_degraded_cost_rows(s3, BUCKET, RUN_DATE)
        assert out["by_callsite"]["replay-concordance"]["unpriced"] == 1

    def test_spill_partition_is_scanned(self, s3):
        """krepis partitions on the record's own ts; the weekly SF's own rows
        land on run_date + 1 (measured 2026-08-29, see cost_coverage.py)."""
        _put_rows(s3, "2026-09-20", "director-plan", [_row(cost_source="unpriced")])
        assert scan_degraded_cost_rows(s3, BUCKET, RUN_DATE)["degraded_total"] == 1

    def test_weekday_pipeline_rows_earlier_in_the_week_are_scanned(self, s3):
        _put_rows(s3, "2026-09-15", "preopen-scan", [_row(cost_source="unpriced")])
        assert scan_degraded_cost_rows(s3, BUCKET, RUN_DATE)["degraded_total"] == 1

    def test_rows_outside_the_window_are_not_scanned(self, s3):
        _put_rows(s3, "2026-09-01", "director-plan", [_row(cost_source="unpriced")])
        assert scan_degraded_cost_rows(s3, BUCKET, RUN_DATE)["degraded_total"] == 0

    def test_non_jsonl_objects_are_ignored(self, s3):
        s3.put_object(Bucket=BUCKET, Key=f"{COST_RAW_PREFIX}/{RUN_DATE}/_SUCCESS",
                      Body=b"not json")
        out = scan_degraded_cost_rows(s3, BUCKET, RUN_DATE)
        assert out["objects_scanned"] == 0
        assert out["degraded_total"] == 0


class TestUnreadableIsNotZero:
    def test_malformed_jsonl_raises_rather_than_counting_zero(self, s3):
        key = f"{COST_RAW_PREFIX}/{RUN_DATE}/krepis-abc/director-plan.0.jsonl"
        s3.put_object(Bucket=BUCKET, Key=key, Body=b"{not json}\n")
        with pytest.raises(CostPricingUnmeasured) as exc:
            scan_degraded_cost_rows(s3, BUCKET, RUN_DATE)
        assert "malformed" in str(exc.value)

    def test_non_object_record_raises(self, s3):
        key = f"{COST_RAW_PREFIX}/{RUN_DATE}/krepis-abc/director-plan.0.jsonl"
        s3.put_object(Bucket=BUCKET, Key=key, Body=b'["a list, not a row"]\n')
        with pytest.raises(CostPricingUnmeasured):
            scan_degraded_cost_rows(s3, BUCKET, RUN_DATE)

    def test_list_failure_raises_rather_than_counting_zero(self, s3):
        class _DeniedPaginator:
            def paginate(self, **_kw):
                raise RuntimeError("AccessDenied")

        class _Denied:
            def get_paginator(self, _name):
                return _DeniedPaginator()

        with pytest.raises(CostPricingUnmeasured) as exc:
            scan_degraded_cost_rows(_Denied(), BUCKET, RUN_DATE)
        assert "could not list" in str(exc.value)

    def test_get_object_failure_raises_rather_than_counting_zero(self, s3):
        _put_rows(s3, RUN_DATE, "director-plan", [_row()])

        class _ReadDenied:
            def __init__(self, inner):
                self._inner = inner

            def get_paginator(self, name):
                return self._inner.get_paginator(name)

            def get_object(self, **_kw):
                raise RuntimeError("AccessDenied")

        with pytest.raises(CostPricingUnmeasured) as exc:
            scan_degraded_cost_rows(_ReadDenied(s3), BUCKET, RUN_DATE)
        assert "could not read" in str(exc.value)

    def test_truncated_listing_refuses_rather_than_under_reporting(self, s3):
        for i in range(3):
            _put_rows(s3, RUN_DATE, "director-plan", [_row(cost_source="unpriced")], seq=i)
        with pytest.raises(CostPricingUnmeasured) as exc:
            scan_degraded_cost_rows(s3, BUCKET, RUN_DATE, max_objects=2)
        assert "truncated" in str(exc.value)


class TestFormatting:
    def test_worst_callsite_first_and_truncation_is_explicit(self):
        by = {f"site-{i}": {"unpriced": i, "usage_unreported": 0, "total": i}
              for i in range(1, 8)}
        out = format_by_callsite(by, limit=2)
        assert out.startswith("site-7 (unpriced=7)")
        assert "+5 more callsite(s)" in out

    def test_empty_breakdown_says_none(self):
        assert format_by_callsite({}) == "none"
