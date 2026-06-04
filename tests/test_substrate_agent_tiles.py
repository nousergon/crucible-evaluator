"""Tests for grading/tiles/substrate.py + agent.py — Tiles 5 & 6."""

from datetime import UTC, datetime, timedelta

import boto3
import pytest
from moto import mock_aws

from grading.tiles.agent import build_agent_tile
from grading.tiles.substrate import PRICE_CACHE_PREFIX, build_substrate_tile

BUCKET = "alpha-engine-research"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _comp(tile, name):
    return next(c for c in tile["components"] if c["name"] == name)


class TestSubstrate:
    def test_price_cache_freshness_fresh_green(self, s3):
        s3.put_object(Bucket=BUCKET, Key=f"{PRICE_CACHE_PREFIX}A.parquet", Body=b"x")
        tile = build_substrate_tile(BUCKET, s3_client=s3, as_of=datetime.now(UTC))
        pcf = _comp(tile, "price_cache_freshness")
        assert pcf["value"] is not None
        assert pcf["value"] < 1.0  # ~0 days old
        assert pcf["status"] == "GREEN"

    def test_price_cache_freshness_stale_red(self, s3):
        s3.put_object(Bucket=BUCKET, Key=f"{PRICE_CACHE_PREFIX}A.parquet", Body=b"x")
        future = datetime.now(UTC) + timedelta(days=20)  # > 14d red-line
        tile = build_substrate_tile(BUCKET, s3_client=s3, as_of=future)
        assert _comp(tile, "price_cache_freshness")["status"] == "RED"

    def test_price_cache_absent_missing_input(self, s3):
        tile = build_substrate_tile(BUCKET, s3_client=s3)
        assert _comp(tile, "price_cache_freshness")["status"] == "N/A-MISSING-INPUT"

    def test_sf_success_is_not_impl_with_producer_reason(self, s3):
        tile = build_substrate_tile(BUCKET, s3_client=s3)
        sf = _comp(tile, "sf_success_rate_4w")
        assert sf["status"] == "N/A-NOT-IMPL"
        assert "pipeline_status" in sf["status_reason"]

    def test_tile_has_nine_components(self, s3):
        tile = build_substrate_tile(BUCKET, s3_client=s3)
        assert tile["n_components"] == 9
        assert tile["module"] == "substrate"

    def test_tile_watch_when_only_freshness_real(self, s3):
        # price_cache fresh GREEN but critical sf/data_quality/schema N/A-NOT-IMPL → WATCH.
        s3.put_object(Bucket=BUCKET, Key=f"{PRICE_CACHE_PREFIX}A.parquet", Body=b"x")
        tile = build_substrate_tile(BUCKET, s3_client=s3, as_of=datetime.now(UTC))
        assert tile["status"] == "WATCH"


class TestAgent:
    def test_all_components_na(self, s3):
        tile = build_agent_tile(BUCKET, s3_client=s3)
        assert tile["module"] == "agent"
        statuses = {c["status"] for c in tile["components"]}
        assert statuses <= {"N/A-NOT-IMPL", "N/A-MISSING-INPUT"}
        # critical N/A → tile WATCH (transparency, never a false GREEN).
        assert tile["status"] == "WATCH"
        assert tile["numeric_grade"] is None

    def test_stance_source_missing_input(self, s3):
        tile = build_agent_tile(BUCKET, s3_client=s3)
        ss = _comp(tile, "stance_source_provenance")
        assert ss["status"] == "N/A-MISSING-INPUT"
        assert "stance_source" in ss["status_reason"]

    def test_judge_kappa_not_impl_names_producer(self, s3):
        tile = build_agent_tile(BUCKET, s3_client=s3)
        jk = _comp(tile, "judge_calibration_cohen_kappa")
        assert jk["status"] == "N/A-NOT-IMPL"
        assert "decision_artifacts" in jk["status_reason"]

    def test_seven_components(self, s3):
        tile = build_agent_tile(BUCKET, s3_client=s3)
        assert tile["n_components"] == 7
