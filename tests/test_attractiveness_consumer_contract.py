"""Consumer-side contract test: research tile vs attractiveness_eval.json v1.

M0 contract discipline: ``backtest/{date}/attractiveness_eval.json`` is a
FROZEN schema_version-1 producer/consumer contract (producer: the backtester's
attractiveness evaluator, landing in a parallel PR; consumer: the research
tile's attractiveness_ic / attractiveness_trajectory_ic /
scanner_feed_counterfactual components). This test grades the tile against the
canonical fixture (tests/fixtures/attractiveness_eval_v1.json) so any consumer
drift from the frozen shape fails HERE, in this repo's CI — mirroring the
cross-repo consumer-contract pattern (test_scanner_consumer_contract.py in
crucible-research).

Forward-compat is part of the contract: the producer deploys independently, so
an ABSENT artifact (or status="insufficient_data") must grade an honest,
specific N/A — never a crash, never a silent omission.
"""

import json
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from grading.tiles.research import build_research_tile

BUCKET = "alpha-engine-research"
RUN_DATE = "2026-07-04"
FIXTURE = Path(__file__).parent / "fixtures" / "attractiveness_eval_v1.json"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def _put_att(s3, doc, date=RUN_DATE):
    s3.put_object(Bucket=BUCKET, Key=f"backtest/{date}/attractiveness_eval.json",
                  Body=json.dumps(doc).encode())


def _comp(tile, name):
    return next(c for c in tile["components"] if c["name"] == name)


class TestFixtureShape:
    def test_fixture_carries_the_frozen_v1_shape(self):
        doc = _fixture()
        assert doc["schema_version"] == 1
        assert doc["status"] in ("ok", "insufficient_data")
        assert doc["horizon_days"] == 21
        for k in ("date_ic_mean", "date_ic_t", "date_ic_p", "n_eval_dates",
                  "pooled_ic", "pooled_ic_p", "n"):
            assert k in doc["composite_ic"], f"composite_ic missing {k}"
        for pillar, blk in doc["pillar_ic"].items():
            for k in ("date_ic_mean", "date_ic_p", "n_eval_dates"):
                assert k in blk, f"pillar_ic[{pillar}] missing {k}"
        assert set(doc["suggested_pillar_weights"]) == set(doc["pillar_ic"])
        assert doc["shrinkage"]["method"] == "demiguel_1overN"
        assert set(doc["trajectory_ic"]) == {"pre_repricing_score", "attr_slope_z"}
        for entry in doc["counterfactual"]["top_n"]:
            assert set(entry) == {"n", "sector_balanced", "capture_rate", "mean_alpha_21d"}
        assert set(doc["counterfactual"]["live_gate"]) == {
            "capture_rate", "mean_alpha_21d", "n_survivors"}


class TestAttractivenessIc:
    def test_grades_from_date_clustered_ic(self, s3):
        _put_att(s3, _fixture())
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "attractiveness_ic")
        assert m["criticality"] == "supporting"
        assert m["value"] == pytest.approx(0.062)
        assert m["n_samples"] == 11
        assert m["estimator"] == "date_clustered_rank_ic_vs_21d_alpha"
        assert m["measurement_horizon"] == "21d"
        # p=0.041 < 0.10 and N=11 >= 8 → significant, lib-derived status.
        assert m["status"] == "GREEN"
        assert m["reliability"] == "high"
        # Reason carries the per-pillar IC summary + suggested weights.
        assert "quant=+0.055" in m["status_reason"]
        assert "quant=0.45" in m["status_reason"]
        assert "demiguel_1overN" in m["status_reason"]

    def test_insignificant_grades_watch_low_reliability(self, s3):
        doc = _fixture()
        doc["composite_ic"]["date_ic_p"] = 0.34
        _put_att(s3, doc)
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "attractiveness_ic")
        assert m["status"] == "WATCH"
        assert m["reliability"] == "low"
        assert "accumulating" in m["status_reason"]

    def test_small_n_eval_dates_grades_watch_low_reliability(self, s3):
        doc = _fixture()
        doc["composite_ic"]["n_eval_dates"] = 5  # below the 8-week floor
        _put_att(s3, doc)
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "attractiveness_ic")
        assert m["status"] == "WATCH"
        assert m["reliability"] == "low"

    def test_absent_artifact_grades_specific_na(self, s3):
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "attractiveness_ic")
        assert m["status"] == "N/A-MISSING-INPUT"
        assert "config#1389" in m["status_reason"]

    def test_insufficient_data_status_grades_na(self, s3):
        _put_att(s3, {"schema_version": 1, "status": "insufficient_data",
                      "as_of": RUN_DATE, "horizon_days": 21})
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "attractiveness_ic")
        assert m["status"] == "N/A-MISSING-INPUT"
        assert "insufficient" in m["status_reason"]


class TestTrajectoryIc:
    def test_grades_pre_repricing_ic_and_names_the_gate(self, s3):
        _put_att(s3, _fixture())
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3),
                  "attractiveness_trajectory_ic")
        assert m["criticality"] == "diagnostic"
        assert m["value"] == pytest.approx(0.071)
        assert m["n_samples"] == 11
        # p=0.029 → significant.
        assert m["status"] == "GREEN"
        # The reason names the config#1392 observe→cutover gate it evidences,
        # and carries the attr_slope_z ride-along.
        assert "config#1392" in m["status_reason"]
        assert "attr_slope_z" in m["status_reason"]

    def test_absent_grades_na_naming_gate(self, s3):
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3),
                  "attractiveness_trajectory_ic")
        assert m["status"] == "N/A-MISSING-INPUT"
        assert "config#1392" in m["status_reason"]


class TestScannerFeedCounterfactual:
    def test_prize_delta_at_matched_breadth(self, s3):
        _put_att(s3, _fixture())
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3),
                  "scanner_feed_counterfactual")
        assert m["criticality"] == "supporting"
        # live_gate n_survivors=27 → matched-breadth row is top-25 (non-balanced,
        # the first n=25 row); prize = 0.0121 − 0.0063.
        assert m["value"] == pytest.approx(0.0121 - 0.0063)
        assert m["n_samples"] == 27
        # Positive prize vs target 0.0 → GREEN; opportunity metric, never RED.
        assert m["status"] == "GREEN"
        assert m["red_line"] is None
        assert "config#1398" in m["status_reason"]
        assert "top-10" in m["status_reason"]  # all cohorts surfaced

    def test_negative_prize_is_watch_never_red(self, s3):
        doc = _fixture()
        doc["counterfactual"]["live_gate"]["mean_alpha_21d"] = 0.09
        _put_att(s3, doc)
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3),
                  "scanner_feed_counterfactual")
        assert m["value"] < 0
        assert m["status"] == "WATCH"

    def test_missing_live_gate_grades_na(self, s3):
        doc = _fixture()
        doc["counterfactual"]["live_gate"] = {}
        _put_att(s3, doc)
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3),
                  "scanner_feed_counterfactual")
        # Without the live-gate baseline there is no like-for-like delta —
        # honest N/A, not a semantics-shifting fallback value.
        assert m["status"] == "N/A-MISSING-INPUT"


class TestWindowedResolution:
    def test_reads_freshest_artifact_within_trailing_window(self, s3):
        # Artifact persisted 3 days before run_date still grades (config#1190).
        _put_att(s3, _fixture(), date="2026-07-01")
        tile = build_research_tile(BUCKET, RUN_DATE, s3_client=s3)
        m = _comp(tile, "attractiveness_ic")
        assert m["value"] == pytest.approx(0.062)
        assert "2026-07-01" in m["source_path"]
