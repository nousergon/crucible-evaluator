"""Consumer-side contract test: research tile vs attractiveness_eval.json.

M0 contract discipline: ``backtest/{date}/attractiveness_eval.json`` is a
producer/consumer contract (producer: the backtester's attractiveness
evaluator; consumer: the research tile's attractiveness_ic /
attractiveness_trajectory_ic / scanner_feed_counterfactual components). The
canonical schema now lives in ``nousergon_lib.contracts`` (config#1861).

config#1861 rename window: the field ``mean_alpha_21d`` (v1) is renamed to
``mean_alpha`` (v2, horizon carried by top-level ``horizon_days``). This
consumer is deployed FIRST and must tolerate BOTH the v1 artifacts still in S3
and the v2 artifacts the producer starts emitting after its own cutover — so
this test parametrizes over both canonical fixtures (v1 + v2) and asserts the
tile reads the counterfactual prize correctly from EITHER. The v2 fixture also
round-trips through the lib schema (contracts.validate). The v1 fixture proves
the dual-tolerance fallback (drop after the ~2026-07-15 S3 window).

Forward-compat is part of the contract: the producer deploys independently, so
an ABSENT artifact (or status="insufficient_data") must grade an honest,
specific N/A — never a crash, never a silent omission.
"""

import json
from pathlib import Path

import boto3
import pytest
from moto import mock_aws
from nousergon_lib import contracts

from grading.tiles.research import build_research_tile

BUCKET = "alpha-engine-research"
RUN_DATE = "2026-07-04"
FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURES / "attractiveness_eval_v1.json"
FIXTURE_V2 = FIXTURES / "attractiveness_eval_v2.json"

# (fixture path, alpha field name for that schema version) for parametrization.
ALL_FIXTURES = [
    pytest.param(FIXTURE, 1, "mean_alpha_21d", id="v1"),
    pytest.param(FIXTURE_V2, 2, "mean_alpha", id="v2"),
]


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
    @pytest.mark.parametrize("path, version, alpha_field", ALL_FIXTURES)
    def test_fixture_carries_the_expected_shape(self, path, version, alpha_field):
        doc = json.loads(path.read_text())
        assert doc["schema_version"] == version
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
        # The counterfactual alpha field is the version-appropriate one:
        # {n,sector_balanced,capture_rate,mean_alpha_21d} for v1,
        # {n,sector_balanced,capture_rate,mean_alpha} for v2.
        for entry in doc["counterfactual"]["top_n"]:
            assert set(entry) == {"n", "sector_balanced", "capture_rate", alpha_field}
        assert set(doc["counterfactual"]["live_gate"]) == {
            "capture_rate", alpha_field, "n_survivors"}

    def test_v2_fixture_conforms_to_lib_schema(self):
        # The v2 fixture must round-trip through the canonical lib contract —
        # this is the schema crucible-evaluator validates in production once the
        # producer emits v2 (config#1861).
        v2 = json.loads(FIXTURE_V2.read_text())
        contracts.validate("attractiveness_eval", v2)  # raises on non-conformance


class TestDualToleranceCounterfactual:
    """The tile must read the counterfactual prize from BOTH schema versions.

    This is the crux of the consumer-tolerant-before-producer-cutover order:
    the v1 fixture exercises the ``mean_alpha_21d`` fallback and the v2 fixture
    the renamed ``mean_alpha`` — both must yield the same matched-breadth prize.
    """

    @pytest.mark.parametrize("path, version, alpha_field", ALL_FIXTURES)
    def test_prize_reads_from_either_schema_version(self, s3, path, version, alpha_field):
        _put_att(s3, json.loads(path.read_text()))
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3),
                  "scanner_feed_counterfactual")
        # live_gate n_survivors=27 → matched-breadth row is top-25 (non-balanced);
        # prize = 0.0121 − 0.0063, identical across v1/v2 (only the field renamed).
        assert m["value"] == pytest.approx(0.0121 - 0.0063)
        assert m["status"] == "GREEN"


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
