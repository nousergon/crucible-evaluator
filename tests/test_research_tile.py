"""Tests for grading/tiles/research.py — Tile 1, precision-from-e2e sourcing."""

import json

import boto3
import pytest
from moto import mock_aws

from grading.tiles.research import build_research_tile

BUCKET = "alpha-engine-research"
RUN_DATE = "2026-06-07"


def _clf(precision, tp, fp, fn=0, tn=0):
    return {"precision": precision, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": tp + fp + fn + tn}


# Realistic classification blocks (with fn/tn) so the base-rate edge is
# meaningful: edge = precision − base_rate, base_rate = (tp+fn)/n_pop.
_E2E = {
    "status": "ok",
    # precision 0.50, base 200/5000=0.04 → edge +0.46
    "scanner_lift": {"lift": -0.0016, "n_passing": 320,
                     "classification": _clf(0.50, 60, 60, fn=140, tn=4740)},
    "team_lift": [
        {"team_id": "tech", "lift": 0.01, "classification": _clf(0.6, 18, 12, fn=12, tn=58)},
        {"team_id": "health", "lift": 0.0, "classification": _clf(0.5, 10, 10, fn=10, tn=70)},
    ],
    # precision 0.55, base 58/200=0.29 → edge +0.26
    "cio_lift": {"lift": -0.0067, "n_advance": 69,
                 "classification": _clf(0.55, 38, 31, fn=20, tn=111)},
    "cio_vs_ranking": {"lift": 0.0003},
}


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _put(s3, name, data):
    s3.put_object(Bucket=BUCKET, Key=f"backtest/{RUN_DATE}/{name}", Body=json.dumps(data).encode())


def _comp(tile, name):
    return next(c for c in tile["components"] if c["name"] == name)


class TestMissingArtifacts:
    def test_all_missing_inputs_loud(self, s3):
        tile = build_research_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert tile["module"] == "research"
        scanner = _comp(tile, "scanner")
        assert scanner["status"] == "N/A-MISSING-INPUT"
        # critical components N/A → tile WATCH (never false GREEN).
        assert tile["status"] == "WATCH"


class TestPrecisionComponents:
    def test_scanner_edge_over_base_rate(self, s3):
        _put(s3, "e2e_lift.json", _E2E)
        tile = build_research_tile(BUCKET, RUN_DATE, s3_client=s3)
        scanner = _comp(tile, "scanner")
        # edge = precision 0.50 − base_rate (200/5000=0.04) = +0.46
        assert scanner["value"] == pytest.approx(0.50 - 0.04)
        assert scanner["ci_method"] == "wilson"
        assert scanner["n_samples"] == 120  # tp+fp (selected)
        assert "base-rate" in scanner["status_reason"]
        assert "return-lift" in scanner["status_reason"]

    def test_sector_teams_pooled_edge(self, s3):
        _put(s3, "e2e_lift.json", _E2E)
        tile = build_research_tile(BUCKET, RUN_DATE, s3_client=s3)
        teams = _comp(tile, "sector_teams_avg")
        # pooled tp=28 fp=22 fn=22 tn=128 → precision 28/50=0.56,
        # base_rate (28+22)/200=0.25 → edge +0.31
        assert teams["value"] == pytest.approx(28 / 50 - 50 / 200)
        assert teams["n_samples"] == 50
        assert teams["criticality"] == "critical"

    def test_cio_edge(self, s3):
        _put(s3, "e2e_lift.json", _E2E)
        tile = build_research_tile(BUCKET, RUN_DATE, s3_client=s3)
        cio = _comp(tile, "cio")
        # precision 0.55 − base_rate (58/200=0.29) = +0.26
        assert cio["value"] == pytest.approx(0.55 - 0.29)
        assert "vs-ranking-lift" in cio["status_reason"]


class TestCompositeScoring:
    def test_monotonic_green(self, s3):
        _put(s3, "e2e_lift.json", _E2E)
        _put(s3, "score_calibration.json", {"status": "ok", "monotonic": True, "beat_spy_pct": 0.56, "n": 200})
        tile = build_research_tile(BUCKET, RUN_DATE, s3_client=s3)
        cs = _comp(tile, "composite_scoring")
        assert cs["status"] == "GREEN"
        assert "monotonic=True" in cs["status_reason"]

    def test_non_monotonic_red(self, s3):
        _put(s3, "e2e_lift.json", _E2E)
        _put(s3, "score_calibration.json", {"status": "ok", "monotonic": False, "beat_spy_pct": 0.48, "n": 200})
        tile = build_research_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert _comp(tile, "composite_scoring")["status"] == "RED"

    def test_absent_missing_input(self, s3):
        _put(s3, "e2e_lift.json", _E2E)
        tile = build_research_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert _comp(tile, "composite_scoring")["status"] == "N/A-MISSING-INPUT"


class TestMacroAndCalibration:
    def test_macro_accuracy_lift(self, s3):
        _put(s3, "e2e_lift.json", _E2E)
        _put(s3, "macro_eval.json", {"status": "ok", "accuracy_lift": 2.0, "assessment": "helpful", "n_evaluated": 40})
        tile = build_research_tile(BUCKET, RUN_DATE, s3_client=s3)
        macro = _comp(tile, "macro_agent")
        assert macro["value"] == pytest.approx(2.0)
        assert macro["status"] == "GREEN"  # +2pp > target 0, N=40 > floor 20

    def test_calibration_ece_lower_is_better(self, s3):
        _put(s3, "e2e_lift.json", _E2E)
        _put(s3, "portfolio_calibration.json", {"status": "ok", "ece": 0.03, "n": 200})
        tile = build_research_tile(BUCKET, RUN_DATE, s3_client=s3)
        cal = _comp(tile, "calibration_diagnostics")
        assert cal["value"] == pytest.approx(0.03)
        assert cal["status"] == "GREEN"  # 0.03 < target 0.05


class TestAspirationalNA:
    def test_not_impl_components(self, s3):
        _put(s3, "e2e_lift.json", _E2E)
        tile = build_research_tile(BUCKET, RUN_DATE, s3_client=s3)
        for name in ("judge_rubric_pass_rate", "pillar_emit_coverage", "signal_volume_adequacy"):
            assert _comp(tile, name)["status"] == "N/A-NOT-IMPL"
