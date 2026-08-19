"""`inference_coverage` grades the set the run was ASKED to score.

alpha-engine-config-I7648. It divided by `n_universe` — the research
population — and the predictor stopped scoring that population at the champion
cutover. Measured live 2026-08-18: all 24 of the day's predictions carried a
`watchlist_source` of `attractiveness_top_20` (20) or `held` (4), while
signals.json declared `universe: 903`, so this graded 23/903 = 2.5% against a
95% target it could not reach — a CRITICAL component permanently RED, and one
of three stated reasons live sizing was being held de-risked.
"""
from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws

from grading.tiles.predictor import LATEST_KEY, build_predictor_tile

BUCKET = "alpha-engine-research"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _components(s3, latest: dict) -> dict:
    """Build the tile with only latest.json present; components by name.

    Every other artifact is absent on purpose — those components grade their
    own honest N/A and are not what this module is about.
    """
    s3.put_object(Bucket=BUCKET, Key=LATEST_KEY, Body=json.dumps(latest).encode())
    tile = build_predictor_tile(BUCKET, "2026-08-18", s3_client=s3)
    return {c["name"]: c for c in tile["components"]}


#: The live 2026-08-18 metrics body, after the producer change
#: (crucible-predictor `intended_scoring_set`).
LIVE = {
    "n_universe": 903,
    "n_universe_covered": 23,
    "n_intended": 24,
    "n_intended_covered": 23,
    "intended_source": ["attractiveness_top_20", "held"],
    "n_predictions_today": 24,
}


def test_the_live_shape_grades_near_100_not_2_percent(s3):
    c = _components(s3, LIVE)["inference_coverage"]
    assert c["value"] == 23 / 24
    assert c["value"] > 0.95
    detail = c["status_reason"]
    assert "asked to score" in detail
    assert "attractiveness_top_20" in detail


def test_funnel_width_is_reported_separately_and_is_supporting(s3):
    """The narrowing is real and must stay visible — just not inside the
    coverage number, where it made both facts unreadable at once."""
    c = _components(s3, LIVE)["inference_funnel_width"]
    assert c["value"] == 24 / 903
    assert c["criticality"] == "supporting"


def test_a_genuine_miss_still_grades_red(s3):
    """The fix must not make the metric unable to fire."""
    c = _components(s3, {**LIVE, "n_intended": 24, "n_intended_covered": 5})["inference_coverage"]
    assert c["value"] < 0.8


def test_absent_n_intended_grades_na_and_never_falls_back(s3):
    """A producer that has not been redeployed must not look like a healthy
    run. It also must not look like the old 2.5%."""
    stale = {k: v for k, v in LIVE.items() if not k.startswith("n_intended")}
    stale.pop("intended_source", None)
    c = _components(s3, stale)["inference_coverage"]
    assert c["value"] is None
    detail = c["status_reason"]
    assert "n_intended" in detail
    # It says WHY it refuses the other denominator, so the next reader does not
    # "fix" the N/A by reinstating it.
    assert "903" in detail and "champion cutover" in detail


def test_zero_intended_grades_na_not_a_division_error(s3):
    c = _components(s3, {**LIVE, "n_intended": 0})["inference_coverage"]
    assert c["value"] is None


def test_funnel_width_is_na_without_both_halves(s3):
    partial = {k: v for k, v in LIVE.items() if k != "n_universe"}
    c = _components(s3, partial)["inference_funnel_width"]
    assert c["value"] is None
