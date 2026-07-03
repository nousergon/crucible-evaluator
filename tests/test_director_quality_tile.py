"""Tests for grading/tiles/director_quality.py — Tile 9 (config#1674)."""

import json

import boto3
import pytest
from moto import mock_aws

from grading.tiles.director_quality import RETRO_TREND_KEY, build_director_quality_tile

BUCKET = "alpha-engine-research"
RUN_DATE = "2026-06-27"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _comp(tile, name):
    return next(c for c in tile["components"] if c["name"] == name)


def _put_trend(s3, grades):
    s3.put_object(
        Bucket=BUCKET, Key=RETRO_TREND_KEY,
        Body=json.dumps({"updated": RUN_DATE, "grades": grades}).encode("utf-8"),
    )


def _grade(prior_run_date, grounding=80, calibration=55, actionability=70, **extra):
    return {
        "retro_run_date": RUN_DATE,
        "prior_run_date": prior_run_date,
        "grounding": grounding,
        "calibration": calibration,
        "actionability": actionability,
        "notes": "Flagged risks mostly held.",
        **extra,
    }


class TestDirectorQuality:
    def test_all_components_na_when_no_trend(self, s3):
        tile = build_director_quality_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert tile["module"] == "director_quality"
        assert tile["n_components"] == 3
        statuses = {c["status"] for c in tile["components"]}
        assert statuses == {"N/A-MISSING-INPUT"}
        # only supporting components N/A → tile stays GREEN (WATCH-only class,
        # never a critical N/A) per module_agg.module_status.
        assert tile["status"] == "GREEN"
        assert tile["numeric_grade"] is None
        for name in ("director_grounding", "director_calibration", "director_actionability"):
            c = _comp(tile, name)
            assert "retro_trend.json" in c["status_reason"]

    def test_all_components_na_when_empty_grades_list(self, s3):
        _put_trend(s3, [])
        tile = build_director_quality_tile(BUCKET, RUN_DATE, s3_client=s3)
        statuses = {c["status"] for c in tile["components"]}
        assert statuses == {"N/A-MISSING-INPUT"}

    def test_graded_case_three_components(self, s3):
        _put_trend(s3, [_grade("2026-06-20", grounding=80, calibration=55, actionability=70)])
        tile = build_director_quality_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert tile["n_components"] == 3

        g = _comp(tile, "director_grounding")
        assert g["value"] == 80
        assert g["n_samples"] == 1
        assert g["criticality"] == "supporting"
        assert g["status"] == "GREEN"  # 80 >= target 75

        c = _comp(tile, "director_calibration")
        assert c["value"] == 55
        assert c["status"] == "WATCH"  # between red_line 40 and target 75

        a = _comp(tile, "director_actionability")
        assert a["value"] == 70
        assert a["status"] == "WATCH"

    def test_red_line_grades_red(self, s3):
        _put_trend(s3, [_grade("2026-06-20", grounding=10, calibration=10, actionability=10)])
        tile = build_director_quality_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert _comp(tile, "director_grounding")["status"] == "RED"
        # supporting RED components never cascade this tile below WATCH.
        assert tile["status"] == "WATCH"

    def test_most_recent_entry_selected_when_multiple_grades(self, s3):
        _put_trend(s3, [
            _grade("2026-06-06", grounding=20, calibration=20, actionability=20),
            _grade("2026-06-13", grounding=50, calibration=50, actionability=50),
            _grade("2026-06-20", grounding=90, calibration=90, actionability=90),
        ])
        tile = build_director_quality_tile(BUCKET, RUN_DATE, s3_client=s3)
        # grades[-1] (most recent prior_run_date) wins, not the first or a max.
        assert _comp(tile, "director_grounding")["value"] == 90
        assert _comp(tile, "director_calibration")["value"] == 90
        assert _comp(tile, "director_actionability")["value"] == 90
        assert "2026-06-20" in _comp(tile, "director_grounding")["status_reason"]

    def test_judge_model_included_in_reason_when_present(self, s3):
        """config#1673's judge_model/resolved_model fields, when present, stay
        visible on the card so a judge-model regime break is legible."""
        _put_trend(s3, [_grade("2026-06-20", judge_model="claude-opus-4-8", resolved_model="claude-opus-4-8-20260601")])
        tile = build_director_quality_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert "claude-opus-4-8" in _comp(tile, "director_grounding")["status_reason"]

    def test_judge_model_absent_does_not_crash(self, s3):
        """Older rows (or a repo where config#1673 hasn't landed) lack
        judge_model entirely — must degrade gracefully, never raise."""
        row = _grade("2026-06-20")
        assert "judge_model" not in row
        _put_trend(s3, [row])
        tile = build_director_quality_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert _comp(tile, "director_grounding")["value"] == 80

    def test_source_path(self, s3):
        _put_trend(s3, [_grade("2026-06-20")])
        tile = build_director_quality_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert _comp(tile, "director_grounding")["source_path"] == f"s3://{BUCKET}/{RETRO_TREND_KEY}"
