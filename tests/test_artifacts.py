"""Tests for grading/artifacts.py — S3 artifact reader + gaps report."""

import json

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from grading.artifacts import (
    ARTIFACT_MAP,
    DEFAULT_ARTIFACT_MAX_AGE_DAYS,
    get_json_windowed,
    read_scorecard_inputs,
)

BUCKET = "alpha-engine-research"
RUN_DATE = "2026-06-06"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _put(s3, filename, data):
    s3.put_object(
        Bucket=BUCKET,
        Key=f"backtest/{RUN_DATE}/{filename}",
        Body=json.dumps(data).encode("utf-8"),
    )


def _put_metrics(s3, overall, status="ok"):
    body = {"run_date": RUN_DATE, "status": status, **overall}
    s3.put_object(
        Bucket=BUCKET,
        Key=f"backtest/{RUN_DATE}/metrics.json",
        Body=json.dumps(body).encode("utf-8"),
    )


class TestReadScorecardInputs:
    def test_empty_bucket_all_missing(self, s3):
        inputs, report = read_scorecard_inputs(BUCKET, RUN_DATE, s3_client=s3)
        assert inputs == {}
        # metrics.json + every mapped artifact missing.
        assert "metrics.json" in report.missing
        assert len(report.missing) == len(ARTIFACT_MAP) + 1
        assert report.read == []

    def test_reads_present_artifacts(self, s3):
        _put(s3, "macro_eval.json", {"status": "ok", "accuracy_lift": 2.0, "alpha_lift": 0.5})
        _put(s3, "exit_timing.json", {"status": "ok", "n_roundtrips": 10,
                                      "summary": {"avg_capture_ratio": 0.6}, "diagnosis": "exits_well_timed"})

        inputs, report = read_scorecard_inputs(BUCKET, RUN_DATE, s3_client=s3)
        assert inputs["macro_eval"]["accuracy_lift"] == 2.0
        assert inputs["exit_timing"]["diagnosis"] == "exits_well_timed"
        assert "macro_eval.json" in report.read
        assert "exit_timing.json" in report.read
        # The unmapped/absent ones land in missing.
        assert "shadow_book.json" in report.missing

    def test_veto_result_maps_from_veto_analysis_filename(self, s3):
        # The param is veto_result but the file is veto_analysis.json.
        _put(s3, "veto_analysis.json", {"status": "ok", "recommended_threshold": 0.65,
                                        "thresholds": [{"confidence": 0.65, "precision": 0.7}]})
        inputs, report = read_scorecard_inputs(BUCKET, RUN_DATE, s3_client=s3)
        assert "veto_result" in inputs
        assert inputs["veto_result"]["recommended_threshold"] == 0.65
        assert "veto_analysis.json" in report.read

    def test_calibration_and_excursion_map_from_portfolio_filenames(self, s3):
        _put(s3, "portfolio_calibration.json", {"status": "ok", "ece": 0.04})
        _put(s3, "portfolio_excursion.json", {"status": "ok", "mean_mfe_mae_ratio": 1.6})
        inputs, _ = read_scorecard_inputs(BUCKET, RUN_DATE, s3_client=s3)
        assert inputs["calibration_diagnostics"]["ece"] == 0.04
        assert inputs["excursion_summary"]["mean_mfe_mae_ratio"] == 1.6

    def test_signal_quality_reconstructed_from_metrics(self, s3):
        _put_metrics(s3, {"accuracy_10d": 0.58, "avg_alpha_10d": 1.5, "n_10d": 50})
        inputs, report = read_scorecard_inputs(BUCKET, RUN_DATE, s3_client=s3)
        sq = inputs["signal_quality"]
        assert sq["status"] == "ok"
        assert sq["overall"]["accuracy_10d"] == 0.58
        assert sq["overall"]["avg_alpha_10d"] == 1.5
        # report_card / run_date / status must NOT leak into overall.
        assert "run_date" not in sq["overall"]
        assert "report_card" not in sq["overall"]
        assert "metrics.json" in report.read

    def test_metrics_report_card_excluded_from_overall(self, s3):
        body = {"run_date": RUN_DATE, "status": "ok", "accuracy_10d": 0.5,
                "report_card": {"overall": {"grade": 70}}}
        s3.put_object(Bucket=BUCKET, Key=f"backtest/{RUN_DATE}/metrics.json",
                      Body=json.dumps(body).encode("utf-8"))
        inputs, _ = read_scorecard_inputs(BUCKET, RUN_DATE, s3_client=s3)
        assert inputs["signal_quality"]["overall"] == {"accuracy_10d": 0.5}

    def test_non_nosuchkey_error_raises(self, s3):
        # Reading from a bucket that doesn't exist → NoSuchBucket, not NoSuchKey.
        with pytest.raises(ClientError):
            read_scorecard_inputs("nonexistent-bucket-xyz", RUN_DATE, s3_client=s3)

    def test_report_as_dict_shape(self, s3):
        _put(s3, "macro_eval.json", {"status": "ok"})
        _, report = read_scorecard_inputs(BUCKET, RUN_DATE, s3_client=s3)
        d = report.as_dict()
        assert d["run_date"] == RUN_DATE
        assert d["bucket"] == BUCKET
        assert d["prefix"] == f"backtest/{RUN_DATE}"
        assert d["n_read"] == len(d["artifacts_read"])
        assert d["n_missing"] == len(d["artifacts_missing"])
        assert d["artifacts_read"] == sorted(d["artifacts_read"])


class TestGetJsonWindowed:
    """The artifact-resilience keystone (config#1190): grade off the freshest
    artifact within the trailing window, so a partial/retried/off-cycle Saturday
    run still grades instead of reading N/A. A corrupt mid-write is skipped."""

    TPL = "backtest/{date}/e2e_lift.json"

    def _put_on(self, s3, date, data, raw=False):
        body = data if raw else json.dumps(data).encode("utf-8")
        s3.put_object(Bucket=BUCKET, Key=f"backtest/{date}/e2e_lift.json", Body=body)

    def test_exact_date_age_zero(self, s3):
        self._put_on(s3, "2026-06-20", {"status": "ok"})
        doc, src_date, age, key = get_json_windowed(s3, BUCKET, self.TPL, "2026-06-20")
        assert doc == {"status": "ok"} and src_date == "2026-06-20" and age == 0
        assert key == "backtest/2026-06-20/e2e_lift.json"

    def test_finds_earlier_artifact_within_window(self, s3):
        # SF never ran on run_date 2026-06-20, but a partial run on 06-18 produced it.
        self._put_on(s3, "2026-06-18", {"status": "ok", "from": "partial"})
        doc, src_date, age, key = get_json_windowed(s3, BUCKET, self.TPL, "2026-06-20")
        assert doc["from"] == "partial" and src_date == "2026-06-18" and age == 2

    def test_outside_window_is_none(self, s3):
        # Older than the window → genuinely N/A (not silently graded stale).
        old = "2026-06-01"  # > DEFAULT_ARTIFACT_MAX_AGE_DAYS before run_date
        self._put_on(s3, old, {"status": "ok"})
        doc, src_date, age, key = get_json_windowed(s3, BUCKET, self.TPL, "2026-06-20")
        assert doc is None and src_date is None and key is None

    def test_freshest_wins_over_older(self, s3):
        self._put_on(s3, "2026-06-15", {"v": "older"})
        self._put_on(s3, "2026-06-19", {"v": "fresher"})
        doc, src_date, _, _ = get_json_windowed(s3, BUCKET, self.TPL, "2026-06-20")
        assert doc["v"] == "fresher" and src_date == "2026-06-19"

    def test_corrupt_candidate_skipped_for_older_good(self, s3):
        # A crashed mid-write leaves an empty file on the freshest date; the
        # resolver skips it and returns the last GOOD artifact.
        self._put_on(s3, "2026-06-19", b"", raw=True)        # corrupt/empty
        self._put_on(s3, "2026-06-17", {"v": "good"})
        doc, src_date, _, _ = get_json_windowed(s3, BUCKET, self.TPL, "2026-06-20")
        assert doc["v"] == "good" and src_date == "2026-06-17"

    def test_window_cap_respected(self, s3):
        self._put_on(s3, "2026-06-10", {"status": "ok"})
        doc, *_ = get_json_windowed(s3, BUCKET, self.TPL, "2026-06-20", max_age_days=3)
        assert doc is None  # 10 days back, window only 3
        doc2, *_ = get_json_windowed(s3, BUCKET, self.TPL, "2026-06-20", max_age_days=DEFAULT_ARTIFACT_MAX_AGE_DAYS)
        assert doc2 == {"status": "ok"}  # within default 10d

    def test_non_iso_run_date_falls_back_to_exact(self, s3):
        self._put_on(s3, "latest", {"status": "ok"})
        doc, src_date, age, _ = get_json_windowed(s3, BUCKET, "backtest/{date}/e2e_lift.json", "latest")
        assert doc == {"status": "ok"} and age == 0
