"""Tests for grading/artifacts.py — S3 artifact reader + gaps report."""

import json

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from grading.artifacts import ARTIFACT_MAP, read_scorecard_inputs

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
