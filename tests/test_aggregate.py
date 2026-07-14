"""Tests for grading/aggregate.py — build/write report card + parity compare."""

import json

import boto3
import pytest
from moto import mock_aws

from grading.aggregate import (
    LATEST_REPORT_CARD_KEY,
    build_report_card,
    compare_to_backtester,
    latest_report_card_key,
    report_card_key,
    write_report_card,
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


def _seed_full(s3):
    """Seed a realistic full artifact set (mirrors the scorecard full-data test)."""
    s3.put_object(
        Bucket=BUCKET, Key=f"backtest/{RUN_DATE}/metrics.json",
        Body=json.dumps({
            "run_date": RUN_DATE, "status": "ok",
            "accuracy_10d": 0.58, "avg_alpha_10d": 1.5, "n_10d": 50,
        }).encode("utf-8"),
    )
    _put(s3, "e2e_lift.json", {
        "status": "ok",
        "scanner_lift": {"lift": 1.2, "n_passing": 55, "n_universe": 900},
        "team_lift": [{"team_id": "technology", "lift": 2.5, "lift_vs_quant": 1.1, "n_picks": 12}],
        "cio_lift": {"lift": 1.5, "advance_avg": 2.1, "reject_avg": -0.3, "n_advance": 15, "n_reject": 20},
        "cio_vs_ranking": {"lift": 0.8, "cio_beats_ranking": True, "n_dates": 8,
                           "n_picks": 15, "avg_overlap": 0.6, "cio_avg": 2.1, "ranking_avg": 1.3},
    })
    _put(s3, "predictor_sizing.json", {
        "status": "ok", "overall_rank_ic": 0.06,
        "recent_positive_weeks": 6, "recent_total_weeks": 8, "sizing_lift": 0.3, "n_samples": 100,
    })
    _put(s3, "veto_analysis.json", {
        "status": "ok", "recommended_threshold": 0.65,
        "thresholds": [{"confidence": 0.65, "precision": 0.68, "lift": 13.0}],
    })
    _put(s3, "veto_value.json", {"net_value": 420.0})
    _put(s3, "trigger_scorecard.json", {
        "status": "ok",
        "triggers": [{"trigger": "pullback", "n_trades": 20, "avg_slippage_vs_signal": -0.3, "win_rate_vs_spy": 0.55}],
        "summary": {"total_entries": 35, "avg_slippage_vs_signal": -0.4, "win_rate_vs_spy": 0.57, "avg_realized_alpha": 1.2},
    })
    _put(s3, "shadow_book.json", {
        "status": "ok", "n_blocked": 12, "n_traded": 35, "guard_lift": 1.5,
        "blocked_beat_spy_pct": 0.33, "assessment": "appropriate",
    })
    _put(s3, "exit_timing.json", {
        "status": "ok", "n_roundtrips": 28,
        "summary": {"avg_capture_ratio": 0.62, "avg_realized_return": 1.8},
        "diagnosis": "exits_could_improve",
    })


class TestBuildReportCard:
    def test_empty_bucket_insufficient_with_provenance(self, s3):
        card = build_report_card(BUCKET, RUN_DATE, s3_client=s3)
        assert card["status"] == "insufficient_data"
        prov = card["_provenance"]
        assert prov["run_date"] == RUN_DATE
        assert prov["artifacts"]["n_read"] == 0
        assert prov["artifacts"]["n_missing"] > 0
        assert "grader_source" in prov

    def test_full_artifacts_produce_ok_status(self, s3):
        _seed_full(s3)
        card = build_report_card(BUCKET, RUN_DATE, s3_client=s3)
        assert card["status"] == "ok"
        assert card["overall"]["grade"] is not None
        assert card["research"]["grade"] is not None
        assert card["predictor"]["grade"] is not None
        assert card["executor"]["grade"] is not None
        # Provenance records exactly what was read.
        assert "e2e_lift.json" in card["_provenance"]["artifacts"]["artifacts_read"]

    def test_partial_artifacts_grade_na_loudly(self, s3):
        # Only macro present → research partial, others N/A; the gap is in provenance.
        _put(s3, "macro_eval.json", {"status": "ok", "accuracy_lift": 2.0, "alpha_lift": 0.5})
        card = build_report_card(BUCKET, RUN_DATE, s3_client=s3)
        assert card["predictor"]["grade"] is None
        assert "shadow_book.json" in card["_provenance"]["artifacts"]["artifacts_missing"]

    def test_v2_tiles_attached(self, s3):
        # RC v2 MetricRecord tiles are nested under "tiles" (portfolio + predictor).
        card = build_report_card(BUCKET, RUN_DATE, s3_client=s3)
        assert set(card["tiles"]) == {
            "portfolio_outcome", "predictor", "research", "executor",
            "backtester", "substrate", "agent", "behavioral", "director_quality",
        }
        for tile in card["tiles"].values():
            assert "status" in tile and "components" in tile
        # Unified v2 overall status rolls up the tiles.
        assert card["tiles_overall_status"] in (
            "GREEN", "WATCH", "RED", "N/A-NOT-RUN",
        )

    def test_tiles_carry_per_tile_freshness_stamps(self, s3):
        # config-I2556: every tile carries as_of + source_artifact_dates.
        _seed_full(s3)
        card = build_report_card(BUCKET, RUN_DATE, s3_client=s3)
        for name, tile in card["tiles"].items():
            assert "as_of" in tile and tile["as_of"], f"{name} missing as_of"
            assert "source_artifact_dates" in tile
            assert isinstance(tile["source_artifact_dates"], list)
        # research reads dated backtest/{RUN_DATE}/*.json artifacts seeded
        # above → its source_artifact_dates should surface that real date.
        assert RUN_DATE in card["tiles"]["research"]["source_artifact_dates"]


class TestWriteReportCard:
    def test_writes_latest_and_dated_by_default(self, s3):
        # config-I2556 staged back-compat default: snapshot defaults True, so
        # today's behavior (dated card on every non-dry write) is preserved,
        # AND the new standing `latest` pointer is now also written.
        _seed_full(s3)
        card = build_report_card(BUCKET, RUN_DATE, s3_client=s3)
        written = write_report_card(BUCKET, RUN_DATE, card, s3_client=s3)
        assert written["dated_key"] == f"evaluator/{RUN_DATE}/report_card.json"
        assert written["latest_key"] == "evaluator/latest/report_card.json"
        # Does NOT clobber the backtester's grading.json namespace.
        assert written["dated_key"] != f"backtest/{RUN_DATE}/grading.json"
        for key in (written["dated_key"], written["latest_key"]):
            obj = s3.get_object(Bucket=BUCKET, Key=key)
            roundtrip = json.loads(obj["Body"].read())
            assert roundtrip["status"] == card["status"]
            assert roundtrip["overall"]["letter"] == card["overall"]["letter"]

    def test_snapshot_false_writes_latest_only(self, s3):
        _seed_full(s3)
        card = build_report_card(BUCKET, RUN_DATE, s3_client=s3)
        written = write_report_card(BUCKET, RUN_DATE, card, s3_client=s3, snapshot=False)
        assert written["dated_key"] is None
        assert written["latest_key"] == "evaluator/latest/report_card.json"
        # latest was written...
        s3.get_object(Bucket=BUCKET, Key=written["latest_key"])
        # ...but the dated weekly key was NOT.
        listing = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"evaluator/{RUN_DATE}/")
        assert listing.get("KeyCount", 0) == 0

    def test_report_card_key_helper(self):
        assert report_card_key("2026-01-02") == "evaluator/2026-01-02/report_card.json"

    def test_latest_report_card_key_helper(self):
        assert latest_report_card_key() == "evaluator/latest/report_card.json"
        assert LATEST_REPORT_CARD_KEY == "evaluator/latest/report_card.json"


class TestCompareToBacktester:
    def test_no_backtester_grading(self, s3):
        _seed_full(s3)
        card = build_report_card(BUCKET, RUN_DATE, s3_client=s3)
        result = compare_to_backtester(BUCKET, RUN_DATE, card, s3_client=s3)
        assert result["status"] == "no_backtester_grading"
        assert result["mismatches"] == {}

    def test_identical_grading_no_mismatch(self, s3):
        _seed_full(s3)
        card = build_report_card(BUCKET, RUN_DATE, s3_client=s3)
        # Seed the backtester grading.json as a copy of the evaluator card
        # (minus provenance) → parity must be clean.
        bt = {k: v for k, v in card.items() if k != "_provenance"}
        s3.put_object(Bucket=BUCKET, Key=f"backtest/{RUN_DATE}/grading.json",
                      Body=json.dumps(bt).encode("utf-8"))
        result = compare_to_backtester(BUCKET, RUN_DATE, card, s3_client=s3)
        assert result["status"] == "compared"
        assert result["n_mismatch"] == 0
        assert result["mismatches"] == {}

    def test_differing_overall_letter_flagged(self, s3):
        _seed_full(s3)
        card = build_report_card(BUCKET, RUN_DATE, s3_client=s3)
        bt = {k: v for k, v in card.items() if k != "_provenance"}
        bt["overall"] = {"grade": 10.0, "letter": "F"}  # force a disagreement
        s3.put_object(Bucket=BUCKET, Key=f"backtest/{RUN_DATE}/grading.json",
                      Body=json.dumps(bt).encode("utf-8"))
        result = compare_to_backtester(BUCKET, RUN_DATE, card, s3_client=s3)
        assert result["n_mismatch"] >= 1
        assert "overall" in result["mismatches"]
        assert result["mismatches"]["overall"]["backtester"] == "F"
