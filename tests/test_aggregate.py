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
from grading.freshness_preflight import MissingInputArtifactError, StaleInputArtifactError

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


def _put_eod_pnl(s3, *, n_rows=80, month="01"):
    """Seed a minimal-but-sufficient trades/eod_pnl.csv (80 rows clears the
    dsr/psr n_floor=60) so portfolio_outcome's build path reaches the dsr
    component instead of short-circuiting the whole tile to N/A on a
    missing CSV. Dates default to Jan 2026 so callers grading a LATER
    RUN_DATE (e.g. the freshness-preflight fixtures below) can still exercise
    the dsr path without also having to satisfy the freshness gate via this
    same CSV — use ``_put_eod_pnl_fresh`` when the gate itself is under test."""
    s3.put_object(
        Bucket=BUCKET, Key="trades/eod_pnl.csv",
        Body=(
            "date,portfolio_nav,daily_return_pct,spy_return_pct,"
            "daily_alpha_pct,positions_snapshot,created_at\n"
            + "\n".join(
                f"2026-{month}-{(i % 28) + 1:02d},{1_000_000 * (1 + 0.001 * i):.2f},"
                f"0.12,0.04,0.08,{{}},2026-01-01T00:00:00+00:00"
                for i in range(n_rows)
            )
            + "\n"
        ).encode("utf-8"),
    )


def _put_eod_pnl_fresh(s3, run_date=RUN_DATE, *, n_rows=80):
    """eod_pnl.csv whose freshest row IS run_date — satisfies both the dsr
    n_floor AND the freshness preflight's eod_reconcile_pnl check."""
    import datetime as _dt

    end = _dt.date.fromisoformat(run_date)
    rows = [
        (end - _dt.timedelta(days=i)).isoformat()
        for i in range(n_rows)
    ]
    s3.put_object(
        Bucket=BUCKET, Key="trades/eod_pnl.csv",
        Body=(
            "date,portfolio_nav,daily_return_pct,spy_return_pct,"
            "daily_alpha_pct,positions_snapshot,created_at\n"
            + "\n".join(
                f"{d},{1_000_000 * (1 + 0.001 * i):.2f},0.12,0.04,0.08,{{}},"
                "2026-01-01T00:00:00+00:00"
                for i, d in enumerate(rows)
            )
            + "\n"
        ).encode("utf-8"),
    )


def _seed_freshness_inputs(s3, run_date=RUN_DATE):
    """Seed every artifact `grading.freshness_preflight.assert_input_freshness`
    hard-requires, all dated exactly `run_date` (the trivially-fresh case) —
    the baseline every test that isn't itself exercising the freshness gate
    should call, so it can focus on the behavior it actually names."""
    s3.put_object(
        Bucket=BUCKET, Key="predictor/weights/meta/manifest.json",
        Body=json.dumps({
            "training_date": run_date,
            "walk_forward": {"n_folds": 16},
            "meta_model_oos_ic_cpcv": {"status": "ok", "n_combos": 4, "mean_ic": 0.1, "frac_positive": 0.75, "ics": [0.1, 0.1, 0.1, 0.1]},
        }).encode("utf-8"),
    )
    s3.put_object(
        Bucket=BUCKET, Key=f"signals/{run_date}/signals.json",
        Body=json.dumps({"market_regime": "neutral"}).encode("utf-8"),
    )
    _put_eod_pnl_fresh(s3, run_date)


def _seed_full(s3):
    """Seed a realistic full artifact set (mirrors the scorecard full-data test)."""
    _seed_freshness_inputs(s3)
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
    def test_empty_bucket_hard_fails_freshness_preflight(self, s3):
        # alpha-engine-config#3058: an empty bucket has NO declared input
        # artifacts at all — this is a MissingInputArtifactError (hard fail),
        # not the old "insufficient_data" graded card. The preflight runs
        # BEFORE any tile/scorecard computation.
        with pytest.raises(MissingInputArtifactError):
            build_report_card(BUCKET, RUN_DATE, s3_client=s3)

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
        # Provenance also records the freshness preflight that ran first.
        assert card["_provenance"]["freshness_preflight"]["run_date"] == RUN_DATE
        checked_ids = {c["artifact_id"] for c in card["_provenance"]["freshness_preflight"]["checks"]}
        assert "e2e_lift_json" in checked_ids
        assert "eod_reconcile_pnl" in checked_ids

    def test_partial_artifacts_grade_na_loudly(self, s3):
        # Only macro present (+ the hard-required freshness inputs) → research
        # partial, others N/A; the gap is in provenance. This is testing the
        # TILES' own graceful-N/A posture for artifacts the preflight does NOT
        # hard-gate (shadow_book.json etc.) — distinct from the freshness gate.
        _seed_freshness_inputs(s3)
        _put(s3, "metrics.json", {"run_date": RUN_DATE, "status": "ok"})
        _put(s3, "e2e_lift.json", {"status": "ok"})
        _put(s3, "macro_eval.json", {"status": "ok", "accuracy_lift": 2.0, "alpha_lift": 0.5})
        card = build_report_card(BUCKET, RUN_DATE, s3_client=s3)
        assert card["predictor"]["grade"] is None
        assert "shadow_book.json" in card["_provenance"]["artifacts"]["artifacts_missing"]

    def test_v2_tiles_attached(self, s3):
        # RC v2 MetricRecord tiles are nested under "tiles" (portfolio + predictor).
        _seed_full(s3)
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


class TestCumulativeTrialCountWiring:
    """config#2454: build_report_card reads the shared cumulative
    trial-count counter and threads it into portfolio_outcome's dsr
    metric via n_trials — best-effort, never blocks the card build."""

    def test_dsr_na_when_counter_artifact_absent(self, s3):
        """No cumulative_trial_count.json yet (counter not seeded/backfilled)
        → n_trials stays None → dsr keeps reporting N/A, exactly the
        pre-#2454 behavior. Never an error."""
        _seed_freshness_inputs(s3)
        _put(s3, "metrics.json", {"run_date": RUN_DATE, "status": "ok"})
        _put(s3, "e2e_lift.json", {"status": "ok"})
        card = build_report_card(BUCKET, RUN_DATE, s3_client=s3)
        dsr = next(
            c for c in card["tiles"]["portfolio_outcome"]["components"]
            if c["name"] == "dsr"
        )
        assert dsr["status"] == "N/A-NOT-IMPL"

    def test_dsr_computed_when_counter_seeded(self, s3):
        """A seeded cumulative_trial_count.json flows through to a real
        (non-N/A) dsr value."""
        from nousergon_lib.quant.stats.trial_accumulator import (
            increment_trial_count,
        )
        increment_trial_count(
            "gamma_sweep", 250, RUN_DATE, bucket=BUCKET, s3_client=s3,
        )
        _seed_freshness_inputs(s3)
        _put(s3, "metrics.json", {"run_date": RUN_DATE, "status": "ok"})
        _put(s3, "e2e_lift.json", {"status": "ok"})
        card = build_report_card(BUCKET, RUN_DATE, s3_client=s3)
        dsr = next(
            c for c in card["tiles"]["portfolio_outcome"]["components"]
            if c["name"] == "dsr"
        )
        assert dsr["status"] != "N/A-NOT-IMPL"
        assert dsr["value"] is not None

    def test_counter_read_failure_degrades_gracefully(self, s3, monkeypatch):
        """A broken/corrupt counter artifact must not fail the whole report-
        card build — dsr just falls back to N/A this cycle."""
        _seed_freshness_inputs(s3)
        _put(s3, "metrics.json", {"run_date": RUN_DATE, "status": "ok"})
        _put(s3, "e2e_lift.json", {"status": "ok"})
        s3.put_object(
            Bucket=BUCKET, Key="backtest/cumulative_trial_count.json",
            Body=b"{not valid json",
        )
        card = build_report_card(BUCKET, RUN_DATE, s3_client=s3)
        assert card is not None
        dsr = next(
            c for c in card["tiles"]["portfolio_outcome"]["components"]
            if c["name"] == "dsr"
        )
        assert dsr["status"] == "N/A-NOT-IMPL"


class TestWriteReportCard:
    def test_writes_latest_only_by_default(self, s3):
        # config-I2556: snapshot defaults False — an absent flag refreshes the
        # standing `latest` pointer only; production callers (the Saturday
        # advisory-child freeze, the Sunday ModelZoo re-grade) both pass the
        # flag explicitly (nousergon-data PR #832, merged 2026-07-14).
        _seed_full(s3)
        card = build_report_card(BUCKET, RUN_DATE, s3_client=s3)
        written = write_report_card(BUCKET, RUN_DATE, card, s3_client=s3)
        assert written["dated_key"] is None
        assert written["latest_key"] == "evaluator/latest/report_card.json"
        obj = s3.get_object(Bucket=BUCKET, Key=written["latest_key"])
        roundtrip = json.loads(obj["Body"].read())
        assert roundtrip["status"] == card["status"]
        assert roundtrip["overall"]["letter"] == card["overall"]["letter"]

    def test_writes_latest_and_dated_when_snapshot_true(self, s3):
        _seed_full(s3)
        card = build_report_card(BUCKET, RUN_DATE, s3_client=s3)
        written = write_report_card(BUCKET, RUN_DATE, card, s3_client=s3, snapshot=True)
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

    def test_snapshot_true_reasserts_freshness_and_raises_if_now_stale(self, s3):
        # alpha-engine-config#3058: the snapshot step re-asserts freshness
        # independent of build_report_card's own gate — belt-and-braces so a
        # frozen weekly record (the worst-case artifact per the issue) can
        # never be produced from a stale state, even if a future/alternate
        # caller builds a card once and snapshots it later. Simulate that by
        # building a valid card, then rot the e2e_lift.json instance so the
        # SAME run_date now reads stale at snapshot time.
        _seed_full(s3)
        card = build_report_card(BUCKET, RUN_DATE, s3_client=s3)
        s3.delete_object(Bucket=BUCKET, Key=f"backtest/{RUN_DATE}/e2e_lift.json")
        with pytest.raises(MissingInputArtifactError):
            write_report_card(BUCKET, RUN_DATE, card, s3_client=s3, snapshot=True)
        # And the moving `latest` pointer must not have been written either —
        # the gate runs BEFORE either put_object.
        listing = s3.list_objects_v2(Bucket=BUCKET, Prefix="evaluator/")
        assert listing.get("KeyCount", 0) == 0

    def test_snapshot_false_does_not_reassert_freshness(self, s3):
        # A bare latest-only refresh never freezes a weekly record, so it
        # stays governed by build_report_card's own preflight only — no
        # redundant re-check here.
        _seed_full(s3)
        card = build_report_card(BUCKET, RUN_DATE, s3_client=s3)
        s3.delete_object(Bucket=BUCKET, Key=f"backtest/{RUN_DATE}/e2e_lift.json")
        written = write_report_card(BUCKET, RUN_DATE, card, s3_client=s3, snapshot=False)
        assert written["latest_key"] == "evaluator/latest/report_card.json"

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
