"""Tests for grading/tiles/substrate.py + agent.py — Tiles 5 & 6."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

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

    def test_sf_success_na_when_no_arns_discoverable(self, s3, monkeypatch):
        # moto has no state machines → no ARNs discoverable → N/A-NOT-IMPL.
        monkeypatch.delenv("EVALUATOR_SF_ARNS", raising=False)
        tile = build_substrate_tile(BUCKET, s3_client=s3, sfn_client=boto3.client("stepfunctions", region_name="us-east-1"))
        sf = _comp(tile, "sf_success_rate_4w")
        assert sf["status"] == "N/A-MISSING-INPUT"
        assert "discoverable" in sf["status_reason"]

    def test_tile_has_ten_components(self, s3):
        # 10 = the original 9 + the new unattended_first_pass_rate (config#1059).
        tile = build_substrate_tile(BUCKET, s3_client=s3)
        assert tile["n_components"] == 10
        assert tile["module"] == "substrate"

    def test_tile_watch_when_only_freshness_real(self, s3):
        # price_cache fresh GREEN but critical sf/data_quality/schema N/A-NOT-IMPL → WATCH.
        s3.put_object(Bucket=BUCKET, Key=f"{PRICE_CACHE_PREFIX}A.parquet", Body=b"x")
        tile = build_substrate_tile(BUCKET, s3_client=s3, as_of=datetime.now(UTC))
        assert tile["status"] == "WATCH"


_ARNS = "arn:aws:states:us-east-1:0:stateMachine:alpha-engine-saturday-pipeline"


def _run(status, days_ago, now, role="weekly"):
    # Production runs carry a pipeline_role; role=None mimics an untracked
    # smoke/manual run (excluded from the rate). `days_ago` also keys the cycle
    # (one cycle = one SF + one start UTC-date), so distinct days = distinct
    # cycles, same day = same cycle (e.g. scheduled run + same-day recovery).
    return SimpleNamespace(status=status, start_utc=now - timedelta(days=days_ago), pipeline_role=role)


class TestSfSuccessRate:
    def test_cycle_rate_green_when_all_succeed(self, s3, monkeypatch):
        # 4 distinct days, each a clean scheduled run → 4/4 distinct cycles clean.
        now = datetime.now(UTC)
        monkeypatch.setenv("EVALUATOR_SF_ARNS", _ARNS)
        monkeypatch.setattr(
            "alpha_engine_lib.pipeline_status.list_recent_pipeline_runs",
            lambda arn, **kw: [_run("SUCCEEDED", 1, now), _run("SUCCEEDED", 8, now),
                               _run("SUCCEEDED", 15, now), _run("SUCCEEDED", 22, now)],
        )
        tile = build_substrate_tile(BUCKET, s3_client=s3, as_of=now, sfn_client=object())
        sf = _comp(tile, "sf_success_rate_4w")
        assert sf["value"] == pytest.approx(1.0)
        assert sf["n_samples"] == 4
        assert sf["status"] == "GREEN"
        # unattended: all 4 scheduled cycles succeeded with no recovery → 1.0
        unatt = _comp(tile, "unattended_first_pass_rate")
        assert unatt["value"] == pytest.approx(1.0)
        assert unatt["n_samples"] == 4

    def test_cycle_rate_red_when_failures_dominate(self, s3, monkeypatch):
        # 4 distinct cycles: 1 clean, 3 failed (no recovery) → cycle_rate 0.25.
        now = datetime.now(UTC)
        monkeypatch.setenv("EVALUATOR_SF_ARNS", _ARNS)
        monkeypatch.setattr(
            "alpha_engine_lib.pipeline_status.list_recent_pipeline_runs",
            lambda arn, **kw: [_run("SUCCEEDED", 1, now), _run("FAILED", 2, now),
                               _run("FAILED", 3, now), _run("TIMED_OUT", 4, now)],
        )
        tile = build_substrate_tile(BUCKET, s3_client=s3, as_of=now, sfn_client=object())
        sf = _comp(tile, "sf_success_rate_4w")
        assert sf["value"] == pytest.approx(0.25)
        assert sf["status"] == "RED"

    def test_recovered_cycle_counts_clean_but_not_unattended(self, s3, monkeypatch):
        # THE config#1059 false-RED fix: each cycle's scheduled run FAILS then a
        # same-day recovery SUCCEEDS. The OLD per-execution rate = 50% (a false
        # P0 RED). New cycle_rate = 100% (every cycle ultimately completed clean)
        # but unattended_first_pass_rate = 0% (each needed an operator). 3 cycles
        # on distinct days so N clears the floor and a status is gradeable.
        now = datetime.now(UTC)
        monkeypatch.setenv("EVALUATOR_SF_ARNS", _ARNS)
        monkeypatch.setattr(
            "alpha_engine_lib.pipeline_status.list_recent_pipeline_runs",
            lambda arn, **kw: [
                _run("FAILED", 1, now, role="weekly"), _run("SUCCEEDED", 1, now, role="recovery"),
                _run("FAILED", 8, now, role="weekly"), _run("SUCCEEDED", 8, now, role="recovery"),
                _run("FAILED", 15, now, role="weekly"), _run("SUCCEEDED", 15, now, role="recovery"),
            ],
        )
        tile = build_substrate_tile(BUCKET, s3_client=s3, as_of=now, sfn_client=object())
        sf = _comp(tile, "sf_success_rate_4w")
        assert sf["value"] == pytest.approx(1.0)  # 3/3 distinct cycles clean
        assert sf["n_samples"] == 3
        assert sf["status"] == "GREEN"  # no longer a false RED
        unatt = _comp(tile, "unattended_first_pass_rate")
        assert unatt["value"] == pytest.approx(0.0)  # every scheduled run needed recovery
        assert unatt["n_samples"] == 3
        assert unatt["status"] == "RED"  # 0% unattended — the genuine target, surfaced honestly

    def test_scheduled_success_no_recovery_is_unattended(self, s3, monkeypatch):
        # Scheduled run succeeds outright, no recovery → both metrics 1.0.
        now = datetime.now(UTC)
        monkeypatch.setenv("EVALUATOR_SF_ARNS", _ARNS)
        monkeypatch.setattr(
            "alpha_engine_lib.pipeline_status.list_recent_pipeline_runs",
            lambda arn, **kw: [_run("SUCCEEDED", 1, now, role="weekly")],
        )
        tile = build_substrate_tile(BUCKET, s3_client=s3, as_of=now, sfn_client=object())
        assert _comp(tile, "sf_success_rate_4w")["value"] == pytest.approx(1.0)
        assert _comp(tile, "unattended_first_pass_rate")["value"] == pytest.approx(1.0)

    def test_operator_only_cycle_excluded_from_unattended(self, s3, monkeypatch):
        # A cycle with ONLY an operator/recovery run (no scheduled run) counts
        # toward cycle_rate but NOT the unattended denominator.
        now = datetime.now(UTC)
        monkeypatch.setenv("EVALUATOR_SF_ARNS", _ARNS)
        monkeypatch.setattr(
            "alpha_engine_lib.pipeline_status.list_recent_pipeline_runs",
            lambda arn, **kw: [_run("SUCCEEDED", 1, now, role="recovery")],
        )
        tile = build_substrate_tile(BUCKET, s3_client=s3, as_of=now, sfn_client=object())
        assert _comp(tile, "sf_success_rate_4w")["value"] == pytest.approx(1.0)
        # no scheduled cycle → unattended is N/A-NOT-RUN, not a fabricated number
        assert _comp(tile, "unattended_first_pass_rate")["status"] == "N/A-NOT-RUN"

    def test_window_excludes_old_and_running(self, s3, monkeypatch):
        now = datetime.now(UTC)
        monkeypatch.setenv("EVALUATOR_SF_ARNS", _ARNS)
        monkeypatch.setattr(
            "alpha_engine_lib.pipeline_status.list_recent_pipeline_runs",
            lambda arn, **kw: [_run("SUCCEEDED", 1, now), _run("RUNNING", 1, now),
                               _run("FAILED", 40, now)],  # RUNNING excluded; 40d old excluded
        )
        tile = build_substrate_tile(BUCKET, s3_client=s3, as_of=now, sfn_client=object())
        sf = _comp(tile, "sf_success_rate_4w")
        # day-1 cycle: SUCCEEDED + RUNNING(non-terminal, dropped) → 1 clean cycle.
        assert sf["n_samples"] == 1
        assert sf["value"] == pytest.approx(1.0)

    def test_untracked_none_role_excluded(self, s3, monkeypatch):
        # role=None (smoke/manual) runs must NOT count toward production reliability.
        now = datetime.now(UTC)
        monkeypatch.setenv("EVALUATOR_SF_ARNS", _ARNS)
        monkeypatch.setattr(
            "alpha_engine_lib.pipeline_status.list_recent_pipeline_runs",
            lambda arn, **kw: [_run("SUCCEEDED", 1, now, role="weekly"),
                               _run("FAILED", 2, now, role=None),  # untracked smoke → excluded
                               _run("FAILED", 3, now, role=None)],
        )
        tile = build_substrate_tile(BUCKET, s3_client=s3, as_of=now, sfn_client=object())
        sf = _comp(tile, "sf_success_rate_4w")
        assert sf["n_samples"] == 1  # only the role-carrying cycle counts
        assert sf["value"] == pytest.approx(1.0)
        assert "DISTINCT" in sf["status_reason"]

    def test_no_terminal_runs_not_run(self, s3, monkeypatch):
        now = datetime.now(UTC)
        monkeypatch.setenv("EVALUATOR_SF_ARNS", _ARNS)
        monkeypatch.setattr(
            "alpha_engine_lib.pipeline_status.list_recent_pipeline_runs",
            lambda arn, **kw: [_run("RUNNING", 1, now)],
        )
        tile = build_substrate_tile(BUCKET, s3_client=s3, as_of=now, sfn_client=object())
        sf = _comp(tile, "sf_success_rate_4w")
        assert sf["status"] == "N/A-NOT-RUN"
        assert _comp(tile, "unattended_first_pass_rate")["status"] == "N/A-NOT-RUN"


RUN_DATE = "2026-06-20"


def _put_agent_quality(s3, body):
    import json
    s3.put_object(
        Bucket=BUCKET, Key=f"backtest/{RUN_DATE}/agent_quality.json",
        Body=json.dumps(body).encode(),
    )


class TestAgent:
    def test_all_components_na_when_no_producer(self, s3):
        tile = build_agent_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert tile["module"] == "agent"
        statuses = {c["status"] for c in tile["components"]}
        assert statuses <= {"N/A-NOT-IMPL", "N/A-MISSING-INPUT"}
        # critical N/A → tile WATCH (transparency, never a false GREEN).
        assert tile["status"] == "WATCH"
        assert tile["numeric_grade"] is None

    def test_producer_components_missing_input_name_producer(self, s3):
        """Absent agent_quality.json → the 5 wired components are MISSING-INPUT
        (not NOT-IMPL) and name the producer issue — the live contract anchor."""
        tile = build_agent_tile(BUCKET, RUN_DATE, s3_client=s3)
        for name in ("agent_validation_failure_rate", "cost_per_signal",
                     "retry_storm_count", "agent_latency_p95", "judge_rubric_distribution"):
            c = _comp(tile, name)
            assert c["status"] == "N/A-MISSING-INPUT", name
            assert "agent_quality.json" in c["status_reason"], name
            assert "config#1149" in c["status_reason"], name

    def test_stance_source_missing_input(self, s3):
        tile = build_agent_tile(BUCKET, RUN_DATE, s3_client=s3)
        ss = _comp(tile, "stance_source_provenance")
        assert ss["status"] == "N/A-MISSING-INPUT"
        assert "stance_source" in ss["status_reason"]

    def test_judge_kappa_not_impl_names_producer(self, s3):
        tile = build_agent_tile(BUCKET, RUN_DATE, s3_client=s3)
        jk = _comp(tile, "judge_calibration_cohen_kappa")
        assert jk["status"] == "N/A-NOT-IMPL"
        assert "decision_artifacts" in jk["status_reason"]

    def test_seven_components(self, s3):
        tile = build_agent_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert tile["n_components"] == 7

    def test_wired_validation_failure_rate_grades(self, s3):
        """A healthy producer artifact grades the critical component GREEN."""
        _put_agent_quality(s3, {
            "status": "ok",
            "agent_validation_failure_rate": {"value": 0.01, "n": 320},
            "cost_per_signal": {"value": 0.40, "n": 25},
            "retry_storm_count": {"value": 0, "n": 48},
            "agent_latency_p95": {"value": 8200.0, "n": 48},
            "judge_rubric_distribution": {"value": 0.30, "n": 60},
        })
        tile = build_agent_tile(BUCKET, RUN_DATE, s3_client=s3)
        vfr = _comp(tile, "agent_validation_failure_rate")
        assert vfr["value"] == 0.01
        assert vfr["n_samples"] == 320
        assert vfr["status"] == "GREEN"  # 1% < 2% target, lower-is-better
        assert _comp(tile, "retry_storm_count")["status"] == "GREEN"
        assert _comp(tile, "agent_latency_p95")["value"] == 8200.0

    def test_wired_validation_failure_rate_red(self, s3):
        _put_agent_quality(s3, {
            "status": "ok",
            "agent_validation_failure_rate": {"value": 0.18, "n": 300},
        })
        tile = build_agent_tile(BUCKET, RUN_DATE, s3_client=s3)
        vfr = _comp(tile, "agent_validation_failure_rate")
        assert vfr["value"] == 0.18
        assert vfr["status"] == "RED"  # 18% > 10% red-line
        # the un-supplied blocks stay MISSING-INPUT, not crashing.
        assert _comp(tile, "cost_per_signal")["status"] == "N/A-MISSING-INPUT"

    def test_non_ok_status_degrades_to_missing_input(self, s3):
        _put_agent_quality(s3, {"status": "error", "agent_validation_failure_rate": {"value": 0.5}})
        tile = build_agent_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert _comp(tile, "agent_validation_failure_rate")["status"] == "N/A-MISSING-INPUT"
