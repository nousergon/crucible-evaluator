"""Tests for grading/handler.py — the grading Lambda entrypoint."""

import json
import logging
import sys
import types
from datetime import datetime, timezone

import boto3
import pytest
from moto import mock_aws

from grading import handler as H

BUCKET = "alpha-engine-research"
# A TRADING day (Fri) — the handler normalizes any calendar date to the trading
# day, so the TestHandler keys/assertions below use a trading-day constant to
# make that normalization a no-op. (Was "2026-06-07", a Sunday — which silently
# encoded the pre-fix bug of keying on the calendar day.)
RUN_DATE = "2026-06-05"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        # Patch the module-level boto3 default client used by the tiles/aggregate
        # by seeding via this client and relying on moto's global interception.
        yield client


def _seed_eod(s3):
    s3.put_object(
        Bucket=BUCKET, Key="trades/eod_pnl.csv",
        Body=(f"date,portfolio_nav,daily_return_pct,spy_return_pct,daily_alpha_pct,positions_snapshot,created_at\n"
              f"{RUN_DATE},1000000,0.1,0.05,0.05,{{}},x\n").encode("utf-8"),
    )
    _seed_freshness_inputs(s3)


def _seed_freshness_inputs(s3):
    """Every OTHER artifact grading.freshness_preflight.assert_input_freshness
    hard-requires (eod_pnl.csv is seeded separately by _seed_eod above, since
    several tests want control over its exact row shape), all dated exactly
    RUN_DATE — the trivially-fresh baseline so TestHandler's tests can focus
    on the handler-level behavior they actually name rather than the
    freshness gate itself (that gate has its own dedicated coverage in
    tests/test_freshness_preflight.py)."""
    s3.put_object(
        Bucket=BUCKET, Key=f"backtest/{RUN_DATE}/metrics.json",
        Body=json.dumps({"run_date": RUN_DATE, "status": "ok"}).encode("utf-8"),
    )
    s3.put_object(
        Bucket=BUCKET, Key=f"backtest/{RUN_DATE}/e2e_lift.json",
        Body=json.dumps({"status": "ok"}).encode("utf-8"),
    )
    s3.put_object(
        Bucket=BUCKET, Key="predictor/weights/meta/manifest.json",
        Body=json.dumps({
            "training_date": RUN_DATE,
            "meta_model_oos_ic_cpcv": {"status": "ok", "n_combos": 4, "mean_ic": 0.1, "frac_positive": 0.75, "ics": [0.1, 0.1, 0.1, 0.1]},
        }).encode("utf-8"),
    )
    s3.put_object(
        Bucket=BUCKET, Key=f"signals/{RUN_DATE}/signals.json",
        Body=json.dumps({"market_regime": "neutral"}).encode("utf-8"),
    )


class TestResolveRunDate:
    def test_explicit_trading_date_passes_through(self):
        # 2026-01-02 is a Friday (trading day) → normalization is a no-op.
        assert H._resolve_run_date({"date": "2026-01-02"}) == "2026-01-02"

    def test_explicit_calendar_date_normalized_to_trading_day(self):
        # The real bug (2026-06-07): the SF threads the CALENDAR run_date, but
        # the backtester + evaluate.py write backtest/{trading_day}/. The grader
        # must read the SAME trading day or it grades 0/18 (insufficient_data).
        assert H._resolve_run_date({"date": "2026-06-07"}) == "2026-06-05"  # Sun → Fri
        assert H._resolve_run_date({"date": "2026-06-06"}) == "2026-06-05"  # Sat → Fri

    def test_env_override_normalized_to_trading_day(self, monkeypatch):
        # Env escape hatch is also normalized — a weekend env value still keys
        # on the trading day.
        monkeypatch.setenv("EVALUATOR_RUN_DATE", "2026-06-07")
        assert H._resolve_run_date({}) == "2026-06-05"

    def test_falls_back_to_trading_day(self, monkeypatch):
        monkeypatch.delenv("EVALUATOR_RUN_DATE", raising=False)
        rd = H._resolve_run_date({})
        # now_dual().trading_day is an ISO date string (already a trading day).
        assert isinstance(rd, str) and len(rd) == 10 and rd[4] == "-"

    def test_env_override(self, monkeypatch):
        # 2025-12-31 is a Wednesday (trading day) → passes through unchanged.
        monkeypatch.setenv("EVALUATOR_RUN_DATE", "2025-12-31")
        assert H._resolve_run_date({}) == "2025-12-31"


class TestHandler:
    def test_writes_report_card_and_returns_summary(self, s3):
        _seed_eod(s3)
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET, "snapshot": True})
        assert out["status"] == "ok"
        assert out["run_date"] == RUN_DATE
        assert out["report_card_key"] == f"evaluator/{RUN_DATE}/report_card.json"
        # config-I2556: latest.json convention key + resolved snapshot flag,
        # additive on the summary.
        assert out["latest_key"] == "evaluator/latest/report_card.json"
        assert out["snapshot"] is True
        # all 9 tiles present in the per-tile status map.
        assert set(out["tile_status"]) == {
            "portfolio_outcome", "predictor", "research", "executor",
            "backtester", "substrate", "agent", "behavioral", "director_quality",
            "contribution_lift",
        }
        assert out["tiles_overall_status"] in ("GREEN", "WATCH", "RED", "N/A-NOT-RUN")
        # both written objects round-trip.
        for key in (out["report_card_key"], out["latest_key"]):
            obj = s3.get_object(Bucket=BUCKET, Key=key)
            card = json.loads(obj["Body"].read())
            assert card["tiles_overall_status"] == out["tiles_overall_status"]

    def test_handler_publishes_the_self_test_and_carries_its_verdict(self, s3):
        """End-to-end: the artifact Brian asked for actually lands, every cycle.

        The wiring is what fails silently — a battery that runs and is never
        published is indistinguishable from one that never ran.
        """
        _seed_eod(s3)
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET, "snapshot": True})
        assert out["self_test_verdict"] in ("PASS", "FAIL", "UNKNOWN")
        assert out["degraded_self_test"] is (out["self_test_verdict"] != "PASS")
        assert out["self_test_key"] == f"evaluator/{RUN_DATE}/self_test.json"

        # sf-pipeline-policy §2.3a rule 3: the CARD carries the verdict too —
        # the surface that presents the run's numbers must say whether the
        # arithmetic behind them was checked.
        card = json.loads(s3.get_object(Bucket=BUCKET, Key=out["report_card_key"])["Body"].read())
        assert card["self_test"]["verdict"] == out["self_test_verdict"]
        assert card["degraded_self_test"] is out["degraded_self_test"]

        body = json.loads(s3.get_object(Bucket=BUCKET, Key=out["self_test_key"])["Body"].read())
        assert body["schema"] == "evaluator_self_test-1.0.0"
        assert body["run_date"] == RUN_DATE
        assert body["verdict"] == out["self_test_verdict"]
        assert body["libraries"]["nousergon-lib"]
        assert body["cases"] and all(
            {"case", "inputs", "expected", "actual", "abs_error", "tolerance", "verdict"}
            <= set(c) for c in body["cases"]
        )

    def test_no_write_skips_persist(self, s3):
        _seed_eod(s3)
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET, "write": False})
        assert out["report_card_key"] is None
        assert out["latest_key"] is None
        # nothing written under evaluator/ at all (neither dated nor latest).
        listing = s3.list_objects_v2(Bucket=BUCKET, Prefix="evaluator/")
        assert listing.get("KeyCount", 0) == 0

    def test_snapshot_true_writes_dated_key(self, s3):
        _seed_eod(s3)
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET, "snapshot": True})
        assert out["snapshot"] is True
        assert out["report_card_key"] == f"evaluator/{RUN_DATE}/report_card.json"
        s3.get_object(Bucket=BUCKET, Key=out["report_card_key"])  # exists

    def test_snapshot_false_skips_dated_key_writes_latest_only(self, s3):
        _seed_eod(s3)
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET, "snapshot": False})
        assert out["snapshot"] is False
        assert out["report_card_key"] is None
        assert out["latest_key"] == "evaluator/latest/report_card.json"
        s3.get_object(Bucket=BUCKET, Key=out["latest_key"])  # latest exists
        # the dated weekly REPORT CARD was NOT written. Asserted on that key
        # specifically rather than on the whole dated prefix being empty: the
        # prefix legitimately carries other per-run artifacts (self_test.json,
        # which publishes on every cycle because it grades the image, not the
        # card), and an emptiness assertion would silently forbid any of them.
        keys = {
            o["Key"] for o in
            s3.list_objects_v2(Bucket=BUCKET, Prefix=f"evaluator/{RUN_DATE}/").get("Contents", [])
        }
        assert f"evaluator/{RUN_DATE}/report_card.json" not in keys

    def test_snapshot_absent_defaults_false(self, s3):
        # config-I2556: nousergon-data PR #832 (both the Saturday advisory-child
        # freeze and the Sunday ModelZoo re-grade tail invoke) merged
        # 2026-07-14 and passes this flag explicitly, so an absent flag now
        # means "refresh latest only" — no dated weekly snapshot.
        _seed_eod(s3)
        event = {"date": RUN_DATE, "bucket": BUCKET}
        assert "snapshot" not in event  # sanity: no explicit flag passed
        out = H.handler(event)
        assert out["snapshot"] is False
        assert out["report_card_key"] is None
        assert out["latest_key"] == "evaluator/latest/report_card.json"

    def test_latest_written_every_non_dry_invoke_regardless_of_snapshot(self, s3):
        # config-I2556 core behavior: `latest` is refreshed on EVERY non-dry
        # invocation, whether or not this cycle also freezes a dated snapshot.
        _seed_eod(s3)
        for snap in (True, False):
            out = H.handler({"date": RUN_DATE, "bucket": BUCKET, "snapshot": snap})
            assert out["latest_key"] == "evaluator/latest/report_card.json"
            s3.get_object(Bucket=BUCKET, Key=out["latest_key"])

    def test_real_graded_counts_present(self, s3):
        _seed_eod(s3)
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET, "write": False})
        # portfolio outcome has eod data → some real-graded components.
        assert out["real_graded"]["portfolio_outcome"] > 0
        assert "agent" in out["real_graded"]

    def test_dry_run_computes_but_does_not_persist(self, s3):
        # Friday-PM preflight (ROADMAP L4504): dry_run exercises the full
        # read+compute path but must NOT write the degenerate preflight card.
        _seed_eod(s3)
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET, "dry_run": True})
        assert out["status"] == "ok"
        assert out["dry_run"] is True
        assert out["report_card_key"] is None
        # compute still ran (tiles graded), proving it's a dry execution, not a skip.
        assert out["real_graded"]["portfolio_outcome"] > 0
        listing = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"evaluator/{RUN_DATE}/")
        assert listing.get("KeyCount", 0) == 0

    def test_explicit_write_overrides_dry_run(self, s3):
        # Operator escape hatch: an explicit write=True wins even under dry_run.
        _seed_eod(s3)
        out = H.handler({
            "date": RUN_DATE, "bucket": BUCKET, "dry_run": True, "write": True,
            "snapshot": True,
        })
        assert out["dry_run"] is True
        assert out["report_card_key"] == f"evaluator/{RUN_DATE}/report_card.json"


class TestExperimentRecordWiring:
    """alpha-engine-config#3077 Phase C: experiment_record.v1 rides alongside
    the report card write, fail-SOFT — a bug in it must never fail the run."""

    def test_writes_experiment_record_alongside_report_card(self, s3):
        _seed_eod(s3)
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET, "snapshot": True})
        assert out["experiment_record_key"] == f"experiments/reference/records/{RUN_DATE}.json"
        obj = s3.get_object(Bucket=BUCKET, Key=out["experiment_record_key"])
        record = json.loads(obj["Body"].read())
        assert record["schema_version"] == 1
        assert record["experiment_id"] == "reference"
        assert record["run_date"] == RUN_DATE
        # _seed_eod only seeds the 5 hard-gated freshness artifacts; several
        # other backtest artifacts are legitimately-optional-and-unseeded in
        # this fixture (scanner_opt/cio_opt/sizing_ab/etc — see grading/
        # artifacts.py's own module docstring), so this record is honestly
        # "partial", not "complete".
        assert record["status"] == "partial"
        # latest.json pointer also refreshed.
        latest = s3.get_object(Bucket=BUCKET, Key="experiments/reference/records/latest.json")
        assert json.loads(latest["Body"].read()) == record

    def test_no_write_skips_experiment_record_too(self, s3):
        _seed_eod(s3)
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET, "write": False})
        assert out["experiment_record_key"] is None
        listing = s3.list_objects_v2(Bucket=BUCKET, Prefix="experiments/")
        assert listing.get("KeyCount", 0) == 0

    def test_dry_run_skips_experiment_record(self, s3):
        _seed_eod(s3)
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET, "dry_run": True})
        assert out["experiment_record_key"] is None
        listing = s3.list_objects_v2(Bucket=BUCKET, Prefix="experiments/")
        assert listing.get("KeyCount", 0) == 0

    def test_experiment_record_failure_does_not_fail_the_handler(self, s3, monkeypatch):
        # A bug in the NEW secondary-artifact code path must never turn a
        # healthy report-card cycle into a failed run — the report card
        # write already succeeded by the time this runs.
        _seed_eod(s3)

        def _boom(*a, **k):
            raise RuntimeError("synthetic experiment_record bug")

        monkeypatch.setattr(H, "build_experiment_record", _boom)
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET, "snapshot": True})
        assert out["status"] == "ok"
        assert out["report_card_key"] == f"evaluator/{RUN_DATE}/report_card.json"
        assert out["experiment_record_key"] is None
        # report card itself still round-trips fine.
        s3.get_object(Bucket=BUCKET, Key=out["report_card_key"])


class TestCheckDeployDriftDispatch:
    """config#2348: action=check_deploy_drift short-circuits BEFORE the normal
    report-card build path — no S3/bucket resolution, no tile compute."""

    def test_dispatches_to_deploy_drift_probe(self, s3, monkeypatch):
        # `s3` (mock_aws) so the stage-coverage call this dispatch also makes
        # (config-I7214/I7334) resolves against moto rather than reaching
        # real AWS — krepis.stage_coverage is genuinely importable now, so
        # this exercises the REAL library call, not a fake.
        captured = {}

        def _fake_check_deploy_drift(*, function_name):
            captured["function_name"] = function_name
            return {"has_drift": False, "function_name": function_name}

        import grading.deploy_drift as dd
        monkeypatch.setattr(dd, "check_deploy_drift", _fake_check_deploy_drift)

        class _Ctx:
            function_name = "alpha-engine-evaluator"

        out = H.handler({"action": "check_deploy_drift"}, context=_Ctx())
        assert out["has_drift"] is False
        assert out["function_name"] == "alpha-engine-evaluator"
        assert out["stage_coverage"]["stage"] == "EvaluatorDeployDriftCheck"
        assert captured["function_name"] == "alpha-engine-evaluator"

    def test_does_not_touch_bucket_or_s3(self, monkeypatch):
        # No S3 client/bucket resolution should occur — this must be a pure,
        # pre-boot gate. Deliberately don't provide the `s3` fixture / moto
        # mock_aws context; a real boto3 call here would error/hang.
        import grading.deploy_drift as dd
        monkeypatch.setattr(
            dd, "check_deploy_drift",
            lambda *, function_name: {"has_drift": False, "function_name": function_name},
        )
        out = H.handler({"action": "check_deploy_drift"}, context=None)
        assert out["has_drift"] is False


class TestCanaryProbeDispatch:
    """config#3058 follow-up: action=canary is the deploy BOOT probe. It must
    short-circuit BEFORE build_report_card so it NEVER runs the hard
    input-freshness preflight — a deploy runs on off-cycle days when the
    current trading day's weekly backtest artifacts legitimately don't exist,
    and the old {"write": false} canary hard-failed every such deploy on a
    MissingInputArtifactError that reflects a healthy image, not a broken one.
    """

    def test_returns_ok_without_building_the_card(self, monkeypatch):
        # If the probe touched build_report_card at all, this would raise —
        # proving the probe is decoupled from the freshness-gated build path.
        def _boom(*a, **k):
            raise AssertionError("canary probe must not call build_report_card")

        monkeypatch.setattr(H, "build_report_card", _boom)
        out = H.handler({"action": "canary"})
        assert out["status"] == "ok"
        assert out["probe"] == "canary"
        assert out["run_date"]  # run-date resolution exercised

    def test_immune_to_missing_input_artifacts(self):
        # The exact failure that broke the deploy: no artifacts in the bucket
        # (so assert_input_freshness WOULD raise MissingInputArtifactError if
        # the build ran). The boot probe must pass regardless.
        with mock_aws():
            boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
            out = H.handler({"action": "canary", "bucket": BUCKET})
        assert out["status"] == "ok"
        assert out["probe"] == "canary"


# ── Stage-output coverage (config-I7214, config-I7334) ────────────────────────
#
# The single shared implementation is `krepis.stage_coverage` (relocated from
# `nousergon_lib.stage_coverage`, which never shipped on any released
# nousergon-lib version — config-I7334). These tests inject a fake module into
# sys.modules so the call sites can be exercised without a real krepis
# resolve; `TestStageCoverageImportDegrades` below separately proves the
# import-failure degrade path (the REAL live behavior before I7334's pin bump).

def _install_fake_stage_coverage(monkeypatch, calls=None):
    """Inject a stand-in `krepis.stage_coverage` module so the handler's
    `from krepis.stage_coverage import assert_stage_coverage` resolves
    to a controllable fake (Python's import system checks sys.modules before
    any finder/loader lookup, so this needs no real submodule on disk)."""
    calls = calls if calls is not None else []
    fake_mod = types.ModuleType("krepis.stage_coverage")

    def _fake_assert_stage_coverage(stage, *, run_date, window_start):
        calls.append({"stage": stage, "run_date": run_date, "window_start": window_start})
        return {"stage": stage, "status": "COVERED", "run_date": run_date}

    fake_mod.assert_stage_coverage = _fake_assert_stage_coverage
    monkeypatch.setitem(sys.modules, "krepis.stage_coverage", fake_mod)
    return calls


class TestStageCoverageReportCard:
    """(grading Lambda, work dispatch) — the ReportCard SF stage."""

    def test_verdict_lands_under_stage_coverage_with_correct_stage_name(self, s3, monkeypatch):
        _seed_eod(s3)
        calls = _install_fake_stage_coverage(monkeypatch)
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET, "snapshot": True})
        assert out["stage_coverage"] == {"stage": "ReportCard", "status": "COVERED", "run_date": RUN_DATE}
        assert [c["stage"] for c in calls] == ["ReportCard"]
        assert calls[0]["run_date"] == RUN_DATE


class TestStageCoverageEvaluatorDeployDriftCheck:
    """(grading Lambda, check_deploy_drift dispatch) — the
    EvaluatorDeployDriftCheck SF stage. An infrastructure/gate stage: it
    declares no durable artifact, but must still record that it declared
    nothing rather than being silently un-considered."""

    def test_verdict_lands_under_stage_coverage_with_correct_stage_name(self, monkeypatch):
        import grading.deploy_drift as dd
        monkeypatch.setattr(dd, "check_deploy_drift", lambda *, function_name: {"has_drift": False})
        calls = _install_fake_stage_coverage(monkeypatch)

        out = H.handler({"action": "check_deploy_drift"}, context=None)

        assert out["stage_coverage"]["stage"] == "EvaluatorDeployDriftCheck"
        assert [c["stage"] for c in calls] == ["EvaluatorDeployDriftCheck"]
        # never confused with the real work stage this Lambda also backs.
        assert out["stage_coverage"]["stage"] != "ReportCard"


def _block_stage_coverage_import(monkeypatch):
    """Force `from krepis.stage_coverage import assert_stage_coverage` to
    raise ImportError regardless of whether the real module is actually
    resolvable in this test environment — the module being absent from
    sys.modules is not sufficient to prove an import failure once krepis
    provides it (I7334 fixed exactly that gap)."""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "krepis.stage_coverage":
            raise ImportError("No module named 'krepis.stage_coverage'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "krepis.stage_coverage", raising=False)
    monkeypatch.setattr(builtins, "__import__", _fake_import)


class TestStageCoverageImportDegrades:
    """Observe mode cannot break the stage it observes: a ModuleNotFoundError
    from the lib must never change the handler's own outcome — logged loud
    AND the `stage_coverage` key is ALWAYS present (never absent, per I7334):
    status UNMEASURED with a reason, distinguishable from a real COVERED
    verdict, plus an alarmable CloudWatch metric."""

    def test_report_card_outcome_unchanged_when_module_absent(self, s3, monkeypatch):
        _seed_eod(s3)
        _block_stage_coverage_import(monkeypatch)
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET, "snapshot": True})
        assert out["status"] == "ok"
        assert out["report_card_key"] == f"evaluator/{RUN_DATE}/report_card.json"
        assert out["stage_coverage"]["status"] == "UNMEASURED"
        assert "krepis.stage_coverage" in out["stage_coverage"]["reason"]
        assert out["stage_coverage"]["stage"] == "ReportCard"

    def test_deploy_drift_check_outcome_unchanged_when_module_absent(self, monkeypatch):
        import grading.deploy_drift as dd
        monkeypatch.setattr(dd, "check_deploy_drift", lambda *, function_name: {"has_drift": False})
        _block_stage_coverage_import(monkeypatch)
        out = H.handler({"action": "check_deploy_drift"}, context=None)
        assert out["has_drift"] is False
        assert out["stage_coverage"]["status"] == "UNMEASURED"
        assert out["stage_coverage"]["stage"] == "EvaluatorDeployDriftCheck"

    def test_unavailable_verdict_never_shares_shape_with_a_real_pass(self, s3, monkeypatch):
        """The loud-unavailable signal must be distinguishable on the exact
        surface a reader sees: the payload's `stage_coverage.status`."""
        _seed_eod(s3)
        _block_stage_coverage_import(monkeypatch)
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET, "snapshot": True})
        assert out["stage_coverage"]["status"] != "COVERED"
        assert out["stage_coverage"]["status"] != "COVERED_NO_OUTPUT"

    def test_unavailable_publishes_alarmable_cloudwatch_metric(self, s3, monkeypatch):
        # `s3` already runs inside `mock_aws()`, which intercepts every
        # boto3 client (including the cloudwatch one this path constructs).
        _seed_eod(s3)
        _block_stage_coverage_import(monkeypatch)
        H.handler({"date": RUN_DATE, "bucket": BUCKET, "snapshot": True})

        cw = boto3.client("cloudwatch", region_name="us-east-1")
        stats = cw.list_metrics(Namespace="AlphaEngine", MetricName="StageCoverage")
        matches = [
            m for m in stats["Metrics"]
            if {"Name": "Stage", "Value": "ReportCard"} in m["Dimensions"]
            and {"Name": "Status", "Value": "UNMEASURED"} in m["Dimensions"]
        ]
        assert matches, f"no UNMEASURED/ReportCard StageCoverage metric published: {stats['Metrics']}"


class TestStageCoverageNeverEnablesEnforcement:
    """OBSERVE MODE ONLY (config-I7214): no shipped call site may pass an
    enforcement-enabling argument. `assert_stage_coverage` takes exactly
    (stage, run_date, window_start) from every call site in this repo — any
    extra kwarg would be how an enforcement flag could sneak in."""

    def test_call_sites_pass_only_the_observe_mode_signature(self, s3, monkeypatch):
        seen_kwargs = []

        def _capture(stage, **kwargs):
            seen_kwargs.append(set(kwargs))
            return {"stage": stage, "status": "COVERED"}

        fake_mod = types.ModuleType("krepis.stage_coverage")
        fake_mod.assert_stage_coverage = _capture
        monkeypatch.setitem(sys.modules, "krepis.stage_coverage", fake_mod)

        _seed_eod(s3)
        H.handler({"date": RUN_DATE, "bucket": BUCKET, "snapshot": True})
        import grading.deploy_drift as dd
        monkeypatch.setattr(dd, "check_deploy_drift", lambda *, function_name: {"has_drift": False})
        H.handler({"action": "check_deploy_drift"}, context=None)

        assert seen_kwargs
        for kwargs in seen_kwargs:
            assert kwargs == {"run_date", "window_start"}


# ── config-I8155: a refused verdict must not kill the stage it observes ──────


def test_a_blank_run_date_records_unmeasured_and_never_raises(monkeypatch, caplog):
    """krepis now REFUSES to build a verdict without the SF execution's own
    run_date. That refusal is correct at the library — one execution's verdicts
    landed under two date prefixes on 2026-08-22 because it used to fall back to
    the cycle date — but an observer that can kill the stage it observes is a
    new failure mode bolted onto the one it reports."""
    import types

    from grading import handler as h

    class _Contract(ValueError):
        pass

    fake = types.ModuleType("krepis.stage_coverage")

    def _refuse(stage, **kwargs):
        raise _Contract("run_date is REQUIRED and must be non-empty")

    fake.assert_stage_coverage = _refuse
    monkeypatch.setitem(sys.modules, "krepis.stage_coverage", fake)

    result: dict = {}
    with caplog.at_level(logging.CRITICAL):
        out = h._record_stage_coverage(
            "ReportCard", run_date="", started=datetime.now(timezone.utc), result=result
        )

    assert out["stage_coverage"]["status"] == h._STAGE_COVERAGE_STATUS_UNMEASURED
    assert "refused" in out["stage_coverage"]["reason"]
    assert out["stage_coverage"]["is_finding"] is False
    assert any("REFUSED" in r.message for r in caplog.records)


def test_a_normal_run_date_still_reaches_the_library(monkeypatch):
    import types

    from grading import handler as h

    seen: list = []
    fake = types.ModuleType("krepis.stage_coverage")
    fake.assert_stage_coverage = lambda stage, **kw: seen.append((stage, kw)) or {
        "stage": stage, "status": "COVERED", "run_date": kw["run_date"]
    }
    monkeypatch.setitem(sys.modules, "krepis.stage_coverage", fake)

    out = h._record_stage_coverage(
        "ReportCard",
        run_date="2026-08-22",
        started=datetime.now(timezone.utc),
        result={},
    )
    assert seen[0][1]["run_date"] == "2026-08-22"
    assert out["stage_coverage"]["run_date"] == "2026-08-22"
