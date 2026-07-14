"""Tests for grading/handler.py — the grading Lambda entrypoint."""

import json

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
        Body=(b"date,portfolio_nav,daily_return_pct,spy_return_pct,daily_alpha_pct,positions_snapshot,created_at\n"
              b"2026-06-06,1000000,0.1,0.05,0.05,{},x\n"),
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
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET})
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
        }
        assert out["tiles_overall_status"] in ("GREEN", "WATCH", "RED", "N/A-NOT-RUN")
        # both written objects round-trip.
        for key in (out["report_card_key"], out["latest_key"]):
            obj = s3.get_object(Bucket=BUCKET, Key=key)
            card = json.loads(obj["Body"].read())
            assert card["tiles_overall_status"] == out["tiles_overall_status"]

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
        # the dated weekly key was NOT written.
        listing = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"evaluator/{RUN_DATE}/")
        assert listing.get("KeyCount", 0) == 0

    def test_snapshot_absent_defaults_true_back_compat(self, s3):
        # config-I2556 staged back-compat: an absent flag must preserve
        # today's behavior (dated card written on every non-dry invoke) until
        # the nousergon-data callers passing the flag explicitly are merged.
        _seed_eod(s3)
        event = {"date": RUN_DATE, "bucket": BUCKET}
        assert "snapshot" not in event  # sanity: no explicit flag passed
        out = H.handler(event)
        assert out["snapshot"] is True
        assert out["report_card_key"] == f"evaluator/{RUN_DATE}/report_card.json"

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
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET, "dry_run": True, "write": True})
        assert out["dry_run"] is True
        assert out["report_card_key"] == f"evaluator/{RUN_DATE}/report_card.json"


class TestCheckDeployDriftDispatch:
    """config#2348: action=check_deploy_drift short-circuits BEFORE the normal
    report-card build path — no S3/bucket resolution, no tile compute."""

    def test_dispatches_to_deploy_drift_probe(self, monkeypatch):
        captured = {}

        def _fake_check_deploy_drift(*, function_name):
            captured["function_name"] = function_name
            return {"has_drift": False, "function_name": function_name}

        import grading.deploy_drift as dd
        monkeypatch.setattr(dd, "check_deploy_drift", _fake_check_deploy_drift)

        class _Ctx:
            function_name = "alpha-engine-evaluator"

        out = H.handler({"action": "check_deploy_drift"}, context=_Ctx())
        assert out == {"has_drift": False, "function_name": "alpha-engine-evaluator"}
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
