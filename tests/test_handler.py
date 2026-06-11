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
        # all 7 tiles present in the per-tile status map.
        assert set(out["tile_status"]) == {
            "portfolio_outcome", "predictor", "research", "executor",
            "backtester", "substrate", "agent", "behavioral",
        }
        assert out["tiles_overall_status"] in ("GREEN", "WATCH", "RED", "N/A-NOT-RUN")
        # the written object round-trips.
        obj = s3.get_object(Bucket=BUCKET, Key=out["report_card_key"])
        card = json.loads(obj["Body"].read())
        assert card["tiles_overall_status"] == out["tiles_overall_status"]

    def test_no_write_skips_persist(self, s3):
        _seed_eod(s3)
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET, "write": False})
        assert out["report_card_key"] is None
        # nothing written under evaluator/.
        listing = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"evaluator/{RUN_DATE}/")
        assert listing.get("KeyCount", 0) == 0

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
