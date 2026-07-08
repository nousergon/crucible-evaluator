"""Tests for grading/history.py — cross-cycle trend history (config#1836).

Covers the loader contract (prior CARDS are the SSOT; N/A weeks skipped, never
zero-filled; short history WARNs with the found-card count; N bounded at 13)
and the closes-when assertion: ``build_report_card`` with 3 prior weekly cards
in mocked S3 threads history-derived ``trend_4w``/``trend_13w`` (and non-default
glyphs) onto the named critical components, skipping N/A weeks.
"""

import json
import logging

import boto3
import pytest
from moto import mock_aws

from grading.aggregate import build_report_card, report_card_key
from grading.history import (
    MAX_HISTORY_CARDS,
    CardHistory,
    _CARD_KEY_RE,
    load_card_history,
)

BUCKET = "alpha-engine-research"
RUN_DATE = "2026-07-04"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _card(sharpe=None, ic=None, ic_status="WATCH"):
    """A minimal prior report card carrying RC v2 tiles."""
    tiles = {}
    if sharpe is not None:
        tiles["portfolio_outcome"] = {"components": [
            {"name": "sharpe_ratio", "value": sharpe, "status": "WATCH"},
        ]}
    else:
        # An N/A week for sharpe — must be SKIPPED, never zero-filled.
        tiles["portfolio_outcome"] = {"components": [
            {"name": "sharpe_ratio", "value": None, "status": "N/A-MISSING-INPUT"},
        ]}
    if ic is not None:
        tiles["research"] = {"components": [
            {"name": "research_composite_ic", "value": ic, "status": ic_status},
        ]}
    return {"tiles": tiles}


def _put_card(s3, date, card):
    s3.put_object(Bucket=BUCKET, Key=report_card_key(date),
                  Body=json.dumps(card).encode())


class TestCardKeyPattern:
    def test_regex_matches_aggregate_key_layout(self):
        # Structural guard: history's key regex must stay in lockstep with the
        # writer's key builder (grading.aggregate.report_card_key).
        assert _CARD_KEY_RE.match(report_card_key("2026-07-04"))
        assert not _CARD_KEY_RE.match("evaluator/2026-07-04/other.json")
        assert not _CARD_KEY_RE.match("backtest/2026-07-04/report_card.json")


class TestLoadCardHistory:
    def test_empty_bucket_warns_with_count(self, s3, caplog):
        with caplog.at_level(logging.WARNING, logger="grading.history"):
            h = load_card_history(BUCKET, RUN_DATE, s3_client=s3)
        assert h.n_cards_found == 0
        assert h.prior_values("portfolio_outcome", "sharpe_ratio") == []
        assert h.trends_for("portfolio_outcome", "sharpe_ratio", 1.0) == {}
        assert any("found 0 prior report card" in r.message for r in caplog.records)

    def test_extracts_series_oldest_to_newest_and_skips_na_weeks(self, s3, caplog):
        _put_card(s3, "2026-06-13", _card(sharpe=0.5, ic=0.01))
        _put_card(s3, "2026-06-20", _card(sharpe=None, ic=0.02))   # sharpe N/A week
        _put_card(s3, "2026-06-27", _card(sharpe=0.7, ic=0.03))
        with caplog.at_level(logging.WARNING, logger="grading.history"):
            h = load_card_history(BUCKET, RUN_DATE, s3_client=s3)
        assert h.n_cards_found == 3
        # N/A week skipped — no zero-filling.
        assert h.prior_values("portfolio_outcome", "sharpe_ratio") == [0.5, 0.7]
        assert h.prior_values("research", "research_composite_ic") == [0.01, 0.02, 0.03]
        # Short history (3 < 13) → WARN names the found-card count.
        assert any("found 3 prior report card" in r.message for r in caplog.records)

    def test_only_cards_strictly_before_run_date(self, s3):
        _put_card(s3, "2026-06-27", _card(sharpe=0.5))
        _put_card(s3, RUN_DATE, _card(sharpe=9.9))       # same-day: excluded
        _put_card(s3, "2026-07-11", _card(sharpe=9.9))   # future: excluded
        h = load_card_history(BUCKET, RUN_DATE, s3_client=s3)
        assert h.prior_values("portfolio_outcome", "sharpe_ratio") == [0.5]

    def test_bounded_at_max_history_cards(self, s3):
        # 15 prior weekly cards; only the latest 13 may be read.
        import datetime as dt
        day = dt.date(2026, 7, 4)
        for i in range(1, 16):
            d = (day - dt.timedelta(weeks=i)).isoformat()
            _put_card(s3, d, _card(sharpe=float(i)))
        h = load_card_history(BUCKET, RUN_DATE, s3_client=s3)
        assert h.n_cards_found == MAX_HISTORY_CARDS == 13
        vals = h.prior_values("portfolio_outcome", "sharpe_ratio")
        assert len(vals) == 13
        # Latest 13 priors, chronological: weeks 13..1 → values 13.0 .. 1.0.
        assert vals == [float(i) for i in range(13, 0, -1)]

    def test_corrupt_card_skipped_with_warn(self, s3, caplog):
        _put_card(s3, "2026-06-20", _card(sharpe=0.5))
        s3.put_object(Bucket=BUCKET, Key=report_card_key("2026-06-27"),
                      Body=b"{ not json")
        with caplog.at_level(logging.WARNING, logger="grading.history"):
            h = load_card_history(BUCKET, RUN_DATE, s3_client=s3)
        assert h.prior_values("portfolio_outcome", "sharpe_ratio") == [0.5]
        assert any("corrupt prior card" in r.message for r in caplog.records)


class TestTrendsFor:
    def test_appends_current_value_and_windows(self):
        h = CardHistory({("research", "research_composite_ic"):
                         [float(i) for i in range(1, 15)]}, 14)
        t = h.trends_for("research", "research_composite_ic", 99.0)
        assert t["trend_4w"] == [12.0, 13.0, 14.0, 99.0]
        assert t["trend_13w"] == [float(i) for i in range(3, 15)] + [99.0]
        assert len(t["trend_13w"]) == 13

    def test_na_current_value_uses_prior_only(self):
        h = CardHistory({("research", "cio"): [0.1, 0.2]}, 2)
        t = h.trends_for("research", "cio", None)
        assert t["trend_4w"] == [0.1, 0.2]

    def test_unknown_component_returns_empty(self):
        h = CardHistory({}, 5)
        assert h.trends_for("research", "nope", 1.0) == {}


class TestBuildReportCardThreadsTrends:
    """Closes-when (config#1836): build_report_card with 3 prior cards threads
    history-derived trends onto the named critical components and skips N/A
    weeks."""

    def _seed_priors(self, s3):
        # 3 prior weekly cards; the middle week is N/A for sharpe_ratio.
        _put_card(s3, "2026-06-13", _card(sharpe=0.50, ic=0.010))
        _put_card(s3, "2026-06-20", _card(sharpe=None, ic=0.020))
        _put_card(s3, "2026-06-27", _card(sharpe=0.70, ic=0.030))

    def _seed_eod_pnl(self, s3, n_days=70):
        rows = ["date,portfolio_nav,daily_return_pct,spy_return_pct,daily_alpha_pct"]
        nav = 100000.0
        for i in range(n_days):
            r = 0.10 + (i % 3) * 0.05           # gently positive daily % returns
            nav *= 1 + r / 100
            rows.append(f"2026-{3 + i // 28:02d}-{i % 28 + 1:02d},{nav:.2f},{r},{0.05},{r - 0.05}")
        s3.put_object(Bucket=BUCKET, Key="trades/eod_pnl.csv",
                      Body="\n".join(rows).encode())

    def _comp(self, card, tile, name):
        return next(c for c in card["tiles"][tile]["components"] if c["name"] == name)

    def test_trends_threaded_and_na_weeks_skipped(self, s3):
        self._seed_priors(s3)
        self._seed_eod_pnl(s3)
        card = build_report_card(BUCKET, RUN_DATE, s3_client=s3)

        sharpe = self._comp(card, "portfolio_outcome", "sharpe_ratio")
        assert sharpe["value"] is not None
        # Prior [0.50, 0.70] (N/A week SKIPPED) + current value appended.
        assert sharpe["trend_4w"] == [0.50, 0.70, pytest.approx(sharpe["value"])]
        assert sharpe["trend_13w"] == sharpe["trend_4w"]

        # research_composite_ic is N/A this cycle (no e2e_lift seeded) — prior
        # history still rides so the trajectory stays visible.
        ric = self._comp(card, "research", "research_composite_ic")
        assert ric["trend_4w"] == [0.010, 0.020, 0.030]

    def test_non_default_glyph_on_improving_series(self, s3):
        # Monotonic improvement across 3 priors + current → a non-default glyph
        # (the "populated trend arrays and non-default glyphs" closes-when).
        _put_card(s3, "2026-06-13", _card(sharpe=-2.0))
        _put_card(s3, "2026-06-20", _card(sharpe=-1.5))
        _put_card(s3, "2026-06-27", _card(sharpe=-1.0))
        self._seed_eod_pnl(s3)  # positive returns → current sharpe > -1.0
        card = build_report_card(BUCKET, RUN_DATE, s3_client=s3)
        sharpe = self._comp(card, "portfolio_outcome", "sharpe_ratio")
        assert len(sharpe["trend_4w"]) == 4
        assert sharpe["trend_decoration"] == "↑↑"

    def test_no_prior_cards_trends_stay_default(self, s3):
        self._seed_eod_pnl(s3)
        card = build_report_card(BUCKET, RUN_DATE, s3_client=s3)
        sharpe = self._comp(card, "portfolio_outcome", "sharpe_ratio")
        assert sharpe["trend_4w"] is None
        assert sharpe["trend_13w"] is None
        assert sharpe["trend_decoration"] == "→"


class TestLoadCardHistoryFailLoud:
    """config#1958 gap-closer: a REAL S3 error must raise, never degrade to
    empty history (only 404/corrupt/short degrade, each with a WARN)."""

    def test_non_404_get_error_raises(self):
        from unittest.mock import MagicMock

        from botocore.exceptions import ClientError

        s3 = MagicMock()
        s3.get_paginator.return_value.paginate.return_value = [
            {"Contents": [{"Key": "evaluator/2026-06-27/report_card.json"}]}
        ]
        s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "denied"}}, "GetObject"
        )
        try:
            load_card_history("b", "2026-07-04", s3_client=s3)
        except ClientError as e:
            assert e.response["Error"]["Code"] == "AccessDenied"
        else:
            raise AssertionError("AccessDenied should have raised, not degraded")
