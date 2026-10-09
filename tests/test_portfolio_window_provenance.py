"""alpha-engine-config-I11089 — every Portfolio Outcome figure states its window.

The card used to print ``N=144``, ``N=42`` and ``N=102`` side by side on one tile
under one ``since_inception`` label, and the reader had to infer which dates and
which artifact each number came from. Each component now carries a ``window``:
start/end dates, trading-day count, basis, source artifact(s), and ``undeclared``
wherever a field cannot be read.

These tests fail when that provenance is MISSING from any component the tile
emits, or when it DISAGREES with the data the value was computed over — the
dates are recomputed here from the fixture itself, never read back from the
code under test.
"""

from __future__ import annotations

import csv
import io
import json

import boto3
import pytest
from moto import mock_aws

import grading.tiles.portfolio_outcome as po
from director.report_card_digest import summarize_report_card
from grading.tiles.portfolio_outcome import (
    EOD_PNL_KEY,
    MODEL_ZOO_LEADERBOARD_KEY,
    SIGNALS_KEY_TEMPLATE,
    UNDECLARED,
    build_portfolio_outcome_tile,
)

BUCKET = "alpha-engine-research"
_HEADER = "date,portfolio_nav,daily_return_pct,spy_return_pct,daily_alpha_pct,positions_snapshot,created_at"

# Components whose N is a count of the eod_pnl.csv rows they were computed
# over. For these, n_samples and the window's trading_days are the same fact
# and must agree. (alpha_trend's N is an autocorrelation-adjusted EFFECTIVE N,
# psr/dsr take theirs from the lib, and the rest have their own window basis —
# all still carry a window, asserted separately.)
_ROW_COUNTED = {
    "sharpe_ratio", "information_ratio", "alpha_vs_spy", "max_drawdown",
    "sortino_ratio", "calmar_ratio", "cvar_95_daily", "hit_rate_daily",
    "beta_vs_spy", "max_dd_duration_days",
}
_EOD_BASIS = _ROW_COUNTED | {"psr", "dsr", "alpha_trend"}
_REQUIRED = {"basis", "sources", "start", "end", "trading_days", "strategy_epoch", "label"}


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _csv(n: int, *, bad_rows: int = 0) -> str:
    """n parseable trading-day rows starting 2026-03-02 (weekdays), plus
    ``bad_rows`` rows whose daily_return_pct does not parse."""
    import datetime as dt

    rows = [_HEADER]
    d = dt.date(2026, 3, 2)
    nav = 1_000_000.0
    i = 0
    while i < n:
        if d.weekday() < 5:
            p = 0.12 if i % 5 else -0.07
            s = 0.05 if i % 3 else -0.02
            nav *= 1 + p / 100.0
            rows.append(f"{d.isoformat()},{nav:.2f},{p},{s},{p - s},{{}},x")
            i += 1
        d += dt.timedelta(days=1)
    for j in range(bad_rows):
        rows.append(f"2025-12-{10 + j:02d},1000000,not-a-number,0.1,0.1,{{}},x")
    return "\n".join(rows) + "\n"


def _parsed_dates(body: str) -> list[str]:
    """The dates the tile SHOULD have measured, recomputed independently."""
    out = []
    for r in csv.DictReader(io.StringIO(body)):
        try:
            float(r["portfolio_nav"])
            float(r["daily_return_pct"])
            float(r["spy_return_pct"])
        except (TypeError, ValueError):
            continue
        out.append(r["date"])
    return sorted(out)


def _put(s3, key, body):
    s3.put_object(Bucket=BUCKET, Key=key, Body=body.encode("utf-8") if isinstance(body, str) else body)


def _by_name(tile):
    return {c["name"]: c for c in tile["components"] if not c.get("unreported")}


def _assert_window_matches(w: dict, dates: list[str]):
    assert w["start"] == min(dates)
    assert w["end"] == max(dates)
    assert w["trading_days"] == len(dates)
    assert w["start"] in w["label"] and w["end"] in w["label"]
    assert f"{len(dates)} trading days" in w["label"]


class TestEveryComponentCarriesAWindow:
    def test_full_tile_no_component_without_provenance(self, s3):
        body = _csv(90)
        _put(s3, EOD_PNL_KEY, body)
        tile = build_portfolio_outcome_tile(BUCKET, s3_client=s3, n_trials=12)
        comps = _by_name(tile)
        assert comps, "tile emitted no components"
        missing = [n for n, c in comps.items() if not isinstance(c.get("window"), dict)]
        assert not missing, f"components with no window provenance: {missing}"
        for name, c in comps.items():
            w = c["window"]
            assert _REQUIRED <= set(w), f"{name}: window lacks {_REQUIRED - set(w)}"
            assert w["sources"] and all(isinstance(s, str) and s for s in w["sources"]), name
            # Either a real window or an explicit `undeclared` — never blank.
            if w["start"] == UNDECLARED:
                assert w["end"] == UNDECLARED and w["trading_days"] is None, name
            else:
                assert w["start"] <= w["end"] and w["trading_days"] >= 1, name
            # The deployed-strategy question (I1084/I11089) is answered, not implied.
            assert w["strategy_epoch"] == UNDECLARED

    def test_missing_eod_pnl_every_window_is_undeclared(self, s3):
        tile = build_portfolio_outcome_tile(BUCKET, s3_client=s3)
        assert tile["window"]["start"] == UNDECLARED
        for name, c in _by_name(tile).items():
            w = c["window"]
            assert w["start"] == w["end"] == UNDECLARED, name
            assert w["trading_days"] is None, name
            assert w["label"].startswith(UNDECLARED), name


class TestWindowAgreesWithTheData:
    def test_eod_components_window_is_exactly_the_parsed_rows(self, s3):
        body = _csv(90, bad_rows=2)
        _put(s3, EOD_PNL_KEY, body)
        tile = build_portfolio_outcome_tile(BUCKET, s3_client=s3, n_trials=12)
        dates = _parsed_dates(body)
        assert len(dates) == 90
        comps = _by_name(tile)
        for name in _EOD_BASIS:
            w = comps[name]["window"]
            assert w["basis"] == "eod_pnl_daily_rows", name
            assert w["sources"][0] == f"s3://{BUCKET}/{EOD_PNL_KEY}", name
            _assert_window_matches(w, dates)
            assert w["rows_excluded"] == 2, name
        _assert_window_matches(tile["window"], dates)

    def test_row_counted_n_equals_window_trading_days(self, s3):
        body = _csv(75)
        _put(s3, EOD_PNL_KEY, body)
        comps = _by_name(build_portfolio_outcome_tile(BUCKET, s3_client=s3))
        for name in _ROW_COUNTED:
            c = comps[name]
            assert c["n_samples"] == c["window"]["trading_days"], (
                f"{name}: N={c['n_samples']} but window says "
                f"{c['window']['trading_days']} trading days"
            )

    def test_dsr_names_the_trial_counter_as_a_source(self, s3):
        _put(s3, EOD_PNL_KEY, _csv(70))
        dsr = _by_name(build_portfolio_outcome_tile(BUCKET, s3_client=s3, n_trials=5))["dsr"]
        assert f"s3://{BUCKET}/backtest/cumulative_trial_count.json" in dsr["window"]["sources"]

    def test_source_object_pins_the_exact_artifact(self, s3):
        _put(s3, EOD_PNL_KEY, _csv(65))
        etag = s3.head_object(Bucket=BUCKET, Key=EOD_PNL_KEY)["ETag"].strip('"')
        tile = build_portfolio_outcome_tile(BUCKET, s3_client=s3)
        assert tile["window"]["source_object"]["etag"] == etag
        assert _by_name(tile)["sharpe_ratio"]["window"]["source_object"]["etag"] == etag

    def test_regime_window_is_the_joined_dates_not_the_tile_window(self, s3):
        body = _csv(80)
        _put(s3, EOD_PNL_KEY, body)
        dates = _parsed_dates(body)
        # Only the last 50 days carry a regime tag: 25 bull, 25 bear.
        tagged = dates[30:]
        for i, d in enumerate(tagged):
            _put(s3, SIGNALS_KEY_TEMPLATE.format(date=d),
                 json.dumps({"market_regime": "bull" if i < 25 else "bear"}))
        rwa = _by_name(build_portfolio_outcome_tile(BUCKET, s3_client=s3))["regime_weighted_alpha"]
        w = rwa["window"]
        assert w["basis"] == "regime_joined_daily_rows"
        _assert_window_matches(w, tagged)
        assert rwa["n_samples"] == w["trading_days"] == 50
        assert w["rows_excluded"] == 30

    def test_attribution_window_is_its_sessions(self, s3, monkeypatch):
        body = _csv(70)
        _put(s3, EOD_PNL_KEY, body)
        sessions = _parsed_dates(body)[5:47]
        payload = {
            "status": "ok",
            "sessions_used": len(sessions),
            "sessions_skipped": {"prior positions_snapshot absent": 4},
            "source_paths": [f"s3://{BUCKET}/{EOD_PNL_KEY}", f"s3://{BUCKET}/sectors.json"],
            "per_session": [{"date": d} for d in sessions],
            "brinson_fachler": {"residual_pct_of_active": 0.01, "selection": 0.02},
            "active_return": {"active_return_ex_index_sleeve": -0.01, "beta_adjusted_alpha": -0.02},
        }
        monkeypatch.setattr(po, "build_attribution", lambda bucket, s3_client=None: payload)
        comps = _by_name(build_portfolio_outcome_tile(BUCKET, s3_client=s3))
        for name in ("attribution_residual_pct_of_active", "bf_selection_effect",
                     "active_return_ex_index_sleeve", "beta_adjusted_alpha"):
            w = comps[name]["window"]
            assert w["basis"] == "attribution_sessions", name
            _assert_window_matches(w, sessions)
            assert comps[name]["n_samples"] == w["trading_days"], name
            assert w["rows_excluded"] == 4, name
            assert w["sources"] == payload["source_paths"], name

    def test_pbo_has_no_calendar_window_and_says_so(self, s3):
        _put(s3, EOD_PNL_KEY, _csv(65))
        _put(s3, MODEL_ZOO_LEADERBOARD_KEY, json.dumps({"date": "2026-09-28", "selection_pbo": {
            "status": "ok", "n_splits": 44, "n_specs": 5, "pbo": 0.09,
            "selected_counts": {"a": 30, "b": 14},
        }}))
        w = _by_name(build_portfolio_outcome_tile(BUCKET, s3_client=s3))["pbo"]["window"]
        assert w["basis"] == "cscv_splits_of_one_model_rotation"
        assert (w["start"], w["end"], w["trading_days"]) == (UNDECLARED, UNDECLARED, None)
        assert w["as_of"] == "2026-09-28"


class TestWindowHelperIsDerivedFromDates:
    """``_window`` derives start/end/count from the dates it is handed, so a
    caller cannot hand it a count that disagrees with its own dates."""

    def test_counts_and_flags_duplicate_dates(self):
        w = po._window(basis="b", sources=["s3://x/y"],
                       dates=["2026-03-03", "2026-03-02", "2026-03-03"])
        assert (w["start"], w["end"], w["trading_days"]) == ("2026-03-02", "2026-03-03", 3)
        assert w["duplicate_dates"] == 1

    def test_no_dates_is_undeclared_not_a_guess(self):
        w = po._window(basis="b", sources=["s3://x/y"], dates=[])
        assert (w["start"], w["end"], w["trading_days"]) == (UNDECLARED, UNDECLARED, None)


class TestTheCardStatesIt:
    def test_digest_renders_tile_and_component_windows(self, s3):
        body = _csv(70)
        _put(s3, EOD_PNL_KEY, body)
        tile = build_portfolio_outcome_tile(BUCKET, s3_client=s3)
        tile.pop("_regime_index", None)
        card = {"_provenance": {"run_date": "2026-10-03"}, "tiles_overall_status": "WATCH",
                "tiles": {"portfolio_outcome": tile}}
        # Pin every component so each one is rendered in full.
        text = summarize_report_card(card, pinned_components={c["name"] for c in tile["components"]})
        dates = _parsed_dates(body)
        assert f"  - window: {dates[0]}→{dates[-1]}, 70 trading days" in text
        sharpe_line = next(ln for ln in text.splitlines() if ln.startswith("  - sharpe_ratio "))
        assert f"window {dates[0]}→{dates[-1]}, 70 trading days" in sharpe_line
        pbo_line = next(ln for ln in text.splitlines() if ln.startswith("  - pbo "))
        assert f"window {UNDECLARED}" in pbo_line
