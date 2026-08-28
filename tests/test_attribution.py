"""Benchmark-relative attribution (I8188 deliverables 6-7).

The producer/consumer contract test for the ``evaluator/latest/attribution.json``
artifact lives at the bottom (M0 contract discipline: a versioned schema and a
contract test at birth).
"""

from __future__ import annotations

import json

import pytest

from grading.attribution import (
    INPUT_CLOSURE_NAV_BPS,
    SCHEMA_VERSION,
    beta_vs_benchmark,
    build_attribution,
    read_benchmark_sector_weights,
    read_sector_etf_returns,
    session_brinson,
    session_portfolio_groups,
)
from grading.sectors import INDEX_SLEEVE


# ─────────────────────────────────────────────────────────────────────────────
# fakes
# ─────────────────────────────────────────────────────────────────────────────

class _Body:
    def __init__(self, data: bytes):
        self._d = data

    def read(self):
        return self._d


class _S3:
    """Minimal S3 double keyed by object key."""

    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects
        self.written: dict[str, bytes] = {}

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            from botocore.exceptions import ClientError
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": _Body(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, ContentType=None):  # noqa: N803
        self.written[Key] = Body


def _pos(sector, mv, pnl=0.0, shares=10):
    return {"sector": sector, "market_value": mv, "daily_return_usd": pnl,
            "shares": shares, "closing_price": mv / shares}


def _row(date, nav, snap, **extra):
    r = {
        "date": date, "portfolio_nav": str(nav),
        "positions_snapshot": json.dumps(snap),
        "interest_usd": "0", "rotation_realized_usd": "0",
        "pricing_timing_usd": "0", "unattributed_true_usd": "0",
    }
    r.update({k: str(v) for k, v in extra.items()})
    return r


# ─────────────────────────────────────────────────────────────────────────────
# the portfolio leg — no plug
# ─────────────────────────────────────────────────────────────────────────────

class TestPortfolioLeg:
    def test_weights_and_returns_are_beginning_of_day(self):
        prior = _row("2026-08-26", 1_000_000, {
            "AAA": _pos("Technology", 300_000),
            "BBB": _pos("Healthcare", 200_000),
        })
        today = _row("2026-08-27", 1_006_000, {
            "AAA": _pos("Technology", 306_000, 6_000),
            "BBB": _pos("Healthcare", 200_000, 0),
        }, nav_change_usd=6_000)
        g = session_portfolio_groups(prior, today)
        assert g["available"] is True
        assert g["weights"]["Technology"] == pytest.approx(0.30)
        assert g["weights"]["Healthcare"] == pytest.approx(0.20)
        assert g["returns"]["Technology"] == pytest.approx(0.02)
        assert g["cash_weight"] == pytest.approx(0.50)
        assert g["input_closure_usd"] == pytest.approx(0.0)

    def test_the_two_spellings_of_one_sector_land_in_one_group(self):
        prior = _row("2026-08-26", 1_000_000, {
            "AAA": _pos("Information Technology", 300_000),
            "BBB": _pos("Technology", 100_000),
        })
        today = _row("2026-08-27", 1_004_000, {
            "AAA": _pos("Information Technology", 303_000, 3_000),
            "BBB": _pos("Technology", 101_000, 1_000),
        }, nav_change_usd=4_000)
        g = session_portfolio_groups(prior, today)
        assert set(g["weights"]) == {"Technology"}
        assert g["weights"]["Technology"] == pytest.approx(0.40)

    def test_held_spy_is_carved_into_its_own_sleeve(self):
        prior = _row("2026-08-26", 1_000_000, {
            "SPY": _pos("Broad Market / Index", 400_000),
            "AAA": _pos("Technology", 100_000),
        })
        today = _row("2026-08-27", 1_005_000, {
            "SPY": _pos("Broad Market / Index", 404_000, 4_000),
            "AAA": _pos("Technology", 101_000, 1_000),
        }, nav_change_usd=5_000)
        g = session_portfolio_groups(prior, today)
        assert INDEX_SLEEVE not in g["weights"]
        assert g["index_weight"] == pytest.approx(0.40)
        assert g["index_pnl_usd"] == pytest.approx(4_000)
        assert g["input_closure_usd"] == pytest.approx(0.0)

    def test_the_named_cash_lines_are_named_never_a_remainder(self):
        """THE POINT OF I8188. The cash sleeve is the SUM of measured schema
        columns. What does not reconcile is reported, not absorbed."""
        prior = _row("2026-08-26", 1_000_000, {"AAA": _pos("Technology", 500_000)})
        today = _row("2026-08-27", 1_001_400, {
            "AAA": _pos("Technology", 501_000, 1_000)},
            nav_change_usd=1_400, interest_usd=100,
            rotation_realized_usd=200, pricing_timing_usd=50,
            unattributed_true_usd=50,
        )
        g = session_portfolio_groups(prior, today)
        assert g["named_cash_lines"] == {
            "interest_usd": 100.0, "rotation_realized_usd": 200.0,
            "pricing_timing_usd": 50.0, "unattributed_true_usd": 50.0,
        }
        assert g["cash_pnl_usd"] == pytest.approx(400.0)
        assert g["input_closure_usd"] == pytest.approx(0.0)

    def test_an_unaccounted_dollar_is_reported_not_plugged(self):
        prior = _row("2026-08-26", 1_000_000, {"AAA": _pos("Technology", 500_000)})
        today = _row("2026-08-27", 1_009_000, {
            "AAA": _pos("Technology", 501_000, 1_000)}, nav_change_usd=9_000)
        g = session_portfolio_groups(prior, today)
        assert g["input_closure_usd"] == pytest.approx(8_000.0), (
            "an $8,000 gap must SURVIVE as a measured number — folding it into "
            "the cash sleeve is exactly the plug this work removes"
        )

    def test_missing_prior_snapshot_is_unavailable_not_empty(self):
        prior = {"date": "2026-08-26", "portfolio_nav": "1000000",
                 "positions_snapshot": ""}
        today = _row("2026-08-27", 1_001_000, {}, nav_change_usd=1_000)
        assert session_portfolio_groups(prior, today)["available"] is False


# ─────────────────────────────────────────────────────────────────────────────
# the decomposition — exact closure
# ─────────────────────────────────────────────────────────────────────────────

class TestBrinsonClosure:
    def test_effects_sum_exactly_to_active_return(self):
        prior = _row("2026-08-26", 1_000_000, {
            "AAA": _pos("Technology", 300_000),
            "BBB": _pos("Healthcare", 200_000),
            "SPY": _pos("Broad Market / Index", 100_000),
        })
        today = _row("2026-08-27", 1_009_000, {
            "AAA": _pos("Technology", 306_000, 6_000),
            "BBB": _pos("Healthcare", 202_000, 2_000),
            "SPY": _pos("Broad Market / Index", 101_000, 1_000),
        }, nav_change_usd=9_000)
        g = session_portfolio_groups(prior, today)
        bf = session_brinson(
            g,
            {"Technology": 0.4, "Healthcare": 0.3, "Financial Services": 0.3},
            {"Technology": 0.015, "Healthcare": 0.005, "Financial Services": -0.002},
        )
        assert bf is not None
        assert bf.total_effect == pytest.approx(bf.active_return, abs=1e-12)

    def test_the_portfolio_leg_ties_to_the_nav_based_return(self):
        prior = _row("2026-08-26", 1_000_000, {
            "AAA": _pos("Technology", 400_000),
        })
        today = _row("2026-08-27", 1_008_000, {
            "AAA": _pos("Technology", 408_000, 8_000),
        }, nav_change_usd=8_000)
        g = session_portfolio_groups(prior, today)
        bf = session_brinson(g, {"Technology": 1.0}, {"Technology": 0.01})
        assert bf.portfolio_return == pytest.approx(8_000 / 1_000_000, abs=1e-12)

    def test_a_benchmark_sector_with_no_return_today_is_dropped_and_renormalised(self):
        prior = _row("2026-08-26", 1_000_000, {"AAA": _pos("Technology", 500_000)})
        today = _row("2026-08-27", 1_005_000, {
            "AAA": _pos("Technology", 505_000, 5_000)}, nav_change_usd=5_000)
        g = session_portfolio_groups(prior, today)
        bf = session_brinson(
            g,
            {"Technology": 0.5, "Utilities": 0.5},
            {"Technology": 0.02},  # no Utilities close for the date
        )
        assert bf.benchmark_return == pytest.approx(0.02)

    def test_no_benchmark_returns_at_all_yields_none_not_zero(self):
        prior = _row("2026-08-26", 1_000_000, {"AAA": _pos("Technology", 500_000)})
        today = _row("2026-08-27", 1_005_000, {
            "AAA": _pos("Technology", 505_000, 5_000)}, nav_change_usd=5_000)
        g = session_portfolio_groups(prior, today)
        assert session_brinson(g, {"Technology": 1.0}, {}) is None


# ─────────────────────────────────────────────────────────────────────────────
# the benchmark leg
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmarkLeg:
    def _sectors_art(self):
        return json.dumps({"as_of": "2026-08-27", "spy_sector_weights": {
            "Technology": 0.374, "Financial Services": 0.1224,
            "Healthcare": 0.091,
        }}).encode()

    def test_weights_are_canonicalised_and_normalised_to_one(self):
        s3 = _S3({"market_data/sectors/latest.json": self._sectors_art()})
        got = read_benchmark_sector_weights("b", s3_client=s3)
        assert got["available"] is True
        assert sum(got["weights"].values()) == pytest.approx(1.0)
        assert got["as_of"] == "2026-08-27"

    def test_absent_weights_are_unavailable_with_a_reason(self):
        got = read_benchmark_sector_weights("b", s3_client=_S3({}))
        assert got["available"] is False
        assert "absent" in got["reason"]

    def test_a_price_return_etf_history_raises(self):
        """I8188 defect 3: a price-return benchmark leg against a total-return
        portfolio leg is the defect, not an inconvenience."""
        art = json.dumps({
            "adjustment_basis": "split_adjusted",
            "closes": [["2026-08-26", 100.0], ["2026-08-27", 101.0]],
        }).encode()
        s3 = _S3({"market_data/close_history/XLK.json": art})
        with pytest.raises(ValueError, match="dividend_adjusted"):
            read_sector_etf_returns("b", s3_client=s3)

    def test_dividend_adjusted_history_becomes_daily_total_returns(self):
        art = json.dumps({
            "adjustment_basis": "dividend_adjusted",
            "closes": [["2026-08-26", 100.0], ["2026-08-27", 101.0]],
        }).encode()
        s3 = _S3({"market_data/close_history/XLK.json": art})
        got = read_sector_etf_returns("b", s3_client=s3)
        assert got["Technology"]["2026-08-27"] == pytest.approx(0.01)
        assert "2026-08-26" not in got["Technology"]  # no prior close


def test_beta_is_ols_on_the_daily_series():
    port = [0.02, -0.01, 0.03, 0.00]
    assert beta_vs_benchmark(port, [0.01, -0.005, 0.015, 0.0]) == pytest.approx(2.0)
    assert beta_vs_benchmark([0.01], [0.01]) is None
    assert beta_vs_benchmark(port, [0.0] * 4) is None


# ─────────────────────────────────────────────────────────────────────────────
# artifact contract (M0)
# ─────────────────────────────────────────────────────────────────────────────

class TestArtifactContract:
    def _bucket(self, n_sessions=40):
        objects = {}
        art = {"adjustment_basis": "dividend_adjusted", "closes": []}
        from grading.sectors import SECTOR_TO_ETF
        dates = [f"2026-06-{d:02d}" for d in range(1, n_sessions + 2)]
        for etf in SECTOR_TO_ETF.values():
            closes = [[d, 100.0 * (1.001 ** i)] for i, d in enumerate(dates)]
            objects[f"market_data/close_history/{etf}.json"] = json.dumps(
                {**art, "closes": closes}).encode()
        objects["market_data/sectors/latest.json"] = json.dumps({
            "as_of": "2026-08-27",
            "spy_sector_weights": {"Technology": 0.6, "Healthcare": 0.4},
        }).encode()
        rows = []
        nav = 1_000_000.0
        for i, d in enumerate(dates):
            snap = {"AAA": _pos("Technology", nav * 0.5, 500.0 if i else 0.0),
                    "BBB": _pos("Healthcare", nav * 0.3, 200.0 if i else 0.0)}
            rows.append({
                "date": d, "portfolio_nav": f"{nav}",
                "positions_snapshot": json.dumps(snap),
                "nav_change_usd": "700" if i else "",
                "daily_return_pct": "0.07", "spy_return_pct": "0.10",
                "interest_usd": "0", "rotation_realized_usd": "0",
                "pricing_timing_usd": "0", "unattributed_true_usd": "0",
            })
            nav += 700.0
        header = list(rows[0])
        csv_text = ",".join(header) + "\n" + "\n".join(
            ",".join('"' + str(r[h]).replace('"', '""') + '"' for h in header)
            for r in rows
        )
        objects["trades/eod_pnl.csv"] = csv_text.encode()
        return _S3(objects)

    def test_payload_declares_its_schema_and_closes(self):
        payload = build_attribution("b", s3_client=self._bucket(), run_date="2026-08-27")
        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["status"] == "ok", payload.get("reason")
        bf = payload["brinson_fachler"]
        assert bf["linking"] == "carino"
        assert abs(bf["residual_pct_of_active"]) < 1.0, (
            "deliverable 6's closes-when: residual under 1% of active return"
        )
        assert bf["total_effect"] == pytest.approx(bf["bf_active_return"], abs=1e-9)

    def test_the_three_return_quantities_close_exactly(self):
        p = build_attribution("b", s3_client=self._bucket())
        act = p["active_return"]["active_return_vs_spy"]
        bf_act = p["brinson_fachler"]["bf_active_return"]
        tracking = p["benchmark"]["proxy_tracking"]
        assert act == pytest.approx(bf_act + tracking, abs=1e-9), (
            "the sector-ETF proxy's gap to SPY must be PUBLISHED, so that "
            "active = decomposed + tracking holds exactly"
        )

    def test_the_benchmark_leg_declares_its_basis_and_its_approximation(self):
        b = build_attribution("b", s3_client=self._bucket())["benchmark"]
        assert b["basis"] == "total_return"
        assert b["construction"] == "spdr_sector_etf_proxy"
        assert b["weights_are_current_proxy"] is True
        assert b["weights_as_of"] == "2026-08-27"

    def test_factor_attribution_is_an_admitted_gap_not_a_zero(self):
        act = build_attribution("b", s3_client=self._bucket())["active_return"]
        assert act["factor_attribution"] is None
        assert "NOT IMPLEMENTED" in act["factor_attribution_reason"]

    def test_too_few_sessions_is_insufficient_data_never_a_number(self):
        p = build_attribution("b", s3_client=self._bucket(n_sessions=5))
        assert p["status"] == "insufficient_data"
        assert p["brinson_fachler"] is None

    def test_missing_eod_pnl_is_missing_input(self):
        p = build_attribution("b", s3_client=_S3({}))
        assert p["status"] == "missing_input"
        assert p["brinson_fachler"] is None

    def test_input_closure_bound_is_declared_and_breaches_are_listed(self):
        p = build_attribution("b", s3_client=self._bucket())
        assert p["input_closure"]["bound_nav_bps"] == INPUT_CLOSURE_NAV_BPS
        assert isinstance(p["input_closure"]["breaches"], list)

    def test_write_lands_the_pointer_and_the_dated_copy(self):
        from grading.attribution import ATTRIBUTION_LATEST_KEY, write_attribution
        s3 = self._bucket()
        keys = write_attribution({"schema_version": SCHEMA_VERSION}, bucket="b",
                                 run_date="2026-08-27", s3_client=s3)
        assert keys == [ATTRIBUTION_LATEST_KEY, "evaluator/2026-08-27/attribution.json"]
        assert set(s3.written) == set(keys)


class TestWiring:
    """The decomposition must reach a surface, or it is unobserved."""

    def test_the_tile_carries_the_payload_and_the_four_components(self):
        from grading.tiles.portfolio_outcome import _build_attribution_components

        comps, payload = _build_attribution_components("b", s3_client=_S3({}))
        names = [c.name for c in comps]
        assert names == [
            "attribution_residual_pct_of_active", "bf_selection_effect",
            "active_return_ex_index_sleeve", "beta_adjusted_alpha",
        ]
        assert payload["status"] == "missing_input"

    def test_a_failed_build_renders_na_never_a_zero_effect(self):
        """A zero allocation effect and an ABSENT one are different facts."""
        from grading.tiles.portfolio_outcome import _build_attribution_components

        comps, _ = _build_attribution_components("b", s3_client=_S3({}))
        for c in comps:
            assert c.value is None
            assert c.status == "N/A-MISSING-INPUT"
            assert "attribution not built this cycle" in c.status_reason

    def test_the_report_card_writer_persists_the_artifact(self):
        from grading.aggregate import write_report_card

        s3 = _S3({})
        scorecard = {"tiles": {"portfolio_outcome": {
            "attribution": {"schema_version": SCHEMA_VERSION, "status": "ok"},
        }}}
        write_report_card("b", "2026-08-27", scorecard, s3_client=s3)
        assert "evaluator/latest/attribution.json" in s3.written

    def test_a_write_failure_never_takes_the_report_card_with_it(self):
        from grading.aggregate import write_report_card

        class _Boom(_S3):
            def put_object(self, Bucket, Key, Body, ContentType=None):  # noqa: N803
                if "attribution" in Key:
                    raise RuntimeError("boom")
                super().put_object(Bucket, Key, Body, ContentType)

        s3 = _Boom({})
        scorecard = {"tiles": {"portfolio_outcome": {
            "attribution": {"schema_version": SCHEMA_VERSION, "status": "ok"},
        }}}
        out = write_report_card("b", "2026-08-27", scorecard, s3_client=s3)
        assert out["latest_key"] == "evaluator/latest/report_card.json"
