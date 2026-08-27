"""Tests for grading/tiles/behavioral.py — Tile 7 (L4514/config#698)."""

import json

import boto3
import pytest
from moto import mock_aws

from grading.tiles.behavioral import build_behavioral_tile

BUCKET = "alpha-engine-research"
RUN_DATE = "2026-06-13"  # a Saturday — the report-card cadence


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _put_ba(s3, data):
    s3.put_object(Bucket=BUCKET, Key=f"backtest/{RUN_DATE}/behavioral_anomaly.json",
                  Body=json.dumps(data).encode())


def _put_shadow(s3, date, tripwire):
    s3.put_object(Bucket=BUCKET, Key=f"predictor/optimizer_shadow/{date}.json",
                  Body=json.dumps({"turnover_tripwire": tripwire}).encode())


def _comp(tile, name):
    return next(c for c in tile["components"] if c["name"] == name)


FULL_BA = {
    "status": "ok",
    "decision_reversal": {"status": "ok", "window_days": 10, "n_exits": 20,
                          "n_reversals": 1, "reversal_rate": 0.05, "offenders": []},
    "conviction_stability": {"status": "ok", "window_days": 90, "n_tickers": 24,
                             "median_score_std": 3.2, "p90_score_std": 8.1,
                             "high_variance": []},
    "cost_adjusted_quality": {"status": "ok", "n_roundtrips": 30,
                              "median_gross_alpha_pct": 1.4, "median_slippage_pct": 0.05,
                              "median_net_alpha_pct": 1.35, "cost_drag_threshold": 0.25,
                              "n_winners": 18, "n_cost_dragged_winners": 2,
                              "cost_drag_fraction": 0.111},
    "portfolio_state_drift": {"status": "ok", "n_days": 22, "n_unparseable": 0,
                              "median_daily_drift": 0.03, "max_daily_drift": 0.12,
                              "spike_threshold": 0.25, "n_spike_days": 0,
                              "spike_days": []},
}

OK_TRIPWIRE = {"status": "ok", "turnover_one_way": 0.04, "daily_band": 0.10,
               "daily_breach": False, "rolling_days": 5, "n_days_used": 5,
               "rolling_sum": 0.18, "rolling_band": 0.60, "rolling_breach": False}


class TestMissing:
    def test_all_absent_is_transparent_na(self, s3):
        tile = build_behavioral_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert tile["module"] == "behavioral"
        assert tile["n_components"] == 5
        for name in ("turnover", "decision_reversal", "conviction_stability",
                     "cost_adjusted_quality", "portfolio_state_drift"):
            assert _comp(tile, name)["status"] == "N/A-MISSING-INPUT"

    def test_component_insufficient_data_is_na_with_detail(self, s3):
        _put_ba(s3, {"status": "insufficient_data",
                     "decision_reversal": {"status": "insufficient_data", "n_trades": 0}})
        comp = _comp(build_behavioral_tile(BUCKET, RUN_DATE, s3_client=s3),
                     "decision_reversal")
        assert comp["status"] == "N/A-MISSING-INPUT"
        assert "insufficient_data" in (comp.get("na_detail") or comp.get("status_reason") or "")


class TestPopulated:
    def test_full_artifact_grades_green(self, s3):
        _put_ba(s3, FULL_BA)
        _put_shadow(s3, RUN_DATE, OK_TRIPWIRE)
        tile = build_behavioral_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert _comp(tile, "decision_reversal")["status"] == "GREEN"   # 5% < 10% target
        assert _comp(tile, "conviction_stability")["status"] == "GREEN"  # 3.2 < 5
        assert _comp(tile, "cost_adjusted_quality")["status"] == "GREEN"  # 1.35 > 0.5
        assert _comp(tile, "portfolio_state_drift")["status"] == "GREEN"  # 3% < 5%
        assert _comp(tile, "turnover")["status"] == "GREEN"  # 0.18 < 0.30 (band/2)

    def test_lower_is_better_red_on_churn(self, s3):
        ba = dict(FULL_BA)
        ba["decision_reversal"] = {"status": "ok", "window_days": 10, "n_exits": 20,
                                   "n_reversals": 8, "reversal_rate": 0.40, "offenders": []}
        _put_ba(s3, ba)
        comp = _comp(build_behavioral_tile(BUCKET, RUN_DATE, s3_client=s3),
                     "decision_reversal")
        assert comp["status"] == "RED"  # 40% > 30% red-line, direction inferred

    def test_no_component_is_critical_soak_posture(self, s3):
        _put_ba(s3, FULL_BA)
        _put_shadow(s3, RUN_DATE, OK_TRIPWIRE)
        tile = build_behavioral_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert all(c["criticality"] in ("supporting", "diagnostic")
                   for c in tile["components"])


class TestShadowWalkback:
    def test_tripwire_found_on_prior_weekday(self, s3):
        _put_ba(s3, FULL_BA)
        _put_shadow(s3, "2026-06-12", OK_TRIPWIRE)  # Friday before the Saturday card
        comp = _comp(build_behavioral_tile(BUCKET, RUN_DATE, s3_client=s3), "turnover")
        assert comp["status"] == "GREEN"
        assert "2026-06-12" in comp["source_path"]

    def test_tripwire_beyond_lookback_is_na(self, s3):
        _put_shadow(s3, "2026-06-01", OK_TRIPWIRE)  # 12 days back — outside window
        comp = _comp(build_behavioral_tile(BUCKET, RUN_DATE, s3_client=s3), "turnover")
        assert comp["status"] == "N/A-MISSING-INPUT"

    def test_tripwire_disabled_status_is_na(self, s3):
        _put_shadow(s3, RUN_DATE, {"status": "disabled"})
        comp = _comp(build_behavioral_tile(BUCKET, RUN_DATE, s3_client=s3), "turnover")
        assert comp["status"] == "N/A-MISSING-INPUT"
        assert "disabled" in (comp.get("na_detail") or comp.get("status_reason") or "")


# ── a BREACH is a measurement, not an absence (alpha-engine-config-I8752) ──
#
# This tile gated on `status == "ok"` while crucible-executor's tripwire
# `status` was a hardcoded literal that never changed — so the distinction had
# no teeth and the component always graded. Now that the executor derives its
# status (`breach_daily` / `breach_rolling`), the old gate would have blanked
# this component to N/A on exactly the weeks the turnover number matters:
# reporting "no ok tripwire block" about a tripwire that had a number and was
# paging about it.
#
# Live shape it would have hidden, measured 2026-08-27: eight consecutive
# sessions with rolling_sum 0.69..0.95 against a 0.60 band.

BREACH_ROLLING_TRIPWIRE = {
    "status": "breach_rolling", "turnover_one_way": 0.170, "daily_band": 0.25,
    "daily_breach": False, "rolling_days": 5, "n_days_used": 5,
    "rolling_sum": 0.921, "rolling_band": 0.60, "rolling_breach": True,
}

BREACH_DAILY_TRIPWIRE = {
    "status": "breach_daily", "turnover_one_way": 0.30, "daily_band": 0.25,
    "daily_breach": True, "rolling_days": 5, "n_days_used": 5,
    "rolling_sum": 0.95, "rolling_band": 0.60, "rolling_breach": True,
}


class TestBreachIsAMeasurement:
    def test_rolling_breach_still_grades_the_component(self, s3):
        _put_ba(s3, FULL_BA)
        _put_shadow(s3, RUN_DATE, BREACH_ROLLING_TRIPWIRE)

        comp = _comp(build_behavioral_tile(BUCKET, RUN_DATE, s3_client=s3), "turnover")

        assert comp["status"] != "N/A-MISSING-INPUT", comp
        assert comp["value"] == 0.921
        # The breach is stated, not merely survived.
        blob = " ".join(str(v) for v in comp.values() if isinstance(v, str))
        assert "rolling_breach=True" in blob, comp

    def test_daily_breach_still_grades_the_component(self, s3):
        _put_ba(s3, FULL_BA)
        _put_shadow(s3, RUN_DATE, BREACH_DAILY_TRIPWIRE)

        comp = _comp(build_behavioral_tile(BUCKET, RUN_DATE, s3_client=s3), "turnover")

        assert comp["status"] != "N/A-MISSING-INPUT", comp
        assert comp["value"] == 0.95

    def test_a_breach_grades_worse_than_a_quiet_week(self, s3):
        """The point of surfacing it: 0.921 is past the 0.60 red line, so the
        component must not read the same as a 0.18 week."""
        _put_ba(s3, FULL_BA)
        _put_shadow(s3, RUN_DATE, OK_TRIPWIRE)
        quiet = _comp(build_behavioral_tile(BUCKET, RUN_DATE, s3_client=s3), "turnover")

        _put_shadow(s3, RUN_DATE, BREACH_ROLLING_TRIPWIRE)
        breaching = _comp(build_behavioral_tile(BUCKET, RUN_DATE, s3_client=s3), "turnover")

        assert quiet["status"] != breaching["status"], (quiet, breaching)

    def test_statuses_with_no_measurement_are_still_na(self, s3):
        """`disabled`, `no_turnover_metric` and `error` genuinely carry no
        number and must keep falling through."""
        for status in ("disabled", "no_turnover_metric", "error"):
            _put_shadow(s3, RUN_DATE, {"status": status})
            comp = _comp(build_behavioral_tile(BUCKET, RUN_DATE, s3_client=s3), "turnover")
            assert comp["status"] == "N/A-MISSING-INPUT", (status, comp)

    def test_the_measured_vocabulary_matches_the_executors(self):
        """Cross-repo contract pin.

        The executor owns this vocabulary
        (crucible-executor/executor/turnover_tripwire.py::TRIPWIRE_STATUSES).
        It is held here as a literal rather than imported — the S3 artifact's
        field vocabulary is the shared contract and a non-optional cross-repo
        package import is not worth taking for six strings — so drift has to
        fail HERE instead of silently N/A-ing the component.

        If this fails, the executor changed its vocabulary: reconcile the two
        rather than widening this set to make the test pass.
        """
        from grading.tiles.behavioral import _TRIPWIRE_MEASURED_STATUSES

        assert _TRIPWIRE_MEASURED_STATUSES == {"ok", "breach_daily", "breach_rolling"}
        # The complement, stated so a new executor status cannot quietly join
        # either side without this assertion being revisited.
        assert not (_TRIPWIRE_MEASURED_STATUSES & {"disabled", "no_turnover_metric", "error"})
