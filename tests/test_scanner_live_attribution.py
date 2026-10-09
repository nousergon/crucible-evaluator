"""The ``scanner`` component grades what the live scanner published.

alpha-engine-config-I11985 / I11155. On the 2026-10-02 card the research
tile's ``scanner`` WATCH (edge +3.7%, 21d-alpha-lift +0.87%, N=651) was the
retired tech_score gate's record from ``scanner_evaluations`` — 2026-04-12..
2026-07-17, 78 days old by its own ``source_freshness`` stamp — and the weekly
Director cited it as this week's scanner. The ``scanner_lift`` block below is
that artifact's block, verbatim except for trimming; the ``scanner_lift_live``
block is the backtester's read of the published cuts
(``candidates/{run_date}/candidates.json::scanner_eval_log``) over the same
data on 2026-10-06.
"""

import json

import boto3
import pytest
from moto import mock_aws

from grading.history import CardHistory, _extract_series
from grading.scorecard import _grade_scanner
from grading.tiles.research import build_research_tile

BUCKET = "alpha-engine-research"
RUN_DATE = "2026-10-03"
RETIRED_ARM = "tech_score_baseline (retired from live feed 2026-06-29)"
LIVE_ARM = (
    "scanner_live_cut (the cut the live scanner published — "
    "candidates/{run_date}/candidates.json::scanner_eval_log.quant_filter_pass)"
)

# backtest/2026-10-02/e2e_lift.json::scanner_lift (trimmed).
RETIRED = {
    "lift": -0.0034, "n_universe": 14376, "n_passing": 653,
    "classification": {"precision": 0.441, "tp": 288, "fp": 365, "fn": 6054, "tn": 7669, "n": 14376},
    "classification_21d": {"precision": 0.5084, "tp": 331, "fp": 320, "fn": 6436, "tn": 7260, "n": 14347},
    "lift_21d_log": {"lift": 0.00868, "n_selected": 651},
    "first_eval_date": "2026-04-12", "last_eval_date": "2026-07-17",
    "arm": RETIRED_ARM,
    "source_freshness": {
        "measurement": "scanner_lift", "run_date": "2026-10-03", "stale": True,
        "stale_tables": ["scanner_evaluations"], "max_age_days": 14,
        "sources": [{"table": "scanner_evaluations", "present": True,
                     "newest_date": "2026-07-17", "age_days": 78, "stale": True,
                     "retired_date": "2026-07-12"}],
    },
}

# crucible-backtester scanner_lift_live over the same research.db + the
# published candidates.json cuts, measured 2026-10-06.
LIVE = {
    "status": "ok", "lift": 0.0039, "n_universe": 10788, "n_passing": 719,
    "classification": {"precision": 0.4882, "tp": 351, "fp": 368, "fn": 4385, "tn": 5684, "n": 10788},
    "classification_21d": {"precision": 0.4656, "tp": 223, "fp": 256, "fn": 2274, "tn": 4421, "n": 7174},
    "lift_21d_log": {"lift": 0.0196, "n_selected": 479},
    "arm": LIVE_ARM,
    "source": "s3://alpha-engine-research/candidates/{run_date}/candidates.json::scanner_eval_log",
    "cohort_rule": "latest_candidates_artifact_per_iso_week",
    "cohort_dates": ["2026-07-10", "2026-07-17", "2026-07-24", "2026-07-31", "2026-08-07",
                     "2026-08-14", "2026-08-21", "2026-08-28", "2026-09-04", "2026-09-11",
                     "2026-09-18", "2026-09-25", "2026-10-02"],
    "newest_published_cohort": "2026-10-02",
    "n_cohorts_matured_21d": 8,
    "first_matured_eval_date_21d": "2026-07-10",
    "last_matured_eval_date_21d": "2026-08-28",
}

RETIRED_EDGE = 0.5084 - (331 + 6436) / 14347
LIVE_EDGE = 0.4656 - (223 + 2274) / 7174


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _put(s3, name, data):
    s3.put_object(Bucket=BUCKET, Key=f"backtest/{RUN_DATE}/{name}", Body=json.dumps(data).encode())


def _scanner(s3, e2e, history=None):
    _put(s3, "e2e_lift.json", e2e)
    tile = build_research_tile(BUCKET, RUN_DATE, s3_client=s3, history=history)
    return next(c for c in tile["components"] if c["name"] == "scanner")


class TestResearchTileScanner:
    def test_grades_the_live_cut_not_the_retired_gate(self, s3):
        """The misattributed case: the 2026-10-02 card graded RETIRED here."""
        comp = _scanner(s3, {"scanner_lift": RETIRED, "scanner_lift_live": LIVE})
        assert comp["value"] == pytest.approx(LIVE_EDGE)
        assert comp["value"] != pytest.approx(RETIRED_EDGE)
        assert comp["n_samples"] == 223 + 256
        assert comp["arm"] == LIVE_ARM
        reason = comp["status_reason"]
        assert "21d-alpha-lift +1.96%" in reason
        assert "2026-07-10..2026-08-28" in reason      # matured live cohorts
        assert "newest published 2026-10-02" in reason

    def test_retired_read_is_stated_as_dated_history_beside_it(self, s3):
        comp = _scanner(s3, {"scanner_lift": RETIRED, "scanner_lift_live": LIVE})
        reason = comp["status_reason"]
        assert "HISTORICAL, not graded" in reason
        assert RETIRED_ARM in reason
        assert "2026-04-12..2026-07-17" in reason
        assert "newest 2026-07-17, 78d old" in reason
        # The lift the Director quoted belongs to the history clause only.
        assert reason.index("+0.87%") > reason.index("HISTORICAL")

    def test_frozen_retired_read_alone_is_not_graded_as_this_week(self, s3):
        """Producer without the live block: N/A naming the history, never WATCH."""
        comp = _scanner(s3, {"scanner_lift": RETIRED})
        assert comp["status"] == "N/A-MISSING-INPUT"
        assert comp["value"] is None
        assert "not this week's scanner" in comp["status_reason"]
        assert "2026-04-12..2026-07-17" in comp["status_reason"]
        assert comp["arm"] == LIVE_ARM

    def test_failed_live_block_says_why(self, s3):
        live = {"status": "insufficient_data", "reason": "no candidates.json artifact "
                "carries a scanner_eval_log", "arm": LIVE_ARM}
        comp = _scanner(s3, {"scanner_lift": RETIRED, "scanner_lift_live": live})
        assert comp["status"] == "N/A-MISSING-INPUT"
        assert "scanner_lift_live insufficient_data" in comp["status_reason"]

    def test_legacy_artifact_without_stamp_still_grades_as_before(self, s3):
        legacy = {k: v for k, v in RETIRED.items() if k != "source_freshness"}
        comp = _scanner(s3, {"scanner_lift": legacy})
        assert comp["value"] == pytest.approx(RETIRED_EDGE)
        assert comp["arm"] == RETIRED_ARM

    def test_trend_never_splices_the_retired_arm_onto_the_live_one(self, s3):
        def card(value, arm):
            return {"tiles": {"research": {"components": [
                {"name": "scanner", "value": value, "status": "WATCH", "arm": arm}]}}}
        cards = [card(0.0367, RETIRED_ARM), card(0.0367, RETIRED_ARM), card(0.11, LIVE_ARM)]
        history = CardHistory(_extract_series(cards), len(cards),
                              _extract_series(cards, by_arm=True))
        comp = _scanner(s3, {"scanner_lift": RETIRED, "scanner_lift_live": LIVE}, history)
        assert comp["trend_4w"] == pytest.approx([0.11, LIVE_EDGE])


class TestCounterfactualLabels:
    ATT = {
        "status": "ok", "horizon_days": 21,
        "composite_ic": {"date_ic_mean": 0.01, "date_ic_p": 0.5, "n_eval_dates": 27},
        "counterfactual": {
            "top_n": [{"n": 20, "sector_balanced": True, "capture_rate": 0.02, "mean_alpha": 0.0005,
                       "n_cycles": 7, "population_mean_alpha": 0.00739,
                       "excess_vs_population": -0.0069, "excess_t": -0.99, "excess_p": 0.36,
                       "excess_ci95": [-0.024, 0.010]}],
            "live_gate": {"capture_rate": 0.0639, "mean_alpha": 0.02644, "n_survivors": 332,
                          "population_mean_alpha": 0.00739, "excess_vs_population": 0.01905,
                          "excess_t": 2.37, "excess_p": 0.0559},
            "n_cycles": 7,
            "holding_rule": "weekly rebalance, 21d hold, equal-weight, no intra-period turnover",
            "source_freshness": RETIRED["source_freshness"] | {
                "measurement": "attractiveness_eval.counterfactual"},
        },
    }

    def _comp(self, s3, name):
        _put(s3, "attractiveness_eval.json", self.ATT)
        tile = build_research_tile(BUCKET, RUN_DATE, s3_client=s3)
        return next(c for c in tile["components"] if c["name"] == name)

    def test_live_gate_leg_is_named_as_the_retired_arm(self, s3):
        """I11155 read live_gate vs-population +0.0191 as the live feed's own read."""
        reason = self._comp(s3, "scanner_feed_counterfactual")["status_reason"]
        assert f"arm {RETIRED_ARM}, not the live feed" in reason
        assert "scanner_evaluations (newest 2026-07-17, 78d old, STALE" in reason

    def test_basket_states_its_cycles_come_from_the_frozen_table(self, s3):
        comp = self._comp(s3, "scanner_basket_return")
        assert "scanner_evaluations (newest 2026-07-17" in comp["status_reason"]
        assert comp["value"] == pytest.approx(-0.0069)  # grading unchanged


class TestScorecardScanner:
    def test_grades_the_live_cut(self):
        out = _grade_scanner({"scanner_lift": RETIRED, "scanner_lift_live": LIVE}, None)
        assert out["arm"] == LIVE_ARM
        assert out["detail"]["precision"] == "46.6%"
        assert "live_arm_graded_by" not in out["detail"]
        assert "8 matured of 13" in out["detail"]["cohorts"]

    def test_frozen_retired_read_alone_is_na(self):
        out = _grade_scanner({"scanner_lift": RETIRED}, None)
        assert out["letter"] == "N/A"
        assert out["grade"] is None
        assert "2026-04-12..2026-07-17" in out["reason"]
