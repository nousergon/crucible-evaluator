"""alpha-engine-config-I12354 — frozen reliability-LOW research ICs grade N/A-stale.

Brian's ruling (2026-10-16 00:09Z, config#11498): ``judge_outcome_ic``,
``attractiveness_trajectory_ic`` and ``scanner_basket_return`` cannot
accumulate any more, so they must not present as live levels ("WATCH,
accumulating"). Stale ``scanner_evaluations`` (the producer's own
``source_freshness.stale`` stamp) or an N unchanged across the last report
cards grades them N/A-stale (frozen); a growing one still grades normally.
"""

import json
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from grading.history import CardHistory, _extract_n_series
from grading.tiles.research import build_research_tile

BUCKET = "alpha-engine-research"
RUN_DATE = "2026-10-16"
FIXTURE = Path(__file__).parent / "fixtures" / "attractiveness_eval_v1.json"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _put(s3, name, doc):
    s3.put_object(Bucket=BUCKET, Key=f"backtest/{RUN_DATE}/{name}", Body=json.dumps(doc).encode())


def _comp(tile, name):
    return next(c for c in tile["components"] if c["name"] == name)


def _att(*, n_cycles=7, traj_dates=2, stale=False) -> dict:
    doc = json.loads(FIXTURE.read_text())
    doc["trajectory_ic"]["pre_repricing_score"].update(
        n_eval_dates=traj_dates, date_ic_p=None, status="accruing")
    cf = doc["counterfactual"]
    cf["holding_rule"] = "weekly rebalance, 21d hold, equal-weight"
    cf["top_n"].append({
        "n": 20, "sector_balanced": True, "capture_rate": 0.44, "mean_alpha_21d": 0.0135,
        "population_mean_alpha": 0.0058, "excess_vs_population": 0.0077, "excess_t": 1.1,
        "excess_p": 0.30, "excess_ci95": [-0.005, 0.02], "n_cycles": n_cycles,
    })
    if stale:
        cf["source_freshness"] = {"stale": True, "run_date": RUN_DATE, "sources": [{
            "table": "scanner_evaluations", "newest_date": "2026-07-17", "age_days": 91,
            "stale": True, "retired_date": "2026-07-12"}]}
    return doc


def _aq(n_dates=12) -> dict:
    return {"status": "ok", "judge_outcome_ic": {
        "status": "ok", "horizon_days": 21, "n_unattributable": 5780,
        "overall": {"date_ic_mean": 0.01, "date_ic_p": 0.6, "n_eval_dates": n_dates,
                    "pooled_ic": 0.01, "pooled_ic_p": 0.6, "n": 170}}}


def _history(**ns) -> CardHistory:
    """A CardHistory whose prior cards carried each metric at the given N list."""
    series = {}
    for name, seq in ns.items():
        series[("research", name)] = [(f"2026-09-{18 + 7 * i:02d}", n) for i, n in enumerate(seq)]
    return CardHistory({}, 3, None, series)


def _put_all(s3, **kw):
    _put(s3, "attractiveness_eval.json", _att(**{k: v for k, v in kw.items() if k != "aq"}))
    _put(s3, "agent_quality.json", _aq(kw.get("aq", 12)))


def _assert_frozen(m):
    assert m["status"] == "N/A-NOT-IMPL"
    assert m["derived_letter"] == "N/A"
    assert m["value"] is None
    assert m["permanent_na"] is True          # leaves the coverage denominator
    assert m["criticality"] == "diagnostic"   # cannot drag the tile
    assert m["status_reason"].startswith("N/A-stale (frozen)")
    assert "accumulating" not in m["status_reason"].replace("not accumulating", "")


class TestScannerBasketReturn:
    def test_stale_source_grades_frozen_and_names_the_dead_source(self, s3):
        _put_all(s3, stale=True)
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "scanner_basket_return")
        _assert_frozen(m)
        assert m["n_samples"] == 7
        assert "scanner_evaluations" in m["status_reason"]
        assert "2026-07-17" in m["status_reason"] and "2026-07-12" in m["status_reason"]

    def test_unchanged_n_grades_frozen_without_a_freshness_stamp(self, s3):
        _put_all(s3)
        h = _history(scanner_basket_return=[7, 7])
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3, history=h), "scanner_basket_return")
        _assert_frozen(m)
        assert "unchanged since the 2026-09-18 card" in m["status_reason"]

    def test_growing_still_grades_watch_accumulating(self, s3):
        _put_all(s3)
        h = _history(scanner_basket_return=[5, 6])
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3, history=h), "scanner_basket_return")
        assert m["status"] == "WATCH"
        assert m["criticality"] == "critical"
        assert "WATCH, accumulating" in m["status_reason"]
        assert not m.get("permanent_na")


class TestAttractivenessTrajectoryIc:
    def test_unchanged_n_grades_frozen(self, s3):
        _put_all(s3)
        h = _history(attractiveness_trajectory_ic=[2, 2, 2])
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3, history=h),
                  "attractiveness_trajectory_ic")
        _assert_frozen(m)
        assert m["n_samples"] == 2
        assert "unchanged since the 2026-09-18 card" in m["status_reason"]

    def test_growing_still_grades_watch_accumulating(self, s3):
        _put_all(s3, traj_dates=3)
        h = _history(attractiveness_trajectory_ic=[1, 2])
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3, history=h),
                  "attractiveness_trajectory_ic")
        assert m["status"] == "WATCH"
        assert "WATCH, accumulating" in m["status_reason"]


class TestJudgeOutcomeIc:
    def test_unchanged_n_grades_frozen_and_names_the_unattributable_joins(self, s3):
        _put_all(s3)
        h = _history(judge_outcome_ic=[12, 12])
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3, history=h), "judge_outcome_ic")
        _assert_frozen(m)
        assert m["n_samples"] == 12
        assert "5780" in m["status_reason"]

    def test_growing_still_grades_watch_accumulating(self, s3):
        _put_all(s3, aq=13)
        h = _history(judge_outcome_ic=[11, 12])
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3, history=h), "judge_outcome_ic")
        assert m["status"] == "WATCH"
        assert "WATCH, accumulating" in m["status_reason"]


class TestNoHistoryAndRecovery:
    def test_no_history_keeps_prior_grading_for_n_based_metrics(self, s3):
        _put_all(s3)
        tile = build_research_tile(BUCKET, RUN_DATE, s3_client=s3)
        for name in ("judge_outcome_ic", "attractiveness_trajectory_ic", "scanner_basket_return"):
            assert _comp(tile, name)["status"] == "WATCH"

    def test_one_prior_card_is_not_enough_to_call_it_frozen(self, s3):
        _put_all(s3)
        h = _history(judge_outcome_ic=[12])
        assert _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3, history=h),
                     "judge_outcome_ic")["status"] == "WATCH"

    def test_extract_n_series_keeps_na_weeks_and_orders_by_date(self):
        cards = [
            ("2026-10-02", {"tiles": {"research": {"components": [
                {"name": "judge_outcome_ic", "value": None, "status": "N/A-NOT-IMPL", "n_samples": 12}]}}}),
            ("2026-10-09", {"tiles": {"research": {"components": [
                {"name": "judge_outcome_ic", "value": None, "status": "N/A-NOT-IMPL", "n_samples": 12},
                {"name": "other", "value": 1.0, "n_samples": None}]}}}),
        ]
        out = _extract_n_series(cards)
        assert out == {("research", "judge_outcome_ic"): [("2026-10-02", 12), ("2026-10-09", 12)]}
