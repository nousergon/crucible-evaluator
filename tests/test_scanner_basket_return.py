"""scanner_basket_return (alpha-engine-config-I7213) — consumer-side tests.

The number Brian asked for directly: top-20 attractiveness-feed basket mean
21d alpha vs the PIT population mean, sector-neutral. Sourced from the same
``backtest/{date}/attractiveness_eval.json :: counterfactual`` block as
``attractiveness_ic`` / ``scanner_feed_counterfactual`` (tests in
``test_attractiveness_consumer_contract.py``), but reading fields
(``excess_vs_population``, ``population_mean_alpha``, ``excess_t``,
``excess_p``, ``excess_ci95``, ``holding_rule``) that the paired
``crucible-backtester`` producer PR adds — this consumer deploys FIRST, so it
must grade an honest, specific N/A-MISSING-INPUT (naming which field) until
the producer's first Saturday run, and self-activate on first emission.
"""

import json
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from grading.tiles.research import build_research_tile

BUCKET = "alpha-engine-research"
RUN_DATE = "2026-07-04"
FIXTURE_V1 = Path(__file__).parent / "fixtures" / "attractiveness_eval_v1.json"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _v1_fixture() -> dict:
    return json.loads(FIXTURE_V1.read_text())


def _put_att(s3, doc, date=RUN_DATE):
    s3.put_object(Bucket=BUCKET, Key=f"backtest/{date}/attractiveness_eval.json",
                  Body=json.dumps(doc).encode())


def _comp(tile, name):
    return next(c for c in tile["components"] if c["name"] == name)


# A doc carrying the full I7213 producer shape: n=20 rows (balanced +
# unbalanced) with excess_vs_population / population_mean_alpha / excess_t /
# excess_p / excess_ci95, plus the top-level holding_rule string.
def _doc_with_producer_fields() -> dict:
    doc = _v1_fixture()
    doc["counterfactual"]["holding_rule"] = (
        "weekly rebalance, 21d hold, equal-weight, no intra-period turnover"
    )
    doc["counterfactual"]["top_n"] += [
        {
            "n": 20, "sector_balanced": True, "capture_rate": 0.44,
            "mean_alpha_21d": 0.0135, "population_mean_alpha": 0.0058,
            "excess_vs_population": 0.0077, "excess_t": 3.12, "excess_p": 0.008,
            "excess_ci95": [0.0029, 0.0125], "n_cycles": 11,
        },
        {
            "n": 20, "sector_balanced": False, "capture_rate": 0.47,
            "mean_alpha_21d": 0.0151, "population_mean_alpha": 0.0058,
            "excess_vs_population": 0.0093, "excess_t": 3.40, "excess_p": 0.006,
            "excess_ci95": [0.0041, 0.0145], "n_cycles": 11,
        },
    ]
    return doc


class TestScannerBasketReturnWired:
    def test_grades_from_sector_balanced_top20_excess(self, s3):
        _put_att(s3, _doc_with_producer_fields())
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "scanner_basket_return")
        assert m["criticality"] == "critical"
        assert m["metric_type"] == "log_return"
        assert m["arm"] == (
            "scanner_attractiveness (live champion feed — universe-board "
            "attractiveness_score, config-I2994)"
        )
        assert m["value"] == pytest.approx(0.0077)
        assert m["n_samples"] == 11
        assert m["estimator"] == "date_clustered_mean_excess_vs_pit_population"
        assert m["measurement_horizon"] == "21d"
        # p=0.008 < 0.10 and N=11 >= 8 → significant.
        assert m["status"] == "GREEN"
        assert m["reliability"] == "high"
        # Raw (unbalanced) variant surfaced alongside, per I7213 spec.
        assert "raw (unbalanced) top-20: excess_vs_population=+0.0093" in m["status_reason"]
        assert "Holding rule: weekly rebalance" in m["status_reason"]
        assert "alpha-engine-config-I7213" in m["status_reason"]

    def test_insignificant_grades_watch_low_reliability(self, s3):
        doc = _doc_with_producer_fields()
        for row in doc["counterfactual"]["top_n"]:
            if row.get("n") == 20 and row.get("sector_balanced") is True:
                row["excess_p"] = 0.41
        _put_att(s3, doc)
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "scanner_basket_return")
        assert m["status"] == "WATCH"
        assert m["reliability"] == "low"
        assert "accumulating" in m["status_reason"]

    def test_negative_excess_grades_red(self, s3):
        doc = _doc_with_producer_fields()
        for row in doc["counterfactual"]["top_n"]:
            if row.get("n") == 20 and row.get("sector_balanced") is True:
                row["excess_vs_population"] = -0.004
                row["excess_ci95"] = [-0.009, 0.001]
        _put_att(s3, doc)
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "scanner_basket_return")
        assert m["status"] == "RED"


class TestScannerBasketReturnDegradesHonestly:
    def test_absent_artifact_grades_missing_input_naming_producer(self, s3):
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "scanner_basket_return")
        assert m["status"] == "N/A-MISSING-INPUT"
        assert "config#1389" in m["status_reason"]

    def test_artifact_without_n20_row_grades_missing_input_naming_row(self, s3):
        # Today's live shape: top_n carries only 10/25/50 — no n=20 cohort at
        # all until the producer PR ships.
        _put_att(s3, _v1_fixture())
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "scanner_basket_return")
        assert m["status"] == "N/A-MISSING-INPUT"
        assert "n=20, sector_balanced=True" in m["status_reason"]
        assert "alpha-engine-config-I7213" in m["status_reason"]

    def test_n20_row_present_but_missing_excess_field_names_it(self, s3):
        doc = _v1_fixture()
        doc["counterfactual"]["holding_rule"] = "weekly rebalance, 21d hold, equal-weight, no intra-period turnover"
        doc["counterfactual"]["top_n"].append(
            {"n": 20, "sector_balanced": True, "capture_rate": 0.44, "mean_alpha_21d": 0.0135}
        )
        _put_att(s3, doc)
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "scanner_basket_return")
        assert m["status"] == "N/A-MISSING-INPUT"
        assert "'excess_vs_population'" in m["status_reason"]

    def test_n20_row_missing_holding_rule_names_it(self, s3):
        doc = _v1_fixture()
        doc["counterfactual"]["top_n"].append(
            {
                "n": 20, "sector_balanced": True, "capture_rate": 0.44,
                "mean_alpha_21d": 0.0135, "population_mean_alpha": 0.0058,
                "excess_vs_population": 0.0077, "excess_t": 3.12, "excess_p": 0.008,
                "excess_ci95": [0.0029, 0.0125], "n_cycles": 11,
            }
        )
        _put_att(s3, doc)
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "scanner_basket_return")
        assert m["status"] == "N/A-MISSING-INPUT"
        assert "'holding_rule'" in m["status_reason"]

    def test_insufficient_data_status_grades_na(self, s3):
        _put_att(s3, {"schema_version": 1, "status": "insufficient_data",
                      "as_of": RUN_DATE, "horizon_days": 21})
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "scanner_basket_return")
        assert m["status"] == "N/A-MISSING-INPUT"
        assert "insufficient" in m["status_reason"]
