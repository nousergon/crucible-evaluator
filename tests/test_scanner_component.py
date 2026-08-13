"""`_grade_scanner` — arm labelling, canonical horizon, reachable recall.

alpha-engine-config-I2318 / I1458. Fixtures are the MEASURED live
``scanner_lift`` block from ``s3://alpha-engine-research/backtest/2026-08-07/
e2e_lift.json`` so the regressions are pinned to real values, not invented ones.
"""

import pytest

from grading.scorecard import (
    _grade_scanner,
    _max_achievable_recall,
    _selection_edge_pp,
)

# Verbatim from the live 2026-08-07 artifact.
LIVE_SCANNER_LIFT = {
    "universe_avg": 0.0008,
    "passing_avg": -0.0024,
    "lift": -0.0031,
    "n_universe": 15274,
    "n_passing": 653,
    "classification": {
        "precision": 0.441, "recall": 0.0437, "f1": 0.0795, "accuracy": 0.5634,
        "tp": 288, "fp": 365, "fn": 6304, "tn": 8317, "n": 15274,
    },
    "classification_21d": {
        "precision": 0.4925, "recall": 0.0412, "f1": 0.076, "accuracy": 0.5263,
        "tp": 262, "fp": 270, "fn": 6103, "tn": 6818, "n": 13453,
    },
    "arm": "tech_score_baseline (retired from live feed 2026-06-29)",
}


def _e2e(sl=None):
    return {"scanner_lift": sl if sl is not None else LIVE_SCANNER_LIFT}


class TestSelectionEdge:
    def test_edge_is_precision_over_base_rate(self):
        # 21d: base rate (262+6103)/13453 = 47.31%, precision 49.25% -> +1.94pp
        edge = _selection_edge_pp(LIVE_SCANNER_LIFT["classification_21d"])
        assert edge == pytest.approx(1.94, abs=0.02)

    def test_5d_edge_is_near_zero_despite_44pct_precision(self):
        """44.1% precision against a 43.2% base rate is ~nothing."""
        edge = _selection_edge_pp(LIVE_SCANNER_LIFT["classification"])
        assert edge == pytest.approx(0.94, abs=0.02)

    def test_none_on_malformed_or_empty(self):
        assert _selection_edge_pp(None) is None
        assert _selection_edge_pp({"precision": 0.5}) is None
        assert _selection_edge_pp({"tp": 0, "fp": 0, "fn": 0, "tn": 0}) is None


class TestRecallReachability:
    def test_live_recall_band_is_unreachable(self):
        """653 selections / 6592 positives caps recall at 9.9% < 10% baseline."""
        assert _max_achievable_recall(LIVE_SCANNER_LIFT["classification"]) == pytest.approx(
            0.0991, abs=0.001
        )

    def test_recall_not_graded_and_says_why(self):
        out = _grade_scanner(_e2e(), None)
        assert out["detail"]["recall_graded"] is False
        assert "below the 10% grading baseline" in out["detail"]["recall_not_graded_reason"]
        # The measured recall is still reported — suppressed from GRADING only.
        assert "recall" in out["detail"]

    def test_recall_is_graded_when_the_band_is_reachable(self):
        """A less selective filter gets its recall graded normally."""
        sl = dict(LIVE_SCANNER_LIFT)
        sl["classification_21d"] = {
            "precision": 0.6, "recall": 0.30, "f1": 0.4,
            "tp": 300, "fp": 200, "fn": 700, "tn": 800,
        }
        out = _grade_scanner(_e2e(sl), None)
        assert _max_achievable_recall(sl["classification_21d"]) == pytest.approx(0.5)
        assert out["detail"].get("recall_graded") is not False
        assert "recall_not_graded_reason" not in out["detail"]


class TestArmLabelling:
    def test_arm_is_carried_to_detail_and_top_level(self):
        out = _grade_scanner(_e2e(), None)
        assert out["arm"] == LIVE_SCANNER_LIFT["arm"]
        assert out["detail"]["arm"] == LIVE_SCANNER_LIFT["arm"]

    def test_points_at_the_component_that_grades_the_live_arm(self):
        out = _grade_scanner(_e2e(), None)
        assert "attractiveness_ic" in out["detail"]["live_arm_graded_by"]

    def test_absent_arm_does_not_fabricate_one(self):
        sl = {k: v for k, v in LIVE_SCANNER_LIFT.items() if k != "arm"}
        out = _grade_scanner(_e2e(sl), None)
        assert "arm" not in out
        assert "arm" not in out["detail"]


class TestHorizon:
    def test_grades_the_canonical_21d_block(self):
        out = _grade_scanner(_e2e(), None)
        assert out["detail"]["horizon"] == "21d"
        assert out["detail"]["precision"] == "49.2%"
        # 5d survives as a labelled diagnostic.
        assert out["detail"]["precision_5d_diagnostic"] == "44.1%"

    def test_falls_back_to_5d_when_21d_absent(self):
        sl = {k: v for k, v in LIVE_SCANNER_LIFT.items() if k != "classification_21d"}
        out = _grade_scanner(_e2e(sl), None)
        assert out["detail"]["horizon"] == "5d"
        assert out["detail"]["precision"] == "44.1%"


class TestDisplay:
    def test_lift_no_longer_renders_a_real_edge_as_zero(self):
        """-0.0031 is -0.31pp. It used to print '-0.00%'."""
        out = _grade_scanner(_e2e(), None)
        assert out["detail"]["lift"] == "-0.31%"

    def test_universe_basis_is_stated(self):
        out = _grade_scanner(_e2e(), None)
        assert "eval_date" in out["detail"]["n_universe_basis"]

    def test_base_rate_is_published_next_to_precision(self):
        out = _grade_scanner(_e2e(), None)
        assert out["detail"]["base_rate"] == "47.3%"


class TestGrade:
    def test_insufficient_data_still_returns_na(self):
        assert _grade_scanner({"scanner_lift": {}}, None)["letter"] == "N/A"
        assert _grade_scanner(None, None)["letter"] == "N/A"

    def test_grade_is_finite_and_bounded(self):
        out = _grade_scanner(_e2e(), None)
        assert 0.0 <= out["grade"] <= 100.0

    def test_a_genuinely_strong_scanner_outgrades_the_live_one(self):
        """Directional sanity: the composite must move on real skill."""
        strong = dict(LIVE_SCANNER_LIFT)
        strong["classification_21d"] = {
            "precision": 0.70, "recall": 0.30, "f1": 0.42,
            "tp": 350, "fp": 150, "fn": 800, "tn": 1000,
        }
        strong["lift"] = 0.02
        assert _grade_scanner(_e2e(strong), None)["grade"] > _grade_scanner(_e2e(), None)["grade"]
