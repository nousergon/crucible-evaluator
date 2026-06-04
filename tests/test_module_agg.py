"""Tests for grading/module_agg.py — hierarchical aggregation."""

from grading.metric_record import build_metric
from grading.module_agg import (
    build_tile,
    module_status,
    numeric_grade,
    overall_status,
)

SRC = "s3://b/x"


def _crit(status, *, p=None):
    return build_metric(
        name=f"c_{status}", module="m", metric_type="ratio", n_floor=60, value=1.0,
        n_samples=120, target=0.5, red_line=0.0, criticality="critical",
        source_path=SRC, status=status, bh_fdr_adjusted_p=p, reason="x",
    )


def _sup(status):
    return build_metric(
        name=f"s_{status}", module="m", metric_type="ratio", n_floor=60, value=1.0,
        n_samples=120, target=0.5, red_line=0.0, criticality="supporting",
        source_path=SRC, status=status, reason="x",
    )


def _diag(status):
    return build_metric(
        name=f"d_{status}", module="m", metric_type="ratio", n_floor=60, value=1.0,
        n_samples=120, criticality="diagnostic", source_path=SRC, status=status, reason="x",
    )


class TestModuleStatus:
    def test_empty_not_run(self):
        assert module_status([]) == "N/A-NOT-RUN"

    def test_critical_red_fails(self):
        assert module_status([_crit("RED"), _crit("GREEN")]) == "RED"

    def test_critical_not_impl_is_watch(self):
        assert module_status([_crit("N/A-NOT-IMPL"), _crit("GREEN")]) == "WATCH"

    def test_two_critical_watch_bh_significant_red(self):
        # Two tiny p-values → BH significant → RED.
        assert module_status([_crit("WATCH", p=0.001), _crit("WATCH", p=0.002)]) == "RED"

    def test_two_critical_watch_not_significant_watch(self):
        assert module_status([_crit("WATCH", p=0.9), _crit("WATCH", p=0.8)]) == "WATCH"

    def test_two_critical_watch_no_pvalues_watch(self):
        # No p-values → not BH-significant → WATCH, not RED.
        assert module_status([_crit("WATCH"), _crit("WATCH")]) == "WATCH"

    def test_supporting_red_is_watch(self):
        assert module_status([_crit("GREEN"), _sup("RED")]) == "WATCH"

    def test_all_green(self):
        assert module_status([_crit("GREEN"), _sup("GREEN"), _diag("GREEN")]) == "GREEN"

    def test_critical_na_is_watch(self):
        assert module_status([_crit("N/A-MISSING-INPUT"), _sup("GREEN")]) == "WATCH"

    def test_only_diagnostic_na_stays_green(self):
        assert module_status([_crit("GREEN"), _diag("N/A-LOW-N")]) == "GREEN"


class TestOverallStatus:
    def test_portfolio_red_cascades(self):
        assert overall_status({"portfolio_outcome": "RED", "research": "GREEN"}) == "RED"

    def test_cascade_module_red(self):
        assert overall_status({"portfolio_outcome": "GREEN", "predictor": "RED"}) == "RED"

    def test_portfolio_watch(self):
        assert overall_status({"portfolio_outcome": "WATCH", "research": "GREEN"}) == "WATCH"

    def test_two_watch(self):
        assert overall_status({"portfolio_outcome": "GREEN", "research": "WATCH", "executor": "WATCH"}) == "WATCH"

    def test_all_green(self):
        assert overall_status({"portfolio_outcome": "GREEN", "research": "GREEN"}) == "GREEN"

    def test_empty(self):
        assert overall_status({}) == "N/A-NOT-RUN"


class TestNumericGrade:
    def test_excludes_na_and_diagnostic(self):
        # One GREEN critical scorable; N/A + diagnostic excluded.
        comps = [_crit("GREEN"), _diag("GREEN"), _crit("N/A-NOT-IMPL")]
        g = numeric_grade(comps)
        assert g is not None and 0 <= g <= 100

    def test_red_drags_grade_down(self):
        green_only = numeric_grade([_crit("GREEN")])
        with_red = numeric_grade([_crit("GREEN"), _crit("RED")])
        assert with_red < green_only

    def test_all_na_returns_none(self):
        assert numeric_grade([_crit("N/A-NOT-IMPL"), _diag("GREEN")]) is None


class TestBuildTile:
    def test_shape(self):
        # supporting RED → module WATCH (a supporting WATCH would NOT escalate).
        tile = build_tile("portfolio_outcome", [_crit("GREEN"), _sup("RED")])
        assert tile["module"] == "portfolio_outcome"
        assert tile["status"] == "WATCH"
        assert tile["letter"] == "C"
        assert tile["n_components"] == 2
        assert len(tile["components"]) == 2
        # components are JSON-serializable dicts (model_dump mode=json).
        assert isinstance(tile["components"][0], dict)
        assert tile["components"][0]["name"]
