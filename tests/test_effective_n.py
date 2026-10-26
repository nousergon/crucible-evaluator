"""grading/effective_n.py — one definition of effective N (alpha-engine-config-I11094 / -I10065)."""

from grading.effective_n import (
    PARITY_MIN_EFFECTIVE_N,
    declared_effective_n,
    non_overlapping_windows,
)


def test_formula_matches_the_producer():
    # crucible-backtester pipeline_common.push_predictor_rolling_metrics (I8701):
    # max(1, n_dates // forward_days), 0 when there are no dates.
    assert non_overlapping_windows(22, 21) == 1
    assert non_overlapping_windows(105, 21) == 5
    assert non_overlapping_windows(5, 21) == 1
    assert non_overlapping_windows(0, 21) == 0


def test_declared_value_wins_over_the_formula():
    assert declared_effective_n({"n_effective": 3, "n_prediction_dates": 210, "horizon_days": 21}) == 3


def test_dates_without_a_horizon_are_undeclared():
    assert declared_effective_n({"n_prediction_dates": 210}) is None
    assert declared_effective_n({"n_prediction_dates": 210}, horizon_days=21) == 10


def test_row_counts_are_never_an_effective_n():
    for block in ({"n": 3533}, {"n_rows": 689, "rolling_n": 689}, {}, None, "x"):
        assert declared_effective_n(block, horizon_days=21) is None


def test_malformed_counts_are_refused():
    for bad in (True, -1, 2.5, "4"):
        assert declared_effective_n({"n_effective": bad}) is None


def test_parity_floor_is_the_rulings_two():
    assert PARITY_MIN_EFFECTIVE_N == 2
