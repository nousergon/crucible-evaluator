"""Drift guard: this repo's risk-ratio paths vs nousergon_lib (config-I7597).

`grading/tiles/portfolio_outcome.py`'s three annualized-ratio adapters all call
`nousergon_lib.quant.riskstats` now — `_ann_ir` was the last one still deriving
`mean / std * sqrt(252)` locally. This file pins that they agree with the
library over a fixed corpus including the degenerate series (zero volatility,
no downside days, n < 2), and that the tile's `nan`-for-undefined sentinel is
applied consistently across all three.

`grading/attestation.py` and `grading/self_test.py` deliberately do NOT call the
library: they are known-answer checks written out from the metric definition, so
a check that called the thing it checks would be vacuous. `test_attestation_
expectations_agree_with_the_library` below is the bridge between the two — it is
the one place the independent expectation and the library are required to meet.

CORPUS is kept byte-identical to
`nousergon-lib/tests/test_quant_riskstats_drift_corpus.py`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from nousergon_lib.quant.riskstats import sharpe_ratio, sortino_ratio

from grading.tiles.portfolio_outcome import _ann_ir, _ann_sharpe, _ann_sortino

# Keep byte-identical to the nousergon-lib copy.
CORPUS: dict[str, list[float]] = {
    "mixed": [0.01, -0.02, 0.015, -0.005, 0.03, -0.01, 0.0, 0.02, -0.03, 0.005],
    "all_positive": [0.01, 0.02, 0.005, 0.03, 0.015],
    "all_negative": [-0.01, -0.02, -0.005, -0.04],
    "all_zero": [0.0, 0.0, 0.0, 0.0, 0.0],
    "zero_vol_positive": [0.01] * 8,
    "zero_vol_negative": [-0.01] * 8,
    "two_obs": [0.01, -0.01],
    "single_obs": [0.02],
    "empty": [],
    "tiny_downside": [0.01, 0.02, 0.03, -1e-9],
}

_TOL = dict(rel=1e-9, abs=1e-12)


def _arr(name: str) -> np.ndarray:
    return np.asarray(CORPUS[name], dtype=float)


@pytest.mark.parametrize("name", sorted(CORPUS))
@pytest.mark.parametrize(
    "adapter,lib",
    [(_ann_sharpe, sharpe_ratio), (_ann_ir, sharpe_ratio), (_ann_sortino, sortino_ratio)],
)
def test_adapters_match_the_library(name, adapter, lib) -> None:
    got = adapter(_arr(name))
    want = lib(CORPUS[name])
    if want is None:
        assert math.isnan(got), f"{name}/{adapter.__name__}: expected nan, got {got}"
    else:
        assert got == pytest.approx(want, **_TOL), f"{name}/{adapter.__name__}"


def test_ir_is_sharpe_on_the_active_series() -> None:
    """IR and Sharpe are the same statistic on different inputs."""
    a = _arr("mixed")
    assert _ann_ir(a) == pytest.approx(_ann_sharpe(a), **_TOL)


def test_undefined_is_nan_never_a_measured_zero() -> None:
    for name in ("zero_vol_positive", "single_obs", "empty"):
        for adapter in (_ann_sharpe, _ann_ir, _ann_sortino):
            assert math.isnan(adapter(_arr(name))), f"{name}/{adapter.__name__}"
    # No downside day: Sortino undefined, Sharpe defined.
    assert math.isnan(_ann_sortino(_arr("all_positive")))
    assert not math.isnan(_ann_sharpe(_arr("all_positive")))


def test_sortino_uses_the_full_sample_denominator() -> None:
    """config-I7271's convention, pinned here so a silent switch fails."""
    r = CORPUS["mixed"]
    n, n_down = len(r), sum(1 for x in r if x < 0)
    got = _ann_sortino(_arr("mixed"))
    n_down_variant = sortino_ratio(r, denominator="downside") if _has_denominator() else None
    if n_down_variant is not None:
        assert got / n_down_variant == pytest.approx(math.sqrt(n / n_down), rel=1e-9)
    else:
        # Older pinned library without the `denominator` parameter — assert the
        # convention directly from the definition instead.
        mean = sum(r) / n
        dd = math.sqrt(sum(min(0.0, x) ** 2 for x in r) / n)
        assert got == pytest.approx(mean / dd * math.sqrt(252), **_TOL)


def _has_denominator() -> bool:
    import inspect

    return "denominator" in inspect.signature(sortino_ratio).parameters


def test_attestation_expectations_agree_with_the_library() -> None:
    """The one place the independent KAT and the library must meet.

    `grading/attestation.py` writes its expected Sharpe/Sortino out from the
    definition on purpose — a library that switched to a population sd would
    inflate every card with no other symptom. That independence is only useful
    if something checks the two agree TODAY.
    """
    from grading.attestation import FROZEN_RETURNS, _expected_sharpe, _expected_sortino

    r = list(FROZEN_RETURNS)
    assert _expected_sharpe() == pytest.approx(sharpe_ratio(r), **_TOL)
    assert _expected_sortino() == pytest.approx(sortino_ratio(r), **_TOL)
