"""Tests for the `pbo` component of the Portfolio Outcome tile
(alpha-engine-config-I9684).

Two layers, deliberately:

1. A CONSUMER-CONTRACT layer over the shared CSCV engine
   (`nousergon_lib.quant.stats.pbo.cscv_pbo`) against the two cases with a
   KNOWN answer — a strictly dominant configuration (PBO → 0) and pure noise
   (PBO → 0.5). The evaluator grades against that engine's output, so a silent
   change in its semantics must fail here rather than on the live card.
2. A WIRING layer over `_build_pbo_component`, which owns the honest-N/A
   posture: a missing leaderboard, a leaderboard with no verdict, an
   underpowered split count, and the two graded outcomes.
"""

import json

import boto3
import numpy as np
import pytest
from moto import mock_aws

from grading.tiles.portfolio_outcome import (
    MODEL_ZOO_LEADERBOARD_KEY,
    _PBO_MIN_SPLITS,
    _build_pbo_component,
)

BUCKET = "alpha-engine-research"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _put_leaderboard(s3, payload):
    s3.put_object(
        Bucket=BUCKET,
        Key=MODEL_ZOO_LEADERBOARD_KEY,
        Body=json.dumps(payload).encode("utf-8"),
    )


def _block(**over):
    base = {
        "status": "ok",
        "n_splits": 44,
        "n_specs": 5,
        "spec_ids": ["a", "b", "c", "d", "e"],
        "pbo": 0.09,
        "mean_logit": 0.97,
        "selected_counts": {"a": 30, "b": 14},
        "pbo_target": 0.2,
        "pbo_pass": True,
    }
    base.update(over)
    return base


# ── 1. Known-answer contract over the shared CSCV engine ────────────────────
class TestCscvEngineKnownAnswers:
    def test_strictly_dominant_config_gives_pbo_zero(self):
        """One column is better than every other on EVERY split. The in-sample
        winner is therefore also the out-of-sample top rank at every held-out
        split, so it never lands in the bottom half: PBO == 0."""
        from nousergon_lib.quant.stats.pbo import cscv_pbo

        rng = np.random.default_rng(7)
        m = rng.normal(0.0, 0.01, size=(40, 6))
        m[:, 0] = 1.0  # strictly dominant on every split
        out = cscv_pbo(m)
        assert out["status"] == "ok"
        assert out["pbo"] == 0.0
        assert out["selected_counts"] == {0: 40}

    def test_pure_noise_gives_pbo_near_one_half(self):
        """With no real signal the in-sample winner is a noise winner, so its
        out-of-sample rank is uniform and PBO → 0.5."""
        from nousergon_lib.quant.stats.pbo import cscv_pbo

        rng = np.random.default_rng(11)
        out = cscv_pbo(rng.normal(size=(400, 20)))
        assert out["status"] == "ok"
        assert 0.4 <= out["pbo"] <= 0.6

    def test_engine_declares_min_splits_four(self):
        """The component's n_floor is the ENGINE's declared floor, not a number
        the evaluator picked. If the engine's default moves, this fails."""
        import inspect

        from nousergon_lib.quant.stats.pbo import cscv_pbo

        assert (
            inspect.signature(cscv_pbo).parameters["min_splits"].default
            == _PBO_MIN_SPLITS
        )


# ── 2. Component wiring / honest-N/A posture ────────────────────────────────
class TestPboComponent:
    def test_missing_leaderboard_is_not_impl_naming_the_producer(self, s3):
        c = _build_pbo_component(BUCKET, s3_client=s3).model_dump()
        assert c["name"] == "pbo"
        assert c["status"] == "N/A-NOT-IMPL"
        assert c["value"] is None
        assert MODEL_ZOO_LEADERBOARD_KEY in c["status_reason"]
        assert "ModelZooSelect" in c["status_reason"]

    def test_leaderboard_without_verdict_is_not_impl(self, s3):
        _put_leaderboard(s3, {"date": "2026-08-28", "candidates": []})
        c = _build_pbo_component(BUCKET, s3_client=s3).model_dump()
        assert c["status"] == "N/A-NOT-IMPL"
        assert "selection_pbo" in c["status_reason"]

    def test_engine_insufficient_is_low_n_not_a_number(self, s3):
        _put_leaderboard(s3, {"date": "2026-08-28", "selection_pbo": {
            "status": "insufficient", "reason": "needs >=2 aligned combos",
            "n_splits": 0, "n_specs": 1, "pbo": None}})
        c = _build_pbo_component(BUCKET, s3_client=s3).model_dump()
        assert c["status"] == "N/A-LOW-N"
        assert c["value"] is None
        assert "needs >=2 aligned combos" in c["status_reason"]

    def test_below_split_floor_is_underpowered_never_green(self, s3):
        _put_leaderboard(s3, {"date": "2026-08-28",
                              "selection_pbo": _block(n_splits=_PBO_MIN_SPLITS - 1, pbo=0.0)})
        c = _build_pbo_component(BUCKET, s3_client=s3).model_dump()
        assert c["status"] == "N/A-LOW-N"
        assert "UNDERPOWERED" in c["status_reason"]

    def test_clears_the_declared_bar_is_green(self, s3):
        _put_leaderboard(s3, {"date": "2026-08-28", "selection_pbo": _block(pbo=0.09)})
        c = _build_pbo_component(BUCKET, s3_client=s3).model_dump()
        assert c["status"] == "GREEN"
        assert c["value"] == pytest.approx(0.09)
        assert c["n_samples"] == 44
        assert c["n_floor"] == _PBO_MIN_SPLITS
        # The bar comes from the registry (SYSTEM_OPTIMIZED.md §12), not source.
        assert c["target"] == pytest.approx(0.2)
        assert c["red_line"] is None

    def test_misses_the_declared_bar_is_watch_never_red(self, s3):
        """§12 declares a target and NO red line. A miss must not manufacture a
        RED against a bar nobody wrote — and must not be mis-oriented into a
        higher-is-better GREEN either."""
        _put_leaderboard(s3, {"date": "2026-08-28", "selection_pbo": _block(pbo=0.62)})
        c = _build_pbo_component(BUCKET, s3_client=s3).model_dump()
        assert c["status"] == "WATCH"
        assert c["value"] == pytest.approx(0.62)
        assert "does NOT clear" in c["status_reason"]

    def test_degenerate_selection_is_stated_in_the_reason(self, s3):
        """A single spec winning every split makes PBO structurally low. The
        card must say so beside the number rather than publish a bare GREEN."""
        _put_leaderboard(s3, {"date": "2026-08-28", "selection_pbo": _block(
            pbo=0.0, n_specs=3, selected_counts={"champion-arch": 44},
            dropped_misaligned_specs=["horizon-60d", "horizon-90d"])})
        c = _build_pbo_component(BUCKET, s3_client=s3).model_dump()
        assert c["status"] == "GREEN"
        r = c["status_reason"]
        assert "DEGENERATE" in r
        assert "n_specs=3" in r
        assert "horizon-60d" in r
        assert "2026-08-28" in r

    def test_ungraded_registry_row_raises_never_silent_green(self, s3, monkeypatch):
        """A null target would make every branch fall through to an
        unconditional pass. Grading against no bar must raise."""
        import grading.tiles.portfolio_outcome as po
        from grading.metric_record import MetricContractError

        class _NoBar:
            target = None
            red_line = None

        monkeypatch.setattr(po, "resolve_band", lambda *a, **k: _NoBar())
        _put_leaderboard(s3, {"date": "2026-08-28", "selection_pbo": _block()})
        with pytest.raises(MetricContractError, match="declares no target"):
            po._build_pbo_component(BUCKET, s3_client=s3)

    def test_corrupt_leaderboard_raises_never_renders_a_number(self, s3):
        s3.put_object(Bucket=BUCKET, Key=MODEL_ZOO_LEADERBOARD_KEY, Body=b"{not json")
        with pytest.raises(ValueError, match="not valid JSON"):
            _build_pbo_component(BUCKET, s3_client=s3)


class TestPboOnTheTile:
    def test_pbo_is_on_the_tile_even_when_eod_pnl_is_absent(self, s3):
        """pbo's input is the model-zoo leaderboard, not eod_pnl.csv. A missing
        EOD export says nothing about whether the rotation computed a CSCV
        verdict, so pbo must still carry its real value."""
        from grading.tiles.portfolio_outcome import build_portfolio_outcome_tile

        _put_leaderboard(s3, {"date": "2026-08-28", "selection_pbo": _block(pbo=0.09)})
        tile = build_portfolio_outcome_tile(BUCKET, s3_client=s3)
        pbo = next(c for c in tile["components"] if c["name"] == "pbo")
        assert pbo["status"] == "GREEN"
        assert pbo["value"] == pytest.approx(0.09)
