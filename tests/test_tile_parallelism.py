"""tests/test_tile_parallelism.py — the ten tiles build concurrently, and a
failing tile still fails the card (alpha-engine-config-I9702).

``build_report_card`` built its ten tiles strictly sequentially, so the
ReportCard stage's wall-clock was the SUM of ten independent network-bound
latencies and one slow tile set the whole stage's duration. This file pins the
two properties of the fix that a future refactor could quietly undo, in
OPPOSITE directions:

  * the builders actually overlap (a pool that serialises is the defect back);
  * the card's error contract is UNCHANGED — a tile that raises still raises,
    deterministically, in declaration order. Wrapping the pool in a
    swallow would drop a voter out of the composite denominator and RAISE the
    headline grade because a measurement broke, which is the inversion the
    I7210 failure-scores-zero rule exists to prevent.
"""

from __future__ import annotations

import threading
import time

import pytest

import grading.aggregate as aggregate
from tests.test_aggregate import BUCKET, RUN_DATE, _seed_full, s3  # noqa: F401

_TILE_BUILDER_NAMES = [
    "build_portfolio_outcome_tile",
    "build_predictor_tile",
    "build_research_tile",
    "build_executor_tile",
    "build_backtester_tile",
    "build_substrate_tile",
    "build_agent_tile",
    "build_behavioral_tile",
    "build_director_quality_tile",
    "build_contribution_lift_tile",
]


class TestTilesBuildConcurrently:
    def test_builders_overlap_in_time(self, s3, monkeypatch):  # noqa: F811
        """Peak concurrent builders > 1 — the whole point of the pool.

        Each builder is wrapped with a small delay so overlap is observable:
        under sequential execution the peak is 1 however slow each one is.
        """
        _seed_full(s3)
        lock = threading.Lock()
        state = {"inflight": 0, "peak": 0}

        def _wrap(fn):
            def _inner(*args, **kwargs):
                with lock:
                    state["inflight"] += 1
                    state["peak"] = max(state["peak"], state["inflight"])
                try:
                    time.sleep(0.05)
                    return fn(*args, **kwargs)
                finally:
                    with lock:
                        state["inflight"] -= 1
            return _inner

        for name in _TILE_BUILDER_NAMES:
            monkeypatch.setattr(aggregate, name, _wrap(getattr(aggregate, name)))

        card = aggregate.build_report_card(BUCKET, RUN_DATE, s3_client=s3)

        assert state["peak"] > 1, (
            "the ten tile builders ran sequentially — peak concurrency was 1"
        )
        assert state["peak"] <= aggregate._TILE_POOL_WORKERS
        assert set(card["tiles"]) == {
            "portfolio_outcome", "predictor", "research", "executor", "backtester",
            "substrate", "agent", "behavioral", "director_quality", "contribution_lift",
        }

    def test_every_declared_builder_is_published(self, s3):  # noqa: F811
        """Membership cannot drift between what is BUILT and what is PUBLISHED.

        Before the pool these were two separate literals (a build call and a
        dict entry); now the dict is a projection of the builder list, and this
        pins that they stay the same ten.
        """
        _seed_full(s3)
        card = aggregate.build_report_card(BUCKET, RUN_DATE, s3_client=s3)
        assert len(card["tiles"]) == len(_TILE_BUILDER_NAMES) == 10

    def test_a_raising_tile_still_fails_the_card(self, s3, monkeypatch):  # noqa: F811
        """Error semantics unchanged: no tile is swallowed by the pool."""
        _seed_full(s3)

        def _boom(*args, **kwargs):
            raise RuntimeError("tile exploded")

        monkeypatch.setattr(aggregate, "build_behavioral_tile", _boom)
        with pytest.raises(RuntimeError, match="tile exploded"):
            aggregate.build_report_card(BUCKET, RUN_DATE, s3_client=s3)

    def test_the_first_failing_tile_in_declaration_order_is_the_one_raised(
        self, s3, monkeypatch,  # noqa: F811
    ):
        """Deterministic, and the same one sequential execution would have
        surfaced — so an operator reading the SF failure sees a stable cause."""
        _seed_full(s3)

        def _boom(label):
            def _inner(*args, **kwargs):
                raise RuntimeError(label)
            return _inner

        # `research` is declared before `behavioral`; both raise.
        monkeypatch.setattr(aggregate, "build_research_tile", _boom("research-first"))
        monkeypatch.setattr(aggregate, "build_behavioral_tile", _boom("behavioral-later"))
        with pytest.raises(RuntimeError, match="research-first"):
            aggregate.build_report_card(BUCKET, RUN_DATE, s3_client=s3)


class TestRegimeIndexRidesOutOfTheCard:
    def test_index_is_popped_off_the_card_before_it_is_written(self, s3):  # noqa: F811
        """`_regime_index` is build output, not card content — same convention
        as `_threshold_leaderboard`. The handler pops it and persists it under
        its own `write` flag, so the pre-promotion canary still writes nothing."""
        _seed_full(s3)
        card = aggregate.build_report_card(BUCKET, RUN_DATE, s3_client=s3)
        assert "_regime_index" in card
        assert "_regime_index" not in card["tiles"]["portfolio_outcome"]
