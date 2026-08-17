"""The threshold slot's cohort, challenger arm, scoring and promotion CLI.

The load-bearing assertions are the ones about what happens when there is NOT
enough data — which is the real state today (17 cards on 2026-08-16 against a
26-paired-card floor). `insufficient` must render, must carry its counts, and
must never be mistaken for a pass or for absence (champion-challenger §5.1,
§7.2).
"""

from __future__ import annotations

import json

import pytest

from grading.thresholds.challenger import propose_bands
from grading.thresholds.cohort import Cohort, CardRow, Cell, load_cohort
from grading.thresholds.promote import (
    PromotionRefused,
    check_evidence,
    experiments_entry,
    swap_champion,
)
from grading.thresholds.registry import load_registry
from grading.thresholds.scoring import (
    arm_metric_name,
    build_arm_components,
    leaderboard_key,
    score_slot,
)

METRIC = ("portfolio_outcome", "sharpe_ratio")
BUCKET = "alpha-engine-research-test"


def _card(date: str, *, sharpe: float, alpha_level: float) -> CardRow:
    return CardRow(
        date=date,
        cells={METRIC: Cell(value=sharpe, n_samples=200, n_floor=60)},
        objective_level=alpha_level,
    )


def _cohort(rows, horizon=1) -> Cohort:
    return Cohort(rows=rows, horizon_cycles=horizon, n_cards_loaded=len(rows))


class TestCohortPairing:
    def test_objective_is_the_delta_of_cumulative_alpha(self):
        c = _cohort([_card("2026-01-03", sharpe=1.2, alpha_level=0.10),
                     _card("2026-01-10", sharpe=1.1, alpha_level=0.13)])
        assert c.objective(0) == pytest.approx(0.03)

    def test_unpaired_tail_card_yields_no_observation(self):
        c = _cohort([_card("2026-01-03", sharpe=1.2, alpha_level=0.10),
                     _card("2026-01-10", sharpe=1.1, alpha_level=0.13)])
        assert c.objective(1) is None
        assert c.paired_indices() == [0]

    def test_missing_objective_anchor_is_not_zero_filled(self):
        rows = [_card("2026-01-03", sharpe=1.2, alpha_level=0.10),
                CardRow(date="2026-01-10", cells={}, objective_level=None),
                _card("2026-01-17", sharpe=1.0, alpha_level=0.14)]
        c = _cohort(rows, horizon=1)
        assert c.objective(0) is None      # the far end of the window is missing
        assert c.objective(1) is None      # ...and so is the near end
        assert c.n_paired == 0             # an unpaired card, never a zero

        # At a 2-cycle horizon the gap is stepped over rather than filled.
        assert _cohort(rows, horizon=2).objective(0) == pytest.approx(0.04)

    def test_load_cohort_reads_prior_cards_from_s3(self, tmp_path):
        pytest.importorskip("moto")
        import boto3
        from moto import mock_aws

        with mock_aws():
            s3 = boto3.client("s3", region_name="us-east-1")
            s3.create_bucket(Bucket=BUCKET)
            for date, alpha in (("2026-01-03", 0.10), ("2026-01-10", 0.13)):
                s3.put_object(
                    Bucket=BUCKET, Key=f"evaluator/{date}/report_card.json",
                    Body=json.dumps({"tiles": {"portfolio_outcome": {"components": [
                        {"name": "alpha_vs_spy", "value": alpha, "status": "GREEN",
                         "n_samples": 200, "n_floor": 60},
                        {"name": "sharpe_ratio", "value": 1.1, "status": "GREEN",
                         "n_samples": 200, "n_floor": 60},
                    ]}}}).encode(),
                )
            cohort = load_cohort(BUCKET, "2026-01-17", s3_client=s3)
        assert cohort.dates == ["2026-01-03", "2026-01-10"]
        assert cohort.rows[0].objective_level == pytest.approx(0.10)
        assert ("portfolio_outcome", "sharpe_ratio") in cohort.rows[0].cells

    def test_na_readings_are_skipped_never_zero_filled(self):
        pytest.importorskip("moto")
        import boto3
        from moto import mock_aws

        with mock_aws():
            s3 = boto3.client("s3", region_name="us-east-1")
            s3.create_bucket(Bucket=BUCKET)
            s3.put_object(
                Bucket=BUCKET, Key="evaluator/2026-01-03/report_card.json",
                Body=json.dumps({"tiles": {"portfolio_outcome": {"components": [
                    {"name": "sharpe_ratio", "value": None, "status": "N/A-LOW-N"},
                    {"name": "psr", "value": 0.9, "status": "N/A-MISSING-INPUT"},
                ]}}}).encode(),
            )
            cohort = load_cohort(BUCKET, "2026-01-17", s3_client=s3)
        assert cohort.rows[0].cells == {}
        assert cohort.rows[0].objective_level is None
        assert cohort.warnings and "cannot anchor the objective" in cohort.warnings[0]


class TestChallengerAbstains:
    def test_short_history_is_insufficient_with_its_counts(self):
        rows = [_card(f"2026-01-{d:02d}", sharpe=1.0 + i * 0.01, alpha_level=0.01 * i)
                for i, d in enumerate(range(3, 25, 7))]
        proposals = propose_bands(_cohort(rows))
        band = proposals[METRIC]
        assert band.status == "insufficient"
        assert not band.usable
        assert band.target is None and band.red_line is None
        assert "paired card" in band.reason and "floor" in band.reason

    def test_seventeen_cards_is_below_the_floor(self):
        """The measured state on 2026-08-16 — and the correct output is `insufficient`."""
        rows = [_card(f"2026-{(i // 4) + 1:02d}-{(i % 4) * 7 + 1:02d}",
                      sharpe=1.0 + 0.01 * i, alpha_level=0.01 * i) for i in range(17)]
        band = propose_bands(_cohort(rows))[METRIC]
        assert band.status == "insufficient"
        assert load_registry().slot.n_floor_cards > 17

    def test_degenerate_separation_proposes_nothing(self):
        reg = _relaxed(load_registry(), floor_cards=3, floor_status=2)
        rows = []
        level = 0.0
        # A HIGHER sharpe consistently precedes a WORSE objective — the metric
        # is anti-predictive, so no band may be invented.
        for i, sh in enumerate([2.0, 0.5, 2.2, 0.4, 2.1, 0.6]):
            rows.append(_card(f"2026-01-{i + 1:02d}", sharpe=sh, alpha_level=level))
            level += -0.01 if sh > 1 else 0.01
        rows.append(_card("2026-02-01", sharpe=1.0, alpha_level=level))
        band = propose_bands(_cohort(rows), reg)[METRIC]
        assert band.status == "degenerate"
        assert band.target is None and band.red_line is None
        assert "did not separate" in band.reason

    def test_bands_are_proposed_when_the_metric_separates(self):
        reg = _relaxed(load_registry(), floor_cards=3, floor_status=2)
        rows = []
        level = 0.0
        for i, sh in enumerate([2.0, 0.5, 2.2, 0.4, 2.1, 0.6]):
            rows.append(_card(f"2026-01-{i + 1:02d}", sharpe=sh, alpha_level=level))
            level += 0.01 if sh > 1 else -0.01
        rows.append(_card("2026-02-01", sharpe=1.0, alpha_level=level))
        band = propose_bands(_cohort(rows), reg)[METRIC]
        assert band.status == "proposed"
        assert band.target > band.red_line
        assert band.n_positive >= 2 and band.n_negative >= 2


def _relaxed(registry, *, floor_cards: int, floor_status: int):
    """A registry copy with lowered floors — for exercising the SCORED path.

    The real floors are deliberately out of reach of today's 17 cards; a test
    that quietly lowered them in place would be testing a different slot.
    """
    import copy
    from dataclasses import replace

    scoring = dict(registry.slot.scoring)
    scoring["n_floor_cards"] = floor_cards
    scoring["n_floor_per_status"] = floor_status
    slot = replace(registry.slot, scoring=scoring)
    return replace(copy.copy(registry), slot=slot)


class TestScoring:
    def _separating_cohort(self):
        rows = []
        level = 0.0
        for i, sh in enumerate([2.0, 0.5, 2.2, 0.4, 2.1, 0.6, 2.3, 0.45]):
            rows.append(_card(f"2026-01-{i + 1:02d}", sharpe=sh, alpha_level=level))
            level += 0.01 if sh > 1 else -0.01
        rows.append(_card("2026-02-01", sharpe=1.0, alpha_level=level))
        return _cohort(rows)

    def test_insufficient_is_a_result_not_a_pass(self):
        doc = score_slot(self._separating_cohort())
        assert [a["arm"] for a in doc["arms"]] == list(load_registry().slot.arms)
        for arm in doc["arms"]:
            assert arm["status"] == "insufficient"
            assert arm["brier"] is None
            assert "insufficient" in arm["reason"]
        assert doc["promotion"]["eligible"] == []
        assert "never promoted against an unscored incumbent" in doc["promotion"]["reason"]

    def test_champion_is_always_an_arm_on_the_leaderboard(self):
        doc = score_slot(self._separating_cohort())
        champion = [a for a in doc["arms"] if a["role"] == "champion"]
        assert len(champion) == 1
        assert champion[0]["arm"] == load_registry().slot.champion

    def test_both_arms_score_and_the_brier_is_bounded(self):
        reg = _relaxed(load_registry(), floor_cards=3, floor_status=2)
        doc = score_slot(self._separating_cohort(), reg)
        scored = [a for a in doc["arms"] if a["status"] == "scored"]
        assert scored, doc["arms"]
        for arm in scored:
            assert 0.0 <= arm["brier"] <= 1.0
            assert arm["n_observations"] > 0

    def test_challenger_proposals_are_shadow_output_on_the_artifact(self):
        doc = score_slot(self._separating_cohort())
        keys = {(b["module"], b["metric"]) for b in doc["challenger_proposed_bands"]}
        assert METRIC in keys
        assert doc["schema_version"] == "threshold_leaderboard.v1"
        assert doc["cohort"]["horizon_cycles"] == 1  # the cohort under test

    def test_leaderboard_key_is_dated_under_evaluator(self):
        assert leaderboard_key("2026-08-15") == "evaluator/2026-08-15/threshold_leaderboard.json"


class TestArmRecords:
    def test_every_arm_gets_a_record_every_cycle(self):
        doc = score_slot(_cohort([_card("2026-01-03", sharpe=1.0, alpha_level=0.0),
                                  _card("2026-01-10", sharpe=1.0, alpha_level=0.01)]))
        comps = build_arm_components(doc, module="substrate", source_path="s3://b/k")
        names = {c.name for c in comps}
        assert names == {arm_metric_name(a) for a in load_registry().slot.arms}

    def test_insufficient_renders_as_na_never_green(self):
        doc = score_slot(_cohort([_card("2026-01-03", sharpe=1.0, alpha_level=0.0),
                                  _card("2026-01-10", sharpe=1.0, alpha_level=0.01)]))
        for comp in build_arm_components(doc, module="substrate", source_path="s3://b/k"):
            assert comp.status.startswith("N/A")
            assert comp.value is None
            assert "floor" in comp.status_reason
            assert comp.criticality == "diagnostic"

    def test_absent_leaderboard_is_a_recorded_miss(self):
        comps = build_arm_components(None, module="substrate", source_path="s3://b/k",
                                     error="scoring blew up: KeyError: x")
        assert comps and all(c.status == "N/A-NOT-RUN" for c in comps)
        assert all("blew up" in c.status_reason for c in comps)

    def test_arm_records_carry_no_bar(self):
        """§8 — machine health never borrows the experiment vocabulary."""
        comps = build_arm_components(None, module="substrate", source_path="s3://b/k")
        assert all(c.target is None and c.red_line is None for c in comps)


class TestPromotion:
    LEADERBOARD = {
        "champion": "declared_v2",
        "arms": [
            {"arm": "declared_v2", "role": "champion", "status": "scored", "brier": 0.24,
             "n_cards_paired": 30, "n_observations": 400, "reason": "r"},
            {"arm": "history_bands_v1", "role": "challenger", "status": "scored",
             "brier": 0.19, "n_cards_paired": 30, "n_observations": 400, "reason": "r"},
        ],
    }

    def test_evidence_accepted_when_the_lead_clears_the_margin(self):
        line = check_evidence(self.LEADERBOARD, "history_bands_v1", 0.02)
        assert "0.1900" in line and "lead 0.0500" in line

    def test_a_lead_under_the_margin_is_refused(self):
        with pytest.raises(PromotionRefused, match="hysteresis margin"):
            check_evidence(self.LEADERBOARD, "history_bands_v1", 0.10)

    def test_insufficient_challenger_is_refused(self):
        doc = json.loads(json.dumps(self.LEADERBOARD))
        doc["arms"][1].update(status="insufficient", brier=None,
                              reason="insufficient: 17 paired card(s) vs floor 26")
        with pytest.raises(PromotionRefused, match="not a near-pass"):
            check_evidence(doc, "history_bands_v1", 0.02)

    def test_unscored_incumbent_blocks_promotion(self):
        doc = json.loads(json.dumps(self.LEADERBOARD))
        doc["arms"][0].update(status="insufficient", brier=None)
        with pytest.raises(PromotionRefused, match="never promoted against an unscored"):
            check_evidence(doc, "history_bands_v1", 0.02)

    def test_swap_rewrites_exactly_the_champion_line(self):
        text = "slot:\n  id: s\n  champion: declared_v2\n  arms:\n    - declared_v2\n"
        out = swap_champion(text, "history_bands_v1")
        assert "  champion: history_bands_v1\n" in out
        assert "    - declared_v2\n" in out

    def test_swap_refuses_an_ambiguous_file(self):
        text = "  champion: a\n  champion: b\n"
        with pytest.raises(PromotionRefused, match="refusing to guess"):
            swap_champion(text, "c")

    def test_forced_promotion_records_its_rationale(self):
        entry = experiments_entry(from_arm="a", to_arm="b", evidence="E",
                                  leaderboard_ref="s3://x", rationale="operator call")
        assert "Forced against the gate" in entry and "operator call" in entry
        assert "--to a" in entry  # the reversal command names the arm being replaced
