"""Coverage + weight-provenance on the report card (``alpha-engine-config-I7202``).

The defect: ``_weighted_avg`` removes a null component from the denominator and
rescales the remaining weights to sum to 1. A missing component does not lower
the grade — it vanishes. On ``evaluator/2026-08-07/report_card.json`` (overall
55.68, C+, ``status: "ok"``) FOUR of the declared weights contributed nothing —
``research.cio`` (0.20), ``research.sector_teams_avg`` (0.25),
``executor.position_sizing`` (0.10) and ``executor.excursion`` (0.15, whose key
was not even emitted) — and no field anywhere said so.

The acceptance test is ``TestRecomputeFromCardAlone``: a reader must be able to
reproduce the published overall grade **parsing only the emitted JSON**. That is
the difference between a number and a verifiable number, and
``_recompute_overall`` below is deliberately written as a CONSUMER would write
it — no import from ``grading.scorecard``, no shared constant — so that a
producer-side weight change that is not stamped onto the artifact fails here.

RULING 2026-08-18 (``alpha-engine-config-I7210``, both decisions = option (a)):

* decision 1 — ``research.cio`` (0.20) and ``research.sector_teams_avg`` (0.25)
  are REMOVED from the weight table and DECLARED retired on the artifact. They
  graded the six-team + CIO graph retired 2026-07-12, so 45% of research's
  declared weight was renormalized away every cycle.
* decision 2 — a ``failed`` / ``failed_timeout`` component SCORES 0.0 and STAYS
  in the denominator at its full declared weight. It used to drop out and let
  the survivors renormalize, which made the grade go UP when the system broke.

Both move the published grade. That break with earlier cards is deliberate and
dated: ``grading_weights.version`` advances to ``2026-08-18`` and the stamped
rule names the regime, so a reader can tell which one a card was computed under.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from grading.scorecard import (
    EXECUTOR_WEIGHTS,
    OVERALL_WEIGHTS,
    PREDICTOR_WEIGHTS,
    RESEARCH_WEIGHTS,
    SKIP_CLASSES,
    compute_scorecard,
)

#: A VERBATIM published card from the PRE-ruling regime
#: (``s3://alpha-engine-research/evaluator/2026-08-07/report_card.json``,
#: grading_weights.version 2026-08-13). Never regenerate it — it is the only
#: check that the stamped weights reproduce a number the system really shipped.
FIXTURE = Path(__file__).parent / "fixtures" / "report_card_2026-08-07.json"

#: The POST-ruling artifact shape (config-I7210, 2026-08-18): research declares
#: no cio/sector_teams_avg weight, the retirement is declared on the card, and
#: the stamped rule describes the failure-scores-zero regime.
FIXTURE_POST = Path(__file__).parent / "fixtures" / "report_card_post_i7210.json"

#: The CURRENT artifact shape (alpha-engine-config-I9005, 2026-08-31): the
#: composite declares FOUR voters, the product-outcome tile among them, and the
#: card carries the outcome grade that voted.
#:
#: A NEW file rather than a regenerated ``FIXTURE_POST``. ``FIXTURE_POST`` is
#: the frozen evidence for the I7210 ruling's continuity claim — regenerating
#: it would delete the only record of the regime that claim is about, and the
#: test below would then be comparing the producer to itself.
FIXTURE_I9005 = Path(__file__).parent / "fixtures" / "report_card_post_i9005.json"

# ---------------------------------------------------------------------------
# The reader-side recomputation. Imports nothing from the producer on purpose.
# ---------------------------------------------------------------------------

_SECTIONS = ("research", "predictor", "executor")


_FAILED_CLASSES = {"failed", "failed_timeout"}


def _recompute_level(
    weights: dict[str, float],
    grades: dict[str, float | None],
    skip_classes: dict[str, str],
) -> float | None:
    """One level of the rule the card states in ``grading_weights.rule``.

    A component whose ``skip_class`` is a FAILURE scores 0.0 and stays in the
    denominator; any other null is dropped and the survivors renormalize
    (alpha-engine-config-I7210, ruled 2026-08-18).
    """
    num = den = 0.0
    for name, w in weights.items():
        grade = grades.get(name)
        if grade is None:
            if skip_classes.get(name) not in _FAILED_CLASSES:
                continue
            grade = 0.0  # failure scores zero AT FULL DECLARED WEIGHT
        num += w * grade
        den += w
    return (num / den) if den else None


def _recompute_section(card: dict, section: str) -> float | None:
    components = card[section]["components"]
    weights = card["grading_weights"][section]
    grades = {
        name: (components.get(name) or {}).get("grade")
        if isinstance(components.get(name), dict) else None
        for name in weights
    }
    skip_classes = (card[section].get("coverage") or {}).get("skip_classes") or {}
    return _recompute_level(weights, grades, skip_classes)


def _recompute_overall(card: dict) -> float | None:
    """Reproduce ``card['overall']['grade']`` from the card's own JSON.

    Four voters since alpha-engine-config-I9005: the three v1 sections, each
    recomputed from its own component block, plus the product-outcome grade the
    card carries at ``card['portfolio_outcome']['grade']``. A pre-I9005 card
    has no such key and no ``portfolio_outcome`` entry in its stamped weight
    table, so it recomputes exactly as it always did — which is what keeps the
    verbatim 2026-08-07 fixture a live check rather than a rewritten one.
    """
    grades = {s: _recompute_section(card, s) for s in _SECTIONS}
    outcome = card.get("portfolio_outcome")
    grades["portfolio_outcome"] = (
        outcome.get("grade") if isinstance(outcome, dict) else None
    )
    skip_classes = (card["overall"].get("coverage") or {}).get("skip_classes") or {}
    return _recompute_level(
        card["grading_weights"]["overall"], grades, skip_classes,
    )


def _roundtrip(card: dict) -> dict:
    """Force the assertion through real JSON — the artifact, not the dict."""
    return json.loads(json.dumps(card))


# ---------------------------------------------------------------------------
# Input builders
# ---------------------------------------------------------------------------


def _full_inputs() -> dict:
    """Inputs that populate every declared component — the never-yet-observed
    complete card. Coverage must read 1.0 and the qualifier COMPLETE."""
    return {
        "signal_quality": {
            "status": "ok",
            "overall": {"accuracy_21d": 0.58, "avg_alpha_21d": 1.2},
            "by_score_bucket": [{"bucket": "90+", "accuracy_21d": 0.66}],
        },
        "e2e_lift": {
            "status": "ok",
            "scanner_lift": {
                "n_passing": 600, "n_universe": 15000, "lift": 0.8,
                "classification": {"precision": 0.44, "recall": 0.2, "f1": 0.27},
            },
            "team_lift": [
                {
                    "team_id": f"t{i}", "n_picks": 20, "lift": 0.5,
                    "lift_vs_quant": 0.3,
                    "classification": {"precision": 0.5, "recall": 0.3, "f1": 0.37},
                }
                for i in range(6)
            ],
            "cio_lift": {
                "n_advance": 100, "n_reject": 90, "lift": 0.4,
                "advance_avg": 0.01, "reject_avg": -0.01,
                "classification": {"precision": 0.5, "recall": 0.45, "f1": 0.47},
            },
            "cio_vs_ranking": {"lift": 0.3},
        },
        "macro_eval": {"status": "ok", "accuracy_lift": 1.0, "alpha_lift": 0.4},
        "score_calibration": {"monotonic": True},
        "veto_result": {
            "status": "ok", "recommended_threshold": 0.65,
            "thresholds": [
                {"confidence": 0.65, "precision": 0.57, "recall": 0.21,
                 "f1": 0.31, "lift": 4.0},
            ],
        },
        "veto_value": {"net_value": 400.0},
        "trigger_scorecard": {
            "status": "ok",
            "summary": {
                "avg_slippage_vs_signal": -0.2, "win_rate_vs_spy": 0.5,
                "avg_realized_alpha": 0.8, "total_entries": 300,
            },
            "triggers": [
                {"trigger": "gap", "avg_slippage_vs_signal": -0.1,
                 "win_rate_vs_spy": 0.52, "n_trades": 40},
            ],
        },
        "shadow_book": {
            "status": "ok", "assessment": "appropriate", "guard_lift": 0.6,
            "n_blocked": 1233,
            "classification": {"precision": 0.63, "recall": 0.9, "f1": 0.74},
        },
        "exit_timing": {
            "status": "ok", "diagnosis": "exits_could_improve", "n_roundtrips": 120,
            "summary": {"avg_capture_ratio": 0.6, "avg_realized_return": 0.5},
        },
        "sizing_ab": {
            "status": "ok", "sharpe_diff": 0.12, "alpha_diff": 0.5,
            "assessment": "sizing_helps",
        },
        "predictor_sizing": {
            "status": "ok", "overall_rank_ic": 0.0278, "sizing_lift": 0.01,
            "recent_positive_weeks": 7, "recent_total_weeks": 8,
            "weekly_ic": [0.01] * 8,
        },
        "portfolio_stats": {
            "sharpe_ratio": 0.35, "sortino_ratio": 0.56, "calmar_ratio": 0.22,
            "cvar_95": -0.02, "information_ratio_spy": 0.4, "max_drawdown": -0.018,
        },
        "scanner_opt": {"leakage_pct": 0.1},
        "cio_opt": {},
        "calibration_diagnostics": {"status": "ok", "ece": 0.03, "n": 400},
        "action_entropy": {
            "status": "ok", "entropy_normalized": 0.796, "most_common": "HOLD",
            "most_common_fraction": 0.637, "alarm": False, "n": 900,
        },
        "excursion_summary": {
            "status": "ok", "mean_mfe_mae_ratio": 1.6, "pct_high_quality": 0.45,
            "median_mfe_mae_ratio": 1.4, "pct_mfe_gt_mae": 0.6, "n": 120,
        },
        # The product-outcome voter (alpha-engine-config-I9005). Handed in
        # already graded by grading/aggregate.py::_outcome_voter from Tile 0;
        # `compute_scorecard` never reads eod_pnl.csv itself. A "complete" card
        # is one where all FOUR declared voters graded, so it belongs here.
        "portfolio_outcome": {
            "grade": 39.0, "letter": "F", "tile_status": "RED",
            "n_components": 18, "n_graded": 18, "effective_coverage": 1.0,
            "source": "tiles.portfolio_outcome.numeric_grade",
        },
    }


def _today_inputs() -> dict:
    """The shape the pipeline actually produces: research graph retired,
    sizing A/B and portfolio excursion never persisted."""
    inputs = _full_inputs()
    inputs["e2e_lift"] = dict(inputs["e2e_lift"])
    inputs["e2e_lift"]["team_lift"] = []
    inputs["e2e_lift"]["cio_lift"] = {
        "status": "retired", "retired_date": "2026-07-12",
        "note": "six-team+CIO graph retired (config#1580 / config-I2993)",
    }
    inputs["e2e_lift"]["research_graph_retired"] = {
        "retired_date": "2026-07-12",
        "reason": "six-team + macro-economist + CIO research orchestration retired (config#1580)",
    }
    del inputs["sizing_ab"]
    del inputs["excursion_summary"]
    return inputs


# ---------------------------------------------------------------------------


class TestWeightTables:
    """The declared weights are the contract; a table that does not sum to 1
    makes ``weight_present`` uninterpretable."""

    @pytest.mark.parametrize(
        "table",
        [RESEARCH_WEIGHTS, PREDICTOR_WEIGHTS, EXECUTOR_WEIGHTS, OVERALL_WEIGHTS],
    )
    def test_each_table_sums_to_one(self, table):
        assert math.isclose(sum(table.values()), 1.0, abs_tol=1e-9)

    def test_declared_component_names_match_what_the_card_emits(self):
        card = compute_scorecard(**_full_inputs())
        for section, table in (
            ("research", RESEARCH_WEIGHTS),
            ("predictor", PREDICTOR_WEIGHTS),
            ("executor", EXECUTOR_WEIGHTS),
        ):
            emitted = set(card[section]["components"]) - {"sector_teams"}
            missing = set(table) - emitted
            assert not missing, (
                f"{section}: weights declared for components the card never "
                f"emits: {sorted(missing)} — a weight that cannot be traced to a "
                f"component block is not recomputable by a reader"
            )


class TestRecomputeFromCardAlone:
    """DELIVERABLE 2 — the acceptance test.

    Parses only the emitted JSON and reproduces ``overall.grade``.
    """

    @pytest.mark.parametrize(
        "inputs, label",
        [(_full_inputs(), "complete"), (_today_inputs(), "as-produced-today")],
    )
    def test_overall_is_reproducible(self, inputs, label):
        card = _roundtrip(compute_scorecard(**inputs))
        assert card["overall"]["grade"] == pytest.approx(
            _recompute_overall(card), abs=1e-9,
        ), f"{label}: overall grade not reproducible from the card alone"

    def test_section_grades_are_reproducible(self):
        card = _roundtrip(compute_scorecard(**_today_inputs()))
        for section in _SECTIONS:
            assert card[section]["grade"] == pytest.approx(
                _recompute_section(card, section), abs=1e-9,
            ), section

    def test_a_card_carrying_a_FAILURE_is_reproducible(self):
        """The regime the 2026-08-18 ruling introduced, from the card alone.

        A reader must be able to reproduce a grade that a failure dragged down
        — which needs ``coverage.skip_classes`` on the artifact, not just the
        weights. Without it the stated rule is unexecutable.
        """
        inputs = _today_inputs()
        inputs["exit_timing"] = {"status": "timeout", "reason": "stage timed out at 300s"}
        card = _roundtrip(compute_scorecard(**inputs))
        assert card["executor"]["coverage"]["skip_classes"]["exit_rules"] == "failed_timeout"
        for section in _SECTIONS:
            assert card[section]["grade"] == pytest.approx(
                _recompute_section(card, section), abs=1e-9,
            ), section
        assert card["overall"]["grade"] == pytest.approx(
            _recompute_overall(card), abs=1e-9,
        )

    def test_the_stamped_weights_reproduce_a_REAL_published_card(self):
        """The strongest available check: the weights this producer now stamps
        must reproduce the number the system actually published on 2026-08-07.

        If they do not, the table on the artifact is not the table that graded
        the system, and every "recompute it yourself" claim is false. The
        fixture's grade blocks are verbatim from
        ``s3://alpha-engine-research/evaluator/2026-08-07/report_card.json``.
        """
        card = json.loads(FIXTURE.read_text())
        assert card["overall"]["grade"] == pytest.approx(55.68560475209855, abs=1e-9)
        assert _recompute_overall(card) == pytest.approx(
            card["overall"]["grade"], abs=1e-9,
        )

    def test_the_renormalization_rule_is_stated_on_the_artifact(self):
        # A reader needs the RULE as much as the numbers; without it the
        # weights alone reproduce the wrong answer whenever anything is null.
        card = _roundtrip(compute_scorecard(**_today_inputs()))
        rule = card["grading_weights"]["rule"]
        assert "removed from the denominator" in rule
        assert "rescaled" in rule

    def test_the_stamped_rule_does_not_call_the_failure_case_renormalization(self):
        """The stamped words must describe what the arithmetic DOES.

        Under the 2026-08-18 ruling a failed component is scored 0 at full
        weight — that is the opposite of renormalizing it away, and a rule text
        that still said "a null is removed from the denominator" full stop
        would send a reader to a different number than the one published.
        """
        card = _roundtrip(compute_scorecard(**_today_inputs()))
        rule = card["grading_weights"]["rule"]
        assert "failed_timeout" in rule
        assert "STAYS in the denominator" in rule
        assert "NOT renormalized away" in rule
        # And the regime is dated, so two cards can be told apart.
        assert card["grading_weights"]["version"] == "2026-08-31"

    def test_the_weight_table_version_advances_with_the_published_grade(self):
        """A reader comparing two cards must see the TABLE changed, not the
        system. Removing 45% of research's declared weight moves the published
        number; a version left at 2026-08-13 would hide that behind a diff."""
        card = _roundtrip(compute_scorecard(**_today_inputs()))
        assert card["grading_weights"]["version"] > "2026-08-13"


class TestCoverageIsPublishedAtEveryLevel:
    """DELIVERABLE 1 — a reader must see WHICH level lost coverage."""

    @pytest.fixture
    def card(self):
        return _roundtrip(compute_scorecard(**_today_inputs()))

    @pytest.mark.parametrize("level", ["overall", "research", "predictor", "executor"])
    def test_every_level_carries_a_coverage_block(self, card, level):
        cov = card[level]["coverage"]
        for key in (
            "weight_present", "weight_present_effective", "components_skipped",
            "skips", "weights", "qualifier", "weight_failed", "components_failed",
        ):
            assert key in cov, f"{level}.coverage missing {key}"

    def test_every_skip_names_a_reason_and_a_class(self, card):
        for level in ("overall", "research", "predictor", "executor"):
            for skip in card[level]["coverage"]["skips"]:
                assert skip["reason"], f"{level}/{skip['component']}: empty reason"
                assert skip["skip_class"] in SKIP_CLASSES
                assert skip["weight"] > 0

    def test_research_no_longer_declares_the_retired_pair(self, card):
        """Decision 1, ruled 2026-08-18 (config-I7210).

        ``cio`` and ``sector_teams_avg`` graded a graph retired 2026-07-12.
        They carried 45% of the declared research weight and were renormalized
        away every cycle, so research's declared table was fiction. They are
        gone from the table — research's declared weight now equals its actual
        weight and ``weight_present`` reads 1.0.
        """
        assert "cio" not in RESEARCH_WEIGHTS
        assert "sector_teams_avg" not in RESEARCH_WEIGHTS
        cov = card["research"]["coverage"]
        assert cov["components_skipped"] == []
        assert cov["weight_present"] == pytest.approx(1.0)
        assert cov["qualifier"] == "COMPLETE"

    def test_the_surviving_research_weights_are_the_old_ones_rescaled(self):
        """The ruling removes dead weight; it does not re-opine on the live
        components. Every surviving pair keeps its old RATIO."""
        old = {"scanner": 0.10, "macro_agent": 0.10,
               "composite_scoring": 0.20, "calibration_diagnostics": 0.15}
        assert set(RESEARCH_WEIGHTS) == set(old)
        for name, w_old in old.items():
            assert RESEARCH_WEIGHTS[name] == pytest.approx(w_old / 0.55, abs=1e-12)
        assert math.isclose(sum(RESEARCH_WEIGHTS.values()), 1.0, abs_tol=1e-9)

    def test_the_retirement_is_DECLARED_on_the_artifact_not_silently_deleted(self, card):
        """``observability-policy.md`` §8.3: RETIRED is not ABSENT.

        A reader of a post-ruling card must be able to see WHY research's
        declared weights changed without diffing two commits of the producer.
        """
        retired = card["grading_weights"]["retired_components"]["research"]
        by_name = {d["component"]: d for d in retired}
        # alpha-engine-config-I8184 added momentum_regime_ic — diagnostic-only,
        # never weight-tabled (weight_was is None), stamped for the same
        # declared-not-silently-deleted reason as cio/sector_teams_avg.
        assert set(by_name) == {"cio", "sector_teams_avg", "momentum_regime_ic"}
        for name, weight_was in (("cio", 0.20), ("sector_teams_avg", 0.25)):
            d = by_name[name]
            assert d["lifecycle"] == "RETIRED"
            assert d["weight_was"] == pytest.approx(weight_was)
            assert d["retired_date"] == "2026-07-12"
            assert d["removed_from_weight_table"] == "2026-08-18"
            assert "config-I2993" in d["reference"]
            assert "I7210" in d["ruling"]
            assert d["superseded_by"]
        mri_d = by_name["momentum_regime_ic"]
        assert mri_d["lifecycle"] == "RETIRED"
        assert mri_d["weight_was"] is None
        assert mri_d["retired_date"] == "2026-07-17"
        assert "I7827" in mri_d["reference"] or "I8184" in mri_d["ruling"]
        # The 45% that used to be renormalized away is stated as a number —
        # summed only over weight-tabled retirees (weight_was is not None).
        assert sum(
            d["weight_was"] for d in retired if d["weight_was"] is not None
        ) == pytest.approx(0.45)

    def test_the_retired_components_are_still_emitted_and_marked_unweighted(self, card):
        """Removing a weight is not deleting the evidence. The blocks are still
        computed and published; each says on its own face that it does not
        vote, so a reader is never left inferring it from the weight table."""
        for name in ("cio", "sector_teams_avg"):
            block = card["research"]["components"][name]
            assert block["lifecycle"] == "RETIRED"
            assert block["weighted"] is False
            assert block["retired_date"] == "2026-07-12"

    def test_executor_reports_the_two_never_persisted_producers(self, card):
        cov = card["executor"]["coverage"]
        assert set(cov["components_skipped"]) == {"position_sizing", "excursion"}
        assert cov["weight_present"] == pytest.approx(0.75)

    def test_overall_coverage_is_full_at_its_own_level_and_the_card_says_otherwise(self, card):
        """The regression this whole file exists for.

        All three module grades are non-null, so the overall level is 100%
        covered *at its own level* — which is exactly the reading that made
        55.68 look complete. The effective number is what stops that.
        """
        cov = card["overall"]["coverage"]
        assert cov["weight_present"] == pytest.approx(1.0)
        # Research is now fully covered (the retired pair no longer declares
        # weight); the executor's two never-persisted producers still are not.
        # Four declared voters since alpha-engine-config-I9005: the outcome
        # tile at 0.50 with its own leaf coverage, and the three process
        # modules over the remaining 0.50 in the ruled I7210 ratios.
        w = OVERALL_WEIGHTS
        assert cov["weight_present_effective"] == pytest.approx(
            w["portfolio_outcome"] * 1.0
            + w["research"] * 1.0
            + w["predictor"] * 1.0
            + w["executor"] * 0.75,
        )
        assert cov["weight_present_effective"] < 1.0

    def test_a_complete_card_reports_full_coverage(self):
        card = _roundtrip(compute_scorecard(**_full_inputs()))
        for level in ("overall", "research", "predictor", "executor"):
            cov = card[level]["coverage"]
            assert cov["qualifier"] == "COMPLETE", level
            assert cov["weight_present"] == pytest.approx(1.0), level
            assert cov["weight_present_effective"] == pytest.approx(1.0), level
            assert cov["components_skipped"] == [], level
            assert cov["renormalized"] is False, level


class TestPartialGradeNeverRendersAsAPlainLetter:
    """DELIVERABLE 3, satisfied WITHOUT inventing a threshold.

    The qualifier keys on a measured fact (declared weight did not all vote),
    not on a floor. The floor itself stays unset — see
    ``DEFAULT_COVERAGE_FLOOR``'s rationale and the report on I7202.
    """

    def test_partial_card_is_qualified(self):
        card = _roundtrip(compute_scorecard(**_today_inputs()))
        assert card["overall"]["coverage"]["qualifier"] == "PARTIAL"
        assert card["overall"]["display"] != card["overall"]["letter"]
        assert "PARTIAL" in card["overall"]["display"]

    def test_complete_card_renders_the_bare_letter(self):
        card = _roundtrip(compute_scorecard(**_full_inputs()))
        assert card["overall"]["display"] == card["overall"]["letter"]

    def test_no_floor_is_asserted(self):
        card = _roundtrip(compute_scorecard(**_today_inputs()))
        cov = card["overall"]["coverage"]
        assert cov["floor"] is None
        assert cov["provisional"] is False
        assert "unmeasured" in cov["floor_status"]


class TestFailureIsNeverJustCoverage:
    """Brian ruling 2026-08-13: anything timing out is FAILED.
    Brian ruling 2026-08-18 (config-I7210 decision 2): a failed component
    SCORES 0.0 and STAYS in the denominator at its full declared weight.

    Dropping a failure from the denominator meant the grade went UP when the
    system broke — the property that disqualified the card as a track record.
    """

    def _timed_out(self, base=None):
        inputs = base() if base else _today_inputs()
        inputs["exit_timing"] = {"status": "timeout", "reason": "stage timed out at 300s"}
        return _roundtrip(compute_scorecard(**inputs))

    def test_timeout_classifies_as_failed_not_insufficient_data(self):
        card = self._timed_out()
        cov = card["executor"]["coverage"]
        assert cov["skip_classes"]["exit_rules"] == "failed_timeout"
        assert "exit_rules" in cov["components_failed"]

    def test_failed_weight_is_reported_separately_from_coverage(self):
        card = self._timed_out()
        cov = card["executor"]["coverage"]
        assert cov["weight_failed"] == pytest.approx(0.15)
        assert cov["weight_scored_zero"] == pytest.approx(0.15)
        # And it is loud in the rendering, distinctly from ordinary partial.
        assert cov["qualifier"] == "PARTIAL-FAILURE-SCORED-ZERO"
        assert "SCORED ZERO ON FAILURE" in card["executor"]["display"]

    # -- decision 2: the arithmetic, asserted numerically -------------------

    def test_a_failure_drags_the_grade_down_by_its_FULL_declared_weight(self):
        """THE decision-2 assertion, on an otherwise COMPLETE card.

        Every other executor component is present, so the denominator is the
        full 1.0 both times and the drop is exactly w_exit_rules * g_exit_rules
        — the failed component voting 0 at its declared 0.15, with nothing
        rescaled onto the survivors.
        """
        healthy = _roundtrip(compute_scorecard(**_full_inputs()))
        broken = self._timed_out(base=_full_inputs)

        g_exit = healthy["executor"]["components"]["exit_rules"]["grade"]
        assert g_exit is not None and g_exit > 0
        w_exit = EXECUTOR_WEIGHTS["exit_rules"]

        assert broken["executor"]["coverage"]["weight_in_denominator"] == pytest.approx(1.0)
        assert broken["executor"]["grade"] == pytest.approx(
            healthy["executor"]["grade"] - w_exit * g_exit, abs=1e-9,
        )
        # ... and it reaches the headline at the overall weight, undiluted.
        assert broken["overall"]["grade"] == pytest.approx(
            healthy["overall"]["grade"] - OVERALL_WEIGHTS["executor"] * w_exit * g_exit,
            abs=1e-9,
        )
        # The direction is the whole point: breaking never improves the grade.
        assert broken["overall"]["grade"] < healthy["overall"]["grade"]

    def test_a_failure_is_NOT_renormalized_onto_the_survivors(self):
        """The old behaviour, asserted as the thing that must NOT happen.

        Under renormalization the executor grade would have been the survivors'
        weighted average over 0.85 — a number STRICTLY HIGHER than the healthy
        grade whenever exit_rules graded below the rest.
        """
        healthy = _roundtrip(compute_scorecard(**_full_inputs()))
        broken = self._timed_out(base=_full_inputs)

        comps = healthy["executor"]["components"]
        survivors = {k: v for k, v in EXECUTOR_WEIGHTS.items() if k != "exit_rules"}
        renormalized = (
            sum(w * comps[k]["grade"] for k, w in survivors.items())
            / sum(survivors.values())
        )
        assert broken["executor"]["grade"] != pytest.approx(renormalized, abs=1e-6)
        assert broken["executor"]["coverage"]["renormalized"] is False

    def test_a_NON_failure_absence_still_renormalizes(self):
        """Decision 2 must not turn every absence into a zero.

        ``_today_inputs`` drops sizing_ab and excursion_summary — legitimate
        never-persisted producers, not failures. Those 0.25 of declared weight
        are still removed from the denominator and the survivors still rescale.
        """
        card = _roundtrip(compute_scorecard(**_today_inputs()))
        cov = card["executor"]["coverage"]
        assert set(cov["components_skipped"]) == {"position_sizing", "excursion"}
        assert {cov["skip_classes"][c] for c in cov["components_skipped"]} <= {
            "input_absent", "insufficient_data", "not_implemented",
        }
        assert cov["weight_scored_zero"] == 0.0
        assert cov["renormalized"] is True
        assert cov["weight_in_denominator"] == pytest.approx(0.75)
        assert cov["renormalization_factor"] == pytest.approx(1 / 0.75, abs=1e-6)

        comps = card["executor"]["components"]
        survivors = {k: w for k, w in EXECUTOR_WEIGHTS.items()
                     if k not in cov["components_skipped"]}
        assert card["executor"]["grade"] == pytest.approx(
            sum(w * comps[k]["grade"] for k, w in survivors.items())
            / sum(survivors.values()),
            abs=1e-9,
        )

    def test_a_retired_component_is_not_scored_zero(self):
        """A retirement is an absence, not a failure — it must never be priced
        as one. The retired pair no longer declares weight at all, and nothing
        that remains classifies as failed."""
        card = _roundtrip(compute_scorecard(**_today_inputs()))
        cov = card["research"]["coverage"]
        assert cov["weight_scored_zero"] == 0.0
        assert cov["components_failed"] == []

    def test_coverage_and_the_arithmetic_agree_on_the_SAME_failures(self):
        """The drift test.

        The composite arithmetic and the coverage block are two views of one
        classification pass (``_resolve_components``). If they ever became two
        passes again, this fails: the components the coverage block names as
        failed are exactly the ones the grade priced at 0, checked by
        recomputing the grade from the coverage block's own verdict.
        """
        import grading.scorecard as sc

        inputs = _today_inputs()
        inputs["exit_timing"] = {"status": "timeout", "reason": "stage timed out at 300s"}
        inputs["macro_eval"] = {"status": "error", "reason": "macro grader crashed"}
        card = _roundtrip(compute_scorecard(**inputs))

        for section, table in (
            ("research", sc.RESEARCH_WEIGHTS),
            ("predictor", sc.PREDICTOR_WEIGHTS),
            ("executor", sc.EXECUTOR_WEIGHTS),
        ):
            cov = card[section]["coverage"]
            named = set(cov["components_failed"])
            from_classes = {
                c for c, k in cov["skip_classes"].items() if k in _FAILED_CLASSES
            }
            assert named == from_classes, section
            # weight_failed is the sum of exactly those declared weights ...
            assert cov["weight_failed"] == pytest.approx(
                sum(table[c] for c in named), abs=1e-6,
            ), section
            # ... and the published grade is the one that rule produces.
            assert card[section]["grade"] == pytest.approx(
                _recompute_section(card, section), abs=1e-9,
            ), section

        assert "macro_agent" in card["research"]["coverage"]["components_failed"]
        assert "exit_rules" in card["executor"]["coverage"]["components_failed"]

    def test_zero_is_emitted_as_a_value_when_nothing_failed(self):
        # `no data` must never be indistinguishable from `no failures`.
        card = _roundtrip(compute_scorecard(**_today_inputs()))
        for level in ("overall", "research", "predictor", "executor"):
            cov = card[level]["coverage"]
            assert cov["weight_failed"] == 0.0
            assert cov["weight_scored_zero"] == 0.0
            assert cov["components_failed"] == []
            assert cov["qualifier"] != "PARTIAL-FAILURE-SCORED-ZERO"

    def test_a_masked_failure_reaches_the_HEADLINE_number(self):
        """The one that matters for showing the grade externally.

        A component that timed out inside the executor must not become a clean
        C+ at the top of the card. The overall level reads 100% covered (all
        three module grades non-null); the propagated fields are what stop that
        reading, and now the number itself carries the loss too.
        """
        card = self._timed_out()
        cov = card["overall"]["coverage"]
        assert cov["qualifier"] == "PARTIAL-FAILURE-SCORED-ZERO"
        assert "exit_rules" in cov["components_failed"]
        assert cov["weight_failed"] == pytest.approx(
            OVERALL_WEIGHTS["executor"] * EXECUTOR_WEIGHTS["exit_rules"],
        )
        assert "SCORED ZERO ON FAILURE" in card["overall"]["display"]

    def test_a_retirement_is_not_a_failure(self):
        card = _roundtrip(compute_scorecard(**_today_inputs()))
        cov = card["research"]["coverage"]
        assert cov["components_failed"] == []
        assert cov["weight_failed"] == 0.0


class TestContinuityAcrossTheRuling:
    """What the 2026-08-18 ruling does and does not move on the headline.

    The ruling deliberately breaks comparability with pre-2026-08-18 cards.
    These tests state exactly WHERE the break is, measured rather than
    asserted, so the claim in the PR body is checkable and stays true.
    """

    def test_removing_the_retired_pair_does_not_move_the_LAST_published_number(self):
        """MEASURED, on the real 2026-08-07 card.

        The two removed components were already renormalized away, and the
        survivors are rescaled PROPORTIONALLY over the 0.55 that actually
        voted, so the arithmetic is identical term by term. Decision 1 changes
        what the declared table MEANS (weight_present 0.55 → 1.0, an honest
        denominator instead of a fictional one) without silently repricing the
        live components on the way through. If someone later re-weights the
        survivors on judgement rather than proportion, this test fails and the
        change has to be argued as the grade change it is.
        """
        card = json.loads(FIXTURE.read_text())
        comps = card["research"]["components"]

        old = card["grading_weights"]["research"]
        assert set(old) - set(RESEARCH_WEIGHTS) == {"cio", "sector_teams_avg"}

        def _avg(table):
            num = den = 0.0
            for name, w in table.items():
                g = (comps.get(name) or {}).get("grade")
                if g is None:
                    continue
                num += w * g
                den += w
            return num / den

        assert _avg(old) == pytest.approx(card["research"]["grade"], abs=1e-9)
        assert _avg(RESEARCH_WEIGHTS) == pytest.approx(
            card["research"]["grade"], abs=1e-9,
        )

    def test_the_break_is_in_the_DENOMINATOR_not_the_number(self):
        """Same card, same grade, honest coverage: the pre-ruling card reported
        research at 55% of its declared weight; the post-ruling one reports the
        same grade at 100%, because the 45% it never had is no longer claimed."""
        pre = json.loads(FIXTURE.read_text())
        post = json.loads(FIXTURE_POST.read_text())
        # The real 2026-08-07 card predates the coverage block, so its coverage
        # is derived here the way a reader would have had to: from the stamped
        # weights and which components carried a grade.
        pre_comps = pre["research"]["components"]
        pre_present = sum(
            w for name, w in pre["grading_weights"]["research"].items()
            if (pre_comps.get(name) or {}).get("grade") is not None
        )
        assert pre_present == pytest.approx(0.55)
        assert post["research"]["coverage"]["weight_present"] == pytest.approx(1.0)
        assert pre["grading_weights"]["version"] == "2026-08-13"
        assert post["grading_weights"]["version"] == "2026-08-18"

    def test_decision_2_does_not_move_a_card_with_no_failure(self):
        """No card in the measured window carries a ``failed`` skip, so the
        scoring change is latent until something actually breaks — which is the
        point: it costs nothing on a healthy week and is unmissable on a bad
        one."""
        card = _roundtrip(compute_scorecard(**_today_inputs()))
        for level in _SECTIONS + ("overall",):
            assert card[level]["coverage"]["weight_scored_zero"] == 0.0
        assert card["overall"]["grade"] == pytest.approx(
            _recompute_overall(card), abs=1e-9,
        )

    def test_the_current_fixture_is_the_shape_the_producer_emits(self):
        """Guards the fixture against drifting from the producer — a stale
        fixture asserting a regime nobody publishes any more is worse than
        none. Points at the CURRENT regime (I9005); ``FIXTURE_POST`` stays
        frozen as the I7210 evidence the test above reads."""
        live = _roundtrip(compute_scorecard(**_today_inputs()))
        fixture = json.loads(FIXTURE_I9005.read_text())
        assert fixture["grading_weights"]["research"] == live["grading_weights"]["research"]
        assert fixture["grading_weights"]["rule"] == live["grading_weights"]["rule"]
        assert (
            fixture["grading_weights"]["retired_components"]
            == live["grading_weights"]["retired_components"]
        )
        assert fixture["overall"]["grade"] == pytest.approx(
            live["overall"]["grade"], abs=1e-9,
        )
        # And it is recomputable from itself, like any published card.
        assert fixture["overall"]["grade"] == pytest.approx(
            _recompute_overall(fixture), abs=1e-9,
        )


class TestCoverageCannotFailTheRun:
    """The Saturday-run constraint: coverage degrades, the card still ships."""

    def test_grade_is_unchanged_when_coverage_raises(self, monkeypatch):
        import grading.scorecard as sc

        good = compute_scorecard(**_today_inputs())
        monkeypatch.setattr(
            sc, "_coverage",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        degraded = compute_scorecard(**_today_inputs())

        assert degraded["overall"]["grade"] == good["overall"]["grade"]
        for section in _SECTIONS:
            assert degraded[section]["grade"] == good[section]["grade"]

    def test_coverage_renders_unknown_rather_than_absent(self, monkeypatch):
        import grading.scorecard as sc

        monkeypatch.setattr(
            sc, "_coverage",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        card = _roundtrip(compute_scorecard(**_today_inputs()))
        for level in ("overall", "research", "predictor", "executor"):
            cov = card[level]["coverage"]
            assert cov["qualifier"] == "UNKNOWN"
            assert cov["error"] == sc.COVERAGE_UNKNOWN_MARKER
            assert "UNKNOWN" in card[level]["display"]

    def test_the_totally_empty_card_still_builds(self):
        card = _roundtrip(compute_scorecard())
        assert card["status"] == "insufficient_data"
        assert card["overall"]["grade"] is None
        assert card["overall"]["coverage"]["qualifier"] == "UNGRADED"
        assert card["overall"]["coverage"]["weight_present"] == 0.0


class TestTheProductOutcomeVotes:
    """alpha-engine-config-I9005 — the concealment this closes.

    Until 2026-08-31 the composite graded ``research`` / ``predictor`` /
    ``executor`` only. The one tile measuring what the system PRODUCES was
    computed every cycle, published on the card, and then excluded from the
    number every surface reads: the live 2026-08-31 card carried
    ``overall.display = "C+ (PARTIAL — 93% of declared weight)"`` and
    ``tiles_overall_status = "RED"`` simultaneously, with
    ``overall.coverage.census_scope.tiles_out_of_scope`` naming
    ``portfolio_outcome: "RED"`` (all four read from
    ``s3://alpha-engine-research/evaluator/latest/report_card.json``,
    2026-08-31).

    No threshold moves here. Every ``target`` / ``red_line`` / ``n_floor``
    behind the outcome tile still comes from
    ``grading/thresholds/registry.yaml``; what changes is whether the
    already-measured verdict is allowed to vote.
    """

    def test_the_outcome_tile_carries_declared_weight(self):
        assert "portfolio_outcome" in OVERALL_WEIGHTS
        assert OVERALL_WEIGHTS["portfolio_outcome"] == pytest.approx(0.50)

    def test_the_process_modules_keep_their_ruled_RATIOS(self):
        """I9005 adds a voter; it does not re-opine on the I7210 table.

        The three process weights must stay in exactly the 0.40 / 0.25 / 0.35
        proportion Brian ruled on 2026-08-18. If someone later re-weights them
        on judgement while adding a voter, this fails and the change has to be
        argued as the re-ruling it is.
        """
        process = {k: OVERALL_WEIGHTS[k] for k in _SECTIONS}
        total = sum(process.values())
        assert total == pytest.approx(1.0 - OVERALL_WEIGHTS["portfolio_outcome"])
        for name, ruled in (("research", 0.40), ("predictor", 0.25), ("executor", 0.35)):
            assert process[name] / total == pytest.approx(ruled, abs=1e-9), name

    def test_a_RED_outcome_lowers_the_headline(self):
        """The property the exclusion removed. Same process inputs, two outcome
        grades: the worse outcome must produce the worse headline."""
        good = dict(_today_inputs())
        good["portfolio_outcome"] = {"grade": 90.0, "letter": "A"}
        bad = dict(_today_inputs())
        bad["portfolio_outcome"] = {"grade": 39.0, "letter": "F"}

        g = _roundtrip(compute_scorecard(**good))
        b = _roundtrip(compute_scorecard(**bad))

        assert b["overall"]["grade"] < g["overall"]["grade"]
        assert b["overall"]["grade"] == pytest.approx(
            g["overall"]["grade"] - OVERALL_WEIGHTS["portfolio_outcome"] * (90.0 - 39.0),
            abs=1e-9,
        )
        # ... and it is recomputable from the card alone, both times.
        for card in (g, b):
            assert card["overall"]["grade"] == pytest.approx(
                _recompute_overall(card), abs=1e-9,
            )

    def test_the_voting_grade_is_on_the_card(self):
        """A published number whose inputs live only in source is not
        verifiable — the same reason ``grading_weights`` is stamped."""
        card = _roundtrip(compute_scorecard(**_today_inputs()))
        block = card["portfolio_outcome"]
        assert block["grade"] == pytest.approx(39.0)
        assert block["source"] == "tiles.portfolio_outcome.numeric_grade"
        assert card["grading_weights"]["portfolio_outcome_weight"] == pytest.approx(0.50)
        assert card["grading_weights"]["process_half"] == {
            "research": 0.40, "predictor": 0.25, "executor": 0.35,
        }

    def test_an_ABSENT_outcome_reproduces_the_pre_I9005_composite_EXACTLY(self):
        """Why the proportional split is the safe one.

        When the outcome does not grade, the I7210 non-failure-absence rule
        renormalizes its weight away and the three survivors rescale to exactly
        0.40 / 0.25 / 0.35 — the pre-I9005 table. So a cycle with no
        ``eod_pnl.csv`` publishes the number it always did, and the only cards
        whose headline moves are the ones where the outcome IS measured.
        """
        inputs = dict(_today_inputs())
        del inputs["portfolio_outcome"]
        card = _roundtrip(compute_scorecard(**inputs))

        comps = {s: _recompute_section(card, s) for s in _SECTIONS}
        pre_i9005 = sum(
            w * comps[name]
            for name, w in (("research", 0.40), ("predictor", 0.25), ("executor", 0.35))
        )
        assert card["overall"]["grade"] == pytest.approx(pre_i9005, abs=1e-9)

    def test_an_ABSENT_outcome_is_NOT_scored_zero(self):
        """A missing measurement must never be published as a measured zero —
        that would be a bar this layer has no authority to set."""
        inputs = dict(_today_inputs())
        del inputs["portfolio_outcome"]
        card = _roundtrip(compute_scorecard(**inputs))
        cov = card["overall"]["coverage"]
        assert cov["skip_classes"]["portfolio_outcome"] == "input_absent"
        assert "portfolio_outcome" not in cov["components_failed"]
        assert cov["weight_scored_zero"] == 0.0
        # And the loss of half the declared weight is LOUD, not silent.
        assert cov["weight_present"] == pytest.approx(0.50)
        assert card["overall"]["display"] != card["overall"]["letter"]

    def test_status_counts_the_outcome_voter(self):
        """`ok` may not mean "three of four declared voters graded"."""
        inputs = dict(_today_inputs())
        del inputs["portfolio_outcome"]
        assert _roundtrip(compute_scorecard(**inputs))["status"] == "partial"
        assert _roundtrip(compute_scorecard(**_full_inputs()))["status"] == "ok"
