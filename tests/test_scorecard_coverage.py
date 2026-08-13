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

FIXTURE = Path(__file__).parent / "fixtures" / "report_card_2026-08-07.json"

# ---------------------------------------------------------------------------
# The reader-side recomputation. Imports nothing from the producer on purpose.
# ---------------------------------------------------------------------------

_SECTIONS = ("research", "predictor", "executor")


def _recompute_overall(card: dict) -> float | None:
    """Reproduce ``card['overall']['grade']`` from the card's own JSON.

    Implements exactly the rule the card states in ``grading_weights.rule``:
    drop null components, renormalize over the surviving declared weights.
    """
    weights = card["grading_weights"]
    section_grades: dict[str, float | None] = {}
    for section in _SECTIONS:
        components = card[section]["components"]
        num = den = 0.0
        for name, w in weights[section].items():
            entry = components.get(name)
            grade = entry.get("grade") if isinstance(entry, dict) else None
            if grade is None:
                continue
            num += w * grade
            den += w
        section_grades[section] = (num / den) if den else None

    num = den = 0.0
    for name, w in weights["overall"].items():
        g = section_grades[name]
        if g is None:
            continue
        num += w * g
        den += w
    return (num / den) if den else None


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
            "overall": {"accuracy_10d": 0.58, "avg_alpha_10d": 1.2},
            "by_score_bucket": [{"bucket": "90+", "accuracy_10d": 0.66}],
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
        weights = card["grading_weights"]
        for section in _SECTIONS:
            components = card[section]["components"]
            num = den = 0.0
            for name, w in weights[section].items():
                entry = components.get(name)
                g = entry.get("grade") if isinstance(entry, dict) else None
                if g is None:
                    continue
                num += w * g
                den += w
            expected = (num / den) if den else None
            assert card[section]["grade"] == pytest.approx(expected, abs=1e-9)

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

    def test_research_reports_the_retired_pair(self, card):
        cov = card["research"]["coverage"]
        assert set(cov["components_skipped"]) == {"cio", "sector_teams_avg"}
        assert cov["weight_present"] == pytest.approx(0.55)
        assert cov["skip_classes"] == {"cio": "retired", "sector_teams_avg": "retired"}

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
        assert cov["weight_present_effective"] == pytest.approx(
            0.40 * 0.55 + 0.25 * 1.0 + 0.35 * 0.75,
        )
        assert cov["weight_present_effective"] < 0.75

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

    A failed component dropped from the denominator inflates the grade. This
    build does not change how a failure scores — it makes the masking loud.
    """

    def _timed_out(self):
        inputs = _today_inputs()
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
        # And it is loud in the rendering, distinctly from ordinary partial.
        assert cov["qualifier"] == "PARTIAL-MASKED-FAILURE"
        assert "DROPPED ON FAILURE" in card["executor"]["display"]

    def test_zero_is_emitted_as_a_value_when_nothing_failed(self):
        # `no data` must never be indistinguishable from `no failures`.
        card = _roundtrip(compute_scorecard(**_today_inputs()))
        for level in ("overall", "research", "predictor", "executor"):
            cov = card[level]["coverage"]
            assert cov["weight_failed"] == 0.0
            assert cov["components_failed"] == []
            assert cov["qualifier"] != "PARTIAL-MASKED-FAILURE"

    def test_a_masked_failure_reaches_the_HEADLINE_number(self):
        """The one that matters for showing the grade externally.

        A component that timed out inside the executor must not become a clean
        C+ at the top of the card. Without propagation the overall level reads
        100% covered (all three module grades non-null) and says nothing.
        """
        card = self._timed_out()
        cov = card["overall"]["coverage"]
        assert cov["qualifier"] == "PARTIAL-MASKED-FAILURE"
        assert "exit_rules" in cov["components_failed"]
        assert cov["weight_failed"] == pytest.approx(0.35 * 0.15)
        assert "DROPPED ON FAILURE" in card["overall"]["display"]

    def test_a_retirement_is_not_a_failure(self):
        card = _roundtrip(compute_scorecard(**_today_inputs()))
        cov = card["research"]["coverage"]
        assert cov["components_failed"] == []
        assert cov["weight_failed"] == 0.0


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
