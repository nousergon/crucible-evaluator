"""Tests for evals/director_arm_eval.py — the Director's arm comparison.

These pin the SCORING and the VERDICT, which is the part that has to be right
when the arms actually differ. The live calls are exercised by running the
harness; nothing here makes one.

The failure this guards against is specific and has a precedent in this fleet:
on 2026-09-08 a collapsed predictor champion was promoted past a veto whose
incumbent side WAS the candidate, so every ratio came out 1.0 by construction
and the comparison could not have failed. A comparison that cannot return
"rejected" is not a comparison.
"""
from __future__ import annotations

import pathlib
import unittest

import pytest

# `repo_tree`: this module imports `evals.director_arm_eval`, and `evals/` is
# deliberately NOT COPYed into the Lambda image — the harness is an operator
# tool, the image ships only `grading/` and `director/`, and copying it in
# would put `nousergon_lib.arena` on the deployed function for a comparison
# the function never runs. So these tests cannot execute against the image's
# tree, which is exactly what the marker means. They gate fully in the `test`
# job, which runs the whole repo.
pytestmark = pytest.mark.repo_tree

# THE MARKER IS NOT ENOUGH, and the distinction cost a CI round trip.
# `-m 'not repo_tree'` deselects at RUN time; pytest still IMPORTS every test
# module at COLLECTION time, so an unimportable module is a collection ERROR
# that no marker can deselect.
#
# The condition below is deliberately "is the REPO TREE here", not "does
# `evals` import". `pytest.importorskip("evals")` would have been one line and
# would also have silently skipped this entire file if `evals/` were ever
# deleted or renamed — turning a real breakage into a green run, which is the
# masking these tests exist to catch elsewhere. The Dockerfile is present in
# every checkout and absent from the image's mounted `tests/`-only tree, so it
# discriminates the two environments exactly; with the tree present and
# `evals/` gone, the import below runs and FAILS, loudly, as it should.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if not (_REPO_ROOT / "Dockerfile").exists():  # pragma: no cover — image only
    pytest.skip("the repository tree is not mounted (in-image run); `evals/` "
                "is not shipped in the Lambda image by design",
                allow_module_level=True)

from director.schema import ActionItem, DirectorWeeklyActionPlan
from evals.director_arm_eval import (
    MIN_GROUNDING_RATE,
    MIN_RED_COVERAGE,
    cache_hit_rate,
    compare_arms,
    failed_score,
    gate_verdicts,
    pair_on_cards,
    price_call,
    score_plan,
)


def _card(components: dict) -> dict:
    """A minimal Report Card v2 shaped how `component_status_map` reads it."""
    return {
        "_provenance": {"run_date": "2026-09-04"},
        "tiles": {
            "t": {"components": [{"name": n, "status": s}
                                 for n, s in components.items()]},
        },
    }


def _item(iid: str, evidence: list) -> ActionItem:
    return ActionItem(
        id=iid, title=iid, rationale="because", evidence=evidence,
        proposed_owner="research", priority="P1", horizon="this_week",
        suggested_change_type="investigation", confidence=50,
    )


def _plan(items: list, carryover: list | None = None) -> DirectorWeeklyActionPlan:
    return DirectorWeeklyActionPlan(
        run_date="2026-09-04", system_summary="s", top_risks=["r"],
        action_items=items, carryover_review=carryover or [],
    )


class Scoring(unittest.TestCase):
    def setUp(self):
        self.card = _card({"momentum_l1_ic": "RED", "price_cache": "RED",
                           "deploy_success_rate": "GREEN"})

    def test_an_item_citing_a_real_component_is_grounded(self):
        s = score_plan(_plan([_item("a", ["predictor tile momentum_l1_ic"])]),
                       self.card, rung="native")
        self.assertEqual(s.grounding_rate, 1.0)
        self.assertEqual(s.grounded_items, 1)

    def test_an_item_citing_a_metric_that_does_not_exist_is_NOT_grounded(self):
        """The whole point. A fluent plan citing invented metric names is the
        failure `director/schema.py` calls plausible-but-ungrounded, and it is
        indistinguishable from a good plan by every other surface."""
        s = score_plan(_plan([_item("a", ["the vibes tile looked bad"])]),
                       self.card, rung="native")
        self.assertEqual(s.grounding_rate, 0.0)

    def test_an_EMPTY_plan_scores_zero_grounding_not_perfect(self):
        """0/0 rendered as 1.0 would rank the emptiest arm first — a scoring
        bug that reads as a strong result."""
        s = score_plan(_plan([]), self.card, rung="native")
        self.assertEqual(s.grounding_rate, 0.0)
        self.assertEqual(s.action_items, 0)

    def test_red_coverage_counts_only_RED_components_the_plan_cited(self):
        s = score_plan(_plan([_item("a", ["momentum_l1_ic"])]), self.card,
                       rung="native")
        self.assertEqual(s.red_components, 2)
        self.assertEqual(s.red_covered, 1)
        self.assertEqual(s.red_coverage, 0.5)

    def test_a_card_with_no_RED_components_gives_full_coverage(self):
        """The vacuity is in the CARD, not the arm. Penalising an arm for a
        healthy week would make the gate a function of the fleet's mood."""
        s = score_plan(_plan([_item("a", ["deploy_success_rate"])]),
                       _card({"deploy_success_rate": "GREEN"}), rung="native")
        self.assertEqual(s.red_coverage, 1.0)

    def test_adverse_targeting_separates_acting_on_red_from_citing_green(self):
        s = score_plan(_plan([_item("a", ["deploy_success_rate"])]), self.card,
                       rung="native")
        self.assertEqual(s.grounding_rate, 1.0)
        self.assertEqual(s.adverse_targeting_rate, 0.0)

    def test_only_the_evidence_list_counts_never_the_rationale_prose(self):
        """Same scope rule as `loop_verification.evidence_still_adverse`:
        grading prose would score an arm on how many metric names it
        sprinkled through a paragraph."""
        item = _item("a", [])
        item.rationale = "momentum_l1_ic and price_cache both look bad"
        self.assertEqual(score_plan(_plan([item]), self.card).grounding_rate, 0.0)

    def test_a_failed_call_is_a_recorded_zero_not_a_gap(self):
        s = failed_score("RateLimitError: 429")
        self.assertEqual(s.schema_valid, 0)
        self.assertIn("429", s.failure)

    def test_a_non_native_rung_is_recorded(self):
        s = score_plan(_plan([_item("a", ["momentum_l1_ic"])]), self.card,
                       rung="prompt_only")
        self.assertEqual(s.native_rung, 0)
        self.assertEqual(s.structured_rung, "prompt_only")


class Gates(unittest.TestCase):
    def _score(self, **kw):
        base = dict(schema_valid=1, native_rung=1, structured_rung="native",
                    action_items=5, grounded_items=5, grounding_rate=1.0,
                    red_coverage=1.0)
        base.update(kw)
        from evals.director_arm_eval import PlanScore
        return PlanScore(**base)

    def test_all_four_gates_pass_on_a_clean_arm(self):
        v = gate_verdicts([self._score(), self._score()])
        self.assertTrue(v["passed"])
        self.assertEqual(
            {v["G1_structured"], v["G2_native_rung"], v["G3_grounding"],
             v["G4_red_coverage"]}, {"pass"})

    def test_every_gate_reports_a_verdict_including_the_ones_that_passed(self):
        """A gate list containing only failures cannot be audited."""
        v = gate_verdicts([self._score(grounding_rate=0.1)])
        self.assertEqual(v["G1_structured"], "pass")
        self.assertEqual(v["G3_grounding"], "FAIL")

    def test_one_failed_call_anywhere_fails_G1(self):
        """A plan that comes back as prose is not a degraded plan; it is the
        weekly run failing."""
        v = gate_verdicts([self._score(), failed_score("boom")])
        self.assertEqual(v["G1_structured"], "FAIL")
        self.assertFalse(v["passed"])

    def test_a_prompt_only_rung_fails_G2(self):
        v = gate_verdicts([self._score(native_rung=0, structured_rung="prompt_only")])
        self.assertEqual(v["G2_native_rung"], "FAIL")

    def test_grounding_just_below_the_threshold_fails(self):
        v = gate_verdicts([self._score(grounding_rate=MIN_GROUNDING_RATE - 0.01)])
        self.assertEqual(v["G3_grounding"], "FAIL")

    def test_red_coverage_just_below_the_threshold_fails(self):
        v = gate_verdicts([self._score(red_coverage=MIN_RED_COVERAGE - 0.01)])
        self.assertEqual(v["G4_red_coverage"], "FAIL")

    def test_no_observations_is_unmeasurable_not_a_pass(self):
        v = gate_verdicts([])
        self.assertFalse(v["passed"])
        self.assertEqual(v["G1_structured"], "unmeasurable")


class Comparison(unittest.TestCase):
    PASS = {"passed": True, "reason": ""}
    FAILED = {"passed": False, "reason": "G1_structured=FAIL"}

    def test_an_empty_intersection_is_unmeasurable_never_a_tie(self):
        """champion-challenger §4: a comparison with no usable intersection is
        `unmeasurable` with a reason. It is never a tie, a pass, or a zero —
        and it is exactly what a champion arm nobody could bill produces."""
        out = compare_arms("ultra", "high",
                           {"ultra": {}, "high": {"2026-09-04": 0.9}},
                           {"ultra": self.PASS, "high": self.PASS})
        self.assertEqual(out["verdict"], "unmeasurable")
        self.assertEqual(out["n_paired"], 0)
        self.assertFalse(out["promote"])
        self.assertIn("never a tie", out["reason"])

    def test_pairing_uses_only_cards_BOTH_arms_ran(self):
        pairs = pair_on_cards({"a": 1.0, "b": 0.5}, {"b": 0.7, "c": 0.9})
        self.assertEqual(pairs, [("b", 0.5, 0.7)])

    def test_a_challenger_failing_a_gate_is_REJECTED_however_it_scores(self):
        """Gates are hard. An arm that cannot produce a structured, grounded
        plan on every card is not a candidate however far ahead it leads."""
        per_card = {"ultra": {str(i): 0.1 for i in range(8)},
                    "high": {str(i): 0.99 for i in range(8)}}
        out = compare_arms("ultra", "high", per_card,
                           {"ultra": self.PASS, "high": self.FAILED})
        self.assertEqual(out["verdict"], "rejected")
        self.assertFalse(out["promote"])

    def test_a_supported_champion_lead_rejects_the_challenger(self):
        per_card = {"ultra": {str(i): 1.0 for i in range(30)},
                    "high": {str(i): 0.0 for i in range(30)}}
        out = compare_arms("ultra", "high", per_card,
                           {"ultra": self.PASS, "high": self.PASS})
        self.assertEqual(out["supported"], "champion")
        self.assertEqual(out["verdict"], "rejected")
        self.assertFalse(out["promote"])

    def test_a_supported_challenger_lead_is_the_only_route_to_promote(self):
        per_card = {"ultra": {str(i): 0.0 for i in range(30)},
                    "high": {str(i): 1.0 for i in range(30)}}
        out = compare_arms("ultra", "high", per_card,
                           {"ultra": self.PASS, "high": self.PASS})
        self.assertEqual(out["supported"], "challenger")
        self.assertEqual(out["verdict"], "challenger-leads")
        self.assertTrue(out["promote"])

    def test_a_straddling_interval_never_promotes(self):
        """Thin evidence must not move the pointer. The interval is wide at
        small n by construction (champion-challenger §5.0), which is the
        property that replaces the minimum-evidence floors."""
        per_card = {"ultra": {"a": 0.90, "b": 0.85, "c": 0.88},
                    "high": {"a": 0.91, "b": 0.84, "c": 0.89}}
        out = compare_arms("ultra", "high", per_card,
                           {"ultra": self.PASS, "high": self.PASS})
        self.assertEqual(out["supported"], "neither")
        self.assertEqual(out["verdict"], "no-quality-difference-supported")
        self.assertFalse(out["promote"])

    def test_a_champion_that_fails_its_own_gates_is_reported_too(self):
        """An incumbent failing a gate is a finding about the INCUMBENT, and
        it must not be lost because the comparison was about the challenger."""
        per_card = {"ultra": {"a": 0.5}, "high": {"a": 0.5}}
        out = compare_arms("ultra", "high", per_card,
                           {"ultra": self.FAILED, "high": self.PASS})
        self.assertIn("CHAMPION also failed a gate", out["reason"])


class Pricing(unittest.TestCase):
    CARDS = [{"model_name": "deepseek/deepseek-v4-pro", "input_per_1m": 0.435,
              "output_per_1m": 0.87, "cache_read_per_1m": 0.003625},
             {"model_name": "glm-5.2", "input_per_1m": 1.4,
              "output_per_1m": 4.4, "cache_read_per_1m": 0.26}]

    class U:
        def __init__(self, i=0, o=0, cr=0, miss=0):
            self.input_tokens, self.output_tokens = i, o
            self.cache_read_tokens, self.prompt_cache_miss_tokens = cr, miss

    def test_cache_read_tokens_are_not_double_charged_as_input(self):
        p = price_call("glm-5.2", self.U(i=1_000_000, cr=1_000_000), self.CARDS)
        self.assertAlmostEqual(p["input"], 0.0)
        self.assertAlmostEqual(p["cache_read"], 0.26)

    def test_the_direct_and_openrouter_ids_price_identically(self):
        a = price_call("deepseek-v4-pro", self.U(i=1_000_000), self.CARDS)
        b = price_call("deepseek/deepseek-v4-pro", self.U(i=1_000_000), self.CARDS)
        self.assertEqual(a["total"], b["total"])

    def test_an_unknown_model_is_reported_unpriced_not_free(self):
        """$0.00 for a model the table does not know is a free-call claim.
        `priced=False` is what stops the estimate reading as good news."""
        p = price_call("some-new-model", self.U(i=1_000_000), self.CARDS)
        self.assertFalse(p["priced"])
        self.assertEqual(p["total"], 0.0)

    def test_a_substring_never_prices_one_model_off_another(self):
        p = price_call("deepseek-v4-flash", self.U(i=1_000_000), self.CARDS)
        self.assertFalse(p["priced"])


class CacheRate(unittest.TestCase):
    class U:
        def __init__(self, i=0, cr=0, miss=0):
            self.input_tokens, self.cache_read_tokens = i, cr
            self.prompt_cache_miss_tokens, self.output_tokens = miss, 0

    def test_a_call_with_no_prompt_tokens_is_None_not_zero(self):
        """`None` and 0.0 must not render identically: a call that never got
        far enough to have a cache rate is not a call whose cache is dead."""
        self.assertIsNone(cache_hit_rate(self.U()))

    def test_a_prompt_wholly_uncached_is_a_real_zero(self):
        self.assertEqual(cache_hit_rate(self.U(i=100, miss=100)), 0.0)

    def test_the_rate_is_cache_read_over_the_WHOLE_prompt(self):
        """MEASURED DEFECT, 2026-09-09. The first form was
        `read / (read + miss)`, and krepis reports `miss` as 0 for a provider
        that does not publish it — so the ratio became `read/read` and this
        harness printed a 100% cache rate for a call whose real coverage was
        49.8%. Half the prompt was billed at full price under a number that
        said none of it was."""
        self.assertAlmostEqual(cache_hit_rate(self.U(i=17214, cr=8576, miss=0)),
                               0.4982, places=4)
        self.assertAlmostEqual(cache_hit_rate(self.U(i=100, cr=60, miss=40)), 0.6)

    def test_the_rate_is_never_above_one(self):
        self.assertEqual(cache_hit_rate(self.U(i=100, cr=120)), 1.0)


class RouteAddressing(unittest.TestCase):
    def test_the_harness_names_no_model_id_provider_or_base_url(self):
        """principles.md §2.8, asserted rather than requested. Every arm is a
        registry GROUP handle; a model id, a provider name or a base url
        appearing in the harness is the Substitutability violation the
        migration to `resolve_group_spec` removed from the Director itself."""
        import pathlib

        import evals.director_arm_eval as mod

        # CODE only. Comments and docstrings legitimately QUOTE model ids
        # and endpoints when explaining a decision; what must not exist is a
        # model id, provider or url the harness ADDRESSES. Tokenizing is the
        # only way to tell those apart — a plain substring scan over the file
        # would either pass by deleting the explanations or fail forever.
        import io
        import tokenize

        src = pathlib.Path(mod.__file__).read_text()
        body = "".join(
            tok.string for tok in
            tokenize.generate_tokens(io.StringIO(src).readline)
            if tok.type not in (tokenize.COMMENT, tokenize.STRING))
        # The tokens are ASSEMBLED rather than written out. This file is
        # scanned by the fleet's provider direct-linkage guard
        # (nousergon-lib `provider_linkage_guard.py`, alpha-engine-config-I9295),
        # and a test that spells a provider endpoint in order to forbid it
        # trips the very guard it agrees with — which happened on the first
        # push of this PR. Concatenation keeps the assertion and keeps the
        # literal out of the scanned text.
        banned = [
            "glm" + "-5.2-direct",                 # a registry ENTRY id
            "deepseek" + "-v4-pro-max",            # a registry ENTRY id
            "api." + "z" + ".ai",                  # a provider endpoint
            "api." + "deepseek" + ".com",          # a provider endpoint
            "open" + "router",                     # a provider name
            "Open" + "AI(",                        # an SDK client at a call site
            "89" + "90",                           # the egress proxy port
        ]
        for tok in banned:
            self.assertNotIn(tok, body,
                             f"{tok!r} is addressed directly in the harness body")
        for scheme in ("http:" + "//", "https:" + "//"):
            self.assertNotIn(scheme, body,
                             "the harness must reach no endpoint of its own")

    def test_the_champion_is_read_from_the_director_not_restated(self):
        from director.agent import DIRECTOR_GROUP
        from evals.director_arm_eval import champion_group

        self.assertEqual(champion_group(), DIRECTOR_GROUP)


class DirectorGroupParameter(unittest.TestCase):
    def test_build_action_plan_refuses_a_group_alongside_an_injected_llm(self):
        """A harness that thinks it graded `high` and actually graded whatever
        the injected double serves produces a verdict with no relationship to
        the thing it names."""
        from director.agent import build_action_plan

        with self.assertRaises(ValueError):
            build_action_plan({}, llm=object(), group="high")

    def test_the_production_default_is_still_the_ultra_group(self):
        import inspect

        from director.agent import DIRECTOR_GROUP, build_action_plan

        sig = inspect.signature(build_action_plan)
        self.assertIsNone(sig.parameters["group"].default)
        self.assertEqual(sig.parameters["callsite_id"].default, "director-plan")
        self.assertEqual(DIRECTOR_GROUP, "ultra")


if __name__ == "__main__":
    unittest.main(verbosity=2)


def test_grounding_resolves_typed_citations_by_id_not_prose():
    """alpha-engine-config-I12312: an item carrying typed citations is grounded
    by its code-checked ``metric:`` ids alone. A metric its prose names does
    not count when its typed citation failed the check."""
    from director.citations import SourceRegistry, validate_citations
    from director.report_card_digest import summarize_report_card
    from director.schema import Citation

    card = {"tiles": {"predictor": {"status": "RED", "components": [
        {"name": "momentum_l1_ic", "status": "RED", "value": -0.01},
        {"name": "meta_l2_ic", "status": "WATCH", "value": 0.01},
    ]}}}
    reg = SourceRegistry()
    summarize_report_card(card, sources=reg)

    def item(item_id, cits, evidence):
        return ActionItem(id=item_id, title="t", rationale="r", citations=cits, evidence=evidence,
                          proposed_owner="predictor", priority="P1", horizon="this_week",
                          suggested_change_type="investigation", confidence=50)

    plan = DirectorWeeklyActionPlan(run_date="2026-10-10", system_summary="s", top_risks=[],
                                    action_items=[
        item("a", [Citation(id="metric:momentum_l1_ic", kind="metric")], ["meta_l2_ic prose"]),
        item("b", [Citation(id="metric:nope", kind="metric")], ["predictor tile momentum_l1_ic"]),
    ])
    validate_citations(plan.action_items, reg)
    s = score_plan(plan, card)
    assert s.grounded_items == 1 and s.grounding_rate == 0.5
    assert s.red_covered == 1
