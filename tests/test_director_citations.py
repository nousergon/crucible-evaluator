"""Typed, code-checked citations for the Director plan and retro judge
(alpha-engine-config-I12312; the citation-identity half of I12101).

Covers: the structured-output schema (what the model is and is not asked to
produce), the offered-source registry (every tag in the prompt is a registered
id and vice versa), the deterministic validator (valid / unknown / kind
mismatch / quote / empty), backward compatibility with legacy prose-evidence
plans on disk, and an end-to-end run of both calls with a stubbed LLM.
"""

from __future__ import annotations

import json
import re

import pytest

from director import agent, retro
from director.carryover import merge_plan_into_ledger
from director.citations import (
    CITATION_CHECK_KEY,
    CITATION_CONTRACT_VERSION,
    CITATION_SOURCES_KEY,
    GROUNDED,
    KIND_MISMATCH,
    QUOTE_NOT_FOUND,
    UNGROUNDED,
    UNKNOWN_ID,
    VALID,
    SourceRegistry,
    check_citation,
    source_id,
    valid_citation_ids,
    validate_citations,
    validate_grade_citations,
)
from director.issue_filer import (
    find_reconfirm_match,
    parse_evidence_from_body,
    render_issue,
)
from director.loop_verification import (
    citation_mode,
    component_status_map,
    evidence_label,
    item_cited_metrics,
    item_evidence_still_adverse,
    verify_and_correct,
)
from director.report_card_digest import summarize_report_card
from director.schema import (
    ActionItem,
    Citation,
    DirectorWeeklyActionPlan,
    RetroCitation,
    RetroGrade,
)
from director.verdict import stamp_plan_artifact

RUN_DATE = "2026-10-10"
#: A rendered tag. The kind prefix is part of the pattern so the instruction
#: text's own ``{cite=<id>}`` placeholder is not mistaken for a tag.
_TAG = re.compile(r"\{cite=((?:metric|tile|card|carryover|issue|artifact|plan_item):[^}]+)\}")

CARD = {
    "_provenance": {"run_date": RUN_DATE},
    "tiles_overall_status": "RED",
    "attestation": {"verdict": "PASS", "as_of": {}},
    "degraded_component_census": False,
    "tiles": {
        "predictor": {"status": "RED", "letter": "F", "numeric_grade": 40.0, "components": [
            {"name": "momentum_l1_ic", "criticality": "critical", "status": "RED",
             "value": -0.0015, "target": 0.03, "red_line": 0.0, "status_reason": "dead L1."},
            {"name": "meta_l2_ic", "criticality": "critical", "status": "GREEN", "value": 0.05},
        ]},
        "portfolio_outcome": {"status": "WATCH", "letter": "C", "components": [
            {"name": "excess_return", "criticality": "critical", "status": "WATCH", "value": -0.01,
             "window": {"label": "13w", "sources": ["s3://alpha-engine-research/trades/eod_pnl.csv"]}},
        ]},
        "substrate": {"status": "GREEN", "letter": "A", "components": [
            {"name": "artifact_freshness", "criticality": "critical", "status": "GREEN", "value": 1.0},
        ]},
    },
}


def _item(item_id="revive-momentum-l1", citations=None, evidence=None, **kw) -> ActionItem:
    return ActionItem(
        id=item_id, title=kw.pop("title", "Revive momentum L1"),
        rationale=kw.pop("rationale", "momentum_l1_ic = -0.0015 (RED)."),
        citations=citations or [], evidence=evidence or [],
        proposed_owner="predictor", priority="P1", horizon="this_week",
        suggested_change_type="investigation", confidence=60, **kw,
    )


def _plan(items) -> DirectorWeeklyActionPlan:
    return DirectorWeeklyActionPlan(
        run_date=RUN_DATE, system_summary="s", top_risks=["r"], action_items=items,
    )


# ── schema ──────────────────────────────────────────────────────────────────


class TestSchema:
    def test_llm_schema_asks_for_citations_but_never_for_the_check(self):
        schema = json.dumps(DirectorWeeklyActionPlan.model_json_schema())
        cit = DirectorWeeklyActionPlan.model_json_schema()["$defs"]["Citation"]
        assert set(cit["properties"]) == {"id", "kind", "quote"}
        assert cit["required"] == ["id", "kind"]
        assert cit["properties"]["kind"]["enum"] == [
            "metric", "tile", "card", "carryover", "issue", "artifact", "plan_item"]
        # The verdicts are code's, not the model's.
        assert "check_reason" not in schema
        assert "citation_grounding" not in schema
        item = DirectorWeeklyActionPlan.model_json_schema()["$defs"]["ActionItem"]
        assert "citations" in item["properties"] and "evidence" in item["properties"]

    def test_off_vocabulary_kind_parses_and_is_recorded_not_raised(self):
        c = Citation(id="metric:momentum_l1_ic", kind="component")
        reg = SourceRegistry()
        reg.offer("metric", "momentum_l1_ic", "line")
        assert check_citation(c, reg)[0] == KIND_MISMATCH

    def test_legacy_plan_on_disk_still_parses(self):
        legacy = {
            "run_date": "2026-08-22", "system_summary": "s", "top_risks": [],
            "action_items": [{
                "id": "x", "title": "t", "rationale": "r",
                "evidence": ["predictor tile momentum_l1_ic"], "proposed_owner": "predictor",
                "priority": "P0", "horizon": "this_week",
                "suggested_change_type": "structural", "confidence": 50,
            }],
        }
        plan = DirectorWeeklyActionPlan.model_validate(legacy)
        assert plan.action_items[0].citations == []
        assert plan.action_items[0].citation_grounding is None

    def test_retro_grade_citations_carry_a_dimension(self):
        props = RetroGrade.model_json_schema()["$defs"]["RetroCitation"]["properties"]
        assert set(props) == {"id", "kind", "quote", "dimension"}
        g = RetroGrade(prior_run_date="d", grounding=1, calibration=1, actionability=1)
        assert g.citations == []


# ── offered-source registry ────────────────────────────────────────────────


class TestRegistry:
    def test_source_id_grammar(self):
        assert source_id("metric", " momentum_l1_ic ") == "metric:momentum_l1_ic"
        with pytest.raises(ValueError):
            source_id("prose", "x")
        with pytest.raises(ValueError):
            source_id("metric", "  ")

    def test_untagged_digest_is_unchanged_without_a_registry(self):
        assert "{cite=" not in summarize_report_card(CARD)

    def test_every_tag_in_the_digest_is_a_registered_id_and_vice_versa(self):
        reg = SourceRegistry()
        text = summarize_report_card(CARD, sources=reg)
        assert set(_TAG.findall(text)) == set(reg.ids())
        ids = set(reg.ids())
        assert {"metric:momentum_l1_ic", "metric:excess_return", "tile:predictor",
                "tile:portfolio_outcome", "tile:substrate", "card:overall",
                "card:attestation", "card:census",
                f"artifact:evaluator/{RUN_DATE}/report_card.json",
                "artifact:s3://alpha-engine-research/trades/eod_pnl.csv"} <= ids
        # A GREEN component the digest only COUNTS is not rendered by name, so
        # it is not citable.
        assert "metric:meta_l2_ic" not in ids
        assert "metric:artifact_freshness" not in ids

    def test_pinned_green_component_becomes_citable(self):
        reg = SourceRegistry()
        summarize_report_card(CARD, pinned_components={"meta_l2_ic"}, sources=reg)
        assert "metric:meta_l2_ic" in reg

    def test_prompt_tags_equal_registry_including_carryover_and_issues(self):
        ledger = {"items": [{"id": "fix-pnl", "title": "Fix PnL", "status": "carried_over",
                             "priority": "P1", "proposed_owner": "data",
                             "evidence": ["excess_return"]}]}
        messages, reg = agent.build_messages_with_sources(
            CARD, carryover=ledger,
            roadmap_digest="#123 [P1] some open work\nnot an issue line",
            resolved_digest="#99 (closed 2026-10-01) prior fix",
        )
        human = messages[1][1]
        assert set(_TAG.findall(human)) == set(reg.ids())
        assert {"carryover:fix-pnl", "issue:alpha-engine-config-I123",
                "issue:alpha-engine-config-I99"} <= set(reg.ids())
        assert "not an issue line\n" in human + "\n"  # untagged line passes through
        assert "CITATIONS (required, code-checked)" in human
        # build_messages (the old surface) renders the same prompt.
        assert agent.build_messages(
            CARD, carryover=ledger, roadmap_digest="#123 [P1] some open work\nnot an issue line",
            resolved_digest="#99 (closed 2026-10-01) prior fix",
        ) == messages

    def test_elided_carryover_rows_are_not_citable(self):
        n = agent.CARRYOVER_PROMPT_MAX_ITEMS + 5
        ledger = {"items": [
            {"id": f"row-{i:02d}", "title": f"row {i}", "status": "carried_over",
             "priority": "P3", "proposed_owner": "data"} for i in range(n)
        ]}
        messages, reg = agent.build_messages_with_sources(CARD, carryover=ledger)
        rows = reg.ids_of_kind("carryover")
        assert 0 < len(rows) < n
        for i in range(n):
            sid = f"carryover:row-{i:02d}"
            assert (sid in reg) == (("{cite=" + sid + "}") in messages[1][1])

    def test_manifest_is_deterministic_and_round_trips(self):
        a, b = SourceRegistry(), SourceRegistry()
        summarize_report_card(CARD, sources=a)
        summarize_report_card(CARD, sources=b)
        assert a.manifest() == b.manifest()
        assert a.manifest()["contract"] == CITATION_CONTRACT_VERSION
        again = SourceRegistry.from_manifest(a.manifest())
        assert again.ids() == a.ids()
        assert again.manifest()["sha256"] == a.manifest()["sha256"]


# ── validator ──────────────────────────────────────────────────────────────


@pytest.fixture
def reg():
    r = SourceRegistry()
    summarize_report_card(CARD, sources=r)
    return r


class TestValidator:
    def test_valid_with_and_without_quote(self, reg):
        assert check_citation(Citation(id="metric:momentum_l1_ic", kind="metric"), reg) == (VALID, None)
        # Quote whitespace is collapsed for the comparison; the id never is.
        q = Citation(id="metric:momentum_l1_ic", kind="metric", quote="value  -0.0015")
        assert check_citation(q, reg)[0] == VALID
        assert check_citation(Citation(id="metric:momentum_l1_ic", kind="metric",
                                       quote="reason: dead L1."), reg)[0] == VALID

    def test_unknown_id_is_never_repaired(self, reg):
        for bad in ("metric:Momentum_L1_IC", "metric: momentum_l1_ic", "momentum_l1_ic",
                    "metric:momentum_l1", "predictor tile momentum_l1_ic"):
            status, reason = check_citation(Citation(id=bad, kind="metric"), reg)
            assert status == UNKNOWN_ID, bad
            assert repr(bad) in reason

    def test_kind_mismatch_and_quote_not_found(self, reg):
        assert check_citation(Citation(id="tile:predictor", kind="metric"), reg)[0] == KIND_MISMATCH
        assert check_citation(Citation(id="metric:momentum_l1_ic", kind="metric",
                                       quote="value 0.9"), reg)[0] == QUOTE_NOT_FOUND

    def test_validate_plan_records_invalid_and_flags_ungrounded(self, reg):
        good = _item("good", [Citation(id="metric:momentum_l1_ic", kind="metric"),
                              Citation(id="metric:ghost", kind="metric")])
        bad_only = _item("bad-only", [Citation(id="metric:ghost2", kind="metric")])
        empty = _item("empty")
        block = validate_citations([good, bad_only, empty], reg)
        assert block["n_items"] == 3 and block["grounded_items"] == 1
        assert block["ungrounded_item_ids"] == ["bad-only", "empty"]
        assert block["counts"][VALID] == 1 and block["counts"][UNKNOWN_ID] == 2
        assert {r["id"] for r in block["invalid"]} == {"metric:ghost", "metric:ghost2"}
        # Recorded on the citation, never dropped from the item.
        assert [c.check for c in good.citations] == [VALID, UNKNOWN_ID]
        assert len(bad_only.citations) == 1 and bad_only.citations[0].check == UNKNOWN_ID
        assert (good.citation_grounding, bad_only.citation_grounding, empty.citation_grounding) \
            == (GROUNDED, UNGROUNDED, UNGROUNDED)
        assert block["offered_sha256"] == reg.manifest()["sha256"]

    def test_validate_works_on_ledger_dicts(self, reg):
        row = {"id": "x", "citations": [{"id": "tile:predictor", "kind": "tile"}]}
        block = validate_citations([row], reg)
        assert row["citations"][0]["check"] == VALID and row["citation_grounding"] == GROUNDED
        assert block["grounded_items"] == 1

    def test_unchecked_citations_carry_no_identity(self):
        row = {"citations": [{"id": "metric:momentum_l1_ic", "kind": "metric"}]}
        assert valid_citation_ids(row) == []
        assert valid_citation_ids({"evidence": ["x"]}) is None


# ── consumers: loop verification, issue filer ──────────────────────────────


STATUS_MAP = component_status_map(CARD)


def _typed_row(*cits, **kw) -> dict:
    return {"id": "r", "title": kw.get("title", "t"), "rationale": kw.get("rationale", ""),
            "evidence": kw.get("evidence", []), "citations": list(cits)}


class TestLoopVerification:
    def test_typed_row_resolves_by_id_not_by_prose(self):
        row = _typed_row({"id": "metric:momentum_l1_ic", "kind": "metric", "check": VALID},
                         evidence=["portfolio_outcome excess_return"])
        assert citation_mode(row) == "typed"
        assert item_cited_metrics(row, STATUS_MAP) == {"momentum_l1_ic": "RED"}
        assert item_evidence_still_adverse(row, STATUS_MAP) == "adverse"
        # The annotation scope still reads prose too.
        assert item_cited_metrics(row, STATUS_MAP, wide=True) == {
            "momentum_l1_ic": "RED", "excess_return": "WATCH"}

    def test_typed_row_with_no_valid_metric_is_unverifiable_even_if_prose_names_one(self):
        row = _typed_row({"id": "metric:momentum_l1_ic", "kind": "metric", "check": UNKNOWN_ID},
                         {"id": "tile:predictor", "kind": "tile", "check": VALID},
                         evidence=["predictor tile momentum_l1_ic"])
        assert item_cited_metrics(row, STATUS_MAP) == {}
        assert item_evidence_still_adverse(row, STATUS_MAP) == "unverifiable"

    def test_typed_green_metric_reads_recovered(self):
        row = _typed_row({"id": "metric:meta_l2_ic", "kind": "metric", "check": VALID})
        assert item_evidence_still_adverse(row, STATUS_MAP) == "recovered"

    def test_legacy_prose_row_keeps_the_regex_path(self):
        row = {"id": "old", "evidence": ["predictor tile momentum_l1_ic"]}
        assert citation_mode(row) == "legacy"
        assert item_evidence_still_adverse(row, STATUS_MAP) == "adverse"
        assert evidence_label(row) == "predictor tile momentum_l1_ic"

    def test_evidence_label_names_valid_ids(self):
        row = _typed_row({"id": "metric:momentum_l1_ic", "kind": "metric", "check": VALID},
                         {"id": "metric:ghost", "kind": "metric", "check": UNKNOWN_ID})
        assert evidence_label(row) == "metric:momentum_l1_ic"

    def test_verify_and_correct_counts_the_identity_path(self):
        rows = [
            {**_typed_row({"id": "metric:momentum_l1_ic", "kind": "metric", "check": VALID}),
             "id": "typed", "issue_number": 1},
            {"id": "legacy", "issue_number": 2, "evidence": ["meta_l2_ic"]},
        ]
        calls = []

        def gh(method, url, token, body=None):
            calls.append((method, url, body))
            if method == "GET" and url.endswith("/comments"):
                return 200, []
            if method == "GET":
                return 200, {"state": "closed", "labels": []}
            return 201, {}

        out = verify_and_correct(rows, CARD, repo="o/r", token="t", gh_request=gh)
        assert out["resolved_typed"] == 1 and out["resolved_legacy"] == 1
        assert out["reopened_issues"] == [1] and out["closed_verified"] == 1
        reopen_comment = [b for m, u, b in calls if m == "POST" and u.endswith("/1/comments")][0]
        assert "metric:momentum_l1_ic still reads RED/WATCH" in reopen_comment["body"]


class TestIssueFiler:
    def test_typed_item_files_ids_and_a_citation_check_section(self, reg):
        item = _item("x", [Citation(id="metric:momentum_l1_ic", kind="metric"),
                           Citation(id="metric:ghost", kind="metric")],
                     evidence=["L1 IC has been dead for weeks"])
        validate_citations([item], reg)
        _, body = render_issue(item, RUN_DATE)
        assert "## Evidence\nmetric:momentum_l1_ic, L1 IC has been dead for weeks\n" in body
        assert "## Citation check\n1 of 2 citation(s) valid" in body
        assert "`metric:ghost` (metric) — unknown_id" in body
        assert "Ungrounded" not in body
        # The Evidence parser reads the ids and stops before the check section.
        assert parse_evidence_from_body(body) == {
            "metric:momentum_l1_ic", "l1 ic has been dead for weeks"}

    def test_ungrounded_item_is_flagged_on_the_issue(self, reg):
        item = _item("x")
        validate_citations([item], reg)
        _, body = render_issue(item, RUN_DATE)
        assert "## Citation check\n0 of 0 citation(s) valid" in body
        assert "**Ungrounded**" in body

    def test_legacy_item_renders_exactly_as_before(self):
        item = _item("x", evidence=["predictor.momentum_l1_ic"])
        _, body = render_issue(item, RUN_DATE)
        assert "## Citation check" not in body
        assert "## Evidence\npredictor.momentum_l1_ic\n" in body

    def test_reconfirm_match_on_typed_ids(self, reg):
        item = _item("new-slug", [Citation(id="metric:momentum_l1_ic", kind="metric")])
        validate_citations([item], reg)
        prior_body = "## Evidence\nmetric:momentum_l1_ic\n\n## Closes when\n..."
        unrelated = "## Evidence\nmetric:excess_return\n\n## Closes when\n..."
        got = find_reconfirm_match(item, [{"number": 5, "body": prior_body},
                                          {"number": 6, "body": unrelated}])
        assert [p["number"] for p in got] == [5]


# ── end to end, stubbed LLMs ───────────────────────────────────────────────


class _PlanLLM:
    """Reads the prompt it is given and cites from it — one valid citation with
    a quote, one invented id, and one item with no citations at all."""

    def __init__(self):
        self.human = None

    def invoke(self, messages):
        self.human = messages[1][1]
        offered = set(_TAG.findall(self.human))
        assert "metric:momentum_l1_ic" in offered
        return _plan([
            _item("revive-momentum-l1", [
                Citation(id="metric:momentum_l1_ic", kind="metric", quote="RED"),
                Citation(id="card:attestation", kind="card"),
                Citation(id="metric:momentum_l1_ic_v2", kind="metric"),
            ], evidence=["L1 dead"]),
            _item("no-cite", title="Vague item", rationale="things look bad"),
        ])


class _JudgeLLM:
    def __init__(self):
        self.human = None

    def invoke(self, messages):
        self.human = messages[1][1]
        return RetroGrade(
            prior_run_date="", grounding=70, calibration=60, actionability=80,
            citations=[
                RetroCitation(id="plan_item:revive-momentum-l1", kind="plan_item",
                              dimension="grounding"),
                RetroCitation(id="metric:momentum_l1_ic", kind="metric", dimension="calibration",
                              quote="value -0.0015"),
                RetroCitation(id="plan_item:invented", kind="plan_item", dimension="actionability"),
            ],
        )


def test_end_to_end_plan_then_ledger_then_loop_then_retro():
    llm = _PlanLLM()
    plan = agent.build_action_plan(CARD, run_date=RUN_DATE, llm=llm)

    # Per-citation and per-item stamps.
    good, vague = plan.action_items
    assert [c.check for c in good.citations] == [VALID, VALID, UNKNOWN_ID]
    assert good.citation_grounding == GROUNDED and vague.citation_grounding == UNGROUNDED

    # Plan-level block + the offered set, both archived in action_plan.json.
    body = json.loads(stamp_plan_artifact(plan, {"verdict": "PASS"}))
    check = body[CITATION_CHECK_KEY]
    assert check["n_items"] == 2 and check["grounded_items"] == 1
    assert check["ungrounded_item_ids"] == ["no-cite"]
    assert check["invalid"] == [{
        "item_id": "revive-momentum-l1", "id": "metric:momentum_l1_ic_v2", "kind": "metric",
        "status": UNKNOWN_ID, "reason": "'metric:momentum_l1_ic_v2' was not offered in the input"}]
    manifest = body[CITATION_SOURCES_KEY]
    assert set(manifest["sources"]) == set(_TAG.findall(llm.human))
    assert check["offered_sha256"] == manifest["sha256"]
    # The archived plan is re-checkable from the artifact alone.
    reparsed = DirectorWeeklyActionPlan.model_validate(body)
    again = validate_citations(reparsed.action_items, SourceRegistry.from_manifest(manifest))
    assert again["counts"] == check["counts"]
    assert body["action_items"][0]["citations"][0]["check"] == VALID

    # Ledger rows keep the stamped citations; loop verification resolves by id.
    ledger = merge_plan_into_ledger({"items": []}, plan, RUN_DATE)
    row = next(r for r in ledger["items"] if r["id"] == "revive-momentum-l1")
    assert citation_mode(row) == "typed"
    assert item_evidence_still_adverse(row, STATUS_MAP) == "adverse"

    # Next week's Director prompt tags the carried rows as citable.
    _, reg2 = agent.build_messages_with_sources(CARD, carryover=ledger)
    assert "carryover:revive-momentum-l1" in reg2

    # Retro judge: prior items are offered as plan_item ids, and the judge
    # sees each item's deterministic check.
    judge = _JudgeLLM()
    grade = retro.grade_prior_plan(json.loads(plan.model_dump_json()), CARD, llm=judge)
    assert "{cite=plan_item:revive-momentum-l1}" in judge.human
    assert "citations [grounded]: metric:momentum_l1_ic (valid)" in judge.human
    # A checked item with no citations is shown as ungrounded, not omitted.
    assert "citations [ungrounded]: none" in judge.human
    assert [c.check for c in grade.citations] == [VALID, VALID, UNKNOWN_ID]
    gcheck = getattr(grade, CITATION_CHECK_KEY)
    assert gcheck["grounding"] == GROUNDED
    assert gcheck["valid_by_dimension"] == {"grounding": 1, "calibration": 1}
    assert [r["id"] for r in gcheck["invalid"]] == ["plan_item:invented"]
    assert set(getattr(grade, CITATION_SOURCES_KEY)["sources"]) == set(_TAG.findall(judge.human))
    assert grade.prior_run_date == RUN_DATE


def test_retro_of_a_legacy_plan_still_grades():
    legacy = {"run_date": "2026-08-22", "system_summary": "s", "top_risks": ["r"],
              "action_items": [{"id": "old", "title": "Old", "rationale": "r",
                                "evidence": ["predictor tile momentum_l1_ic"],
                                "proposed_owner": "predictor", "priority": "P1"}]}
    judge = _JudgeLLM()
    grade = retro.grade_prior_plan(legacy, CARD, llm=judge)
    assert "[evidence: predictor tile momentum_l1_ic]" in judge.human
    assert "citations [" not in judge.human
    assert "{cite=plan_item:old}" in judge.human
    # The judge's citation of an item the legacy plan did not contain is invalid.
    assert [c.check for c in grade.citations] == [UNKNOWN_ID, VALID, UNKNOWN_ID]
    assert validate_grade_citations(grade, SourceRegistry())["grounding"] == UNGROUNDED

