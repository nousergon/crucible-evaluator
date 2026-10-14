"""Tests for the PROPOSAL half of the item-validity contract
(alpha-engine-config-I11990, slice 2 second part).

Slice 2 part 1 (crucible-evaluator-PR345) stopped the weekly loop pass from
grading a closed or ruled item on its cited metrics' color. It left the other
three readers of the old notion of "done" untouched, and the 2026-10-09 plan
shows the result: action 5, ``validate-21d-return-column-stall`` — "Land
residual #2972 ... the single unblock for five starved 21d metrics" — carried
at P0 for the twelfth week against a disproved premise and a standing sunset
ruling. These tests pin the three remaining readers:

* the carry-over prompt states each row's declared closure or ruling;
* the plan artifact carries a deterministic per-item verdict;
* the issue filer does not file an item a declared record holds, and caps a
  re-confirm only on DECLARED identity, never on one shared metric (I11984).

Rows are shaped like the live ledger (``director/carryover_ledger.json``,
2026-10-10). No live GitHub / AWS.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from director import agent as A
from director import handler as H
from director import issue_filer as IF
from director import item_validity_gate as G
from director.schema import ActionItem, DirectorWeeklyActionPlan

_AS_OF = date(2026, 10, 9)

#: The 2026-10-09 card's shape as far as the records can read it: component
#: statuses only. It carries none of the three measures the shipped records'
#: reopen conditions are over (a CloudWatch lag gauge, an SF-cycle signature
#: count, the I9760 flag), which is the production state today.
_CARD = {
    "_provenance": {"run_date": "2026-10-09"},
    "tiles": {
        "predictor": {"components": [{"name": "momentum_l1_ic", "status": "RED", "value": -0.024, "n": 16}]},
        "substrate": {"components": [{"name": "sf_success_rate_4w", "status": "RED"}]},
    },
}


def _item(slug, *, priority="P1", evidence=("predictor tile: momentum_l1_ic",), targets=(),
          title=None, confidence=80) -> ActionItem:
    return ActionItem(
        id=slug, title=title or f"Do {slug}", rationale="grounded", evidence=list(evidence),
        proposed_owner="data", priority=priority, horizon="this_week",
        suggested_change_type="investigation", confidence=confidence, targets=list(targets),
    )


def _ledger_row(slug, issue, **extra) -> dict:
    return {"id": slug, "title": f"row {slug}", "issue_number": issue, "priority": "P0",
            "status": "carried_over", "proposed_owner": "data",
            "evidence": ["predictor tile: momentum_l1_ic"], "carry_count": 12, **extra}


def _assess(items, ledger_rows=(), *, record_set=None):
    return G.assess_plan_items([i.model_dump() for i in items], list(ledger_rows), _CARD,
                               as_of=_AS_OF, record_set=record_set)


# ── canonical_ref ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("ref", [
    "#2972", "I2972", "2972", "alpha-engine-config-I2972", "alpha-engine-config#2972",
    "nousergon/alpha-engine-config#2972", " alpha-engine-config-i2972 ",
])
def test_every_form_an_issue_ref_arrives_in_is_one_identity(ref):
    assert G.canonical_ref(ref) == "alpha-engine-config-I2972"


@pytest.mark.parametrize("ref,expected", [
    ("crucible-evaluator-PR235", "crucible-evaluator-PR235"),  # another repo: verbatim
    ("land residual #2972", "land residual #2972"),             # prose is never parsed
    ("", None), ("   ", None), (None, None), (2972, None),
])
def test_a_ref_that_is_not_an_issue_number_is_never_invented_into_one(ref, expected):
    assert G.canonical_ref(ref) == expected


# ── assess_plan_items: the per-item verdict on the plan ────────────────────


def test_the_carried_2978_row_is_not_valid_to_file_however_its_title_now_reads():
    """Action 5 on 2026-10-09: the item is the work of the ledger row's own
    tracked issue, I2978, whose 2026-09-08 sunset ruling reopens only on the
    I9760 flag. The card does not carry that flag, so the item is HELD — not
    graded green, and not filed."""
    item = _item("validate-21d-return-column-stall", priority="P0",
                 title="Land residual #2972 (21d return-column fix)")
    block = _assess([item], [_ledger_row("validate-21d-return-column-stall", 2978)])
    (v,) = block["items"]
    assert v["valid_to_file"] is False
    assert v["decision"] == "hold_unevaluable"
    assert v["matched"] == ["alpha-engine-config-I2978"]
    assert v["targets"] == ["alpha-engine-config-I2978"]
    assert "I9760" in v["reason"]
    assert block["n_not_valid"] == 1 and block["records"] == "loaded"


@pytest.mark.parametrize("slug,issue", [
    ("halt-or-derisk-live-deployment", 964),
    ("validate-portfolio-report-deployed-strategy", 1084),
])
def test_the_other_carried_sunset_rows_are_held_too(slug, issue):
    (v,) = _assess([_item(slug, priority="P0")], [_ledger_row(slug, issue)])["items"]
    assert (v["valid_to_file"], v["matched"]) == (False, [f"alpha-engine-config-I{issue}"])


def test_a_new_slug_declaring_the_disproved_premise_is_held_by_its_declaration():
    """A changed slug is not a changed premise: the item names I2972, and
    I2972's reopen condition (the 21d lag gauge) has no observation on the card."""
    (v,) = _assess([_item("land-21d-writer-fix", targets=["#2972"])])["items"]
    assert v["valid_to_file"] is False
    assert "alpha-engine-config-I2972" in v["matched"]


def test_a_new_item_sharing_only_a_metric_is_new_work_at_its_own_urgency():
    """I11984's lesson, at the proposal gate: citing ``momentum_l1_ic`` or
    ``sf_success_rate_4w`` is not identity with I2972 or I7644."""
    item = _item("cadence-detector-run-scope", priority="P1",
                 evidence=["substrate tile: sf_success_rate_4w", "predictor tile: momentum_l1_ic"])
    (v,) = _assess([item])["items"]
    assert (v["valid_to_file"], v["basis"], v["matched"], v["targets"]) == (True, G.BASIS_NO_RECORD, [], [])


def test_a_ledger_row_with_no_record_carries_forward_as_before():
    (v,) = _assess([_item("substrate-pipeline-reliability", priority="P0")],
                   [_ledger_row("substrate-pipeline-reliability", 1059)])["items"]
    assert (v["valid_to_file"], v["basis"]) == (True, G.BASIS_NO_RECORD)


def test_targets_stored_on_the_carried_row_are_identity_too():
    row = _ledger_row("re-run-21d-audit", 12001, targets=["alpha-engine-config-I11093"])
    (v,) = _assess([_item("re-run-21d-audit")], [row])["items"]
    assert v["valid_to_file"] is False
    assert "alpha-engine-config-I11093" in v["matched"]


def test_absent_records_withhold_nothing():
    rs = G.RecordSet(status="absent", source="x")
    block = _assess([_item("land-21d-writer-fix", targets=["#2972"])], record_set=rs)
    assert block["items"][0]["valid_to_file"] is True
    assert block["items"][0]["basis"] == G.BASIS_RECORDS_ABSENT


def test_unreadable_records_hold_only_items_with_identity():
    rs = G.RecordSet(status="error", error="ValueError: bad", source="x")
    block = _assess([_item("land-21d-writer-fix", targets=["#2972"]), _item("brand-new")], record_set=rs)
    with_identity, without = block["items"]
    assert (with_identity["valid_to_file"], with_identity["basis"]) == (False, G.BASIS_RECORDS_UNREADABLE)
    assert without["valid_to_file"] is True
    assert block["records_error"] == "ValueError: bad"


def test_not_valid_to_file_reads_only_explicit_false():
    block = _assess([_item("land-21d-writer-fix", targets=["#2972"]), _item("brand-new")])
    assert set(G.not_valid_to_file(block)) == {"land-21d-writer-fix"}
    assert G.not_valid_to_file(None) == {}
    assert G.not_valid_to_file({"items": "garbage"}) == {}


def test_the_block_is_strict_json():
    block = _assess([_item("land-21d-writer-fix", targets=["#2972"])],
                    [_ledger_row("validate-21d-return-column-stall", 2978)])
    assert json.loads(json.dumps(block, allow_nan=False)) == block


# ── the schema: additive, so old plans and ledger rows still parse ─────────


def test_targets_is_additive_with_an_empty_default():
    old = {"id": "x", "title": "t", "rationale": "r", "evidence": [], "proposed_owner": "data",
           "priority": "P1", "horizon": "this_week", "suggested_change_type": "investigation",
           "confidence": 50}
    assert ActionItem.model_validate(old).targets == []


# ── issue filer ────────────────────────────────────────────────────────────


class _FakeGitHub:
    def __init__(self, closed=()):
        self.posted: list[dict] = []
        self._closed = list(closed)
        self._n = 900

    def __call__(self, method, url, token, body=None):
        if method == "GET" and "state=all" in url:
            return 200, []
        if method == "GET" and "state=closed" in url:
            return (200, list(self._closed)) if "page=1" in url else (200, [])
        if method == "POST" and url.endswith("/issues"):
            self.posted.append(body)
            self._n += 1
            return 201, {"number": self._n, "html_url": f"https://x/{self._n}"}
        raise AssertionError(f"unexpected call {method} {url}")


def _closed(number, evidence):
    _, body = IF.render_issue(_item("old-slug", evidence=evidence), "2026-09-01")
    return {"number": number, "title": f"[director] prior (id=old-{number})", "body": body,
            "closed_at": "2026-09-20T00:00:00Z"}


def _plan(*items) -> DirectorWeeklyActionPlan:
    return DirectorWeeklyActionPlan(run_date="2026-10-09", system_summary="s", top_risks=["r"],
                                    action_items=list(items))


def test_a_new_defect_sharing_one_metric_with_a_closed_investigation_keeps_its_priority():
    """The I11984 regression: a new cadence-detector defect cited
    ``sf_success_rate_4w``, which an unrelated closed coverage investigation
    also cited, and was filed as a P2 re-confirm at confidence 40. It is now
    filed at its own P1, with the prior closure linked."""
    gh = _FakeGitHub(closed=[_closed(11612, ["substrate tile: sf_success_rate_4w"])])
    item = _item("cadence-detector-run-scope", priority="P1",
                 evidence=["substrate tile: sf_success_rate_4w"], confidence=70)
    res = IF.file_director_issues(_plan(item), "2026-10-09", token="t", gh_request=gh)
    (posted,) = gh.posted
    assert posted["labels"] == ["area:director-proposals", "P1"]
    assert "Re-confirm" not in posted["title"]
    assert "Confidence:** 70/100" in posted["body"]
    assert "## Shares cited evidence with closed proposal(s)\n- #11612" in posted["body"]
    assert res["issues"][0]["reconfirm_of"] == []
    assert res["issues"][0]["shares_evidence_with"] == [11612]


def test_the_shared_evidence_note_leaves_the_evidence_section_and_marker_intact():
    gh = _FakeGitHub(closed=[_closed(11612, ["substrate tile: sf_success_rate_4w"])])
    item = _item("cadence-detector-run-scope", evidence=["substrate tile: sf_success_rate_4w", "b"])
    IF.file_director_issues(_plan(item), "2026-10-09", token="t", gh_request=gh)
    body = gh.posted[0]["body"]
    assert IF.parse_evidence_from_body(body) == {"substrate tile: sf_success_rate_4w", "b"}
    assert "id=cadence-detector-run-scope · idempotency marker" in body


@pytest.mark.parametrize("target", ["#1168", "alpha-engine-config-I1168", "I1168"])
def test_a_declared_re_do_of_a_closed_proposal_is_still_a_capped_re_confirm(target):
    """config#2307 survives on DECLARED identity: an item that names the prior
    closure it re-does files as the re-confirm variant, capped at P2."""
    gh = _FakeGitHub(closed=[_closed(1168, ["metrics.json::cio_selection_skill"]),
                             _closed(1200, ["metrics.json::unrelated"])])
    item = _item("validate-reliability-low-research-metrics", priority="P1",
                 evidence=["metrics.json::other"], targets=[target])
    res = IF.file_director_issues(_plan(item), "2026-10-09", token="t", gh_request=gh)
    (posted,) = gh.posted
    assert posted["labels"] == ["area:director-proposals", "P2"]
    assert "Re-confirm" in posted["title"] and "#1168" in posted["title"]
    assert "#1200" not in posted["title"]
    assert res["issues"][0]["reconfirm_of"] == [1168]


def test_find_declared_reconfirm_needs_a_declared_target():
    prior = [_closed(1168, ["m"])]
    assert IF.find_declared_reconfirm(_item("x", evidence=["m"]), prior) == []
    assert IF.find_declared_reconfirm(_item("x", targets=["#1168"]), prior) == prior
    assert IF.find_declared_reconfirm(_item("x", targets=["#9999"]), prior) == []


def test_an_item_a_declared_record_holds_is_not_filed_and_is_named():
    gh = _FakeGitHub()
    held = _item("land-21d-writer-fix", priority="P0", targets=["#2972"])
    fresh = _item("brand-new-defect", priority="P1", evidence=["x"])
    block = _assess([held, fresh])
    res = IF.file_director_issues(_plan(held, fresh), "2026-10-09", token="t", gh_request=gh,
                                  item_validity=block)
    assert [p["title"] for p in gh.posted] == ["[director] Do brand-new-defect (id=brand-new-defect)"]
    assert res["n_filed"] == 1 and res["n_withheld"] == 1
    (w,) = res["withheld"]
    assert w["slug"] == "land-21d-writer-fix"
    assert w["decision"] == "hold_unevaluable"
    assert "alpha-engine-config-I2972" in w["matched"]


def test_every_item_withheld_files_nothing_and_fetches_nothing_more():
    calls: list[str] = []

    def gh(method, url, token, body=None):
        calls.append(f"{method} {url}")
        return (200, []) if method == "GET" else (500, {})

    held = _item("land-21d-writer-fix", targets=["#2972"])
    res = IF.file_director_issues(_plan(held), "2026-10-09", token="t", gh_request=gh,
                                  item_validity=_assess([held]))
    assert res["status"] == "ok" and res["n_filed"] == 0 and res["n_withheld"] == 1
    assert not any("state=closed" in c or c.startswith("POST") for c in calls), calls


def test_no_validity_block_withholds_nothing():
    gh = _FakeGitHub()
    res = IF.file_director_issues(_plan(_item("land-21d-writer-fix", targets=["#2972"])),
                                  "2026-10-09", token="t", gh_request=gh)
    assert res["n_filed"] == 1 and res["withheld"] == []


# ── handler wiring ─────────────────────────────────────────────────────────


def test_the_handler_assessment_reads_the_ledger_the_plan_was_built_from():
    plan = _plan(_item("validate-21d-return-column-stall", priority="P0"), _item("brand-new"))
    ledger = {"items": [_ledger_row("validate-21d-return-column-stall", 2978)]}
    block = H._assess_plan_items_best_effort(plan, ledger, _CARD, "2026-10-09")
    assert block["as_of"] == "2026-10-09"
    assert [i["valid_to_file"] for i in block["items"]] == [False, True]


def test_a_failed_plan_assessment_is_named_and_withholds_nothing(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(H, "assess_plan_items", _boom)
    block = H._assess_plan_items_best_effort(_plan(_item("x")), {}, _CARD, "2026-10-09")
    assert block["records"] == "assessment_error" and "kaboom" in block["error"]
    assert G.not_valid_to_file(block) == {}


def test_the_filer_is_handed_the_block_and_the_withheld_items_reach_the_sf_output(monkeypatch):
    seen: dict = {}

    def _file(plan, run_date, *, token, item_validity=None):
        seen["item_validity"] = item_validity
        return {"status": "ok", "n_filed": 0, "issues": [],
                "withheld": [{"slug": "land-21d-writer-fix", "reason": "held"}]}

    monkeypatch.setattr(H, "file_director_issues", _file)
    monkeypatch.setattr(H, "_issue_filing_enabled", lambda: True)
    block = {"items": []}
    attested = {"verdict": "PASS", "present": True, "as_of": {}}
    out = H._file_issues_best_effort(_plan(_item("x")), "2026-10-09", "tok",
                                     verdict_block=attested, item_validity=block)
    assert seen["item_validity"] is block
    assert out["director_issues_withheld"] == [{"slug": "land-21d-writer-fix", "reason": "held"}]


# ── the carry-over prompt states the declared record ───────────────────────


def _ctx(rows, card=_CARD):
    return A._carryover_context({"items": rows}, card)


def test_a_carried_ruled_row_is_annotated_with_its_declared_record():
    rows = [_ledger_row("validate-21d-return-column-stall", 2978),
            _ledger_row("substrate-pipeline-reliability", 1059)]
    text = _ctx(rows)
    block_2978 = text.split("[validate-21d-return-column-stall]")[1].split("  - [")[0]
    assert "DECLARED CLOSED/RULED (alpha-engine-config-I2978)" in block_2978
    assert "Do NOT restate it as an action item" in block_2978
    block_1059 = text.split("[substrate-pipeline-reliability]")[1].split("  - [")[0]
    assert "DECLARED" not in block_1059


def test_the_annotation_drops_and_reorders_nothing():
    rows = [_ledger_row("validate-21d-return-column-stall", 2978),
            _ledger_row("substrate-pipeline-reliability", 1059)]
    with_records = _ctx(rows)
    shown_ids = [line.split("]")[0].split("[")[1] for line in with_records.splitlines()
                 if line.startswith("  - [")]
    assert shown_ids == [r["id"] for r in A.select_carryover_rows(rows)[0]]
    assert A._CARRYOVER_COUNT_MARKER + "2" in with_records


def test_a_met_reopen_condition_says_so_instead(monkeypatch):
    card = json.loads(json.dumps(_CARD))
    card["tiles"]["governance"] = {"components": [
        {"name": "alpha-engine-config-I9760:phase4_cancelled_or_v1_retained", "value": 1}]}
    text = _ctx([_ledger_row("validate-21d-return-column-stall", 2978)], card)
    assert "Its declared reopen condition is met" in text
    assert "DECLARED CLOSED/RULED" not in text


def test_unreadable_records_are_said_once_and_annotate_nothing(monkeypatch):
    monkeypatch.setattr(A, "load_records",
                        lambda: G.RecordSet(status="error", error="ValueError: bad", source="x"))
    text = _ctx([_ledger_row("validate-21d-return-column-stall", 2978)])
    assert text.count("could not be read (ValueError: bad)") == 1
    assert "DECLARED CLOSED/RULED" not in text


def test_an_annotation_failure_never_sinks_the_prompt(monkeypatch):
    def _boom():
        raise RuntimeError("kaboom")

    monkeypatch.setattr(A, "load_records", _boom)
    text = _ctx([_ledger_row("validate-21d-return-column-stall", 2978)])
    assert "[validate-21d-return-column-stall]" in text
    assert "item-validity check failed this week (RuntimeError)" in text


def test_absent_records_leave_the_prompt_as_it_was(monkeypatch):
    rows = [_ledger_row("validate-21d-return-column-stall", 2978)]
    monkeypatch.setattr(A, "load_records", lambda: G.RecordSet(status="absent", source="x"))
    text = _ctx(rows)
    assert "DECLARED" not in text and "item-validity" not in text


def test_the_plan_instruction_asks_for_targets():
    messages = A.build_messages(_CARD, carryover={"items": []})
    human = messages[-1][1]
    assert "`targets`" in human and "CLOSED or RULED" in human


def test_the_shipped_records_file_is_where_the_gate_reads():
    assert Path(G.RECORDS_PATH).name == "item_validity_records.json"
    assert G.load_records().status == "loaded"
