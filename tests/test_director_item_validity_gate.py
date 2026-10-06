"""Tests for director/item_validity_gate.py — the weekly Director reading the
item-validity contract before it grades a closed item
(alpha-engine-config-I11990, slice 2 first part).

The rows below are shaped like the LIVE ledger (2026-10-02): the
``validate-21d-return-column-stall`` row tracks issue 2978, the
``halt-or-derisk-live-deployment`` row tracks 964, and so on. The observations
are the known-answer fixture's ``current`` block from crucible-evaluator-PR338,
so every verdict here is the contract's own answer, read through the wiring.
No live GitHub / AWS.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from director import handler as H
from director import item_validity as IV
from director import item_validity_gate as G
from director import loop_verification as LV
from director import verdict as V
from director.emailer import build_director_digest

_FIXTURE = Path(__file__).parent / "fixtures" / "director_item_validity_known_answers.json"
_AS_OF = date(2026, 10, 4)


@pytest.fixture(scope="module")
def ka() -> dict:
    return json.loads(_FIXTURE.read_text())


def _obs(observations) -> dict:
    return {o.measure: o for o in (IV.Observation.model_validate(x) for x in observations)}


def _current(ka, override=()):
    merged = {o["measure"]: o for o in ka["current"]}
    for o in override:
        merged[o["measure"]] = o
    return _obs(merged.values())


def _shipped():
    rs = G.load_records()
    assert rs.status == "loaded", rs.error
    return rs


def _row(id_, issue=None, priority="P0", **extra):
    return {"id": id_, "issue_number": issue, "priority": priority, "status": "carried_over",
            "evidence": ["momentum_l1_ic"], **extra}


# ── the shipped records file ───────────────────────────────────────────────


def test_shipped_records_are_the_pr338_known_answer_records_unchanged(ka):
    """The weekly run reads exactly the declarations the contract's
    known-answer tests prove. A different reopen threshold is Brian's ruling on
    alpha-engine-config-I11990 and lands in BOTH files with that ruling — never
    in the shipped file alone."""
    rs = _shipped()
    shipped = {r.item: r for r in rs.records}
    fixture = {r.item: r for r in (IV.parse_record(x) for x in ka["records"])}
    assert shipped == fixture


def test_every_shipped_completion_carries_its_required_evidence():
    for rec in _shipped().records:
        if rec.completion is not None:
            assert IV.missing_evidence(rec.completion) == (), rec.item


# ── per-row verdicts on live-ledger-shaped rows ────────────────────────────


def test_a_ruled_sunset_row_is_not_valid_to_grade_while_its_ruling_stands(ka):
    """Ledger row ``validate-21d-return-column-stall`` tracks I2978, closed by
    the 2026-09-08 sunset ruling. Its portfolio/predictor metrics are RED; that
    is not the ruling's invalid-when, so the row is not graded on color."""
    v = G.assess_row(_row("validate-21d-return-column-stall", 2978), _shipped(), _current(ka), as_of=_AS_OF)
    assert v["valid_to_grade"] is False
    assert v["basis"] == G.BASIS_RECORD
    assert v["decision"] == "refuse"
    assert v["matched"] == ["alpha-engine-config-I2978"]
    assert "ruling's reopen condition" in v["reason"] and "not met" in v["reason"]


@pytest.mark.parametrize("slug,issue", [
    ("halt-or-derisk-live-deployment", 964),
    ("validate-portfolio-report-deployed-strategy", 1084),
])
def test_the_other_sunset_rows_are_refused_too(ka, slug, issue):
    v = G.assess_row(_row(slug, issue), _shipped(), _current(ka), as_of=_AS_OF)
    assert (v["valid_to_grade"], v["decision"]) == (False, "refuse")
    assert v["matched"] == [f"alpha-engine-config-I{issue}"]


def test_the_ruling_becomes_gradable_only_when_its_own_condition_is_met(ka):
    override = [{"measure": "alpha-engine-config-I9760:phase4_cancelled_or_v1_retained",
                 "observed_at": "2026-10-09", "value": 1, "source": "synthetic"}]
    v = G.assess_row(_row("halt-or-derisk-live-deployment", 964), _shipped(),
                     _current(ka, override), as_of=_AS_OF)
    assert (v["valid_to_grade"], v["decision"]) == (True, "allow")
    assert v["reason"].startswith("declared condition met")


def test_an_explicit_target_on_a_disproved_premise_is_refused_and_flips_on_recurrence(ka):
    row = _row("land-residual-2972", 11999, targets=["alpha-engine-config-I2972"])
    assert G.assess_row(row, _shipped(), _current(ka), as_of=_AS_OF)["valid_to_grade"] is False
    recur = [{"measure": "universe_returns_horizon_grading_lag_trading_days{HorizonDays=21}",
              "observed_at": "2026-10-09", "value": 3, "source": "synthetic"}]
    assert G.assess_row(row, _shipped(), _current(ka, recur), as_of=_AS_OF)["valid_to_grade"] is True


def test_a_matching_work_scope_is_identity_without_any_issue_number(ka):
    row = _row("validate-21d-label-surface-v2", None, work_scope={
        "anchor": "nousergon-data:universe_returns 21d-horizon label surface",
        "acceptance": "21d-grades-not-corrupted-by-label-stall"})
    v = G.assess_row(row, _shipped(), _current(ka), as_of=_AS_OF)
    assert v["valid_to_grade"] is False
    assert set(v["matched"]) == {"alpha-engine-config-I11093", "alpha-engine-config-I2978"}


def test_a_row_no_record_covers_is_graded_on_color_as_before(ka):
    """I11984's lesson: sharing a metric with a closed item is not identity."""
    row = _row("substrate-pipeline-reliability", 1059, evidence=["sf_success_rate_4w"])
    v = G.assess_row(row, _shipped(), _current(ka), as_of=_AS_OF)
    assert (v["valid_to_grade"], v["basis"], v["matched"]) == (True, G.BASIS_NO_RECORD, [])


def test_prose_issue_references_are_never_identity(ka):
    row = _row("validate-21d-return-column-stall-copy", 11998,
               title="Land residual #2972 (21d return-column fix)")
    v = G.assess_row(row, _shipped(), _current(ka), as_of=_AS_OF)
    assert v["basis"] == G.BASIS_NO_RECORD


def test_a_condition_the_card_cannot_observe_is_held_never_graded():
    """In production the observations are the card's components only. The
    sunset ruling's condition is over an I9760 flag the card does not carry, so
    the row is held unevaluable — named, not graded, not green."""
    card = {"tiles": {"portfolio_outcome": {"components": [
        {"name": "alpha_vs_spy", "status": "RED", "value": -0.12, "n": 144}]}}}
    rows = [_row("halt-or-derisk-live-deployment", 964)]
    summary = G.assess_ledger(rows, card, as_of=_AS_OF, record_set=_shipped())
    v = rows[0][G.VERDICT_KEY]
    assert (v["valid_to_grade"], v["decision"]) == (False, "hold_unevaluable")
    assert "alpha-engine-config-I9760:phase4_cancelled_or_v1_retained" in v["reason"]
    assert summary["director_item_validity_not_valid"] == 1


# ── absent vs unreadable records ───────────────────────────────────────────


def test_absent_records_grade_everything_as_before_and_say_so(tmp_path):
    rs = G.load_records(tmp_path / "missing.json")
    assert rs.status == "absent"
    v = G.assess_row(_row("halt-or-derisk-live-deployment", 964), rs, {}, as_of=_AS_OF)
    assert (v["valid_to_grade"], v["basis"]) == (True, G.BASIS_RECORDS_ABSENT)


@pytest.mark.parametrize("body", [
    "{not json",
    json.dumps({"records": "nope"}),
    json.dumps({"records": [{"item": "x", "schema_version": 99,
                             "scope": {"anchor": "a", "acceptance": "b"}}]}),
    json.dumps({"records": [{"item": "x", "scope": {"anchor": "a", "acceptance": "b"}, "bogus": 1}]}),
    json.dumps({"records": [{"item": "x", "scope": {"anchor": "a", "acceptance": "b"}}] * 2}),
])
def test_unreadable_records_fail_closed(tmp_path, body):
    p = tmp_path / "records.json"
    p.write_text(body)
    rs = G.load_records(p)
    assert rs.status == "error" and rs.error
    rows = [_row("a", 1), _row("b", None)]
    summary = G.assess_ledger(rows, {}, as_of=_AS_OF, record_set=rs)
    assert all(r[G.VERDICT_KEY]["valid_to_grade"] is False for r in rows)
    assert all(r[G.VERDICT_KEY]["basis"] == G.BASIS_RECORDS_UNREADABLE for r in rows)
    assert summary["director_item_validity"] == "error"
    assert summary["director_item_validity_error"] == rs.error


# ── verify_and_correct obeys the stamp ─────────────────────────────────────


class _GH:
    """Closed issues only; records every call so a mutation is visible."""

    def __init__(self, issues):
        self.issues = issues
        self.calls: list[tuple[str, str]] = []

    def __call__(self, method, url, token, body=None):
        self.calls.append((method, url))
        if method == "GET" and url.endswith("/comments"):
            return 200, []
        if method == "GET":
            return 200, self.issues[int(url.rsplit("/", 1)[-1])]
        return 201, {"number": 1}


_RED_CARD = {"tiles": {"predictor": {"components": [{"name": "momentum_l1_ic", "status": "RED"}]}}}


def test_a_closed_row_not_valid_to_grade_is_neither_graded_nor_mutated():
    row = _row("validate-21d-return-column-stall", 2978)
    row[G.VERDICT_KEY] = {"valid_to_grade": False, "reason": "refused"}
    gh = _GH({2978: {"state": "closed", "state_reason": "not_planned", "labels": []}})
    out = LV.verify_and_correct([row], _RED_CARD, repo="o/r", token="t", gh_request=gh)
    assert out["closed_not_valid_to_grade"] == 1
    assert out["not_valid_to_grade_issues"] == [2978]
    assert out["closed_ruled_unrecovered"] == 0 and out["corrections"] == 0
    assert all(m == "GET" for m, _ in gh.calls), gh.calls
    assert "ruled_unrecovered_notified" not in row


def test_a_closed_row_valid_to_grade_is_graded_exactly_as_before():
    row = _row("some-repair", 4321)
    row[G.VERDICT_KEY] = {"valid_to_grade": True, "reason": "no record"}
    gh = _GH({4321: {"state": "closed", "labels": []}})
    out = LV.verify_and_correct([row], _RED_CARD, repo="o/r", token="t", gh_request=gh)
    assert out["closed_unrecovered"] == 1 and out["reopened_issues"] == [4321]
    assert out["closed_not_valid_to_grade"] == 0


# ── handler wiring ─────────────────────────────────────────────────────────


def _pass_card():
    return {"tiles_overall_status": "GREEN", "tiles": {}, "attestation": {
        "schema": "report_card_attestation-1.0.0", "run_date": "2026-10-03", "verdict": "PASS",
        "as_of": {"backtester": "2026-10-03T09:41:02Z", "evaluator_stage": "2026-10-03T10:02:55Z"},
        "evaluator": {"verdict": "PASS", "n_checks": 6}, "backtester": {"verdict": "PASS", "n_checks": 6},
        "evaluator_stage": {"verdict": "PASS", "n_checks": 4}, "promotion_withheld": False,
        "reason": "All three halves attested."}}


def test_rows_are_stamped_before_verify_and_after_backfill(monkeypatch):
    seen: list = []
    monkeypatch.setattr(H, "backfill_issue_numbers",
                        lambda items, repo=None, token=None: (items[0].update(issue_number=964), 1)[1])
    monkeypatch.setattr(H, "verify_and_correct",
                        lambda items, card, repo=None, token=None: (seen.append(dict(items[0])), {})[1])
    card = _pass_card()
    ledger = {"items": [{"id": "halt-or-derisk-live-deployment", "priority": "P0"}]}
    out = H._verify_loop_best_effort(ledger, card, "tok", verdict_block=V.read_card_verdict(card),
                                     run_date="2026-10-03")
    assert out["director_loop"] == "ok"
    stamp = seen[0][G.VERDICT_KEY]
    # The backfilled issue number made the row I964's work, so it matched the ruling.
    assert stamp["matched"] == ["alpha-engine-config-I964"]
    assert stamp["valid_to_grade"] is False and stamp["as_of"] == "2026-10-03"
    assert out["director_item_validity"] == "loaded"
    assert out["director_item_validity_items"][0]["id"] == "halt-or-derisk-live-deployment"


def test_the_statement_is_made_even_when_mutations_are_withheld(monkeypatch):
    monkeypatch.setattr(H, "backfill_issue_numbers", lambda items, repo=None, token=None: 0)
    ledger = {"items": [_row("validate-21d-return-column-stall", 2978)]}
    out = H._verify_loop_best_effort(ledger, {"tiles": {}}, "tok", verdict_block={}, run_date="2026-10-03")
    assert out["director_loop"] == "mutations_withheld"
    assert out["director_item_validity_not_valid"] == 1
    assert ledger["items"][0][G.VERDICT_KEY]["valid_to_grade"] is False


def test_unreadable_records_make_the_loop_partial(monkeypatch):
    monkeypatch.setattr(H, "backfill_issue_numbers", lambda items, repo=None, token=None: 0)
    monkeypatch.setattr(H, "verify_and_correct", lambda items, card, repo=None, token=None: {})
    monkeypatch.setattr(H, "assess_ledger", lambda items, card, as_of: G.assess_ledger(
        items, card, as_of=as_of, record_set=G.RecordSet(status="error", error="ValueError: bad")))
    card = _pass_card()
    out = H._verify_loop_best_effort({"items": [_row("a", 1)]}, card, "tok",
                                     verdict_block=V.read_card_verdict(card), run_date="2026-10-03")
    assert out["director_loop"] == "partial"


def test_a_failed_assessment_overwrites_last_weeks_stamp_fail_closed(monkeypatch):
    def _boom(items, card, as_of):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(H, "assess_ledger", _boom)
    rows = [_row("a", 1)]
    rows[0][G.VERDICT_KEY] = {"valid_to_grade": True, "as_of": "2026-09-26"}
    out = H._assess_item_validity_best_effort(rows, {}, "2026-10-03")
    assert out["director_item_validity"] == "error"
    assert rows[0][G.VERDICT_KEY]["valid_to_grade"] is False
    assert rows[0][G.VERDICT_KEY]["as_of"] == "2026-10-03"


# ── the digest states it per item ──────────────────────────────────────────


def _plan():
    return {"run_date": "2026-10-03", "system_summary": "s", "top_risks": [], "action_items": []}


def test_digest_names_each_record_decided_item_and_why(ka):
    rows = [_row("validate-21d-return-column-stall", 2978), _row("substrate-pipeline-reliability", 1059)]
    summary = G.assess_ledger(rows, {}, as_of=_AS_OF, record_set=_shipped())
    _, plain, html = build_director_digest(_plan(), "2026-10-03",
                                           loop_summary={"director_loop": "ok", **summary})
    assert "1 of 2 ledger item(s) NOT valid to grade" in plain
    assert "validate-21d-return-column-stall (#2978): NOT valid to grade — held, unevaluable" in plain
    assert "substrate-pipeline-reliability" not in plain  # counted, not listed
    assert "validate-21d-return-column-stall (#2978): NOT valid to grade" in html


@pytest.mark.parametrize("status,needle", [
    ("absent", "records file ABSENT"),
    ("error", "records UNREADABLE (ValueError: bad)"),
])
def test_digest_says_when_the_records_could_not_be_used(status, needle):
    summary = {"director_loop": "ok", "director_item_validity": status,
               "director_item_validity_error": "ValueError: bad"}
    _, plain, html = build_director_digest(_plan(), "2026-10-03", loop_summary=summary)
    assert needle in plain and needle in html


def test_digest_unchanged_when_no_assessment_ran():
    _, plain, _ = build_director_digest(_plan(), "2026-10-03",
                                        loop_summary={"director_loop": "skipped",
                                                      "director_loop_reason": "no GH token"})
    assert "Item validity" not in plain
