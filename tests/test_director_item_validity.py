"""Known-answer tests for director/item_validity.py — the typed item
completion / validity contract (alpha-engine-config-I11990, slice 1).

The fixture file restates real authoritative closures and rulings and the
items the 2026-10-02 Director plan actually produced. These are the acceptance
cases I11990 names:

  * I2972 stays disproved while its labels are current;
  * I7644 is not re-proposed as unlanded;
  * ruled sunset closes stay closed unless their own invalid-when is met;
  * a new defect sharing evidence with a closed item keeps its urgency;
  * missing closure evidence is unevaluable, never green;
  * fixed-but-still-negative work does not reopen on color.

A changed expected answer is a changed contract — edit the fixture only with
the issue that authorises the change.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from director import item_validity as IV

_FIXTURE = Path(__file__).parent / "fixtures" / "director_item_validity_known_answers.json"


@pytest.fixture(scope="module")
def ka() -> dict:
    return json.loads(_FIXTURE.read_text())


def _records(ka):
    return [IV.parse_record(r) for r in ka["records"]]


def _current(observations):
    return {o.measure: o for o in (IV.Observation.model_validate(x) for x in observations)}


def _as_of(ka) -> date:
    return date.fromisoformat(ka["as_of"])


# ── fixture integrity ──────────────────────────────────────────────────────


def test_every_fixture_record_carries_complete_closure_evidence(ka):
    """A known-answer record that is itself unevaluable would make every
    'stays closed' answer below vacuous."""
    for rec in _records(ka):
        if rec.completion is not None:
            assert IV.missing_evidence(rec.completion) == (), rec.item


def test_every_fixture_proposal_parses_and_expected_answers_are_exhaustive(ka):
    ids = {p["proposal"]["id"] for p in ka["proposals"]}
    assert len(ids) == len(ka["proposals"])
    for cond in ka["changed_conditions"]:
        assert set(cond["expected_decisions"]) <= ids, cond["name"]
    assert set(ka["expected_reconciliation"]) == {r["item"] for r in ka["records"]}


# ── reconciliation today ───────────────────────────────────────────────────


def test_reconciliation_known_answers(ka):
    current = _current(ka["current"])
    got = {r.item: IV.reconcile_record(r, current, as_of=_as_of(ka)).disposition.value
           for r in _records(ka)}
    assert got == ka["expected_reconciliation"]


def test_i2972_stays_disproved_while_its_outcome_metric_is_red(ka):
    current = _current(ka["current"])
    assert current["momentum_l1_ic"].status == "RED"  # the color that used to re-track it
    rec = next(r for r in _records(ka) if r.item == "alpha-engine-config-I2972")
    rc = IV.reconcile_record(rec, current, as_of=_as_of(ka))
    assert rc.disposition is IV.Disposition.STAYS_CLOSED
    assert "premise_disproved" in rc.reason


# ── proposal gate ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("idx", range(7))
def test_proposal_known_answers(ka, idx):
    case = ka["proposals"][idx]
    proposal = IV.Proposal.model_validate(case["proposal"])
    got = IV.gate_proposal(proposal, _records(ka), _current(ka["current"]), as_of=_as_of(ka))
    exp = case["expected"]
    assert got.decision.value == exp["decision"], (proposal.id, got.reason)
    assert got.priority == exp["priority"]
    assert sorted(got.matched) == sorted(exp["matched"])


def test_fixture_proposal_count_matches_parametrization(ka):
    assert len(ka["proposals"]) == 7


def test_new_defect_sharing_evidence_with_a_closed_item_keeps_its_urgency(ka):
    """I11984: the precondition is real shared evidence with closed I7644 —
    the overlap that capped it P1 → P2 under metric-overlap identity."""
    case = next(p for p in ka["proposals"] if p["proposal"]["id"] == "run-scope-cadence-branch-attribution")
    proposal = IV.Proposal.model_validate(case["proposal"])
    prior = next(r for r in _records(ka) if r.item == case["shares_evidence_with"])
    shared = set(proposal.evidence) & {o.measure for o in prior.completion.evidence}
    assert shared, "fixture premise: the new defect must share evidence with the closed item"
    assert not IV.same_work(proposal, prior)
    got = IV.gate_proposal(proposal, _records(ka), _current(ka["current"]), as_of=_as_of(ka))
    assert (got.decision, got.priority) == (IV.Decision.ALLOW, "P1")


def test_gate_never_changes_priority(ka):
    for case in ka["proposals"]:
        proposal = IV.Proposal.model_validate(case["proposal"])
        got = IV.gate_proposal(proposal, _records(ka), _current(ka["current"]), as_of=_as_of(ka))
        assert got.priority == proposal.priority


# ── genuine changed conditions stay detectable ─────────────────────────────


@pytest.mark.parametrize("idx", range(3))
def test_changed_conditions_are_detected(ka, idx):
    cond = ka["changed_conditions"][idx]
    current = {**_current(ka["current"]), **_current(cond["override"])}
    as_of = max(o.observed_at for o in current.values())
    recs = _records(ka)
    by_item = {r.item: r for r in recs}
    for item, expected in cond["expected_reconciliation"].items():
        got = IV.reconcile_record(by_item[item], current, as_of=as_of)
        assert got.disposition.value == expected, (cond["name"], item, got.reason)
    proposals = {p["proposal"]["id"]: IV.Proposal.model_validate(p["proposal"]) for p in ka["proposals"]}
    for pid, expected in cond["expected_decisions"].items():
        got = IV.gate_proposal(proposals[pid], recs, current, as_of=as_of)
        assert got.decision.value == expected, (cond["name"], pid, got.reason)


def test_fixture_changed_condition_count_matches_parametrization(ka):
    assert len(ka["changed_conditions"]) == 3


# ── contract rules (synthetic) ─────────────────────────────────────────────

_D = date(2026, 10, 4)
_SCOPE = IV.WorkScope(anchor="crucible-executor:sizing", acceptance="mae-floor")


def _obs(measure, *, value=None, status=None, n=None, on=_D):
    return IV.Observation(measure=measure, observed_at=on, source="test", value=value, status=status, n=n)


def _repair(**over):
    base = dict(
        kind=IV.CompletionKind.REPAIR_VERIFIED, closed_on=date(2026, 9, 1),
        closure_ref="x#1", change_ref="repo-PR1",
        evidence=(_obs("mae_floor_breaches", value=0),),
        reopen_when=IV.Predicate(measure="mae_floor_breaches", op="gt", threshold=0),
    )
    base.update(over)
    return IV.ItemCompletion(**base)


def test_fixed_but_still_negative_does_not_reopen_on_color():
    rec = IV.ItemRecord(item="x-I1", scope=_SCOPE, completion=_repair())
    current = {
        "mae_floor_breaches": _obs("mae_floor_breaches", value=0),
        "alpha_vs_spy": _obs("alpha_vs_spy", value=-0.12, status="RED"),
    }
    assert IV.reconcile_record(rec, current, as_of=_D).disposition is IV.Disposition.STAYS_CLOSED


@pytest.mark.parametrize("drop", ["closure_ref", "change_ref", "evidence", "reopen_when"])
def test_missing_closure_evidence_is_unevaluable_never_green(drop):
    rec = IV.ItemRecord(item="x-I1", scope=_SCOPE, completion=_repair(**{drop: None if drop != "evidence" else ()}))
    rc = IV.reconcile_record(rec, {"mae_floor_breaches": _obs("mae_floor_breaches", value=0)}, as_of=_D)
    assert rc.disposition is IV.Disposition.UNEVALUABLE
    assert rc.missing


def test_record_with_neither_completion_nor_ruling_is_unevaluable():
    rc = IV.reconcile_record(IV.ItemRecord(item="x-I1", scope=_SCOPE), {}, as_of=_D)
    assert rc.disposition is IV.Disposition.UNEVALUABLE


def test_absent_current_observation_is_unevaluable_not_clear():
    rec = IV.ItemRecord(item="x-I1", scope=_SCOPE, completion=_repair())
    assert IV.reconcile_record(rec, {}, as_of=_D).disposition is IV.Disposition.UNEVALUABLE


def test_unevaluable_match_holds_a_proposal_rather_than_allowing_it():
    rec = IV.ItemRecord(item="x-I1", scope=_SCOPE, completion=_repair(change_ref=None))
    got = IV.gate_proposal(IV.Proposal(id="p", priority="P1", scope=_SCOPE), [rec], {}, as_of=_D)
    assert got.decision is IV.Decision.HOLD_UNEVALUABLE
    assert got.priority == "P1"


def test_color_reopen_condition_is_only_admissible_for_outcome_recovered():
    color = IV.Predicate(measure="alpha_vs_spy", op="status_in", statuses=("RED",))
    ev = (_obs("alpha_vs_spy", value=0.01, status="GREEN"),)
    for kind in (IV.CompletionKind.REPAIR_VERIFIED, IV.CompletionKind.PREMISE_DISPROVED,
                 IV.CompletionKind.GUARD_REFUSING):
        c = _repair(kind=kind, evidence=ev, reopen_when=color, premise="p", guard="g")
        assert any("card-color" in m for m in IV.missing_evidence(c)), kind
    ok = _repair(kind=IV.CompletionKind.OUTCOME_RECOVERED, evidence=ev, reopen_when=color)
    assert IV.missing_evidence(ok) == ()
    rec = IV.ItemRecord(item="x-I1", scope=_SCOPE, completion=ok)
    red = {"alpha_vs_spy": _obs("alpha_vs_spy", value=-0.1, status="RED")}
    assert IV.reconcile_record(rec, red, as_of=_D).disposition is IV.Disposition.REOPEN_CONDITION_MET


def test_reopen_condition_must_be_over_an_observed_measure():
    c = _repair(reopen_when=IV.Predicate(measure="alpha_vs_spy", op="lt", threshold=0))
    assert any("observed in evidence" in m for m in IV.missing_evidence(c))


def test_guard_refusing_does_not_reopen_while_guarded_outcome_stays_bad():
    c = IV.ItemCompletion(
        kind=IV.CompletionKind.GUARD_REFUSING, closed_on=date(2026, 10, 2), closure_ref="x#2",
        guard="alpha_floor", evidence=(_obs("alpha_floor.passing_candidates_refused", value=0),),
        reopen_when=IV.Predicate(measure="alpha_floor.passing_candidates_refused", op="gt", threshold=0),
    )
    rec = IV.ItemRecord(item="x-I2", scope=_SCOPE, completion=c)
    current = {
        "alpha_floor.passing_candidates_refused": _obs("alpha_floor.passing_candidates_refused", value=0),
        "alpha_vs_spy": _obs("alpha_vs_spy", value=-0.09, status="RED"),
    }
    assert IV.reconcile_record(rec, current, as_of=_D).disposition is IV.Disposition.STAYS_CLOSED


@pytest.mark.parametrize("n,as_of,expected", [
    (10, date(2026, 10, 4), IV.Disposition.ACCUMULATING),
    (30, date(2026, 10, 4), IV.Disposition.MATURED),
    (10, date(2026, 11, 2), IV.Disposition.STALLED),
])
def test_monitor_accumulating_matures_or_stalls_never_reopens_on_color(n, as_of, expected):
    c = IV.ItemCompletion(
        kind=IV.CompletionKind.MONITOR_ACCUMULATING, closed_on=date(2026, 9, 1), closure_ref="x#3",
        monitored_measure="psr", required_n=30, matures_on=date(2026, 11, 1),
        evidence=(_obs("psr", n=5, status="WATCH"),),
    )
    rec = IV.ItemRecord(item="x-I3", scope=_SCOPE, completion=c)
    current = {"psr": _obs("psr", n=n, status="RED")}
    assert IV.reconcile_record(rec, current, as_of=as_of).disposition is expected
    gate = IV.gate_proposal(IV.Proposal(id="p", priority="P2", scope=_SCOPE), [rec], current, as_of=as_of)
    assert (gate.decision is IV.Decision.REFUSE) == (expected is IV.Disposition.ACCUMULATING)


def test_ruling_without_declared_condition_never_reopens():
    rec = IV.ItemRecord(
        item="x-I4", scope=_SCOPE,
        ruling=IV.Ruling(decision="sunset", decided_on=date(2026, 9, 8), source_ref="x#4"),
    )
    current = {"alpha_vs_spy": _obs("alpha_vs_spy", value=-0.5, status="RED")}
    assert IV.reconcile_record(rec, current, as_of=_D).disposition is IV.Disposition.STAYS_CLOSED
    fresh_slug = IV.Proposal(id="brand-new-slug", priority="P0", scope=_SCOPE)
    assert IV.gate_proposal(fresh_slug, [rec], current, as_of=_D).decision is IV.Decision.REFUSE


def test_ruling_outranks_a_completion_on_the_same_record():
    rec = IV.ItemRecord(
        item="x-I5", scope=_SCOPE, completion=_repair(),
        ruling=IV.Ruling(decision="no_action", decided_on=date(2026, 9, 8), source_ref="x#5"),
    )
    current = {"mae_floor_breaches": _obs("mae_floor_breaches", value=9)}
    assert IV.reconcile_record(rec, current, as_of=_D).disposition is IV.Disposition.STAYS_CLOSED


def test_same_anchor_different_acceptance_is_different_work():
    rec = IV.ItemRecord(item="x-I6", scope=_SCOPE, completion=_repair())
    other = IV.Proposal(id="p", priority="P1",
                        scope=IV.WorkScope(anchor=" Crucible-Executor:sizing ", acceptance="exit-criteria"))
    same = IV.Proposal(id="q", priority="P1",
                       scope=IV.WorkScope(anchor="crucible-executor:SIZING", acceptance=" MAE-floor "))
    assert not IV.same_work(other, rec)
    assert IV.same_work(same, rec)


# ── schema boundary ────────────────────────────────────────────────────────


def test_newer_schema_version_is_refused_not_guessed():
    with pytest.raises(ValueError, match="schema_version"):
        IV.parse_record({"schema_version": IV.ITEM_VALIDITY_SCHEMA_VERSION + 1,
                         "item": "x", "scope": {"anchor": "a", "acceptance": "b"}})


def test_unknown_completion_kind_and_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        IV.parse_record({"item": "x", "scope": {"anchor": "a", "acceptance": "b"},
                         "completion": {"kind": "llm_says_done", "closed_on": "2026-10-01"}})
    with pytest.raises(ValidationError):
        IV.parse_record({"item": "x", "scope": {"anchor": "a", "acceptance": "b"}, "verdict": "green"})


def test_records_round_trip_through_json(ka):
    for rec in _records(ka):
        assert IV.parse_record(json.loads(rec.model_dump_json())) == rec


def test_observations_from_card_reads_status_value_and_n():
    card = {"tiles": {"predictor": {"components": [
        {"name": "momentum_l1_ic", "status": "RED", "value": -0.0244, "n": 16},
        {"name": "slim_cache_freshness", "status": "N/A-MISSING-INPUT", "value": None},
        {"name": "flag", "status": "GREEN", "value": True},
    ]}}}
    obs = IV.observations_from_card(card, as_of=_D)
    assert (obs["momentum_l1_ic"].status, obs["momentum_l1_ic"].value, obs["momentum_l1_ic"].n) == ("RED", -0.0244, 16)
    assert obs["slim_cache_freshness"].value is None
    assert obs["flag"].value is None
