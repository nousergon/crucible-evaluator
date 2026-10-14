"""
item_validity_gate.py — the weekly Director's reading of the item-validity
contract (alpha-engine-config-I11990, slice 2, first part: the grading pass).

What this decides
-----------------
``director/item_validity.py`` (crucible-evaluator-PR338) defines WHEN a closed
or ruled item may be judged again. Until this module nothing in the weekly run
read it: ``loop_verification.verify_and_correct`` graded every closed ledger
item on one rule — "do the metrics it cited still read RED/WATCH?" — and acted
on the answer (reopen, notice comment, fresh re-track issue). That rule is
wrong for most of the items it touches: a disproved premise (I2972), a verified
repair (I7644) and a human sunset ruling (I964 / I1084 / I2978) all sit next to
metrics that stay RED for reasons the item never claimed to fix.

This module answers ONE question per ledger item, before any grading runs:

    Is this item valid to grade on its cited-metric color this week — and if
    not, why not?

The answer is stamped on the ledger row as ``item_validity`` (persisted with
the ledger, so the per-item statement is durable) and summarised for the
digest and the SF output. ``verify_and_correct`` then skips the color grade —
and every mutation that follows from it — for an item that is not valid to
grade.

How the answer is reached
-------------------------
* **Identity is the contract's, never the slug.** A ledger row is the work of
  its own tracked issue (``alpha-engine-config-I<issue_number>``), plus any
  explicit ``targets`` / ``work_scope`` the row carries. Nothing writes those
  two fields yet; they are read when present so the slice-2 schema addition and
  ledger sweep plug in without a second change here. Prose is never parsed for
  issue references, and a shared metric is never identity.
* **The decision is ``item_validity.gate_proposal``'s, unchanged.** ``allow``
  with a matched record is valid to grade (the record's own reopen condition is
  met, or its monitor matured / stalled). ``refuse`` is not valid: the record
  stays closed / is still accumulating. ``hold_unevaluable`` is not valid: the
  closure evidence or the current observation its condition needs is missing —
  never read as green and never as permission. ``allow`` with NO matched record
  is the pre-I11990 rule, stated as such: graded on cited-metric color.
* **Observations come from this week's report card only**
  (``item_validity.observations_from_card``). A record whose condition is over a
  measure the card does not carry (a CloudWatch gauge, a ruling flag) is
  therefore ``hold_unevaluable`` until a producer supplies that observation —
  named in the reason, so the gap is visible rather than defaulted.
* **No threshold lives here.** Every value a condition compares against is
  declared per record in ``item_validity_records.json``; this module only reads
  them.

Absent vs unreadable
--------------------
* Records file **absent** → every item is graded as before, and the report says
  so (``records_absent``). That is the tolerant path for a deploy that does not
  carry the file.
* Records file **unreadable** (bad JSON, a schema version newer than this code,
  a record failing validation, a duplicate item) → fail CLOSED: no closed item
  is graded this week, and the report names the error. Grading an item whose
  declared validity could not be read is exactly the defect I11990 exists to
  stop, so the safe direction is to grade nothing and say why.

Determinism: pure functions of the ledger rows, the records file and the card.
No LLM produces or edits a verdict here.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from director.item_validity import (
    ITEM_VALIDITY_SCHEMA_VERSION,
    Decision,
    ItemRecord,
    Proposal,
    WorkScope,
    gate_proposal,
    observations_from_card,
    parse_record,
)

logger = logging.getLogger(__name__)

#: The declared records the weekly run reads. Shipped with the code (the
#: Dockerfile COPYs ``director/``), so a change to any declared reopen
#: condition is a reviewed PR — the threshold values in it are Brian's call
#: (alpha-engine-config-I11990), not this module's.
RECORDS_PATH = Path(__file__).with_name("item_validity_records.json")

#: The repo every ledger ``issue_number`` lives in (``issue_filer.DEFAULT_REPO``
#: = ``nousergon/alpha-engine-config``), as the canonical-ref prefix the
#: records use.
ISSUE_REF_PREFIX = "alpha-engine-config-I"

#: The ledger-row key the per-item verdict is stamped under, and that
#: ``loop_verification.verify_and_correct`` reads.
VERDICT_KEY = "item_validity"

BASIS_RECORD = "item_validity_record"
BASIS_NO_RECORD = "no_record"
BASIS_RECORDS_ABSENT = "records_absent"
BASIS_RECORDS_UNREADABLE = "records_unreadable"

#: Longest reason carried per item onto the summary / ledger row. The full
#: reconciliation is reproducible from the records file and the dated card.
_MAX_REASON_CHARS = 400

_PRIORITIES = {"P0", "P1", "P2", "P3"}


@dataclass(frozen=True)
class RecordSet:
    """The declared records as the weekly run read them."""

    status: str  # "loaded" | "absent" | "error"
    records: tuple[ItemRecord, ...] = ()
    error: str | None = None
    source: str = ""


def load_records(path: str | Path | None = None) -> RecordSet:
    """Read and validate the declared records. Never raises."""
    p = Path(path) if path is not None else RECORDS_PATH
    try:
        raw = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return RecordSet(status="absent", source=str(p))
    except OSError as e:
        return RecordSet(status="error", error=f"{type(e).__name__}: {e}"[:_MAX_REASON_CHARS], source=str(p))
    try:
        data = json.loads(raw)
        rows = data.get("records") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            raise ValueError("records file has no 'records' list")
        records = tuple(parse_record(r) for r in rows)
        seen: set[str] = set()
        for r in records:
            if r.item in seen:
                raise ValueError(f"duplicate record for {r.item}")
            seen.add(r.item)
    except ValueError as e:  # json / pydantic ValidationError / newer schema_version
        return RecordSet(status="error", error=f"{type(e).__name__}: {e}"[:_MAX_REASON_CHARS], source=str(p))
    return RecordSet(status="loaded", records=records, source=str(p))


def item_ref(row: Mapping) -> str | None:
    """The canonical ref of the row's own tracked issue, or ``None``."""
    number = row.get("issue_number")
    if isinstance(number, bool):
        return None
    if isinstance(number, int) and number > 0:
        return f"{ISSUE_REF_PREFIX}{number}"
    if isinstance(number, str) and number.strip().isdigit():
        return f"{ISSUE_REF_PREFIX}{int(number.strip())}"
    return None


def _row_scope(row: Mapping) -> WorkScope | None:
    scope = row.get("work_scope")
    if not isinstance(scope, Mapping):
        return None
    try:
        return WorkScope.model_validate(dict(scope))
    except ValueError:
        return None


#: A bare or repo-qualified reference to an ``alpha-engine-config`` issue, in
#: the forms an action item's ``targets`` arrives in: ``#2972``, ``I2972``,
#: ``2972``, ``alpha-engine-config-I2972``, ``alpha-engine-config#2972``,
#: ``nousergon/alpha-engine-config#2972``. Anything else (another repo's ref,
#: prose) is kept verbatim: it can still equal a record's ``item`` exactly, and
#: it is never parsed into a number it does not state.
_ISSUE_REF_RE = re.compile(
    r"^(?:(?:nousergon/)?alpha-engine-config\s*(?:-\s*i|#)|#|i)?\s*#?\s*(\d+)$",
    re.IGNORECASE,
)


def canonical_ref(ref: object) -> str | None:
    """``alpha-engine-config-I<N>`` for an issue reference in any of the forms
    above, the stripped string for any other non-empty string, else ``None``."""
    if not isinstance(ref, str):
        return None
    text = ref.strip()
    if not text:
        return None
    m = _ISSUE_REF_RE.match(text)
    return f"{ISSUE_REF_PREFIX}{int(m.group(1))}" if m else text


def proposal_for_row(row: Mapping) -> Proposal:
    """The row as the contract's ``Proposal``: its own issue plus any explicit
    ``targets`` / ``work_scope``. ``priority`` is carried for the type only —
    this verdict never reads or changes it."""
    targets: list[str] = []
    own = item_ref(row)
    if own:
        targets.append(own)
    extra = row.get("targets")
    if isinstance(extra, (list, tuple)):
        targets.extend(c for c in (canonical_ref(t) for t in extra) if c)
    priority = str(row.get("priority") or "").strip().upper()
    return Proposal(
        id=str(row.get("id") or "(no id)"),
        priority=priority if priority in _PRIORITIES else "P3",
        scope=_row_scope(row),
        targets=tuple(dict.fromkeys(targets)),
    )


def _verdict(*, valid: bool, basis: str, reason: str, as_of: date,
             decision: str | None = None, matched: tuple[str, ...] = ()) -> dict:
    return {
        "schema_version": ITEM_VALIDITY_SCHEMA_VERSION,
        "as_of": as_of.isoformat(),
        "valid_to_grade": valid,
        "basis": basis,
        "decision": decision,
        "matched": list(matched),
        "reason": reason[:_MAX_REASON_CHARS],
    }


def assess_row(row: Mapping, record_set: RecordSet, observations: Mapping, *, as_of: date) -> dict:
    """One row's verdict: is it valid to grade on cited-metric color, and why."""
    if record_set.status == "absent":
        return _verdict(
            valid=True, basis=BASIS_RECORDS_ABSENT, as_of=as_of,
            reason="no item-validity records file; graded on cited-metric color (pre-I11990 rule)",
        )
    if record_set.status != "loaded":
        return _verdict(
            valid=False, basis=BASIS_RECORDS_UNREADABLE, as_of=as_of,
            reason=f"item-validity records unreadable ({record_set.error}); not graded until they read",
        )
    decision = gate_proposal(proposal_for_row(row), record_set.records, observations, as_of=as_of)
    if not decision.matched:
        return _verdict(
            valid=True, basis=BASIS_NO_RECORD, as_of=as_of, decision=decision.decision.value,
            reason="no declared completion or ruling shares this item's work; graded on cited-metric color",
        )
    if decision.decision is Decision.ALLOW:
        reason = f"declared condition met — {decision.reason}"
    elif decision.decision is Decision.HOLD_UNEVALUABLE:
        reason = f"held, unevaluable — {decision.reason}"
    else:
        reason = f"refused — {decision.reason}"
    return _verdict(
        valid=decision.decision is Decision.ALLOW, basis=BASIS_RECORD, as_of=as_of,
        decision=decision.decision.value, matched=decision.matched, reason=reason,
    )


def assess_ledger(rows: list[dict], card: Mapping | None, *, as_of: date,
                  record_set: RecordSet | None = None) -> dict:
    """Stamp every row's verdict under :data:`VERDICT_KEY` and return the
    summary the handler folds into the run output and the digest.

    Mutates ``rows`` in place, like the rest of the loop pass: the stamp is
    persisted by the same ledger write, so next week's reader and the console
    see what this week decided and why."""
    rs = record_set if record_set is not None else load_records()
    observations = observations_from_card(card or {}, as_of=as_of)
    items: list[dict] = []
    for row in rows:
        verdict = assess_row(row, rs, observations, as_of=as_of)
        row[VERDICT_KEY] = verdict
        items.append({
            "id": row.get("id"),
            "issue_number": row.get("issue_number"),
            "valid_to_grade": verdict["valid_to_grade"],
            "basis": verdict["basis"],
            "decision": verdict["decision"],
            "matched": verdict["matched"],
            "reason": verdict["reason"],
        })
    if rs.status == "error":
        logger.error(
            "Director item validity: records UNREADABLE at %s (%s) — no closed item is "
            "graded this run (alpha-engine-config-I11990, fail closed).", rs.source, rs.error,
        )
    summary = {
        "director_item_validity": rs.status,
        "director_item_validity_records": len(rs.records),
        "director_item_validity_valid": sum(1 for i in items if i["valid_to_grade"]),
        "director_item_validity_not_valid": sum(1 for i in items if not i["valid_to_grade"]),
        "director_item_validity_items": items,
    }
    if rs.error:
        summary["director_item_validity_error"] = rs.error
    return summary


def not_valid_to_grade(row: Mapping) -> bool:
    """True when this run's verdict says the row must not be color-graded.

    Only an explicit ``valid_to_grade: False`` withholds: a row the assessment
    never stamped (a direct caller, a pre-I11990 ledger) is graded as before."""
    verdict = row.get(VERDICT_KEY)
    return isinstance(verdict, Mapping) and verdict.get("valid_to_grade") is False


# ── the proposal half: this week's plan items (I11990 slice 2, second part) ──
#
# The grading pass above decides whether a CLOSED item may be graded on color.
# This half asks the contract's other question of every item the plan just
# PROPOSED: may it stand as open work at all, given the declared closures and
# rulings? It is read by two consumers, neither of which may decide it for
# itself:
#
#   * the plan artifact, which carries the per-item verdict as an extra
#     (``PLAN_ITEM_VALIDITY_KEY``) — so the dated plan states, next to each
#     item, whether a declared record holds it and why;
#   * ``issue_filer.file_director_issues``, which does not file an item the
#     verdict says is not valid to file.
#
# Identity is the same as the grading pass: the item's own ``targets`` (what
# the proposer declared it lands or redoes), plus — when the item carries a
# ledger row forward — that row's tracked issue and stored ``targets``. A plan
# item restating ``validate-21d-return-column-stall`` is the work of I2978 by
# the ledger, whatever its title now says about #2972.

#: The plan-artifact extra the per-item verdicts are stamped under. An EXTRA on
#: ``DirectorWeeklyActionPlan`` (``extra="allow"``), never a declared field: a
#: verdict the LLM is asked to produce about its own items is not a verdict
#: (``verdict.stamp_plan_artifact`` gives the same reasoning).
PLAN_ITEM_VALIDITY_KEY = "action_item_validity"


def _plan_item_row(item: Mapping, ledger_row: Mapping | None) -> dict:
    """The fields :func:`proposal_for_row` reads, for one plan item."""
    targets: list = list(item.get("targets") or [])
    row: dict = {"id": item.get("id"), "priority": item.get("priority"), "issue_number": None}
    if ledger_row is not None:
        row["issue_number"] = ledger_row.get("issue_number")
        targets.extend(ledger_row.get("targets") or [])
        if isinstance(ledger_row.get("work_scope"), Mapping):
            row["work_scope"] = ledger_row["work_scope"]
    row["targets"] = targets
    return row


def assess_plan_items(items: list[Mapping], ledger_rows: list[Mapping] | None,
                      card: Mapping | None, *, as_of: date,
                      record_set: RecordSet | None = None) -> dict:
    """Every plan item's verdict: may it stand as open work this week, and why.

    ``items`` are the plan's action items as dicts (``ActionItem.model_dump()``);
    ``ledger_rows`` are the carry-over ledger's active rows as the plan was built
    from them (their ``issue_number`` is identity for an item carried forward).

    Returns the block stamped onto the plan under :data:`PLAN_ITEM_VALIDITY_KEY`.
    ``items[*].valid_to_file`` is the one field the issue filer obeys.

    Records **unreadable** fails closed only where it can matter: an item with
    no identity at all (no targets, no tracked issue) cannot match ANY record,
    so its verdict does not depend on the records and it stays valid; an item
    WITH identity is held, because whether its declared record refuses it is
    exactly what could not be read."""
    rs = record_set if record_set is not None else load_records()
    observations = observations_from_card(card or {}, as_of=as_of)
    by_id = {r.get("id"): r for r in (ledger_rows or []) if isinstance(r, Mapping)}
    out: list[dict] = []
    for item in items:
        row = _plan_item_row(item, by_id.get(item.get("id")))
        has_identity = bool(proposal_for_row(row).targets) or "work_scope" in row
        if rs.status == "error" and not has_identity:
            verdict = _verdict(
                valid=True, basis=BASIS_NO_RECORD, as_of=as_of,
                reason="no targets and no tracked issue — no declared record can share this "
                       "work, so the unreadable records file does not bear on it",
            )
        else:
            verdict = assess_row(row, rs, observations, as_of=as_of)
        out.append({
            "id": item.get("id"),
            "issue_number": row.get("issue_number"),
            "targets": list(proposal_for_row(row).targets),
            "valid_to_file": verdict["valid_to_grade"],
            "basis": verdict["basis"],
            "decision": verdict["decision"],
            "matched": verdict["matched"],
            "reason": verdict["reason"],
        })
    block = {
        "schema_version": ITEM_VALIDITY_SCHEMA_VERSION,
        "as_of": as_of.isoformat(),
        "records": rs.status,
        "n_records": len(rs.records),
        "n_items": len(out),
        "n_not_valid": sum(1 for i in out if not i["valid_to_file"]),
        "items": out,
    }
    if rs.error:
        block["records_error"] = rs.error
    return block


def not_valid_to_file(block: Mapping | None) -> dict[str, dict]:
    """``{item id: verdict}`` for every item ``block`` says may not be filed.
    An absent or malformed block withholds nothing — the pre-I11990 path."""
    if not isinstance(block, Mapping):
        return {}
    return {
        str(i.get("id")): dict(i)
        for i in (block.get("items") or [])
        if isinstance(i, Mapping) and i.get("valid_to_file") is False
    }
