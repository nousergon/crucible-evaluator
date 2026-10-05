"""
item_validity.py — the typed item completion / validity contract
(alpha-engine-config-I11990, slice 1 of 2: contract + known-answer fixtures).

Why this exists
---------------
Until this module the Director had ONE notion of "done": the metrics an item
cited read GREEN on the current card. Three code paths act on that one notion,
and each is wrong for most of the items it touches:

  * ``loop_verification.evidence_still_adverse`` re-tracks a closed item
    whenever any cited metric is RED/WATCH — so an investigation whose
    premise was DISPROVED (alpha-engine-config-I2972: the "stalled 21d writer"
    was normal 21-trading-day maturation) comes back every week, because
    ``momentum_l1_ic`` is still RED for reasons that have nothing to do with
    the writer.
  * ``loop_verification._file_new_issue_for_ruled_unrecovered`` files a fresh
    issue against a HUMAN sunset ruling (I964 / I1084 / I2978) on color alone,
    with no evidence that anything the ruling relied on has changed.
  * ``issue_filer.find_reconfirm_match`` treats one shared metric name as
    work identity, so a genuinely new defect (I11984, run-scope cadence
    attribution) was capped from P1 to P2 because an unrelated closed coverage
    investigation also cited ``sf_success_rate_4w``.

The contract
------------
An item's closure is a typed, evidence-backed record. There are exactly five
ways a piece of work can be complete, and each has its own evidence
requirement and its own — narrow — reopen condition:

  ``repair_verified``       a defect was fixed AND the fix was observed working.
                            Needs the merged change and a post-fix observation
                            of the DEFECT's own signature. Reopens only when
                            that signature recurs, never on outcome color.
  ``premise_disproved``     an investigation answered its question "no".
                            Needs the disproved claim and the measurement that
                            disproved it. Reopens only when that measurement
                            flips.
  ``monitor_accumulating``  the instrument is in place; the answer needs N or
                            calendar time. Needs the monitored measure, its
                            current N, and a required N or maturity date. It is
                            not unfinished work: it matures, or it STALLS
                            (maturity date passed short of N), and only a stall
                            is new work.
  ``guard_refusing``        a safety guard is refusing, correctly. Needs the
                            guard and the refusal observation. Reopens only on
                            evidence the guard itself misbehaves, never because
                            what it guards is still bad.
  ``outcome_recovered``     the economic outcome itself crossed its line. The
                            ONLY kind whose reopen condition may be a card
                            color (``status_in``).

A human RULING (sunset / no-action / close) is an authority overlay, not a
sixth completion kind: a ruled close stays closed unless the ruling's own
declared ``reopen_when`` is met by a current observation. A ruling that
declares none is never reopened or re-tracked autonomously. A changed slug is
not a changed premise — identity here is the WORK SCOPE (causal anchor +
acceptance key), never the slug and never a shared metric citation.

Missing closure evidence is ``unevaluable`` — named and counted, never read as
green and never read as permission to file fresh work.

Determinism: every verdict here is a pure function of typed records and
observations. No LLM produces or edits a completion verdict.

Slice 2 (not in this module yet): wire ``reconcile_record`` / ``gate_proposal``
into ``loop_verification.verify_and_correct``, ``issue_filer`` (replacing the
metric-overlap reconfirm match) and ``agent._carryover_context``; add an
additive, versioned ``work_scope`` / ``targets`` to ``schema.ActionItem``; and
sweep the stored ledger through this code, recording corrected / retained /
unevaluable per row. This slice changes no runtime behavior.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable, Mapping
from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#: Bumped on any non-additive change to the records below. Additive fields keep
#: the version; a consumer reading a NEWER version than it knows refuses rather
#: than guessing (see :func:`parse_record`).
ITEM_VALIDITY_SCHEMA_VERSION = 1


class CompletionKind(str, Enum):
    REPAIR_VERIFIED = "repair_verified"
    PREMISE_DISPROVED = "premise_disproved"
    MONITOR_ACCUMULATING = "monitor_accumulating"
    GUARD_REFUSING = "guard_refusing"
    OUTCOME_RECOVERED = "outcome_recovered"


class Disposition(str, Enum):
    """What reconciliation concludes about one closed / standing record today."""

    STAYS_CLOSED = "stays_closed"
    REOPEN_CONDITION_MET = "reopen_condition_met"
    ACCUMULATING = "accumulating"
    MATURED = "matured"
    STALLED = "stalled"
    UNEVALUABLE = "unevaluable"


class Decision(str, Enum):
    """What the proposal gate concludes about one NEW proposed item."""

    ALLOW = "allow"
    REFUSE = "refuse"
    HOLD_UNEVALUABLE = "hold_unevaluable"


_FROZEN = ConfigDict(frozen=True, extra="forbid")


class Observation(BaseModel):
    """One measured fact, with where and when it was read."""

    model_config = _FROZEN

    measure: str
    observed_at: date
    source: str = Field(min_length=1, description="Artifact / metric namespace / card the value was read from.")
    value: float | None = None
    status: str | None = Field(default=None, description="Card status (GREEN/WATCH/RED/N/A-*) when the measure is a card component.")
    n: int | None = None


_OPS = {
    "gt": operator.gt, "ge": operator.ge, "lt": operator.lt,
    "le": operator.le, "eq": operator.eq, "ne": operator.ne,
}


class Predicate(BaseModel):
    """A deterministic condition over ONE measure's current observation."""

    model_config = _FROZEN

    measure: str
    op: Literal["gt", "ge", "lt", "le", "eq", "ne", "status_in"]
    threshold: float | None = None
    statuses: tuple[str, ...] = ()

    def evaluate(self, current: Mapping[str, Observation]) -> bool | None:
        """``True`` / ``False``, or ``None`` when the input is absent — an
        unmeasured condition is never read as met or as clear."""
        obs = current.get(self.measure)
        if obs is None:
            return None
        if self.op == "status_in":
            if obs.status is None:
                return None
            return obs.status.upper() in {s.upper() for s in self.statuses}
        if obs.value is None or self.threshold is None:
            return None
        return bool(_OPS[self.op](obs.value, self.threshold))


class WorkScope(BaseModel):
    """Work identity: the causal anchor the work changes or answers about,
    plus a stable acceptance key. Two items are the same work iff both match.
    Metric citations are deliberately NOT part of identity."""

    model_config = _FROZEN

    anchor: str = Field(min_length=1, description="repo:path::symbol, or the artifact/instrument answered about.")
    acceptance: str = Field(min_length=1, description="Stable key for what 'done' means for this work.")

    def identity(self) -> tuple[str, str]:
        return (" ".join(self.anchor.lower().split()), " ".join(self.acceptance.lower().split()))


class Ruling(BaseModel):
    """A human ruling that closed the item. Authority, not evidence."""

    model_config = _FROZEN

    decision: Literal["sunset", "no_action", "close", "reject"]
    decided_on: date
    source_ref: str = Field(min_length=1, description="Issue comment / Decision Queue record carrying the ruling.")
    reopen_when: Predicate | None = Field(
        default=None,
        description="The ruling's declared invalidity condition. None = never reopened or re-tracked autonomously.",
    )


class ItemCompletion(BaseModel):
    """An evidence-backed claim that the work is complete in one of five ways."""

    model_config = _FROZEN

    kind: CompletionKind
    closed_on: date
    closure_ref: str | None = None
    evidence: tuple[Observation, ...] = ()
    reopen_when: Predicate | None = None
    # kind-specific evidence
    change_ref: str | None = None          # repair_verified: merged PR / commit
    premise: str | None = None             # premise_disproved: the claim disproved
    guard: str | None = None               # guard_refusing: the guard refusing
    monitored_measure: str | None = None   # monitor_accumulating
    required_n: int | None = None          # monitor_accumulating
    matures_on: date | None = None         # monitor_accumulating


class ItemRecord(BaseModel):
    """One work item's durable completion / ruling state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = ITEM_VALIDITY_SCHEMA_VERSION
    item: str = Field(min_length=1, description="Canonical ref, e.g. alpha-engine-config-I2972.")
    scope: WorkScope
    completion: ItemCompletion | None = None
    ruling: Ruling | None = None


class Proposal(BaseModel):
    """The slice of a proposed action item the gate reads."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    priority: Literal["P0", "P1", "P2", "P3"]
    scope: WorkScope | None = None
    targets: tuple[str, ...] = Field(default=(), description="Items this proposal claims to land / redo.")
    evidence: tuple[str, ...] = ()


class Reconciliation(BaseModel):
    model_config = _FROZEN

    item: str
    disposition: Disposition
    reason: str
    missing: tuple[str, ...] = ()


class ProposalDecision(BaseModel):
    model_config = _FROZEN

    proposal: str
    decision: Decision
    priority: str
    reason: str
    matched: tuple[str, ...] = ()


def parse_record(data: Mapping) -> ItemRecord:
    """Parse a stored record, refusing a schema newer than this code knows."""
    version = data.get("schema_version", ITEM_VALIDITY_SCHEMA_VERSION)
    if not isinstance(version, int) or version > ITEM_VALIDITY_SCHEMA_VERSION:
        raise ValueError(
            f"item-validity record {data.get('item')!r} has schema_version={version!r}; "
            f"this code reads <= {ITEM_VALIDITY_SCHEMA_VERSION}"
        )
    return ItemRecord.model_validate(data)


# ── evidence requirements ──────────────────────────────────────────────────


def missing_evidence(completion: ItemCompletion) -> tuple[str, ...]:
    """What ``completion`` lacks to stand as its declared kind. Empty = complete.

    The cross-kind rule is the one I11990 is about: a reopen condition must be
    stated over a measure the closure actually observed, and a card-color
    (``status_in``) condition is allowed ONLY for ``outcome_recovered``. A
    repair, a disproof or a guard is not undone by an outcome staying red.
    """
    k = completion.kind
    missing: list[str] = []
    if not completion.closure_ref:
        missing.append("closure_ref")
    if not completion.evidence:
        missing.append("evidence")
    measures = {o.measure for o in completion.evidence}

    if k is CompletionKind.REPAIR_VERIFIED and not completion.change_ref:
        missing.append("change_ref")
    if k is CompletionKind.PREMISE_DISPROVED and not completion.premise:
        missing.append("premise")
    if k is CompletionKind.GUARD_REFUSING and not completion.guard:
        missing.append("guard")
    if k is CompletionKind.MONITOR_ACCUMULATING:
        if not completion.monitored_measure:
            missing.append("monitored_measure")
        if completion.required_n is None and completion.matures_on is None:
            missing.append("required_n|matures_on")
    elif completion.reopen_when is None:
        missing.append("reopen_when")

    rw = completion.reopen_when
    if rw is not None:
        if rw.measure not in measures:
            missing.append(f"reopen_when.measure observed in evidence ({rw.measure})")
        if rw.op == "status_in" and k is not CompletionKind.OUTCOME_RECOVERED:
            missing.append(f"reopen_when not a card-color condition for {k.value}")
    return tuple(missing)


# ── reconciliation ─────────────────────────────────────────────────────────


def reconcile_record(
    record: ItemRecord, current: Mapping[str, Observation], *, as_of: date,
) -> Reconciliation:
    """Today's disposition of one record. Pure; never raises on a valid record."""
    ref = record.item
    if record.ruling is not None:
        ruling = record.ruling
        if ruling.reopen_when is None:
            return Reconciliation(
                item=ref, disposition=Disposition.STAYS_CLOSED,
                reason=f"human {ruling.decision} ruling {ruling.decided_on} declares no reopen condition",
            )
        met = ruling.reopen_when.evaluate(current)
        if met is None:
            return Reconciliation(
                item=ref, disposition=Disposition.UNEVALUABLE,
                reason=f"ruling's reopen condition on {ruling.reopen_when.measure} has no current observation",
            )
        if met:
            return Reconciliation(
                item=ref, disposition=Disposition.REOPEN_CONDITION_MET,
                reason=f"ruling's declared reopen condition on {ruling.reopen_when.measure} is met",
            )
        return Reconciliation(
            item=ref, disposition=Disposition.STAYS_CLOSED,
            reason=f"ruling's reopen condition on {ruling.reopen_when.measure} is not met",
        )

    completion = record.completion
    if completion is None:
        return Reconciliation(
            item=ref, disposition=Disposition.UNEVALUABLE,
            reason="no authoritative completion or ruling record", missing=("completion|ruling",),
        )
    gaps = missing_evidence(completion)
    if gaps:
        return Reconciliation(
            item=ref, disposition=Disposition.UNEVALUABLE,
            reason=f"{completion.kind.value} closure lacks required evidence", missing=gaps,
        )

    if completion.kind is CompletionKind.MONITOR_ACCUMULATING:
        obs = current.get(completion.monitored_measure or "")
        if obs is None or obs.n is None:
            return Reconciliation(
                item=ref, disposition=Disposition.UNEVALUABLE,
                reason=f"no current N for monitored {completion.monitored_measure}",
            )
        if completion.required_n is not None and obs.n >= completion.required_n:
            return Reconciliation(
                item=ref, disposition=Disposition.MATURED,
                reason=f"{completion.monitored_measure} N={obs.n} reached required {completion.required_n}",
            )
        if completion.matures_on is not None and as_of > completion.matures_on:
            return Reconciliation(
                item=ref, disposition=Disposition.STALLED,
                reason=f"{completion.monitored_measure} N={obs.n} short of required past {completion.matures_on}",
            )
        return Reconciliation(
            item=ref, disposition=Disposition.ACCUMULATING,
            reason=f"{completion.monitored_measure} N={obs.n} still accumulating",
        )

    assert completion.reopen_when is not None  # guaranteed by missing_evidence
    met = completion.reopen_when.evaluate(current)
    if met is None:
        return Reconciliation(
            item=ref, disposition=Disposition.UNEVALUABLE,
            reason=f"reopen condition on {completion.reopen_when.measure} has no current observation",
        )
    if met:
        return Reconciliation(
            item=ref, disposition=Disposition.REOPEN_CONDITION_MET,
            reason=f"{completion.kind.value}: declared reopen condition on {completion.reopen_when.measure} is met",
        )
    return Reconciliation(
        item=ref, disposition=Disposition.STAYS_CLOSED,
        reason=f"{completion.kind.value}: reopen condition on {completion.reopen_when.measure} not met",
    )


# ── identity + proposal gate ───────────────────────────────────────────────


def same_work(proposal: Proposal, record: ItemRecord) -> bool:
    """Work identity: an explicit target, or an equal WorkScope. Shared metric
    citations never make two items the same work."""
    if record.item in proposal.targets:
        return True
    return proposal.scope is not None and proposal.scope.identity() == record.scope.identity()


_ALLOWING = {Disposition.REOPEN_CONDITION_MET, Disposition.MATURED, Disposition.STALLED}


def gate_proposal(
    proposal: Proposal,
    records: Iterable[ItemRecord],
    current: Mapping[str, Observation],
    *,
    as_of: date,
) -> ProposalDecision:
    """May ``proposal`` be filed as new work, given the closed / ruled records?

    The proposal's priority is returned UNCHANGED in every branch: this gate
    refuses or allows, it never quietly downgrades. Same-work is decided by
    :func:`same_work`; a proposal matching no record is new work at its own
    urgency, however many metrics it shares with closed items."""
    matched = [r for r in records if same_work(proposal, r)]
    if not matched:
        return ProposalDecision(
            proposal=proposal.id, decision=Decision.ALLOW, priority=proposal.priority,
            reason="no closed or ruled item shares this work scope",
        )
    recs = [(r, reconcile_record(r, current, as_of=as_of)) for r in matched]
    refs = tuple(r.item for r, _ in recs)
    blocking = [(r, rc) for r, rc in recs if rc.disposition not in _ALLOWING]
    unevaluable = [rc for _, rc in blocking if rc.disposition is Disposition.UNEVALUABLE]
    if unevaluable:
        return ProposalDecision(
            proposal=proposal.id, decision=Decision.HOLD_UNEVALUABLE, priority=proposal.priority,
            reason="; ".join(f"{rc.item}: {rc.reason}" for rc in unevaluable), matched=refs,
        )
    if blocking:
        return ProposalDecision(
            proposal=proposal.id, decision=Decision.REFUSE, priority=proposal.priority,
            reason="; ".join(f"{rc.item}: {rc.reason}" for _, rc in blocking), matched=refs,
        )
    return ProposalDecision(
        proposal=proposal.id, decision=Decision.ALLOW, priority=proposal.priority,
        reason="; ".join(f"{rc.item}: {rc.reason}" for _, rc in recs), matched=refs,
    )


def observations_from_card(card: Mapping, *, as_of: date, source: str = "report_card") -> dict[str, Observation]:
    """Card components as ``{name: Observation}`` (status / value / N) — the
    adapter slice 2 feeds :func:`reconcile_record` from. Non-numeric values are
    kept as ``None`` rather than coerced."""
    out: dict[str, Observation] = {}
    for tile in (card.get("tiles") or {}).values():
        for c in (tile or {}).get("components", []) or []:
            name = c.get("name")
            if not name:
                continue
            value = c.get("value")
            n = c.get("n")
            out[str(name)] = Observation(
                measure=str(name), observed_at=as_of, source=source,
                value=float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None,
                status=str(c["status"]) if c.get("status") is not None else None,
                n=int(n) if isinstance(n, int) and not isinstance(n, bool) else None,
            )
    return out
