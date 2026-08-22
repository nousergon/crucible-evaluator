"""verdict.py — the Director's §2.3a consumer half.

WHY THIS EXISTS
---------------
The Report Card has carried a runtime correctness verdict at
``report_card["attestation"]`` since ``config-I6973``/``I7039``: the worst of
three independently-versioned known-answer batteries — this image's quant
primitives, the simulation engine's fills/fees/NAV arithmetic, and the Evaluator
stage's ranking metrics (IC, hit rate, calibration).

The Director read that card and never read that field.

``sf-pipeline-policy.md`` §2.3a rule 1 is explicit that this is the defect, not a
gap in coverage: *the verdict is consumed by every stage whose output depends on
it being true — not merely by whatever happens to read it today. A stage that
reports, grades, promotes or acts on the run's numbers depends on those numbers
being uncontaminated, whether or not it currently reads the check that says so.*

The Director is the strongest instance of that sentence anywhere in the fleet.
It does not merely render the week's numbers, it **acts on them**, and every one
of those actions is durable and outward-facing:

- it files ``area:director-proposals`` GitHub issues into the fleet backlog,
- it **reopens** issues whose cited metric it judges unrecovered,
- it **escalates** carried-over items to Brian's Decision Queue,
- it emails a weekly digest that Brian reads as the system's own read of itself.

Run that off arithmetic nobody checked and the contamination stops being a
number on a page: it becomes tracked work, a reopened issue, and a reserved
matter on the queue — each of which outlives the cycle that produced it and
carries no memory of having been derived from unverified inputs.

Each of those four is a **mutation into a system other than this one**, which is
the property the gate is actually about — see ``MUTATING_ACTIONS``.

WHAT IT DOES
------------
``read_card_verdict`` normalizes the card's verdict onto the closed vocabulary,
and ``actions_withheld`` says whether the Director may exercise its **mutating**
authority this cycle.

**What the gate withholds is authority, not the verification pass** (Brian
ruling 2026-08-22, ``alpha-engine-config-I8187``). The withheld set is split by
MUTATION CLASS — ``MUTATING_ACTIONS`` (file an issue, reopen an issue, escalate a
carried item to the Decision Queue) stop; ``UNGATED_ACTIONS`` (the ledger's
``issue_number`` backfill, the carry-over reconciliation that only annotates) run
under every verdict. The distinction is mutates-the-world vs reads-the-world,
never attestation-clean vs not.

That split is not a softening of §2.3a; it is what §2.3a already meant. Gating a
read on a run-quality flag buys no safety and costs self-correction: the
``issue_number`` backfill sat inside this gate through three consecutive UNKNOWN
cycles (2026-08-13, -08-14, -08-21), which is why ``issue_number`` is null on all
28 ledger rows (``alpha-engine-config-I8179``) — and ``verify_and_correct``
``continue``s past every row without one. So the withholding did not merely pause
the corrections, it dismantled the state those corrections need, and the longer
the attestation stayed UNKNOWN the less able the system became to correct itself
when it cleared.

**The absent case is the whole point.** A card written before the attestation
producer existed, a card whose attestation block failed to build, and a card
carrying an unrecognised verdict string are all indistinguishable from a clean
run to any consumer that tests truthiness or uses ``.get(..., True)``. Every one
of them resolves to ``UNKNOWN`` here, and ``UNKNOWN`` withholds. §2.3a rule 2 —
a missing verdict propagates as ``UNKNOWN``, never as a pass.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not fail the Director. §2.3a: *withholding the guarantee is not the same
as failing the pipeline… a verdict stage that dies must not kill stages that do
not depend on it.* An ``UNKNOWN`` here is frequently the ordinary consequence of
a spot reclaim on the box that produced the backtest — killing the weekly
advisory on that would trade one blindness for an outage. The plan is still
built, still persisted, and still emailed; it is persisted and emailed **marked**,
and the actions that create durable state elsewhere are the ones that stop.

By the same argument one step further (I8187): an action that creates durable
state *here* — a backfilled ``issue_number``, an annotation on a carried row — is
on the same side of the line as rendering, not on the side of a reopen. Stopping
it trades one blindness for a different one, and this module's own history is the
evidence.
"""
from __future__ import annotations

import json
import logging
from typing import Any

# The closed vocabulary and the only correct read of it, defined once in this
# repo by the module that produces the block. Imported rather than restated:
# a second copy of `verdict == "PASS"` is a second place the "missing reads as
# pass" bug can be reintroduced (`policy-shared-code` — same repo, so the fix
# is an import, not a lift).
from grading.attestation import FAIL, PARTIAL, PASS, UNKNOWN, verdict_is_pass
from grading.pipeline_gates import (
    PIPELINE_GATES_KEY,
    gates_unmeasured,
    read_gate_state,
)

logger = logging.getLogger(__name__)

__all__ = [  # re-exported: the Director's callers read the key from here
    "PIPELINE_GATES_KEY", "ATTESTATION_KEY", "GATED_ACTIONS", "MUTATING_ACTIONS",
    "UNGATED_ACTIONS", "actions_withheld", "is_gated_action",
    "log_verdict", "read_card_verdict", "read_pipeline_gates",
    "stamp_plan_artifact", "withheld_reason", "withheld_summary",
]

#: The card key carrying the §2.3a block. Cross-repo contract with
#: ``crucible-evaluator grading/attestation.py::build_run_attestation``, whose
#: emitted body is stamped ``report_card_attestation-1.0.0``.
ATTESTATION_KEY = "attestation"

_VALID_VERDICTS = frozenset({PASS, FAIL, PARTIAL, UNKNOWN})

#: The Director actions gated on the verdict, split by MUTATION CLASS.
#:
#: Brian ruling 2026-08-22 (``alpha-engine-config-I8187``), option (c): the gate
#: is on the AUTHORITY an action exercises, never on a global run-quality flag.
#: The test for membership here is the one ``overseer-policy.md`` §6 applies to
#: autonomous action — a T2 write into somebody else's system is not the same
#: act as a T0 read that informs a human or a prompt, and one verdict flag must
#: not govern both.
#:
#: Membership test: does this action create state that OUTLIVES this cycle in a
#: system OTHER than the Director's own? A GitHub issue filed, an issue
#: reopened, a reserved matter placed on Brian's Decision Queue — yes. A read,
#: an annotation, a write into the Director's own ledger — no.
#:
#: ``loop_verification`` is deliberately absent from both tuples: it was the
#: atom this ruling dissolved. The pass is not one authority — it is a
#: non-mutating backfill and reconciliation carrying two mutating actions, and
#: naming the pass rather than the actions is what withheld the backfill for
#: three consecutive cycles (``alpha-engine-config-I8179``: ``issue_number``
#: null on all 28 ledger rows, because the thing that fills it was inside the
#: gate). A gate named after a code path grows to cover whatever that path
#: later does; a gate named after an authority does not.
MUTATING_ACTIONS: tuple[str, ...] = (
    "issue_filing",           # files area:director-proposals issues into the fleet backlog
    "issue_reopen",           # reopens a closed issue whose cited metric reads unrecovered
    "carryover_escalation",   # puts a carried item on Brian's Decision Queue
)

#: The actions that RUN REGARDLESS of the verdict — non-mutating, or writing
#: only to the Director's own ledger. Withholding these buys no safety and costs
#: self-correction: with the backfill inside the gate, the longer the attestation
#: stays UNKNOWN the LESS able the system becomes to correct itself when it
#: clears. Declared as a named tuple, not left implicit, so the ran-regardless
#: set is emitted on every surface beside the withheld set — a list that appears
#: only when something stopped cannot be distinguished from a producer that
#: stopped emitting.
UNGATED_ACTIONS: tuple[str, ...] = (
    "issue_number_backfill",    # GET-only against GitHub; writes ledger rows in place
    "carryover_reconciliation", # annotates ledger rows against this week's card
)

#: Back-compatible name for the withheld set. Kept as an alias rather than
#: deleted because it is re-exported and read by the plan artifact's consumers;
#: it now means "the MUTATING actions", which is the only thing the verdict ever
#: had grounds to withhold.
GATED_ACTIONS: tuple[str, ...] = MUTATING_ACTIONS

#: The reason string recorded against every withheld action. A skip with no
#: recorded reason is the silent swallow the fleet's fail-loud rule forbids, and
#: on this path it would be indistinguishable from the feature being disabled.
WITHHELD_REASON_TEMPLATE = (
    "correctness verdict {verdict} — sf-pipeline-policy.md §2.3a: the Director "
    "may not MUTATE another system on numbers whose correctness was not "
    "established. Non-mutating work (ledger backfill, reconciliation used to "
    "annotate) is unaffected and ran. {detail}"
)


def read_card_verdict(card: Any) -> dict:
    """Normalize the report card's correctness verdict. Never raises.

    Returns a block carrying the verdict, the per-half ``as_of`` stamps, and a
    human-readable reason. Every degenerate input — ``None``, a non-mapping
    card, an absent ``attestation`` key, a non-mapping block, a verdict string
    outside the closed vocabulary — resolves to ``UNKNOWN`` with the specific
    cause recorded, because "the producer never ran" and "the producer said the
    numbers are wrong" are different findings and must not collapse.
    """
    if not isinstance(card, dict):
        return {
            "verdict": UNKNOWN,
            "as_of": {},
            "present": False,
            "reason": (
                "no report card was supplied to the verdict reader — the "
                "correctness guarantee cannot be established, so it is withheld."
            ),
        }

    block = card.get(ATTESTATION_KEY)
    if not isinstance(block, dict):
        return {
            "verdict": UNKNOWN,
            "as_of": {},
            "present": False,
            "reason": (
                f"the report card carries no `{ATTESTATION_KEY}` block. Either it "
                "was written before the verdict producer existed, or the producer "
                "did not run this cycle. Nothing checked whether the arithmetic "
                "behind this card's numbers is still right; this is an absence of "
                "evidence and is never read as a pass."
            ),
        }

    raw = block.get("verdict")
    # `isinstance` first: an unhashable value (a dict or list where a verdict
    # string belongs) raises on a set membership test, and a normalizer that can
    # raise is a normalizer that fails open at the call site that wrapped it.
    verdict = raw if isinstance(raw, str) and raw in _VALID_VERDICTS else UNKNOWN
    result = {
        "verdict": verdict,
        "as_of": block.get("as_of") if isinstance(block.get("as_of"), dict) else {},
        "present": True,
        "schema": block.get("schema"),
        "run_date": block.get("run_date"),
        "promotion_withheld": bool(block.get("promotion_withheld")),
    }
    # config#7199: `contamination` joins the arithmetic halves. It is surfaced
    # here as its own field, not folded into the combined verdict alone, because
    # the digest sentence "the numbers are right" and the sentence "the numbers
    # could not have seen the future" are different assurances to a reader.
    result["arithmetic_verdict"] = (
        block.get("arithmetic_verdict")
        if block.get("arithmetic_verdict") in _VALID_VERDICTS else UNKNOWN
    )
    result["contamination_verdict"] = (
        block.get("contamination_verdict")
        if block.get("contamination_verdict") in _VALID_VERDICTS else UNKNOWN
    )
    result["contamination_coverage_fraction"] = block.get(
        "contamination_coverage_fraction"
    )
    for half in ("evaluator", "backtester", "evaluator_stage", "contamination"):
        h = block.get(half)
        if isinstance(h, dict):
            result[f"{half}_verdict"] = (
                h.get("verdict") if h.get("verdict") in _VALID_VERDICTS else UNKNOWN
            )

    # Value comparison, not identity: `raw` comes from a JSON-parsed card and
    # `json.loads` does not intern scalar values, so `verdict is UNKNOWN` is
    # False whenever the card literally said "UNKNOWN". This branch happens to
    # reach the right outcome either way, but the identity form is the same
    # defect that made the pipeline-gate verdict permanently UNKNOWN
    # (alpha-engine-config-I7614) — it is removed here so the pattern does not
    # survive anywhere in the attestation read path.
    if verdict == UNKNOWN and raw != UNKNOWN:
        result["reason"] = (
            f"the card's attestation verdict {raw!r} is not one of "
            f"{sorted(_VALID_VERDICTS)} — treated as UNKNOWN, never as a pass. A "
            "verdict vocabulary that silently accepts new truthy strings is not a "
            "verdict."
        )
    else:
        result["reason"] = block.get("reason") or f"attestation verdict {verdict}."
    return result


def read_pipeline_gates(gate_state: Any, card: Any = None) -> dict:
    """The SF's pre-spend correctness-gate state, for the Director's surfaces.

    ``alpha-engine-config-I7282``. Two sources, in precedence order:

    1. ``gate_state`` — the ``Director`` Task's own payload block, threaded
       straight off the live execution record. Authoritative, because it
       describes THIS execution.
    2. the report card's ``pipeline_gates`` block — the same data as the
       ReportCard stage saw it. Used only when the SF sent nothing, so a
       dropped payload field degrades to a stale-but-real answer rather than to
       silence.

    Neither present resolves to ``UNKNOWN`` with the cause recorded, never to a
    pass (§2.3a rule 2).

    **Deliberately does NOT feed** :func:`actions_withheld`. The attestation
    verdict answers "are these numbers right", and acting on numbers that are
    not is what ``GATED_ACTIONS`` exists to prevent. The pre-spend gates answer
    a different question — "did the pipeline's own contract and pin invariants
    get checked before it spent" — and an unmeasured one of those does not make
    the week's metrics wrong. Wiring it into the gate would also, today, stop
    Director issue filing outright and permanently, because
    ``PipelineContractGate`` has been ``UNKNOWN`` on every production run since
    it existed (``alpha-engine-config-I7281``): a silent, indefinite outage
    introduced by an observability fix. It is rendered on every surface and
    gates nothing, until I7281 makes a MEASURED result reachable.
    """
    if isinstance(gate_state, dict):
        return read_gate_state(gate_state)
    if isinstance(card, dict):
        from_card = card.get(PIPELINE_GATES_KEY)
        if isinstance(from_card, dict) and from_card.get("verdict"):
            block = dict(from_card)
            block["source"] = "report_card"
            return block
    return read_gate_state(None)


def is_gated_action(action: str) -> bool:
    """Whether ``action`` is one the verdict may withhold. Raises on a name in
    neither tuple.

    Fail-loud on purpose. The two wrong defaults are both silent: returning
    ``False`` for an unrecognised name lets a new mutating action ship ungated
    the moment someone spells it differently, and returning ``True`` reinstates
    exactly the over-broad withholding this split removed. A caller naming an
    action nobody declared is a bug in the caller, and it says so.
    """
    if action in MUTATING_ACTIONS:
        return True
    if action in UNGATED_ACTIONS:
        return False
    raise ValueError(
        f"unknown Director action {action!r} — declare it in MUTATING_ACTIONS or "
        f"UNGATED_ACTIONS (director/verdict.py). Known: "
        f"{sorted(MUTATING_ACTIONS + UNGATED_ACTIONS)}"
    )


def actions_withheld(verdict_block: dict, action: str | None = None) -> bool:
    """True when the verdict withholds MUTATING AUTHORITY.

    Not "true when the Director must do less". The subject of this gate is the
    authority an action exercises, never the run's overall quality — Brian
    ruling 2026-08-22 (``alpha-engine-config-I8187``), ``sf-pipeline-policy.md``
    §2.3a. A non-PASS verdict says the numbers were not established correct; the
    only thing that follows is that the Director may not turn those numbers into
    durable state in somebody else's system. It does not follow that it may not
    read, annotate, or repair its own ledger — and withholding those is what made
    the outage compound rather than merely persist.

    ``action=None`` answers the class question: may the Director exercise
    mutating authority at all this cycle? That is the form the plan artifact,
    the email banner and the SF stage output all want, and its truth value is
    unchanged from before this split.

    ``action=<name>`` answers it for one declared action, and is the form a call
    site should use: ``UNGATED_ACTIONS`` members answer ``False`` under every
    verdict, so a caller cannot accidentally re-gate a read by passing the
    verdict block around.

    Deliberately expressed as ``not verdict_is_pass(...)`` rather than as a test
    against ``FAIL``: ``UNKNOWN`` withholds exactly as hard as ``FAIL`` does.
    They differ in what they say about the numbers — ``FAIL`` is evidence the
    arithmetic moved, ``UNKNOWN`` is absence of evidence either way — and not at
    all in what the Director is permitted to do next.
    """
    if action is not None and not is_gated_action(action):
        return False
    return not verdict_is_pass((verdict_block or {}).get("verdict"))


def withheld_reason(verdict_block: dict) -> str:
    """The reason recorded against each withheld action, and logged once."""
    vb = verdict_block or {}
    return WITHHELD_REASON_TEMPLATE.format(
        verdict=vb.get("verdict", UNKNOWN),
        detail=(vb.get("reason") or "").strip(),
    ).strip()


def withheld_summary(verdict_block: dict) -> dict:
    """The keys the Director's SF output carries so the withholding is visible
    in the execution history, not only in the artifact.

    A stage that quietly did less than usual and returned ``status: ok`` is the
    same blindness one layer up.
    """
    vb = verdict_block or {}
    withheld = actions_withheld(vb)
    summary = {
        "correctness_verdict": vb.get("verdict", UNKNOWN),
        "correctness_verdict_present": bool(vb.get("present")),
        "correctness_as_of": vb.get("as_of") or {},
        "director_actions_withheld": withheld,
    }
    # alpha-engine-config-I8187: the ran-regardless set is emitted in BOTH
    # verdict states, and unconditionally. A reader who sees only what stopped
    # cannot tell a narrowed gate from a gate that stopped being applied, and
    # the whole content of this ruling is which actions are on which side.
    summary["director_actions_ungated_list"] = list(UNGATED_ACTIONS)
    if withheld:
        summary["director_actions_withheld_list"] = list(MUTATING_ACTIONS)
        summary["correctness_verdict_reason"] = vb.get("reason")
    # alpha-engine-config-I7282 — §2.3a rule 3 on the SF stage output. Emitted
    # in BOTH polarities (a key that appears only when something is wrong cannot
    # be distinguished from a producer that stopped emitting).
    gates = vb.get(PIPELINE_GATES_KEY) or {}
    if gates:
        summary["pipeline_gates_verdict"] = gates.get("verdict", UNKNOWN)
        summary["degraded_pipeline_gates"] = gates_unmeasured(gates)
        summary["pipeline_gates_unmeasured"] = list(gates.get("unmeasured") or [])
    return summary


def stamp_plan_artifact(plan: Any, verdict_block: dict) -> bytes:
    """Serialize ``director/{run_date}/action_plan.json`` carrying the verdict.

    The verdict is stamped onto the serialized body rather than added to
    :class:`director.schema.DirectorWeeklyActionPlan`, because that model is the
    **LLM's structured-output schema**: a field added there becomes a field the
    model is asked to produce, and a correctness verdict generated by the thing
    being verified is not a verdict. The plan model already declares
    ``extra="allow"``, so a consumer that round-trips the artifact through it
    keeps the block.

    §2.3a rule 3 — every surface presenting the run's results carries the verdict
    state. The action plan is the artifact the console Director page renders, so
    it is one of those surfaces.
    """
    if hasattr(plan, "model_dump_json"):
        body = json.loads(plan.model_dump_json())
    elif isinstance(plan, dict):
        body = dict(plan)
    else:  # pragma: no cover — defended, not expected
        raise TypeError(f"cannot serialize a plan of type {type(plan).__name__}")

    vb = verdict_block or {}
    body[ATTESTATION_KEY] = vb
    body["advisory_unverified"] = actions_withheld(vb)
    if body["advisory_unverified"]:
        body["advisory_unverified_reason"] = withheld_reason(vb)
        body["actions_withheld"] = list(MUTATING_ACTIONS)
        # I8187: named beside the withheld set, on the artifact the console
        # Director page renders. "loop_verification" used to appear in the list
        # above and no longer does — not because it stopped being gated, but
        # because it was never one authority. This key is what says so to a
        # reader comparing this artifact to a pre-ruling one.
        body["actions_ran_regardless"] = list(UNGATED_ACTIONS)
    return json.dumps(body, indent=2).encode("utf-8")


def log_verdict(verdict_block: dict, run_date: str) -> None:
    """Emit the one log line that makes the decision reconstructable later.

    ``principles.md`` §2.1: after this runs unattended, someone must be able to
    reconstruct why it did what it did from durable artifacts alone.
    """
    vb = verdict_block or {}
    if actions_withheld(vb):
        logger.error(
            "Director: correctness verdict %s for %s — WITHHOLDING mutating "
            "authority (%s); still running (%s). %s",
            vb.get("verdict", UNKNOWN), run_date, ", ".join(MUTATING_ACTIONS),
            ", ".join(UNGATED_ACTIONS), vb.get("reason", ""),
        )
    else:
        logger.info(
            "Director: correctness verdict PASS for %s — acting authority intact.",
            run_date,
        )
