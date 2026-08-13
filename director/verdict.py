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

WHAT IT DOES
------------
``read_card_verdict`` normalizes the card's verdict onto the closed vocabulary,
and ``actions_withheld`` says whether the Director may exercise its acting
authority this cycle.

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
from grading.attestation import FAIL, PASS, UNKNOWN, verdict_is_pass

logger = logging.getLogger(__name__)

#: The card key carrying the §2.3a block. Cross-repo contract with
#: ``crucible-evaluator grading/attestation.py::build_run_attestation``, whose
#: emitted body is stamped ``report_card_attestation-1.0.0``.
ATTESTATION_KEY = "attestation"

_VALID_VERDICTS = frozenset({PASS, FAIL, UNKNOWN})

#: The Director actions gated on the verdict. Each creates state that OUTLIVES
#: this cycle in a system other than this one, which is the test for membership
#: — rendering is not gated (a marked page is more useful than a blank one),
#: mutating a shared tracker is.
GATED_ACTIONS: tuple[str, ...] = ("issue_filing", "loop_verification")

#: The reason string recorded against every withheld action. A skip with no
#: recorded reason is the silent swallow the fleet's fail-loud rule forbids, and
#: on this path it would be indistinguishable from the feature being disabled.
WITHHELD_REASON_TEMPLATE = (
    "correctness verdict {verdict} — sf-pipeline-policy.md §2.3a: the Director "
    "may not act on numbers whose correctness was not established. {detail}"
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
    for half in ("evaluator", "backtester", "evaluator_stage"):
        h = block.get(half)
        if isinstance(h, dict):
            result[f"{half}_verdict"] = (
                h.get("verdict") if h.get("verdict") in _VALID_VERDICTS else UNKNOWN
            )

    if verdict is UNKNOWN and raw != UNKNOWN:
        result["reason"] = (
            f"the card's attestation verdict {raw!r} is not one of "
            f"{sorted(_VALID_VERDICTS)} — treated as UNKNOWN, never as a pass. A "
            "verdict vocabulary that silently accepts new truthy strings is not a "
            "verdict."
        )
    else:
        result["reason"] = block.get("reason") or f"attestation verdict {verdict}."
    return result


def actions_withheld(verdict_block: dict) -> bool:
    """True when the Director must not exercise its acting authority.

    Deliberately expressed as ``not verdict_is_pass(...)`` rather than as a test
    against ``FAIL``: ``UNKNOWN`` withholds exactly as hard as ``FAIL`` does.
    They differ in what they say about the numbers — ``FAIL`` is evidence the
    arithmetic moved, ``UNKNOWN`` is absence of evidence either way — and not at
    all in what the Director is permitted to do next.
    """
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
    if withheld:
        summary["director_actions_withheld_list"] = list(GATED_ACTIONS)
        summary["correctness_verdict_reason"] = vb.get("reason")
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
        body["actions_withheld"] = list(GATED_ACTIONS)
    return json.dumps(body, indent=2).encode("utf-8")


def log_verdict(verdict_block: dict, run_date: str) -> None:
    """Emit the one log line that makes the decision reconstructable later.

    ``principles.md`` §2.1: after this runs unattended, someone must be able to
    reconstruct why it did what it did from durable artifacts alone.
    """
    vb = verdict_block or {}
    if actions_withheld(vb):
        logger.error(
            "Director: correctness verdict %s for %s — WITHHOLDING %s. %s",
            vb.get("verdict", UNKNOWN), run_date, ", ".join(GATED_ACTIONS),
            vb.get("reason", ""),
        )
    else:
        logger.info(
            "Director: correctness verdict PASS for %s — acting authority intact.",
            run_date,
        )
