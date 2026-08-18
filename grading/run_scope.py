"""run_scope.py — the Report Card's consumer half for the run's own scope.

Tracked as ``alpha-engine-config-I7620``.

WHY THIS EXISTS
---------------
The card renders grades. It has never rendered the DENOMINATOR those grades were
computed over, and the denominator moves: the weekly pipeline carries 29 skip
gates, and an operator flipping one changes which producers ran without changing
anything the card says.

That is not hypothetical. ``skip_parity: true`` has been set on the live
``alpha-engine-saturday`` EventBridge target since 2026-08-13 by a recorded
ruling (``config-I7309``). The 2026-08-14 Director plan reported the resulting
absence as *"contamination attestation absent … the producer never ran this
cycle"* and withheld ``issue_filing`` and ``loop_verification`` for the cycle.
The producer did not never run. It was switched off on purpose, and no surface
could tell the difference.

``nousergon-data``'s ``RunScope`` stage now derives that fact every run and
writes ``backtest/{date}/run_scope.json``. This module is the reading half.

THE RULE THAT MATTERS
---------------------
**An absent or unreadable scope block means "grade nothing", never "everything
ran".** A card that quietly grades the full stage list against a run that
dispatched three of them is worse than a card that says it does not know: the
first is confidently wrong and the second is merely uninformative. So every
degenerate input here — absent artifact, non-mapping block, a disposition
outside the closed vocabulary, a producer that grew a fifth value — resolves to
``UNKNOWN`` with an empty graded set and the cause recorded.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not move the card's ``status``. Same reasoning as ``pipeline_gates.py``:
a field that is permanently amber is a field nobody reads, and the scope is
legitimately narrow on every partial rerun. The scope is rendered BESIDE the
grade, which is the thing that was missing — not folded INTO it.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: The card key carrying this block. Cross-repo contract with
#: ``nousergon-data infrastructure/lambdas/weekly-run-scope/run_scope.py``,
#: whose emitted body is stamped ``run_scope-1.0.0``.
RUN_SCOPE_KEY = "run_scope"

MEASURED = "MEASURED"
UNKNOWN = "UNKNOWN"

DISABLED = "DISABLED"
ENABLED_COMPLETED = "ENABLED_COMPLETED"
ENABLED_FAILED = "ENABLED_FAILED"
NOT_REACHED = "NOT_REACHED"

#: The producer's closed vocabulary, restated here because this is the boundary.
#: A value outside it is never graded — a producer that starts emitting
#: ``"PROBABLY_FINE"`` must withhold, not slip through a truthiness test.
DISPOSITIONS = frozenset({DISABLED, ENABLED_COMPLETED, ENABLED_FAILED, NOT_REACHED})

#: The two that were DISPATCHED. Grading follows dispatch, not success, so a
#: stage cannot silently drop out of the denominator by crashing.
GRADED_DISPOSITIONS = frozenset({ENABLED_COMPLETED, ENABLED_FAILED})


def read_run_scope(block: Any) -> dict:
    """Normalize the run-scope artifact for the card. Never raises.

    Returns ``verdict`` (MEASURED / UNKNOWN), ``graded_stages``, ``disabled``,
    ``not_reached``, ``failed``, the counts, and a human-readable ``statement``.
    """
    if not isinstance(block, dict):
        return _unknown(
            "no run_scope artifact was supplied to the card — which stages this "
            "run dispatched is not established, so nothing on it is graded "
            "against a known denominator."
        )

    if block.get("degraded"):
        return _unknown(
            "the run-scope producer ran and could not derive the scope "
            f"({block.get('degraded_reason', 'no cause recorded')}). This cycle "
            "is unmeasured, not narrow."
        )

    stages = block.get("stages")
    if not isinstance(stages, dict) or not stages:
        return _unknown(
            "the run_scope artifact carries no stage rows — an empty scope is "
            "an absence of evidence and is never read as a complete run."
        )

    buckets: dict[str, list[str]] = {d: [] for d in sorted(DISPOSITIONS)}
    unrecognised: list[str] = []
    for name, row in sorted(stages.items()):
        disposition = row.get("disposition") if isinstance(row, dict) else None
        if disposition in DISPOSITIONS:
            buckets[disposition].append(name)
        else:
            # Recorded, never graded, and never silently dropped: a stage the
            # card cannot classify is a stage the card cannot claim to have
            # covered.
            unrecognised.append(name)

    graded = sorted(
        name for d in GRADED_DISPOSITIONS for name in buckets[d]
    )
    total = len(stages)
    result = {
        "verdict": MEASURED,
        "present": True,
        "run_date": block.get("run_date"),
        "graded_stages": graded,
        "disabled_stages": buckets[DISABLED],
        "failed_stages": buckets[ENABLED_FAILED],
        "not_reached_stages": buckets[NOT_REACHED],
        "unrecognised_stages": unrecognised,
        "graded_count": len(graded),
        "stage_count": total,
        "disabled_by": sorted({
            row.get("disabled_by") for row in stages.values()
            if isinstance(row, dict) and row.get("disabled_by")
        }),
    }
    result["statement"] = _statement(result)
    return result


def _unknown(reason: str) -> dict:
    return {
        "verdict": UNKNOWN,
        "present": False,
        "run_date": None,
        "graded_stages": [],
        "disabled_stages": [],
        "failed_stages": [],
        "not_reached_stages": [],
        "unrecognised_stages": [],
        "graded_count": 0,
        "stage_count": 0,
        "disabled_by": [],
        "statement": f"SCOPE UNKNOWN — {reason}",
    }


def _statement(block: dict) -> str:
    """The sentence that goes beside the grade.

    "GREEN" over an unstated denominator is not a falsifiable claim. Every
    surface in this fleet that has gone quietly green did it by shrinking its
    own scope unannounced — most recently the 2026-08-16 execution that
    terminated SUCCEEDED having dispatched 3 of 29 stages.
    """
    parts = [
        f"{block['graded_count']} of {block['stage_count']} gated stages "
        "dispatched and graded"
    ]
    if block["disabled_stages"]:
        flags = ", ".join(block["disabled_by"]) or "operator flag"
        parts.append(f"{len(block['disabled_stages'])} disabled by {flags}")
    if block["not_reached_stages"]:
        parts.append(f"{len(block['not_reached_stages'])} never reached")
    if block["failed_stages"]:
        parts.append(
            f"{len(block['failed_stages'])} dispatched and did NOT complete "
            f"({', '.join(block['failed_stages'])})"
        )
    if block["unrecognised_stages"]:
        parts.append(
            f"{len(block['unrecognised_stages'])} carrying a disposition this "
            "card does not recognise, and therefore not graded"
        )
    return "; ".join(parts) + "."


def scope_unknown(block: Any) -> bool:
    """True when the card may not claim a denominator.

    Expressed as ``!= MEASURED`` rather than ``== UNKNOWN`` so a future third
    verdict withholds rather than passing. Value comparison, not identity —
    this block crosses a JSON boundary, and the identity form is what made the
    pipeline-gate verdict permanently UNKNOWN (``config-I7614``).
    """
    return (block or {}).get("verdict") != MEASURED


def log_run_scope(block: dict) -> None:
    """One line, both polarities. A surface that logs only the bad case cannot
    be distinguished from a producer that stopped running."""
    logger.info("run scope: %s", block.get("statement", "(no statement)"))
