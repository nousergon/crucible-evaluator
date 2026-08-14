"""pipeline_gates.py — the Report Card's consumer half for the weekly SF's
pre-spend correctness gates.

WHY THIS EXISTS
---------------
``sf-pipeline-policy.md`` §2.3a rule 3: *every surface presenting the run's
results carries the verdict state. A report card, dashboard or notification
rendering the run's numbers without saying whether the correctness check ran is
asserting a guarantee nobody established.*

The weekly pipeline runs two pre-spend correctness gates before it spends a
cent — ``LibPinDriftCheck`` (are the cross-repo library pins coherent?) and
``PipelineContractCheck`` (does the declared pipeline contract still validate?)
— plus four fail-open degradation families that record when a stage produced its
output without its guarantee. Until ``alpha-engine-config-I7282`` the
``ReportCard`` Lambda's SF payload was ``{date, dry_run, snapshot}`` and the
``Director``'s was ``{date, dry_run}``. Neither carried any of it. The card
therefore rendered the week's numbers identically whether the gates passed,
failed, or never ran.

That was not hypothetical. ``PipelineContractGate`` fetches
``private-docs/PIPELINE_CONTRACT.yaml`` from ``raw.githubusercontent.com``
unauthenticated, and ``nousergon/alpha-engine-config`` is a PRIVATE repository:
the URL returns HTTP 404, measured live 2026-08-13. That gate has never measured
anything on any production run (``alpha-engine-config-I7281``), and
``LibPinDriftCheck`` has been unmeasured since crucible-predictor#422
(``alpha-engine-config-I7171``). Every Report Card ever issued was produced under
two unrun correctness gates and said so nowhere.

THE VOCABULARY IS THE PRODUCER'S
--------------------------------
``MEASURED`` / ``UNKNOWN`` are ``crucible-predictor``'s, minted in
``inference/lib_pin_drift.py`` and ``inference/pipeline_contract_check.py``
(crucible-predictor#489, ``alpha-engine-config-I7277``). An ``UNKNOWN`` payload
deliberately OMITS its verdict key (``has_drift`` / ``has_violation``) and its
evidence list, so a consumer keyed on either sees an absent key rather than an
empty-and-clean-looking one. This module never reconstructs a verdict from those
keys; it reads ``status`` and nothing else, which is the only field that
distinguishes "checked, nothing found" from "could not check".

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not fail the card, and it does not move the card's ``status`` to
``partial``. Two different reasons, both load-bearing:

* **Not failing** is §2.3a's own instruction — *withholding the guarantee is not
  the same as failing the pipeline*. Every number is still computed, every tile
  still renders; what is withheld is the claim that anything checked the inputs.
* **Not moving ``status``** because ``status`` is the card's own build quality
  (``ok`` / ``partial`` / ``insufficient_data``), already owned by
  ``degraded_attestation``. The pipeline-contract gate has been ``UNKNOWN`` on
  *every* run since it existed, so folding it into ``status`` would paint that
  field permanently yellow and make "this build is incomplete" indistinguishable
  from "an upstream probe is broken". A surface that is always amber is a
  surface nobody reads, which is the same blindness §2.3a exists to end, one
  layer over. The honesty is carried instead in a field of its own — present in
  both polarities, with a sentence a human reads — and echoed into the Director
  digest, which is what Brian actually opens.

Nothing here raises. Every degenerate input resolves to ``UNKNOWN`` with the
specific cause recorded, because "the SF did not send the gate state" and "the
gate ran and could not measure" are different findings and must not collapse.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Version of the SF -> evaluator ``gate_state`` payload contract. The schema is
#: ``grading/contracts/sf_gate_state.v1.schema.json``; the SoT copy lives at
#: ``nousergon-data/infrastructure/contracts/sf_gate_state.v1.schema.json`` and
#: both are pinned by digest in each repo's contract test.
SCHEMA = "sf_gate_state-1.0.0"
SCHEMA_VERSION = 1

#: Where the normalized block sits on the report card, the Director's verdict
#: block and the stamped action plan. One name on all three so a reader who finds
#: it on one surface knows what to grep for on the others.
PIPELINE_GATES_KEY = "pipeline_gates"

#: The producer's closed vocabulary (crucible-predictor, config-I7277). Anything
#: outside it is UNKNOWN — a status vocabulary that silently accepts new truthy
#: strings is not a status.
MEASURED = "MEASURED"
UNKNOWN = "UNKNOWN"
_VALID_STATUSES = frozenset({MEASURED, UNKNOWN})

#: The pre-spend correctness gates, keyed as the SF payload carries them, mapped
#: to the human name the statement uses. Ordered — the statement lists them in
#: pipeline order, which is the order an operator debugs them in.
GATE_LABELS: dict[str, str] = {
    "lib_pin_drift": "LibPinDriftCheck (cross-repo library pin coherence)",
    "pipeline_contract": "PipelineContractCheck (declared pipeline contract validity)",
}

#: The SF's fail-open degradation families, in the order ``CheckGateDegradedNotify``
#: evaluates them. Each is an SF-controlled boolean seeded ``false`` at
#: ``InitializeInput`` and set ``true`` by exactly one Pass state — so it is
#: present in BOTH polarities on the execution record, and an absent one here
#: means the SF did not send it, never that it was false.
DEGRADED_FAMILY_LABELS: dict[str, str] = {
    "gate_degraded": "pre-spend gates (LibPinDriftCheck/PipelineContractCheck/AcquireMutex)",
    "health_check_degraded": "tail health checks (SaturdayHealthCheck/WeeklySubstrateHealthCheck)",
    "parity_degraded": "parity verdict (pit_parity/replay)",
    "research_predictor_degraded": "an internal ResearchPredictorParallel fail-open",
}


def _statement(verdict: str, unmeasured: list[tuple[str, str]], degraded: list[str],
               families_unreported: list[str], n_gates: int, top_reason: str) -> str:
    """The one sentence a human reads. Present in both polarities.

    Deliberately phrased ``NOT VERIFIED`` rather than ``FAILED``: a gate that
    could not run is an absence of evidence, and a reader who cannot tell it
    apart from a detected defect will eventually discard the card on a week when
    the numbers were fine. The distinction between the two is the entire content
    of §2.3a rule 2, and it survives only if the words survive.
    """
    # alpha-engine-config-I7312: gated on `not degraded` as well as the verdict,
    # deliberately belt-and-braces. The verdict now folds `degraded` in, so this
    # branch is unreachable with a fired family — but the sentence ASSERTS "no
    # fail-open degradation", and a claim that can only be true because of a
    # computation two functions away is how this was wrong in the first place.
    # Read the evidence here, not the summary of it.
    if verdict == MEASURED and not degraded:
        head = (
            f"VERIFIED — all {n_gates} pre-spend correctness gates reported "
            "MEASURED this cycle, and the Step Function recorded no fail-open "
            "degradation."
        )
        return head

    if unmeasured:
        named = "; ".join(
            f"{GATE_LABELS.get(key, key)} — {reason}" for key, reason in unmeasured
        )
        head = (
            f"NOT VERIFIED — {len(unmeasured)} of {n_gates} pre-spend correctness "
            f"gates did not run this cycle: {named}."
        )
    else:
        head = f"NOT VERIFIED — {top_reason}"
    head += (
        " Every number on this card was computed normally; what is missing is the "
        "check that says the inputs behind them were sound. Treat the grades as "
        "UNATTESTED, not as wrong."
    )
    if families_unreported and unmeasured:
        head += (
            " The Step Function also reported no value for "
            + ", ".join(families_unreported)
            + " — unreported is not false."
        )
    if degraded:
        head += (
            " Fail-open degradation recorded on this run: "
            + "; ".join(DEGRADED_FAMILY_LABELS.get(d, d) for d in degraded)
            + "."
        )
    return head


def read_gate_state(gate_state: Any) -> dict:
    """Normalize the SF-supplied ``gate_state`` payload. Never raises.

    Returns a block carrying, in both polarities:

    ``verdict``
        ``MEASURED`` only when ALL THREE hold: every gate in
        :data:`GATE_LABELS` reported ``MEASURED``, every family in
        :data:`DEGRADED_FAMILY_LABELS` was reported, and none of them fired.
        ``UNKNOWN`` otherwise — including when the SF sent nothing at all,
        which is the case for every Report Card written before
        ``alpha-engine-config-I7282`` was in.

        The third clause is ``alpha-engine-config-I7312``. A fired family means
        something on this run failed OPEN, so the attestation is withheld —
        which is not the same as saying the grades are wrong, and the statement
        keeps that distinction.
    ``gates``
        per-gate ``{status, reason}``, one entry per gate, always all of them.
    ``unmeasured``
        the gate keys whose status is not ``MEASURED`` — empty list on a clean
        run, never absent.
    ``degraded_families``
        the SF fail-open families that fired.
    ``statement``
        the human sentence; see :func:`_statement`.
    """
    present = isinstance(gate_state, dict)
    if not present:
        gate_state = {}
        top_reason = (
            "the Step Function supplied no gate_state to this stage — the "
            "correctness gates' verdicts never reached the surface that renders "
            "their run's numbers (sf-pipeline-policy.md §2.3a rule 3). This is an "
            "absence of evidence and is never read as a pass."
        )
    else:
        top_reason = ""

    gates: dict[str, dict] = {}
    unmeasured: list[tuple[str, str]] = []
    for key in GATE_LABELS:
        raw = gate_state.get(key)
        if not isinstance(raw, dict):
            status, reason = UNKNOWN, (
                "absent from the gate_state payload" if present
                else "no gate_state supplied"
            )
        else:
            status_raw = raw.get("status")
            status = status_raw if status_raw in _VALID_STATUSES else UNKNOWN
            if status is UNKNOWN and status_raw not in (None, UNKNOWN):
                reason = (
                    f"unrecognised status {status_raw!r} — treated as UNKNOWN, "
                    "never as a pass"
                )
            else:
                reason = str(raw.get("reason") or "gate reported no measurement")
        gates[key] = {"status": status, "reason": reason if status is UNKNOWN else None}
        if status is not MEASURED:
            unmeasured.append((key, reason))

    degraded = [
        fam for fam in DEGRADED_FAMILY_LABELS
        if gate_state.get(fam) is True
    ]
    # A family the SF did not send is NOT false — it is unreported. Recorded
    # separately so "no degradation" and "nobody said" stay distinguishable.
    families_unreported = [
        fam for fam in DEGRADED_FAMILY_LABELS
        if not isinstance(gate_state.get(fam), bool)
    ]

    # alpha-engine-config-I7312: `degraded` — the families that actually FIRED
    # — belongs in the verdict, not only in the narrative below it.
    #
    # The MEASURED statement asserts TWO things: that every gate reported
    # MEASURED, *and* that the Step Function recorded no fail-open degradation.
    # Computing the verdict from `unmeasured` + `families_unreported` alone made
    # the second half unfalsifiable — and because `_statement` returns early on
    # MEASURED, its `if degraded:` clause was unreachable on exactly the runs
    # that needed it. A run rendered "VERIFIED ... no fail-open degradation"
    # while this same block carried `degraded_families: ["gate_degraded"]`.
    #
    # The combination is reachable without any probe going UNKNOWN, because
    # THREE SF states write `$.gate_degraded=true` that GATE_LABELS does not
    # cover: EvaluatorGateDegraded, EvaluatorDirectorGateDegraded, and
    # SetMutexAcquireDegradedFlag. It stayed hidden only because
    # `pipeline_contract` had been UNKNOWN on every production run to date
    # (alpha-engine-config-I7281), which forced the verdict UNKNOWN anyway.
    #
    # A fired family is not evidence the grades are WRONG — the non-MEASURED
    # branch says exactly that ("UNATTESTED, not wrong"), which is why folding
    # it in here is safe: it withholds the attestation without claiming a defect.
    verdict = (
        MEASURED
        if not unmeasured and not families_unreported and not degraded
        else UNKNOWN
    )
    if families_unreported and not unmeasured:
        top_reason = top_reason or (
            "the SF reported no value for the fail-open degradation "
            f"{'families' if len(families_unreported) > 1 else 'family'} "
            f"{', '.join(families_unreported)} — unreported is not false."
        )
    if degraded and not unmeasured:
        # Without this the head renders "NOT VERIFIED — " with an empty reason:
        # this is the one path where the verdict is UNKNOWN and NEITHER a gate
        # nor an unreported family explains why.
        top_reason = top_reason or (
            f"{len(degraded)} fail-open "
            f"{'degradations were' if len(degraded) > 1 else 'degradation was'} "
            "recorded on this run, so the run's pre-spend protection was "
            "incomplete even though every gate that DID report reported MEASURED."
        )

    block = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "verdict": verdict,
        "present": present,
        "gates": gates,
        "unmeasured": [k for k, _ in unmeasured],
        "degraded_families": degraded,
        "families_unreported": families_unreported,
        "reason": top_reason,
        "statement": _statement(
            verdict, unmeasured, degraded, families_unreported,
            len(GATE_LABELS), top_reason,
        ),
    }
    return block


def gates_unmeasured(block: Any) -> bool:
    """True when the card may not present its numbers as gate-verified.

    Expressed as ``!= MEASURED`` rather than ``== UNKNOWN`` so a future third
    status value withholds by default instead of passing by omission.
    """
    return (block or {}).get("verdict") != MEASURED


def log_gate_state(block: dict, run_date: str) -> None:
    """The one log line that makes the decision reconstructable later
    (``principles.md`` §2.1)."""
    if gates_unmeasured(block):
        logger.error(
            "PIPELINE GATES %s for %s — %s",
            (block or {}).get("verdict", UNKNOWN), run_date,
            (block or {}).get("statement", ""),
        )
    else:
        logger.info("Pipeline gates MEASURED for %s.", run_date)
