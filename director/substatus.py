"""substatus.py — the Director stage's status is the WORST of its sub-results.

``sf-pipeline-policy.md`` §2.3b, clause
``SFP-2.3b-stage-status-is-the-worst-substatus``: *a stage that returns named
sub-results returns a status no better than the worst of them; a sub-result
reporting an error is a stage degradation; and a sub-result status that is
neither a declared pass nor a declared error is reported ``unclassified``,
never defaulted to pass.*

**The measured instance this module closes** (``alpha-engine-config-I11299``).
The Director returns one flat summary carrying four independent legs, each with
its own status string, under a single enclosing ``status``. On the 2026-09-19
scheduled weekly run AND on ``watch-rerun-2026-09-18-1`` and ``-3`` it returned
``status: "ok"`` over ``retro: "error"`` — the retro judge had REFUSED to
publish a grade because it was served the same upstream model that produced the
plan it was grading (self-grading bias; root cause
``alpha-engine-config-I8202``, registry invariant 15 compares entry ids rather
than served models). Two of those three executions terminated
``ExecutionSucceeded`` with ``degraded_summary.degraded: false``. The weekly
cycle has therefore published no RetroGrade for at least two consecutive cycles
with nothing anywhere going red — ``alpha-engine-config-I10169`` one repo over.

**Why the fan-out is carried a SECOND time as a dict.** The conformance probe
for the clause (``nous-ergon-ops/scripts/sf_substatus_honesty.py``) descends
only into sub-results that are DICTS carrying their own ``status`` key
(``_scan_node``: ``if key in EXCLUDED_SUBRESULT_KEYS or not isinstance(value,
dict): continue``). Every Director leg is a BARE STRING on the flat summary, so
the probe read the 2026-09-19 scheduled execution and reported *0 stage(s)
reported a pass over an errored sub-result, across 88 stage result(s)* —
measured 2026-09-21, blind to the very instance the clause was amended for.
:func:`build_sub_statuses` emits the same facts as a dict of dicts under
``sub_statuses`` so the existing probe sees them with no change to the probe,
while every existing consumer of the flat ``retro`` / ``director_loop`` /
``director_issues`` / ``deploy_success`` strings (the digest emailer, the
console Director row, the tests) keeps reading exactly what it read before.
Additive, never a rename.

**The vocabulary, closed at both ends.**

``PASS_SUB_STATUSES`` are the outcomes in which the leg did what it was asked
to do, INCLUDING the two shapes of "correctly did less":

* ``skipped`` — the leg had nothing to do (no prior plan on the first cycle) or
  declined a call it could not finish inside the invocation budget. §2.3b's own
  carve-out, and ``alpha-engine-config-I11299`` names it explicitly: *do not
  widen the degraded flag so far that a legitimately-skipped retro marks a run
  degraded — ``error`` and ``skipped`` are different sub-statuses.*
* ``disabled`` — the channel is switched off by configuration.
* ``withheld`` / ``mutations_withheld`` — §2.3a withholding. The leg was gated
  OFF by a correctness verdict that is itself already on this summary
  (``withheld_summary``), already threaded to the SF, and already the subject
  of its own degraded family. Counting it a second time here would degrade
  every run whose attestation is UNKNOWN for a reason §2.3a has already
  reported, which is double-counting, not honesty.

``ERROR_SUB_STATUSES`` are the outcomes in which work the stage was asked for
did not happen and the leg said so — ``error`` (including a judge's
correctness REFUSAL, which is a correct verdict about an incorrect setup and
still means no RetroGrade exists) and ``partial`` (the loop-verification pass
ran against a ledger it could not fully resolve).

Anything else is ``unclassified`` and degrades. A reader that treats an
unrecognised status as healthy goes quiet the first time a producer invents a
word for failure.
"""

from __future__ import annotations

from typing import Any

#: The Director's named sub-results, in the order they are reported. Each is a
#: bare status string on the flat summary; the key here is also the key used
#: under ``sub_statuses``.
#:
#: ``pipeline_gates_verdict`` is deliberately NOT a member. It is not a leg of
#: this stage's own work — it is the SF's pre-spend gate verdict, threaded IN
#: as ``gate_state``, carried here for transparency, and already owned by
#: ``$.gate_degraded`` and its own ``Mark*Degraded`` family in the state
#: machine. Folding it in would make the Director report a degradation that
#: another stage already reports (``alpha-engine-config-I11112`` owns the
#: unmeasured-gate problem itself).
SUB_RESULT_KEYS: tuple[str, ...] = (
    "retro",
    "director_loop",
    "director_issues",
    "deploy_success",
)

#: Detail field carrying the leg's own explanation, per sub-result key. Read
#: onto ``sub_statuses[key]["detail"]`` so the SF output names WHY without a
#: consumer having to know which flat field each leg uses.
_DETAIL_FIELDS: dict[str, tuple[str, ...]] = {
    "retro": ("retro_error", "retro_reason"),
    "director_loop": ("director_loop_error", "director_loop_reason"),
    "director_issues": ("director_issues_error", "director_issues_reason"),
    "deploy_success": ("deploy_success_error", "deploy_success_reason"),
}

PASS_SUB_STATUSES: frozenset[str] = frozenset(
    {"ok", "skipped", "disabled", "withheld", "mutations_withheld"}
)

ERROR_SUB_STATUSES: frozenset[str] = frozenset({"error", "partial"})

#: What a sub-result status outside BOTH vocabularies is reported as — never a
#: pass (§2.3b, "the vocabulary is closed at both ends").
UNCLASSIFIED = "unclassified"

#: The stage statuses this module produces. ``degraded`` is deliberately NOT a
#: member of the conformance probe's ``PASS_STATUSES``, which is what makes the
#: honest stage stop being reported as a violation.
STAGE_OK = "ok"
STAGE_DEGRADED = "degraded"

#: Longest detail string carried onto the SF output. The underlying text stays
#: whole on the summary's own ``*_error`` field and in the Lambda log; this is
#: a bounded projection for a surface a Choice state reads.
_MAX_DETAIL_CHARS = 400


def classify(status: Any) -> str:
    """Map one sub-result status onto ``pass`` / ``error`` / ``unclassified``.

    A missing leg (``None``) is ``pass``: the leg did not run at all in this
    invocation, which is the ``disabled``/``dry_run`` shape, not a failure.
    A non-string is ``unclassified`` — a leg that reported a number or a dict
    where a status belongs has told us nothing, and nothing is not health.
    """
    if status is None:
        return "pass"
    if not isinstance(status, str):
        return UNCLASSIFIED
    if status in PASS_SUB_STATUSES:
        return "pass"
    if status in ERROR_SUB_STATUSES:
        return "error"
    return UNCLASSIFIED


def _detail_for(summary: dict, key: str) -> str | None:
    for field in _DETAIL_FIELDS.get(key, ()):
        value = summary.get(field)
        if isinstance(value, str) and value.strip():
            return value[:_MAX_DETAIL_CHARS]
    return None


def build_sub_statuses(summary: dict) -> dict[str, dict]:
    """The Director's named legs as a dict of dicts the §2.3b probe can read.

    Only legs PRESENT on ``summary`` are emitted — a summary from a path that
    never ran the retro (the deploy-drift probe, the dry run, the disabled
    no-op) declares three legs it does not have if absence is filled in, and a
    fabricated ``ok`` is exactly the defect this module exists to remove.
    """
    out: dict[str, dict] = {}
    for key in SUB_RESULT_KEYS:
        if key not in summary:
            continue
        raw = summary.get(key)
        verdict = classify(raw)
        entry: dict[str, Any] = {
            "status": raw if isinstance(raw, str) else UNCLASSIFIED,
        }
        if verdict == UNCLASSIFIED:
            # Reported in the STATUS field too, not only as a verdict: the
            # probe and every other reader key on `status`, and leaving an
            # unrecognised word there would let it be read as a pass by
            # anything whose pass-set happens to contain it.
            entry["status"] = UNCLASSIFIED
            entry["reported_status"] = raw
        entry["verdict"] = verdict
        detail = _detail_for(summary, key)
        if detail is not None:
            entry["detail"] = detail
        out[key] = entry
    return out


def stage_status(sub_statuses: dict[str, dict]) -> str:
    """``degraded`` when any leg is an error or unclassified, else ``ok``."""
    for entry in sub_statuses.values():
        if entry.get("verdict") in ("error", UNCLASSIFIED):
            return STAGE_DEGRADED
    return STAGE_OK


def apply_substatus_honesty(summary: dict) -> dict:
    """Stamp ``sub_statuses`` / ``degraded_sub_results`` and demote ``status``.

    Mutates and returns ``summary``. Called on the ONE path that fans out —
    the enabled, non-dry Director run — immediately before the stage-coverage
    record is taken, so the honest status is what reaches S3, the SF output and
    the digest alike rather than being recomputed by each of them.

    ``status`` is only ever demoted, never promoted: a caller that already set
    something other than ``ok`` (a future terminal sub-state) keeps it.
    """
    sub_statuses = build_sub_statuses(summary)
    if not sub_statuses:
        return summary
    summary["sub_statuses"] = sub_statuses
    summary["degraded_sub_results"] = sorted(
        key
        for key, entry in sub_statuses.items()
        if entry.get("verdict") in ("error", UNCLASSIFIED)
    )
    if summary["degraded_sub_results"] and summary.get("status") == STAGE_OK:
        summary["status"] = STAGE_DEGRADED
    return summary
