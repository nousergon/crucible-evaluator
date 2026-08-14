"""
carryover.py — the Director's carry-over ledger (read / merge).

``s3://alpha-engine-research/director/carryover_ledger.json`` is the append/merge
record that makes "carry-over tasks are fine" structural rather than folklore:
each ``ActionItem.id`` is tracked across weeks with its status transitions and
first/last-seen run dates. This is the system-level instantiation of the
"reminders must be written down" rule — the plan is persisted, not
emailed-and-lost.

Phase-E scope: read the ledger + merge a new plan into it (upsert by id). The
dashboard surface + self-grade trend are later phases.

**The ledger is bounded (alpha-engine-config-I7311).** Until 2026-08-14
``merge_plan_into_ledger`` was upsert-only and nothing ever left: measured on
the live artifact, 41 of 41 rows carried ``status="carried_over"``, not one had
ever reached ``resolved``, and 17 of them had a ``last_seen`` of 2026-06-18 to
2026-07-17 — items the Director had not re-proposed for four to eight weekly
cycles and never would again. Every one of them was still rendered into the
prompt by ``agent._carryover_context`` AND still required an explicit
disposition line in the plan's ``carryover_review``, which is why that list had
exactly one entry per ledger row (41) in the 2026-08-13 plan.

That made the Director's LLM call duration a monotonically increasing function
of an artifact that only grew. Measured plan-call wall time against the same
route and model: 87-107s (2026-08-04, ledger 29-31) → 135s (2026-08-13, ledger
37) → 205s (2026-08-14, ledger 41) → two consecutive attempts past the 340s
per-attempt ceiling on the same day, which hard-failed the weekly SF. The
ceiling was not wrong; it was a fixed line under a rising curve.

Retirement is therefore **structural, not cosmetic**: it is what makes the
prompt's carry-over section, and the output section that mirrors it, bounded by
construction rather than by whatever the model happens to emit.

Retirement is on AGE and SET SIZE only — never on ``status="resolved"``; see
:func:`_retirement_reason` for why the obvious trigger is the wrong one.

**Retire, never delete.** The old docstring's refusal — "we don't silently drop
them" — was right about the risk and wrong about the remedy. A retired row is
moved to ``retired_items`` in the SAME artifact, with the reason and the run
that retired it stamped on it, so every prior commitment stays reconstructible
from durable state alone. What changes is only what the *prompt* carries.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime

import boto3
from botocore.exceptions import ClientError

from director.schema import DirectorWeeklyActionPlan

logger = logging.getLogger(__name__)

LEDGER_KEY = "director/carryover_ledger.json"

# ── The two bounds on the ACTIVE set (alpha-engine-config-I7311) ─────────────
#
# Both are needed, and neither is redundant:
#
#   * The age bound retires what has gone stale. An id the Director has not
#     re-proposed for four consecutive weekly cycles is one it has stopped
#     proposing — the model never sets ``status="resolved"`` on an item it
#     simply drops, so staleness is the only signal that exists. 28 days is
#     four Saturday cycles, which is one full cycle of slack past the three a
#     genuinely-carried item shows.
#
#   * The size bound retires the tail when the Director proposes faster than
#     the age bound retires. Without it the active set is bounded only by
#     (items per plan × 4 weeks), which is not a bound anyone chose. 40 is
#     ~1.6× the largest plan on record (25 action items, 2026-08-13
#     measurement in ``agent._default_llm``), so it is slack in the normal
#     case and a real ceiling in the pathological one.
#
# Ordering for the size bound is (most recently seen first, then priority, then
# id) — so what survives is what the Director most recently still believed in,
# and a P0 outranks a P3 at the same recency. Deterministic: the same ledger
# retires the same rows on a re-run.
RETIREMENT_STALE_DAYS = 28
ACTIVE_LEDGER_MAX_ITEMS = 40

_PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


def _parse_run_date(value) -> date | None:
    """``YYYY-MM-DD`` → ``date``; ``None`` for anything unparseable.

    Never raises. An unparseable ``last_seen`` is treated as "age unknown" by
    the caller, which then declines to retire on age — the conservative
    direction: a row whose age cannot be established stays visible rather than
    disappearing on a formatting accident.
    """
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.strptime(value.strip()[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _retirement_reason(row: dict, run_day: date | None) -> str | None:
    """Why this row leaves the active set on age, or ``None`` to keep it.

    The size bound is not here — it is a property of the SET, not the row, and
    is applied by :func:`partition_ledger` after this.

    **``status == "resolved"`` is deliberately NOT a retirement trigger.** It
    reads like the obvious one and it is the wrong one: ``loop_verification``'s
    reopen-if-unrecovered pass runs over the same rows and exists precisely to
    catch an item declared resolved whose evidence is still adverse. Retiring
    on the resolution would hide the row from the check that disputes it — the
    close-and-look-away failure the whole loop-verification pass was built to
    stop. A resolved item simply stops being re-proposed, so it ages out
    through the same staleness rule as everything else, one full month later,
    by which time the reopen check has had four passes at it.
    """
    if run_day is None:
        return None
    last_seen = _parse_run_date(row.get("last_seen"))
    if last_seen is None:
        return None
    age_days = (run_day - last_seen).days
    if age_days > RETIREMENT_STALE_DAYS:
        return f"stale:{age_days}d"
    return None


def partition_ledger(rows: list[dict], run_date: str) -> tuple[list[dict], list[dict]]:
    """Split ledger rows into ``(active, retiring)`` for ``run_date``.

    Pure and total — no I/O, no exceptions on malformed rows, and the same
    input always yields the same split. It is separated from
    :func:`merge_plan_into_ledger` so the bound can be tested (and reasoned
    about) without constructing a plan.

    A retiring row is returned with ``retired_on`` / ``retired_reason``
    stamped, which is what makes the retirement reconstructible from the
    artifact alone rather than from this source file.
    """
    run_day = _parse_run_date(run_date)
    active: list[dict] = []
    retiring: list[dict] = []
    for row in rows:
        reason = _retirement_reason(row, run_day)
        if reason is None:
            active.append(row)
        else:
            retiring.append({**row, "retired_on": run_date, "retired_reason": reason})

    if len(active) > ACTIVE_LEDGER_MAX_ITEMS:
        active.sort(
            key=lambda r: (
                _parse_run_date(r.get("last_seen")) or date.min,
                -_PRIORITY_ORDER.get(str(r.get("priority")), len(_PRIORITY_ORDER)),
                str(r.get("id", "")),
            ),
            reverse=True,
        )
        overflow = active[ACTIVE_LEDGER_MAX_ITEMS:]
        active = active[:ACTIVE_LEDGER_MAX_ITEMS]
        retiring.extend(
            {**row, "retired_on": run_date,
             "retired_reason": f"over_cap:{ACTIVE_LEDGER_MAX_ITEMS}"}
            for row in overflow
        )
    return active, retiring


def load_ledger(bucket: str, s3_client=None) -> dict:
    """Load the carry-over ledger (``{"items": [...]}``); empty if absent.

    Fail-loud on a real S3 error (a corrupt/unreadable ledger must not be
    silently treated as empty — that would drop every prior commitment).
    NoSuchKey is the legitimate first-run empty state.
    """
    s3 = s3_client or boto3.client("s3")
    try:
        resp = s3.get_object(Bucket=bucket, Key=LEDGER_KEY)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return {"items": []}
        logger.error("Ledger read failed s3://%s/%s: %s", bucket, LEDGER_KEY, e)
        raise
    return json.loads(resp["Body"].read())


def merge_plan_into_ledger(ledger: dict, plan: DirectorWeeklyActionPlan, run_date: str) -> dict:
    """Upsert this week's action items into the ledger by stable id.

    New id → appended with ``first_seen``=this run; existing id → status +
    ``last_seen`` updated. Items in the ledger but NOT in this plan keep their
    prior state (the plan's ``carryover_review`` is the authoritative
    disposition; we don't silently drop them). Returns the merged ledger.

    **Bounded since alpha-engine-config-I7311.** After the upsert, rows stale
    past :data:`RETIREMENT_STALE_DAYS` or beyond
    :data:`ACTIVE_LEDGER_MAX_ITEMS` move from ``items`` to ``retired_items``
    with ``retired_on`` / ``retired_reason`` stamped. Only ``items`` reaches
    the prompt, so the Director's call size — and therefore its duration — no
    longer grows without limit. ``retired_items`` accumulates in the same
    artifact: nothing is deleted, and a retired row can be read back.

    Retirement runs on the MERGED set, never the pre-merge one, so an id the
    Director re-proposed this run has its ``last_seen`` refreshed first and
    cannot be retired by the same call that renewed it.

    Also tracks (config#3145 — close the Director loop):
      - ``carry_count``: consecutive weekly runs this id has appeared with a
        non-``resolved`` status. Reset to 0 the week it resolves (or on first
        appearance); incremented every week it doesn't. This is the "weeks
        carried" the carryover-escalation check (``loop_verification.py``)
        thresholds on — the ledger previously had only first/last-seen DATES,
        not a run-count, so "carried >= 2 weeks" had nothing to read.
      - ``escalated``: sticky one-shot flag set once a carried item has been
        auto-escalated to the Decision Queue, so it isn't re-escalated every
        subsequent week. Cleared back to ``False`` when the item resolves (a
        later re-carry of a since-resolved id starts its escalation clock
        fresh).
      - ``issue_number``: preserved from the existing row (not derivable from
        the plan itself — populated by ``loop_verification.backfill_issue_numbers``
        against the live GitHub state).
    """
    items = {it["id"]: it for it in (ledger.get("items") or [])}
    for ai in plan.action_items:
        existing = items.get(ai.id)
        row = ai.model_dump()
        if existing:
            row["first_seen"] = existing.get("first_seen", run_date)
            row["issue_number"] = existing.get("issue_number")
            if ai.status == "resolved":
                row["carry_count"] = 0
                row["escalated"] = False
            else:
                row["carry_count"] = existing.get("carry_count", 0) + 1
                row["escalated"] = existing.get("escalated", False)
        else:
            row["first_seen"] = run_date
            row["issue_number"] = None
            row["carry_count"] = 0
            row["escalated"] = False
        row["last_seen"] = run_date
        items[ai.id] = row

    active, retiring = partition_ledger(list(items.values()), run_date)
    if retiring:
        logger.info(
            "Director ledger: retired %d of %d rows (%s) — active set now %d "
            "(alpha-engine-config-I7311). Retired rows are kept under "
            "'retired_items'; only 'items' reaches the prompt.",
            len(retiring), len(items), ", ".join(
                f"{r.get('id')}={r.get('retired_reason')}" for r in retiring
            ), len(active),
        )

    def _key(r):
        return (r.get("first_seen", ""), r.get("id", ""))

    return {
        "updated": run_date,
        "items": sorted(active, key=_key),
        "retired_items": sorted(
            list(ledger.get("retired_items") or []) + retiring, key=_key
        ),
    }


def write_ledger(bucket: str, ledger: dict, s3_client=None) -> str:
    s3 = s3_client or boto3.client("s3")
    s3.put_object(
        Bucket=bucket, Key=LEDGER_KEY,
        Body=json.dumps(ledger, indent=2, default=str).encode("utf-8"),
        ContentType="application/json",
    )
    return LEDGER_KEY
