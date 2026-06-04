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
"""

from __future__ import annotations

import json
import logging

import boto3
from botocore.exceptions import ClientError

from director.schema import DirectorWeeklyActionPlan

logger = logging.getLogger(__name__)

LEDGER_KEY = "director/carryover_ledger.json"


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
    """
    items = {it["id"]: it for it in (ledger.get("items") or [])}
    for ai in plan.action_items:
        existing = items.get(ai.id)
        row = ai.model_dump()
        if existing:
            row["first_seen"] = existing.get("first_seen", run_date)
        else:
            row["first_seen"] = run_date
        row["last_seen"] = run_date
        items[ai.id] = row
    return {
        "updated": run_date,
        "items": sorted(items.values(), key=lambda r: (r.get("first_seen", ""), r.get("id", ""))),
    }


def write_ledger(bucket: str, ledger: dict, s3_client=None) -> str:
    s3 = s3_client or boto3.client("s3")
    s3.put_object(
        Bucket=bucket, Key=LEDGER_KEY,
        Body=json.dumps(ledger, indent=2, default=str).encode("utf-8"),
        ContentType="application/json",
    )
    return LEDGER_KEY
