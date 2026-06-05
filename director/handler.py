"""
handler.py — the Director's Lambda entrypoint (Layer C).

Designed to run as the FINAL task of the Saturday SF, immediately after the
``ReportCard`` state (which writes ``evaluator/{date}/report_card.json``). The
Director reads that fresh card, weighs the week's issues, and emits an advisory
``DirectorWeeklyActionPlan`` → ``director/{date}/action_plan.json`` + merges the
carry-over ledger.

**The switch:** gated behind ``DIRECTOR_ENABLED`` (env, default OFF), checked at
request time so flipping it on/off needs no redeploy. OFF → no-op (returns
``status: disabled``), so the Director SF state can be wired non-fatally now and
activated after the foundation validates on a clean Saturday cycle.

Fail-loud on a genuine error (the SF state's own Catch makes it non-fatal — an
advisory failure must never break the run that produced the real trading
artifacts). The Anthropic key + langchain are only needed when the flag is on.

Lambda handler reference: ``director.handler.handler``.
"""

from __future__ import annotations

import json
import logging
import os

import boto3

from director.agent import build_action_plan
from director.carryover import load_ledger, merge_plan_into_ledger, write_ledger
from grading.handler import _resolve_run_date

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "alpha-engine-research"


def _enabled() -> bool:
    return os.environ.get("DIRECTOR_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


def _load_report_card(s3, bucket: str, run_date: str) -> dict | None:
    from botocore.exceptions import ClientError
    key = f"evaluator/{run_date}/report_card.json"
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        raise
    return json.loads(resp["Body"].read())


def _dry_run_probe(bucket: str, run_date: str, card: dict | None, s3) -> dict:
    """Preflight probe: exercise the Director's bootstrap/import/IAM surface with
    no Opus call and no S3 write.

    What it validates (the Saturday-fatal-break classes):
      - the ``langchain-anthropic`` lazy import + ``ChatAnthropic`` construction,
      - the SSM ``ANTHROPIC_API_KEY`` fetch (the ``ReadAnthropicSecret`` IAM grant),
      - the carry-over-ledger S3 read + (when a card is present) the digest build.

    The card is normally ABSENT on a preflight (the dry ``ReportCard`` upstream did
    not write one), so a missing card is expected here — we still construct the
    client and read the ledger, then stop short of the digest. Anything that raises
    (broken import, revoked key grant, unreadable ledger) propagates to the SF
    state's non-fatal Catch as a caught preflight failure.
    """
    from director.agent import _default_llm, build_messages

    _default_llm()  # raises on a missing langchain dep or a broken SSM key-fetch grant
    ledger = load_ledger(bucket, s3_client=s3)  # validates ledger read IAM
    digest_built = False
    if card is not None:
        build_messages(card, carryover=ledger)  # exercises the digest path
        digest_built = True

    summary = {
        "status": "dry_run",
        "run_date": run_date,
        "card_present": card is not None,
        "llm_constructed": True,
        "digest_built": digest_built,
        "ledger_size": len(ledger.get("items", [])),
    }
    logger.info("Director preflight probe ok: %s", summary)
    return summary


def handler(event: dict | None = None, context=None) -> dict:
    """Build + persist the weekly Director action plan (flag-gated)."""
    event = event or {}
    if not _enabled():
        logger.info("Director disabled (DIRECTOR_ENABLED off) — no-op.")
        return {"status": "disabled", "reason": "DIRECTOR_ENABLED is off"}

    bucket = event.get("bucket") or os.environ.get("EVALUATOR_BUCKET") or DEFAULT_BUCKET
    run_date = _resolve_run_date(event)
    dry_run = bool(event.get("dry_run", False))
    s3 = boto3.client("s3")

    card = _load_report_card(s3, bucket, run_date)

    if dry_run:
        # Friday-PM Preflight Pipeline (SF passes dry_run=$.research_dry). Exercise
        # the Saturday-fatal-break surface — container boot, the langchain-anthropic
        # lazy import, the SSM ANTHROPIC_API_KEY fetch (validates the
        # ReadAnthropicSecret IAM grant), the S3 reads — but make NO paid Opus call
        # and NO write (no action_plan, no carry-over-ledger mutation: that ledger
        # is shared + non-date-scoped, so a preflight write would pollute the real
        # Saturday run, ROADMAP L4504). Fail-loud (the SF state's Catch is non-fatal):
        # a broken import / revoked key grant surfaces as a caught preflight failure
        # ~18h before the real Saturday Director would hit it.
        return _dry_run_probe(bucket, run_date, card, s3)

    if card is None:
        # The ReportCard state should have produced it; absence is a real gap
        # the SF Catch + freshness monitor surface. Fail loud.
        raise RuntimeError(
            f"Director: no report_card.json at evaluator/{run_date}/ — the ReportCard "
            "state must run before the Director."
        )

    ledger = load_ledger(bucket, s3_client=s3)
    roadmap_digest = event.get("roadmap_digest")  # optional; Phase H wires the live ROADMAP read
    plan = build_action_plan(card, run_date=run_date, carryover=ledger, roadmap_digest=roadmap_digest)

    plan_key = f"director/{run_date}/action_plan.json"
    s3.put_object(
        Bucket=bucket, Key=plan_key,
        Body=plan.model_dump_json(indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    merged = merge_plan_into_ledger(ledger, plan, run_date)
    ledger_key = write_ledger(bucket, merged, s3_client=s3)

    summary = {
        "status": "ok",
        "run_date": run_date,
        "n_action_items": len(plan.action_items),
        "n_top_risks": len(plan.top_risks),
        "action_plan_key": plan_key,
        "ledger_key": ledger_key,
        "ledger_size": len(merged.get("items", [])),
    }
    logger.info("Director plan written: %s", summary)
    return summary


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    """Local manual validation: ``DIRECTOR_ENABLED=1 ANTHROPIC_API_KEY=… \\
    python -m director.handler --date 2026-05-30 [--no-write]``."""
    import argparse

    parser = argparse.ArgumentParser(description="Run the Director against a report card (manual validation).")
    parser.add_argument("--date", default=None)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--no-write", action="store_true",
                        help="Build the plan + print it, but do not write to S3.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.no_write:
        # Build-only path (no S3 writes) for eyeballing plan quality.
        os.environ.setdefault("DIRECTOR_ENABLED", "1")
        s3 = boto3.client("s3")
        run_date = args.date or _resolve_run_date({})
        card = _load_report_card(s3, args.bucket, run_date)
        if card is None:
            print(f"No report_card at evaluator/{run_date}/"); return 1
        plan = build_action_plan(card, run_date=run_date, carryover=load_ledger(args.bucket, s3_client=s3))
        print(plan.model_dump_json(indent=2))
        return 0

    out = handler({"date": args.date, "bucket": args.bucket})
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
