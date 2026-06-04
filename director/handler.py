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


def handler(event: dict | None = None, context=None) -> dict:
    """Build + persist the weekly Director action plan (flag-gated)."""
    event = event or {}
    if not _enabled():
        logger.info("Director disabled (DIRECTOR_ENABLED off) — no-op.")
        return {"status": "disabled", "reason": "DIRECTOR_ENABLED is off"}

    bucket = event.get("bucket") or os.environ.get("EVALUATOR_BUCKET") or DEFAULT_BUCKET
    run_date = _resolve_run_date(event)
    s3 = boto3.client("s3")

    card = _load_report_card(s3, bucket, run_date)
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
