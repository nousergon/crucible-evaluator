"""
handler.py — the evaluator's grading-layer Lambda entrypoint (Layer B producer).

Runs as a Saturday-SF state after the terminal evaluation states (so it reads
fresh artifacts): assembles the Report Card v2 from the persisted per-module
analysis artifacts and writes ``evaluator/{date}/report_card.json``. This is
what makes the report card *produced* on a cadence rather than only buildable
from the CLI.

Failure isolation (RC v2 / director-plan §5): the SF state wraps this in its own
Catch and is **non-fatal** — a grading failure must never break the run that
produced the real trading artifacts. So the handler is **fail-loud** (it raises
on a genuine error rather than writing a half-card), and the SF Catch + the
freshness monitor on ``evaluator/{date}/report_card.json`` surface the failure.
This is the no-silent-fails carve-out for secondary observability hung off a
primary path that records the failure.

Lambda handler reference: ``grading.handler.handler``.
"""

from __future__ import annotations

import json
import logging
import os

from alpha_engine_lib.dates import now_dual

from grading.aggregate import build_report_card, write_report_card

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "alpha-engine-research"


def _resolve_run_date(event: dict) -> str:
    """Run date for the report card.

    Precedence: explicit ``event['date']`` (the SF passes the normalized
    RUN_DATE, mirroring the backtester) → ``EVALUATOR_RUN_DATE`` env →
    ``now_dual().trading_day`` (last closed NYSE session — never ahead of now,
    per the date conventions). Matches the ``backtest/{date}`` keying the tiles
    read from.
    """
    explicit = (event or {}).get("date") or os.environ.get("EVALUATOR_RUN_DATE")
    if explicit:
        return explicit
    return now_dual().trading_day


def handler(event: dict | None = None, context=None) -> dict:
    """Build + persist the Report Card v2 for a run date.

    Returns a compact summary (the SF state output): overall status, per-tile
    statuses, real-graded coverage, and the S3 key written.
    """
    event = event or {}
    bucket = event.get("bucket") or os.environ.get("EVALUATOR_BUCKET") or DEFAULT_BUCKET
    run_date = _resolve_run_date(event)
    write = event.get("write", True)  # SF runs write the card; dry-run can disable

    logger.info("Building Report Card v2 for %s (bucket=%s, write=%s)", run_date, bucket, write)
    card = build_report_card(bucket, run_date)

    tiles = card.get("tiles", {})
    tile_status = {name: t.get("status") for name, t in tiles.items()}
    real_graded = {
        name: sum(1 for c in t.get("components", []) if not str(c.get("status", "")).startswith("N/A"))
        for name, t in tiles.items()
    }

    key = None
    if write:
        key = write_report_card(bucket, run_date, card)

    summary = {
        "status": "ok",
        "run_date": run_date,
        "bucket": bucket,
        "tiles_overall_status": card.get("tiles_overall_status"),
        "tile_status": tile_status,
        "real_graded": real_graded,
        "report_card_key": key,
        "artifacts": card.get("_provenance", {}).get("artifacts", {}),
    }
    logger.info(
        "Report Card v2 %s: overall=%s tiles=%s",
        run_date, summary["tiles_overall_status"], tile_status,
    )
    return summary


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    """Local invoke: ``python -m grading.handler --date 2026-06-07 [--no-write]``."""
    import argparse

    parser = argparse.ArgumentParser(description="Invoke the grading Lambda handler locally.")
    parser.add_argument("--date", default=None)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = handler({"date": args.date, "bucket": args.bucket, "write": not args.no_write})
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
