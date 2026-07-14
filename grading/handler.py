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

from krepis.dates import now_dual, resolve_trading_day
from krepis.logging import setup_logging

from grading.aggregate import build_report_card, write_report_card

# Structured logging + flow-doctor. Passing a flow-doctor.yaml attaches a
# FlowDoctorHandler at ERROR (off under pytest), so every log.error() routes
# through flow-doctor's capture -> dedupe -> diagnose -> alert dispatch without
# explicit plumbing. The yaml ships in the Lambda task root (Dockerfile COPY);
# its ${VAR} secrets resolve from SSM at runtime via the role's existing
# parameter/alpha-engine/* read. Mirrors research/lambda/handler.py. The
# grading state is non-fatal (SF Catch) and fail-loud by design; flow-doctor
# adds capture/diagnosis on top without changing that contract.
_FLOW_DOCTOR_EXCLUDE_PATTERNS: list[str] = []
_FLOW_DOCTOR_YAML = os.path.join(
    os.environ.get(
        "LAMBDA_TASK_ROOT",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ),
    "flow-doctor.yaml",
)
setup_logging(
    "evaluator-grading",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
)

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "alpha-engine-research"


def _resolve_run_date(event: dict) -> str:
    """Run date for the report card, normalized to the NYSE trading day.

    Precedence: explicit ``event['date']`` (the SF threads the CALENDAR
    run_date) → ``EVALUATOR_RUN_DATE`` env → ``now_dual().trading_day`` (last
    closed NYSE session). Whichever wins is then normalized to the trading day
    via :func:`krepis.dates.resolve_trading_day` so the tiles read the
    ``backtest/{trading_day}`` keys the backtester + evaluate.py actually wrote
    (calendar-day is deprecated as an artifact key — see DATE_CONVENTIONS).

    History: the grading layer once trusted ``event['date']`` verbatim → it
    read ``backtest/2026-06-07`` (a Sunday) and graded 0/18 artifacts while the
    real artifacts sat at ``backtest/2026-06-05``. The normalizer (formerly a
    local ``_to_trading_day``, now the shared ``krepis.dates.resolve_trading_day``
    the backtester also uses) keys the report card + Director on the SAME
    trading day the producers wrote.
    """
    explicit = (event or {}).get("date") or os.environ.get("EVALUATOR_RUN_DATE")
    raw = explicit if explicit else now_dual().trading_day
    return resolve_trading_day(raw)


def handler(event: dict | None = None, context=None) -> dict:
    """Build + persist the Report Card v2 for a run date.

    Returns a compact summary (the SF state output): overall status, per-tile
    statuses, real-graded coverage, and the S3 key written.

    ``action="check_deploy_drift"`` (config#2348) is a separate, lightweight
    Step Function gate: compares this Lambda's baked image SHA against
    ``origin/main`` HEAD and returns immediately — no report-card build, no
    S3 reads beyond the local stamp file. See ``grading/deploy_drift.py``.
    """
    event = event or {}
    if event.get("action") == "check_deploy_drift":
        from grading.deploy_drift import _resolve_function_name, check_deploy_drift
        return check_deploy_drift(function_name=_resolve_function_name(context))

    bucket = event.get("bucket") or os.environ.get("EVALUATOR_BUCKET") or DEFAULT_BUCKET
    run_date = _resolve_run_date(event)
    # dry_run = the Friday-PM Preflight Pipeline (SF passes dry_run=$.research_dry,
    # the canonical shell-run-dry signal). It still exercises the full read+compute
    # path — container boot, lib/numpy/pandas imports, S3-read IAM/transport across
    # the backtest/predictor/trades artifacts, the tile compute — but does NOT
    # persist the (degenerate, mostly-N/A) preflight card. Explicit `write` wins.
    dry_run = bool(event.get("dry_run", False))
    write = event.get("write", not dry_run)

    logger.info(
        "Building Report Card v2 for %s (bucket=%s, write=%s, dry_run=%s)",
        run_date, bucket, write, dry_run,
    )
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
        "dry_run": dry_run,
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
