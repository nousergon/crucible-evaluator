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
from datetime import datetime, timezone

import boto3

from krepis.dates import now_dual, resolve_trading_day
from krepis.logging import setup_logging

from grading.aggregate import build_report_card, write_report_card
from grading.experiment_record import build_experiment_record, write_experiment_record
from grading.pipeline_gates import gates_unmeasured, log_gate_state, read_gate_state
from grading.self_test import run_self_test, verdict_is_pass, write_self_test

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


def _canary_probe(bucket: str) -> dict:
    """Deploy boot-probe (invoked as ``{"action": "canary"}`` by
    ``infrastructure/deploy.sh`` after pushing the new image).

    A deploy canary's job is to prove the artifact it just shipped can BOOT
    and its handler wiring is healthy — NOT to assert the state of weekly
    production data. Reaching this function already proves the container
    initialized and every module-level first-party import resolved under the
    new image (``grading.aggregate`` pulls in all tile modules at import
    time) — exactly the failure a dependency bump introduces (this deploy was
    triggered by #137's langchain-anthropic bump, the canonical thing a boot
    probe must catch). On top of that it exercises run-date resolution and
    boto3 S3-client construction (region/credential wiring), then returns
    ``status: ok``.

    It DELIBERATELY does NOT call :func:`grading.aggregate.build_report_card`,
    and therefore never runs the hard input-freshness preflight
    (``assert_input_freshness``, alpha-engine-config#3058). That preflight is
    a deliberate, unconditional, Brian-ruled fail-loud gate for the WEEKLY
    ASSESSMENT — a real build on stale/absent data is "COMPLETELY USELESS" and
    must hard-fail — and it stays fully in force on every real
    build/CLI/snapshot path (untouched here). But a deploy runs on ARBITRARY
    days: the current trading day's weekly ``backtest/{date}/metrics.json``
    (and its siblings) legitimately do not exist yet on any off-cycle deploy,
    so routing the deploy canary through the freshness-gated build made every
    off-cycle deploy hard-fail on a MissingInputArtifactError that reflects a
    healthy image, not a broken one. Input freshness is the weekly run's gate;
    boot health is the deploy's — keep them on separate paths.
    """
    run_date = _resolve_run_date({})
    boto3.client("s3")  # construct (region/credential wiring) — no read; freshness-agnostic
    logger.info("Canary boot-probe OK (run_date=%s, bucket=%s)", run_date, bucket)
    return {"status": "ok", "probe": "canary", "run_date": run_date, "bucket": bucket}


def _record_stage_coverage(stage: str, run_date: str, started: datetime, result: dict) -> dict:
    """Merge the end-of-stage output-coverage verdict into a handler's return
    payload, under ``stage_coverage`` (config-I7214, the ruled rescope).

    Called from EACH SF-Task-mapped return point of a handler, immediately
    before it returns to Step Functions — the assertion belongs in the
    stage's own handler asserting its own declared output, never a
    standalone end-of-run SF state. OBSERVE MODE ONLY: the single
    enforcement flag lives in ``nousergon_lib.stage_coverage`` and is never
    enabled from a call site here — a coverage miss is recorded, never
    raised, before Saturday 2026-08-15 02:00 PT (and after, only on a
    separate ruled flip).

    ``ImportError`` means the ``nousergon-lib`` git-tag pin in
    ``requirements.txt`` predates the module (a separate wave bumps the
    pin) — logged loud so an absent assertion is never mistaken for a
    covered stage, never swallowed, and the handler's own outcome is
    unchanged either way.
    """
    try:
        from nousergon_lib.stage_coverage import assert_stage_coverage
        result["stage_coverage"] = assert_stage_coverage(stage, run_date=run_date, window_start=started)
    except ImportError as exc:
        logger.error("stage-coverage assertion unavailable for %s: %s", stage, exc)
    return result


def handler(event: dict | None = None, context=None) -> dict:
    """Build + persist the Report Card v2 for a run date.

    Returns a compact summary (the SF state output): overall status, per-tile
    statuses, real-graded coverage, and the S3 key written.

    ``action="check_deploy_drift"`` (config#2348) is a separate, lightweight
    Step Function gate: compares this Lambda's baked image SHA against
    ``origin/main`` HEAD and returns immediately — no report-card build, no
    S3 reads beyond the local stamp file. See ``grading/deploy_drift.py``.

    ``action="canary"`` (config#3058 follow-up) is the deploy boot-probe:
    ``infrastructure/deploy.sh`` invokes it post-deploy to smoke-test the
    freshly-pushed image and returns immediately — see ``_canary_probe``.
    """
    event = event or {}
    _started = datetime.now(timezone.utc)
    if event.get("action") == "check_deploy_drift":
        from grading.deploy_drift import _resolve_function_name, check_deploy_drift
        result = check_deploy_drift(function_name=_resolve_function_name(context))
        # Infrastructure/gate stage: declares no durable artifact — the lib
        # returns COVERED_NO_OUTPUT for it, but it must still record that it
        # declared nothing (config-I7214 §"declares nothing" != "never considered").
        return _record_stage_coverage(
            "EvaluatorDeployDriftCheck", run_date=_resolve_run_date({}), started=_started, result=result,
        )

    bucket = event.get("bucket") or os.environ.get("EVALUATOR_BUCKET") or DEFAULT_BUCKET

    if event.get("action") == "canary":
        # Deploy-time boot probe, not an SF-declared stage — no coverage call.
        return _canary_probe(bucket)

    run_date = _resolve_run_date(event)
    # dry_run = the Friday-PM Preflight Pipeline (SF passes dry_run=$.research_dry,
    # the canonical shell-run-dry signal). It still exercises the full read+compute
    # path — container boot, lib/numpy/pandas imports, S3-read IAM/transport across
    # the backtest/predictor/trades artifacts, the tile compute — but does NOT
    # persist the (degenerate, mostly-N/A) preflight card. Explicit `write` wins.
    dry_run = bool(event.get("dry_run", False))
    write = event.get("write", not dry_run)

    # `snapshot` (config-I2556): the evaluator report card is now a PERSISTENT
    # surface — every non-dry invocation rebuilds the full card and overwrites
    # the standing `evaluator/latest/report_card.json` pointer (see
    # write_report_card). `snapshot=True` ALSO freezes the dated
    # `evaluator/{run_date}/report_card.json` weekly record.
    #
    # Default is False: `feat/weekly-sf-advisory-child-and-sunday-zoo`
    # (nousergon-data PR #832) merged 2026-07-14 and both production callers
    # now pass this flag explicitly — the Saturday advisory-child `ReportCard`
    # state (`infrastructure/step_function_advisory.json`) passes `true` for
    # the weekly freeze; the Sunday ModelZoo `GradingLambdaReGrade` state
    # (`infrastructure/step_function_modelzoo.json`) passes `false` for the
    # re-grade-only tail invoke. An absent flag now means "refresh latest
    # only" — the honest default for the persistent-surface model, where a
    # frozen weekly snapshot is the deliberate exception, not the norm.
    snapshot = bool(event.get("snapshot", False))

    logger.info(
        "Building Report Card v2 for %s (bucket=%s, write=%s, dry_run=%s, snapshot=%s)",
        run_date, bucket, write, dry_run, snapshot,
    )
    # ── Published known-answer SELF-TEST (Brian, 2026-08-13) ───────────────
    # `grading/self_test.py` drives the PRODUCTION portfolio-outcome tile over a
    # frozen in-memory eod_pnl.csv and compares Sharpe / Sortino / Calmar /
    # CVaR(95) / max-drawdown against expectations derived on paper from each
    # metric's definition, plus two metamorphic relations and the
    # undefined-vs-measured-zero contract.
    #
    # It runs HERE, before the card is built, because the question it answers is
    # "can the arithmetic in THIS image be trusted" — a question whose answer
    # must not depend on the card build succeeding. It runs on every invocation
    # (the battery is ~0.5s over 100 in-memory rows, no network, no market data)
    # and `run_self_test` never raises: any failure resolves to FAIL or UNKNOWN
    # inside the returned body. sf-pipeline-policy.md §2.3a — withholding a
    # guarantee beats failing the run, so a non-PASS verdict never aborts the
    # card; it is carried in this stage's terminal output instead.
    self_test = run_self_test(run_date=run_date)
    self_test_pass = verdict_is_pass(self_test.get("verdict"))
    if not self_test_pass:
        logger.error(
            "SELF-TEST %s for %s (%d disagreed, %d could not run) on %s — the "
            "Report Card's risk metrics are NOT attested this cycle.",
            self_test.get("verdict"), run_date, self_test.get("n_failed", 0),
            self_test.get("n_errored", 0), self_test.get("libraries"),
        )

    # ── The SF's correctness-gate state (alpha-engine-config-I7282) ──────────
    # sf-pipeline-policy.md §2.3a rule 3. `gate_state` is the ReportCard Task's
    # new payload block: the two pre-spend gate probe payloads plus the four
    # fail-open degradation families, each seeded at InitializeInput so it is
    # present in BOTH polarities (a field that appears only on the bad path
    # cannot be distinguished from a producer that broke). Read here — never
    # `.get(..., True)`-style — so an absent block resolves to UNKNOWN, which is
    # precisely the pre-I7282 world and must not read as a clean run.
    gate_block = read_gate_state(event.get("gate_state"))
    log_gate_state(gate_block, run_date)

    card = build_report_card(bucket, run_date, self_test=self_test,
                             gate_state=event.get("gate_state"))

    tiles = card.get("tiles", {})
    tile_status = {name: t.get("status") for name, t in tiles.items()}
    real_graded = {
        name: sum(1 for c in t.get("components", []) if not str(c.get("status", "")).startswith("N/A"))
        for name, t in tiles.items()
    }

    latest_key = None
    dated_key = None
    if write:
        written = write_report_card(bucket, run_date, card, snapshot=snapshot)
        latest_key = written["latest_key"]
        dated_key = written["dated_key"]

    # experiment_record.v1 (alpha-engine-config#3077 Phase C): a per-run index
    # binding what ran to what it emitted, for the results renderer. Isolated,
    # fail-SOFT best-effort — this is a NEW secondary-observability artifact
    # riding on top of the report-card build; a bug in it must never turn a
    # healthy report-card cycle into a failed SF state (the report card write
    # above is the primary deliverable and is already fully persisted by the
    # time this runs). Skipped entirely on a dry run — dry runs never persist
    # a report card, so a record built from one would only ever describe a
    # cycle that emitted nothing (an uninformative, permanently-"failed"
    # record every Friday-PM Preflight Pipeline run).
    # Persist the self-test beside the card. Isolated and fail-SOFT for the same
    # reason `experiment_record` below is: this is secondary observability riding
    # on top of the report-card build, and a bug in it must never turn a healthy
    # grading cycle into a failed SF state. The verdict still reaches the SF
    # output via `summary` regardless of whether the write succeeded — so a lost
    # artifact degrades the evidence, never the verdict.
    self_test_key_written = None
    if write:
        try:
            self_test_key_written = write_self_test(bucket, run_date, self_test)
        except Exception as exc:  # noqa: BLE001 — secondary artifact, never fatal
            logger.warning(
                "self_test emission failed for %s (verdict=%s is still carried in "
                "this stage's output; the report card is unaffected): %s",
                run_date, self_test.get("verdict"), exc, exc_info=True,
            )

    experiment_record_key = None
    if write:
        try:
            record = build_experiment_record(
                bucket, run_date, card, report_card_key=dated_key or latest_key,
            )
            written_record = write_experiment_record(bucket, run_date, record)
            experiment_record_key = written_record["dated_key"]
        except Exception as exc:  # noqa: BLE001 — secondary artifact, never fatal
            logger.warning(
                "experiment_record emission failed for %s (report card already "
                "persisted above; this does not fail the run): %s",
                run_date, exc, exc_info=True,
            )

    summary = {
        "status": "ok",
        "dry_run": dry_run,
        "run_date": run_date,
        "bucket": bucket,
        "tiles_overall_status": card.get("tiles_overall_status"),
        "tile_status": tile_status,
        "real_graded": real_graded,
        "report_card_key": dated_key,
        "latest_key": latest_key,
        "snapshot": snapshot,
        "artifacts": card.get("_provenance", {}).get("artifacts", {}),
        "experiment_record_key": experiment_record_key,
        # §2.3a rule 3 — every surface presenting the run's results carries the
        # verdict state. `degraded_self_test` is derived from the verdict rather
        # than set independently, so the two can never disagree, and a missing
        # or unrecognised verdict reads as degraded (never as a pass).
        "self_test_verdict": self_test.get("verdict"),
        "degraded_self_test": not self_test_pass,
        "self_test_key": self_test_key_written,
        # alpha-engine-config-I7282 — the same clause, applied to the gates that
        # run BEFORE this stage. Carried in the SF stage output as well as in the
        # card, so the withholding is visible in the execution history and not
        # only in the artifact: a stage that quietly rendered an unverified card
        # and returned status "ok" is the same blindness one layer up.
        "pipeline_gates_verdict": gate_block["verdict"],
        "degraded_pipeline_gates": gates_unmeasured(gate_block),
        "pipeline_gates_unmeasured": gate_block["unmeasured"],
        "pipeline_gates_statement": gate_block["statement"],
    }
    _record_stage_coverage("ReportCard", run_date=run_date, started=_started, result=summary)
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
    parser.add_argument("--no-snapshot", action="store_true",
                         help="skip the dated weekly snapshot; refresh evaluator/latest/report_card.json only")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = handler({
        "date": args.date, "bucket": args.bucket, "write": not args.no_write,
        "snapshot": not args.no_snapshot,
    })
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
