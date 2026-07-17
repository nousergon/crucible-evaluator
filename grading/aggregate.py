"""
aggregate.py — native report-card build (Layer B orchestrator).

Reads the persisted analysis artifacts (``grading/artifacts.py``), runs the
pure grader (``grading/scorecard.py``), attaches provenance, and writes the
report card to the evaluator's own S3 namespace.

OBSERVE / PARALLEL-RUN (Phase C increment 1):
  The evaluator writes ``evaluator/{date}/report_card.json`` — a NEW key,
  deliberately NOT the backtester's ``backtest/{date}/grading.json``. During
  the soak both graders run; we compare letter grades (``--compare``) to verify
  the evaluator reproduces the backtester's in-process grading from the
  persisted artifacts. Cutover (dashboard reads the evaluator key; backtester
  drops its in-process grading call) is a later Phase C step, after the
  parallel run shows parity. This honours the S3-contract-safety "write both
  for ≥1 week" rule.

The Lambda handler + Saturday-SF wiring arrive in Phase F; this module exposes
``build_report_card`` / ``write_report_card`` for that handler and a thin CLI
for manual observe runs.
"""

from __future__ import annotations

import argparse
import json
import logging

import boto3
from botocore.exceptions import ClientError

from grading.artifacts import read_scorecard_inputs
from grading.history import load_card_history
from grading.scorecard import compute_scorecard
from grading.module_agg import overall_status
from nousergon_lib.quant.stats.trial_accumulator import read_cumulative_trial_count
from grading.tiles.agent import build_agent_tile
from grading.tiles.backtester import build_backtester_tile
from grading.tiles.behavioral import build_behavioral_tile
from grading.tiles.director_quality import build_director_quality_tile
from grading.tiles.executor import build_executor_tile
from grading.tiles.portfolio_outcome import build_portfolio_outcome_tile
from grading.tiles.predictor import build_predictor_tile
from grading.tiles.research import build_research_tile
from grading.tiles.substrate import build_substrate_tile

logger = logging.getLogger(__name__)

# The evaluator's own report-card namespace (NOT backtest/{date}/grading.json).
REPORT_CARD_PREFIX = "evaluator"
REPORT_CARD_FILENAME = "report_card.json"

# The standing, continuously-maintained pointer (config-I2556): every
# non-dry-run invocation of the grading handler overwrites this key with a
# freshly-rebuilt full card, regardless of the `snapshot` flag. The dated
# `report_card_key(run_date)` below stays the FROZEN weekly record, written
# only when `snapshot=True` — the deliberate archival copy OF this standing
# card, not a second independent build. Deliberately keyed beside the dated
# convention (`evaluator/latest/report_card.json`, same filename, "latest" in
# place of the date segment) rather than a flat `evaluator/latest.json`, so it
# reads as the same namespace/convention as the dated keys. Deliberately NOT a
# `YYYY-MM-DD`-shaped path segment — `history.py`'s `_CARD_KEY_RE` (dated-only)
# and its S3 `list_objects_v2` prefix walk must never pick this key up as a
# weekly card instance (see test_history.py's regression test).
LATEST_REPORT_CARD_KEY = f"{REPORT_CARD_PREFIX}/latest/{REPORT_CARD_FILENAME}"

# Provenance: the grader source this build instantiates. Bump when scorecard.py
# is re-synced from the backtester (until the Phase C cutover removes the
# backtester copy).
GRADER_SOURCE = "alpha-engine-evaluator/grading/scorecard.py (ported from backtester @f46e7e6)"


def build_report_card(
    bucket: str,
    run_date: str,
    s3_client=None,
) -> dict:
    """Read artifacts → grade → attach provenance. Pure of writes."""
    inputs, report = read_scorecard_inputs(bucket, run_date, s3_client=s3_client)
    scorecard = compute_scorecard(**inputs)

    # Cross-cycle trend history (config#1836): prior weekly CARDS are the SSOT
    # for graded values — the tiles thread trend_4w/trend_13w from these into
    # their critical score-vs-return components. Short/absent history WARNs
    # inside the loader and degrades to empty trends (never blocks the build).
    history = load_card_history(bucket, run_date, s3_client=s3_client)

    # config#2454: DSR (portfolio_outcome's dsr metric) needs the cumulative
    # count of strategy configurations trialed since inception across ALL
    # 4 backtester sweep producers (optimizer_param_sweep / gamma_sweep /
    # cov_estimator_sweep / predictor_param_sweep) — the multiple-testing
    # correction Bailey & Lopez de Prado's DSR formula deflates the observed
    # Sharpe by. crucible-backtester increments the shared counter after
    # each producer's real (non-skipped) cycle; read it here so
    # build_portfolio_outcome_tile can compute a real dsr value instead of
    # emitting N/A-NOT-IMPL. Best-effort: an artifact-read failure (e.g. the
    # counter hasn't been backfilled/seeded yet) degrades to n_trials=None,
    # which portfolio_outcome.py already treats as its pre-existing N/A path
    # — never blocks the report-card build.
    n_trials: int | None = None
    try:
        trial_state = read_cumulative_trial_count(bucket, s3_client=s3_client)
        if trial_state.get("total"):
            n_trials = int(trial_state["total"])
    except Exception as exc:  # noqa: BLE001 — advisory read, dsr degrades to N/A
        logger.warning(
            "build_report_card: cumulative_trial_count read failed (dsr will "
            "report N/A this cycle): %s", exc,
        )

    # RC v2 MetricRecord tiles (value + CI + N + status), nested under "tiles".
    # These read their own sources independently of the backtest/{date}/
    # artifacts and land alongside the v1 raw-dict scorecard (research /
    # predictor / executor) during the migration. The unified overall_status
    # roll-up (module_agg.overall_status) activates once research + executor
    # also migrate to MetricRecords (later Phase C increments).
    #   - portfolio_outcome (Tile 0): trades/eod_pnl.csv
    #   - predictor (Tile 2): predictor metrics + weights manifest (LEAK-FREE IC)
    #   - research (Tile 1): backtest/{date}/e2e_lift + score_calibration + macro_eval + portfolio_calibration
    #   - executor (Tile 3): backtest/{date}/trigger_scorecard + shadow_book + exit_timing + portfolio_excursion
    #   - backtester (Tile 4): grading.json coverage audit + parity + attribution FDR + freshness + rollbacks
    #     + live-vs-backtest-promised IC drift (backtest_vs_live_parity, config#1153)
    #   - substrate (Tile 5): price-cache freshness (+ SF/data-quality producers N/A until wired)
    #   - agent (Tile 6): agent-quality transparency shell (producers not yet persisted)
    #   - behavioral (Tile 7): backtest/{date}/behavioral_anomaly + optimizer_shadow
    #     tripwire (L4514/config#698 — all components supporting/diagnostic during soak)
    #   - director_quality (Tile 9): director/retro_trend.json — the Director's own
    #     weekly Phase-G retro grade of its PRIOR plan (config#1674 — WATCH-only,
    #     never cascades to overall RED, same class as agent/behavioral)
    # NINE tiles total; the historical numbering skips 8 (0–7 then 9) — there
    # is no Tile 8. This dict is the membership source of truth (pinned by
    # tests/test_aggregate.py + test_handler.py).
    tiles = {
        "portfolio_outcome": build_portfolio_outcome_tile(
            bucket, s3_client=s3_client, history=history, n_trials=n_trials,
        ),
        "predictor": build_predictor_tile(bucket, run_date, s3_client=s3_client, history=history),
        "research": build_research_tile(bucket, run_date, s3_client=s3_client, history=history),
        "executor": build_executor_tile(bucket, run_date, s3_client=s3_client),
        "backtester": build_backtester_tile(bucket, run_date, s3_client=s3_client, history=history),
        "substrate": build_substrate_tile(bucket, run_date, s3_client=s3_client),
        "agent": build_agent_tile(bucket, run_date, s3_client=s3_client),
        "behavioral": build_behavioral_tile(bucket, run_date, s3_client=s3_client),
        "director_quality": build_director_quality_tile(bucket, run_date, s3_client=s3_client),
    }
    scorecard["tiles"] = tiles
    # Unified RC v2 overall status — worst-of (portfolio outcome leads; a RED in
    # any cascade module fails overall), per module_agg.overall_status. The
    # Backtester / Substrate / Agent tiles join later; overall_status tolerates
    # their absence. Distinct from the v1 scorecard["overall"] letter.
    scorecard["tiles_overall_status"] = overall_status(
        {name: t["status"] for name, t in tiles.items()}
    )

    scorecard["_provenance"] = {
        "run_date": run_date,
        "grader_source": GRADER_SOURCE,
        "artifacts": report.as_dict(),
    }
    logger.info(
        "Report card for %s: status=%s overall=%s (%d artifacts read, %d absent)",
        run_date, scorecard["status"], scorecard["overall"]["letter"],
        report.as_dict()["n_read"], report.as_dict()["n_missing"],
    )
    return scorecard


def report_card_key(run_date: str) -> str:
    return f"{REPORT_CARD_PREFIX}/{run_date}/{REPORT_CARD_FILENAME}"


def latest_report_card_key() -> str:
    return LATEST_REPORT_CARD_KEY


def write_report_card(
    bucket: str,
    run_date: str,
    scorecard: dict,
    s3_client=None,
    *,
    snapshot: bool = False,
) -> dict:
    """Persist the report card (config-I2556: persistent surface + weekly snapshot).

    Always overwrites the standing ``evaluator/latest/report_card.json``
    pointer with this (full-rebuild) card — the continuously-maintained
    surface any producer can refresh on its own cadence by tail-invoking the
    grading Lambda. ``snapshot=True`` ALSO writes the dated
    ``evaluator/{run_date}/report_card.json`` — the frozen weekly record that
    ``history.py``'s cross-cycle trend loader and the Director's advisory read
    (``director/handler.py``) consume; a moving ``latest`` must never leak
    into either of those (stable-snapshot inputs).

    ``snapshot`` DEFAULT: ``False`` (mirrored by ``grading.handler.handler``'s
    ``event.get("snapshot", False)``). ``feat/weekly-sf-advisory-child-and-
    sunday-zoo`` (nousergon-data PR #832) merged 2026-07-14 — both production
    callers now pass this flag explicitly (``True`` for the Saturday
    advisory-child freeze, ``False`` for the Sunday ModelZoo re-grade tail
    invoke), so an absent flag no longer needs to preserve the old
    always-dated behavior; it now means "refresh latest only," the correct
    default for the persistent-surface model where a frozen weekly snapshot
    is the deliberate exception.

    Returns ``{"latest_key": str, "dated_key": str | None}`` (``dated_key`` is
    ``None`` when ``snapshot=False``).
    """
    s3 = s3_client or boto3.client("s3")
    body = json.dumps(scorecard, indent=2, default=str).encode("utf-8")

    latest_key = LATEST_REPORT_CARD_KEY
    s3.put_object(
        Bucket=bucket, Key=latest_key, Body=body, ContentType="application/json",
    )
    logger.info("Wrote report card to s3://%s/%s (latest)", bucket, latest_key)

    dated_key = None
    if snapshot:
        dated_key = report_card_key(run_date)
        s3.put_object(
            Bucket=bucket, Key=dated_key, Body=body, ContentType="application/json",
        )
        logger.info("Wrote report card to s3://%s/%s (weekly snapshot)", bucket, dated_key)

    return {"latest_key": latest_key, "dated_key": dated_key}


def _letters(scorecard: dict) -> dict[str, str]:
    """Flatten a scorecard to {path: letter} for parity comparison."""
    out: dict[str, str] = {"overall": scorecard.get("overall", {}).get("letter", "N/A")}
    for module in ("research", "predictor", "executor"):
        mod = scorecard.get(module) or {}
        out[module] = mod.get("letter", "N/A")
        for comp_name, comp in (mod.get("components") or {}).items():
            # sector_teams is a list; the rest are component dicts.
            if isinstance(comp, dict) and "letter" in comp:
                out[f"{module}.{comp_name}"] = comp["letter"]
    return out


def compare_to_backtester(
    bucket: str,
    run_date: str,
    scorecard: dict,
    s3_client=None,
) -> dict:
    """Diff the evaluator's letter grades vs the backtester's grading.json.

    Observe-mode parity check. Returns a dict of {path: {evaluator, backtester}}
    for every path where the two disagree (plus a summary). A clean parallel run
    has ``mismatches == {}`` on the paths the backtester also grades.
    """
    s3 = s3_client or boto3.client("s3")
    try:
        resp = s3.get_object(Bucket=bucket, Key=f"backtest/{run_date}/grading.json")
        bt = json.loads(resp["Body"].read())
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return {"status": "no_backtester_grading", "mismatches": {}}
        raise

    ev_letters = _letters(scorecard)
    bt_letters = _letters(bt)
    mismatches: dict[str, dict] = {}
    for path, bt_letter in bt_letters.items():
        ev_letter = ev_letters.get(path, "MISSING")
        if ev_letter != bt_letter:
            mismatches[path] = {"evaluator": ev_letter, "backtester": bt_letter}
    return {
        "status": "compared",
        "n_paths": len(bt_letters),
        "n_mismatch": len(mismatches),
        "mismatches": mismatches,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the evaluator report card from S3 artifacts (observe mode).")
    parser.add_argument("--date", required=True, help="run date (ISO, e.g. 2026-06-06)")
    parser.add_argument("--bucket", default="alpha-engine-research", help="S3 bucket")
    parser.add_argument("--write", action="store_true", help="persist to the evaluator namespace (always overwrites evaluator/latest/report_card.json)")
    parser.add_argument("--no-snapshot", dest="snapshot", action="store_false", default=True,
                         help="with --write: skip the dated evaluator/{date}/report_card.json weekly snapshot (writes latest only)")
    parser.add_argument("--compare", action="store_true", help="diff letter grades vs the backtester's grading.json")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    scorecard = build_report_card(args.bucket, args.date)
    print(json.dumps(scorecard, indent=2, default=str))

    if args.compare:
        parity = compare_to_backtester(args.bucket, args.date, scorecard)
        print("\n--- parity vs backtester grading.json ---")
        print(json.dumps(parity, indent=2, default=str))

    if args.write:
        written = write_report_card(args.bucket, args.date, scorecard, snapshot=args.snapshot)
        print(f"\nWrote s3://{args.bucket}/{written['latest_key']} (latest)")
        if written["dated_key"]:
            print(f"Wrote s3://{args.bucket}/{written['dated_key']} (weekly snapshot)")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
