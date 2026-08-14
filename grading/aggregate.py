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
from grading.attestation import build_run_attestation, verdict_is_pass
from grading.freshness_preflight import assert_input_freshness
from grading.history import load_card_history
from grading.scorecard import compute_scorecard
from grading.pipeline_gates import gates_unmeasured, read_gate_state
from grading.self_test import run_self_test
from grading.self_test import verdict_is_pass as self_test_is_pass
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
    self_test: dict | None = None,
    gate_state: dict | None = None,
) -> dict:
    """Read artifacts → grade → attach provenance. Pure of writes.

    Hard input-freshness preflight FIRST (alpha-engine-config#3058, Brian
    ruling 2026-07-20): "if the evaluator is evaluating on stale data its
    report is COMPLETELY USELESS — it should hard-fail before evaluating
    stale outputs." ``assert_input_freshness`` raises
    ``MissingInputArtifactError``/``StaleInputArtifactError`` (uncaught here
    by design — the SF state must fail loud, rc != 0) before ANY tile reads
    an artifact. This runs unconditionally, including on a ``skip_*``-
    flagged partial rerun — the exact scenario (a recovery rerun that skips
    the producer stage) that makes a consumer-side gate load-bearing.
    """
    freshness_provenance = assert_input_freshness(bucket, run_date, s3_client=s3_client)

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

    # config#2885: top-level degraded_staleness flag — true when ANY tile
    # reports a stale artifact (detected by scanning tile components' na_detail
    # for the "stale input" reason text each per-call-site staleness check
    # produces). The Director agent's prompt MUST check this before treating
    # the card as ground truth (advisory hardening — freshness_preflight.py
    # already hard-fails the snapshot path on the core inputs, but per-tile
    # staleness catches edge cases the preflight doesn't cover).
    stale_tiles: list[str] = []
    for name, t in tiles.items():
        for c in (t.get("components") or []):
            if isinstance(c, dict) and "stale input" in (c.get("na_detail") or ""):
                stale_tiles.append(name)
                break
    scorecard["degraded_staleness"] = bool(stale_tiles)
    if stale_tiles:
        scorecard["stale_tiles"] = sorted(stale_tiles)

    # sf-pipeline-policy §2.3a — the run's CORRECTNESS VERDICT, distinct from the
    # freshness gate above. Freshness answers "are the inputs current"; this
    # answers "is the arithmetic that produced them still right". Both halves are
    # known-answer batteries run at RUNTIME on the deployed wheels: the
    # evaluator's own quant primitives here, and the backtest engine's verdict
    # read from backtest/{run_date}/attestation.json.
    #
    # A missing verdict propagates as UNKNOWN, never as a pass (rule 2), and every
    # surface presenting the run's numbers carries the state (rule 3): the card
    # here, the Backtester tile's `numeric_attestation` critical component, and the
    # Director digest. Never raises — a dead verdict stage must not kill the card,
    # and must equally not let it render as verified.
    attestation = build_run_attestation(bucket, run_date, s3_client=s3_client)
    scorecard["attestation"] = attestation
    scorecard["degraded_attestation"] = not verdict_is_pass(attestation["verdict"])

    # The published known-answer SELF-TEST (Brian, 2026-08-13; config#7238). Same
    # clause, second surface: §2.3a rule 3 — every surface presenting the run's
    # results carries the verdict state. The attestation above is the terse
    # machine verdict; this carries the EVIDENCE behind it — every case's inputs,
    # hand-derived expectation, observation, error and tolerance, plus the
    # importlib.metadata version of every quant distribution actually loaded into
    # this image. Full body at evaluator/{run_date}/self_test.json.
    #
    # `self_test` is threaded in by `grading/handler.py`, which runs the battery
    # BEFORE this build so the answer to "can this image's arithmetic be trusted"
    # does not depend on the card build succeeding. It falls back to running the
    # battery here so a caller that forgets cannot silently drop the verdict —
    # `run_self_test` never raises, and a missing one would otherwise read as a
    # card with nothing to declare rather than a card nobody checked.
    #
    # `degraded_self_test` is DERIVED from the verdict rather than set
    # independently, so the two can never disagree and an absent or unrecognised
    # verdict reads as degraded — never as a pass.
    self_test_body = self_test if self_test is not None else run_self_test(run_date)
    scorecard["self_test"] = self_test_body
    scorecard["degraded_self_test"] = not self_test_is_pass(self_test_body.get("verdict"))
    # config#7199 — the contamination claim, flagged SEPARATELY from the
    # arithmetic one. "Did we compute this right" and "could the input have seen
    # the future" are two questions; a single degraded flag answers neither
    # specifically, and the second is the one an external reader asks first.
    scorecard["degraded_contamination"] = not verdict_is_pass(
        attestation.get("contamination_verdict")
    )
    scorecard["contamination_verdict"] = attestation.get("contamination_verdict")

    # alpha-engine-config-I7282 — §2.3a rule 3, third surface and the one that
    # was carrying NOTHING. `attestation` and `self_test` above answer "is the
    # arithmetic behind these numbers right"; this answers the prior question
    # "did the pipeline's own pre-spend correctness gates run at all this
    # cycle". Until this landed the SF sent the ReportCard Lambda
    # {date, dry_run, snapshot} and the card rendered identically whether the
    # gates passed, failed, or (as PipelineContractGate has done on every
    # production run since it existed — alpha-engine-config-I7281) never
    # measured anything.
    #
    # `gate_state` is threaded in by grading/handler.py from the SF payload.
    # A caller that omits it does NOT get a pass: read_gate_state(None) resolves
    # to UNKNOWN with the cause recorded, which is exactly the pre-I7282 state
    # of the world and must read as such rather than as silence.
    #
    # `degraded_pipeline_gates` is DERIVED from the verdict, never set
    # independently, so the two cannot disagree. `status` is deliberately NOT
    # moved to "partial" by an unmeasured gate — see pipeline_gates.py's module
    # docstring for why (a permanently-amber field is a field nobody reads).
    gate_block = read_gate_state(gate_state)
    scorecard["pipeline_gates"] = gate_block
    scorecard["degraded_pipeline_gates"] = gates_unmeasured(gate_block)

    # A card whose correctness verdict did not come back cannot present itself
    # as a complete build. On 2026-08-07 the contamination check timed out and
    # the card was still written status "ok", degraded_staleness false, grade
    # 55.7 — nothing distinguished a non-answer from a pass. `partial` is the
    # existing vocabulary for "this card is not fully established" (the
    # report_card schema enum is exactly ok | partial | insufficient_data), so
    # this reuses it rather than minting a fourth value no consumer handles.
    # `insufficient_data` is never upgraded — a card that could not be graded at
    # all stays that way.
    if scorecard.get("status") == "ok" and scorecard["degraded_attestation"]:
        scorecard["status"] = "partial"
        scorecard["status_reason"] = (
            f"correctness verdict {attestation['verdict']} — "
            f"{attestation.get('reason', '')}"
        )

    # The frozen write-contract's version stamp
    # (nousergon_lib/contracts/report_card.schema.json, `schema_version` const 1).
    # Found unstamped 2026-08-12 while adding the producer contract test below
    # (config-I7039): the contract has declared this REQUIRED since config#2343
    # and the producer has never emitted it, so every card this repo has ever
    # written is invalid against its own contract — invisibly, because nothing
    # validated. Consumers are tolerant on read, so this is additive and safe;
    # `tests/test_report_card_producer_contract.py` is what stops it recurring.
    scorecard["schema_version"] = 1

    scorecard["_provenance"] = {
        "run_date": run_date,
        "grader_source": GRADER_SOURCE,
        "artifacts": report.as_dict(),
        "freshness_preflight": freshness_provenance,
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

    Freshness preflight mirror (alpha-engine-config#3058): ``snapshot=True``
    freezes the dated weekly record — the worst-case outcome the issue names
    is a FROZEN stale report, so the gate is re-asserted here, independent of
    whether the caller already ran it via ``build_report_card``. Belt-and-
    braces: a future caller that builds a card once and snapshots it later
    (or re-snapshots a previously-built card) must not bypass the gate. Not
    re-run for a bare ``latest``-only refresh (``snapshot=False``) — that
    path never freezes a weekly record, so it stays governed by
    ``build_report_card``'s own preflight.
    """
    s3 = s3_client or boto3.client("s3")

    if snapshot:
        assert_input_freshness(bucket, run_date, s3_client=s3)

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
