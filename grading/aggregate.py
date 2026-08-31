"""
aggregate.py — native report-card build (Layer B orchestrator).

Reads the persisted analysis artifacts (``grading/artifacts.py``), runs the
pure grader (``grading/scorecard.py``), attaches provenance, and writes the
report card to the evaluator's own S3 namespace.

OBSERVE / PARALLEL-RUN (Phase C increment 1):
  The evaluator writes ``evaluator/{date}/report_card.json`` — a NEW key,
  deliberately NOT the backtester's ``backtest/{date}/grading.json``. This
  honours the S3-contract-safety "write both for ≥1 week" rule.

  RC v3 T1 (config-I7474, 2026-08-16): the ``--compare`` backtester-parity
  soak check (``compare_to_backtester`` / ``_letters``) is deleted — the
  backtester's v1 letter grade is retired as a rendered surface (its
  ``analysis/grading.py`` no longer emits a "letter" field at all), so a
  letter-vs-letter diff against it is no longer meaningful.

The Lambda handler + Saturday-SF wiring arrive in Phase F; this module exposes
``build_report_card`` / ``write_report_card`` for that handler and a thin CLI
for manual observe runs.
"""

from __future__ import annotations

import argparse
import json
import logging

import boto3

from grading.artifacts import read_scorecard_inputs
from grading.coverage import (
    CENSUS_UNKNOWN_MARKER,
    replace_evaluator_coverage,
    stamp_composite_scope,
)
from grading.attestation import build_run_attestation, verdict_is_pass
from grading.freshness_preflight import assert_input_freshness
from grading.history import load_card_history
from grading.scorecard import compute_scorecard
from grading.pipeline_gates import gates_unmeasured, read_gate_state
from grading.run_scope import RUN_SCOPE_KEY, log_run_scope, read_run_scope, scope_unknown
from grading.self_test import run_self_test
from grading.self_test import verdict_is_pass as self_test_is_pass
from grading.module_agg import overall_status
from grading.thresholds.scoring import build_leaderboard
from nousergon_lib.quant.stats.trial_accumulator import read_cumulative_trial_count
from grading.tiles.agent import build_agent_tile
from grading.tiles.backtester import build_backtester_tile
from grading.tiles.behavioral import build_behavioral_tile
from grading.tiles.contribution_lift import build_contribution_lift_tile
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
    dry_run: bool = False,
    run_scope: dict | None = None,
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
    # dry_run threads to the freshness preflight AND to build_run_attestation
    # (alpha-engine-config-I7392): on the Friday shell run every producer ran
    # --preflight-only and wrote nothing, so the gate records UNMEASURED instead
    # of raising and the attestation marks the halves whose artifact is absent
    # NOT_IN_SCOPE instead of UNKNOWN. The real run is untouched — see
    # assert_input_freshness for why that does not weaken the 2026-07-20 ruling
    # and for what still raises on the dry path, and
    # _mark_rehearsal_out_of_scope for the three ways the attestation half is
    # deliberately narrow.
    #
    # Threading it to the preflight ALONE is what left the rehearsal paging: the
    # card stopped hard-failing and started logging
    # "report card attestation UNKNOWN ... the producer never ran this cycle"
    # every Friday instead, most recently at 2026-08-29T01:44Z from execution
    # offcycle-shell-20260829-004717 — a run in which nothing failed.
    freshness_provenance = assert_input_freshness(
        bucket, run_date, s3_client=s3_client, dry_run=dry_run,
    )

    inputs, report = read_scorecard_inputs(
        bucket, run_date, s3_client=s3_client, run_scope_payload=run_scope,
    )
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
    # config#7476 — the threshold champion/challenger slot. Both arms are scored
    # on predictive validity against the REALIZED next-cycle objective, every
    # cycle, whether or not anything could be promoted (champion-challenger §3).
    # It reads only PRIOR cards, so it is built before the tiles and rendered on
    # the substrate tile as one machine-health record per arm.
    #
    # Fail-SOFT with the reason carried, never silent (§7.2): a bug in the
    # scoring must not fail the report card, but the failure reaches the card as
    # an N/A-NOT-RUN naming it — an unscored cycle is unrecoverable and must
    # never read as absence.
    threshold_leaderboard: dict | None = None
    threshold_leaderboard_error: str | None = None
    try:
        threshold_leaderboard = build_leaderboard(bucket, run_date, s3_client=s3_client)
    except Exception as exc:  # noqa: BLE001 — secondary scoring, reported not raised
        threshold_leaderboard_error = (
            f"threshold slot scoring failed for {run_date}: {type(exc).__name__}: {exc}"
        )
        logger.error(threshold_leaderboard_error, exc_info=True)

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
    #   - contribution_lift (Tile 10, RC v3 T5, config-I7473): backtest/{date}/
    #     contribution_lift.json — each component's measured marginal
    #     contribution to the T2 objective. Not in _CASCADE_MODULES (module_agg.py);
    #     each record's own `module` field names its REAL owning tile
    #     (research/predictor/executor/behavioral) so it renders beside that
    #     component on any surface grouping by a record's own `.module`.
    # TEN tiles total; the historical numbering skips 8 (0–7, 9, 10) — there
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
        "substrate": build_substrate_tile(
            bucket, run_date, s3_client=s3_client,
            threshold_leaderboard=threshold_leaderboard,
            threshold_leaderboard_error=threshold_leaderboard_error,
        ),
        "agent": build_agent_tile(bucket, run_date, s3_client=s3_client),
        "behavioral": build_behavioral_tile(bucket, run_date, s3_client=s3_client),
        "director_quality": build_director_quality_tile(bucket, run_date, s3_client=s3_client),
        "contribution_lift": build_contribution_lift_tile(bucket, run_date, s3_client=s3_client),
    }
    # alpha-engine-config-I8177 — `evaluator_coverage` grades THIS card, not the
    # legacy v1 `grading.json`.
    #
    # The backtester tile builds the record from `backtest/{date}/grading.json`,
    # a 14-leaf artifact covering research/predictor/executor only. It cannot
    # see agent, substrate, behavioral, contribution_lift, portfolio_outcome,
    # director_quality or backtester — seven of these ten tiles. On 2026-08-22
    # it rendered 0.857 (WATCH) while the real surface was 78/125 = 0.624,
    # below the red-line: the metric named for the coverage cliff was
    # structurally unable to see it.
    #
    # It has to happen HERE rather than inside the tile builder: the census is
    # over every tile, and no single builder can see its siblings. The tile
    # keeps emitting its record (so its own shape and tests stand); this
    # substitutes the value once the full census exists.
    #
    # I8193: the denominator is the THRESHOLD REGISTRY roster, not the set of
    # components that happened to render. A tile deleted from the dict above,
    # or a builder that silently returns fewer records, now shows up as N
    # UNREPORTED components inside the denominator — coverage falls. Counting
    # what rendered made the number rise when a tile vanished, which is the
    # same "true number about a smaller world" this metric was repointed to
    # escape.
    census = replace_evaluator_coverage(tiles)
    # Same defect, one level up (alpha-engine-config-I8177): the v1 `overall`
    # block reported `components_declared: 3` / `qualifier: COMPLETE` on a card
    # carrying 125 leaf components, 47 of them N/A, and `tiles_overall_status:
    # RED`. True of its three modules, false of the card it ships on. This
    # stamps what it covers and demotes the qualifier; it does not touch the
    # grade, whose weights are a Brian ruling (`alpha-engine-config-I7210`).
    stamp_composite_scope(scorecard, tiles)
    scorecard["tiles"] = tiles
    # Handed to the handler (which persists it as its own artifact) under a
    # leading underscore, the same convention `_provenance` uses for a key that
    # is build output rather than card content. The handler pops it before the
    # card is written, so the card itself does not carry the whole leaderboard.
    scorecard["_threshold_leaderboard"] = threshold_leaderboard
    # Unified RC v2 overall status — worst-of (portfolio outcome leads; a RED in
    # any cascade module fails overall), per module_agg.overall_status. The
    # Backtester / Substrate / Agent tiles join later; overall_status tolerates
    # their absence. Distinct from the v1 scorecard["overall"] letter.
    scorecard["tiles_overall_status"] = overall_status(
        {name: t["status"] for name, t in tiles.items()}
    )
    # alpha-engine-config-I8177 — tiles in which NOTHING graded, named at the
    # top of the card rather than inferable only by opening every component
    # list. Zero-length is a VALUE here: an empty list asserts every tile
    # carried at least one real measurement, which is what makes a non-empty
    # one readable as the finding it is. Measured 2026-08-22: `agent` was 11
    # N/A of 11 and rendered WATCH/C, indistinguishable from a tile that
    # measured everything and came out borderline.
    # alpha-engine-config-I8193 — the census's OWN health, at the top of the
    # card. A coverage number is only as good as its denominator, so the two
    # ways that denominator can be wrong get a surface of their own rather
    # than living inside a component's `coverage_census` sub-object: a
    # registered component that rendered nothing (`unreported`), and a graded
    # component with no registry row (`unregistered`). `false` is a VALUE —
    # it asserts the roster and the card agreed this cycle.
    if census is None:
        scorecard["degraded_component_census"] = True
        scorecard["component_census_error"] = CENSUS_UNKNOWN_MARKER
    else:
        missing = list(census["unreported"]) + list(census["unregistered"])
        scorecard["degraded_component_census"] = bool(missing)
        if missing:
            scorecard["component_census_unreported"] = census["unreported"]
            scorecard["component_census_unregistered"] = census["unregistered"]

    scorecard["tiles_unmeasured"] = {
        name: t["status"]
        for name, t in sorted(tiles.items())
        if str(t.get("status", "")).startswith("N/A")
    }

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
    # The scope is DERIVED FIRST because the attestation depends on it: the
    # contamination half's producer is gated by `skip_parity`, and only the
    # scope artifact can tell a stage that was switched off from one that died
    # (config-I7620 follow-up). Read here, rendered onto the card lower down
    # where the denominator belongs — one read, two consumers.
    # The scope arrives IN-BAND with the run (the ReportCard Task's `run_scope`
    # payload key, threaded from $.run_scope_result.Payload) and falls back to
    # the S3 artifact — see artifacts._read_run_scope for why that order was
    # inverted (alpha-engine-config-I7392).
    scope_block = read_run_scope(report.run_scope)
    attestation = build_run_attestation(
        bucket, run_date, s3_client=s3_client, run_scope=scope_block,
        dry_run=dry_run,
    )
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

    # alpha-engine-config-I7620 — the DENOMINATOR. Every grade above is computed
    # over whatever stages this run actually dispatched, and that set moves: the
    # weekly pipeline carries 29 skip gates and an operator flipping one changes
    # which producers ran without changing anything else the card says. The
    # 2026-08-16 execution terminated SUCCEEDED having dispatched 3 of 29, and
    # nothing on any surface said so.
    #
    # `report.run_scope` is the artifact `RunScope` wrote for THIS run. An absent
    # or degraded block resolves to UNKNOWN with an empty graded set — never to
    # "everything ran". A card that grades the full stage list against a run that
    # dispatched three of them is confidently wrong; one that says it does not
    # know is merely uninformative.
    #
    # Deliberately does NOT move `status`: the scope is legitimately narrow on
    # every partial rerun, and a permanently-amber field is a field nobody reads
    # (same reasoning as `degraded_pipeline_gates` above). It is rendered BESIDE
    # the grade, which is what was missing, not folded INTO it.
    scorecard[RUN_SCOPE_KEY] = scope_block
    scorecard["scope_unknown"] = scope_unknown(scope_block)
    log_run_scope(scope_block)

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

    # ── Attribution artifact (alpha-engine-config-I8188 deliverables 6-7) ──
    # The Brinson-Fachler decomposition rides on the portfolio tile; persist it
    # as its own artifact so a consumer can read the per-sector, per-session
    # detail without parsing the whole card. Best-effort: (a) swallowed — a
    # write failure for the SECONDARY artifact; (b) the report card itself is
    # the primary deliverable and is already written above; (c) recording
    # surface — the ERROR log here plus the tile's own components, which carry
    # the same headline numbers and their status.
    try:
        attribution = (scorecard.get("tiles") or {}).get(
            "portfolio_outcome", {}
        ).get("attribution")
        if attribution:
            from grading.attribution import write_attribution

            write_attribution(
                attribution, bucket=bucket,
                run_date=run_date if snapshot else None, s3_client=s3,
            )
    except Exception as e:  # noqa: BLE001 — secondary artifact, see above
        logger.error("attribution artifact write failed for %s: %s", run_date, e)

    dated_key = None
    if snapshot:
        dated_key = report_card_key(run_date)
        s3.put_object(
            Bucket=bucket, Key=dated_key, Body=body, ContentType="application/json",
        )
        logger.info("Wrote report card to s3://%s/%s (weekly snapshot)", bucket, dated_key)

    return {"latest_key": latest_key, "dated_key": dated_key}


# _letters / compare_to_backtester / --compare deleted RC v3 T1
# (config-I7474, 2026-08-16): the backtester-parity soak check compared
# "letter" fields on BOTH sides — crucible-backtester's analysis/grading.py
# (this repo's compute_scorecard input's counterpart) no longer emits a
# "letter" key on any level (T1 retires the v1 A-F letter as a RENDERED
# grade there), so this comparison would now silently read every path as
# "MISSING" vs the evaluator's own (still-populated) letters — a false
# 100%-mismatch reading, not a real signal. grading/scorecard.py itself
# (the compute_scorecard this module still calls below, INCLUDING its
# coverage/grading_weights renormalization block — config-I7202) is NOT
# touched: it is not a pure duplicate of the retired v1 grader and a
# currently-passing producer-contract test
# (tests/test_report_card_producer_contract.py::TestCoverageBlock) depends
# on it. Deleting grading/scorecard.py wholesale, as this issue's deliverable
# 6 literally reads, would break that contract — flagged in the PR body for
# a ruling rather than guessed through.


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the evaluator report card from S3 artifacts (observe mode).")
    parser.add_argument("--date", required=True, help="run date (ISO, e.g. 2026-06-06)")
    parser.add_argument("--bucket", default="alpha-engine-research", help="S3 bucket")
    parser.add_argument("--write", action="store_true", help="persist to the evaluator namespace (always overwrites evaluator/latest/report_card.json)")
    parser.add_argument("--no-snapshot", dest="snapshot", action="store_false", default=True,
                         help="with --write: skip the dated evaluator/{date}/report_card.json weekly snapshot (writes latest only)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    scorecard = build_report_card(args.bucket, args.date)
    print(json.dumps(scorecard, indent=2, default=str))

    if args.write:
        written = write_report_card(args.bucket, args.date, scorecard, snapshot=args.snapshot)
        print(f"\nWrote s3://{args.bucket}/{written['latest_key']} (latest)")
        if written["dated_key"]:
            print(f"Wrote s3://{args.bucket}/{written['dated_key']} (weekly snapshot)")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
