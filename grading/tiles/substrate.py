"""
substrate.py — Tile 5: Substrate Reliability (RC v2).

Operational substrate — a flaky substrate invalidates everything above it
(RC v2 Principle 8). One component is sourceable from S3 today
(``price_cache_freshness`` via the content-derived freshness sentinel,
config#2350); the rest need producers the evaluator can't yet reach (SF/CW
execution history, the
data-quality substrate inventory, GitHub Actions, CFN drift), so they grade a
**transparent N/A-NOT-IMPL whose reason names the producer to build** — the
report card says "the substrate is mostly unmeasured" out loud rather than
hiding it.

``sf_success_rate_4w`` is the headline substrate metric, wired over the 3
Step Function ARNs (Saturday / Weekday / EOD) via
``nousergon_lib.pipeline_status`` — ``list_recent_pipeline_runs`` for
discovery, ``classify_work`` for the per-execution work verdict
(alpha-engine-config-I8069) and ``build_cycle_shape`` for the cycle fold. It
is graded in two scopes published side by side: ``full_scope`` and the
headline ``clean_with_declared_skips``, which puts stages excused by a
REGISTERED cadence-skip witness out of scope (alpha-engine-config-I11987,
option A).

Spec: ``system-report-card-revamp-260522.md`` Tile 5.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from grading.artifacts import get_json, get_json_windowed
# The label a stage excused by a registered cadence-skip witness carries on the
# card — the attestation's own vocabulary for a stage switched off by a recorded
# decision: neither passed nor failed, and its guarantee still missing.
from grading.attestation import NOT_IN_SCOPE
from grading.metric_record import build_metric
from grading.module_agg import build_tile
from grading.producers.cost_pricing import (
    COST_RAW_PREFIX,
    CostPricingUnmeasured,
    format_by_callsite,
    scan_degraded_cost_rows,
)
from grading.producers.deploy_success import DEPLOY_SUCCESS_KEY
from grading.thresholds.scoring import build_arm_components, leaderboard_key
from grading.units import COUNT_EVENTS, COUNT_ROWS, DAYS, FRACTION

logger = logging.getLogger(__name__)

MODULE = "substrate"
# predictor/price_cache/ stopped being written after the Wave-3 PR4 cutover
# moved the live per-ticker cache to reference/price_cache/ (crucible-predictor
# regime/features.py reads the new location; see ARTIFACT_REGISTRY.yaml's
# grandfathered-legacy entry for predictor/price_cache/). Grading raw S3 mtimes
# under the dead legacy prefix produced a monotonically-growing false RED that
# no refresh could ever clear (config#3142) — read the content-derived
# freshness sentinel (config#2350) that the live writer stamps instead.
PRICE_CACHE_FRESHNESS_SENTINEL_KEY = "reference/price_cache/_freshness.json"

# The 3 orchestration Step Functions whose rolling success rate IS the substrate
# headline. ARNs are *discovered* at runtime (list_state_machines) rather than
# hardcoded, so no AWS account id lives in this public repo.
_SF_NAMES = (
    "ne-weekly-freshness-pipeline",
    "ne-preopen-trading-pipeline",
    "ne-postclose-trading-pipeline",
)
_SF_TERMINAL = {"SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"}
_SF_WINDOW_DAYS = 28

#: How deep to walk each SF's execution history for the trailing-window scan.
#:
#: `list_recent_pipeline_runs` is a COUNT-bounded "most recent N" primitive
#: built for the dashboard's recent-executions disclosure; this metric needs a
#: TIME-bounded window. Until the library grows a `since=`-style walk, the
#: count bound has to be set above the busiest SF's real execution rate and
#: the truncation case has to be detected rather than assumed away.
#:
#: Sized from measurement, not guesswork: ne-weekly-freshness-pipeline ran 102
#: terminal executions in the 28 days to 2026-08-22 (its day-gate fires Thu-Sat
#: and reruns are frequent). 400 is ~4x that headroom, and the truncation guard
#: below is what makes an under-estimate loud instead of silent.
#:
#: Cost: each summary is one DescribeExecution on top of one ListExecutions
#: page, so this is O(limit) API calls per SF against a 25-TPS soft limit. It
#: is paid once a week by the grading Lambda, and only walks as deep as the SF
#: actually has executions.
_SF_EXEC_SCAN_LIMIT = 400

# data_quality_incidents — the data collector's per-run quality gate emits
# ``AlphaEngine/Data/daily_append_quality_blocked_count`` (rows EXCLUDED from the
# feature store on a quality failure — the load-bearing incident) +
# ``_warned_count`` (flagged-but-kept) per run (alpha-engine-data
# builders/daily_append.py). We grade the trailing-4w SUM of BLOCKED as the
# incident count; warned rides in the reason. Thresholds calibrated off the
# 2026-06 baseline (~95 blocked / 4w ≈ 0.5%/day of ~900 tickers → GREEN); they
# are a REGRESSION detector (WATCH ≈ 2x, RED ≈ 5x baseline), re-tune once a
# trend establishes. (config#1150 Batch B.)
_DQ_NAMESPACE = "AlphaEngine/Data"
_DQ_BLOCKED_METRIC = "daily_append_quality_blocked_count"
_DQ_WARNED_METRIC = "daily_append_quality_warned_count"

# schema_drift_incidents — the data collector counts every ArcticDB
# StreamDescriptorMismatch / DataError raised on a universe write path
# (``update_batch`` / ``write_batch`` / ``_write_row_backfill_safe``) per run
# and emits ``AlphaEngine/Data/daily_append_schema_drift_count`` (alpha-engine-data
# builders/daily_append.py). UNLIKE data_quality_incidents (a routine row-level
# gate with a ~95/4w steady-state baseline), a schema-drift incident is a HARD
# data-integrity failure: the persisted ArcticDB descriptor no longer matches the
# row being written, the daily_append run FAILS LOUD (counted then re-raised),
# and downstream features are starved until an operator repairs the descriptor.
# It is meant to be ZERO in steady state, so the thresholds are an absolute
# incident count, NOT a regression band: target 0 (any incident is a real event
# worth a WATCH) / red-line 3 (a cluster of ≥3 schema failures in 4w is a
# systemic descriptor regression → RED). (config#1150 Batch B.)
_SCHEMA_DRIFT_METRIC = "daily_append_schema_drift_count"

# watchdog_firings (supporting, config#1151 Batch C) — per-RUN count of backtester
# phases that hit their hard timeout cap and were force-aborted by the phase
# watchdog (crucible-backtester pipeline_common.phase() → substrate_ops.json).
# A firing means a phase burned through its entire silent-compute budget — the
# 2026-04-22 dry-run's 110-min silent stall that motivated the tripwire. Thresholds
# (lower-is-better): 0 firings = GREEN (no phase capped out, healthy); target 0,
# red-line 2 → exactly 1 = WATCH (a single trip can be a transient infra slowdown —
# a slow EBS volume, a one-off GC pause), >=2 = RED (repeated caps = a phase
# systematically exceeding its budget, real degradation needing a root-cause).

# deploy_success_rate (supporting, config#1153 Batch E) — CI/CD deploy-workflow
# health across the code repos, produced weekly by the Director
# (grading.producers.deploy_success) since the evaluator Lambda holds no GitHub
# token. Bands: deploys should almost always ship clean → target 95% / red-line
# 80%. A rollup older than the freshness window means the producer stopped
# running → grade a transparent stale N/A rather than a false-confident number.
_DEPLOY_SUCCESS_MAX_AGE_DAYS = 21

# unpriced_cost_rows (supporting, alpha-engine-config-I11113) — how many rows in
# the trailing-week `decision_artifacts/_cost_raw/` window carry a degraded
# `cost_source` ("unpriced" — krepis could not find a price card for a served
# model; "usage_unreported" — the provider returned no usage to price). The row
# is WRITTEN in both cases and fan-in coverage is satisfied, so `CostCoverageError`
# passes; what is lost is only the dollar figure. That makes this a DEGRADATION,
# not a failure — `supporting` criticality, so it never cascades a tile to RED on
# the critical path, and no new gate.
#
# Bands: target 0 (steady state is every call priced) / red-line 25. A single
# unpriced row is a WATCH-worthy one-off — a model served once before its card
# landed. A SUSTAINED gap is what the 2026-09-12 glm-5.3 promotion produced: one
# missing card silently unpriced every `ultra` call for days, which is tens of
# rows per cycle, not a handful. 25 is set above the one-off and below the
# sustained case deliberately; it is not a hand-tuned literal but the boundary
# between those two shapes, and it lives in the threshold registry like every
# other band (config#7476).
_UNPRICED_COST_ROWS_WINDOW_DAYS = 7


def _cw_metric_sum(cw, namespace: str, metric: str, as_of: datetime, window_days: int) -> float | None:
    """Trailing-``window_days`` SUM of a CloudWatch metric, or None if no data."""
    resp = cw.get_metric_statistics(
        Namespace=namespace,
        MetricName=metric,
        StartTime=as_of - timedelta(days=window_days),
        EndTime=as_of,
        Period=window_days * 86400,
        Statistics=["Sum"],
    )
    points = resp.get("Datapoints") or []
    if not points:
        return None
    return float(sum(p.get("Sum", 0.0) for p in points))

# SCHEDULED-cadence roles: an EventBridge-triggered run that is SUPPOSED to
# complete on its own (the "unattended" target). Everything else role-carrying
# (recovery / operator / operator-replay / backfill / shell-run) is an operator
# intervention — it still completes the cycle, but its presence means the
# scheduled run did NOT succeed unattended. (config#1059 / #970 / L4552d.)
_SCHEDULED_ROLES = {"weekly", "saturday", "daily", "eod"}


def _discover_sf_arns(sfn) -> list[str]:
    """Resolve the 3 pipeline SF ARNs by name (no hardcoded account id).

    Env override ``EVALUATOR_SF_ARNS`` (comma-separated) wins — lets the Lambda
    skip the ListStateMachines call / IAM grant if the ARNs are configured.
    """
    env = os.environ.get("EVALUATOR_SF_ARNS")
    if env:
        return [a.strip() for a in env.split(",") if a.strip()]
    arns: list[str] = []
    paginator = sfn.get_paginator("list_state_machines")
    for page in paginator.paginate():
        for sm in page.get("stateMachines", []):
            if sm.get("name") in _SF_NAMES:
                arns.append(sm["stateMachineArn"])
    return arns


#: Pipelines whose cycle is keyed on the TRADING day (alpha-engine-config-I8809).
#:
#: A scheduled weekly execution carries no ``input.run_date`` and an opaque
#: name, so its key falls through to its own ``startDate`` — the Saturday —
#: while every recovery rerun of the same cycle carries the Friday trading day.
#: Measured 2026-10-05 on the live state machine: the 2026-10-02 cycle's
#: scheduled run keyed ``2026-10-03`` and its three watch-reruns keyed
#: ``2026-10-02``, so the card graded ONE week as TWO cycles — a scheduled
#: failure with no recovery beside it and a recovery with no scheduled run.
#: That split makes a first-pass/recovered count meaningless, which is why
#: ``alpha-engine-config-I11987`` needs it closed. ``resolve_trading_day`` is
#: idempotent on a trading day, so a key that already is one is unchanged.
#: Scoped to the weekly SF: the two daily SFs run ON the session they serve.
_TRADING_DAY_KEYED_SFS = frozenset({"ne-weekly-freshness-pipeline"})

def _read_execution(execution_arn: str, sfn) -> tuple[object, tuple[str, ...], dict]:
    """``(WorkOutcome, entered_states, input)`` for one execution, ONE history walk.

    The per-execution verdict is the same ``classify_work`` predicate
    ``read_work_outcome`` applies (alpha-engine-config-I8069). This reads the
    history itself because the cycle fold needs two things that function
    discards: every entered state name — the registered cadence-skip witness
    (``MarkParityVerdictUnknownByCadence``) is not a spine stage, so
    ``WorkOutcome.stages_entered`` never carries it — and the execution input,
    which says whether the run was a rehearsal. Calling both would walk every
    history twice.
    """
    from nousergon_lib.pipeline_status import RunStatus, classify_work, entered_states_from_history

    desc = sfn.describe_execution(executionArn=execution_arn)
    sm_arn = str(desc.get("stateMachineArn") or "")
    sm_name = sm_arn.rsplit(":", 1)[-1] if sm_arn else execution_arn.split(":")[6]
    events: list = []
    kwargs = {"executionArn": execution_arn, "maxResults": 1000}
    while True:
        page = sfn.get_execution_history(**kwargs)
        events.extend(page.get("events") or [])
        token = page.get("nextToken")
        if not token:
            break
        kwargs["nextToken"] = token
    states = tuple(entered_states_from_history(events))
    start, stop = desc.get("startDate"), desc.get("stopDate")
    duration = (stop - start).total_seconds() if start and stop else None
    outcome = classify_work(
        state_machine_name=sm_name,
        status=RunStatus(str(desc.get("status"))),
        entered_states=list(states),
        duration_sec=duration,
        execution_arn=execution_arn,
        execution_name=str(desc.get("name") or "") or None,
    )
    try:
        payload = json.loads(desc.get("input") or "{}")
    except (TypeError, ValueError):
        payload = {}
    return outcome, states, payload if isinstance(payload, dict) else {}


def _cycle_key(sf_name: str, run_date: str | None, start: datetime) -> str:
    """The cycle an execution belongs to — see ``_TRADING_DAY_KEYED_SFS``."""
    key = run_date or start.date().isoformat()
    if sf_name in _TRADING_DAY_KEYED_SFS:
        from krepis.dates import resolve_trading_day

        try:
            return resolve_trading_day(key)
        except ValueError:
            logger.warning("sf_success_rate: cannot resolve %r to a trading day — keyed as-is", key)
    return key


def _credited(outcome, states):
    """The work verdict a cycle fold may credit for this execution.

    A FAILED / TIMED_OUT / ABORTED execution ENTERED the stage it died in, and
    ``build_cycle_shape`` folds entered stages. Crediting that stage would let
    a cycle whose last run failed inside, say, ``Director`` read as complete —
    the 2026-09-26 scheduled run did exactly that, ``DegradedRun`` on a
    degraded Director after entering it. So the last spine stage a failed
    execution entered, in ENTRY order, is withheld; every earlier one was left
    behind for a later state and is evidence of work done.
    """
    if outcome.reason != "execution_failed" or not outcome.stages_entered:
        return outcome
    spine_entered = set(outcome.stages_entered)
    last = next((s for s in reversed(states) if s in spine_entered), None)
    if last is None:
        return outcome
    return replace(outcome, stages_entered=tuple(s for s in outcome.stages_entered if s != last))


def _fold(sf_name: str, key: str, execs: list) -> dict:
    """Grade one cycle in both scopes. ``execs`` = ``[(status, role, outcome, states)]``.

    - ``full_scope``: the declared spine, every stage, nothing excused. Either
      one execution entered all of it (``classify_work`` ``full_run``, the
      alpha-engine-config-I8069 rule) or the cycle's contributing executions
      did between them (``build_cycle_shape`` union, no declared skip).
    - ``declared``: the same fold, except that a stage excused by a
      REGISTERED cadence-skip witness (``CADENCE_SKIP_MARKER_STAGES`` — the
      cycle's own walk entered the marker state) is out of scope. Brian's
      ruling on alpha-engine-config-I11987, option A. An input flag is never
      read: ``skip_rag_ingestion: true`` on the trigger excuses nothing.

    Both require at least one contributing execution that SUCCEEDED and did
    some spine work — a cycle whose every run failed is not clean in either
    scope however much of the spine they entered — and the fold credits a
    failed execution only with the stages it LEFT (``_credited``).
    """
    from nousergon_lib.pipeline_status import RunStatus, build_cycle_shape

    full_run = any(outcome.did_work for _st, _role, outcome, _s in execs)
    shape = build_cycle_shape(
        pipeline=sf_name,
        run_date=key,
        outcomes=[(_credited(outcome, states), role, states) for _st, role, outcome, states in execs],
    )
    succeeded_with_work = any(
        outcome.status is RunStatus.SUCCEEDED and outcome.stages_entered
        for _st, _role, outcome, _s in execs
    )
    union_complete = bool(shape.stage_spine) and shape.did_work and succeeded_with_work
    full_scope = full_run or (union_complete and not shape.has_declared_skips)
    declared = full_scope or union_complete
    return {
        "full_scope": full_scope,
        "declared": declared,
        "declared_skipped": shape.stages_declared_skipped if declared and not full_scope else (),
        "shape": shape,
    }


def _sf_success_rate(
    sfn, as_of: datetime, window_days: int,
) -> dict | None:
    """Cycle-level success across the 3 SFs over the trailing window.

    Returns two distinct, complementary metrics (config#1059 / #970 / L4552d) —
    the OLD per-execution rate conflated operator-recovered cycles AND scheduled
    failures into one false-RED "everything is broken" number:

    - ``cycle_rate`` (distinct-cycle outcome): a TRADING CYCLE that ultimately
      completed clean = success, REGARDLESS of how many recovery runs it took.
    - ``unattended_rate`` (first-pass / no-operator): the SCHEDULED run
      (pipeline_role ∈ scheduled-cadence) succeeded with NO recovery run in the
      same cycle.

    **Two scopes, side by side (alpha-engine-config-I11987, option A).** Each
    cycle is graded twice by ``_fold``:

    - ``full_scope`` — every declared spine stage ran. This is what
      ``cycle_rate`` meant before I11987, now also crediting a recovery that
      completed the spine across executions (``build_cycle_shape``).
    - ``declared`` (``clean_with_declared_skips``) — stages excused by a
      REGISTERED cadence-skip witness are out of scope (``NOT_IN_SCOPE``). The
      witness is a state the cycle's own walk entered, mapped to the stages it
      excuses by ``nousergon_lib.pipeline_status.CADENCE_SKIP_MARKER_STAGES``;
      absence is never a skip, and the execution input's ``skip_*`` flags are
      never read. ``cycle_rate`` / ``unattended_rate`` are this scope; the
      ``*_full_scope`` keys carry the other, and every declared-only clean
      cycle is counted (``n_cycles_clean_with_declared_skips``) and named per
      stage (``declared_skips``). A declared-out stage's assurance is MISSING,
      not passed — for parity that is the contamination half, which the card's
      attestation reports separately; this rate adds no contamination guarantee.

    Clean cycles are further split ``first_pass`` (a scheduled run alone was
    clean and no recovery ran) versus ``recovered`` (anything else that ended
    clean), so cumulative recovery never reads as unattended success.

    **Cycle key:** ``input.run_date`` falling back to the start date, both
    normalised to the trading day for ``_TRADING_DAY_KEYED_SFS`` (see there).
    Rehearsals (an execution whose input declares ``rehearsal``) are excluded
    entirely and counted: they dry-run the graph under a recovery role, and
    folded into a cycle their entered stages would read as work done.

    **Point in time (alpha-engine-config-I12059):** the window is
    ``[as_of - window_days, as_of]`` at both ends, start AND stop. The listing
    is read now, so a historical ``as_of`` would otherwise grade executions
    that had not started on the day it describes (measured 2026-10-06: the
    10-04 calculation credited a 10-05 postclose run). An execution that
    started after ``as_of`` is excluded and counted per SF; one that stopped
    after ``as_of`` was in flight on the day and is excluded and named, so a
    later recovery never repairs an earlier cutoff's cycle; a terminal
    execution with no stop instant makes its cycle unevaluable (withheld,
    named, and the caller renders N/A — never GREEN).

    **Work verdict (alpha-engine-config-I8069):** every terminal, role-carrying,
    in-window execution is classified with ``classify_work`` against its own
    entered-state history; ``SKIPPED`` executions (e.g. ``WeeklyRunDaySkip``)
    are excluded from every denominator.

    Returns ``{cycle_rate, n_cycles, n_cycles_clean, unattended_rate,
    n_unattended, n_unattended_ok, per_sf, per_sf_unattended, ...full-scope and
    split keys..., scope_detail, truncated}`` or ``None`` when no SF ARNs are
    discoverable.
    """
    from nousergon_lib.pipeline_status import list_recent_pipeline_runs

    arns = _discover_sf_arns(sfn)
    if not arns:
        return None
    cutoff = as_of - timedelta(days=window_days)
    n_cycles = n_clean = n_clean_full = 0
    n_clean_first_pass = n_clean_recovered = 0
    n_unatt = n_unatt_ok = n_unatt_ok_full = 0
    n_rehearsals = 0
    per_sf: dict[str, str] = {}
    per_sf_full: dict[str, str] = {}
    per_sf_unattended: dict[str, str] = {}
    per_sf_unattended_full: dict[str, str] = {}
    declared_skips: dict[str, dict[str, int]] = {}
    scope_detail: dict[str, str] = {}
    truncated: list[str] = []
    n_after_as_of: dict[str, int] = {}
    in_flight_at_as_of: list[str] = []
    temporally_unevaluable: list[str] = []
    for arn in arns:
        sf_name = arn.rsplit(":", 1)[-1]
        runs = list_recent_pipeline_runs(arn, limit=_SF_EXEC_SCAN_LIMIT, client=sfn)
        # WINDOW TRUNCATION (alpha-engine-config-I8183): a count-bounded scan
        # that filled up while its oldest execution is still inside the window
        # describes a shorter period than the metric's name. Detected and
        # refused, never silently published.
        if len(runs) >= _SF_EXEC_SCAN_LIMIT:
            oldest = min((r.start_utc for r in runs if getattr(r, "start_utc", None)), default=None)
            if oldest is not None and oldest > cutoff:
                logger.error(
                    "sf_success_rate: %s returned %d executions (scan limit) whose oldest "
                    "starts %s, still inside the %dd window opening %s — the window was "
                    "TRUNCATED and any rate computed from it would describe a shorter "
                    "period than its own name (alpha-engine-config-I8183). Raise "
                    "_SF_EXEC_SCAN_LIMIT or move to a time-bounded walk.",
                    sf_name, len(runs), oldest, window_days, cutoff,
                )
                truncated.append(sf_name)
        cycles: dict[str, list[tuple[str, str, object, tuple[str, ...]]]] = {}
        # Cycles holding an execution whose terminal instant cannot be placed
        # against ``as_of`` (alpha-engine-config-I12059). Withheld from every
        # denominator below and named, never folded on a guess.
        unevaluable_keys: set[str] = set()
        for r in runs:
            start = getattr(r, "start_utc", None)
            if start is None or start < cutoff:
                continue
            # POINT-IN-TIME (alpha-engine-config-I12059): the window is
            # [as_of - window_days, as_of], both ends. The SF listing is read
            # NOW, so without this bound a historical as_of graded executions
            # that had not started yet on the day it describes.
            if start > as_of:
                n_after_as_of[sf_name] = n_after_as_of.get(sf_name, 0) + 1
                continue
            # PRODUCTION runs only: ad-hoc smoke / legacy runs carry no role.
            role = getattr(r, "pipeline_role", None)
            if role is None:
                continue
            raw = getattr(r, "status", None)
            status = getattr(raw, "value", raw)
            if status not in _SF_TERMINAL:
                continue
            # The CURRENT terminal status is only evidence about as_of if the
            # execution had already stopped by then. One that stopped later was
            # IN FLIGHT on the day, and IN_FLIGHT is excluded from every
            # denominator (alpha-engine-config-I8069) — its later success or
            # failure is not credited to the earlier cutoff.
            end = getattr(r, "end_utc", None)
            name = getattr(r, "name", None) or r.execution_arn.rsplit(":", 1)[-1]
            if end is None:
                # A terminal status with no stop instant: whether it had ended by
                # as_of is unknowable, so its cycle is unevaluable — never GREEN.
                logger.error(
                    "sf_success_rate: %s:%s is %s with no end timestamp — cannot place its "
                    "outcome against as_of %s; its cycle is withheld as unevaluable "
                    "(alpha-engine-config-I12059).", sf_name, name, status, as_of,
                )
                unevaluable_keys.add(_cycle_key(sf_name, getattr(r, "run_date", None), start))
                temporally_unevaluable.append(f"{sf_name}:{name}")
                continue
            if end > as_of:
                in_flight_at_as_of.append(f"{sf_name}:{name}")
                continue
            run_date = getattr(r, "run_date", None)
            if run_date is None:
                logger.warning(
                    "sf_success_rate: %s carries no input.run_date — falling back "
                    "to the start-date cycle key, which splits an evening "
                    "recovery rerun into its own cycle (config-I7644).",
                    getattr(r, "name", "?"),
                )
            outcome, states, payload = _read_execution(r.execution_arn, sfn)
            if payload.get("rehearsal"):
                n_rehearsals += 1
                continue
            if not outcome.counts_as_cycle:
                # SKIPPED or IN_FLIGHT — excluded from the denominator entirely
                # (alpha-engine-config-I8069).
                continue
            # Always a str (alpha-engine-config-I8183): a str/date mix split
            # one calendar day into two cycles.
            key = _cycle_key(sf_name, run_date, start)
            cycles.setdefault(key, []).append((status, role, outcome, states))

        sf_cycles = sf_clean = sf_clean_full = 0
        sf_unatt = sf_unatt_ok = sf_unatt_ok_full = 0
        for key in unevaluable_keys:
            dropped = cycles.pop(key, None)
            scope_detail[f"{sf_name}:{key}"] = (
                f"[unevaluable] an execution of this cycle carries no end timestamp, so "
                f"its outcome cannot be placed against as_of {as_of.isoformat()} — withheld "
                f"from every denominator ({len(dropped or ())} other execution(s) not folded)"
            )
        for day, execs in sorted(cycles.items()):
            sf_cycles += 1
            graded = _fold(sf_name, day, execs)
            shape = graded["shape"]
            detail = "; ".join(outcome.explain() for _st, _role, outcome, _s in execs)
            scope = (
                "clean" if graded["full_scope"]
                else "clean_with_declared_skips" if graded["declared"]
                else "not_clean"
            )
            scope_detail[f"{sf_name}:{day}"] = f"[{scope}] {shape.explain()} | {detail}"
            sf_clean_full += graded["full_scope"]
            sf_clean += graded["declared"]
            for stage in graded["declared_skipped"]:
                bucket = declared_skips.setdefault(sf_name, {})
                bucket[stage] = bucket.get(stage, 0) + 1
            if not graded["declared"]:
                logger.warning("sf_success_rate: cycle %s:%s did not complete its work — %s", sf_name, day, detail)
            # Unattended first-pass: only cycles that HAD a scheduled run count.
            # It succeeded unattended iff a scheduled run ALONE was clean in the
            # scope at hand and no recovery role appears beside it.
            scheduled = [e for e in execs if e[1] in _SCHEDULED_ROLES]
            first_pass_full = first_pass = False
            if scheduled:
                sf_unatt += 1
                had_recovery = any(role not in _SCHEDULED_ROLES for _st, role, _o, _s in execs)
                if not had_recovery:
                    alone = [_fold(sf_name, day, [e]) for e in scheduled]
                    first_pass_full = any(g["full_scope"] for g in alone)
                    first_pass = any(g["declared"] for g in alone)
                sf_unatt_ok_full += first_pass_full
                sf_unatt_ok += first_pass
            if graded["declared"]:
                if first_pass:
                    n_clean_first_pass += 1
                else:
                    n_clean_recovered += 1

        n_cycles += sf_cycles
        n_clean += sf_clean
        n_clean_full += sf_clean_full
        n_unatt += sf_unatt
        n_unatt_ok += sf_unatt_ok
        n_unatt_ok_full += sf_unatt_ok_full
        per_sf[sf_name] = f"{sf_clean}/{sf_cycles}"
        per_sf_full[sf_name] = f"{sf_clean_full}/{sf_cycles}"
        per_sf_unattended[sf_name] = f"{sf_unatt_ok}/{sf_unatt}"
        per_sf_unattended_full[sf_name] = f"{sf_unatt_ok_full}/{sf_unatt}"

    return {
        # Headline scope: clean_with_declared_skips (I11987 option A).
        "cycle_rate": (n_clean / n_cycles) if n_cycles else None,
        "n_cycles": n_cycles,
        "n_cycles_clean": n_clean,
        "unattended_rate": (n_unatt_ok / n_unatt) if n_unatt else None,
        "n_unattended": n_unatt,
        "n_unattended_ok": n_unatt_ok,
        "per_sf": per_sf,
        "per_sf_unattended": per_sf_unattended,
        # Full scope, side by side — never folded into the headline.
        "cycle_rate_full_scope": (n_clean_full / n_cycles) if n_cycles else None,
        "n_cycles_clean_full_scope": n_clean_full,
        "unattended_rate_full_scope": (n_unatt_ok_full / n_unatt) if n_unatt else None,
        "n_unattended_ok_full_scope": n_unatt_ok_full,
        "per_sf_full_scope": per_sf_full,
        "per_sf_unattended_full_scope": per_sf_unattended_full,
        # How the headline's clean cycles were reached.
        "n_cycles_clean_with_declared_skips": n_clean - n_clean_full,
        "n_cycles_clean_first_pass": n_clean_first_pass,
        "n_cycles_clean_recovered": n_clean_recovered,
        # {sf: {stage: n_cycles}} — stages graded NOT_IN_SCOPE by a registered
        # witness on a cycle that is clean only because of it.
        "declared_skips": declared_skips,
        "n_rehearsals_excluded": n_rehearsals,
        # Both polarities, per cycle. A rate rendered without its denominator
        # is not a falsifiable claim, and this is the field that says which
        # cycles were graded narrow and why.
        "scope_detail": scope_detail,
        # Non-empty ⇒ at least one SF's window was cut short by the scan
        # limit, so `cycle_rate` describes a shorter period than its name.
        # The caller renders N/A rather than publishing it.
        "truncated": truncated,
        # Point-in-time eligibility (alpha-engine-config-I12059): the interval
        # graded, and every execution the listing returned that it excluded.
        "as_of": as_of.isoformat(),
        "window_start": cutoff.isoformat(),
        # {sf: n} executions that STARTED after as_of. They still occupy the
        # count-bounded scan, which is why `truncated` is judged on the scan's
        # oldest start rather than on how many executions fell in the window.
        "n_executions_after_as_of": n_after_as_of,
        # Started inside the window, stopped after as_of: in flight on the day.
        "in_flight_at_as_of": in_flight_at_as_of,
        # Non-empty ⇒ a terminal execution had no end timestamp, so its cycle
        # could not be graded at as_of and the denominator is incomplete. The
        # caller renders N/A rather than publishing it.
        "temporally_unevaluable": temporally_unevaluable,
    }


def _pct(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate:.0%}"


def _declared_skips_text(sf: dict) -> str:
    parts = [
        f"{name}: " + ", ".join(f"{stage}×{n}" for stage, n in sorted(stages.items()))
        for name, stages in sorted((sf.get("declared_skips") or {}).items())
    ]
    return "; ".join(parts) or "none"


def _cycle_scope_block(sf: dict) -> dict:
    """The two scopes and the first-pass/recovered split, as structured fields.

    Rides on the SF components as an extra ``cycle_scope`` field
    (``MetricRecord`` allows extras). The headline ``value`` is the
    ``clean_with_declared_skips`` rate; ``full_scope`` is never folded into it.
    """
    return {
        "headline_scope": "clean_with_declared_skips",
        "ruling": "alpha-engine-config-I11987 option A (2026-10-04)",
        "n_cycles": sf["n_cycles"],
        "clean_with_declared_skips": {
            "rate": sf["cycle_rate"], "n_clean": sf["n_cycles_clean"], "per_sf": sf["per_sf"],
        },
        "full_scope": {
            "rate": sf["cycle_rate_full_scope"], "n_clean": sf["n_cycles_clean_full_scope"],
            "per_sf": sf["per_sf_full_scope"],
        },
        "n_clean_only_by_declared_skips": sf["n_cycles_clean_with_declared_skips"],
        "declared_skips": {
            "label": NOT_IN_SCOPE,
            "witness_registry": "nousergon_lib.pipeline_status.CADENCE_SKIP_MARKER_STAGES",
            "stages_by_sf": sf["declared_skips"],
            "assurance": "missing — not a pass; contamination is reported by the attestation",
        },
        "first_pass": {
            "n_clean": sf["n_cycles_clean_first_pass"],
            "unattended_rate": sf["unattended_rate"],
            "unattended_rate_full_scope": sf["unattended_rate_full_scope"],
            "n_scheduled_cycles": sf["n_unattended"],
        },
        "recovered": {"n_clean": sf["n_cycles_clean_recovered"]},
        "n_rehearsals_excluded": sf["n_rehearsals_excluded"],
        # The graded interval and what it excluded (alpha-engine-config-I12059).
        "point_in_time": {
            "as_of": sf.get("as_of"),
            "window_start": sf.get("window_start"),
            "n_executions_after_as_of": sf.get("n_executions_after_as_of", {}),
            "in_flight_at_as_of": sf.get("in_flight_at_as_of", []),
        },
    }


def build_substrate_tile(
    bucket: str,
    run_date: str | None = None,
    s3_client=None,
    *,
    as_of: datetime | None = None,
    sfn_client=None,
    cloudwatch_client=None,
    threshold_leaderboard: dict | None = None,
    threshold_leaderboard_error: str | None = None,
) -> dict:
    """Build the Substrate Reliability tile.

    ``threshold_leaderboard`` (config#7476) is the scored threshold-slot
    leaderboard for this cycle, built by ``grading.thresholds.scoring``. It
    renders as one ``diagnostic`` MetricRecord per arm — machine-health
    vocabulary, answering "was the arm scored?" and never "did it win?"
    (champion-challenger §8). Omitted (standalone CLI / tests) or failed
    (``threshold_leaderboard_error``) → the records grade a loud N/A naming why,
    because an unscored cycle is unrecoverable and must not read as absence.

    ``run_date`` (ISO ``YYYY-MM-DD``) anchors the windowed read of the
    backtester's ``substrate_ops.json`` for ``watchdog_firings`` (config#1151).
    When omitted, ``watchdog_firings`` grades ``N/A-MISSING-INPUT`` rather than
    crashing — the other components are AWS-API sourced and don't need it.
    """
    s3 = s3_client or boto3.client("s3")
    as_of = as_of or datetime.now(UTC)
    components = []

    # 1. price_cache_freshness (critical) — days since the price cache last wrote,
    # per the content-derived sentinel the live writer stamps (config#2350/#3142).
    pc_src = f"s3://{bucket}/{PRICE_CACHE_FRESHNESS_SENTINEL_KEY}"
    sentinel = get_json(s3, bucket, PRICE_CACHE_FRESHNESS_SENTINEL_KEY)
    latest = None
    ts_raw = sentinel.get("timestamp") if isinstance(sentinel, dict) else None
    if ts_raw:
        try:
            latest = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            if latest.tzinfo is None:
                latest = latest.replace(tzinfo=UTC)
        except ValueError:
            latest = None
    if latest is not None:
        age_d = (as_of - latest).total_seconds() / 86400.0
        components.append(build_metric(
            name="price_cache_freshness", module=MODULE, metric_type="duration", criticality="critical",
            estimator="freshness_age",
            value=age_d, unit=DAYS, n_samples=1, n_floor=1, higher_is_better=False,
            source_path=pc_src,
            reason=f"price_cache_freshness = {age_d:.1f}d since the price cache last refreshed vs target 7d / red-line 14d.",
        ))
    else:
        components.append(build_metric(
            name="price_cache_freshness", module=MODULE, metric_type="duration", criticality="critical",
            estimator="freshness_age",
            n_floor=1, higher_is_better=False, source_path=pc_src,
            input_present=False,
            na_detail=f"price_cache_freshness: no readable timestamp at {PRICE_CACHE_FRESHNESS_SENTINEL_KEY}.",
        ))

    # 2. sf_success_rate_4w (critical) + unattended_first_pass_rate (supporting) —
    #    the substrate headline, re-keyed (config#1059 / #970 / L4552d). The OLD
    #    per-EXECUTION rate counted operator-recovered cycles AND scheduled-run
    #    failures as failures, producing a false P0 RED (0.4918) even on a week
    #    where every cycle ultimately completed clean. We now grade two distinct
    #    axes off the SF execution history (nousergon_lib.pipeline_status):
    #      - sf_success_rate_4w   = DISTINCT-CYCLE outcome (clean = recovered or
    #                               not). The honest "did the work get done?" axis.
    #      - unattended_first_pass_rate = scheduled run succeeded w/ NO recovery.
    #                               Surfaces the genuine full-automation target.
    #    Graceful N/A on an SF access error (a secondary read must not fail the
    #    whole report card — WARN-logged, not swallowed).
    sf_src = "stepfunctions:alpha-engine-{saturday,weekday,eod}-pipeline"

    def _na_pair(*, ran=True, input_present=True, cycle_detail, unatt_detail):
        components.append(build_metric(
            name="sf_success_rate_4w", module=MODULE, metric_type="pct", criticality="critical",
            estimator="distinct_cycle_success_4w", measurement_horizon="trailing_4w",
            n_floor=3, source_path=sf_src,
            ran=ran, input_present=input_present, na_detail=cycle_detail,
        ))
        components.append(build_metric(
            name="unattended_first_pass_rate", module=MODULE, metric_type="pct", criticality="supporting",
            estimator="unattended_first_pass_4w", measurement_horizon="trailing_4w",
            n_floor=3, source_path=sf_src,
            ran=ran, input_present=input_present, na_detail=unatt_detail,
        ))

    try:
        sfn = sfn_client or boto3.client(
            "stepfunctions", region_name=os.environ.get("AWS_REGION", "us-east-1")
        )
        sf = _sf_success_rate(sfn, as_of, _SF_WINDOW_DAYS)
        if sf is None:
            _na_pair(
                input_present=False,
                cycle_detail="sf_success_rate_4w: no pipeline SF ARNs discoverable (set EVALUATOR_SF_ARNS or grant states:ListStateMachines).",
                unatt_detail="unattended_first_pass_rate: no pipeline SF ARNs discoverable (set EVALUATOR_SF_ARNS or grant states:ListStateMachines).",
            )
        elif sf.get("truncated"):
            # A rate over a truncated window is a TRUE number about a smaller
            # world than its own name — `sf_success_rate_4w` / `trailing_4w`
            # would be read as 28 days by every consumer. N/A is the honest
            # render; `principles.md` §2.7 — no data is never green, and a
            # number measured over an unknown period is no data.
            _na_pair(
                input_present=False,
                cycle_detail=(f"sf_success_rate_4w: execution scan TRUNCATED for "
                              f"{', '.join(sf['truncated'])} — more than {_SF_EXEC_SCAN_LIMIT} "
                              f"executions inside the {_SF_WINDOW_DAYS}d window, so the scan "
                              f"covered a shorter period than the metric claims. Raise "
                              f"_SF_EXEC_SCAN_LIMIT or move to a time-bounded walk "
                              f"(alpha-engine-config-I8183)."),
                unatt_detail=(f"unattended_first_pass_rate: execution scan TRUNCATED for "
                              f"{', '.join(sf['truncated'])} — same window as above."),
            )
        elif sf.get("temporally_unevaluable"):
            # Same honesty rule as truncation (alpha-engine-config-I12059): a
            # cycle that could not be placed against as_of is missing from the
            # denominator, so any rate here would claim more than was graded.
            _na_pair(
                input_present=False,
                cycle_detail=(f"sf_success_rate_4w: {len(sf['temporally_unevaluable'])} terminal "
                              f"execution(s) carry no end timestamp "
                              f"({', '.join(sf['temporally_unevaluable'])}), so whether each had "
                              f"finished by as_of {sf['as_of']} is unknown and its cycle cannot be "
                              f"graded point-in-time (alpha-engine-config-I12059)."),
                unatt_detail=(f"unattended_first_pass_rate: same unevaluable cycle(s) as above "
                              f"({', '.join(sf['temporally_unevaluable'])})."),
            )
        elif sf["cycle_rate"] is None:
            _na_pair(
                ran=False,
                cycle_detail=f"sf_success_rate_4w: no terminal production-role SF cycles in the last {_SF_WINDOW_DAYS}d.",
                unatt_detail=f"unattended_first_pass_rate: no scheduled-cadence SF cycles in the last {_SF_WINDOW_DAYS}d.",
            )
        else:
            # Distinct-cycle outcome (critical headline), graded in the
            # clean_with_declared_skips scope (alpha-engine-config-I11987,
            # option A) with the full-scope result beside it — in the reason
            # AND as the structured `cycle_scope` block, so no reader has to
            # parse prose to see what the headline excused.
            cycle_scope = _cycle_scope_block(sf)
            components.append(build_metric(
                name="sf_success_rate_4w", module=MODULE, metric_type="pct", criticality="critical",
                estimator="distinct_cycle_success_4w", measurement_horizon="trailing_4w",
                value=sf["cycle_rate"], unit=FRACTION, n_samples=sf["n_cycles"], n_floor=3,
                source_path=sf_src,
                reason=(f"sf_success_rate_4w = {sf['cycle_rate']:.0%} ({sf['n_cycles_clean']}/{sf['n_cycles']} "
                        f"DISTINCT production-role cycles clean_with_declared_skips in {_SF_WINDOW_DAYS}d: "
                        f"{sf['per_sf']}) vs target 95% / red-line 80%. "
                        f"FULL SCOPE (every declared stage ran): "
                        f"{_pct(sf['cycle_rate_full_scope'])} ({sf['n_cycles_clean_full_scope']}/{sf['n_cycles']}: "
                        f"{sf['per_sf_full_scope']}). "
                        f"{sf['n_cycles_clean_with_declared_skips']} cycle(s) clean only because a REGISTERED "
                        f"cadence-skip witness put stages {NOT_IN_SCOPE} ({_declared_skips_text(sf)}) — an "
                        f"input skip_* flag excuses nothing (alpha-engine-config-I11987, option A). "
                        f"{NOT_IN_SCOPE} is not a pass: the guarantee those stages produce is MISSING — for "
                        f"parity that is contamination assurance, reported separately by the card's attestation "
                        f"(contamination_verdict), and this rate adds none. "
                        f"Of {sf['n_cycles_clean']} clean: {sf['n_cycles_clean_first_pass']} on the scheduled "
                        f"first pass, {sf['n_cycles_clean_recovered']} recovered by an operator rerun "
                        f"(config#1059: a recovered cycle still produced its artifacts). "
                        f"A cycle whose only green run left stages NOT_REACHED does not count clean "
                        f"(config-I7644); per-cycle scope: {sf['scope_detail'] or 'no cycles in window'}."),
            ).model_copy(update={"cycle_scope": cycle_scope}))
            # Unattended first-pass (supporting — the true automation target).
            if sf["unattended_rate"] is None:
                components.append(build_metric(
                    name="unattended_first_pass_rate", module=MODULE, metric_type="pct", criticality="supporting",
                    estimator="unattended_first_pass_4w", measurement_horizon="trailing_4w",
                    n_floor=3, source_path=sf_src, ran=False,
                    na_detail=f"unattended_first_pass_rate: no scheduled-cadence cycles in the last {_SF_WINDOW_DAYS}d.",
                ))
            else:
                components.append(build_metric(
                    name="unattended_first_pass_rate", module=MODULE, metric_type="pct", criticality="supporting",
                    estimator="unattended_first_pass_4w", measurement_horizon="trailing_4w",
                    value=sf["unattended_rate"], unit=FRACTION, n_samples=sf["n_unattended"], n_floor=3,
                    source_path=sf_src,
                    reason=(f"unattended_first_pass_rate = {sf['unattended_rate']:.0%} ({sf['n_unattended_ok']}/"
                            f"{sf['n_unattended']} scheduled cycles clean_with_declared_skips with NO operator "
                            f"recovery in {_SF_WINDOW_DAYS}d: {sf['per_sf_unattended']}) vs target 95% / red-line 50%. "
                            f"FULL SCOPE: {_pct(sf['unattended_rate_full_scope'])} "
                            f"({sf['n_unattended_ok_full_scope']}/{sf['n_unattended']}: "
                            f"{sf['per_sf_unattended_full_scope']}). "
                            f"The genuine full-automation target (config#970/L4552d) — distinct from the "
                            f"did-the-work-get-done cycle rate above."),
                ).model_copy(update={"cycle_scope": cycle_scope}))
    except (ClientError, BotoCoreError) as e:
        code = e.response.get("Error", {}).get("Code") if isinstance(e, ClientError) else type(e).__name__
        logger.warning("sf_success_rate_4w: SF API read failed (%s) — grading N/A", e)
        _na_pair(
            ran=False,
            cycle_detail=f"sf_success_rate_4w: Step Functions read failed this cycle ({code}).",
            unatt_detail=f"unattended_first_pass_rate: Step Functions read failed this cycle ({code}).",
        )

    # 3. data_quality_incidents (critical) — trailing-4w SUM of feature-store
    #    rows BLOCKED by the data-quality gate, read from CloudWatch (mirrors the
    #    SF-history read above — the substrate tile is the one tile that reaches
    #    AWS APIs, not just S3). Graceful N/A on a CW access error / missing perm
    #    (a secondary read must not fail the whole card — WARN-logged, not
    #    swallowed). config#1150 Batch B.
    dq_src = f"cloudwatch:{_DQ_NAMESPACE}/{_DQ_BLOCKED_METRIC}"
    try:
        cw = cloudwatch_client or boto3.client(
            "cloudwatch", region_name=os.environ.get("AWS_REGION", "us-east-1")
        )
        blocked = _cw_metric_sum(cw, _DQ_NAMESPACE, _DQ_BLOCKED_METRIC, as_of, _SF_WINDOW_DAYS)
        warned = _cw_metric_sum(cw, _DQ_NAMESPACE, _DQ_WARNED_METRIC, as_of, _SF_WINDOW_DAYS)
        if blocked is None:
            components.append(build_metric(
                name="data_quality_incidents", module=MODULE, metric_type="count", criticality="critical",
                estimator="incident_count_4w", measurement_horizon="trailing_4w",
                n_floor=1, higher_is_better=False, source_path=dq_src,
                ran=False,
                na_detail=f"data_quality_incidents: no {_DQ_BLOCKED_METRIC} datapoints in CloudWatch over {_SF_WINDOW_DAYS}d.",
            ))
        else:
            warned_s = f"{warned:.0f}" if warned is not None else "n/a"
            components.append(build_metric(
                name="data_quality_incidents", module=MODULE, metric_type="count", criticality="critical",
                estimator="incident_count_4w", measurement_horizon="trailing_4w",
                value=blocked, unit=COUNT_ROWS, n_samples=1, n_floor=1,
                higher_is_better=False, source_path=dq_src,
                reason=(f"data_quality_incidents = {blocked:.0f} feature-store rows BLOCKED by the "
                        f"quality gate over {_SF_WINDOW_DAYS}d (warned: {warned_s}) vs target 200 / "
                        f"red-line 500 — a regression detector calibrated off the 2026-06 baseline."),
            ))
    except (ClientError, BotoCoreError) as e:
        code = e.response.get("Error", {}).get("Code") if isinstance(e, ClientError) else type(e).__name__
        logger.warning("data_quality_incidents: CloudWatch read failed (%s) — grading N/A", e)
        components.append(build_metric(
            name="data_quality_incidents", module=MODULE, metric_type="count", criticality="critical",
            estimator="incident_count_4w", measurement_horizon="trailing_4w",
            n_floor=1, higher_is_better=False, source_path=dq_src,
            input_present=False,
            na_detail=f"data_quality_incidents: CloudWatch read failed this cycle ({code}) — grant cloudwatch:GetMetricStatistics to the evaluator role.",
        ))

    # 3b. schema_drift_incidents (critical) — trailing-4w SUM of ArcticDB
    #    StreamDescriptorMismatch / DataError write failures, counted-then-
    #    re-raised by the data collector and emitted to CloudWatch (mirrors the
    #    data_quality read above — the substrate tile is the one tile that reaches
    #    AWS APIs). A schema-drift incident is a HARD data-integrity failure (the
    #    daily_append run fails loud), so the band is an absolute incident count
    #    (target 0 / red-line 3) rather than a regression detector. Graceful N/A
    #    on a CW access error / missing perm — WARN-logged, not swallowed.
    #    config#1150 Batch B.
    sd_src = f"cloudwatch:{_DQ_NAMESPACE}/{_SCHEMA_DRIFT_METRIC}"
    try:
        cw = cloudwatch_client or boto3.client(
            "cloudwatch", region_name=os.environ.get("AWS_REGION", "us-east-1")
        )
        drift = _cw_metric_sum(cw, _DQ_NAMESPACE, _SCHEMA_DRIFT_METRIC, as_of, _SF_WINDOW_DAYS)
        if drift is None:
            components.append(build_metric(
                name="schema_drift_incidents", module=MODULE, metric_type="count", criticality="critical",
                estimator="incident_count_4w", measurement_horizon="trailing_4w",
                n_floor=1, higher_is_better=False, source_path=sd_src,
                ran=False,
                na_detail=(f"schema_drift_incidents: no {_SCHEMA_DRIFT_METRIC} datapoints in CloudWatch "
                           f"over {_SF_WINDOW_DAYS}d (the metric self-activates once a daily_append run "
                           f"emits it — zero is emitted on every clean run, so N/A means the instrumented "
                           f"producer has not yet deployed/run)."),
            ))
        else:
            components.append(build_metric(
                name="schema_drift_incidents", module=MODULE, metric_type="count", criticality="critical",
                estimator="incident_count_4w", measurement_horizon="trailing_4w",
                value=drift, unit=COUNT_EVENTS, n_samples=1, n_floor=1,
                higher_is_better=False, source_path=sd_src,
                reason=(f"schema_drift_incidents = {drift:.0f} ArcticDB StreamDescriptorMismatch / "
                        f"DataError write failures over {_SF_WINDOW_DAYS}d vs target 0 / red-line 3. "
                        f"A schema-drift incident is a HARD data-integrity failure (the daily_append "
                        f"write fails loud) — meant to be zero in steady state; any incident is a WATCH, "
                        f"a cluster (≥3) is a systemic descriptor regression (RED)."),
            ))
    except (ClientError, BotoCoreError) as e:
        code = e.response.get("Error", {}).get("Code") if isinstance(e, ClientError) else type(e).__name__
        logger.warning("schema_drift_incidents: CloudWatch read failed (%s) — grading N/A", e)
        components.append(build_metric(
            name="schema_drift_incidents", module=MODULE, metric_type="count", criticality="critical",
            estimator="incident_count_4w", measurement_horizon="trailing_4w",
            n_floor=1, higher_is_better=False, source_path=sd_src,
            input_present=False,
            na_detail=f"schema_drift_incidents: CloudWatch read failed this cycle ({code}) — grant cloudwatch:GetMetricStatistics to the evaluator role.",
        ))

    # watchdog_firings (supporting, config#1151 Batch C) — how many backtester
    # phases hit their hard timeout cap this run and were force-aborted by the
    # phase watchdog. Read (windowed, config#1190 — a partial/off-cycle producer
    # run still grades) from backtest/{date}/substrate_ops.json, the per-run
    # aggregate pipeline_common.phase() writes. Lower-is-better: 0 = GREEN.
    wf_src = f"s3://{bucket}/backtest/{run_date}/substrate_ops.json" if run_date else f"s3://{bucket}/"
    wf_doc = None
    if run_date:
        wf_doc, _wf_date, _wf_age, _wf_key = get_json_windowed(
            s3, bucket, "backtest/{date}/substrate_ops.json", run_date
        )
        if _wf_key:
            wf_src = f"s3://{bucket}/{_wf_key}"
    wd = (wf_doc or {}).get("watchdog") if isinstance(wf_doc, dict) else None
    if isinstance(wd, dict) and wd.get("firing_count") is not None:
        firings = wd["firing_count"]
        capped = wd.get("capped_phases_run")
        fired_phases = [
            r.get("phase") for r in (wd.get("per_phase") or []) if r.get("watchdog_fired")
        ]
        if firings == 0:
            verdict = "no phase hit its hard cap (healthy)"
        elif firings == 1:
            verdict = "one phase capped out — a single trip can be a transient infra slowdown, WATCH"
        else:
            verdict = "multiple phases capped out — a phase is systematically exceeding its budget, RED"
        detail = f" ({', '.join(p for p in fired_phases if p)})" if fired_phases else ""
        components.append(build_metric(
            name="watchdog_firings", module=MODULE, metric_type="count", criticality="supporting",
            estimator="per_run_phase_timeout_count", measurement_horizon="per_run",
            value=float(firings), unit=COUNT_EVENTS, n_samples=1, n_floor=1,
            higher_is_better=False,
            source_path=wf_src,
            reason=(f"watchdog_firings = {firings} backtester phase(s) hit their hard "
                    f"timeout cap{detail} of {capped} capped phase(s) run vs target 0 / "
                    f"red-line 2 — {verdict}."),
        ))
    else:
        components.append(build_metric(
            name="watchdog_firings", module=MODULE, metric_type="count", criticality="supporting",
            n_floor=1, 
            higher_is_better=False, source_path=wf_src, input_present=False,
            na_detail=(f"watchdog_firings: no substrate_ops.json with a watchdog block in the "
                       f"trailing window ending {run_date} — needs the backtester producer "
                       f"(config#1151) to have run a capped phase."),
        ))

    # 4. deploy_success_rate (supporting) — wired to the Director's weekly GH-API
    #    rollup (config#1153 Batch E). Graded when the producer has run + the
    #    rollup is fresh; transparent N/A naming the producer otherwise.
    ds_src = f"s3://{bucket}/{DEPLOY_SUCCESS_KEY}"
    ds_doc = get_json(s3, bucket, DEPLOY_SUCCESS_KEY)
    ds_age_days = None
    if isinstance(ds_doc, dict):
        gen = ds_doc.get("generated_utc")
        if gen:
            try:
                gen_dt = datetime.fromisoformat(gen)
                if gen_dt.tzinfo is None:
                    gen_dt = gen_dt.replace(tzinfo=UTC)
                ds_age_days = (as_of - gen_dt).total_seconds() / 86400.0
            except ValueError:
                ds_age_days = None
    ds_rate = ds_doc.get("success_rate") if isinstance(ds_doc, dict) else None
    ds_total = (ds_doc.get("total_runs") or 0) if isinstance(ds_doc, dict) else 0
    ds_stale = ds_age_days is not None and ds_age_days > _DEPLOY_SUCCESS_MAX_AGE_DAYS
    if ds_rate is not None and ds_total and not ds_stale:
        ds_succ = ds_doc.get("success_runs", round(float(ds_rate) * ds_total))
        win = ds_doc.get("window_days", "?")
        n_repos = len(ds_doc.get("repos_measured") or [])
        components.append(build_metric(
            name="deploy_success_rate", module=MODULE, metric_type="pct", criticality="supporting",
            estimator="deploy_workflow_success_rate", measurement_horizon=f"trailing_{win}d",
            value=float(ds_rate), unit=FRACTION, n_samples=int(ds_total), n_floor=3,
            source_path=ds_src,
            reason=(f"deploy_success_rate = {float(ds_rate):.0%} ({ds_succ}/{ds_total} terminal "
                    f"deploy-workflow runs across {n_repos} repo(s) succeeded in {win}d) "
                    f"vs target 95% / red-line 80%."),
        ))
    else:
        if ds_stale:
            na = (f"deploy_success_rate: rollup at {DEPLOY_SUCCESS_KEY} is {ds_age_days:.0f}d old "
                  f"(> {_DEPLOY_SUCCESS_MAX_AGE_DAYS}d) — the Director's weekly producer has stopped running.")
        elif isinstance(ds_doc, dict) and not ds_total:
            na = ("deploy_success_rate: rollup present but no terminal deploy-workflow runs in the "
                  "window — nothing to grade.")
        else:
            na = ("deploy_success_rate: no _substrate/deploy_success.json yet — needs the Director's "
                  "weekly GH-API producer (grading.producers.deploy_success) to have run.")
        components.append(build_metric(
            name="deploy_success_rate", module=MODULE, metric_type="pct", criticality="supporting",
            n_floor=3, source_path=ds_src,
            input_present=False, na_detail=na,
        ))

    # 4b. unpriced_cost_rows (supporting, alpha-engine-config-I11113) — the
    #     count of cost rows in the trailing-week _cost_raw window that exist but
    #     cannot be priced. This is the surface deliverable 4 of I11100 asked for
    #     and krepis structurally could not provide: krepis is a pure library with
    #     no deployment and no alerting surface, so the degrade it writes had no
    #     reader anywhere in the fleet. It renders here, beside the other
    #     substrate-integrity counts, because the Report Card is already the
    #     weekly cost surface and a MetricRecord is self-describing — the console
    #     renders it with no dashboard change (policy-console).
    #
    #     "Could not read the window" is a DIFFERENT record from "zero degraded
    #     rows": the producer raises CostPricingUnmeasured and this grades a
    #     loud N/A-MISSING-INPUT naming the partition and the fault, never 0.
    #     Collapsing the two would reproduce exactly the defect class being
    #     closed here (principles.md §2.7).
    ucr_src = f"s3://{bucket}/{COST_RAW_PREFIX}/{{date}}/**/*.jsonl"
    if run_date:
        try:
            scan = scan_degraded_cost_rows(
                s3, bucket, run_date, window_days=_UNPRICED_COST_ROWS_WINDOW_DAYS
            )
        except CostPricingUnmeasured as e:
            logger.warning("unpriced_cost_rows: window unmeasurable (%s) — grading N/A, NOT zero", e)
            components.append(build_metric(
                name="unpriced_cost_rows", module=MODULE, metric_type="count",
                criticality="supporting",
                estimator="degraded_cost_source_row_count",
                measurement_horizon=f"trailing_{_UNPRICED_COST_ROWS_WINDOW_DAYS}d",
                n_floor=1, higher_is_better=False, source_path=ucr_src,
                input_present=False,
                na_detail=(f"unpriced_cost_rows: the {_UNPRICED_COST_ROWS_WINDOW_DAYS}d "
                           f"{COST_RAW_PREFIX}/ window ending {run_date} could not be counted "
                           f"in full ({e}) — this is UNMEASURED, not zero: a row that exists "
                           f"but cannot be priced would be invisible either way, so the count "
                           f"is withheld rather than reported low."),
            ))
        else:
            total = scan["degraded_total"]
            breakdown = format_by_callsite(scan["by_callsite"])
            by_src = scan["by_cost_source"]
            window = f"{scan['partitions'][0]}..{scan['partitions'][-1]}"
            components.append(build_metric(
                name="unpriced_cost_rows", module=MODULE, metric_type="count",
                criticality="supporting",
                estimator="degraded_cost_source_row_count",
                measurement_horizon=f"trailing_{_UNPRICED_COST_ROWS_WINDOW_DAYS}d",
                value=float(total), unit=COUNT_ROWS, n_samples=1, n_floor=1,
                higher_is_better=False, source_path=ucr_src,
                reason=(f"unpriced_cost_rows = {total} of {scan['rows_scanned']} cost row(s) "
                        f"over {scan['objects_scanned']} object(s) in {window} carry a degraded "
                        f"cost_source (unpriced={by_src['unpriced']}, "
                        f"usage_unreported={by_src['usage_unreported']}) vs target 0 / "
                        f"red-line 25 — by callsite: {breakdown}. The rows EXIST and fan-in "
                        f"coverage passes; what is missing is the price, so this cycle's "
                        f"AlphaEngine/Cost understates by whatever those calls cost. "
                        f"A non-zero count means a served model has no active price card "
                        f"(unpriced) or a route returned no usage (usage_unreported)."),
            ))
    else:
        components.append(build_metric(
            name="unpriced_cost_rows", module=MODULE, metric_type="count",
            criticality="supporting",
            estimator="degraded_cost_source_row_count",
            measurement_horizon=f"trailing_{_UNPRICED_COST_ROWS_WINDOW_DAYS}d",
            n_floor=1, higher_is_better=False, source_path=ucr_src,
            input_present=False,
            na_detail=("unpriced_cost_rows: no run_date supplied, so the "
                       f"{COST_RAW_PREFIX}/ window is undefined — the count is "
                       "withheld rather than reported as zero."),
        ))

    # 5-7. Accepted permanent honest-N/A (config#1153 Batch E, operator ruling
    #      2026-07-11 Option A): each of these needs a producer/convention that
    #      the ruling declined to build (only the critical-3 — judge_calibration,
    #      backtest_vs_live, dsr — stay prioritized). They render as N/A but are
    #      marked permanent_na so the cliff-inventory reads them as closed, not
    #      pending. The reason still names what WOULD unblock them if revisited.
    permanent_na = [
        ("alert_noise_ratio", "supporting",
         "alert_noise_ratio: would need the alerts log + a manual actionable/total tagging convention — "
         "not building the tagging convention (config#1153 Option A)."),
        ("changelog_coverage", "diagnostic",
         "changelog_coverage: would need an expected-event-source set defined to compute % writing to the "
         "changelog — not defining that set (config#1153 Option A)."),
        ("iam_drift", "diagnostic",
         "iam_drift: would need a CFN detect-drift → S3 producer + a cross-repo ops IAM grant the evaluator "
         "lacks — not building it (config#1153 Option A)."),
    ]
    for name, crit, detail in permanent_na:
        components.append(build_metric(
            name=name, module=MODULE, metric_type="pct", criticality=crit, n_floor=1,
            source_path=f"s3://{bucket}/", permanent_na_reason=detail,
        ))

    # config#7476 — the threshold champion/challenger slot's own scoring, one
    # record per arm, every cycle (champion-challenger §3).
    components.extend(build_arm_components(
        threshold_leaderboard,
        module=MODULE,
        source_path=f"s3://{bucket}/{leaderboard_key(run_date)}" if run_date
                    else f"s3://{bucket}/evaluator/",
        error=threshold_leaderboard_error,
    ))

    return build_tile(MODULE, components)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Build the Substrate Reliability tile.")
    parser.add_argument("--bucket", default="alpha-engine-research")
    parser.add_argument("--run-date", default=None, help="ISO run date for windowed artifact reads (watchdog_firings).")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(json.dumps(build_substrate_tile(args.bucket, args.run_date), indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
