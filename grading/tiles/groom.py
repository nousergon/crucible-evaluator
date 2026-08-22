"""
groom.py — groom-pipeline components for the Agent tile (config#2151).

Grades the backlog-groom pipeline (the tiered Haiku/Sonnet/Opus groom runs on
``nousergon/alpha-engine-config``) from its per-run S3 artifacts:

    s3://{bucket}/groom/{YYYY-MM-DD}/{run_id}.json

(schema_version 6, producer: ``alpha-engine-config/scripts/groom_driver.py``
``write_run_artifact`` — the PRIMARY run record since config#1808). Every
groom-health defect to date (fresh-skip drift config#2038, dead engagement
scan config#2142, comment churn config#2147, zero-output WET runs config#2148)
was found by operator forensics on these artifacts; these four records make
the same regressions continuously visible on the report card:

  - groom_completion_rate    : (closed + pr_opened) / all queued-issue
                               dispositions in the trailing 7d window
  - groom_wet_per_completion : sum(run_wet) / completions (directional-down)
  - groom_comment_churn      : distinct issues with ≥3 ``commented``
                               dispositions in-window and no completion
  - groom_lost_chunks        : chunk_log entries that died at max_turns

These land on the AGENT tile (Tile 6 — the LLM-agent-quality surface; the
groom runs are LLM-agent runs, evaluator-owner's call per config#2151), so
``MODULE = "agent"`` and :func:`build_groom_components` is called from
``build_agent_tile`` rather than registering a tenth tile.

SOAK POSTURE (mirrors the behavioral tile): every component is
supporting/diagnostic — never critical — and the bands are provisional
(``reliability="medium"``) until the config#2147 churn drain + the config#2135
complete-or-gate contract settle the steady state. Baselines measured
2026-07-04→07-10: completion 7.2% (100/1392), WET/completion ~1.9M, churn 171,
~24 max_turns chunk failures.

Fail-loud posture (``[[feedback_no_silent_fails]]``):
  - A day-prefix with no artifacts is a legitimate state (the groomer is
    cron-driven and can skip days) — skipped, never an error.
  - ZERO artifacts across the whole window means the groomer stopped running
    or the writer broke → every record grades a precise N/A-MISSING-INPUT
    naming the producer (visible degradation, never a silent GREEN) — UNLESS
    the pause-reconcile paused-lane declaration (alpha-engine-config-I8189,
    ``_load_paused_lanes`` / ``PAUSED_LANES_KEY``) names one of the groom
    schedules as deliberately paused, in which case the four records render a
    declared-off accepted-permanent-N/A naming the ruling
    (alpha-engine-config-I6617) instead — a declared pause must not render as
    an undeclared "did not run or its writer broke" accusation
    (observability-policy.md §8.3).
  - A MALFORMED artifact (unparseable JSON / contract-violating shape) is a
    producer contract violation and RAISES — the artifact is written whole via
    ``aws s3 cp`` (no partial-write mode), so corruption is a genuine defect,
    not an expected transient. The grading handler is fail-loud by design; the
    SF Catch + the freshness monitor on ``evaluator/{date}/report_card.json``
    are the recording surface.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import re

import boto3
from botocore.exceptions import ClientError

from krepis.metrics import MetricRecord

from grading.metric_record import build_metric
from grading.units import COUNT_CHUNKS, COUNT_ISSUES, FRACTION, WET_PER_COMPLETION

logger = logging.getLogger(__name__)

# Components carry the agent tile's module id (they are agent-tile members).
MODULE = "agent"

GROOM_PREFIX = "groom"
# Trailing window, inclusive of run_date — matches the issue's weekly baseline
# window (2026-07-04→07-10) and the weekly report-card cadence.
WINDOW_DAYS = 7

# Dispositions that count as a COMPLETION (the groom actually resolved the
# issue or shipped work), vs the engaged-but-open set (commented/labeled) and
# untouched. Values per groom_driver.build_issue_records (schema_version 6).
_COMPLETION_DISPOSITIONS = frozenset({"closed", "pr_opened"})

# An issue is CHURNING when the groomer keeps commenting without ever
# completing it (config#2147): ≥ this many `commented` dispositions in-window
# with zero closed/pr_opened.
_CHURN_MIN_COMMENTS = 3

# A chunk_log entry marking a chunk that died at the agent turn budget
# (config#1946 classifier signature; the entry embeds the truncated error
# detail, so match either the terminal_reason fragment or the SDK's message).
_MAX_TURNS_RE = re.compile(r"max_turns|Reached maximum number of turns")


class GroomArtifactError(ValueError):
    """A groom run artifact violated the schema_version-6 producer contract."""


# The pause-reconcile paused-lane artifact (alpha-engine-config-I8189). Written
# by nousergon-data's infrastructure/pause_reconcile.py (daily 09:50 UTC) from
# infrastructure/automation_pause.py::paused_names() — the single declaration
# surface for "DISABLED is the INTENDED live state" across the pause manifest.
# A declared pause must not render as an undeclared gap (observability-policy.md
# §8.3): DISABLED is DECLARED, never inferred, and symmetrically a declared
# pause must not be mistaken for an inferred one either.
PAUSED_LANES_KEY = "ops/checks/automation-pause-reconcile/paused_lanes.json"

# The EventBridge Scheduler schedule names that drive the groom pipeline
# (infrastructure/automation_pause.json ``paused.scheduler_schedules``, all
# paused 2026-08-07 per Brian ruling alpha-engine-config-I6617). If ANY of
# these is in the live paused-lane set, the groom is declared off — the whole
# pipeline shares one on/off state, so partial membership still means "off".
_GROOM_TRIGGER_NAMES = frozenset({
    "alpha-engine-groom-lane-reconciler-5min",
    "alpha-engine-groom-sweep-0000-daily",
    "alpha-engine-groom-sweep-0800-daily",
    "alpha-engine-groom-sweep-1600-daily",
    "alpha-engine-scheduled-groom-0400-daily",
    "alpha-engine-scheduled-groom-1200-daily",
    "alpha-engine-scheduled-groom-2000-daily",
    "alpha-engine-scheduled-groom-sun0900-weekly",
})

_PAUSE_RULING_REF = "alpha-engine-config-I6617"


def _load_paused_lanes(s3, bucket: str) -> set[str] | None:
    """The live paused-lane set, or ``None`` if it can't be read.

    Never raises: an unreadable declaration is not proof of absence — it means
    this reader could not observe the declaration, which must NOT be treated
    as "not paused" (that would be inference in the wrong direction). Callers
    fall back to the pre-existing N/A-MISSING-INPUT behavior when this returns
    ``None``, exactly as if the artifact had never been added.
    """
    try:
        resp = s3.get_object(Bucket=bucket, Key=PAUSED_LANES_KEY)
        doc = json.loads(resp["Body"].read())
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            logger.info(
                "groom pause-declaration artifact absent at s3://%s/%s — "
                "treating as unreadable (never as 'not paused')",
                bucket, PAUSED_LANES_KEY,
            )
        else:
            logger.warning(
                "groom pause-declaration artifact read failed s3://%s/%s: %s",
                bucket, PAUSED_LANES_KEY, e,
            )
        return None
    except Exception:  # noqa: BLE001 — see docstring: never raise, never infer
        logger.warning(
            "groom pause-declaration artifact unreadable/malformed at s3://%s/%s",
            bucket, PAUSED_LANES_KEY, exc_info=True,
        )
        return None
    paused = doc.get("paused")
    if not isinstance(paused, list):
        logger.warning(
            "groom pause-declaration artifact at s3://%s/%s missing list-typed "
            "'paused' (schema_version=%r)", bucket, PAUSED_LANES_KEY,
            doc.get("schema_version"),
        )
        return None
    return {str(n) for n in paused}


def _window_dates(run_date: str) -> list[str]:
    """The trailing-``WINDOW_DAYS`` ISO dates ending at ``run_date`` inclusive.

    ``run_date`` arrives ISO-normalized from the handler (resolve_trading_day);
    a non-ISO value is a caller contract violation and raises (fail loud).
    """
    end = _dt.date.fromisoformat(run_date)
    return [(end - _dt.timedelta(days=d)).isoformat() for d in range(WINDOW_DAYS - 1, -1, -1)]


def _list_run_keys(s3, bucket: str, date: str) -> list[str]:
    """All run-artifact keys under one day prefix (empty day → empty list)."""
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{GROOM_PREFIX}/{date}/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json"):
                keys.append(obj["Key"])
    return keys


def _read_run_artifact(s3, bucket: str, key: str) -> dict:
    """Read + shape-validate one run artifact. RAISES on a malformed artifact
    (see the module docstring's fail-loud posture — corruption here is a
    producer contract violation, never skipped)."""
    resp = s3.get_object(Bucket=bucket, Key=key)
    try:
        doc = json.loads(resp["Body"].read())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise GroomArtifactError(f"unparseable groom artifact s3://{bucket}/{key}: {e}") from e

    if not isinstance(doc, dict):
        raise GroomArtifactError(f"groom artifact s3://{bucket}/{key} is not a JSON object")
    issues = doc.get("issues")
    chunk_log = doc.get("chunk_log")
    if not isinstance(issues, list) or not isinstance(chunk_log, list):
        raise GroomArtifactError(
            f"groom artifact s3://{bucket}/{key} missing list-typed issues/chunk_log "
            f"(schema_version={doc.get('schema_version')})"
        )
    for rec in issues:
        if not isinstance(rec, dict) or not isinstance(rec.get("disposition"), str):
            raise GroomArtifactError(
                f"groom artifact s3://{bucket}/{key} carries a malformed issue record: {rec!r}"
            )
    run_wet = doc.get("run_wet")
    # run_wet is null by CONTRACT when the producer's fail-safe WET compute
    # errored (schema_version 5 note) — null is legal, a non-number is not.
    if run_wet is not None and not isinstance(run_wet, (int, float)):
        raise GroomArtifactError(
            f"groom artifact s3://{bucket}/{key} carries non-numeric run_wet: {run_wet!r}"
        )
    return doc


def load_groom_runs(s3, bucket: str, run_date: str) -> tuple[list[dict], list[str]]:
    """All run artifacts in the trailing window. Returns ``(runs, keys)``."""
    runs: list[dict] = []
    keys: list[str] = []
    for date in _window_dates(run_date):
        for key in _list_run_keys(s3, bucket, date):
            runs.append(_read_run_artifact(s3, bucket, key))
            keys.append(key)
    return runs, keys


def build_groom_components(bucket: str, run_date: str, s3_client=None) -> list[MetricRecord]:
    """Compute the four groom-pipeline MetricRecords for the Agent tile."""
    s3 = s3_client or boto3.client("s3")
    runs, keys = load_groom_runs(s3, bucket, run_date)
    window = _window_dates(run_date)
    span = f"{window[0]}→{window[-1]}"
    src = f"s3://{bucket}/{GROOM_PREFIX}/"

    if not runs:
        # Zero artifacts across the whole window is ambiguous by construction:
        # either the groomer stopped running / its writer broke, OR it was
        # deliberately paused (alpha-engine-config-I6617, 2026-08-07). §8.3
        # requires DISABLED be declared, never inferred — so check the
        # declaration surface FIRST, and only fall back to the
        # broke-or-stopped read when the declaration itself is unreadable
        # (never silently assume "not paused").
        paused_lanes = _load_paused_lanes(s3, bucket)
        is_paused = bool(paused_lanes) and bool(_GROOM_TRIGGER_NAMES & paused_lanes)

        def _absent(name: str, metric_type, n_floor: int, hib) -> MetricRecord:
            if is_paused:
                return build_metric(
                    name=name, module=MODULE, metric_type=metric_type,
                    criticality="diagnostic", n_floor=1, band="unbanded",
                    source_path=src,
                    permanent_na_reason=(
                        f"{name}: groom paused by Brian ruling {_PAUSE_RULING_REF} "
                        f"(2026-08-07) — zero run artifacts under {src} for {span} "
                        f"is the DECLARED, intended state, not a broken producer. "
                        f"Un-pauses automatically once the pause-lane manifest "
                        f"drops the groom schedules (no code change needed here)."
                    ),
                )
            return build_metric(
                name=name, module=MODULE, metric_type=metric_type, criticality="supporting",
                n_floor=n_floor, higher_is_better=hib,
                source_path=src, input_present=False,
                na_detail=(f"{name}: zero groom run artifacts under {src} for {span} — "
                           f"groomer did not run or its writer broke (producer: "
                           f"alpha-engine-config scripts/groom_driver.py write_run_artifact, "
                           f"config#2151)."),
            )

        return [
            _absent("groom_completion_rate", "pct", 30, True),
            _absent("groom_wet_per_completion", "ratio", 1, False),
            _absent("groom_comment_churn", "count", 10, False),
            _absent("groom_lost_chunks", "count", 5, False),
        ]

    n_runs = len(runs)
    dispositions = [rec for run in runs for rec in run["issues"]]  # sweep runs contribute none
    n_disp = len(dispositions)
    completions = sum(1 for r in dispositions if r["disposition"] in _COMPLETION_DISPOSITIONS)

    components: list[MetricRecord] = []

    # 1. groom_completion_rate (supporting) — share of queued-issue dispositions
    #    that actually resolved the issue (closed) or shipped work (pr_opened).
    #    Higher better; provisional target 30% / red-line 15% (config#2151 —
    #    bands settle after the config#2147 drain + config#2135 contract).
    rate = (completions / n_disp) if n_disp else None
    components.append(build_metric(
        name="groom_completion_rate", module=MODULE, metric_type="pct", criticality="supporting",
        estimator="disposition_share_trailing_window", measurement_horizon=f"{WINDOW_DAYS}d_window",
        reliability="medium",
        value=rate, unit=FRACTION, n_samples=n_disp, n_floor=30,
        higher_is_better=True, source_path=src,
        reason=(f"groom_completion_rate = {rate:.1%} ({completions}/{n_disp} dispositions "
                f"closed/PR'd across {n_runs} runs, {span}) vs provisional target 30% / "
                f"red-line 15% (baseline 7.2%, config#2151)."
                if rate is not None else
                f"groom_completion_rate: {n_runs} run(s) in {span} carried zero queued-issue "
                f"dispositions (sweep-only window) — nothing to grade."),
    ))

    # 2. groom_wet_per_completion (diagnostic) — WET spent per completed issue.
    #    Directional-down / trend-informational during soak (no bands yet —
    #    config#2151 sets only a "directional-down" objective; baseline ~1.9M).
    #    Runs whose fail-safe WET compute recorded null are excluded from the
    #    numerator by producer contract (their spend is unknown, not zero) and
    #    the exclusion is surfaced in the reason string.
    wet_runs = [run["run_wet"] for run in runs if run.get("run_wet") is not None]
    n_null_wet = n_runs - len(wet_runs)
    total_wet = float(sum(wet_runs))
    wet_per = (total_wet / completions) if completions else None
    null_note = f"; {n_null_wet} run(s) with null run_wet excluded" if n_null_wet else ""
    components.append(build_metric(
        name="groom_wet_per_completion", module=MODULE, metric_type="ratio", criticality="diagnostic",
        estimator="wet_sum_over_completions", measurement_horizon=f"{WINDOW_DAYS}d_window",
        reliability="medium",
        value=wet_per, unit=WET_PER_COMPLETION, n_samples=completions, n_floor=1,
        higher_is_better=False, source_path=src,
        reason=(f"groom_wet_per_completion = {wet_per:,.0f} WET ({total_wet:,.0f} WET over "
                f"{len(wet_runs)} runs / {completions} completions, {span}{null_note}) — "
                f"directional-down, informational during soak (baseline ~1.9M, config#2151)."
                if wet_per is not None else
                f"groom_wet_per_completion: 0 completions across {n_runs} runs ({span}) — "
                f"ratio undefined; see groom_completion_rate."),
    ))

    # 3. groom_comment_churn (supporting) — distinct issues the groomer keeps
    #    commenting on (≥3 in-window) without ever completing (config#2147's
    #    burn signature). Lower better; red-line 20 post-drain per config#2151,
    #    provisional target 5.
    per_issue: dict[tuple[str, int], dict[str, int]] = {}
    for rec in dispositions:
        stats = per_issue.setdefault((rec.get("repo"), rec.get("number")), {"commented": 0, "completed": 0})
        if rec["disposition"] == "commented":
            stats["commented"] += 1
        elif rec["disposition"] in _COMPLETION_DISPOSITIONS:
            stats["completed"] += 1
    churn = sum(1 for s in per_issue.values()
                if s["commented"] >= _CHURN_MIN_COMMENTS and s["completed"] == 0)
    n_issues = len(per_issue)
    components.append(build_metric(
        name="groom_comment_churn", module=MODULE, metric_type="count", criticality="supporting",
        estimator="distinct_issue_churn_count", measurement_horizon=f"{WINDOW_DAYS}d_window",
        reliability="medium",
        value=float(churn), unit=COUNT_ISSUES, n_samples=n_issues, n_floor=10,
        higher_is_better=False, source_path=src,
        reason=(f"groom_comment_churn = {churn} issue(s) with ≥{_CHURN_MIN_COMMENTS} commented "
                f"dispositions and no completion in {span} (N={n_issues} distinct issues) vs "
                f"provisional target 5 / red-line 20 (baseline 171 pre-drain, config#2147/#2151)."),
    ))

    # 4. groom_lost_chunks (supporting) — chunk invocations that exhausted the
    #    agent turn budget (max_turns) and forfeited their remaining issues.
    #    Lower better; red-line 5/week binds as steady-state once config#2148
    #    (zero-output WET-run fix) lands — a RED before that is the known,
    #    tracked defect surfacing honestly, not alarm noise.
    lost = sum(1 for run in runs for entry in run["chunk_log"]
               if isinstance(entry, str) and _MAX_TURNS_RE.search(entry))
    components.append(build_metric(
        name="groom_lost_chunks", module=MODULE, metric_type="count", criticality="supporting",
        estimator="chunk_failure_signature_count", measurement_horizon=f"{WINDOW_DAYS}d_window",
        reliability="medium",
        value=float(lost), unit=COUNT_CHUNKS, n_samples=n_runs, n_floor=5,
        higher_is_better=False, source_path=src,
        reason=(f"groom_lost_chunks = {lost} max_turns chunk failure(s) across {n_runs} runs "
                f"({span}, {len(keys)} artifacts) vs target 0 / red-line 5 "
                f"(red-line binds once config#2148 lands; baseline ~24)."),
    ))

    return components
