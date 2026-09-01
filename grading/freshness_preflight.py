"""freshness_preflight.py — hard input-freshness gate for the Evaluator/ReportCard.

alpha-engine-config#3058 (Brian ruling 2026-07-20): "if the evaluator is
evaluating on stale data its report is COMPLETELY USELESS — it should
hard-fail before evaluating stale outputs." In the 2026-07-18 weekly arc the
Evaluator (this repo's grading layer) computed the full Report Card v2 —
e2e-lift metrics, producer-leaderboard point, weekly assessment — against
``backtest/{date}/e2e_lift.json``, itself derived from
``predictor/research_free_backfill/predictor_outcomes_research_free.parquet``,
whose latest cohort was 8+ days stale because the producer had silently
no-oped (config-I3053). Nothing in this repo asserted freshness before
grading; the stale run produced an authoritative-looking report.

config-I3053 fixed the PRODUCER side (crucible-backtester's Saturday SF now
asserts ``assert_champion_feed_fresh`` right after the backfill, when the
live champion depends on it). This module is the CONSUMER-side hard gate for
the assessment plane — defense in depth, the same posture as
``crucible-executor``'s ``executor/champion.py::_check_freshness``, which is
the proven reference implementation this mirrors:

  - freshness is judged on a CONTENT-DERIVED date wherever the artifact
    carries one (``metrics.json``'s ``run_date`` field, ``eod_pnl.csv``'s
    ``date`` column) — S3 ``LastModified`` is deliberately never trusted
    alone, because a no-op rewrite refreshes it while the content stays
    stale (exactly the 2026-07-18 incident's failure mode);
  - for artifacts with no independent content date (``e2e_lift.json`` and
    its siblings persist no ``run_date``/cohort-date of their own — the
    payload's freshness is entirely inherited from whichever upstream
    cohort the backtester happened to have on hand when it ran), the
    STRONGEST available signal is the artifact's own resolved instance date
    — which S3 key under ``backtest/{date}/`` actually answered — asserted
    to fall inside the run's own ISO week. This closes the loophole the
    incident exploited: ``grading.artifacts.get_json_windowed``'s 10-day
    resilience walk-back (deliberately generous, for partial/retried
    Saturday runs) will silently accept last week's artifact with no signal
    that grading happened on stale content;
  - the gated artifacts, their S3 key templates and their cadences are READ
    from ``alpha-engine-config/private-docs/ARTIFACT_REGISTRY.yaml`` at run
    time, via ``grading.artifact_registry`` (the published S3 mirror, typed
    through ``nousergon_lib.artifact_freshness.ArtifactSpec``). This module
    declares only WHICH declared artifacts it hard-gates — a property of the
    grader, not of the registry — and how each one's content date is
    recovered. Everything the registry already declares comes from the
    registry (`alpha-engine-config-I9731`). Cadence drives the window:
    ``saturday_sf``/``sunday_sf`` artifacts must carry data from the run's own
    week; ``eod_sf``/``weekday_sf`` artifacts must carry data from the last
    NYSE trading day (calendar-aware via ``krepis.dates``); ``event_driven``
    artifacts are presence-gated only, because their row declares that their
    age is not a staleness signal and that their liveness rides a separate
    monitored anchor. A cadence with no window rule here RAISES — an
    unrecognised cadence must never silently grade as fresh;
  - the registry is not optional and has no fallback: if it cannot be loaded,
    the preflight raises ``RegistryUnavailableError`` rather than grading
    against nothing. A fallback table IS the drift this gate was rebuilt to
    remove;
  - ANY breach — stale content, or a declared input missing outright — HARD
    FAILS (raises) naming the artifact, its resolved content date, and the
    expected window. No warn-and-continue, no partial report: a caught
    exception here must propagate out of the SF state (rc != 0).

Wired at the single computation chokepoint (``grading.aggregate.
build_report_card``, called by both the Lambda handler and the CLI) so every
caller — including a ``skip_*``-flagged partial rerun, which is exactly the
scenario that makes a consumer-side check load-bearing (config-I3053 image:
a recovery rerun that skips the producer stage) — runs this preflight before
any tile is computed. Mirrored explicitly in ``grading.aggregate.
write_report_card``'s ``snapshot=True`` path (the ReportCard freeze step) so
a frozen weekly record — the worst-case artifact, per the issue — can never
be produced from a build that skipped the gate.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass
from datetime import date as _date
from datetime import timedelta

import boto3
from botocore.exceptions import ClientError

from krepis.dates import is_fresh_in_trading_days
from nousergon_lib.artifact_freshness import ArtifactSpec

from grading.artifact_registry import (
    REGISTRY_BUCKET,
    REGISTRY_KEY,
    RegistryRowMissingError,
    RegistryUnavailableError,
    load_specs,
)
from grading.artifacts import DEFAULT_ARTIFACT_MAX_AGE_DAYS

logger = logging.getLogger(__name__)

__all__ = [
    "GATED_ARTIFACT_IDS",
    "InputArtifactError",
    "MissingInputArtifactError",
    "RegistryRowMissingError",
    "RegistryUnavailableError",
    "StaleInputArtifactError",
    "assert_input_freshness",
]


class InputArtifactError(RuntimeError):
    """Base class for the two named preflight failures below."""


class MissingInputArtifactError(InputArtifactError):
    """A declared input artifact could not be read at all (NoSuchKey/absent
    body/unparseable) — raised, never silently skipped or graded N/A. Only
    the freshness preflight's own declared inputs are hard-required this way;
    the tiles' own optional/known-unwired artifacts keep their existing
    graceful-N/A posture untouched."""


class StaleInputArtifactError(InputArtifactError):
    """A declared input artifact was read successfully but its content date
    (or, absent a content date, its resolved S3 instance date) falls outside
    the cadence-derived freshness window for this run."""


def _week_start(run_date: _date) -> _date:
    """Monday of ``run_date``'s ISO week (weeks run Mon-Sun, matching the
    Saturday-SF cadence: a Saturday run's own week started the Monday four
    days earlier)."""
    return run_date - timedelta(days=run_date.weekday())


def _in_run_week(content_date: _date, run_date: _date) -> bool:
    """Weekly-cadence freshness: content must fall within ``[Monday of
    run_date's week, run_date]`` — never before this week, never after (a
    future-dated artifact is a clock-skew/mislabel bug, not "fresh")."""
    return _week_start(run_date) <= content_date <= run_date


def _parse_date(raw: object) -> _date | None:
    if not raw:
        return None
    try:
        return _date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


def _get_json_body(s3, bucket: str, key: str) -> dict | None:
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("NoSuchKey", "404"):
            return None
        logger.error("freshness_preflight: S3 read failed for s3://%s/%s: %s", bucket, key, e)
        raise
    return json.loads(resp["Body"].read())


def _newest_dated_instance_for(
    s3, bucket: str, spec: "ArtifactSpec", run_date: _date,
    *, max_age_days: int = DEFAULT_ARTIFACT_MAX_AGE_DAYS,
) -> _date | None:
    """Resolved instance date for a date-templated JSON artifact: the freshest
    existing copy at/before ``run_date`` within the same resilience window
    ``grading.artifacts.get_json_windowed`` uses to build the tiles — so this
    preflight judges the SAME instance the tiles will actually grade, not a
    stricter/looser one. The key comes from the artifact's declared
    ``s3_key_template``. Returns ``None`` if no instance exists anywhere in the
    window (a genuinely missing artifact)."""
    for delta in range(max_age_days + 1):
        d = run_date - timedelta(days=delta)
        key = _resolve_key(spec, d)
        try:
            body = _get_json_body(s3, bucket, key)
        except (json.JSONDecodeError, ValueError):
            continue
        if body is not None:
            return d
    return None


@dataclass(frozen=True)
class _CheckOutcome:
    artifact_id: str
    content_date: str
    window: str


# ── Registry-derived key + window resolution ────────────────────────────────
#
# Everything below reads the artifact's declared row rather than a literal.
# `spec.s3_key_template` is the key; `spec.cadence` is the window rule.

#: Cadences whose artifacts must carry data from the RUN'S OWN ISO week.
_WEEKLY_CADENCES: frozenset[str] = frozenset({"saturday_sf", "sunday_sf"})
#: Cadences whose artifacts must carry data from the last NYSE session(s).
_DAILY_CADENCES: frozenset[str] = frozenset({"eod_sf", "weekday_sf"})


def _resolve_key(spec: ArtifactSpec, d: _date) -> str:
    """Render a declared key template for instance date ``d``.

    Mirrors the placeholder set ``nousergon_lib.artifact_freshness._format_key``
    supports (``{date}`` / ``{trading_day}`` / ``{cycle_label}``). A template
    with no placeholder renders to itself, which is how the fixed-key rows
    (``trades/eod_pnl.csv``, ``predictor/weights/meta/manifest.json``) work.
    """
    iso = d.isoformat()
    try:
        return spec.s3_key_template.format(date=iso, trading_day=iso, cycle_label=iso)
    except (KeyError, IndexError) as exc:
        raise RegistryUnavailableError(
            f"{spec.artifact_id}: declared s3_key_template "
            f"{spec.s3_key_template!r} carries a placeholder this preflight "
            f"cannot resolve ({type(exc).__name__}: {exc}). Supported: "
            "{date} / {trading_day} / {cycle_label}."
        ) from exc


def _window_label(spec: ArtifactSpec, run_date: _date, *, max_stale_trading_days: int = 1) -> str:
    """Human-readable window this run judges ``spec`` against."""
    if spec.cadence in _WEEKLY_CADENCES:
        return f"week of {_week_start(run_date).isoformat()}"
    if spec.cadence in _DAILY_CADENCES:
        return f"<= {max_stale_trading_days} trading day(s) of {run_date.isoformat()}"
    return (
        f"event_driven — age is not a staleness signal for this row; its "
        f"producer liveness rides {spec.liveness_via!r}"
    )


def _assert_in_window(
    spec: ArtifactSpec,
    content_date: _date,
    run_date: _date,
    *,
    what: str,
    max_stale_trading_days: int = 1,
) -> str:
    """Apply the cadence-declared freshness window; return its label.

    Raises ``StaleInputArtifactError`` on breach, and
    ``RegistryUnavailableError`` on a cadence this preflight has no rule for —
    never a silent pass. An unrecognised cadence is a registry change this
    grader has not been taught to read, and treating it as fresh is how a
    re-cadenced row becomes an ungated input.
    """
    if spec.cadence in _WEEKLY_CADENCES:
        if not _in_run_week(content_date, run_date):
            raise StaleInputArtifactError(
                f"{spec.artifact_id} is stale: {what}={content_date.isoformat()} is "
                f"outside this week's window [{_week_start(run_date).isoformat()}, "
                f"{run_date.isoformat()}] for the evaluator run at "
                f"run_date={run_date.isoformat()}. Declared cadence "
                f"{spec.cadence!r}, severity {spec.severity!r}, owner "
                f"{spec.owner_repo!r} (ARTIFACT_REGISTRY.yaml)."
            )
    elif spec.cadence in _DAILY_CADENCES:
        if not is_fresh_in_trading_days(content_date, run_date, max_stale=max_stale_trading_days):
            raise StaleInputArtifactError(
                f"{spec.artifact_id} is stale: {what}={content_date.isoformat()} is "
                f"more than {max_stale_trading_days} NYSE trading day(s) behind "
                f"run_date={run_date.isoformat()}. Declared cadence "
                f"{spec.cadence!r}, severity {spec.severity!r}, owner "
                f"{spec.owner_repo!r} (ARTIFACT_REGISTRY.yaml)."
            )
    elif spec.cadence != "event_driven":
        raise RegistryUnavailableError(
            f"{spec.artifact_id}: declared cadence {spec.cadence!r} has no "
            "freshness-window rule in the evaluator preflight. Add one (and a "
            "test) rather than letting an unrecognised cadence grade as fresh."
        )
    return _window_label(spec, run_date, max_stale_trading_days=max_stale_trading_days)


# ── Per-artifact content-date recovery ──────────────────────────────────────
#
# The registry declares WHERE an artifact lives and HOW OFTEN it is written.
# It does not declare how to recover the date its CONTENT was computed for —
# that is per-artifact knowledge (a JSON field, a CSV column, or, where the
# payload persists no date of its own, the resolved instance key). Each
# function below owns exactly that, and nothing else.


def _check_metrics_json(s3, bucket: str, spec: ArtifactSpec, run_date: _date) -> _CheckOutcome:
    """``backtest_metrics`` — carries an explicit ``run_date`` field (excluded
    from the signal_quality ``overall`` payload downstream, but present in the
    raw artifact), the strongest content-derived signal available for the
    research-free-derived e2e counterfactual family: the backtester stamps
    this file with the cohort date it actually computed against, on the SAME
    run that writes ``e2e_lift.json``."""
    key = _resolve_key(spec, run_date)
    body = _get_json_body(s3, bucket, key)
    if body is None:
        raise MissingInputArtifactError(
            f"{spec.artifact_id}: no artifact at s3://{bucket}/{key} — "
            f"required input for the weekly assessment (run_date={run_date.isoformat()})."
        )
    content_date = _parse_date(body.get("run_date"))
    if content_date is None:
        raise MissingInputArtifactError(
            f"{spec.artifact_id}: s3://{bucket}/{key} has no readable "
            "'run_date' field — cannot verify freshness of a content-dated artifact."
        )
    window = _assert_in_window(spec, content_date, run_date, what="content run_date")
    return _CheckOutcome(spec.artifact_id, content_date.isoformat(), window)


def _check_e2e_lift(s3, bucket: str, spec: ArtifactSpec, run_date: _date) -> _CheckOutcome:
    """``backtest_e2e_lift`` — the artifact directly named in the 2026-07-18
    incident: computed from the research-free parquet
    (``predictor_outcomes_research_free``) among other cohorts, but persists
    no cohort-date of its own. The strongest available signal without reading
    the parquet directly (out of this repo's scope — the parquet is
    read+aggregated upstream by crucible-backtester's evaluate.py) is the
    artifact's own RESOLVED S3 instance date: which day's key actually
    answered. Walking back silently past the run's own week (as the tiles'
    resilience window tolerates for partial/retried runs) is exactly the
    loophole a silently no-op'd producer exploits.

    NOTE (`alpha-engine-config-I9731`): the hardcoded table this replaced named
    ``research_producer_leaderboard`` as this entry's registry row. That was
    wrong — the row is ``backtest_e2e_lift`` — and it is the measured instance
    of the drift a hand-maintained mirror produces.
    """
    instance_date = _newest_dated_instance_for(s3, bucket, spec, run_date)
    if instance_date is None:
        raise MissingInputArtifactError(
            f"{spec.artifact_id}: no instance found under s3://{bucket}/"
            f"{spec.s3_key_template} within the {DEFAULT_ARTIFACT_MAX_AGE_DAYS}-day "
            f"resilience window of run_date={run_date.isoformat()}."
        )
    try:
        window = _assert_in_window(spec, instance_date, run_date, what="freshest resolvable instance")
    except StaleInputArtifactError as exc:
        raise StaleInputArtifactError(
            f"{exc} This is the artifact class behind the 2026-07-18 incident "
            "(config-I3053/config#3058): a silently no-op'd research-free-backfill "
            "producer left this week's e2e_lift.json unrefreshed, and grading "
            "proceeded on last week's cohort."
        ) from exc
    return _CheckOutcome(spec.artifact_id, instance_date.isoformat(), window)


def _check_predictor_manifest(s3, bucket: str, spec: ArtifactSpec, run_date: _date) -> _CheckOutcome:
    """``predictor_meta_weights_manifest`` — the model-zoo promotion record the
    Predictor tile grades leak-free CPCV IC from (``meta_model_oos_ic_cpcv``).
    Fixed-key pointer, no ``{date}`` segment, so the only reliable
    content-derived signal is a date-shaped field inside the manifest itself;
    every live-shipped manifest carries one of ``training_date`` / ``run_date``
    / ``date`` (config#1601 / L4468 SSOT).

    Its declared cadence is ``event_driven`` since 2026-08-28
    (`alpha-engine-config-I9018`): ``promote_to_champion`` is the sole writer
    and a week with no promotion is the expected case, so the row's own age
    carries no staleness signal and its producer liveness rides the separately
    monitored ``model_zoo_leaderboard_latest`` anchor. Presence and a readable
    date are still hard-required — an absent or undated manifest is a defect in
    every cadence.

    `alpha-engine-config-I9255`'s ``_incumbent_retained_this_week`` carve-out —
    a second S3 read of ``predictor/model_zoo/promotions/{date}.json`` to prove
    a stale-looking manifest was an intentional no-promotion week — is RETIRED
    by this change (`alpha-engine-config-I9731`). It was a hand-rolled subset of
    exactly what ``cadence: event_driven`` declares, on the one row that now
    declares it; keeping both would be the same duplicated-truth defect one
    layer down.
    """
    key = _resolve_key(spec, run_date)
    body = _get_json_body(s3, bucket, key)
    if body is None:
        raise MissingInputArtifactError(
            f"{spec.artifact_id}: no artifact at s3://{bucket}/{key} — required "
            f"model-zoo promotion-record input for the weekly assessment."
        )
    raw = body.get("training_date") or body.get("run_date") or body.get("date")
    content_date = _parse_date(raw)
    if content_date is None:
        raise MissingInputArtifactError(
            f"{spec.artifact_id}: s3://{bucket}/{key} has no readable "
            "training_date/run_date/date field — cannot verify freshness."
        )
    window = _assert_in_window(spec, content_date, run_date, what="content date")
    return _CheckOutcome(spec.artifact_id, content_date.isoformat(), window)


def _check_signals(s3, bucket: str, spec: ArtifactSpec, run_date: _date) -> _CheckOutcome:
    """``research_signals`` — the research signals input the Portfolio Outcome
    tile joins for ``regime_weighted_alpha``. Key-templated by date, so its own
    key IS the content date; resolved the same way ``e2e_lift.json`` is
    (freshest instance at/before run_date, asserted against the declared
    cadence window) rather than requiring an exact same-day key
    (Friday-anchored trading-day runs legitimately read a slightly earlier
    signals.json)."""
    instance_date = None
    for delta in range(DEFAULT_ARTIFACT_MAX_AGE_DAYS + 1):
        d = run_date - timedelta(days=delta)
        key = _resolve_key(spec, d)
        try:
            resp = s3.head_object(Bucket=bucket, Key=key)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey"):
                continue
            logger.error("freshness_preflight: HEAD failed for s3://%s/%s: %s", bucket, key, e)
            raise
        else:
            del resp
            instance_date = d
            break
    if instance_date is None:
        raise MissingInputArtifactError(
            f"{spec.artifact_id}: no instance found under s3://{bucket}/"
            f"{spec.s3_key_template} within {DEFAULT_ARTIFACT_MAX_AGE_DAYS} days of "
            f"run_date={run_date.isoformat()}."
        )
    window = _assert_in_window(spec, instance_date, run_date, what="freshest resolvable instance")
    return _CheckOutcome(spec.artifact_id, instance_date.isoformat(), window)


def _check_eod_pnl(s3, bucket: str, spec: ArtifactSpec, run_date: _date, *, max_stale_trading_days: int = 1) -> _CheckOutcome:
    """``eod_reconcile_pnl`` — the portfolio-outcome ground truth (NAV /
    alpha-vs-SPY). Carries a real per-row ``date`` column; content-derived
    freshness is ``max(date)`` across all rows, asserted against the declared
    ``eod_sf`` cadence window (calendar-aware via ``krepis.dates`` — a Saturday
    run must not be judged stale merely because no trading happened over the
    weekend)."""
    key = _resolve_key(spec, run_date)
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("NoSuchKey", "404"):
            raise MissingInputArtifactError(
                f"{spec.artifact_id}: no artifact at s3://{bucket}/{key} — required "
                "portfolio-outcome ground-truth input for the weekly assessment."
            ) from e
        logger.error("freshness_preflight: S3 read failed for s3://%s/%s: %s", bucket, key, e)
        raise
    text = resp["Body"].read().decode("utf-8")
    rows = [r for r in csv.DictReader(io.StringIO(text)) if r.get("date")]
    dates = sorted(d for d in (_parse_date(r["date"]) for r in rows) if d is not None)
    if not dates:
        raise MissingInputArtifactError(
            f"{spec.artifact_id}: s3://{bucket}/{key} has no parseable 'date' rows — "
            "cannot verify freshness of a content-dated artifact."
        )
    content_date = dates[-1]
    window = _assert_in_window(
        spec, content_date, run_date,
        what="freshest row date", max_stale_trading_days=max_stale_trading_days,
    )
    return _CheckOutcome(spec.artifact_id, content_date.isoformat(), window)


# The artifacts this grader hard-gates, keyed by their ARTIFACT_REGISTRY.yaml
# `artifact_id`. This tuple is the ONLY registry-adjacent literal left in this
# module, and deliberately so: WHICH declared artifacts a grader refuses to
# grade without is a property of the grader, not of the registry — the registry
# declares 185 artifacts and this gate covers five of them. Everything else
# about each one (bucket, key template, cadence, SLA, severity, owner, liveness
# anchor) is READ from the registry at run time by `grading.artifact_registry`,
# and `tests/test_freshness_preflight_registry_contract.py` fails if any id here
# stops resolving to a live row.
#
# Artifacts the tiles already treat as legitimately-optional/known-unwired
# (veto_value, scanner_opt, cio_opt, sizing_ab, predictor_sizing — see
# grading/artifacts.py module docstring) are deliberately NOT hard-gated here:
# promoting them would turn a documented "not yet persisted" state into a false
# hard-fail.
GATED_ARTIFACT_IDS: tuple[str, ...] = (
    "backtest_metrics",
    "backtest_e2e_lift",
    "predictor_meta_weights_manifest",
    "research_signals",
    "eod_reconcile_pnl",
)

#: artifact_id -> content-date recovery function. Keys must equal
#: :data:`GATED_ARTIFACT_IDS` exactly; the contract test asserts it.
_CHECK_FNS = {
    "backtest_metrics": _check_metrics_json,
    "backtest_e2e_lift": _check_e2e_lift,
    "predictor_meta_weights_manifest": _check_predictor_manifest,
    "research_signals": _check_signals,
    "eod_reconcile_pnl": _check_eod_pnl,
}


def assert_input_freshness(
    bucket: str, run_date: str, s3_client=None, *, dry_run: bool = False,
) -> dict:
    """Hard preflight: raise ``MissingInputArtifactError`` /
    ``StaleInputArtifactError`` naming the artifact, its resolved content
    date, and the expected window, on the FIRST breach found (fail fast —
    the issue's acceptance criteria wants a named-artifact error, not an
    aggregate report). Returns a provenance dict of every check that PASSED
    (only reachable when ALL of them did — any failure raises instead of
    returning), for the caller to fold into ``_provenance``.

    Must run before any metric computation — see ``grading.aggregate.
    build_report_card`` (called by both the Lambda handler and the CLI) and
    ``grading.aggregate.write_report_card``'s ``snapshot=True`` path, which
    both call this as their first step.

    ``dry_run`` — the Friday-PM shell run (`alpha-engine-config-I7392`)
    ==============================================================

    The 2026-07-20 ruling this gate exists for is about a REAL run:
    "if the evaluator is evaluating on stale data its report is COMPLETELY
    USELESS — it should hard-fail before evaluating stale outputs." That
    premise does not hold on the dry path, because a dry run evaluates
    nothing and persists nothing — so it cannot evaluate stale data.

    It could not, however, ever PASS either. The weekly SF's
    ``ApplyShellRunDefaults`` states "every substantive workload now boots +
    runs DRY; ZERO skip-exceptions remain", so every producer runs
    ``--preflight-only`` and writes no artifact. This gate then hard-failed on
    the absence it had just guaranteed. Measured on execution
    ``friday-shell-2026-08-14-validate-final``: the run reached the END of the
    pipeline with every workload stage green, then died here on
    ``metrics.json: no artifact at backtest/2026-08-14/``. Twelve shell runs
    exist and only two ever succeeded, both predating that rewire.

    So under ``dry_run`` every check still RUNS — the S3 read is attempted for
    each artifact, which is precisely what the rehearsal exists to prove
    (container boot, imports, S3 IAM + transport) — but a missing or stale
    artifact is recorded as ``UNMEASURED`` with its reason instead of raising.

    **A dry run still fails loud on a real defect, and that is the
    load-bearing half.** Only ``MissingInputArtifactError`` and
    ``StaleInputArtifactError`` are absorbed. A read that cannot be PERFORMED
    — AccessDenied, a transport error, anything that is not a 404 — is raised
    by ``_get_json_body`` as a ``ClientError`` and propagates untouched, on
    the dry path exactly as on the real one. Absence-of-artifact and
    cannot-read must never collapse to the same verdict: a rehearsal that
    cannot fail is worse than one that always does, because it is green by
    construction and stops being evidence of anything.
    """
    s3 = s3_client or boto3.client("s3")
    try:
        run_d = _date.fromisoformat(run_date)
    except (ValueError, TypeError) as exc:
        raise MissingInputArtifactError(
            f"freshness_preflight: run_date={run_date!r} is not a valid ISO date — "
            "cannot resolve the freshness window."
        ) from exc

    # The declared registry is loaded BEFORE any check runs, and on the dry
    # path exactly as on the real one. It is never absorbed into UNMEASURED:
    # "an input artifact is absent" is a finding about the fleet a rehearsal
    # may record, while "I could not load the predicates I grade against" is a
    # defect in the grader — a preflight that cannot read its own rules and
    # reports clean is worse than one that fails (alpha-engine-config-I9731).
    specs = load_specs(GATED_ARTIFACT_IDS, s3_client=s3)

    checked: list[dict] = []
    unmeasured = 0
    for name in GATED_ARTIFACT_IDS:
        fn = _CHECK_FNS[name]
        try:
            outcome = fn(s3, bucket, specs[name], run_d)
        except (MissingInputArtifactError, StaleInputArtifactError) as exc:
            if not dry_run:
                raise
            # Dry path only. NOTE what is NOT caught here: a ClientError from
            # `_get_json_body` (AccessDenied, transport, anything non-404)
            # propagates, so "the artifact is absent" and "I could not read"
            # stay distinguishable. See this function's docstring.
            unmeasured += 1
            checked.append({
                "artifact_id": name,
                "content_date": None,
                "window": None,
                "status": "UNMEASURED",
                "reason": str(exc),
            })
            logger.warning(
                "freshness_preflight: %s UNMEASURED on the dry path — %s",
                name, exc,
            )
            continue
        checked.append({
            "artifact_id": outcome.artifact_id,
            "content_date": outcome.content_date,
            "window": outcome.window,
            "status": "ok",
        })
        logger.info(
            "freshness_preflight: %s OK (content_date=%s, window=%s)",
            outcome.artifact_id, outcome.content_date, outcome.window,
        )
    provenance = {
        "run_date": run_date,
        "checks": checked,
        # Console/report-card provenance mirrors grading/coverage.py's
        # `denominator_source` shape: name the document the predicates came
        # from, so a card can be traced to the registry revision it was graded
        # against without reading this module (alpha-engine-config-I9731).
        "predicate_source": f"s3://{REGISTRY_BUCKET}/{REGISTRY_KEY}#artifacts",
    }
    if dry_run:
        # Stated in BOTH polarities (sf-pipeline-policy 2.3a): a dry run that
        # measured everything and one that measured nothing must not render
        # identically. `dry_run` is present on every dry invocation, and
        # `unmeasured` says how much of the gate actually had inputs.
        provenance["dry_run"] = True
        provenance["unmeasured"] = unmeasured
        provenance["measured"] = len(GATED_ARTIFACT_IDS) - unmeasured
    return provenance
