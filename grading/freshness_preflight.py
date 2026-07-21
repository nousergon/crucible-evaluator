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
  - cadence comes from ``alpha-engine-config/ARTIFACT_REGISTRY.yaml``:
    ``saturday_sf`` artifacts must carry data from the run's own week;
    ``eod_sf``/daily artifacts must carry data from the last NYSE trading
    day (calendar-aware via ``krepis.dates``);
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

from grading.artifacts import DEFAULT_ARTIFACT_MAX_AGE_DAYS

logger = logging.getLogger(__name__)


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


def _newest_dated_instance(
    s3, bucket: str, prefix: str, filename: str, run_date: _date,
    *, max_age_days: int = DEFAULT_ARTIFACT_MAX_AGE_DAYS,
) -> _date | None:
    """Resolved instance date for a ``backtest/{date}/<filename>`` artifact:
    the freshest existing copy at/before ``run_date`` within the same
    resilience window ``grading.artifacts.get_json_windowed`` uses to build
    the tiles — so this preflight judges the SAME instance the tiles will
    actually grade, not a stricter/looser one. Returns ``None`` if no
    instance exists anywhere in the window (a genuinely missing artifact)."""
    for delta in range(max_age_days + 1):
        d = run_date - timedelta(days=delta)
        key = f"{prefix.format(date=d.isoformat())}/{filename}"
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


def _check_metrics_json(s3, bucket: str, run_date: _date) -> _CheckOutcome:
    """``backtest/{run_date}/metrics.json`` — carries an explicit ``run_date``
    field (excluded from the signal_quality ``overall`` payload downstream,
    but present in the raw artifact), the strongest content-derived signal
    available for the research-free-derived e2e counterfactual family: the
    backtester stamps this file with the cohort date it actually computed
    against, on the SAME run that writes ``e2e_lift.json``."""
    prefix = f"backtest/{run_date.isoformat()}"
    body = _get_json_body(s3, bucket, f"{prefix}/metrics.json")
    if body is None:
        raise MissingInputArtifactError(
            f"metrics.json: no artifact at s3://{bucket}/{prefix}/metrics.json — "
            f"required input for the weekly assessment (run_date={run_date.isoformat()})."
        )
    content_date = _parse_date(body.get("run_date"))
    if content_date is None:
        raise MissingInputArtifactError(
            f"metrics.json: s3://{bucket}/{prefix}/metrics.json has no readable "
            "'run_date' field — cannot verify freshness of a content-dated artifact."
        )
    if not _in_run_week(content_date, run_date):
        raise StaleInputArtifactError(
            f"metrics.json is stale: content run_date={content_date.isoformat()} is "
            f"outside this week's window [{_week_start(run_date).isoformat()}, "
            f"{run_date.isoformat()}] for the evaluator run at run_date={run_date.isoformat()}."
        )
    return _CheckOutcome("metrics_json", content_date.isoformat(), f"week of {_week_start(run_date).isoformat()}")


def _check_e2e_lift(s3, bucket: str, run_date: _date) -> _CheckOutcome:
    """``backtest/{date}/e2e_lift.json`` — the artifact directly named in the
    2026-07-18 incident: computed from the research-free parquet
    (``predictor_outcomes_research_free``) among other cohorts, but persists
    no cohort-date of its own. The strongest available signal without
    reading the parquet directly (out of this repo's scope — the parquet is
    read+aggregated upstream by crucible-backtester's evaluate.py) is the
    artifact's own RESOLVED S3 instance date: which day's
    ``backtest/{date}/e2e_lift.json`` actually answered. Walking back
    silently past the run's own week (as the tiles' resilience window
    tolerates for partial/retried runs) is exactly the loophole a silently
    no-op'd producer exploits — assert the resolved instance falls in-week.
    """
    instance_date = _newest_dated_instance(s3, bucket, "backtest/{date}", "e2e_lift.json", run_date)
    if instance_date is None:
        raise MissingInputArtifactError(
            f"e2e_lift.json: no instance found under s3://{bucket}/backtest/*/e2e_lift.json "
            f"within the {DEFAULT_ARTIFACT_MAX_AGE_DAYS}-day resilience window of "
            f"run_date={run_date.isoformat()}."
        )
    if not _in_run_week(instance_date, run_date):
        raise StaleInputArtifactError(
            f"e2e_lift.json is stale: the freshest resolvable instance is dated "
            f"{instance_date.isoformat()}, outside this week's window "
            f"[{_week_start(run_date).isoformat()}, {run_date.isoformat()}] for the "
            f"evaluator run at run_date={run_date.isoformat()}. This is the artifact "
            "class behind the 2026-07-18 incident (config-I3053/config#3058): a "
            "silently no-op'd research-free-backfill producer left this week's "
            "e2e_lift.json unrefreshed, and grading proceeded on last week's cohort."
        )
    return _CheckOutcome("e2e_lift_json", instance_date.isoformat(), f"week of {_week_start(run_date).isoformat()}")


def _check_predictor_manifest(s3, bucket: str, run_date: _date) -> _CheckOutcome:
    """``predictor/weights/meta/manifest.json`` — the model-zoo promotion
    record the Predictor tile grades leak-free CPCV IC from
    (``meta_model_oos_ic_cpcv``). Fixed-key pointer, no ``{date}`` segment, so
    the only reliable content-derived signal is a date-shaped field inside
    the manifest itself; every live-shipped manifest carries one of
    ``training_date`` / ``run_date`` / ``date`` (config#1601 / L4468 SSOT).
    """
    key = "predictor/weights/meta/manifest.json"
    body = _get_json_body(s3, bucket, key)
    if body is None:
        raise MissingInputArtifactError(
            f"predictor manifest: no artifact at s3://{bucket}/{key} — required "
            f"model-zoo promotion-record input for the weekly assessment."
        )
    raw = body.get("training_date") or body.get("run_date") or body.get("date")
    content_date = _parse_date(raw)
    if content_date is None:
        raise MissingInputArtifactError(
            f"predictor manifest: s3://{bucket}/{key} has no readable "
            "training_date/run_date/date field — cannot verify freshness."
        )
    if not _in_run_week(content_date, run_date):
        raise StaleInputArtifactError(
            f"predictor manifest is stale: content date={content_date.isoformat()} is "
            f"outside this week's window [{_week_start(run_date).isoformat()}, "
            f"{run_date.isoformat()}] for the evaluator run at run_date={run_date.isoformat()}."
        )
    return _CheckOutcome("predictor_meta_weights_manifest", content_date.isoformat(), f"week of {_week_start(run_date).isoformat()}")


def _check_signals(s3, bucket: str, run_date: _date) -> _CheckOutcome:
    """``signals/{date}/signals.json`` — the research signals input the
    Portfolio Outcome tile joins for ``regime_weighted_alpha``. Key-templated
    by date, so its own key IS the content date; resolved the same way
    ``e2e_lift.json`` is (freshest instance at/before run_date, asserted
    in-week) rather than requiring an exact same-day key (Friday-anchored
    trading-day runs legitimately read a slightly earlier signals.json)."""
    instance_date = None
    for delta in range(DEFAULT_ARTIFACT_MAX_AGE_DAYS + 1):
        d = run_date - timedelta(days=delta)
        key = f"signals/{d.isoformat()}/signals.json"
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
            f"signals.json: no instance found under s3://{bucket}/signals/*/signals.json "
            f"within {DEFAULT_ARTIFACT_MAX_AGE_DAYS} days of run_date={run_date.isoformat()}."
        )
    if not _in_run_week(instance_date, run_date):
        raise StaleInputArtifactError(
            f"signals.json is stale: the freshest resolvable instance is dated "
            f"{instance_date.isoformat()}, outside this week's window "
            f"[{_week_start(run_date).isoformat()}, {run_date.isoformat()}] for the "
            f"evaluator run at run_date={run_date.isoformat()}."
        )
    return _CheckOutcome("research_signals", instance_date.isoformat(), f"week of {_week_start(run_date).isoformat()}")


def _check_eod_pnl(s3, bucket: str, run_date: _date, *, max_stale_trading_days: int = 1) -> _CheckOutcome:
    """``trades/eod_pnl.csv`` — the portfolio-outcome ground truth (NAV /
    alpha-vs-SPY). Carries a real per-row ``date`` column; content-derived
    freshness is ``max(date)`` across all rows, asserted within
    ``max_stale_trading_days`` NYSE sessions of ``run_date`` (calendar-aware
    via ``krepis.dates`` — a Saturday run must not be judged stale merely
    because no trading happened over the weekend). ``eod_sf`` (daily)
    cadence per ARTIFACT_REGISTRY.yaml."""
    key = "trades/eod_pnl.csv"
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("NoSuchKey", "404"):
            raise MissingInputArtifactError(
                f"eod_pnl.csv: no artifact at s3://{bucket}/{key} — required "
                "portfolio-outcome ground-truth input for the weekly assessment."
            ) from e
        logger.error("freshness_preflight: S3 read failed for s3://%s/%s: %s", bucket, key, e)
        raise
    text = resp["Body"].read().decode("utf-8")
    rows = [r for r in csv.DictReader(io.StringIO(text)) if r.get("date")]
    dates = sorted(d for d in (_parse_date(r["date"]) for r in rows) if d is not None)
    if not dates:
        raise MissingInputArtifactError(
            f"eod_pnl.csv: s3://{bucket}/{key} has no parseable 'date' rows — "
            "cannot verify freshness of a content-dated artifact."
        )
    content_date = dates[-1]
    if not is_fresh_in_trading_days(content_date, run_date, max_stale=max_stale_trading_days):
        raise StaleInputArtifactError(
            f"eod_pnl.csv is stale: freshest row date={content_date.isoformat()} is "
            f"more than {max_stale_trading_days} NYSE trading day(s) behind "
            f"run_date={run_date.isoformat()}."
        )
    return _CheckOutcome("eod_reconcile_pnl", content_date.isoformat(), f"<= {max_stale_trading_days} trading day(s) of {run_date.isoformat()}")


# Registry of every hard-gated input, in the order the issue enumerates them.
# Each entry maps 1:1 to an ARTIFACT_REGISTRY.yaml row (named in the comment)
# so the cadence + owner are traceable back to the SoT registry. Artifacts
# the tiles already treat as legitimately-optional/known-unwired (veto_value,
# scanner_opt, cio_opt, sizing_ab, predictor_sizing — see grading/artifacts.py
# module docstring) are deliberately NOT hard-gated here: promoting them would
# turn a documented "not yet persisted" state into a false hard-fail.
_CHECKS = (
    ("metrics_json", _check_metrics_json),               # backtest_metrics
    ("e2e_lift_json", _check_e2e_lift),                  # research_producer_leaderboard / research-free counterfactual
    ("predictor_meta_weights_manifest", _check_predictor_manifest),  # predictor_meta_weights_manifest
    ("research_signals", _check_signals),                # research_signals
    ("eod_reconcile_pnl", _check_eod_pnl),                # eod_reconcile_pnl
)


def assert_input_freshness(bucket: str, run_date: str, s3_client=None) -> dict:
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
    """
    s3 = s3_client or boto3.client("s3")
    try:
        run_d = _date.fromisoformat(run_date)
    except (ValueError, TypeError) as exc:
        raise MissingInputArtifactError(
            f"freshness_preflight: run_date={run_date!r} is not a valid ISO date — "
            "cannot resolve the freshness window."
        ) from exc

    checked: list[dict] = []
    for name, fn in _CHECKS:
        outcome = fn(s3, bucket, run_d)
        checked.append({
            "artifact_id": outcome.artifact_id,
            "content_date": outcome.content_date,
            "window": outcome.window,
        })
        logger.info(
            "freshness_preflight: %s OK (content_date=%s, window=%s)",
            outcome.artifact_id, outcome.content_date, outcome.window,
        )
    return {"run_date": run_date, "checks": checked}
