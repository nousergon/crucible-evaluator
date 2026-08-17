"""
artifacts.py — read the raw per-module analysis artifacts the producers persist
to S3 and assemble the keyword inputs for ``scorecard.compute_scorecard``.

This is the seam that makes Option B work (``director-implementation-plan-260604.md``
§2.4): the backtester / predictor run the analyses where the data lives and
persist their raw dicts to ``s3://{bucket}/backtest/{date}/<name>.json``; the
evaluator reads them here and grades natively. No analysis logic lives here —
only the artifact→input mapping and a fail-loud reader.

Fail-loud posture (``[[feedback_no_silent_fails]]``):
  - A *missing* artifact (``NoSuchKey``) is a legitimate state — the producer
    diagnostic legitimately found no data, or hasn't been wired to persist yet.
    We record it in ``ArtifactReport.missing`` + WARN, and pass ``None`` to the
    grader (which renders that component N/A). Absence is recorded, never
    swallowed.
  - Any *other* S3 error (auth, throttling, network, bad bucket) is an upstream
    contract violation and is RAISED — we do not grade on a partial read we
    can't explain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import boto3

# SSoT: artifact resolution lives in nousergon_lib (config#1190).
# These re-exports ensure the evaluator's consumer tiles resolve the SSoT
# symbol, never a local fork — the identity contract test in
# tests/test_artifacts.py asserts this.
from nousergon_lib.artifact_resolution import (
    DEFAULT_ARTIFACT_MAX_AGE_DAYS,
    get_json,
    get_json_windowed,
)

# Back-compat alias for grading.tiles.substrate which historically
# imported the private ``_get_json`` name.
_get_json = get_json

# ---------------------------------------------------------------------------
# Per-tile-family staleness thresholds (config#2885 — tile hardening).
# ---------------------------------------------------------------------------
# Each tile family declares the max acceptable age (in days) of its input
# artifacts before routing dependent components to N/A with a staleness
# reason. "Worker" families (backtester / research) tolerate the full
# 10-day weekly-recovery window; "daily" families (executor) demand
# fresher data. These are evaluated per-call-site against the ``age_days``
# returned by ``get_json_windowed`` — the resolved age of the ACTUAL
# instance the tile will grade, not the requested run_date.

BACKTESTER_ARTIFACT_MAX_AGE_DAYS: int = 10      # weekly cadence, 10d recovery window
EXECUTOR_ARTIFACT_MAX_AGE_DAYS: int = 3          # daily artifacts, 3d tolerance
PREDICTOR_ARTIFACT_MAX_AGE_DAYS: int = 7         # weekly-ish, 7d tolerance
RESEARCH_ARTIFACT_MAX_AGE_DAYS: int = 7          # weekly, 7d tolerance
AGENT_ARTIFACT_MAX_AGE_DAYS: int = 10            # weekly agent producer
BEHAVIORAL_ARTIFACT_MAX_AGE_DAYS: int = 10       # weekly behavioral producer
SUBSTRATE_ARTIFACT_MAX_AGE_DAYS: int = 10        # weekly substrate producer


def artifact_is_stale(age_days: int | None, max_age_days: int) -> bool:
    """True when the artifact's resolved age exceeds the per-family max.

    ``age_days`` comes from ``get_json_windowed``'s third return value —
    the age of the actual instance the tile will grade. ``None`` means the
    artifact resolved to the exact-requested key (no walk-back), so it is
    never stale. A negative age (clock-skew / future-dated instance) is
    treated as not-stale (it is a mislabel, not an age breach).
    """
    if age_days is None or age_days < 0:
        return False
    return age_days > max_age_days


@dataclass
class StalenessRecord:
    """Per-artifact staleness provenance for one tile's input."""

    artifact: str          # e.g. "grading.json"
    age_days: int | None   # resolved age (None = walk-back didn't fire)
    max_age_days: int      # per-family threshold checked


class StalenessRegistry:
    """Per-tile staleness tracker — each tile builder creates one and records
    every ``get_json_windowed`` call's age so the tile dict and the report card
    can surface a top-level ``degraded_staleness`` flag.

    NOT thread-safe (single-threaded Lambda / CLI context).
    """

    def __init__(self) -> None:
        self._records: list[StalenessRecord] = []

    def record(self, artifact: str, age_days: int | None, max_age_days: int) -> None:
        self._records.append(StalenessRecord(artifact, age_days, max_age_days))

    @property
    def any_stale(self) -> bool:
        return any(
            artifact_is_stale(r.age_days, r.max_age_days) for r in self._records
        )

    @property
    def stale_count(self) -> int:
        return sum(
            1 for r in self._records
            if artifact_is_stale(r.age_days, r.max_age_days)
        )

    @property
    def max_age_days(self) -> float | None:
        ages = [r.age_days for r in self._records if r.age_days is not None]
        return max(ages) if ages else None

    def summary(self) -> dict:
        """Return a dict suitable for the tile's staleness section."""
        return {
            "stale_artifact_count": self.stale_count,
            "max_artifact_age_days": self.max_age_days,
            "any_stale": self.any_stale,
        }


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Artifact map: compute_scorecard kwarg -> backtest/{date}/<filename>
# ---------------------------------------------------------------------------
#
# Each entry is (scorecard_param_name -> s3_filename). The filenames are the
# exact keys reporter.py writes under backtest/{date}/ (verified against
# alpha-engine-backtester reporter.py @ f46e7e6). ``signal_quality`` is handled
# separately (reconstructed from metrics.json) — see _read_signal_quality.
#
# NOTE — known producer-persistence gaps as of 2026-06-04 (these inputs are
# computed in the backtester but NOT yet persisted to S3, so they read as
# missing and grade N/A until a backtester PR persists them):
#   veto_value, predictor_sizing, scanner_opt, cio_opt
# and the explicitly-deferred (RC v2 Ph2): sizing_ab.
# (action_entropy is now wired — config#1151 Batch C — and grades from its
# backtest/{date}/action_entropy.json producer.)
# The ArtifactReport surfaces exactly which were absent so the gap is loud and
# drives the follow-up persistence work, rather than silently grading partial.
ARTIFACT_MAP: dict[str, str] = {
    "e2e_lift": "e2e_lift.json",
    "macro_eval": "macro_eval.json",
    "score_calibration": "score_calibration.json",
    "veto_result": "veto_analysis.json",
    "veto_value": "veto_value.json",
    "trigger_scorecard": "trigger_scorecard.json",
    "shadow_book": "shadow_book.json",
    "exit_timing": "exit_timing.json",
    "sizing_ab": "sizing_ab.json",
    "predictor_sizing": "predictor_sizing.json",
    "portfolio_stats": "portfolio_stats.json",
    "scanner_opt": "scanner_opt.json",
    "cio_opt": "cio_opt.json",
    "team_metrics": "team_metrics.json",
    "calibration_diagnostics": "portfolio_calibration.json",
    "action_entropy": "action_entropy.json",
    "excursion_summary": "portfolio_excursion.json",
}
# NOTE (RC v3 T5, config-I7473): contribution_lift.json is deliberately NOT
# added here. ARTIFACT_MAP entries feed straight into
# ``compute_scorecard(**inputs)`` (the legacy v1 0-100 grader below) — every
# key here corresponds to a real ``compute_scorecard`` parameter that
# consumes it. contribution_lift is v2-only (a MetricRecord tile, same class
# as research/predictor/executor/behavioral's own primary artifacts, none of
# which route through this map either); adding an unconsumed key would raise
# ``TypeError: unexpected keyword argument`` the first time the artifact
# exists in S3. ``grading/tiles/contribution_lift.py`` reads
# ``backtest/{date}/contribution_lift.json`` directly via
# ``get_json_windowed``, exactly like every other v2 tile.

# Reserved top-level keys in metrics.json that are NOT part of the
# signal_quality "overall" block (so we can reconstruct overall by exclusion).
_METRICS_NON_OVERALL_KEYS = {"run_date", "status", "report_card"}


@dataclass
class ArtifactReport:
    """Provenance for one report-card build: what was read, what was absent."""

    run_date: str
    bucket: str
    prefix: str
    read: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "run_date": self.run_date,
            "bucket": self.bucket,
            "prefix": self.prefix,
            "artifacts_read": sorted(self.read),
            "artifacts_missing": sorted(self.missing),
            "n_read": len(self.read),
            "n_missing": len(self.missing),
        }




def _read_signal_quality(s3, bucket: str, prefix: str) -> dict | None:
    """Reconstruct the ``signal_quality`` input from ``metrics.json``.

    The backtester does not persist the full signal_quality dict standalone; it
    flattens the ``overall`` block to the top level of ``metrics.json`` (see
    reporter.save: ``{"run_date", "status", **overall, ["report_card"]}``). We
    recover ``{"status", "overall": {...}}`` from that — enough for the
    portfolio + composite-scoring accuracy grades. ``by_score_bucket`` is not
    persisted, so the composite high-bucket sub-grade stays N/A until a
    standalone ``signal_quality.json`` is persisted (filed follow-up).
    """
    metrics = get_json(s3, bucket, f"{prefix}/metrics.json")
    if metrics is None:
        return None
    overall = {k: v for k, v in metrics.items() if k not in _METRICS_NON_OVERALL_KEYS}
    return {"status": metrics.get("status"), "overall": overall}


def read_scorecard_inputs(
    bucket: str,
    run_date: str,
    s3_client=None,
) -> tuple[dict, ArtifactReport]:
    """Assemble the ``compute_scorecard`` kwargs from S3 artifacts.

    Returns ``(inputs, report)`` where ``inputs`` is a kwargs dict suitable for
    ``compute_scorecard(**inputs)`` (absent artifacts simply omitted → grader
    defaults them to None → N/A) and ``report`` records exactly which artifacts
    were read vs absent.
    """
    s3 = s3_client or boto3.client("s3")
    prefix = f"backtest/{run_date}"
    report = ArtifactReport(run_date=run_date, bucket=bucket, prefix=prefix)
    inputs: dict = {}

    # signal_quality is special (reconstructed from metrics.json).
    sq = _read_signal_quality(s3, bucket, prefix)
    if sq is not None:
        inputs["signal_quality"] = sq
        report.read.append("metrics.json")
    else:
        report.missing.append("metrics.json")
        logger.warning(
            "Artifact absent: s3://%s/%s/metrics.json — signal_quality / "
            "portfolio + composite-scoring tiles will grade N/A", bucket, prefix,
        )

    for param, filename in ARTIFACT_MAP.items():
        data = get_json(s3, bucket, f"{prefix}/{filename}")
        if data is not None:
            inputs[param] = data
            report.read.append(filename)
        else:
            report.missing.append(filename)
            logger.warning(
                "Artifact absent: s3://%s/%s/%s — '%s' tile will grade N/A",
                bucket, prefix, filename, param,
            )

    logger.info(
        "Assembled scorecard inputs for %s: %d read, %d absent",
        run_date, len(report.read), len(report.missing),
    )
    return inputs, report
