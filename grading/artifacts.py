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

import json
import logging
from dataclasses import dataclass, field

import boto3
from botocore.exceptions import ClientError

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
# and the explicitly-deferred (RC v2 Ph2): sizing_ab, action_entropy.
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


def _get_json(s3, bucket: str, key: str) -> dict | None:
    """Read one JSON object from S3.

    Returns the parsed dict, or ``None`` if the key does not exist
    (``NoSuchKey``). Raises on any other ClientError (real S3 problem) and on
    malformed JSON.
    """
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("NoSuchKey", "404"):
            return None
        # Auth / throttle / wrong-bucket / network — do NOT swallow.
        logger.error("S3 read failed for s3://%s/%s: %s", bucket, key, e)
        raise
    body = resp["Body"].read()
    return json.loads(body)


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
    metrics = _get_json(s3, bucket, f"{prefix}/metrics.json")
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
        data = _get_json(s3, bucket, f"{prefix}/{filename}")
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
