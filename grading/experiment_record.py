"""experiment_record.py — the reference/champion experiment's per-run index
(``experiment_record.v1``, alpha-engine-config#3077 Phase C).

The results renderer consumes THIS record, never a directory listing (per
``nousergon_lib.contracts.experiment_record.schema.json``'s own docstring).
It binds WHAT ran (a synthesized manifest hash + per-slot fingerprints) to
WHAT the run emitted (the report card + its constituent artifact reads, as a
typed link table with honest ``status: absent`` + ``reason`` for anything
missing — never a silent omission).

Design gap this module bridges (alpha-engine-config#3077 delegates this
judgment call to the implementer): crucible-evaluator's reference/champion
experiment has no declarative ``experiment.v1`` manifest today — it is the
implicit, always-on system report card, not a manifest-declared arm like a
crucible-research challenger. There is also no existing cross-repo git-SHA
map gathered anywhere in this codebase (confirmed: no SCANNER_VERSION /
PREDICTOR_VERSION env var, no VERSION file — the ONLY git SHA this repo ever
resolves is its OWN baked image stamp, via ``grading.deploy_drift.
_read_baked_git_sha``, used solely for deploy-drift detection). Rather than
inventing fake upstream SHAs, this module synthesizes:

  - ``slots``: one entry per product slot (research / model / strategy) with
    an ``artifact@<sha256-of-content-dates>`` fingerprint derived from the
    freshness-preflight's own per-artifact ``content_date`` resolution — the
    one piece of "what version of the world did this run actually see"
    information this repo genuinely possesses for each slot's upstream
    inputs (backtest research/predictor/executor artifacts). This is
    honest and reproducible (same content dates -> same hash) rather than a
    placeholder.
  - ``manifest.hash``: sha256 over the synthesized, canonicalized slot list
    (deterministic — the Compare view's same-config equality check).
  - ``git``: this repo's OWN resolved SHA (baked image stamp when running in
    Lambda, else local ``git rev-parse HEAD`` — the CLI/test fallback) under
    the ``crucible-evaluator`` key. This is the one real, first-party git SHA
    available; it is NOT folded into ``slots`` (those are input-provenance
    fingerprints, not "which grader ran").

See the PR body for the full reasoning; this is documented here too since
it is a durable design decision future readers of this file need.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path

import boto3

from grading.deploy_drift import _read_baked_git_sha

logger = logging.getLogger(__name__)

EXPERIMENT_RECORD_PREFIX = "experiments"
EXPERIMENT_RECORD_FILENAME_TMPL = "{run_date}.json"
LATEST_EXPERIMENT_RECORD_FILENAME = "latest.json"

# The reference/champion experiment id (alpha-engine-config#3077). Mirrors the
# fleet-wide ``ALPHA_ENGINE_EXPERIMENT_ID`` convention (default "reference",
# crucible-research/infrastructure/spot_research_weekly.sh) rather than the
# research-producer arm name ``scanner_predictor_direct`` — that string names
# a specific (currently NOT live; the live champion pointer is still
# "agentic") research-producer arm, not "the reference experiment" the
# evaluator's report card represents as a whole. The evaluator grades the
# WHOLE system's weekly outcome, independent of which research arm is
# currently promoted to champion.
REFERENCE_EXPERIMENT_ID = "reference"

_EVALUATOR_REPO_KEY = "crucible-evaluator"

# The artifact_id keys (freshness_preflight._CHECKS's per-check ids, e.g.
# "research_signals", "predictor_meta_weights_manifest", "metrics_json",
# "e2e_lift_json", "eod_reconcile_pnl") that feed each product slot, for the
# synthesized fingerprint. Best-effort substring membership test — an
# artifact_id not matched here simply doesn't contribute to any slot's
# fingerprint (never an error; new artifact_ids added to _CHECKS later just
# don't count toward a slot's fingerprint until this map is updated).
_SLOT_ARTIFACT_PREFIXES: dict[str, tuple[str, ...]] = {
    "research": ("research", "signals", "e2e_lift", "scanner", "attractiveness"),
    "model": ("predictor",),
    "strategy": ("metrics_json", "eod_reconcile", "trades", "executor", "backtester"),
}


def _local_git_sha() -> str | None:
    """Best-effort local ``git rev-parse HEAD`` (CLI/test fallback when no
    baked image stamp exists). Never raises — returns ``None`` on any
    failure (no git binary, not a repo, detached weirdness, etc.)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        sha = out.stdout.strip()
        return sha or None
    except Exception:  # noqa: BLE001 — best-effort provenance, never fatal
        return None


def _resolve_evaluator_git_sha() -> str | None:
    """This repo's own resolved SHA: baked image stamp first (Lambda), local
    ``git rev-parse HEAD`` fallback (CLI/tests/local invoke)."""
    return _read_baked_git_sha() or _local_git_sha()


def _slot_fingerprint(slot: str, checks: list[dict]) -> tuple[str, str]:
    """Return ``(impl, fingerprint)`` for one product slot, derived from the
    freshness-preflight's per-artifact ``content_date`` resolution.

    ``artifact@<sha256>`` over the sorted ``{artifact_id: content_date}`` map
    for every check whose ``artifact_id`` matches this slot's prefixes —
    deterministic (same inputs, same run) and reproducible from data this
    repo genuinely has in hand at build time, per artifact.
    """
    prefixes = _SLOT_ARTIFACT_PREFIXES[slot]
    matched = {
        c["artifact_id"]: c.get("content_date")
        for c in checks
        if any(p in str(c.get("artifact_id", "")) for p in prefixes)
    }
    canonical = json.dumps(matched, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    return "artifact", f"artifact@{digest}"


def _build_slots(freshness_provenance: dict) -> list[dict]:
    checks = freshness_provenance.get("checks") or []
    slots = []
    for slot in ("research", "model", "strategy"):
        impl, fingerprint = _slot_fingerprint(slot, checks)
        slots.append({"slot": slot, "impl": impl, "fingerprint": fingerprint})
    return slots


def _manifest_hash(slots: list[dict]) -> str:
    canonical = json.dumps(slots, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _artifact_link(
    name: str,
    *,
    key: str | None = None,
    reason: str | None = None,
    contract: str | None = None,
    contract_version: int | None = None,
) -> dict:
    if key is not None:
        entry = {"name": name, "status": "emitted", "key": key}
    else:
        entry = {
            "name": name,
            "status": "absent",
            "reason": reason or "not produced this cycle",
        }
    if contract is not None:
        entry["contract"] = contract
    if contract_version is not None:
        entry["contract_version"] = contract_version
    return entry


def build_experiment_record(
    bucket: str,
    run_date: str,
    card: dict,
    *,
    report_card_key: str | None,
    experiment_id: str = REFERENCE_EXPERIMENT_ID,
) -> dict:
    """Build the ``experiment_record.v1`` payload for one reference-experiment
    weekly run.

    ``card`` is the ALREADY-BUILT report card (``build_report_card``'s
    return value) — this function does no S3 reads of its own; it derives
    the record entirely from ``card["_provenance"]`` (the freshness-preflight
    checks + the artifact read/missing lists already assembled for this
    exact run) and the caller-supplied ``report_card_key`` (``None`` when the
    report card itself was not persisted this cycle, e.g. a dry run).

    Partial runs: every artifact the run's own provenance lists (read OR
    missing) becomes a link-table row — ``status="emitted"`` (read
    successfully / report card written) or ``status="absent"`` (missing,
    carrying the freshness-preflight's own reason where available). A
    missing artifact is NEVER an omission from the table.
    """
    provenance = card.get("_provenance", {})
    freshness_provenance = provenance.get("freshness_preflight", {})
    artifacts_report = provenance.get("artifacts", {})

    slots = _build_slots(freshness_provenance)
    manifest_hash = _manifest_hash(slots)

    artifacts: list[dict] = []

    # The report card itself — the one artifact THIS run directly emits.
    if report_card_key:
        artifacts.append(_artifact_link(
            "report_card", key=report_card_key, contract="report_card",
        ))
    else:
        artifacts.append(_artifact_link(
            "report_card", reason="report card not persisted this cycle (dry run or write=False)",
        ))

    # Every upstream input artifact the freshness preflight + artifact reader
    # already accounted for this run — read (emitted, by definition already
    # in S3 under the backtest/{run_date}/ prefix) or missing (absent).
    read_prefix = f"backtest/{run_date}"
    for artifact_name in artifacts_report.get("artifacts_read", []):
        artifacts.append(_artifact_link(
            artifact_name, key=f"{read_prefix}/{artifact_name}",
        ))
    for artifact_name in artifacts_report.get("artifacts_missing", []):
        artifacts.append(_artifact_link(
            artifact_name,
            reason=f"absent under s3://{bucket}/{read_prefix}/ at build time",
        ))

    n_missing = artifacts_report.get("n_missing", 0)
    if not report_card_key:
        status = "failed"
    elif n_missing:
        status = "partial"
    else:
        status = "complete"

    evaluator_sha = _resolve_evaluator_git_sha()
    git_map: dict[str, str] = {}
    if evaluator_sha:
        git_map[_EVALUATOR_REPO_KEY] = evaluator_sha

    record = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "run_date": run_date,
        "status": status,
        "manifest": {"hash": manifest_hash},
        "slots": slots,
        "artifacts": artifacts,
    }
    if git_map:
        record["git"] = git_map
    return record


def experiment_record_key(run_date: str, experiment_id: str = REFERENCE_EXPERIMENT_ID) -> str:
    return f"{EXPERIMENT_RECORD_PREFIX}/{experiment_id}/records/{run_date}.json"


def latest_experiment_record_key(experiment_id: str = REFERENCE_EXPERIMENT_ID) -> str:
    return f"{EXPERIMENT_RECORD_PREFIX}/{experiment_id}/records/{LATEST_EXPERIMENT_RECORD_FILENAME}"


def write_experiment_record(
    bucket: str,
    run_date: str,
    record: dict,
    s3_client=None,
    *,
    experiment_id: str = REFERENCE_EXPERIMENT_ID,
) -> dict:
    """Persist the experiment record to both the dated key and the standing
    ``latest.json`` pointer (mirrors ``write_report_card``'s latest+dated
    convention). Returns ``{"latest_key": str, "dated_key": str}``."""
    s3 = s3_client or boto3.client("s3")
    body = json.dumps(record, indent=2, default=str).encode("utf-8")

    dated_key = experiment_record_key(run_date, experiment_id=experiment_id)
    s3.put_object(Bucket=bucket, Key=dated_key, Body=body, ContentType="application/json")
    logger.info("Wrote experiment record to s3://%s/%s", bucket, dated_key)

    latest_key = latest_experiment_record_key(experiment_id=experiment_id)
    s3.put_object(Bucket=bucket, Key=latest_key, Body=body, ContentType="application/json")
    logger.info("Wrote experiment record to s3://%s/%s (latest)", bucket, latest_key)

    return {"latest_key": latest_key, "dated_key": dated_key}
