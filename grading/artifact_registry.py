"""artifact_registry.py — the evaluator's read path onto the fleet's declared
artifact registry (`alpha-engine-config-I9731`).

**What this replaces.** ``grading/freshness_preflight.py`` used to carry a
hardcoded Python table of the artifacts it hard-gates: their S3 key templates,
and — in prose comments — the cadence each one is declared at. Its own comment
said what it was: *"Each entry maps 1:1 to an ARTIFACT_REGISTRY.yaml row (named
in the comment)."* A hand-maintained mirror of a declared registry is a second
source of truth, and it had already drifted — the ``e2e_lift.json`` entry named
``research_producer_leaderboard`` as its registry row when the actual row is
``backtest_e2e_lift``, and the predictor manifest was graded on a weekly
staleness bar that its registry row stopped declaring on 2026-08-28
(``cadence: event_driven``, `alpha-engine-config-I9018`).

**Where the registry comes from, and why.** ``ARTIFACT_REGISTRY.yaml`` lives in
the PRIVATE ``alpha-engine-config`` repo; this repo is PUBLIC and runs as a
Lambda. The registry reaches this runtime as the **published S3 mirror**
``s3://alpha-engine-research/_freshness_monitor/ARTIFACT_REGISTRY.yaml``, which
``alpha-engine-config``'s ``sync-artifact-registry.yml`` refreshes on every
merge touching the registry. That is:

  - **Tier-correct.** No registry CONTENT enters this public repo — only
    ``artifact_id`` strings, which are names of the fleet's own artifacts and
    already appear throughout the public repos. The private document stays
    private; the public code carries an access path, not a copy.
  - **Already granted.** ``alpha-engine-evaluator-role`` already holds
    ``s3:GetObject`` on ``_freshness_monitor/*`` plus the matching ``ListBucket``
    prefix condition (`alpha-engine-config-I8156`, verified against the LIVE
    role 2026-09-01), and ``grading/iam_s3_contract.json`` already declares
    ``_freshness_monitor: read``. No IAM change is required by this module.
  - **Not a build-time freeze.** A bundled copy baked into the image would
    reintroduce exactly the drift being removed: the image outlives the merge
    that changed the registry. The mirror is re-read on every report-card
    build, so the drift window is one merge, not one deploy.
  - **Not a library payload.** Publishing the registry's content inside
    ``nousergon-lib`` (a public, PyPI-installed package) would put a private
    document on PyPI. ``nousergon-lib`` supplies the row SHAPE
    (:class:`~nousergon_lib.artifact_freshness.ArtifactSpec`) and ``krepis``
    supplies the loader; neither carries the content.

**Loader reuse, not a fourth parser.** ``krepis.stage_coverage`` already reads
this exact mirror (``load_registry`` / ``index_artifacts``) for the stage-coverage
verdicts this repo's handler already writes, and its ``S3_SURFACE`` declaration
is what put ``_freshness_monitor`` into this repo's IAM contract in the first
place. Adding a second YAML reader here would be the shared-code policy's
second-adoption trigger fired in the wrong direction, so this module composes
the two libraries that already own the halves: krepis loads and indexes the
document, ``nousergon_lib.artifact_freshness.ArtifactSpec`` validates and types
each row.

**Fail loud.** Every failure here raises. There is no fallback table, no
cached-last-good, and no "grade against what we can read" path: a freshness
preflight whose predicates could not be loaded and which reports clean is
strictly worse than one that fails, because it publishes a Report Card that
looks graded and gated nothing. A fallback IS the drift this module exists to
remove.
"""

from __future__ import annotations

import logging
from datetime import date as _date
from typing import Any

from krepis.stage_coverage import DEFAULT_BUCKET as REGISTRY_BUCKET
from krepis.stage_coverage import REGISTRY_KEY, index_artifacts, load_registry
from nousergon_lib.artifact_freshness import ArtifactSpec

logger = logging.getLogger(__name__)

__all__ = [
    "REGISTRY_BUCKET",
    "REGISTRY_KEY",
    "RegistryUnavailableError",
    "RegistryRowMissingError",
    "load_specs",
    "spec_from_row",
]


class RegistryUnavailableError(RuntimeError):
    """The declared artifact registry could not be loaded, parsed, or typed.

    Deliberately NOT a subclass of the preflight's
    ``InputArtifactError`` family: "an input artifact is missing/stale" is a
    finding ABOUT the fleet that the dry rehearsal path is allowed to record
    as UNMEASURED, while "I could not read the predicates I grade against" is
    a defect in the grader itself. Collapsing the two would let a rehearsal
    that loaded nothing render identically to one that checked everything.
    """


class RegistryRowMissingError(RegistryUnavailableError):
    """A gated ``artifact_id`` has no row in the registry.

    Either the row was renamed/dropped upstream, or this repo gates an
    artifact nobody declared. Both are findings and both raise — a gate whose
    subject is undeclared cannot be graded, and skipping it silently is how a
    dropped row becomes an ungated input.
    """


def spec_from_row(row: dict[str, Any]) -> ArtifactSpec:
    """Type one raw registry row as an :class:`ArtifactSpec`.

    ``ArtifactSpec.__post_init__`` is the validator — an unknown cadence, a
    bad severity, a negative SLA or a malformed lineage edge raises there. Any
    such ``ValueError`` is re-raised as :class:`RegistryUnavailableError` so a
    caller has one exception family to reason about, with the offending
    ``artifact_id`` named.
    """
    artifact_id = row.get("artifact_id")
    try:
        created_at = row["created_at"]
        if isinstance(created_at, str):
            created_at = _date.fromisoformat(created_at)
        return ArtifactSpec(
            artifact_id=row["artifact_id"],
            s3_bucket=row.get("s3_bucket", REGISTRY_BUCKET),
            s3_key_template=row["s3_key_template"],
            cadence=row["cadence"],
            sla_minutes_after_cron=int(row["sla_minutes_after_cron"]),
            severity=row["severity"],
            owner_repo=row["owner_repo"],
            created_at=created_at,
            grace_period_cycles=int(row.get("grace_period_cycles", 2)),
            recovery_key_template=row.get("recovery_key_template"),
            calendar_aware=bool(row.get("calendar_aware", True)),
            interval_minutes=row.get("interval_minutes"),
            run_calendar=row.get("run_calendar"),
            active_hours_utc=row.get("active_hours_utc"),
            produces=tuple(row.get("produces") or ()),
            depends_on=tuple(row.get("depends_on") or ()),
            liveness_via=row.get("liveness_via"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RegistryUnavailableError(
            f"ARTIFACT_REGISTRY row {artifact_id!r} could not be typed as an "
            f"ArtifactSpec: {type(exc).__name__}: {exc}"
        ) from exc


def load_specs(
    artifact_ids: "tuple[str, ...] | list[str]",
    *,
    s3_client: Any,
    bucket: str = REGISTRY_BUCKET,
    key: str = REGISTRY_KEY,
) -> dict[str, ArtifactSpec]:
    """Return ``{artifact_id: ArtifactSpec}`` for every requested id.

    Raises :class:`RegistryUnavailableError` when the mirror cannot be read or
    parsed, and :class:`RegistryRowMissingError` naming EVERY absent id (all of
    them, not the first — a renamed row and a dropped row look identical one at
    a time, and an operator needs the whole set to tell a registry edit from a
    registry outage).

    Not cached. One ``GetObject`` per report-card build is the whole cost, and
    a process-level cache in a warm Lambda would serve a registry the merge
    that changed it has already superseded — the drift this module removes,
    reintroduced at a shorter timescale.
    """
    try:
        registry = load_registry(s3_client, bucket=bucket, key=key)
    except Exception as exc:  # noqa: BLE001 — re-raised, never swallowed
        raise RegistryUnavailableError(
            f"could not load the declared artifact registry from "
            f"s3://{bucket}/{key}: {type(exc).__name__}: {exc}. The freshness "
            "preflight grades against this document's rows and has no fallback "
            "table by design — a preflight that cannot load its predicates and "
            "reports clean is worse than one that fails."
        ) from exc

    rows = index_artifacts(registry)
    if not rows:
        raise RegistryUnavailableError(
            f"the declared artifact registry at s3://{bucket}/{key} parsed to "
            "ZERO artifact rows — an empty registry is a sync failure, never a "
            "fleet with nothing to gate."
        )

    missing = [aid for aid in artifact_ids if aid not in rows]
    if missing:
        raise RegistryRowMissingError(
            f"ARTIFACT_REGISTRY (s3://{bucket}/{key}) has no row for "
            f"{sorted(missing)} — the freshness preflight gates {len(artifact_ids)} "
            f"artifact(s) and {len(missing)} of them are undeclared. Either the "
            "row was renamed/dropped upstream (fix the id here) or this gate "
            f"covers an undeclared artifact (add the row). Registry holds "
            f"{len(rows)} rows."
        )

    specs = {aid: spec_from_row(rows[aid]) for aid in artifact_ids}
    logger.info(
        "artifact_registry: resolved %d gated spec(s) from s3://%s/%s (%d rows)",
        len(specs), bucket, key, len(rows),
    )
    return specs
