"""deploy_drift.py — Lambda-SHA drift probe for the evaluator's 2 Lambdas.

config#2348. Before this module, a failed post-merge deploy (CI failure, a
paths-filter miss, or a run nobody re-triggered) left ``alpha-engine-evaluator``
and/or ``alpha-engine-evaluator-director``'s ``:live`` alias pointed at a stale
image while ``main`` kept moving — surfaced only by a Telegram CI-failure
message, easy to miss, with no automated check and no scripted rollback (the
alias repoint lived in operator memory).

Ports the image-SHA drift pattern predictor already runs
(``nousergon_lib.preflight.BasePreflight.check_deploy_drift`` /
``inference/deploy_drift.py``): ``deploy.sh`` stamps the image with
``GIT_SHA`` at build time (``ARG GIT_SHA`` → ``/var/task/GIT_SHA.txt``); this
probe reads that stamp and compares it against ``origin/main`` HEAD via the
GitHub REST API (``nousergon_lib.preflight._fetch_origin_main_sha`` —
unauthenticated, evaluator is a public repo).

Unlike the predictor's version (which also reads a Step-Function-Comment
stamp + a CloudFormation stack tag — the evaluator has neither: no CFN stack,
and per the 2026-07-13 operator ruling on config#2348 this check is a NEW,
standalone pre-boot Step Function state, not wired through any existing
SF-Comment stamping mechanism), this probe is baked-file-vs-GitHub only.

Exposed as ``action=check_deploy_drift`` on BOTH Lambda handlers
(``grading.handler.handler`` and ``director.handler.handler``) so the weekly
SF (``ne-weekly-freshness-pipeline``, owned by ``nousergon-data``) can invoke
each function's ``:live`` alias directly as a Task state before any real work
runs — catching the case where ONE alias got promoted and the other didn't,
not just "the shared image is stale everywhere".

Returns a JSON-serializable dict; the SF's Choice state reads ``has_drift``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# Re-imported as a module-level attribute (not called via a fully-qualified
# path) so `patch.object(deploy_drift, "_fetch_origin_main_sha", ...)` keeps
# working in tests — mirrors crucible-predictor/inference/deploy_drift.py's
# own re-export comment.
from nousergon_lib.preflight import _fetch_origin_main_sha  # noqa: F401

log = logging.getLogger(__name__)

_EVALUATOR_REPO = "nousergon/crucible-evaluator"

# Lambda image convention (matches nousergon_lib.preflight._DEFAULT_GIT_SHA_FILE
# and crucible-predictor's Dockerfile `RUN echo "${GIT_SHA}" > /var/task/GIT_SHA.txt`).
_DEFAULT_GIT_SHA_FILE = Path("/var/task/GIT_SHA.txt")


def _read_baked_git_sha(sha_file: Path | None = None) -> str | None:
    """Return the SHA baked into the image by ``deploy.sh --build-arg GIT_SHA=…``.

    ``None`` when the stamp file is missing (legacy/local image) or holds
    ``"unknown"`` (build-arg omitted) — both are "can't prove drift", not
    "drift confirmed".
    """
    path = sha_file or _DEFAULT_GIT_SHA_FILE
    try:
        sha = path.read_text().strip()
    except FileNotFoundError:
        return None
    if not sha or sha == "unknown":
        return None
    return sha


def _shas_match(deployed: str | None, upstream: str | None) -> bool:
    """Compare a deployed SHA stamp (may be short) to the full upstream SHA.

    Missing either side → return True (can't prove drift → don't raise).
    Mirrors crucible-predictor/inference/deploy_drift.py's ``_shas_match``.
    """
    if not deployed or not upstream:
        return True
    if len(deployed) < 7:
        return True  # malformed stamp — warn elsewhere, don't block
    return upstream.startswith(deployed) or deployed.startswith(upstream)


def check_deploy_drift(
    *,
    function_name: str,
    repo: str = _EVALUATOR_REPO,
    branch: str = "main",
    sha_file: Path | None = None,
    timeout: float = 5.0,
) -> dict:
    """Compare this Lambda's baked image SHA against ``repo@branch`` HEAD.

    ``function_name`` is caller-supplied (the SF passes each Lambda's own
    name so a drift/degraded result names the offending function) — this
    module has no AWS calls of its own; it only reads the local stamp file
    baked into whichever Lambda invokes it.

    Degraded modes (``has_drift=False`` with a diagnostic ``reason``):
    stamp file missing/unknown (legacy image, or local/non-Lambda invoke),
    or GitHub unreachable. Never block on a probe-side failure — this
    mirrors the fail-open posture of every existing SF preflight gate
    (LibPinDriftCheck, PipelineContractCheck) documented in
    ``nousergon-data/infrastructure/step_function.json``.
    """
    baked = _read_baked_git_sha(sha_file)
    upstream = _fetch_origin_main_sha(repo, branch=branch, timeout=timeout)

    stamp_present = baked is not None
    upstream_available = upstream is not None
    has_drift = stamp_present and upstream_available and not _shas_match(baked, upstream)

    if not stamp_present:
        reason = "no_git_sha_stamp"
    elif not upstream_available:
        reason = "github_unreachable"
    elif has_drift:
        reason = "sha_mismatch"
    else:
        reason = "in_sync"

    result = {
        "function_name": function_name,
        "repo": repo,
        "branch": branch,
        "baked_sha": baked,
        "stamp_present": stamp_present,
        "upstream_sha": upstream,
        "has_drift": has_drift,
        "reason": reason,
    }
    log.info(
        "Deploy-drift check (%s): baked=%s upstream=%s has_drift=%s reason=%s",
        function_name,
        (baked or "missing")[:12],
        (upstream or "unavailable")[:12],
        has_drift,
        reason,
    )
    return result


def _resolve_function_name(context) -> str:
    """Best-effort Lambda function name for the result payload.

    Falls back to ``AWS_LAMBDA_FUNCTION_NAME`` (always set inside a real
    Lambda execution env) and finally ``"unknown"`` for local/test invokes.
    """
    name = getattr(context, "function_name", None)
    if name:
        return name
    return os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "unknown")
