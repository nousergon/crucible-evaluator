"""deploy_success.py — producer for the Substrate-tile ``deploy_success_rate``.

The evaluator's Lambda role can read S3 + AWS APIs but holds **no GitHub token**,
so the deploy health of the code repos (did the CI/CD *deploy* workflows ship
cleanly?) is invisible to it — ``deploy_success_rate`` has therefore graded a
transparent ``N/A-NOT-IMPL`` naming "the producer to build" (config#1153 Batch E).

This is that producer. It collects each code repo's **deploy-workflow** run
conclusions over a trailing window and writes a single rollup the substrate tile
reads:

    s3://alpha-engine-research/_substrate/deploy_success.json

It measures the **deploy** workflows specifically (``deploy*.yml`` / any workflow
whose name contains "deploy"), NOT CI/lint/CodeQL — the substrate question is
"did our *shipping* pipeline succeed", not "did every push pass tests". A run
counts when it is (a) on the default branch (``main``/``master`` — PR/dependabot
branch runs are excluded), (b) terminal (``status=completed``), and (c) a real
pass/fail conclusion (``cancelled``/``skipped``/``neutral`` are not deploy
attempts and don't enter the denominator). ``success_rate`` = success-conclusion
runs / terminal deploy runs over the window.

**Scheduling.** Invoked best-effort from ``director.handler`` (the weekly Saturday
SF Director task — the one component holding both a GH token and the research
bucket). It is written as a pure, dependency-light module so a future *standalone*
scheduled invocation (a tiny EventBridge Lambda / GH-Actions cron) can call
``run`` directly without the Director — see the PR note for that follow-on.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime, timedelta

logger = logging.getLogger(__name__)

# Where the substrate tile reads the rollup from (fixed key — latest wins).
DEPLOY_SUCCESS_KEY = "_substrate/deploy_success.json"

# Default code repos with a deploy pipeline. Override with the comma-separated
# ``DEPLOY_SUCCESS_REPOS`` env var. Repos that 404 (renamed/absent) or carry no
# deploy workflow in the window contribute nothing and are recorded per-repo —
# a stale slug degrades the sample, it never breaks the producer.
DEFAULT_REPOS: tuple[str, ...] = (
    "nousergon/crucible-predictor",
    "nousergon/crucible-executor",
    "nousergon/crucible-backtester",
    "nousergon/crucible-research",
    "nousergon/crucible-evaluator",
    "nousergon/crucible-dashboard",
    "nousergon/nousergon-data",
)

DEFAULT_WINDOW_DAYS = 28

_DEFAULT_BRANCHES = frozenset({"main", "master"})
# Only real pass/fail outcomes enter the denominator. cancelled/skipped/neutral/
# action_required/stale are not a completed deploy attempt that passed or failed.
_DEPLOY_CONCLUSIONS = frozenset({"success", "failure", "timed_out", "startup_failure"})


def _is_deploy_run(run: dict) -> bool:
    """True when a workflow run belongs to a *deploy* workflow (not CI/CodeQL)."""
    path = (run.get("path") or "").lower()
    name = (run.get("name") or "").lower()
    basename = path.rsplit("/", 1)[-1]
    return basename.startswith("deploy") or "deploy" in name


def _fetch_repo_deploy_runs(
    repo: str, token: str, since_date: str, *, gh_request, max_pages: int = 10
) -> list[dict] | None:
    """All terminal workflow runs for ``repo`` created at/after ``since_date``.

    Returns the raw run dicts (filtering to deploy/default-branch happens in the
    aggregator), or ``None`` if the repo 404s (renamed/absent → skip)."""
    api = f"https://api.github.com/repos/{repo}/actions/runs"
    runs: list[dict] = []
    for page in range(1, max_pages + 1):
        # ``created=>=DATE`` — the ``>=`` is URL-encoded.
        url = (
            f"{api}?status=completed&created=%3E%3D{since_date}"
            f"&per_page=100&page={page}"
        )
        status, payload = gh_request("GET", url, token)
        if status == 404:
            return None
        if status != 200 or not isinstance(payload, dict):
            logger.warning("deploy_success: GH %s on %s page %d — stopping this repo", status, repo, page)
            break
        batch = payload.get("workflow_runs") or []
        if not batch:
            break
        runs.extend(batch)
        if len(batch) < 100:
            break
    return runs


def build_deploy_success_doc(
    token: str,
    *,
    repos=DEFAULT_REPOS,
    window_days: int = DEFAULT_WINDOW_DAYS,
    now: datetime,
    gh_request=None,
    max_pages: int = 10,
) -> dict:
    """Collect deploy-workflow success across ``repos`` into the rollup doc.

    ``gh_request`` is injected for tests; it defaults to the Director's minimal
    GitHub REST helper (lazily imported so this module carries no hard dependency
    on ``director`` at import time)."""
    if gh_request is None:
        from director.roadmap_pr import _gh_request as gh_request  # lazy: avoid import cycle

    since_date = (now - timedelta(days=window_days)).date().isoformat()
    per_repo: dict[str, dict] = {}
    total = 0
    succ = 0
    for repo in repos:
        raw = _fetch_repo_deploy_runs(repo, token, since_date, gh_request=gh_request, max_pages=max_pages)
        if raw is None:
            per_repo[repo] = {"status": "not_found"}
            continue
        deploy = [
            r for r in raw
            if _is_deploy_run(r)
            and r.get("head_branch") in _DEFAULT_BRANCHES
            and r.get("conclusion") in _DEPLOY_CONCLUSIONS
        ]
        r_total = len(deploy)
        r_succ = sum(1 for r in deploy if r.get("conclusion") == "success")
        per_repo[repo] = {
            "success": r_succ,
            "total": r_total,
            "rate": (r_succ / r_total) if r_total else None,
        }
        total += r_total
        succ += r_succ

    return {
        "schema": "deploy_success/v1",
        "generated_utc": now.astimezone(UTC).replace(microsecond=0).isoformat(),
        "window_days": window_days,
        "since": since_date,
        "repos_measured": [r for r, v in per_repo.items() if v.get("total")],
        "per_repo": per_repo,
        "success_runs": succ,
        "total_runs": total,
        "success_rate": (succ / total) if total else None,
    }


def write_deploy_success_doc(s3, bucket: str, doc: dict, key: str = DEPLOY_SUCCESS_KEY) -> str:
    """Persist the rollup; returns the ``s3://`` URI written."""
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(doc, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    return f"s3://{bucket}/{key}"


def _resolve_repos() -> tuple[str, ...]:
    env = os.environ.get("DEPLOY_SUCCESS_REPOS")
    if env:
        return tuple(r.strip() for r in env.split(",") if r.strip())
    return DEFAULT_REPOS


def run(
    s3,
    bucket: str,
    token: str,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    now: datetime | None = None,
    repos=None,
    gh_request=None,
) -> dict:
    """Build + write the rollup. Returns a compact summary for the caller's log."""
    now = now or datetime.now(UTC)
    repos = repos or _resolve_repos()
    doc = build_deploy_success_doc(
        token, repos=repos, window_days=window_days, now=now, gh_request=gh_request
    )
    uri = write_deploy_success_doc(s3, bucket, doc)
    return {
        "status": "ok",
        "key": DEPLOY_SUCCESS_KEY,
        "uri": uri,
        "success_rate": doc["success_rate"],
        "total_runs": doc["total_runs"],
        "n_repos_measured": len(doc["repos_measured"]),
    }


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    import argparse

    import boto3

    parser = argparse.ArgumentParser(description="Build + write the deploy_success_rate rollup.")
    parser.add_argument("--bucket", default="alpha-engine-research")
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--no-write", action="store_true", help="Print the doc; do not write S3.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    token = (os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or "").strip()
    if not token:
        parser.error("set GH_TOKEN / GITHUB_TOKEN to call the GitHub API")
    doc = build_deploy_success_doc(token, window_days=args.window_days, now=datetime.now(UTC))
    if args.no_write:
        print(json.dumps(doc, indent=2))
        return 0
    uri = write_deploy_success_doc(boto3.client("s3"), args.bucket, doc)
    logger.info("wrote %s (success_rate=%s, n=%s)", uri, doc["success_rate"], doc["total_runs"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
