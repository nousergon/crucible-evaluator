"""
issue_filer.py — Phase H (repointed): file the weekly ``DirectorWeeklyActionPlan``
action items as GitHub **issues** on ``alpha-engine-config``, replacing the prior
ROADMAP.md markdown-append PR channel.

Why: the Alpha Engine backlog migrated from ``ROADMAP.md`` to GitHub Issues on
2026-06-11 (L4610); ``ROADMAP.md`` is now a CI-guarded tombstone. So the
Director's weekly proposals belong in the issue tracker, not as markdown appends
(config#978).

The Director PROPOSES; Brian DISPOSES. One issue per ``ActionItem``, labeled
``area:director-proposals`` + ``priority-unset`` — the Director's *suggested*
priority rides in the body, and a human assigns the real ``P#`` at triage.
Advisory only: no live trading config is touched, and the Director never closes,
prioritizes, or re-orders its own issues.

Design invariants (mirror the retired ``roadmap_pr`` channel):
  - **Idempotent by ``ActionItem.id``.** Each issue title carries an
    ``id=<slug>`` marker; before filing, every existing
    ``area:director-proposals`` issue (``state=all`` — open OR closed) is
    scanned for its slug, and an item whose slug already filed is skipped.
    Weekly re-runs / same-week re-fires never duplicate, and a proposal closed
    at triage is never resurrected.
  - **Reconciled against the live backlog.** The handler feeds an open-issues
    digest INTO the director call so the LLM avoids re-proposing tracked work;
    the slug-skip here is the second line of defense.
  - **Best-effort, fail-loud.** The plan is already persisted to S3; the handler
    wraps this so a token / GitHub error is WARN-logged and recorded in the
    summary, never fatal (the advisory channel must not break the run that
    produced the real artifacts).

Auth: the ``DIRECTOR_GITHUB_TOKEN`` fine-grained PAT (SSM). NOTE: the repoint
requires this token to carry **``issues:write``** on ``alpha-engine-config`` —
the prior ``contents:write`` + ``pull_requests:write`` scopes filed ROADMAP
PRs and are no longer used. Re-scoping the PAT is the operator step that gates
go-live (config#978).
"""

from __future__ import annotations

import logging

from director.roadmap_pr import _SLUG_RE, _gh_request, select_new_items
from director.schema import ActionItem, DirectorWeeklyActionPlan

logger = logging.getLogger(__name__)

DEFAULT_REPO = "cipher813/alpha-engine-config"
PROPOSAL_LABEL = "area:director-proposals"
PRIORITY_UNSET_LABEL = "priority-unset"


def existing_proposal_slugs(
    repo: str, token: str, *, gh_request=_gh_request, max_pages: int = 20
) -> set[str]:
    """Every ``id=<slug>`` already filed as a director-proposal issue (open OR
    closed) — the idempotency key set.

    ``state=all`` so a proposal that was triaged-and-closed (or
    re-labeled/merged into another issue) is never re-filed. ``/issues`` also
    returns PRs; those are skipped defensively."""
    slugs: set[str] = set()
    api = f"https://api.github.com/repos/{repo}"
    for page in range(1, max_pages + 1):
        url = (
            f"{api}/issues?labels={PROPOSAL_LABEL}&state=all"
            f"&per_page=100&page={page}"
        )
        status, items = gh_request("GET", url, token)
        if status != 200 or not isinstance(items, list) or not items:
            break
        for it in items:
            if "pull_request" in it:  # /issues includes PRs — skip
                continue
            text = f"{it.get('title', '')}\n{it.get('body', '') or ''}"
            slugs.update(_SLUG_RE.findall(text))
        if len(items) < 100:
            break
    return slugs


def render_issue(item: ActionItem, run_date: str) -> tuple[str, str]:
    """``(title, body)`` for one action item.

    The title carries the ``id=<slug>`` marker for title-search idempotency; the
    body carries the full backlog discipline + the Director's SUGGESTED priority
    (the live label stays ``priority-unset`` until a human triages)."""
    title = f"[director] {item.title.rstrip().rstrip('.')} (id={item.id})"
    evidence = ", ".join(item.evidence) if item.evidence else "see report card"
    change = item.suggested_change_type.replace("_", " ")
    body = (
        "_Auto-filed by the weekly Director (Layer C, Phase H) — advisory. "
        "Brian triages: assign the real priority + route to an owner. The "
        "Director never closes or prioritizes its own proposals, and writes no "
        "live trading config._\n\n"
        f"**Suggested priority:** {item.priority}  ·  **Owner:** {item.proposed_owner}  ·  "
        f"**Horizon:** {item.horizon}  ·  **Confidence:** {item.confidence}/100  ·  "
        f"**Change type:** {change}\n\n"
        f"## Rationale\n{item.rationale.rstrip()}\n\n"
        f"## Evidence\n{evidence}\n\n"
        "## Closes when\n"
        f"The cited evidence ({evidence}) clears its target/red-line, or the "
        f"proposed {change} ships and is verified.\n\n"
        f"<sub>run_date={run_date} · id={item.id} · idempotency marker — do not edit the id</sub>"
    )
    return title, body


def file_director_issues(
    plan: DirectorWeeklyActionPlan,
    run_date: str,
    *,
    token: str,
    repo: str = DEFAULT_REPO,
    gh_request=_gh_request,
) -> dict:
    """File each NEW action item as an issue, idempotent by slug.

    If nothing new remains, files nothing and returns ``{"status": "nochange"}``.
    A per-issue POST failure raises (the handler's best-effort wrapper records
    it). ``gh_request`` is injected for tests. Returns a compact summary
    (status / n_filed / issues / slugs)."""
    api = f"https://api.github.com/repos/{repo}"
    already = existing_proposal_slugs(repo, token, gh_request=gh_request)
    new_items = select_new_items(plan, already)
    if not new_items:
        logger.info(
            "issue_filer: no new items (all %d already filed) — nothing filed.",
            len(plan.action_items),
        )
        return {"status": "nochange", "n_filed": 0, "reason": "all items already filed as issues"}

    filed: list[dict] = []
    for item in new_items:
        title, body = render_issue(item, run_date)
        status, res = gh_request(
            "POST", f"{api}/issues", token,
            {"title": title, "body": body, "labels": [PROPOSAL_LABEL, PRIORITY_UNSET_LABEL]},
        )
        if status != 201:
            raise RuntimeError(f"issue_filer: create issue for {item.id!r} -> {status}: {res}")
        filed.append({"slug": item.id, "number": res.get("number"), "url": res.get("html_url")})

    summary = {
        "status": "ok",
        "n_filed": len(filed),
        "issues": filed,
        "slugs": [f["slug"] for f in filed],
    }
    logger.info("issue_filer: %s", summary)
    return summary


def open_issues_digest(
    repo: str, token: str, *, gh_request=_gh_request, max_chars: int = 12000, max_pages: int = 10
) -> str:
    """Condense the OPEN backlog (issue number + labels + title) into a digest
    for the director prompt so it doesn't re-propose tracked work.

    Replaces the ROADMAP digest now that the backlog lives in issues. PRs are
    skipped. Best-effort: the handler tolerates an empty/failed digest (the
    slug-skip at file time is the second line of defense)."""
    api = f"https://api.github.com/repos/{repo}"
    lines: list[str] = []
    for page in range(1, max_pages + 1):
        url = f"{api}/issues?state=open&per_page=100&page={page}"
        status, items = gh_request("GET", url, token)
        if status != 200 or not isinstance(items, list) or not items:
            break
        for it in items:
            if "pull_request" in it:
                continue
            labels = ",".join(l.get("name", "") for l in it.get("labels", []) or [])
            lines.append(f"#{it.get('number')} [{labels}] {it.get('title', '')}"[:300])
        if len(items) < 100:
            break
    return "\n".join(lines)[:max_chars]
