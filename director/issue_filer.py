"""
issue_filer.py — Phase H (repointed): file the weekly ``DirectorWeeklyActionPlan``
action items as GitHub **issues** on ``alpha-engine-config``, replacing the prior
ROADMAP.md markdown-append PR channel.

Why: the Alpha Engine backlog migrated from ``ROADMAP.md`` to GitHub Issues on
2026-06-11 (L4610); ``ROADMAP.md`` is now a CI-guarded tombstone. So the
Director's weekly proposals belong in the issue tracker, not as markdown appends
(config#978).

The Director PROPOSES; Brian DISPOSES. One issue per ``ActionItem``, labeled
``area:director-proposals`` + the Director's **suggested ``P#``** (P0–P3) as the
live priority label — so the proposal lands directly in the prioritized backlog
views and is queryable by priority (``gh issue list --label P1``). Brian
re-prioritizes / routes / closes at triage. Advisory only: no live trading config
is touched, and the Director never closes, re-prioritizes, or re-orders its own
issues.

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

DEFAULT_REPO = "nousergon/alpha-engine-config"
PROPOSAL_LABEL = "area:director-proposals"


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
    body restates the Director's assigned priority (also applied as the live
    ``P#`` label) alongside the full backlog discipline."""
    title = f"[director] {item.title.rstrip().rstrip('.')} (id={item.id})"
    evidence = ", ".join(item.evidence) if item.evidence else "see report card"
    change = item.suggested_change_type.replace("_", " ")
    body = (
        "_Auto-filed by the weekly Director (Layer C, Phase H) — advisory. The "
        f"``{item.priority}`` label is the Director's SUGGESTED priority; Brian "
        "re-prioritizes / routes / closes at triage. The Director never closes or "
        "re-prioritizes its own proposals, and writes no live trading config._\n\n"
        f"**Director-assigned priority:** {item.priority}  ·  **Owner:** {item.proposed_owner}  ·  "
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
    """File each NEW action item as an issue, idempotent by slug, labeled
    ``area:director-proposals`` + the item's suggested ``P#``.

    If nothing new remains, files nothing and returns ``{"status": "nochange"}``.
    A ``401``/``403`` on the first POST is the EXPECTED pre-activation state (the
    PAT is not yet scoped ``issues:write`` — gates go-live, config#978): it is
    WARN-logged + recorded as ``{"status": "skipped_unscoped"}`` and filing stops,
    rather than raising — so the channel can ship default-on and self-activate the
    moment the token is re-scoped, with NO Lambda redeploy. Any OTHER non-201
    (422/5xx) still raises (the handler's best-effort wrapper records it).
    ``gh_request`` is injected for tests. Returns a compact summary
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
            {"title": title, "body": body, "labels": [PROPOSAL_LABEL, item.priority]},
        )
        if status in (401, 403):
            # (a) Failure mode: DIRECTOR_GITHUB_TOKEN PAT not yet scoped issues:write
            # (re-scoped at the nousergon org migration, config#978). (c) Recorded
            # surface: this WARN + the returned "skipped_unscoped" status, which the
            # handler folds into its summary — NOT a silent swallow
            # ([[feedback_no_silent_fails]]): a secondary advisory channel whose
            # failure is surfaced, while the plan (primary deliverable) is already
            # persisted. Self-activates on token re-scope, no redeploy.
            logger.warning(
                "issue_filer: HTTP %s on POST /issues for %r — DIRECTOR_GITHUB_TOKEN "
                "lacks issues:write (config#978; re-scope at org migration). Skipping "
                "filing; channel will self-activate once the PAT is re-scoped.",
                status, item.id,
            )
            return {
                "status": "skipped_unscoped",
                "n_filed": len(filed),
                "reason": f"PAT lacks issues:write (HTTP {status})",
                "issues": filed,
                "slugs": [f["slug"] for f in filed],
            }
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


def recently_closed_proposals_digest(
    repo: str, token: str, *, run_date: str, weeks: int = 8,
    gh_request=_gh_request, max_chars: int = 6000, max_pages: int = 10,
) -> str:
    """Condense director-proposal issues CLOSED within ``weeks`` of ``run_date``
    into a digest so the director won't re-litigate a just-resolved concern.

    The open-backlog digest only shows OPEN issues, and the file-time slug-skip
    only catches the EXACT slug — so a recently-investigated concern that's been
    closed can return under a semantically-similar NEW slug, evading both. This
    feeds those recently-closed proposals to the LLM as "already examined — do
    not re-propose" (config#1164 follow-up). ``run_date`` is ``YYYY-MM-DD``; an
    unparseable date or a fetch failure yields an empty digest (best-effort —
    the caller treats absence as "no extra dedup context"). Pages are fetched
    newest-closed-first and the loop stops once a page predates the cutoff."""
    from datetime import date, timedelta

    try:
        y, m, d = (int(x) for x in run_date.split("-"))
        cutoff = (date(y, m, d) - timedelta(weeks=weeks)).isoformat()
    except Exception:  # noqa: BLE001 — bad run_date → no dedup context, not fatal
        return ""
    api = f"https://api.github.com/repos/{repo}"
    lines: list[str] = []
    for page in range(1, max_pages + 1):
        # sort=updated&direction=desc: a closed issue is rarely touched after
        # closing, so updated_at is a safe lower bound on closed_at for paging
        # termination; we still filter PRECISELY on closed_at below.
        url = (f"{api}/issues?state=closed&labels=area:director-proposals"
               f"&sort=updated&direction=desc&per_page=100&page={page}")
        status, items = gh_request("GET", url, token)
        if status != 200 or not isinstance(items, list) or not items:
            break
        page_has_recent = False
        for it in items:
            if "pull_request" in it:
                continue
            closed_at = (it.get("closed_at") or "")[:10]
            if closed_at and closed_at >= cutoff:
                page_has_recent = True
                lines.append(
                    f"#{it.get('number')} (closed {closed_at}) {it.get('title', '')}"[:300])
        # Once a full page has nothing within the window, older pages won't either.
        if len(items) < 100 or not page_has_recent:
            break
    return "\n".join(lines)[:max_chars]
