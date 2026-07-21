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
  - **Deduped against prior CLOSED investigations by evidence, not just slug**
    (config#2307). The exact-slug skip above only catches a re-fire under the
    IDENTICAL slug; the LLM picks slugs freely (schema.py just calls it a
    "stable slug"), so a semantically-identical investigation can re-appear
    under a brand-new slug and sail past the slug check (this is exactly what
    happened re-filing config#1061/#1168 as config#1639 under a fresh id).
    Relying on the LLM to notice the ``recently_closed_proposals_digest`` text
    is advisory-only and was shown NOT to be enforced. So
    ``find_reconfirm_match()`` is a CODE-level check: it compares each new
    item's ``evidence`` (the MetricRecord/tile names cited) against the parsed
    ``## Evidence`` of recently-closed ``area:director-proposals`` issues
    within a lookback window. On a material overlap, the item is filed as a
    lower-confidence **re-confirm** variant (``render_reconfirm_issue``) that
    explicitly links the prior closure(s) and is capped at ``P2``, rather than
    presenting as a fresh top-priority investigation.
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
import re

from director.roadmap_pr import _SLUG_RE, _gh_request, select_new_items
from director.schema import ActionItem, DirectorWeeklyActionPlan

logger = logging.getLogger(__name__)

DEFAULT_REPO = "nousergon/alpha-engine-config"
PROPOSAL_LABEL = "area:director-proposals"

# Reconfirm-dedup window (config#2307): how far back a CLOSED proposal is still
# considered "the same investigation" for evidence-overlap matching. Matches
# the existing recently_closed_proposals_digest() default so the code-enforced
# check and the advisory LLM digest agree on what counts as "recent".
RECONFIRM_LOOKBACK_WEEKS = 8

# An item's evidence is considered "the same metric pair" as a prior closed
# investigation's evidence when at least this many cited names overlap
# (case-insensitive). 1 is enough — evidence lists are short (typically 1-3
# named MetricRecord/tile references) and a single shared metric name is
# already a strong identity signal for "same investigation".
_MIN_EVIDENCE_OVERLAP = 1

_EVIDENCE_SECTION_RE = re.compile(
    r"##\s*Evidence\s*\n(.*?)(?:\n##|\n<sub>|\Z)", re.DOTALL,
)


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


def slug_issue_number_map(
    repo: str, token: str, *, gh_request=_gh_request, max_pages: int = 20
) -> dict[str, int]:
    """``{id=<slug> marker: issue number}`` for every ``area:director-proposals``
    issue (open OR closed) — the backfill source ``loop_verification.py`` uses to
    populate ``issue_number`` on ledger rows that predate that field (config#3145),
    and to self-heal any row that lost track of its issue. Mirrors
    ``existing_proposal_slugs``'s fetch/pagination exactly; kept as a separate
    function (rather than changing that one's return type) since callers of
    ``existing_proposal_slugs`` only need the slug SET."""
    out: dict[str, int] = {}
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
            if "pull_request" in it:
                continue
            text = f"{it.get('title', '')}\n{it.get('body', '') or ''}"
            for slug in _SLUG_RE.findall(text):
                out[slug] = it.get("number")
        if len(items) < 100:
            break
    return out


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
        f"proposed {change} ships and is verified. A merging PR can ADVANCE this; "
        "only the metric recovering closes it — the next weekly Director run "
        "re-checks the cited evidence against a closed-but-still-red issue and "
        "reopens it with the evidence if it hasn't (config#3145).\n\n"
        f"<sub>run_date={run_date} · id={item.id} · idempotency marker — do not edit the id</sub>"
    )
    return title, body


def render_reconfirm_issue(
    item: ActionItem, run_date: str, prior: list[dict]
) -> tuple[str, str, str]:
    """``(title, body, capped_priority)`` for a **re-confirm** variant of
    ``item`` (config#2307):
    a prior CLOSED investigation already covered this same evidence within the
    lookback window, so this files as a lower-confidence check-in that
    EXPLICITLY references the prior closure(s) instead of presenting as a
    fresh top-priority investigation.

    ``prior`` is the list of matching closed-issue dicts (each needs at least
    ``number``; ``closed_at``/``title`` used if present). Priority is capped at
    ``P2`` (never escalates a re-confirm to P0/P1) and confidence is reduced
    (floor 10) to reflect that this is a repeat read of a metric already found
    sound, not a fresh finding."""
    refs = ", ".join(f"#{p['number']}" for p in prior if p.get("number") is not None)
    title = (
        f"[director] Re-confirm: {item.title.rstrip().rstrip('.')} "
        f"(id={item.id}) [re-confirm of {refs}]"
    )
    evidence = ", ".join(item.evidence) if item.evidence else "see report card"
    change = item.suggested_change_type.replace("_", " ")
    capped_priority = item.priority if item.priority in ("P2", "P3") else "P2"
    confidence = max(10, item.confidence - 30)
    prior_lines = "\n".join(
        f"- #{p['number']}"
        + (f" (closed {p['closed_at'][:10]})" if p.get("closed_at") else "")
        + (f" — {p['title']}" if p.get("title") else "")
        for p in prior
    )
    body = (
        "_Auto-filed by the weekly Director (Layer C, Phase H) — advisory "
        "**re-confirm variant** (config#2307). A prior CLOSED investigation of "
        "this same evidence concluded sound / no action within the lookback "
        f"window; this re-files at reduced priority/confidence to note the "
        "metric still reads adverse, rather than opening a fresh top-priority "
        "investigation. The Director never closes or re-prioritizes its own "
        "proposals, and writes no live trading config._\n\n"
        f"**Director-assigned priority:** {capped_priority} (capped — re-confirm, "
        f"was {item.priority})  ·  **Owner:** {item.proposed_owner}  ·  "
        f"**Horizon:** {item.horizon}  ·  **Confidence:** {confidence}/100 "
        f"(reduced — was {item.confidence})  ·  **Change type:** {change}\n\n"
        f"## Prior investigation(s) of this evidence\n{prior_lines}\n\n"
        f"## Rationale\n{item.rationale.rstrip()}\n\n"
        f"## Evidence\n{evidence}\n\n"
        "## Closes when\n"
        f"The cited evidence ({evidence}) clears its target/red-line, or the "
        f"metric has moved materially past the prior closure's read (in which "
        f"case escalate past this re-confirm), or the proposed {change} ships "
        "and is verified.\n\n"
        f"<sub>run_date={run_date} · id={item.id} · idempotency marker — do not "
        f"edit the id · reconfirm_of={refs}</sub>"
    )
    return title, body, capped_priority


def parse_evidence_from_body(body: str) -> set[str]:
    """Recover the ``## Evidence`` names cited on a previously-filed proposal
    issue, normalized for case-insensitive comparison. Mirrors the
    ``", ".join(item.evidence)`` render in ``render_issue``/
    ``render_reconfirm_issue``. Returns an empty set if no Evidence section is
    found (e.g. an issue not filed by the Director) — best-effort, never
    raises on odd formatting."""
    if not body:
        return set()
    m = _EVIDENCE_SECTION_RE.search(body)
    if not m:
        return set()
    raw = m.group(1).strip()
    if not raw or raw == "see report card":
        return set()
    return {piece.strip().lower() for piece in raw.split(",") if piece.strip()}


def fetch_recently_closed_proposals(
    repo: str, token: str, *, run_date: str, weeks: int = RECONFIRM_LOOKBACK_WEEKS,
    gh_request=_gh_request, max_pages: int = 10,
) -> list[dict]:
    """``area:director-proposals`` issues CLOSED within ``weeks`` of
    ``run_date`` as structured dicts (number/title/body/closed_at) — the
    shared fetch behind both the code-enforced re-confirm match
    (``find_reconfirm_match``) and the advisory LLM digest
    (``recently_closed_proposals_digest``). ``run_date`` is ``YYYY-MM-DD``; an
    unparseable date or a fetch failure yields ``[]`` (best-effort). Pages are
    fetched newest-closed-first and the loop stops once a page has nothing
    within the window."""
    from datetime import date, timedelta

    try:
        y, m, d = (int(x) for x in run_date.split("-"))
        cutoff = (date(y, m, d) - timedelta(weeks=weeks)).isoformat()
    except Exception:  # noqa: BLE001 — bad run_date → no dedup context, not fatal
        return []
    api = f"https://api.github.com/repos/{repo}"
    out: list[dict] = []
    for page in range(1, max_pages + 1):
        # sort=updated&direction=desc: a closed issue is rarely touched after
        # closing, so updated_at is a safe lower bound on closed_at for paging
        # termination; we still filter PRECISELY on closed_at below.
        url = (f"{api}/issues?state=closed&labels={PROPOSAL_LABEL}"
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
                out.append(it)
        # Once a full page has nothing within the window, older pages won't either.
        if len(items) < 100 or not page_has_recent:
            break
    return out


def find_reconfirm_match(item: ActionItem, closed_proposals: list[dict]) -> list[dict]:
    """Which of ``closed_proposals`` (structured issue dicts, e.g. from
    ``fetch_recently_closed_proposals``) this NEW ``item`` should be treated as
    a re-confirm OF — i.e. share ``>= _MIN_EVIDENCE_OVERLAP`` cited evidence
    names with ``item.evidence`` (config#2307).

    This is the code-enforced identity check: the exact-slug skip
    (``existing_proposal_slugs``) only catches a re-fire under the IDENTICAL
    slug, and the LLM is free to name a semantically-identical investigation
    with a brand-new slug (as happened re-filing #1061/#1168 under a fresh id
    for #1639) — evidence is the stable identity the slug isn't. Returns the
    matching prior issues (may be more than one), sorted oldest-first so the
    body cites the ORIGINAL investigation first; ``[]`` if no match (file as a
    normal fresh proposal)."""
    item_evidence = {e.strip().lower() for e in (item.evidence or []) if e.strip()}
    if not item_evidence:
        return []
    matches: list[dict] = []
    for prior in closed_proposals:
        prior_evidence = parse_evidence_from_body(prior.get("body") or "")
        if len(item_evidence & prior_evidence) >= _MIN_EVIDENCE_OVERLAP:
            matches.append(prior)
    matches.sort(key=lambda p: (p.get("closed_at") or ""))
    return matches


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
    Before filing, each new item is also checked against recently-CLOSED
    proposals for an evidence match (config#2307,
    ``find_reconfirm_match``/``RECONFIRM_LOOKBACK_WEEKS``); a match files as a
    lower-confidence re-confirm variant (``render_reconfirm_issue``, capped
    ``P2``, explicitly linking the prior closure) instead of a fresh proposal
    at the item's original suggested priority — this is the code-enforced
    backstop for the advisory ``recently_closed_proposals_digest`` prompt
    context, which the LLM is free to (and did) ignore.

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

    closed_proposals = fetch_recently_closed_proposals(
        repo, token, run_date=run_date, gh_request=gh_request,
    )

    filed: list[dict] = []
    for item in new_items:
        reconfirm_of = find_reconfirm_match(item, closed_proposals)
        if reconfirm_of:
            title, body, priority = render_reconfirm_issue(item, run_date, reconfirm_of)
            logger.info(
                "issue_filer: %r matches prior closed evidence (%s) — filing as "
                "re-confirm at %s instead of fresh %s.",
                item.id, ", ".join(f"#{p.get('number')}" for p in reconfirm_of),
                priority, item.priority,
            )
        else:
            title, body = render_issue(item, run_date)
            priority = item.priority
        status, res = gh_request(
            "POST", f"{api}/issues", token,
            {"title": title, "body": body, "labels": [PROPOSAL_LABEL, priority]},
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
        filed.append({
            "slug": item.id, "number": res.get("number"), "url": res.get("html_url"),
            "reconfirm_of": [p.get("number") for p in reconfirm_of] if reconfirm_of else [],
        })

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
    repo: str, token: str, *, run_date: str, weeks: int = RECONFIRM_LOOKBACK_WEEKS,
    gh_request=_gh_request, max_chars: int = 6000, max_pages: int = 10,
) -> str:
    """Condense director-proposal issues CLOSED within ``weeks`` of ``run_date``
    into a digest so the director won't re-litigate a just-resolved concern.

    The open-backlog digest only shows OPEN issues, and the file-time slug-skip
    only catches the EXACT slug — so a recently-investigated concern that's been
    closed can return under a semantically-similar NEW slug, evading both. This
    feeds those recently-closed proposals to the LLM as "already examined — do
    not re-propose" (config#1164 follow-up). This text digest is ADVISORY ONLY
    (the LLM can and did ignore it — config#2307); the enforced backstop is
    ``find_reconfirm_match`` against the same ``fetch_recently_closed_proposals``
    fetch this digest is built from. ``run_date`` is ``YYYY-MM-DD``; an
    unparseable date or a fetch failure yields an empty digest (best-effort —
    the caller treats absence as "no extra dedup context")."""
    closed = fetch_recently_closed_proposals(
        repo, token, run_date=run_date, weeks=weeks, gh_request=gh_request, max_pages=max_pages,
    )
    lines = [
        f"#{it.get('number')} (closed {(it.get('closed_at') or '')[:10]}) {it.get('title', '')}"[:300]
        for it in closed
    ]
    return "\n".join(lines)[:max_chars]
