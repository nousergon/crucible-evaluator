"""
loop_verification.py — Phase H+ (config#3145): close the Director loop by
verifying, each week, that LAST week's filed proposals actually recovered
their cited metric — not just that a PR merged and GitHub auto-closed the
issue.

Why: ``issue_filer.py`` files one GitHub issue per ``ActionItem`` with a
textual "## Closes when" section, but nothing enforces it — GitHub
auto-closes on a PR's "Closes #NN" the moment it merges, regardless of
whether the cited evidence actually cleared its red-line. config#3142 is the
proof case: config#2350 (price_cache freshness coverage gap) closed within a
day by adding MONITOR coverage, while the underlying cache stayed
unrefreshed for weeks and carried in the ledger as still-red — no issue said
"the refresh is broken" until config#3142 surfaced it directly, weeks later.

Two checks this module runs against the FRESH Report Card at the top of each
weekly Director cycle:

  1. **reopen-if-unrecovered** — for every ledger item with a tracked GH
     issue number whose issue is now CLOSED, check whether the evidence it
     cited is still RED/WATCH on the current card. If so, REOPEN the issue
     with a comment naming the still-adverse evidence (the "green !=
     produced" class, config#3053, applied to remediation) — a merged PR
     advanced the item, but only the metric closes it.
  2. **carryover-escalation** — any ledger item carried (status !=
     "resolved") for >= ``ESCALATION_CARRY_THRESHOLD`` consecutive Director
     runs, not already escalated, gets ``gate:decision`` + a structured
     Ask-block comment on its issue, per the ruling-latency principle
     (config-I3123) — operator-plane items stop aging silently in the
     ledger.

``backfill_issue_numbers`` closes the bootstrap gap: the ledger did not track
``issue_number`` before this change, so on first run (and for any item that
otherwise lost track of its issue) it re-derives the mapping from the live
GitHub state by the ``id=<slug>`` marker every filed issue already carries.

Both checks are best-effort against the GitHub API (the same
``_gh_request`` used by ``issue_filer.py`` / ``roadmap_pr.py``) and NEVER
fatal — the plan + ledger (the primary deliverables) are already persisted
by the time this runs; a single item's GitHub call failing is logged and
skipped, not fatal to the pass.
"""

from __future__ import annotations

import logging

from director.issue_filer import slug_issue_number_map
from director.roadmap_pr import _gh_request

logger = logging.getLogger(__name__)

ADVERSE_STATUSES = {"RED", "WATCH"}

# Consecutive non-resolved runs before a ledger item escalates to the
# Decision Queue (config-I3123 ruling-latency principle).
ESCALATION_CARRY_THRESHOLD = 2

_ESCALATION_LABELS = {"gate:decision", "gate:operator"}


def component_status_map(card: dict) -> dict[str, str]:
    """Flatten every tile's components into ``{name.lower(): status}`` — the
    lookup ``evidence_still_adverse`` checks a cited evidence name against."""
    out: dict[str, str] = {}
    for tile in (card.get("tiles") or {}).values():
        for c in tile.get("components", []) or []:
            name = c.get("name")
            if name:
                out[str(name).strip().lower()] = str(c.get("status") or "")
    return out


def evidence_still_adverse(evidence: list[str], status_map: dict[str, str]) -> str:
    """``"adverse"`` / ``"recovered"`` / ``"unverifiable"`` for a CLOSED
    issue's cited evidence against the CURRENT card.

    "unverifiable" (none of the cited names appear on the current card — a
    metric can be renamed/retired between runs) intentionally does NOT
    reopen: we only reopen on POSITIVE evidence the tile is still red, never
    on absence of evidence — reopening on "we can't tell" would be the same
    "confident prescription off a metric we can't see" failure mode
    ARCHITECTURE.md already warns the Director's own output against."""
    found = False
    for name in evidence or []:
        status = status_map.get(str(name).strip().lower())
        if status is None:
            continue
        found = True
        if status in ADVERSE_STATUSES:
            return "adverse"
    return "unverifiable" if not found else "recovered"


def backfill_issue_numbers(
    ledger_items: list[dict], *, repo: str, token: str, gh_request=_gh_request
) -> int:
    """Fill ``issue_number`` on any ledger row missing it, by matching
    ``id=<slug>`` against the live ``area:director-proposals`` issues.
    Mutates ``ledger_items`` in place; returns the count filled. Best-effort:
    a fetch failure leaves rows as-is (they're simply skipped by
    ``verify_and_correct`` this cycle, same as before this ran)."""
    missing = [it for it in ledger_items if not it.get("issue_number")]
    if not missing:
        return 0
    try:
        slug_map = slug_issue_number_map(repo, token, gh_request=gh_request)
    except Exception as e:  # noqa: BLE001 — best-effort backfill
        logger.warning("loop_verification: backfill fetch failed: %s", e)
        return 0
    filled = 0
    for it in missing:
        number = slug_map.get(it.get("id"))
        if number is not None:
            it["issue_number"] = number
            filled += 1
    return filled


def verify_and_correct(
    ledger_items: list[dict],
    card: dict,
    *,
    repo: str,
    token: str,
    gh_request=_gh_request,
) -> dict:
    """Run both loop-closing checks against the ledger + current card.

    Mutates ``ledger_items`` in place (sets ``escalated=True`` on any item
    this pass escalates, so a subsequent ledger write persists the flag and
    next week's run doesn't re-escalate it). Returns a summary dict with
    per-outcome counts + the acted-on issue numbers, suitable for folding
    into the handler's summary / the weekly digest (config#3145 point 4).
    Never raises — a single item's GitHub call failing is logged and
    skipped, the rest of the pass continues."""
    status_map = component_status_map(card)
    api = f"https://api.github.com/repos/{repo}"

    counts = {
        "open": 0, "closed_verified": 0, "closed_unrecovered": 0,
        "closed_unverifiable": 0, "escalated": 0,
    }
    reopened: list[int] = []
    escalated: list[int] = []

    for item in ledger_items:
        number = item.get("issue_number")
        if not number:
            continue
        try:
            status, res = gh_request("GET", f"{api}/issues/{number}", token)
        except Exception as e:  # noqa: BLE001 — one bad item must not sink the pass
            logger.warning("loop_verification: GET issue #%s failed: %s", number, e)
            continue
        if status != 200 or not isinstance(res, dict):
            logger.warning("loop_verification: GET issue #%s -> HTTP %s", number, status)
            continue

        if res.get("state") == "closed":
            outcome = evidence_still_adverse(item.get("evidence") or [], status_map)
            if outcome == "adverse":
                counts["closed_unrecovered"] += 1
                if _reopen_unrecovered(api, number, item, gh_request, token):
                    reopened.append(number)
            elif outcome == "recovered":
                counts["closed_verified"] += 1
            else:
                counts["closed_unverifiable"] += 1
            continue

        counts["open"] += 1
        carry_count = item.get("carry_count", 0)
        already_gated = bool(_labels_of(res) & _ESCALATION_LABELS)
        if carry_count >= ESCALATION_CARRY_THRESHOLD and not item.get("escalated") and not already_gated:
            if _escalate_carryover(api, number, item, carry_count, gh_request, token):
                item["escalated"] = True
                counts["escalated"] += 1
                escalated.append(number)

    return {**counts, "reopened_issues": reopened, "escalated_issues": escalated}


def _labels_of(issue: dict) -> set[str]:
    return {label.get("name") for label in issue.get("labels", []) or [] if label.get("name")}


def _reopen_unrecovered(api: str, number: int, item: dict, gh_request, token: str) -> bool:
    evidence = ", ".join(item.get("evidence") or []) or "the cited evidence"
    status, _ = gh_request("PATCH", f"{api}/issues/{number}", token, {"state": "open"})
    if status not in (200, 201):
        logger.warning("loop_verification: reopen issue #%s -> HTTP %s", number, status)
        return False
    comment = (
        "**Director loop-verification (config#3145):** this issue closed, but "
        f"{evidence} still reads RED/WATCH on the current Report Card — the "
        "closing change advanced the item, it did not recover the metric. "
        "Reopening; the closes-when is metric recovery, not PR merge."
    )
    gh_request("POST", f"{api}/issues/{number}/comments", token, {"body": comment})
    return True


def _escalate_carryover(
    api: str, number: int, item: dict, carry_count: int, gh_request, token: str
) -> bool:
    owner = item.get("proposed_owner", "the proposed owner")
    title = item.get("title", "(untitled)")
    ask = (
        f"**Summary:** Director action item `{item.get('id')}` (\"{title}\") has carried "
        f"{carry_count} consecutive weekly Director runs without resolving.\n"
        f"**Ask:** Should this stay owned by {owner} at its current priority, be "
        "reprioritized, or be dropped as no-longer-relevant?\n"
        f"**Options:** A) Keep as-is, {owner} to act (recommended) B) Reprioritize "
        "C) Drop — no longer relevant\n"
        "**SOTA:** Carried items get an explicit ownership ruling before they age "
        "further (the ruling-latency principle, config-I3123).\n"
        "**Delta:** IS SOTA — no delta.\n"
        "**Consequence of no action:** the item keeps aging silently in the carry-over "
        "ledger with no forcing function.\n\n"
        f"<sub>Director loop-verification (config#3145) · id={item.get('id')} · "
        f"carry_count={carry_count}</sub>"
    )
    status, _ = gh_request(
        "POST", f"{api}/issues/{number}/labels", token, {"labels": ["gate:decision"]},
    )
    if status not in (200, 201):
        logger.warning("loop_verification: label issue #%s -> HTTP %s", number, status)
        return False
    gh_request("POST", f"{api}/issues/{number}/comments", token, {"body": ask})
    return True
