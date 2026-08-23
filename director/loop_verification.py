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
import re

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
    lookup ``resolve_cited_metrics`` checks cited text against."""
    out: dict[str, str] = {}
    for tile in (card.get("tiles") or {}).values():
        for c in tile.get("components", []) or []:
            name = c.get("name")
            if name:
                out[str(name).strip().lower()] = str(c.get("status") or "")
    return out


#: Identifier-shaped tokens inside a free-text citation. Component names are
#: snake_case (``momentum_l1_ic``), and ``[a-z][a-z0-9_]*`` keeps such a name
#: whole while splitting the prose around it — so "predictor tile
#: momentum_l1_ic" yields ``predictor``/``tile``/``momentum_l1_ic`` and the
#: third token matches exactly. Exact token equality, never substring
#: containment: ``scanner`` is itself a live component name, and a substring
#: match would fire it on every ``scanner_*`` sibling.
_CITATION_TOKEN = re.compile(r"[a-z][a-z0-9_]*")


def resolve_cited_metrics(texts, status_map: dict[str, str]) -> dict[str, str]:
    """Every card component named anywhere in ``texts`` → its current status.

    **This is the fix for a measured detection blindness
    (alpha-engine-config-I8178).** The prior implementation looked each
    evidence string up in ``status_map`` WHOLE. The Director does not write
    bare component names: measured against the live ledger on 2026-08-22, 26
    of 28 rows cited their metric as prose — ``"predictor tile
    momentum_l1_ic"``, ``"backtester tile backtest_vs_live_parity"`` — and a
    whole-string lookup matched none of them. So 26 of 28 rows resolved to
    ``"unverifiable"``, and since "unverifiable" deliberately never reopens,
    the reconciliation pass was a no-op on 93% of the ledger while reporting
    a clean run. Token resolution takes that 26 down to 4.

    The blindness outranked the staleness it hid: the check existed, was
    wired, ran, and returned an answer that was structurally incapable of
    being anything but "can't tell"."""
    hits: dict[str, str] = {}
    for text in texts or []:
        for token in _CITATION_TOKEN.findall(str(text).lower()):
            status = status_map.get(token)
            if status is not None:
                hits[token] = status
    return hits


def evidence_still_adverse(evidence: list[str], status_map: dict[str, str]) -> str:
    """``"adverse"`` / ``"recovered"`` / ``"unverifiable"`` for a CLOSED
    issue's cited evidence against the CURRENT card.

    "unverifiable" (none of the cited names appear on the current card — a
    metric can be renamed/retired between runs) intentionally does NOT
    reopen: we only reopen on POSITIVE evidence the tile is still red, never
    on absence of evidence — reopening on "we can't tell" would be the same
    "confident prescription off a metric we can't see" failure mode
    ARCHITECTURE.md already warns the Director's own output against.

    Scope note: this reads the ``evidence`` list ONLY, never the row's title
    or rationale, because its caller REOPENS a closed GitHub issue — a
    mutating, human-visible action that should fire on the citation the item
    formally staked, not on a metric its prose happened to mention.
    ``_carryover_context``'s annotation is non-mutating and deliberately
    reads wider."""
    hits = resolve_cited_metrics(evidence, status_map)
    if not hits:
        return "unverifiable"
    return "adverse" if any(s in ADVERSE_STATUSES for s in hits.values()) else "recovered"


def backfill_issue_numbers(
    ledger_items: list[dict], *, repo: str, token: str, gh_request=_gh_request
) -> int:
    """Fill ``issue_number`` on any ledger row missing it, by matching
    ``id=<slug>`` against the live ``area:director-proposals`` issues.
    Mutates ``ledger_items`` in place and returns the count filled. A fetch
    failure propagates to the handler, which records it explicitly rather than
    rendering a failed repair as zero backfills."""
    missing = [it for it in ledger_items if not it.get("issue_number")]
    if not missing:
        return 0
    slug_map = slug_issue_number_map(repo, token, gh_request=gh_request)
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
        "examined": 0, "skipped_no_issue": 0, "lookup_failed": 0, "corrections": 0,
        "open": 0, "closed_verified": 0, "closed_unrecovered": 0,
        "closed_unverifiable": 0, "escalated": 0,
    }
    reopened: list[int] = []
    escalated: list[int] = []

    for item in ledger_items:
        number = item.get("issue_number")
        if not number:
            counts["skipped_no_issue"] += 1
            continue
        try:
            status, res = gh_request("GET", f"{api}/issues/{number}", token)
        except Exception as e:  # noqa: BLE001 — one bad item must not sink the pass
            logger.warning("loop_verification: GET issue #%s failed: %s", number, e)
            counts["lookup_failed"] += 1
            continue
        if status != 200 or not isinstance(res, dict):
            logger.warning("loop_verification: GET issue #%s -> HTTP %s", number, status)
            counts["lookup_failed"] += 1
            continue

        counts["examined"] += 1
        if res.get("state") == "closed":
            outcome = evidence_still_adverse(item.get("evidence") or [], status_map)
            if outcome == "adverse":
                counts["closed_unrecovered"] += 1
                if _reopen_unrecovered(api, number, item, gh_request, token):
                    reopened.append(number)
                    counts["corrections"] += 1
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
                counts["corrections"] += 1
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
