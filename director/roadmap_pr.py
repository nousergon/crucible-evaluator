"""
roadmap_pr.py — Phase H: render the weekly ``DirectorWeeklyActionPlan`` into
house-style ROADMAP entries and open an **approval-gated PR** against
``alpha-engine-config``.

The Director PROPOSES; Brian DISPOSES — he reviews and merges the PR (his
review IS the gate; there is no soak flag). **No merge call here, no write to
any live trading config.** The only effect is one branch + one PR per run.

Design invariants (mirror director-plan §4.3 + the README):
  - **Idempotent by ``ActionItem.id``.** Each rendered entry embeds an
    ``id=<slug>`` marker; an item whose slug already appears anywhere in
    ROADMAP.md is skipped, so weekly re-runs (and same-week re-fires) never
    duplicate. If nothing new remains, NO PR is opened.
  - **House-style entries.** Each item renders as a real
    ``- [ ] **L#### · P# — [director] …**`` line so Brian can accept it in
    place or route it into a module section on merge. New L-numbers continue
    from the file's current max.
  - **Minimal, deterministic diff.** New entries land under a dedicated
    ``## Director Proposals`` → ``### Week of {run_date}`` subsection; the rest
    of the file is untouched, and the output is a pure function of (plan, run
    date, current ROADMAP text).
  - **Reconciled against the digest.** The handler also feeds the live ROADMAP
    digest INTO the director call so the LLM avoids re-proposing tracked work;
    the slug skip here is the belt-and-suspenders second line of defense.

Auth: a fine-grained PAT scoped to ``alpha-engine-config`` ONLY
(contents:write + pull_requests:write, **no merge**), fetched from SSM via
``alpha_engine_lib.secrets.get_secret`` — the fleet's institutional secret path,
mirroring the cyphering release-queue token pattern. The director Lambda role's
existing ``ssm:GetParameter`` on ``parameter/alpha-engine/*`` already covers the
token param, so no IAM change is required.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import urllib.error
import urllib.request

from director.schema import ActionItem, DirectorWeeklyActionPlan

logger = logging.getLogger(__name__)

DEFAULT_REPO = "cipher813/alpha-engine-config"
DEFAULT_PATH = "private-docs/ROADMAP.md"
DEFAULT_BASE = "main"
TOKEN_SECRET_NAME = "DIRECTOR_GITHUB_TOKEN"  # SSM /alpha-engine/DIRECTOR_GITHUB_TOKEN

_SECTION_HEADER = "## Director Proposals"
_SECTION_PREAMBLE = (
    "> Auto-filed by the weekly Director (Layer C, Phase H). Each line is a "
    "house-style entry carrying an `id=<slug>` idempotency marker — review, then "
    "accept in place or route into the owning module section. **Brian merges; the "
    "Director never self-merges, and nothing here writes live trading config.**"
)

_L_NUMBER_RE = re.compile(r"\bL(\d{3,})\b")
_SLUG_RE = re.compile(r"\bid=([A-Za-z0-9_\-]+)")


# --------------------------------------------------------------------------- #
# Pure rendering / reconciliation (no I/O — fully unit-testable)
# --------------------------------------------------------------------------- #
def parse_max_l_number(roadmap_text: str) -> int:
    """Highest existing ``L####`` in the file (0 if none) — new entries continue
    from ``max + 1``."""
    nums = [int(m) for m in _L_NUMBER_RE.findall(roadmap_text)]
    return max(nums) if nums else 0


def existing_slugs(roadmap_text: str) -> set[str]:
    """Every ``id=<slug>`` already filed — the idempotency key set."""
    return set(_SLUG_RE.findall(roadmap_text))


def render_entry(item: ActionItem, l_number: str, run_date: str) -> str:
    """One house-style ROADMAP line for an action item, carrying its idempotency
    marker. The ``**Closes when:**`` clause is templated from the item's cited
    evidence + change type — Brian tightens it on review."""
    evidence = ", ".join(item.evidence) if item.evidence else "see report card"
    change = item.suggested_change_type.replace("_", " ")
    title = item.title.rstrip().rstrip(".")
    rationale = item.rationale.rstrip()
    return (
        f"- [ ] **{l_number} · {item.priority} — [director] {title}.** "
        f"(added {run_date} by Director; owner={item.proposed_owner}; "
        f"horizon={item.horizon}; confidence={item.confidence}; id={item.id}) "
        f"{rationale} Evidence: {evidence}. "
        f"**Closes when:** the cited evidence ({evidence}) clears its "
        f"target/red-line, or the proposed {change} ships and is verified."
    )


def select_new_items(
    plan: DirectorWeeklyActionPlan, already_filed: set[str]
) -> list[ActionItem]:
    """Action items not yet present in the ROADMAP, deduped by slug (preserving
    plan order). Items with a blank id are always included (can't dedup a blank
    key) — the schema requires a slug, so this is defensive."""
    out: list[ActionItem] = []
    seen: set[str] = set()
    for item in plan.action_items:
        slug = (item.id or "").strip()
        if slug and (slug in already_filed or slug in seen):
            continue
        if slug:
            seen.add(slug)
        out.append(item)
    return out


def render_proposal_entries(
    items: list[ActionItem], run_date: str, start_l: int
) -> list[str]:
    """Render the selected items into house-style lines with sequential
    L-numbers starting at ``start_l``."""
    return [
        render_entry(item, f"L{start_l + i}", run_date)
        for i, item in enumerate(items)
    ]


def upsert_into_roadmap(roadmap_text: str, entries: list[str], run_date: str) -> str:
    """Insert rendered ``entries`` under ``## Director Proposals`` →
    ``### Week of {run_date}``, returning the new file text.

    - Section absent → append the whole section at EOF.
    - Section present, week subsection absent → append a new week subsection at
      the end of the section.
    - Week subsection present → append entries to it (so a same-week re-fire that
      surfaces a *new* item extends the existing block rather than forking one).

    Pure function: with ``entries == []`` it returns the input unchanged (the
    caller skips the PR in that case)."""
    if not entries:
        return roadmap_text

    week_header = f"### Week of {run_date}"
    block = "\n".join(entries)

    lines = roadmap_text.splitlines()
    sec_idx = next((i for i, ln in enumerate(lines) if ln.strip() == _SECTION_HEADER), None)

    if sec_idx is None:
        # Section absent — append at EOF.
        tail = "" if roadmap_text.endswith("\n") else "\n"
        return (
            roadmap_text
            + tail
            + f"\n{_SECTION_HEADER}\n\n{_SECTION_PREAMBLE}\n\n{week_header}\n\n{block}\n"
        )

    # Find the extent of the section: from sec_idx to the next top-level "## " or EOF.
    sec_end = len(lines)
    for i in range(sec_idx + 1, len(lines)):
        if lines[i].startswith("## "):
            sec_end = i
            break

    week_idx = next(
        (i for i in range(sec_idx + 1, sec_end) if lines[i].strip() == week_header),
        None,
    )

    if week_idx is None:
        # New week subsection at the end of the section.
        insert_at = sec_end
        # Trim trailing blank lines inside the section so spacing stays tidy.
        while insert_at - 1 > sec_idx and lines[insert_at - 1].strip() == "":
            insert_at -= 1
        new_block = ["", week_header, "", *entries]
        lines[insert_at:insert_at] = new_block
        return "\n".join(lines) + ("\n" if roadmap_text.endswith("\n") else "")

    # Week subsection exists — append entries at its end.
    week_end = sec_end
    for i in range(week_idx + 1, sec_end):
        if lines[i].startswith("### ") or lines[i].startswith("## "):
            week_end = i
            break
    insert_at = week_end
    while insert_at - 1 > week_idx and lines[insert_at - 1].strip() == "":
        insert_at -= 1
    lines[insert_at:insert_at] = entries
    return "\n".join(lines) + ("\n" if roadmap_text.endswith("\n") else "")


def roadmap_digest(roadmap_text: str, *, max_chars: int = 12000) -> str:
    """Condense the ROADMAP into a digest of open-item title lines for the
    director prompt (so it doesn't re-propose tracked work). Keeps ``[ ]``/``[~]``
    top-level entries, dropping body prose; truncated to ``max_chars``."""
    kept: list[str] = []
    for ln in roadmap_text.splitlines():
        s = ln.strip()
        if s.startswith("- [ ]") or s.startswith("- [~]"):
            # Keep through the bold title (first sentence) to stay compact.
            head = s.split(".**", 1)[0]
            kept.append(head[:300])
    digest = "\n".join(kept)
    return digest[:max_chars]


# --------------------------------------------------------------------------- #
# GitHub REST (fine-grained PAT, stdlib urllib — no new pip dep)
# --------------------------------------------------------------------------- #
def _gh_request(method: str, url: str, token: str, body: dict | None = None) -> tuple[int, dict]:
    """Minimal GitHub REST call. Raises urllib HTTPError only for unexpected
    statuses the callers don't handle; returns (status, parsed_json) otherwise."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "alpha-engine-director")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as resp:  # noqa: S310 - fixed api.github.com host
            return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        payload = e.read().decode("utf-8") if e.fp else ""
        try:
            parsed = json.loads(payload) if payload else {}
        except json.JSONDecodeError:
            parsed = {"raw": payload}
        return e.code, parsed


def open_roadmap_pr(
    plan: DirectorWeeklyActionPlan,
    run_date: str,
    *,
    token: str,
    repo: str = DEFAULT_REPO,
    path: str = DEFAULT_PATH,
    base: str = DEFAULT_BASE,
    gh_request=_gh_request,
) -> dict:
    """Render the plan into ROADMAP entries and open one approval-gated PR.

    Idempotent: items already filed (by slug) are skipped; if nothing new
    remains the function returns ``{"status": "nochange"}`` and opens no PR. A
    same-week re-fire reuses the dated branch and updates it in place. **Never
    calls the merge endpoint.**

    ``gh_request`` is injected for tests. Returns a compact summary
    (status / pr_url / branch / n_filed / l_numbers)."""
    api = f"https://api.github.com/repos/{repo}"

    # 1) Current ROADMAP on base.
    status, file_obj = gh_request("GET", f"{api}/contents/{path}?ref={base}", token)
    if status != 200:
        raise RuntimeError(f"roadmap_pr: GET contents {path}@{base} -> {status}: {file_obj}")
    current_text = base64.b64decode(file_obj["content"]).decode("utf-8")
    base_sha = file_obj["sha"]

    # 2) Reconcile + render.
    new_items = select_new_items(plan, existing_slugs(current_text))
    if not new_items:
        logger.info("roadmap_pr: no new items (all %d already filed) — no PR.", len(plan.action_items))
        return {"status": "nochange", "n_filed": 0, "reason": "all items already in ROADMAP"}
    start_l = parse_max_l_number(current_text) + 1
    entries = render_proposal_entries(new_items, run_date, start_l)
    new_text = upsert_into_roadmap(current_text, entries, run_date)
    l_numbers = [f"L{start_l + i}" for i in range(len(new_items))]

    # 3) Branch off base (idempotent: reuse if it already exists, e.g. a re-fire).
    branch = f"director/roadmap-{run_date}"
    s, ref = gh_request("GET", f"{api}/git/ref/heads/{base}", token)
    if s != 200:
        raise RuntimeError(f"roadmap_pr: GET ref heads/{base} -> {s}: {ref}")
    base_head = ref["object"]["sha"]
    s, made = gh_request(
        "POST", f"{api}/git/refs", token,
        {"ref": f"refs/heads/{branch}", "sha": base_head},
    )
    if s not in (201, 422):  # 422 = ref already exists (re-fire) — acceptable.
        raise RuntimeError(f"roadmap_pr: create branch {branch} -> {s}: {made}")

    # On a re-fire the branch's ROADMAP.md sha differs from base — fetch the
    # branch copy's sha so the update PUT targets the right blob.
    s, branch_file = gh_request("GET", f"{api}/contents/{path}?ref={branch}", token)
    put_sha = branch_file["sha"] if s == 200 else base_sha
    if s == 200:
        branch_text = base64.b64decode(branch_file["content"]).decode("utf-8")
        # Re-reconcile against the branch's current content (handles partial re-fire).
        re_items = select_new_items(plan, existing_slugs(branch_text))
        if not re_items:
            logger.info("roadmap_pr: branch %s already has all items — no commit.", branch)
            # The branch may already carry an open PR — surface it if so.
            existing_pr = _find_open_pr(api, token, branch, base, gh_request)
            return {"status": "nochange", "branch": branch, "n_filed": 0,
                    "pr_url": existing_pr, "reason": "items already on branch"}
        start_l = parse_max_l_number(branch_text) + 1
        entries = render_proposal_entries(re_items, run_date, start_l)
        new_text = upsert_into_roadmap(branch_text, entries, run_date)
        l_numbers = [f"L{start_l + i}" for i in range(len(re_items))]
        new_items = re_items

    # 4) Commit the updated ROADMAP to the branch (no merge).
    commit_msg = (
        f"docs(roadmap): Director weekly proposals — {run_date} "
        f"({len(new_items)} item{'s' if len(new_items) != 1 else ''})"
    )
    s, put_res = gh_request(
        "PUT", f"{api}/contents/{path}", token,
        {
            "message": commit_msg,
            "content": base64.b64encode(new_text.encode("utf-8")).decode("ascii"),
            "sha": put_sha,
            "branch": branch,
        },
    )
    if s not in (200, 201):
        raise RuntimeError(f"roadmap_pr: PUT contents -> {s}: {put_res}")

    # 5) Open the PR (or return the existing one). NO merge call.
    pr_url = _find_open_pr(api, token, branch, base, gh_request)
    if pr_url is None:
        s, pr = gh_request(
            "POST", f"{api}/pulls", token,
            {
                "title": f"Director weekly action plan — {run_date}",
                "head": branch,
                "base": base,
                "body": _pr_body(plan, run_date, l_numbers),
            },
        )
        if s != 201:
            raise RuntimeError(f"roadmap_pr: create PR -> {s}: {pr}")
        pr_url = pr.get("html_url")

    summary = {
        "status": "ok",
        "pr_url": pr_url,
        "branch": branch,
        "n_filed": len(new_items),
        "l_numbers": l_numbers,
    }
    logger.info("roadmap_pr: %s", summary)
    return summary


def _find_open_pr(api: str, token: str, branch: str, base: str, gh_request) -> str | None:
    owner = api.rstrip("/").split("/")[-2]
    s, prs = gh_request("GET", f"{api}/pulls?head={owner}:{branch}&base={base}&state=open", token)
    if s == 200 and isinstance(prs, list) and prs:
        return prs[0].get("html_url")
    return None


def _pr_body(plan: DirectorWeeklyActionPlan, run_date: str, l_numbers: list[str]) -> str:
    risks = "\n".join(f"- {r}" for r in plan.top_risks) or "_none surfaced_"
    items = "\n".join(
        f"- **{ln} · {it.priority}** ({it.proposed_owner}) — {it.title}"
        for ln, it in zip(l_numbers, plan.action_items[: len(l_numbers)])
    )
    return (
        f"**Director weekly action plan — run_date {run_date}** (advisory; review + merge).\n\n"
        f"This PR was opened automatically by the Director (Layer C, Phase H). It "
        f"proposes house-style ROADMAP entries under **## Director Proposals** — "
        f"review, then accept in place or route into the owning module section. "
        f"The Director never self-merges and writes no live trading config.\n\n"
        f"### Whole-system read\n{plan.system_summary}\n\n"
        f"### Top risks\n{risks}\n\n"
        f"### Proposed items ({len(l_numbers)})\n{items}\n\n"
        f"🤖 Generated with [Claude Code](https://claude.com/claude-code)"
    )
