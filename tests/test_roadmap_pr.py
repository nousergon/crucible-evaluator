"""Tests for director/roadmap_pr.py — Phase H ROADMAP-PR channel.

Covers the pure renderer/reconciler (idempotency by slug, sequential L-numbers,
minimal-diff upsert, digest) and the PR orchestrator against a mocked GitHub
transport — including the load-bearing invariant that it NEVER calls a merge
endpoint.
"""

from __future__ import annotations

import base64

import pytest

from director import roadmap_pr as rp
from director.schema import ActionItem, DirectorWeeklyActionPlan


def _item(slug: str, *, priority="P1", owner="research", title=None) -> ActionItem:
    return ActionItem(
        id=slug,
        title=title or f"Do the thing {slug}",
        rationale=f"Because metric X is bad for {slug}",
        evidence=["metrics.json::sharpe_ratio", "research_tile"],
        proposed_owner=owner,
        priority=priority,
        horizon="this_week",
        suggested_change_type="investigation",
        confidence=70,
    )


def _plan(*slugs: str) -> DirectorWeeklyActionPlan:
    return DirectorWeeklyActionPlan(
        run_date="2026-06-05",
        system_summary="System is RED across the board.",
        top_risks=["alpha is negative", "pipeline 43%"],
        action_items=[_item(s) for s in slugs],
    )


ROADMAP_BASE = """# Alpha Engine — Private Development Roadmap

> Some preamble.

## Research
- [ ] **L4100 · P2 — An existing item.** (added 2026-06-01; id=existing-one) body.
- [x] **L4099 · P3 — A done item.** (id=done-thing)

## Predictor
- [~] **L4205 · P1 — In progress.** (id=wip-thing) body.
"""


# --------------------------------------------------------------------------- #
# Pure functions
# --------------------------------------------------------------------------- #
def test_parse_max_l_number():
    assert rp.parse_max_l_number(ROADMAP_BASE) == 4205
    assert rp.parse_max_l_number("no L numbers here") == 0


def test_existing_slugs():
    slugs = rp.existing_slugs(ROADMAP_BASE)
    assert slugs == {"existing-one", "done-thing", "wip-thing"}


def test_render_entry_is_house_style():
    line = rp.render_entry(_item("neg-alpha", priority="P0", owner="operator"), "L4206", "2026-06-05")
    assert line.startswith("- [ ] **L4206 · P0 — [director] ")
    assert "id=neg-alpha" in line
    assert "owner=operator" in line
    assert "**Closes when:**" in line
    assert "Evidence: metrics.json::sharpe_ratio, research_tile" in line


def test_select_new_items_skips_already_filed():
    plan = _plan("existing-one", "brand-new")
    new = rp.select_new_items(plan, rp.existing_slugs(ROADMAP_BASE))
    assert [i.id for i in new] == ["brand-new"]


def test_select_new_items_dedups_within_plan():
    plan = _plan("dup", "dup", "other")
    new = rp.select_new_items(plan, set())
    assert [i.id for i in new] == ["dup", "other"]


def test_render_proposal_entries_sequential_l():
    items = [_item("a"), _item("b"), _item("c")]
    entries = rp.render_proposal_entries(items, "2026-06-05", 4206)
    assert [e.split(" · ")[0] for e in entries] == [
        "- [ ] **L4206",
        "- [ ] **L4207",
        "- [ ] **L4208",
    ]


def test_upsert_creates_section_when_absent():
    out = rp.upsert_into_roadmap(ROADMAP_BASE, ["- [ ] entry one"], "2026-06-05")
    assert "## Director Proposals" in out
    assert "### Week of 2026-06-05" in out
    assert "- [ ] entry one" in out
    # Existing content preserved.
    assert "## Research" in out and "## Predictor" in out


def test_upsert_empty_entries_is_noop():
    assert rp.upsert_into_roadmap(ROADMAP_BASE, [], "2026-06-05") == ROADMAP_BASE


def test_upsert_appends_to_existing_week():
    once = rp.upsert_into_roadmap(ROADMAP_BASE, ["- [ ] entry one"], "2026-06-05")
    twice = rp.upsert_into_roadmap(once, ["- [ ] entry two"], "2026-06-05")
    assert twice.count("### Week of 2026-06-05") == 1  # same subsection reused
    assert "- [ ] entry one" in twice and "- [ ] entry two" in twice
    assert twice.index("entry one") < twice.index("entry two")


def test_upsert_new_week_subsection():
    wk1 = rp.upsert_into_roadmap(ROADMAP_BASE, ["- [ ] entry one"], "2026-06-05")
    wk2 = rp.upsert_into_roadmap(wk1, ["- [ ] entry two"], "2026-06-12")
    assert "### Week of 2026-06-05" in wk2
    assert "### Week of 2026-06-12" in wk2
    assert wk2.count("## Director Proposals") == 1  # one section, two week blocks


def test_roadmap_digest_keeps_open_items_only():
    digest = rp.roadmap_digest(ROADMAP_BASE)
    assert "An existing item" in digest          # [ ]
    assert "In progress" in digest               # [~]
    assert "A done item" not in digest           # [x] dropped
    assert "Some preamble" not in digest


# --------------------------------------------------------------------------- #
# PR orchestrator (mocked GitHub transport)
# --------------------------------------------------------------------------- #
class FakeGitHub:
    """Records every (method, url) and serves canned REST responses. Branch
    content mirrors base until a PUT updates it."""

    def __init__(self, base_text: str = ROADMAP_BASE):
        self.calls: list[tuple[str, str]] = []
        self._base = base_text
        self._branch = base_text  # branch starts as a copy of base
        self.existing_prs: list[dict] = []

    def __call__(self, method, url, token, body=None):
        self.calls.append((method, url))
        b64 = lambda t: base64.b64encode(t.encode()).decode()  # noqa: E731
        if method == "GET" and "/contents/" in url and "ref=main" in url:
            return 200, {"content": b64(self._base), "sha": "base-sha"}
        if method == "GET" and "/git/ref/heads/main" in url:
            return 200, {"object": {"sha": "head-sha"}}
        if method == "POST" and url.endswith("/git/refs"):
            return 201, {"ref": body["ref"]}
        if method == "GET" and "/contents/" in url and "ref=director/roadmap-" in url:
            return 200, {"content": b64(self._branch), "sha": "branch-sha"}
        if method == "PUT" and "/contents/" in url:
            self._branch = base64.b64decode(body["content"]).decode()
            return 200, {"commit": {"sha": "new-commit"}}
        if method == "GET" and "/pulls?" in url:
            return 200, list(self.existing_prs)
        if method == "POST" and url.endswith("/pulls"):
            return 201, {"html_url": "https://github.com/cipher813/alpha-engine-config/pull/99"}
        raise AssertionError(f"unexpected call: {method} {url}")

    @property
    def merge_calls(self):
        return [(m, u) for (m, u) in self.calls if "/merge" in u]


def test_open_roadmap_pr_happy_path():
    gh = FakeGitHub()
    res = rp.open_roadmap_pr(_plan("brand-new", "another-new"), "2026-06-05",
                             token="tok", gh_request=gh)
    assert res["status"] == "ok"
    assert res["pr_url"].endswith("/pull/99")
    assert res["n_filed"] == 2
    assert res["branch"] == "director/roadmap-2026-06-05"
    assert res["l_numbers"] == ["L4206", "L4207"]
    # The updated branch content carries both new slugs as house-style entries.
    assert "id=brand-new" in gh._branch and "id=another-new" in gh._branch
    assert "## Director Proposals" in gh._branch


def test_open_roadmap_pr_never_merges():
    gh = FakeGitHub()
    rp.open_roadmap_pr(_plan("brand-new"), "2026-06-05", token="tok", gh_request=gh)
    assert gh.merge_calls == []
    # Sanity: no PATCH/PUT to a pulls/{n} merge or refs/heads/main fast-forward.
    assert not any(m == "PUT" and "/git/refs/heads/main" in u for m, u in gh.calls)


def test_open_roadmap_pr_nochange_when_all_filed():
    gh = FakeGitHub()
    # Both slugs already exist in the base ROADMAP → nothing to file.
    res = rp.open_roadmap_pr(_plan("existing-one", "wip-thing"), "2026-06-05",
                             token="tok", gh_request=gh)
    assert res["status"] == "nochange"
    assert res["n_filed"] == 0
    # Short-circuits after the first contents GET — no branch/commit/PR.
    assert gh.calls == [("GET",
                         "https://api.github.com/repos/cipher813/alpha-engine-config/"
                         "contents/private-docs/ROADMAP.md?ref=main")]


def test_open_roadmap_pr_returns_existing_open_pr():
    gh = FakeGitHub()
    gh.existing_prs = [{"html_url": "https://github.com/cipher813/alpha-engine-config/pull/42"}]
    res = rp.open_roadmap_pr(_plan("brand-new"), "2026-06-05", token="tok", gh_request=gh)
    assert res["pr_url"].endswith("/pull/42")
    # Did NOT POST a second PR.
    assert not any(m == "POST" and u.endswith("/pulls") for m, u in gh.calls)
