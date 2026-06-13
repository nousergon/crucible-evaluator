"""Tests for director/issue_filer.py — Phase H issue-filing channel (config#978).

Covers idempotency by ActionItem.id across existing area:director-proposals
issues (open OR closed), the issue render (title id-marker + body discipline +
priority-unset labeling), the file orchestrator against a mocked GitHub
transport, and the open-issue backlog digest.
"""

from __future__ import annotations

import pytest

from director import issue_filer as IF
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
        run_date="2026-06-13",
        system_summary="System is RED across the board.",
        top_risks=["alpha is negative"],
        action_items=[_item(s) for s in slugs],
    )


class FakeGitHub:
    """Records calls; serves canned issue REST responses. ``existing`` seeds
    already-filed area:director-proposals issues (title/body carry id= markers).
    ``open_issues`` seeds the open-backlog digest source. POSTs are captured."""

    def __init__(self, existing=None, open_issues=None):
        self.calls: list[tuple[str, str]] = []
        self.posted: list[dict] = []
        self._existing = existing or []          # list of issue dicts (state=all, labeled)
        self._open = open_issues or []           # list of issue dicts (state=open)
        self._next_number = 500

    def __call__(self, method, url, token, body=None):
        self.calls.append((method, url))
        if method == "GET" and "labels=area:director-proposals" in url and "state=all" in url:
            # paginate: page 1 returns all, later pages empty
            return (200, list(self._existing)) if "page=1" in url else (200, [])
        if method == "GET" and "state=open" in url:
            return (200, list(self._open)) if "page=1" in url else (200, [])
        if method == "POST" and url.endswith("/issues"):
            self.posted.append(body)
            self._next_number += 1
            return 201, {"number": self._next_number,
                         "html_url": f"https://github.com/cipher813/alpha-engine-config/issues/{self._next_number}"}
        raise AssertionError(f"unexpected call: {method} {url}")


# ── existing_proposal_slugs ──────────────────────────────────────────────────


def test_existing_slugs_extracts_from_title_and_body():
    gh = FakeGitHub(existing=[
        {"title": "[director] Foo (id=foo-slug)", "body": "x"},
        {"title": "[director] Bar", "body": "marker id=bar-slug here"},
    ])
    slugs = IF.existing_proposal_slugs("r/x", "tok", gh_request=gh)
    assert slugs == {"foo-slug", "bar-slug"}


def test_existing_slugs_skips_prs():
    gh = FakeGitHub(existing=[
        {"title": "[director] Foo (id=foo-slug)", "body": "x"},
        {"title": "A PR (id=pr-slug)", "body": "", "pull_request": {"url": "..."}},
    ])
    slugs = IF.existing_proposal_slugs("r/x", "tok", gh_request=gh)
    assert slugs == {"foo-slug"}
    assert "pr-slug" not in slugs


# ── render_issue ─────────────────────────────────────────────────────────────


def test_render_issue_title_carries_id_marker():
    title, body = IF.render_issue(_item("my-slug", priority="P2", owner="predictor"), "2026-06-13")
    assert title == "[director] Do the thing my-slug (id=my-slug)"
    # suggested priority is in the BODY (label stays priority-unset)
    assert "**Suggested priority:** P2" in body
    assert "Owner:** predictor" in body
    assert "## Closes when" in body
    assert "id=my-slug" in body  # idempotency marker echoed in body


# ── file_director_issues ─────────────────────────────────────────────────────


def test_file_issues_happy_path_labels_and_idempotency():
    # one already filed (existing closed/open issue), one brand new
    gh = FakeGitHub(existing=[{"title": "[director] old (id=already)", "body": ""}])
    res = IF.file_director_issues(_plan("already", "brand-new"), "2026-06-13", token="tok", gh_request=gh)
    assert res["status"] == "ok"
    assert res["n_filed"] == 1
    assert res["slugs"] == ["brand-new"]
    # exactly one POST, with both labels and the id-bearing title
    assert len(gh.posted) == 1
    posted = gh.posted[0]
    assert posted["labels"] == ["area:director-proposals", "priority-unset"]
    assert "id=brand-new" in posted["title"]


def test_file_issues_nochange_when_all_filed():
    gh = FakeGitHub(existing=[
        {"title": "(id=a)", "body": ""}, {"title": "(id=b)", "body": ""},
    ])
    res = IF.file_director_issues(_plan("a", "b"), "2026-06-13", token="tok", gh_request=gh)
    assert res["status"] == "nochange"
    assert res["n_filed"] == 0
    assert gh.posted == []


def test_file_issues_raises_on_non_201():
    def gh(method, url, token, body=None):
        if method == "GET":
            return 200, []
        return 422, {"message": "validation failed"}
    with pytest.raises(RuntimeError, match="create issue"):
        IF.file_director_issues(_plan("x"), "2026-06-13", token="tok", gh_request=gh)


def test_file_issues_never_files_duplicate_for_closed_proposal():
    # a slug present on a CLOSED proposal issue is still skipped (state=all)
    gh = FakeGitHub(existing=[{"title": "[director] done (id=closed-one)", "body": "", "state": "closed"}])
    res = IF.file_director_issues(_plan("closed-one"), "2026-06-13", token="tok", gh_request=gh)
    assert res["status"] == "nochange"
    assert gh.posted == []


# ── open_issues_digest ───────────────────────────────────────────────────────


def test_open_issues_digest_titles_and_labels():
    gh = FakeGitHub(open_issues=[
        {"number": 10, "title": "Fix the thing", "labels": [{"name": "P1"}, {"name": "area:predictor"}]},
        {"number": 11, "title": "A PR", "labels": [], "pull_request": {"url": "..."}},
    ])
    digest = IF.open_issues_digest("r/x", "tok", gh_request=gh)
    assert "#10 [P1,area:predictor] Fix the thing" in digest
    assert "A PR" not in digest  # PRs skipped
