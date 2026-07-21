"""Tests for director/issue_filer.py — Phase H issue-filing channel (config#978).

Covers idempotency by ActionItem.id across existing area:director-proposals
issues (open OR closed), the issue render (title id-marker + body discipline +
suggested-P# labeling), the file orchestrator (incl. the skipped_unscoped
pre-activation path) against a mocked GitHub transport, and the open-issue
backlog digest.
"""

from __future__ import annotations

import pytest

from director import issue_filer as IF
from director.schema import ActionItem, DirectorWeeklyActionPlan


def _item(slug: str, *, priority="P1", owner="research", title=None, evidence=None, confidence=70) -> ActionItem:
    return ActionItem(
        id=slug,
        title=title or f"Do the thing {slug}",
        rationale=f"Because metric X is bad for {slug}",
        evidence=evidence if evidence is not None else ["metrics.json::sharpe_ratio", "research_tile"],
        proposed_owner=owner,
        priority=priority,
        horizon="this_week",
        suggested_change_type="investigation",
        confidence=confidence,
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

    def __init__(self, existing=None, open_issues=None, closed=None):
        self.calls: list[tuple[str, str]] = []
        self.posted: list[dict] = []
        self._existing = existing or []          # list of issue dicts (state=all, labeled)
        self._open = open_issues or []           # list of issue dicts (state=open)
        self._closed = closed or []              # list of issue dicts (state=closed, labeled)
        self._next_number = 500

    def __call__(self, method, url, token, body=None):
        self.calls.append((method, url))
        if method == "GET" and "labels=area:director-proposals" in url and "state=all" in url:
            # paginate: page 1 returns all, later pages empty
            return (200, list(self._existing)) if "page=1" in url else (200, [])
        if method == "GET" and "state=closed" in url and "labels=area:director-proposals" in url:
            return (200, list(self._closed)) if "page=1" in url else (200, [])
        if method == "GET" and "state=open" in url:
            return (200, list(self._open)) if "page=1" in url else (200, [])
        if method == "POST" and url.endswith("/issues"):
            self.posted.append(body)
            self._next_number += 1
            return 201, {"number": self._next_number,
                         "html_url": f"https://github.com/nousergon/alpha-engine-config/issues/{self._next_number}"}
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
    # the Director's suggested priority is restated in the body (and applied as the
    # live P# label by file_director_issues)
    assert "**Director-assigned priority:** P2" in body
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
    # exactly one POST, labeled with the proposal tag + the item's suggested P#
    # (default _item priority=P1), and the id-bearing title
    assert len(gh.posted) == 1
    posted = gh.posted[0]
    assert posted["labels"] == ["area:director-proposals", "P1"]
    assert "id=brand-new" in posted["title"]


def test_file_issues_applies_director_suggested_priority_label():
    # the live priority label reflects each item's OWN suggested P#
    gh = FakeGitHub()
    plan = DirectorWeeklyActionPlan(
        run_date="2026-06-13",
        system_summary="x",
        top_risks=["r"],
        action_items=[_item("crit", priority="P0"), _item("low", priority="P3")],
    )
    IF.file_director_issues(plan, "2026-06-13", token="tok", gh_request=gh)
    labels_by_title = {p["title"]: p["labels"] for p in gh.posted}
    assert any("id=crit" in t and "P0" in lbls for t, lbls in labels_by_title.items())
    assert any("id=low" in t and "P3" in lbls for t, lbls in labels_by_title.items())
    assert all("priority-unset" not in lbls for lbls in labels_by_title.values())


def test_file_issues_skipped_unscoped_on_403_not_raise():
    # pre-activation: PAT lacks issues:write -> POST 403. Recorded skip, not raise,
    # and no items recorded as filed (channel self-activates on token re-scope).
    def gh(method, url, token, body=None):
        if method == "GET":
            return 200, []
        return 403, {"message": "Resource not accessible by personal access token"}
    res = IF.file_director_issues(_plan("x", "y"), "2026-06-13", token="tok", gh_request=gh)
    assert res["status"] == "skipped_unscoped"
    assert res["n_filed"] == 0
    assert "issues:write" in res["reason"]


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


# ── recently_closed_proposals_digest ─────────────────────────────────────────


def test_recently_closed_digest_includes_within_window_excludes_old():
    # run_date 2026-06-20, 8-week window → cutoff 2026-04-25.
    gh = FakeGitHub(closed=[
        {"number": 1168, "title": "[director] Validate CIO skill (id=validate-cio-selection-skill-metric)",
         "closed_at": "2026-06-23T00:00:00Z"},
        {"number": 900, "title": "[director] Ancient resolved thing (id=old-slug)",
         "closed_at": "2026-03-01T00:00:00Z"},   # older than cutoff → excluded
    ])
    digest = IF.recently_closed_proposals_digest("r/x", "tok", run_date="2026-06-20", gh_request=gh)
    assert "#1168 (closed 2026-06-23)" in digest
    assert "validate-cio-selection-skill-metric" in digest
    assert "#900" not in digest          # outside the window
    assert "old-slug" not in digest


def test_recently_closed_digest_skips_prs_and_handles_bad_date():
    gh = FakeGitHub(closed=[
        {"number": 7, "title": "A PR (id=pr-slug)", "closed_at": "2026-06-23T00:00:00Z",
         "pull_request": {"url": "..."}},
    ])
    digest = IF.recently_closed_proposals_digest("r/x", "tok", run_date="2026-06-20", gh_request=gh)
    assert digest == ""               # only a PR present → nothing
    # Unparseable run_date → empty digest, no raise (best-effort).
    assert IF.recently_closed_proposals_digest("r/x", "tok", run_date="not-a-date", gh_request=gh) == ""


# ── parse_evidence_from_body / find_reconfirm_match ──────────────────────────


def _closed_proposal(number, evidence, *, closed_at="2026-06-24T00:00:00Z", title="[director] prior (id=old-slug)"):
    """A structured closed-issue dict carrying a rendered ``## Evidence`` section,
    as ``fetch_recently_closed_proposals``/``FakeGitHub`` would return it."""
    _, body = IF.render_issue(_item("old-slug", evidence=evidence), "2026-06-01")
    return {"number": number, "title": title, "body": body, "closed_at": closed_at}


def test_parse_evidence_from_body_recovers_rendered_evidence():
    _, body = IF.render_issue(_item("x", evidence=["a::b", "tile_c"]), "2026-06-01")
    assert IF.parse_evidence_from_body(body) == {"a::b", "tile_c"}


def test_parse_evidence_from_body_empty_on_no_section_or_placeholder():
    assert IF.parse_evidence_from_body("") == set()
    assert IF.parse_evidence_from_body("no evidence heading here") == set()
    _, body = IF.render_issue(_item("x", evidence=[]), "2026-06-01")
    assert IF.parse_evidence_from_body(body) == set()  # "see report card" placeholder


def test_find_reconfirm_match_on_evidence_overlap():
    prior = _closed_proposal(1168, ["metrics.json::cio_selection_skill", "metrics.json::research_composite_ic"])
    new_item = _item("validate-reliability-low-research-metrics",
                      evidence=["metrics.json::cio_selection_skill", "metrics.json::research_composite_ic"])
    matches = IF.find_reconfirm_match(new_item, [prior])
    assert matches == [prior]


def test_find_reconfirm_match_no_match_on_disjoint_evidence():
    prior = _closed_proposal(1168, ["metrics.json::unrelated_metric"])
    new_item = _item("fresh", evidence=["metrics.json::cio_selection_skill"])
    assert IF.find_reconfirm_match(new_item, [prior]) == []


def test_find_reconfirm_match_no_evidence_never_matches():
    prior = _closed_proposal(1168, ["metrics.json::cio_selection_skill"])
    new_item = _item("no-evidence", evidence=[])
    assert IF.find_reconfirm_match(new_item, [prior]) == []


# ── render_reconfirm_issue ────────────────────────────────────────────────────


def test_render_reconfirm_issue_links_prior_and_caps_priority_confidence():
    item = _item("validate-reliability-low-research-metrics", priority="P1", confidence=80,
                  evidence=["metrics.json::cio_selection_skill"])
    prior = [{"number": 1061, "closed_at": "2026-06-14T00:00:00Z", "title": "[director] first pass"},
             {"number": 1168, "closed_at": "2026-06-24T00:00:00Z", "title": "[director] second pass"}]
    title, body, priority = IF.render_reconfirm_issue(item, "2026-07-03", prior)
    assert "Re-confirm" in title
    assert "id=validate-reliability-low-research-metrics" in title
    assert "#1061" in title and "#1168" in title
    assert priority == "P2"  # capped, P1 -> P2
    assert "#1061" in body and "#1168" in body
    assert "Confidence:** 50/100" in body  # 80 - 30
    assert "reconfirm_of=#1061, #1168" in body


def test_render_reconfirm_issue_caps_confidence_at_floor():
    item = _item("x", confidence=20)
    title, body, priority = IF.render_reconfirm_issue(item, "2026-07-03", [{"number": 1}])
    assert "Confidence:** 10/100" in body  # floor, not negative


def test_render_reconfirm_issue_never_escalates_p2_p3():
    for original in ("P2", "P3"):
        item = _item("x", priority=original)
        _, _, priority = IF.render_reconfirm_issue(item, "2026-07-03", [{"number": 1}])
        assert priority == original


# ── file_director_issues: end-to-end re-confirm dedup (config#2307) ─────────


def test_file_issues_reconfirms_instead_of_fresh_p1_for_prior_closed_evidence():
    """Regression test for config#2307: simulate the exact failure mode that let
    config#1639 re-file as a NEW P1 investigation of the same metric pair
    config#1061 (closed 2026-06-14) and config#1168 (closed 2026-06-24) already
    validated twice as 'noisy but sound, no action'. The new item's slug is
    DELIBERATELY DIFFERENT from any prior slug (the LLM names slugs freely) so
    the exact-slug idempotency skip alone would NOT catch it — only the
    evidence-overlap check does. Assert the Director does NOT file a fresh P1;
    it must file the lower-confidence re-confirm variant referencing #1168."""
    prior_evidence = ["metrics.json::cio_selection_skill", "metrics.json::research_composite_ic"]
    gh = FakeGitHub(closed=[_closed_proposal(1168, prior_evidence, closed_at="2026-06-24T00:00:00Z",
                                              title="[director] Validate reliability-LOW cio_selection_skill + "
                                                    "research_composite_ic (id=validate-cio-selection-skill-metric)")])
    new_item = _item(
        "validate-reliability-low-research-metrics",  # NEW slug, distinct from the #1168 slug
        priority="P1", confidence=75, evidence=prior_evidence,
    )
    plan = DirectorWeeklyActionPlan(
        run_date="2026-07-03", system_summary="x", top_risks=["r"], action_items=[new_item],
    )
    res = IF.file_director_issues(plan, "2026-07-03", token="tok", gh_request=gh)

    assert res["status"] == "ok"
    assert res["n_filed"] == 1
    assert len(gh.posted) == 1
    posted = gh.posted[0]
    # NOT filed at the original suggested P1 — capped to P2 as a re-confirm.
    assert posted["labels"] == ["area:director-proposals", "P2"]
    assert "Re-confirm" in posted["title"]
    assert "#1168" in posted["title"]
    assert "#1168" in posted["body"]
    assert res["issues"][0]["reconfirm_of"] == [1168]


def test_file_issues_files_fresh_p1_when_no_prior_closed_evidence_matches():
    # sanity check: an item with genuinely new evidence still files at full priority.
    gh = FakeGitHub(closed=[_closed_proposal(1168, ["metrics.json::unrelated_metric"])])
    new_item = _item("brand-new-investigation", priority="P1",
                      evidence=["metrics.json::totally_different_metric"])
    plan = DirectorWeeklyActionPlan(
        run_date="2026-07-03", system_summary="x", top_risks=["r"], action_items=[new_item],
    )
    res = IF.file_director_issues(plan, "2026-07-03", token="tok", gh_request=gh)
    assert res["status"] == "ok"
    posted = gh.posted[0]
    assert posted["labels"] == ["area:director-proposals", "P1"]
    assert "Re-confirm" not in posted["title"]
    assert res["issues"][0]["reconfirm_of"] == []


# ── slug_issue_number_map (config#3145 backfill source) ─────────────────────


def test_slug_issue_number_map_extracts_from_title_and_body():
    gh = FakeGitHub(existing=[
        {"title": "[director] Foo (id=foo-slug)", "body": "x", "number": 601},
        {"title": "[director] Bar", "body": "marker id=bar-slug here", "number": 602},
    ])
    m = IF.slug_issue_number_map("r/x", "tok", gh_request=gh)
    assert m == {"foo-slug": 601, "bar-slug": 602}


def test_slug_issue_number_map_skips_prs():
    gh = FakeGitHub(existing=[
        {"title": "[director] Foo (id=foo-slug)", "body": "x", "number": 601,
         "pull_request": {"url": "..."}},
    ])
    assert IF.slug_issue_number_map("r/x", "tok", gh_request=gh) == {}
