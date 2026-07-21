"""Tests for director/loop_verification.py — config#3145, closing the
Director loop: verifying a closed proposal's cited evidence actually
recovered (reopening if not), and escalating carried-over items to the
Decision Queue. No live GitHub/AWS — the transport is a fake ``gh_request``
mirroring tests/test_issue_filer.py's ``FakeGitHub`` convention.
"""

from __future__ import annotations

from director import loop_verification as LV

_CARD = {
    "tiles": {
        "substrate": {"components": [
            {"name": "price_cache_freshness", "status": "RED"},
            {"name": "deploy_success_rate", "status": "GREEN"},
        ]},
        "predictor": {"components": [
            {"name": "momentum_l1_ic", "status": "WATCH"},
        ]},
    },
}


class FakeGitHub:
    """Records calls; serves canned issue GET/PATCH/POST responses keyed by
    issue number. ``issues`` maps number -> {"state": ..., "labels": [...]}."""

    def __init__(self, issues=None, proposal_issues=None):
        self.calls: list[tuple[str, str, dict | None]] = []
        self._issues = issues or {}
        self._proposal_issues = proposal_issues or []
        self.patched_state: dict[int, str] = {}
        self.added_labels: dict[int, list[str]] = {}
        self.comments: dict[int, list[str]] = {}

    def __call__(self, method, url, token, body=None):
        self.calls.append((method, url, body))
        if method == "GET" and "/issues?labels=area:director-proposals" in url:
            return (200, list(self._proposal_issues)) if "page=1" in url else (200, [])
        if method == "GET" and url.rsplit("/", 1)[-1].isdigit():
            number = int(url.rsplit("/", 1)[-1])
            issue = self._issues.get(number)
            return (200, issue) if issue else (404, {})
        if method == "PATCH":
            number = int(url.rsplit("/", 1)[-1])
            if "state" in (body or {}):
                self.patched_state[number] = body["state"]
                self._issues.setdefault(number, {})["state"] = body["state"]
            return 200, {}
        if method == "POST" and url.endswith("/labels"):
            number = int(url.rsplit("/", 2)[-2])
            self.added_labels.setdefault(number, []).extend(body["labels"])
            self._issues.setdefault(number, {}).setdefault("labels", []).extend(
                [{"name": n} for n in body["labels"]]
            )
            return 200, {}
        if method == "POST" and url.endswith("/comments"):
            number = int(url.rsplit("/", 2)[-2])
            self.comments.setdefault(number, []).append(body["body"])
            return 201, {}
        raise AssertionError(f"unexpected call: {method} {url}")


# ── component_status_map / evidence_still_adverse ───────────────────────────


def test_component_status_map_flattens_all_tiles():
    m = LV.component_status_map(_CARD)
    assert m == {
        "price_cache_freshness": "RED",
        "deploy_success_rate": "GREEN",
        "momentum_l1_ic": "WATCH",
    }


def test_evidence_still_adverse_red():
    m = LV.component_status_map(_CARD)
    assert LV.evidence_still_adverse(["price_cache_freshness"], m) == "adverse"


def test_evidence_still_adverse_watch_counts_as_adverse():
    m = LV.component_status_map(_CARD)
    assert LV.evidence_still_adverse(["momentum_l1_ic"], m) == "adverse"


def test_evidence_recovered_when_green():
    m = LV.component_status_map(_CARD)
    assert LV.evidence_still_adverse(["deploy_success_rate"], m) == "recovered"


def test_evidence_unverifiable_when_not_on_card():
    m = LV.component_status_map(_CARD)
    assert LV.evidence_still_adverse(["some_retired_metric"], m) == "unverifiable"


def test_evidence_case_insensitive():
    m = LV.component_status_map(_CARD)
    assert LV.evidence_still_adverse(["PRICE_CACHE_FRESHNESS"], m) == "adverse"


# ── backfill_issue_numbers ───────────────────────────────────────────────────


def test_backfill_fills_missing_by_slug():
    gh = FakeGitHub(proposal_issues=[
        {"title": "[director] Fix cache (id=fix-cache)", "body": "x", "number": 501},
    ])
    items = [{"id": "fix-cache", "issue_number": None}, {"id": "other", "issue_number": 42}]
    n = LV.backfill_issue_numbers(items, repo="r/x", token="tok", gh_request=gh)
    assert n == 1
    assert items[0]["issue_number"] == 501
    assert items[1]["issue_number"] == 42  # untouched — already set


def test_backfill_noop_when_nothing_missing():
    gh = FakeGitHub()
    items = [{"id": "a", "issue_number": 1}]
    assert LV.backfill_issue_numbers(items, repo="r/x", token="tok", gh_request=gh) == 0
    assert gh.calls == []  # never even fetches


# ── verify_and_correct ───────────────────────────────────────────────────────


def _item(id_="i1", *, number=100, evidence=None, carry_count=0, escalated=False,
          status="carried_over", owner="research", title="Do the thing"):
    return {
        "id": id_, "issue_number": number, "evidence": evidence or [],
        "carry_count": carry_count, "escalated": escalated, "status": status,
        "proposed_owner": owner, "title": title,
    }


def test_reopens_closed_issue_still_adverse():
    gh = FakeGitHub(issues={100: {"state": "closed", "labels": []}})
    item = _item(evidence=["price_cache_freshness"])
    result = LV.verify_and_correct([item], _CARD, repo="r/x", token="tok", gh_request=gh)
    assert result["closed_unrecovered"] == 1
    assert result["reopened_issues"] == [100]
    assert gh.patched_state[100] == "open"
    assert "still reads RED/WATCH" in gh.comments[100][0]


def test_closed_and_recovered_is_left_alone():
    gh = FakeGitHub(issues={100: {"state": "closed", "labels": []}})
    item = _item(evidence=["deploy_success_rate"])
    result = LV.verify_and_correct([item], _CARD, repo="r/x", token="tok", gh_request=gh)
    assert result["closed_verified"] == 1
    assert 100 not in gh.patched_state


def test_closed_unverifiable_is_left_alone():
    gh = FakeGitHub(issues={100: {"state": "closed", "labels": []}})
    item = _item(evidence=["some_retired_metric"])
    result = LV.verify_and_correct([item], _CARD, repo="r/x", token="tok", gh_request=gh)
    assert result["closed_unverifiable"] == 1
    assert 100 not in gh.patched_state


def test_escalates_open_item_carried_twice():
    gh = FakeGitHub(issues={100: {"state": "open", "labels": [{"name": "P1"}]}})
    item = _item(carry_count=2)
    result = LV.verify_and_correct([item], _CARD, repo="r/x", token="tok", gh_request=gh)
    assert result["escalated"] == 1
    assert result["escalated_issues"] == [100]
    assert "gate:decision" in gh.added_labels[100]
    assert item["escalated"] is True
    body = gh.comments[100][0]
    assert "**Summary:**" in body and "**Ask:**" in body and "(recommended)" in body


def test_does_not_reescalate_already_escalated_item():
    gh = FakeGitHub(issues={100: {"state": "open", "labels": [{"name": "P1"}]}})
    item = _item(carry_count=3, escalated=True)
    result = LV.verify_and_correct([item], _CARD, repo="r/x", token="tok", gh_request=gh)
    assert result["escalated"] == 0
    assert 100 not in gh.added_labels


def test_does_not_escalate_when_already_gated():
    gh = FakeGitHub(issues={100: {"state": "open", "labels": [{"name": "gate:decision"}]}})
    item = _item(carry_count=2)
    result = LV.verify_and_correct([item], _CARD, repo="r/x", token="tok", gh_request=gh)
    assert result["escalated"] == 0


def test_does_not_escalate_below_threshold():
    gh = FakeGitHub(issues={100: {"state": "open", "labels": []}})
    item = _item(carry_count=1)
    result = LV.verify_and_correct([item], _CARD, repo="r/x", token="tok", gh_request=gh)
    assert result["escalated"] == 0
    assert result["open"] == 1


def test_skips_items_without_issue_number():
    gh = FakeGitHub()
    item = _item(number=None)
    result = LV.verify_and_correct([item], _CARD, repo="r/x", token="tok", gh_request=gh)
    assert result["open"] == 0 and result["closed_unrecovered"] == 0
    assert gh.calls == []


def test_bad_gh_response_is_skipped_not_fatal():
    gh = FakeGitHub(issues={})  # 404 for any lookup
    item = _item(number=999)
    result = LV.verify_and_correct([item], _CARD, repo="r/x", token="tok", gh_request=gh)
    assert result["open"] == 0
    assert result["closed_unrecovered"] == 0
