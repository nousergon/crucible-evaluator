"""config#6915 — the Director's best-effort tail steps decline when out of time.

config#6904 bounded the two LLM calls. It did not bound what runs around them:
six best-effort steps, four of them GitHub-API-bound, none of which consulted
the clock. Two of those steps MUTATE external state — issue filing creates
issues, loop verification reopens them and sets sticky flags — so being killed
partway through leaves GitHub and the ledger disagreeing about what the run
did. And the two digest fetches run BEFORE the plan call, so an unbounded fetch
there shortens the budget the plan is then quoted against, degrading the
primary deliverable rather than a secondary one.

Every step now checks affordability ONCE for the whole batch and reports a skip
reason naming the budget, so a skip is visible in the handler's return value
rather than inferred from an absent field.
"""

from __future__ import annotations

import pytest

from director import handler as H
from director.budget import STEP_ESTIMATE_S, InvocationBudget


class _Clock:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def __call__(self) -> int:
        return int(self.seconds * 1000)


def _starved() -> InvocationBudget:
    """A budget with almost nothing left — every step is unaffordable."""
    return InvocationBudget(_Clock(1))


def _flush() -> InvocationBudget:
    """A budget that can fund anything."""
    return InvocationBudget(_Clock(3600))


class _Tripwire:
    """Records that it was called — a skipped step must never reach the API."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("the API was called despite an exhausted budget")


def test_every_registered_step_has_an_estimate():
    """A step added without an estimate would silently never be guarded."""
    assert set(STEP_ESTIMATE_S) == {
        "backlog_digest",
        "resolved_digest",
        "loop_verification",
        "digest_email",
        "issue_filing",
        "deploy_rollup",
    }
    assert all(v > 0 for v in STEP_ESTIMATE_S.values())


def test_backlog_digest_skips_and_makes_no_request(monkeypatch):
    tripwire = _Tripwire()
    monkeypatch.setattr(H, "open_issues_digest", tripwire)
    assert H._fetch_backlog_digest_best_effort("tok", budget=_starved()) is None
    assert tripwire.calls == 0


def test_resolved_digest_skips_and_makes_no_request(monkeypatch):
    tripwire = _Tripwire()
    monkeypatch.setattr(H, "recently_closed_proposals_digest", tripwire)
    assert H._fetch_resolved_digest_best_effort("tok", "2026-08-10", budget=_starved()) is None
    assert tripwire.calls == 0


#: An attested cycle. The §2.3a verdict gate is default-deny (config-I7039), so
#: a budget test that omitted it would pass for the wrong reason.
_ATTESTED = {"verdict": "PASS", "present": True, "as_of": {}}


def test_issue_filing_skips_whole_batch_and_files_nothing(monkeypatch):
    tripwire = _Tripwire()
    monkeypatch.setattr(H, "file_director_issues", tripwire)
    monkeypatch.setattr(H, "_issue_filing_enabled", lambda: True)

    # config-I7039: the §2.3a gate is checked BEFORE the budget, so this test
    # must declare an attested cycle or it would measure the verdict gate
    # instead of the budget guard it is named for.
    out = H._file_issues_best_effort(object(), "2026-08-10", "tok", budget=_starved(),
                                     verdict_block=_ATTESTED)

    assert out["director_issues"] == "skipped"
    assert "budget" in out["director_issues_reason"]
    assert tripwire.calls == 0


def test_loop_verification_skips_whole_pass_and_mutates_nothing(monkeypatch):
    tripwire = _Tripwire()
    monkeypatch.setattr(H, "backfill_issue_numbers", tripwire)
    monkeypatch.setattr(H, "verify_and_correct", tripwire)
    ledger = {"items": [{"id": "a", "status": "open"}]}

    out = H._verify_loop_best_effort(ledger, {}, "tok", budget=_starved(),
                                     verdict_block=_ATTESTED)

    assert out["director_loop"] == "skipped"
    assert "budget" in out["director_loop_reason"]
    assert tripwire.calls == 0
    assert ledger == {"items": [{"id": "a", "status": "open"}]}, "ledger was mutated by a skipped pass"


def test_deploy_rollup_skips_and_produces_nothing():
    out = H._run_deploy_success_best_effort(object(), "bucket", "tok", budget=_starved())
    assert out["deploy_success"] == "skipped"
    assert "budget" in out["deploy_success_reason"]


@pytest.mark.parametrize(
    "step",
    ["backlog_digest", "resolved_digest", "loop_verification", "digest_email",
     "issue_filing", "deploy_rollup"],
)
def test_a_healthy_invocation_skips_nothing(step):
    from director.budget import skip_if_unaffordable

    assert skip_if_unaffordable(_flush(), step) is None


def test_an_unbudgeted_caller_is_never_blocked():
    """Outside Lambda there is no clock; nothing may be skipped for time."""
    from director.budget import skip_if_unaffordable

    for step in STEP_ESTIMATE_S:
        assert skip_if_unaffordable(None, step) is None
        assert skip_if_unaffordable(InvocationBudget(None), step) is None


# ---------------------------------------------------------------------------
# Per-request cap, not just per-step
# ---------------------------------------------------------------------------

def test_github_requests_carry_a_default_timeout():
    """`urlopen` with no timeout inherits the global default of None."""
    import inspect

    from director.roadmap_pr import DEFAULT_GH_TIMEOUT_S, _gh_request

    assert DEFAULT_GH_TIMEOUT_S > 0
    assert inspect.signature(_gh_request).parameters["timeout"].default == DEFAULT_GH_TIMEOUT_S
    source = inspect.getsource(_gh_request)
    assert "urlopen(req, timeout=timeout)" in source, (
        "the request must pass its timeout through to urlopen — a timeout "
        "parameter that never reaches the socket bounds nothing"
    )


def test_a_late_invocation_shortens_each_request_below_the_default():
    from director.roadmap_pr import DEFAULT_GH_TIMEOUT_S

    seen = {}

    def _fake(method, url, token, body=None, *, timeout=DEFAULT_GH_TIMEOUT_S):
        seen["timeout"] = timeout
        return 200, {}

    import director.roadmap_pr as R

    original = R._gh_request
    R._gh_request = _fake
    try:
        request = H._budgeted_gh_request(InvocationBudget(_Clock(60)))
        request("GET", "https://api.github.com/x", "tok")
    finally:
        R._gh_request = original

    assert seen["timeout"] < DEFAULT_GH_TIMEOUT_S
    assert seen["timeout"] > 0


def test_an_unbounded_budget_leaves_the_request_helper_untouched():
    from director.roadmap_pr import _gh_request

    assert H._budgeted_gh_request(None) is _gh_request
    assert H._budgeted_gh_request(InvocationBudget(None)) is _gh_request
