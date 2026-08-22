"""config#6904 — every Director LLM call is quoted against the time left.

The Director makes two LLM calls in one invocation: the plan (``director/
agent.py``) and the Phase-G retro judge (``director/retro.py``), each with one
internal retry. Their static ceilings sum to ``2×340 + 2×120 = 920s`` against
a **900s** function timeout (``infrastructure/deploy.sh::DIRECTOR_TIMEOUT``),
before any S3 write — and until this change the only thing asserting the
relationship was a code comment that had the arithmetic wrong.

That is the shape that produced the 2026-08-08 failure one ceiling lower: a
healthy in-flight call aborted at the client timeout, the SDK retried, and the
retry hit the wall (``Status: timeout``, config#6050/config#6747). A Lambda
killed at the wall writes no artifact and logs no cause, which is why three
weeks of Director silence went unnoticed.

These tests pin the fix: the ceilings stay individually affordable, and no
call site may go back to handing the client a bare literal.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

from director.budget import (
    DEFAULT_RESERVE_S,
    MIN_VIABLE_CALL_S,
    BudgetExhausted,
    InvocationBudget,
)

_REPO = pathlib.Path(__file__).resolve().parents[1]
_DEPLOY = _REPO / "infrastructure" / "deploy.sh"

# The `docker-image-tests` job mounts only tests/ into the shipped image, which
# deliberately does not carry infrastructure/ (see .github/workflows/ci.yml —
# it already --ignores the other deploy.sh-reading suites for the same reason).
# The two assertions below are repo-shape assertions, not package assertions, so
# they are binding in the `test` job that runs against a full checkout and
# inapplicable inside the image. Everything else in this file exercises the
# image's own packages and must keep running there.
_repo_only = pytest.mark.skipif(
    not _DEPLOY.exists(),
    reason=(
        "infrastructure/deploy.sh is not present — running inside the shipped "
        "image, where this repo-shape assertion does not apply. It is enforced "
        "by the repo-checkout `test` job."
    ),
)


def _deploy_director_timeout() -> int:
    """The function timeout the deploy script actually sets."""
    match = re.search(r"^DIRECTOR_TIMEOUT=(\d+)", _DEPLOY.read_text(), re.MULTILINE)
    assert match, (
        "DIRECTOR_TIMEOUT is no longer assigned in infrastructure/deploy.sh — "
        "this test can no longer see the ceiling the call budgets must fit inside"
    )
    return int(match.group(1))


class _Clock:
    """Stand-in for context.get_remaining_time_in_millis."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def __call__(self) -> int:
        return int(self.seconds * 1000)


class _Context:
    def __init__(self, seconds: float) -> None:
        self.get_remaining_time_in_millis = _Clock(seconds)


# ---------------------------------------------------------------------------
# The invariant the comment used to assert and nothing enforced
# ---------------------------------------------------------------------------

@_repo_only
def test_each_ceiling_is_individually_affordable():
    """Each call's funded worst case must fit the function timeout on its own.

    "Funded worst case" is the call's per-attempt ceiling times the attempts it
    is quoted for, plus any time it is required to leave behind for LLM work
    later in the same invocation. Both terms are part of the contract, so both
    appear here — before 2026-08-22 the plan's downstream obligation to the
    retro judge was expressed as ``attempts=2``, which made this test pass by
    halving the plan call rather than by naming what the halving paid for
    (alpha-engine-config-I7311).
    """
    from director.agent import DIRECTOR_PLAN_CEILING_S
    from director.budget import RETRO_JUDGE_RESERVE_S
    from director.retro import RETRO_JUDGE_CEILING_S

    fn_timeout = _deploy_director_timeout()
    for label, ceiling, attempts, downstream in (
        ("director-plan", DIRECTOR_PLAN_CEILING_S, 1, RETRO_JUDGE_RESERVE_S),
        ("director-retro-judge", RETRO_JUDGE_CEILING_S, 2, 0.0),
    ):
        worst = ceiling * attempts + downstream
        assert worst + DEFAULT_RESERVE_S <= fn_timeout, (
            f"{label}: {ceiling}s × {attempts} attempt(s) + {downstream}s reserved "
            f"for downstream LLM work + {DEFAULT_RESERVE_S}s write reserve "
            f"= {worst + DEFAULT_RESERVE_S}s exceeds the {fn_timeout}s function "
            f"timeout on its own. No runtime budget can rescue a ceiling that "
            f"cannot fit even when it runs first."
        )


@_repo_only
def test_the_ceilings_do_not_fit_together_which_is_why_the_budget_exists():
    """Documents the measured overrun this change addresses.

    If this ever stops being true the derivation is still correct, but the
    justification in `director/budget.py` needs updating rather than silently
    describing a condition that no longer holds.
    """
    from director.agent import DIRECTOR_PLAN_CEILING_S
    from director.retro import RETRO_JUDGE_CEILING_S

    combined = (DIRECTOR_PLAN_CEILING_S + RETRO_JUDGE_CEILING_S) * 2
    fn_timeout = _deploy_director_timeout()
    assert combined > fn_timeout, (
        f"the two ceilings now sum to {combined}s inside a {fn_timeout}s function "
        f"timeout, so the 'static budgets cannot compose' rationale recorded in "
        f"director/budget.py no longer describes reality — update that docstring "
        f"(the runtime derivation stays correct either way)"
    )


# ---------------------------------------------------------------------------
# No call site may go back to a bare literal
# ---------------------------------------------------------------------------

def _is_quote_call(node) -> bool:
    return isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "quote"


def _quote_bound_names(tree) -> set[str]:
    """Local names assigned directly from a ``budget.quote(...)`` call.

    A call site is allowed to bind the quote to a name and pass the name — the
    Director does, because the same quoted number is ALSO the retry loop's
    ``attempt_cost_s``, and computing it twice would let the two drift. What is
    forbidden is a literal, so the test follows the binding rather than
    demanding the call be inlined.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_quote_call(node.value):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    names.add(tgt.id)
    return names


def _llm_client_calls():
    for path in sorted((_REPO / "director").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        bound = _quote_bound_names(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name == "LLMClient":
                yield path.name, (node, bound)


def test_there_are_llm_client_call_sites_to_check():
    assert list(_llm_client_calls()), (
        "no LLMClient(...) construction found under director/ — if the client "
        "moved, retarget this test rather than leaving it vacuously green"
    )


@pytest.mark.parametrize(
    "filename,node",
    [(f, n) for f, n in _llm_client_calls()],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_every_llm_client_timeout_is_a_budget_quote(filename, node):
    """`timeout=` must come from `budget.quote(...)`, never a literal."""
    call, quote_bound_names = node
    timeout = next((kw.value for kw in call.keywords if kw.arg == "timeout"), None)
    assert timeout is not None, (
        f"{filename}: LLMClient constructed with no explicit timeout — it would "
        f"inherit the krepis default (180s), which is the exact regression "
        f"config#6050 fixed"
    )
    quoted = _is_quote_call(timeout) or (
        isinstance(timeout, ast.Name) and timeout.id in quote_bound_names
    )
    assert quoted, (
        f"{filename}: LLMClient timeout is a static value. It must be quoted "
        f"through an InvocationBudget so a late invocation shortens or declines "
        f"the call instead of being killed at the function timeout (config#6904)."
    )


# ---------------------------------------------------------------------------
# Budget arithmetic
# ---------------------------------------------------------------------------

def test_unbounded_budget_returns_the_ceiling_unchanged():
    """Outside Lambda nothing changes — the ceilings apply exactly as before."""
    budget = InvocationBudget.from_context(None)
    assert not budget.bounded
    assert budget.remaining() == float("inf")
    assert budget.quote("plan", 340.0, attempts=2) == 340.0


def test_a_healthy_invocation_still_gets_the_full_ceiling():
    budget = InvocationBudget.from_context(_Context(900))
    assert budget.quote("plan", 340.0, attempts=2) == 340.0


def test_budget_covers_every_attempt_not_just_the_first():
    """Sizing a retried call by one attempt's ceiling is the original defect."""
    budget = InvocationBudget(_Clock(400), reserve_s=40.0)
    # 360s usable, two attempts → 180s each, not 340.
    assert budget.quote("plan", 340.0, attempts=2) == pytest.approx(180.0)
    assert budget.quote("plan", 340.0, attempts=1) == pytest.approx(340.0)


def test_reserve_is_held_back_for_the_writes():
    budget = InvocationBudget(_Clock(200), reserve_s=DEFAULT_RESERVE_S)
    assert budget.remaining() == pytest.approx(200 - DEFAULT_RESERVE_S)


def test_a_doomed_call_is_declined_rather_than_issued():
    budget = InvocationBudget(_Clock(DEFAULT_RESERVE_S + 20), reserve_s=DEFAULT_RESERVE_S)
    with pytest.raises(BudgetExhausted) as exc:
        budget.quote("director-retro-judge", 120.0, attempts=2)
    assert "floor" in str(exc.value)
    assert not budget.can_afford(MIN_VIABLE_CALL_S * 2)


def test_an_exhausted_budget_never_reports_negative_time():
    budget = InvocationBudget(_Clock(1), reserve_s=DEFAULT_RESERVE_S)
    assert budget.remaining() == 0.0


def test_quote_is_taken_fresh_each_time():
    """The clock is a callable, not a snapshot — the retro is quoted late."""
    clock = _Clock(900)
    budget = InvocationBudget(clock, reserve_s=DEFAULT_RESERVE_S)
    assert budget.quote("plan", 340.0, attempts=2) == 340.0
    clock.seconds = 200  # the plan call consumed most of the invocation
    assert budget.quote("judge", 120.0, attempts=2) == pytest.approx(
        (200 - DEFAULT_RESERVE_S) / 2
    )


# ---------------------------------------------------------------------------
# The retro declines rather than failing
# ---------------------------------------------------------------------------

def test_retro_reports_budget_exhaustion_as_a_skip_not_an_error(monkeypatch):
    """A grade we cannot finish is a skip; an error means the judge misbehaved."""
    from director import handler as handler_mod
    from director import retro as retro_mod

    monkeypatch.setattr(handler_mod, "_load_prior_plan", lambda *a, **k: {"run_date": "2026-08-03"})

    def _boom(*_args, **_kwargs):
        raise BudgetExhausted("director-retro-judge: 12s per attempt is all ...")

    monkeypatch.setattr(retro_mod, "grade_prior_plan", _boom)

    out = handler_mod._run_retro_best_effort(
        object(), "bucket", "2026-08-10", {}, budget=InvocationBudget(_Clock(50))
    )
    assert out["retro"] == "skipped"
    assert "budget" in out["retro_reason"]
    assert "retro_error" not in out
