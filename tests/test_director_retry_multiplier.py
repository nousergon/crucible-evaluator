"""config#7126 — retry loops MULTIPLY, and only one of them can see the deadline.

`director/budget.py` quotes every LLM call against the time actually left, and
`tests/test_director_invocation_budget.py` pins that no call site may hand the
client a bare literal. Both were correct and both were defeated by an attempt
count nothing stated.

Measured on the deployed code, 2026-08-13:

  * `LLMClient(max_retries=1)`  -> 2 transport attempts per model call
  * `client.structured()`       -> `attempts=2` by DEFAULT, never passed by the
                                   Director, i.e. a body-level corrective retry
  * `_invoke_with_retry`        -> up to 3 more

so one `build_action_plan` could issue **12** timeout-bounded model calls of up
to 340s each, inside a function whose timeout cannot be raised: 900s is AWS
Lambda's service maximum. `budget.quote(..., attempts=2)` funded two of them.

That is why raising the ceiling 300s -> 900s moved the cliff instead of removing
it, which is the exact outcome `director/budget.py`'s own docstring warns about.

These tests pin the correction: one `invoke()` is one timeout-bounded model
call, and the single remaining retry loop refuses to start an attempt the
invocation cannot fund.

Derivation of the 340s ceiling these budgets are built on — five UNCENSORED
plan calls against the live router edge, same registry `max_tokens`,
Lambda-shaped prompt (11,212 prompt tokens, both backlog digests), all
`finish_reason: stop`: 170.1s / 183.0s / 194.9s / 199.5s / 228.0s, drawing
18.2k-23.3k completion tokens at 99-109 tok/s. 340 = 1.49 x the slowest, i.e.
`sf-pipeline-policy.md` §4's p95 x 1.5, over samples allowed to finish. Every
figure in this file's history before that was censored at a wall.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from director.agent import (
    _CLIENT_MAX_RETRIES,
    _MAX_RETRIES,
    _RETRYABLE,
    _STRUCTURED_ATTEMPTS,
    DIRECTOR_PLAN_CEILING_S,
    _invoke_with_retry,
)
from director.budget import DEFAULT_RESERVE_S, InvocationBudget

_REPO = pathlib.Path(__file__).resolve().parents[1]


class _Clock:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def __call__(self) -> int:
        return int(self.seconds * 1000)


# ---------------------------------------------------------------------------
# One invoke() is one timeout-bounded model call
# ---------------------------------------------------------------------------

def test_the_inner_retry_loops_are_collapsed():
    """Both krepis loops must be single-attempt, or the quote under-counts."""
    assert _CLIENT_MAX_RETRIES == 0, (
        "the SDK transport retry is back. Every transport retry multiplies the "
        "per-attempt timeout by another whole ceiling, and the SDK cannot see "
        "the invocation deadline — retry in _invoke_with_retry instead."
    )
    assert _STRUCTURED_ATTEMPTS == 1, (
        "krepis' body-level corrective retry is back. It defaults to 2, so "
        "inheriting it silently doubles every budget the Director quotes."
    )


def _structured_calls():
    """Every `.structured(...)` call under director/."""
    for path in sorted((_REPO / "director").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "structured":
                yield path.name, node


def _llm_client_calls():
    for path in sorted((_REPO / "director").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and (
                getattr(node.func, "id", None) == "LLMClient"
                or getattr(node.func, "attr", None) == "LLMClient"
            ):
                yield path.name, node


def test_there_are_call_sites_to_check():
    assert list(_structured_calls()), "no .structured(...) call found under director/"
    assert list(_llm_client_calls()), "no LLMClient(...) construction found under director/"


@pytest.mark.parametrize(
    "filename,node", [(f, n) for f, n in _structured_calls()],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_every_structured_call_states_its_attempt_count(filename, node):
    """`attempts=` must be explicit — the default is 2 and it is invisible."""
    attempts = next((kw.value for kw in node.keywords if kw.arg == "attempts"), None)
    assert attempts is not None, (
        f"{filename}: .structured() called without `attempts=`. krepis defaults "
        f"to 2, so this call site can make twice as many timeout-bounded model "
        f"calls as its budget was quoted for (config#7126)."
    )
    assert isinstance(attempts, ast.Name) and attempts.id == "_STRUCTURED_ATTEMPTS", (
        f"{filename}: `attempts=` is a local literal. It must reference the "
        f"shared `_STRUCTURED_ATTEMPTS` so the multiplier stays stated in one "
        f"place and cannot drift between the two Director call sites."
    )


@pytest.mark.parametrize(
    "filename,node", [(f, n) for f, n in _llm_client_calls()],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_every_client_states_its_transport_retry_count(filename, node):
    max_retries = next((kw.value for kw in node.keywords if kw.arg == "max_retries"), None)
    assert max_retries is not None, (
        f"{filename}: LLMClient constructed with no explicit `max_retries` — it "
        f"would inherit krepis' default of 3, i.e. 4 timeout-bounded attempts "
        f"per model call."
    )
    assert isinstance(max_retries, ast.Name) and max_retries.id == "_CLIENT_MAX_RETRIES", (
        f"{filename}: `max_retries=` is a literal. It must reference the shared "
        f"`_CLIENT_MAX_RETRIES` — the retry belongs in the budget-aware loop."
    )


# ---------------------------------------------------------------------------
# The remaining loop refuses to start what it cannot fund
# ---------------------------------------------------------------------------

class _FailingLLM:
    """Raises a retryable error every time, counting attempts."""

    def __init__(self, message: str = "Request timed out.", *, attempt_cost_s=None):
        self.calls = 0
        self._message = message
        if attempt_cost_s is not None:
            self.attempt_cost_s = attempt_cost_s

    def invoke(self, messages):
        self.calls += 1
        raise RuntimeError(self._message)


def test_a_client_timeout_is_retryable():
    """`openai.APITimeoutError` stringifies as 'Request timed out.'

    It matches neither 'timeout' nor 'connection', so before config#7126 the
    one failure the timeout chain exists to bound was the one this loop did not
    retry. The SDK's own retry hid that — and config#7126 removes the SDK's.
    """
    assert any(t in "request timed out." for t in _RETRYABLE), (
        "'Request timed out.' no longer matches any _RETRYABLE token — a client "
        "timeout would now propagate on the first attempt with no retry at all, "
        "since _CLIENT_MAX_RETRIES is 0."
    )


def test_an_unbounded_invocation_retries_as_before():
    """Outside Lambda nothing changes."""
    llm = _FailingLLM(attempt_cost_s=DIRECTOR_PLAN_CEILING_S)
    with pytest.raises(RuntimeError):
        _invoke_with_retry(llm, [], budget=InvocationBudget.from_context(None))
    assert llm.calls == _MAX_RETRIES


def test_a_late_invocation_declines_the_retry_instead_of_being_killed():
    """The gate: one attempt's worth left, so no second attempt is started."""
    # Exactly one full-ceiling attempt's worth after the write reserve — so the
    # first attempt is affordable and a second (ceiling + backoff) is not.
    clock = _Clock(DEFAULT_RESERVE_S + DIRECTOR_PLAN_CEILING_S)
    budget = InvocationBudget(clock, reserve_s=DEFAULT_RESERVE_S)
    llm = _FailingLLM(attempt_cost_s=DIRECTOR_PLAN_CEILING_S)
    with pytest.raises(RuntimeError, match="timed out"):
        _invoke_with_retry(llm, [], budget=budget)
    assert llm.calls == 1, (
        "a retry was started on an invocation that could not fund it — that is "
        "the shape that gets the Lambda killed at the wall with no artifact "
        "and no cause"
    )


def test_a_healthy_invocation_still_gets_its_retries():
    clock = _Clock(DEFAULT_RESERVE_S + DIRECTOR_PLAN_CEILING_S * 10)
    budget = InvocationBudget(clock, reserve_s=DEFAULT_RESERVE_S)
    llm = _FailingLLM(attempt_cost_s=DIRECTOR_PLAN_CEILING_S)
    with pytest.raises(RuntimeError):
        _invoke_with_retry(llm, [], budget=budget)
    assert llm.calls == _MAX_RETRIES


def test_an_adapter_without_a_declared_cost_is_not_gated():
    """Injected test doubles keep the old behaviour rather than silently
    losing their retries to a budget they never opted into."""
    clock = _Clock(DEFAULT_RESERVE_S + 1)
    budget = InvocationBudget(clock, reserve_s=DEFAULT_RESERVE_S)
    llm = _FailingLLM()  # no attempt_cost_s
    with pytest.raises(RuntimeError):
        _invoke_with_retry(llm, [], budget=budget)
    assert llm.calls == _MAX_RETRIES


def test_a_non_retryable_error_is_not_retried():
    llm = _FailingLLM("schema validation failed", attempt_cost_s=DIRECTOR_PLAN_CEILING_S)
    with pytest.raises(RuntimeError, match="schema validation"):
        _invoke_with_retry(llm, [], budget=InvocationBudget.from_context(None))
    assert llm.calls == 1


# ---------------------------------------------------------------------------
# Both adapters declare what an attempt costs
# ---------------------------------------------------------------------------

def test_both_director_adapters_declare_an_attempt_cost():
    from director.agent import _KrepisStructuredDirector
    from director.retro import RETRO_JUDGE_CEILING_S, _KrepisStructuredJudge

    assert _KrepisStructuredDirector.attempt_cost_s == DIRECTOR_PLAN_CEILING_S
    assert _KrepisStructuredJudge.attempt_cost_s == RETRO_JUDGE_CEILING_S


def test_the_plans_worst_case_still_leaves_the_retro_a_viable_quote():
    """The primary deliverable's worst case must DEGRADE the retro, not kill it.

    900s is AWS Lambda's service maximum, so the two ceilings deliberately do
    not fit together at full size — that is the condition `director/budget.py`
    exists to absorb, and `test_director_invocation_budget.py` pins it. What
    must hold is the shape of the absorption: after the plan has spent its
    funded worst case (two full-ceiling attempts), the retro is still quoted
    ABOVE the viability floor, so the Director degrades to a shorter grade
    rather than declining one outright.
    """
    from director.budget import MIN_VIABLE_CALL_S
    from director.retro import RETRO_JUDGE_CEILING_S

    lambda_max = 900  # AWS service maximum for a Lambda function timeout
    plan_worst = DIRECTOR_PLAN_CEILING_S * 2
    left_for_retro = _Clock(lambda_max - plan_worst)
    budget = InvocationBudget(left_for_retro, reserve_s=DEFAULT_RESERVE_S)
    quoted = budget.call_timeout(RETRO_JUDGE_CEILING_S, attempts=2)
    assert quoted >= MIN_VIABLE_CALL_S, (
        f"after the plan's funded worst case ({plan_worst}s) the retro judge is "
        f"quoted {quoted:.0f}s per attempt, below the {MIN_VIABLE_CALL_S}s "
        f"viability floor — the Director would decline its self-grade on every "
        f"slow week rather than shortening it. Reduce a ceiling or the reserve."
    )
