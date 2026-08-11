"""Per-invocation time budget for the Director Lambda (config#6904).

The Director makes two LLM calls in one invocation — the plan
(``director/agent.py``, ``timeout=340, max_retries=1``) and the Phase-G retro
judge (``director/retro.py``, ``timeout=120, max_retries=1``) — plus GitHub
API fetches, S3 writes, a ledger upsert, an email and issue filing. Every one
of those budgets is a static literal, and the relationship between them and
the function's own ``Timeout`` (``infrastructure/deploy.sh::DIRECTOR_TIMEOUT``)
is asserted only in a code comment.

That comment does not hold: ``2×340 + 2×120 = 920s`` against a **900s**
function timeout, before a single S3 write. The 2026-08-08 weekly run died
exactly this way one ceiling lower — a healthy in-flight call aborted, retried,
and the retry hit the wall at ``Status: timeout`` (config#6050, config#6747).
Raising the ceiling moved the cliff; it did not remove it, because nothing
derives the call budgets from the time actually left.

This module makes the remaining time the source of truth. A call is given the
smaller of its static ceiling and what the invocation can still afford, and
work that cannot fit is **skipped with a reason** rather than started and
killed mid-flight — a Lambda killed at the wall writes nothing and logs no
cause, which is what made three weeks of Director silence invisible.

Outside Lambda (tests, local runs) there is no context and no deadline, so the
budget is unbounded and every static ceiling applies exactly as before.
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)

# Time held back from every quote for the work that is NOT the LLM call:
# S3 puts (plan, ledger, retro, substrate rollup), the GitHub issue filing and
# the digest email. Measured on healthy 2026-07 invocations, that tail runs in
# ~20-30s; 45s carries it with headroom without materially shrinking a call.
DEFAULT_RESERVE_S = 45.0

# A call quoted less than this cannot plausibly complete, so issuing it only
# guarantees a timeout inside the timeout. Below the floor we decline instead.
MIN_VIABLE_CALL_S = 30.0


class InvocationBudget:
    """What this invocation can still afford, in seconds.

    ``remaining_ms`` is normally ``context.get_remaining_time_in_millis`` —
    passed as the callable, not its value, because the answer changes as the
    invocation proceeds and every quote must be taken fresh.
    """

    def __init__(
        self,
        remaining_ms: Callable[[], int] | None = None,
        *,
        reserve_s: float = DEFAULT_RESERVE_S,
    ) -> None:
        self._remaining_ms = remaining_ms
        self._reserve_s = reserve_s

    @classmethod
    def from_context(cls, context, *, reserve_s: float = DEFAULT_RESERVE_S) -> "InvocationBudget":
        """Build from a Lambda context; unbounded when there is none."""
        getter = getattr(context, "get_remaining_time_in_millis", None)
        return cls(getter if callable(getter) else None, reserve_s=reserve_s)

    @property
    def bounded(self) -> bool:
        return self._remaining_ms is not None

    def remaining(self) -> float:
        """Seconds left for LLM work, after the reserve. ``inf`` when unbounded."""
        if self._remaining_ms is None:
            return float("inf")
        return max(0.0, self._remaining_ms() / 1000.0 - self._reserve_s)

    def call_timeout(self, ceiling: float, *, attempts: int = 2) -> float:
        """Per-attempt timeout for a call the client may make ``attempts`` times.

        The client retries internally, so the budget has to cover EVERY attempt
        — sizing a retried call by a single attempt's ceiling is how 2×340
        became a 900s wall. Returns ``ceiling`` unchanged when unbounded.
        """
        if attempts < 1:
            raise ValueError(f"attempts must be >= 1, got {attempts}")
        available = self.remaining()
        if available == float("inf"):
            return ceiling
        return min(ceiling, available / attempts)

    def can_afford(self, seconds: float) -> bool:
        return self.remaining() >= seconds

    def quote(self, label: str, ceiling: float, *, attempts: int = 2) -> float:
        """``call_timeout`` that RAISES rather than issuing a doomed call.

        Raises:
            BudgetExhausted: the quote is below ``MIN_VIABLE_CALL_S``.
        """
        timeout = self.call_timeout(ceiling, attempts=attempts)
        if timeout < MIN_VIABLE_CALL_S:
            raise BudgetExhausted(
                f"{label}: {timeout:.0f}s per attempt is all this invocation can "
                f"afford ({self.remaining():.0f}s left after the {self._reserve_s:.0f}s "
                f"write reserve, over {attempts} attempts) — below the "
                f"{MIN_VIABLE_CALL_S:.0f}s floor. Declining rather than starting a "
                f"call the function timeout will kill mid-flight."
            )
        if timeout < ceiling:
            logger.warning(
                "%s: budget-limited to %.0fs per attempt (static ceiling %.0fs, "
                "%.0fs remaining) — this invocation is running late.",
                label, timeout, ceiling, self.remaining(),
            )
        return timeout


class BudgetExhausted(RuntimeError):
    """Not enough invocation time left to attempt this call."""


#: Unbounded budget — the default for callers with no Lambda context.
UNBOUNDED = InvocationBudget()


#: Time each best-effort tail step is assumed to need, in seconds (config#6915).
#: These are affordability floors, not deadlines: a step is skipped when the
#: invocation cannot fund it, so an over-estimate costs a skipped advisory step
#: and an under-estimate risks the kill this module exists to prevent. Sized
#: from healthy 2026-07 invocations with headroom, and deliberately generous.
STEP_ESTIMATE_S = {
    "backlog_digest": 60.0,
    "resolved_digest": 60.0,
    "loop_verification": 120.0,
    "digest_email": 30.0,
    "issue_filing": 90.0,
    "deploy_rollup": 60.0,
}


def skip_if_unaffordable(budget, step: str, *, estimate: float | None = None) -> str | None:
    """Return a human-readable skip reason, or ``None`` when the step can run.

    A single check for a WHOLE batch — never per item. A partially-executed
    batch is worse than a skipped one for the two steps that mutate GitHub
    state (issue filing, loop verification): a kill mid-loop leaves issues
    filed or reopened with no record of where it stopped.
    """
    if budget is None:
        return None
    need = STEP_ESTIMATE_S.get(step) if estimate is None else estimate
    if need is None:
        raise KeyError(f"no budget estimate registered for step {step!r}")
    if budget.can_afford(need):
        return None
    return (
        f"invocation budget exhausted: {step} needs ~{need:.0f}s, "
        f"{budget.remaining():.0f}s left after the write reserve"
    )
