"""Where the Director runs, and the time budget each home affords it.

The Director has two homes and runs the same code in both
(alpha-engine-config-I11936, Brian's 2026-10-03 ruling to move it off Lambda):

* ``LAMBDA`` — ``alpha-engine-evaluator-director``, invoked by ``handler.handler``.
  Kept working so a rollback is a Step Function edit, not a code change.
* ``WEEKLY_SPOT`` — the weekly SF's launcher box (``DispatchWeeklyFreshnessSpot``),
  driven over SSM by ``director/box_run.py`` via ``infrastructure/director_on_box.sh``.

Both homes share ONE plan-call bound shape: a static total-duration ceiling
(``plan_ceiling_s``) armed as krepis ``total_timeout`` on the streamed call, and
an invocation wall (``wall_s``) that every LLM call and tail step is quoted
against (``director/budget.py``). Only the numbers differ. Nothing here removes
a bound: the box has no 900s cap, but it has its own hard ceiling, derived below.

WHY THE LAMBDA HOME STOPPED FITTING. AWS Lambda's 900s function maximum leaves
the plan call 600s (``agent.DIRECTOR_PLAN_CEILING_S``: 900 - 240 retro reserve -
45 write reserve, rounded down). Measured since 2026-09-19, COMPLETED plan calls
took 379-596s — the slowest of them 4s inside the ceiling — and on 2026-10-03
``watch-rerun-2026-10-02-1`` hit ``StreamTotalTimeoutError`` at 600s while the
model was still streaming reasoning (19,440 chunks, zero answer characters). A
full-length answer on that trajectory needs ~1,500s. No Lambda budget reaches it.
"""

from __future__ import annotations

from dataclasses import dataclass

from director.budget import DEFAULT_RESERVE_S, RETRO_JUDGE_RESERVE_S, STEP_ESTIMATE_S


@dataclass(frozen=True)
class HostProfile:
    """The time a Director home can afford.

    ``wall_s`` is the whole invocation's wall clock, the quantity
    ``InvocationBudget`` counts down from. ``plan_ceiling_s`` is the plan call's
    static per-attempt AND total-duration ceiling; the budget may quote it lower
    on a late invocation, never higher.
    """

    name: str
    wall_s: float
    plan_ceiling_s: float


#: AWS Lambda's service maximum — not a number we chose, and not raisable.
LAMBDA_WALL_S = 900.0

#: The Lambda home's plan ceiling. Kept equal to the historical
#: ``agent.DIRECTOR_PLAN_CEILING_S`` (600s) — see that constant's derivation; the
#: Lambda path is unchanged by the move.
LAMBDA_PLAN_CEILING_S = 600.0

# ── The weekly-spot home ─────────────────────────────────────────────────────
#
# PLAN CEILING. Derived from measurement, with the worst case named:
#
#     completed plan calls since 2026-09-19 : 379-596s   (max 596s)
#     a full-length answer, extrapolated    : ~1,500s    (the 10-03 trajectory)
#
#     1,800s = 1.2 x the ~1,500s worst case
#            = 3.0 x the slowest COMPLETED call (596s)
#
# It stays a HARD bound — armed as krepis `total_timeout` on the streamed call,
# exactly as on Lambda — because the Director's SF failure is terminal
# (config#6408): an unbounded stream would hold the whole weekly run, not just
# this stage. 1,800s is the ceiling on what one plan call may cost, not a target.
WEEKLY_SPOT_PLAN_CEILING_S = 1800.0

# INVOCATION WALL. One term per thing the invocation owes, the same shape as
# the Lambda derivation in `agent.py`:
#
#       120 = pre-plan backlog + resolved digests (2 x 60, STEP_ESTIMATE_S)
#   + 1,800 = the plan call (WEEKLY_SPOT_PLAN_CEILING_S)
#   +   240 = Phase-G retro judge (RETRO_JUDGE_RESERVE_S)
#   +   300 = tail: loop verification 120 + digest email 30 + issue filing 90
#             + deploy rollup 60 (STEP_ESTIMATE_S)
#   +    45 = write reserve (DEFAULT_RESERVE_S)
#   = 2,505, rounded up to 2,700 (45 min)
#
# so a plan call that uses its whole ceiling still leaves every best-effort tail
# step affordable, rather than the tail being skipped on exactly the weeks the
# plan was long. `tests/test_director_hosting.py` recomputes the sum from the
# live constants, so a raised step estimate that breaks it fails in review.
WEEKLY_SPOT_WALL_S = 2700.0

#: Time `infrastructure/director_on_box.sh` may spend BEFORE the Python wall
#: starts: the crucible-evaluator clone/pin, the gitleaks install and — on the
#: first Director run on a fresh box — the evaluator venv build (nousergon-lib
#: from git, numpy/pandas wheels). A cold pip index is the slow case.
WEEKLY_SPOT_PROVISION_ALLOWANCE_S = 900.0

#: The SSM `executionTimeout` the weekly SF's `Director` state must carry: the
#: wall plus the provisioning allowance. SSM kills the command at this bound —
#: the outermost guard, behind the in-process wall that is meant to bind first.
#: nousergon-data's `tests/test_sf_director_on_spot_wiring.py` pins the SF side
#: to this number; `box_run` additionally clamps its wall to the deadline the SF
#: actually exported, so a smaller SF value shortens the quote rather than
#: SIGKILLing a call mid-flight.
WEEKLY_SPOT_SSM_EXECUTION_TIMEOUT_S = WEEKLY_SPOT_WALL_S + WEEKLY_SPOT_PROVISION_ALLOWANCE_S

LAMBDA = HostProfile("lambda", LAMBDA_WALL_S, LAMBDA_PLAN_CEILING_S)
WEEKLY_SPOT = HostProfile("weekly-spot", WEEKLY_SPOT_WALL_S, WEEKLY_SPOT_PLAN_CEILING_S)


def owed_seconds(plan_ceiling_s: float) -> float:
    """Everything one invocation owes when its plan call uses its full ceiling.

    The sum the wall must cover — see the WEEKLY_SPOT_WALL_S derivation.
    """
    pre_plan = STEP_ESTIMATE_S["backlog_digest"] + STEP_ESTIMATE_S["resolved_digest"]
    tail = (
        STEP_ESTIMATE_S["loop_verification"]
        + STEP_ESTIMATE_S["digest_email"]
        + STEP_ESTIMATE_S["issue_filing"]
        + STEP_ESTIMATE_S["deploy_rollup"]
    )
    return pre_plan + plan_ceiling_s + RETRO_JUDGE_RESERVE_S + tail + DEFAULT_RESERVE_S


# ── The box's exit-code contract with the Step Function ──────────────────────
#
# The SF cannot read a Python return value from an SSM command, and it must not
# parse the command's stdout: SSM keeps only the FIRST ~24,000 characters, and
# `States.StringToJson` on a malformed body raises States.Runtime, which no
# Catch can take (both measured on the weekly SF, nousergon-data
# `ReadModelZooArenaCycle`). The two facts the SF routes on —
# `status == "degraded"` (CheckDirectorSubResults) and `retro_refused`
# (CheckDirectorRetroRefused) — are therefore carried in the exit code, which
# `ssm:getCommandInvocation` returns as an integer the SF compares whole.
#
# Every other non-zero code (a raised exception is 1, a missing interpreter 127,
# a kill 137 …) means the Director FAILED, which stays terminal (config#6408).
EXIT_CLEAN = 0
EXIT_DEGRADED = 20
EXIT_RETRO_REFUSED = 21
EXIT_DEGRADED_AND_RETRO_REFUSED = 22

#: Codes the SF routes as a COMPLETED Director. Anything else is a failure.
COMPLETED_EXIT_CODES = frozenset(
    {EXIT_CLEAN, EXIT_DEGRADED, EXIT_RETRO_REFUSED, EXIT_DEGRADED_AND_RETRO_REFUSED}
)


def exit_code_for(summary: dict) -> int:
    """Map a completed run's summary onto the exit-code contract above."""
    degraded = summary.get("status") == "degraded"
    refused = summary.get("retro_refused") is True
    if degraded and refused:
        return EXIT_DEGRADED_AND_RETRO_REFUSED
    if degraded:
        return EXIT_DEGRADED
    if refused:
        return EXIT_RETRO_REFUSED
    return EXIT_CLEAN
