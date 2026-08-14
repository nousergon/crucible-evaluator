"""
agent.py — the Director agent (Layer C): one structured LLM call over the
Report Card v2 → a DirectorWeeklyActionPlan.

Not LangGraph — a single ``krepis.llm.LLMClient`` structured call routed
through ``krepis.router.resolve_group_structured("ultra")`` to the ultra
group's first entry that is both **reachable from this execution context**
and healthy, wrapped in a small rate-limit retry. The report card is
condensed to a digest (the issues + trends), last week's plan is supplied as
carry-over context, and the model emits the structured plan directly (krepis
structured-output + Pydantic — no freeform parsing).

**Where it runs decides what it can reach.** This Lambda declares
``exec_context="lambda"`` and krepis filters the chain by the registry's
``reachable_from`` (model-router-policy R28/R29). No registry entry declares
``lambda``, so the router is the only path from here — and an unreachable
router is an **outage**, not a licence to reach a public endpoint. That is the
whole fix for alpha-engine-config-I6183, where this module passed
``exclude_route="litellm_proxy"`` and quietly served ``glm-5.2`` at
openrouter.ai, DLP-unscanned, while logging a healthy-looking route.

The context names where this runs, **not how it is attached**, and the
distinction is load-bearing: §3.4a R27a forbids reaching the router by network
position, so an unreachable router is fixed on the router side
(alpha-engine-config-I6194) and never by attaching this function to the
router's VPC — which is what caused a 2h20m fleet-wide SSM outage on
2026-08-03 (nous-ergon-ops-I417).

The LLM is injectable (``llm=``) so the build/validate + tests run without a
key or krepis installed; ``_default_llm()`` lazily constructs the real client.
The registry reaches this Lambda by S3 download at handler startup
(``handler.py::_ensure_registry``), which sets ``LLM_MODEL_REGISTRY_PATH`` —
krepis Tier 1. An AppConfig path is also wired and permanently shadowed by
it; removing the dead one is alpha-engine-config-I6187.
Migrated from claude-opus-4-8 direct Anthropic → OpenRouter 2026-07-24, then
→ krepis.router.resolve_group_structured("ultra") 2026-08-02.
"""

from __future__ import annotations

import json
import logging
import os
import time

from director.budget import UNBOUNDED
from director.report_card_digest import summarize_report_card
from director.schema import DirectorWeeklyActionPlan

logger = logging.getLogger(__name__)

# The Director addresses a REGISTRY MODEL GROUP, never a model id. The group's
# ordered chain (LLM_MODEL_REGISTRY.yaml) is what provides redundancy — as of
# 2026-08-01 `ultra` is kimi-k3-direct -> glm-5.2-direct -> glm-5.2 ->
# deepseek-v4-pro-max, deliberately spanning providers.
#
# Why this replaced a pinned model (config#6050 follow-on, 2026-08-01): the
# prior code built ModelSpec(provider="openrouter", model="glm-5.2") with an
# OpenRouter key read in this call site. That pins the group's THIRD entry, so
# when OpenRouter returned 402 (credits exhausted) on watch-rerun-2026-08-01-6
# the Director failed outright — the two direct routes ahead of it in the chain
# were never tried. A pinned model cannot fall back, whatever the registry says.
#
# It also violated `principles.md` §2.8 four ways at once: a model id, a
# provider name, a provider api key, and an SDK client all constructed at the
# call site. Addressing the group moves every one of those into the registry.
DIRECTOR_GROUP = "ultra"

# Per-attempt ceiling for the plan call (config#6050): under the router edge's
# 360s proxy_read_timeout, so the client owns the deadline and the failure is
# attributable. config#6904 made it a ceiling rather than the budget — see
# director/budget.py and the invariant test in
# tests/test_director_invocation_budget.py.
DIRECTOR_PLAN_CEILING_S = 340.0

# Where this module runs, declared rather than inferred (model-router-policy
# R29). krepis filters the fallback chain by the registry's `reachable_from`,
# so this is the whole of what the Director says about routing — and the
# reason it can no longer end up on a public endpoint by accident.
#
# `lambda` names WHERE THIS RUNS, not how it is attached. It says nothing
# about a VPC, and it must not: model-router-policy §3.4a R27a is categorical
# that reaching the router may not depend on host, VPC, subnet, security group
# or private IP, and that if making a consumer work requires changing the
# consumer's network attachment, the ROUTER is mis-exposed.
#
# That rule was written because of this Lambda. On 2026-08-03 the available
# answer to "the Director cannot reach the router" was to attach it to the
# router's VPC; the SSM endpoint that then had to be created to restore its
# egress carried a VPC-wide private-DNS override behind a security group that
# blocked the VPC, and the whole fleet lost SSM for 2h20m (nous-ergon-ops-I417).
#
# What the declaration buys: no registry entry is reachable from `lambda`, so
# krepis can only resolve the router route or fail closed. An unreachable
# router is an OUTAGE here, never a licence to reach a public endpoint
# (model-router-policy §5). The remedy when it is unreachable is on the router
# side — one TLS-terminated authenticated exposure, alpha-engine-config-I6194.
#
# Overridable by env so a local dry-run can resolve against laptop
# reachability without editing code; the Lambda sets nothing and gets the
# correct default.
DIRECTOR_EXEC_CONTEXT = os.environ.get("KREPIS_EXEC_CONTEXT", "lambda")

_DIRECTOR_SCHEMA_NAME = "DirectorWeeklyActionPlan"
_MAX_RETRIES = 3

# `openai.APITimeoutError` stringifies as **"Request timed out."** — which
# matches neither "timeout" nor "connection". Measured 2026-08-13. So until
# this line, the one failure mode the whole timeout chain exists to bound was
# the one failure mode this loop did not retry; the retry that rescued the
# 2026-08-09 run came from the SDK's own `max_retries`, which config#7126
# removes (see `_CLIENT_MAX_RETRIES`). Retrying it here is therefore not new
# behaviour — it is the same retry, moved to the only loop that can see the
# invocation deadline.
_RETRYABLE = (
    "overloaded", "rate", "429", "529", "timeout", "timed out", "connection",
)

# ── The attempt multiplier (config#7126) ─────────────────────────────────────
#
# A single `llm.invoke()` used to be able to make FOUR timeout-bounded model
# calls, and nothing in this module said so:
#
#   * `LLMClient(max_retries=1)`         → 2 transport attempts per model call
#   * `client.structured()`              → `attempts=2` by DEFAULT (krepis'
#                                          body-level corrective retry, which
#                                          this call site never passed)
#   ⇒ 2 × 2 = 4 × the per-attempt timeout, per invoke.
#
# `_invoke_with_retry` then wrapped that in up to 3 more, for a worst case of
# 12 × 340s = 4080s inside a function whose timeout **cannot be raised**: 900s
# is AWS Lambda's service maximum, not a number we chose.
#
# Meanwhile `budget.quote(..., attempts=2)` funded TWO. The budget module was
# doing exactly what it was written to do and was quoting against an attempt
# count that was wrong by 2× at the client and 6× overall — which is why
# raising the ceiling from 300s to 900s moved the cliff instead of removing it,
# the precise outcome `director/budget.py`'s docstring warns about.
#
# The fix is not a bigger number anywhere. It is **one retry loop instead of
# three**, and it is the one that can see the deadline. Both inner loops are
# collapsed to a single attempt each, so one `invoke()` is exactly one
# timeout-bounded model call, and `_invoke_with_retry` — which now consults the
# budget before every retry — owns the count.
_CLIENT_MAX_RETRIES = 0
_STRUCTURED_ATTEMPTS = 1

# The `auth_token_type` -> credential-name mapping used to live here, as the
# THIRD copy in the fleet (groomer_krepis_adapter.py, groom_driver.py, here) —
# its own comment said so and named the lift as a follow-up. krepis 0.31.1 owns
# it (`router._AUTH_TOKEN_SECRET`, applied inside `resolve_group_spec`), so the
# copy is deleted rather than maintained. A consumer that re-derives a
# credential name is holding a routing fact at layer 5.
#
# `_EXPECTED_ROUTE_SCHEMA` went the same way: `resolve_group_spec` branches on
# the schema version and refuses rather than guessing. Two places asserting the
# same contract is how they drift.


def _assert_routed_through_the_proxy(route: dict) -> None:
    """From a Lambda the Director routes through the LiteLLM proxy. No exceptions.

    This is a guard against a **corrupted input**, not a routing decision, and
    the distinction is what keeps it on the right side of `model-router-policy`
    §2's layer-5 rule. The Director does not choose a route here; it refuses one
    that cannot be conformant. Which entries are reachable from `lambda` remains
    entirely the registry's statement about itself (R28/R29).

    It exists because that input has been wrong, in production, more than once,
    and every time the symptom was identical: a paid `ultra` call to
    openrouter.ai, DLP-unscanned, logging a healthy route (R26,
    alpha-engine-config-I6183).

    - `exclude_route="litellm_proxy"` was passed by this module to make the
      Lambda succeed while the network path to the proxy was down. It worked,
      and nothing failed for the weeks the proxy path stayed broken.
    - The registry the Lambda reads is an S3 copy that was published by hand.
      It lagged the repo, carried no `reachable_from`, and krepis read the
      omission as universal reachability.
    - krepis' health probe spoke plain HTTP at the router's TLS edge, called it
      unreachable, and fell through — while that URL was serving 23 models.

    A CI test cannot catch any of those: the code was correct in all three, and
    the artifact it consumed at runtime was not. So the assertion has to run
    where the artifact is read.

    Raising loses the week's advisory action plan. That is the intended trade
    (R20): the Director is advisory and its stage is non-fatal, while an
    unscanned egress is a policy breach that bills real money and reports
    success.
    """
    if DIRECTOR_EXEC_CONTEXT != "lambda":
        # On a laptop or the dashboard box a direct provider route is legitimate
        # — the egress proxy is on loopback there and R27d permits it.
        return
    actual = route.get("route")
    if actual != "litellm_proxy":
        raise RuntimeError(
            f"Director resolved route={actual!r} from exec_context="
            f"{DIRECTOR_EXEC_CONTEXT!r}. The only conformant route from a "
            f"Lambda is 'litellm_proxy' (model-router-policy R26): no direct "
            f"provider endpoint is reachable from here, so this resolution "
            f"came from a registry copy that is stale or wrong, or from a "
            f"resolver that skipped the proxy. Refusing to make a paid, "
            f"DLP-unscanned call. Resolved: model={route.get('deployment_id')!r} "
            f"provider={route.get('provider')!r} "
            f"api_base_url={route.get('api_base_url')!r}; "
            f"skipped={route.get('skipped_entries')!r}"
        )


def _warn_on_degraded_route(route: dict) -> None:
    """Alert when the group's declared primary is not what will serve.

    ``model-router-policy`` R12 is explicit that serving from a fallback is an
    *alert*, not a log line. krepis already returns everything needed —
    ``primary_registry_id`` and a ``skipped_entries`` list with a reason per
    entry — and this module used to throw all of it away, logging one INFO
    line that read identically whether the primary served or the third
    fallback did (alpha-engine-config-I6185).

    Emits ``DirectorRouteFallback`` on **every** run: 1 on degradation, **0 on
    the healthy path**. A metric that only appears when something is wrong is
    indistinguishable from a dead emitter, and `principles.md` §2.7 is that
    *no data* is never rendered as green.

    Emitted as a **CloudWatch Embedded Metric Format** log line, not a
    ``PutMetricData`` call. Log delivery is service-side, costs nothing, and
    needs no `monitoring` interface endpoint (~$7.30/mo) for one weekly data
    point.

    (Corrected 2026-08-03: this docstring used to justify EMF by the Lambda
    being "VPC-attached to a subnet with no internet route". It was detached
    from the VPC on 2026-08-02 and `model-router-policy` §3.4a R27a now forbids
    re-attaching it, so that reason is void. EMF remains the right choice on
    the cost argument alone — but a stale reason in a comment is how the next
    reader concludes the constraint still applies.)

    Never raises. A telemetry failure must not take down the weekly plan.
    """
    skipped = route.get("skipped_entries") or []
    primary = route.get("primary_registry_id") or route.get("primary_model")
    served = route.get("registry_id")

    # On the proxy path `registry_id` is a GROUP HANDLE — `litellm:group:ultra`
    # — not an entry id, so it can never equal `primary_registry_id`
    # (`kimi-k3-direct`). Comparing them is a category error, and it fired on
    # the very first healthy run through the router: every invocation logged
    # DEGRADED and pinned DirectorRouteFallback at 1.
    #
    # That is worse than no metric. A fallback alarm that is always on is
    # indistinguishable from one that is stuck, and it trains the reader to
    # ignore the week it means something — the same failure mode as a metric
    # that never emits, arrived at from the other direction.
    #
    # The consumer genuinely cannot know which entry served through the proxy:
    # LiteLLM walks the fallback chain internally and the resolution contract
    # reports the group, by design. So on this route the only degradation
    # signal the consumer HAS is `skipped_entries` — entries the resolver
    # itself refused before handing over. Which is the honest answer, not a
    # weakened one: R12's "serving from a fallback is an alert" is owed by the
    # layer that knows, and for the proxy path that is the router's own
    # telemetry (R21), not this module.
    if route.get("route") == "litellm_proxy":
        degraded = bool(skipped)
    else:
        degraded = bool(skipped) or (primary is not None and served != primary)

    if degraded:
        logger.warning(
            "Director route DEGRADED: group=%s primary=%s served=%s route=%s "
            "context=%s — skipped: %s",
            DIRECTOR_GROUP, primary, served, route.get("route"),
            route.get("exec_context"),
            "; ".join(
                f"{s.get('registry_id')}: {s.get('reason')}" for s in skipped
            ) or "(none recorded)",
        )

    try:
        print(json.dumps({
            "_aws": {
                "Timestamp": int(time.time() * 1000),
                "CloudWatchMetrics": [{
                    "Namespace": "AlphaEngine/Director",
                    "Dimensions": [["Group"]],
                    "Metrics": [{"Name": "DirectorRouteFallback", "Unit": "Count"}],
                }],
            },
            "Group": DIRECTOR_GROUP,
            "DirectorRouteFallback": 1 if degraded else 0,
            "served": served,
            "primary": primary,
            "route": route.get("route"),
            "exec_context": route.get("exec_context"),
        }))
    except Exception:
        logger.exception(
            "Director: failed to emit DirectorRouteFallback — the fallback "
            "alarm is blind for this run"
        )


# ── The latency signal (alpha-engine-config-I7311) ───────────────────────────
#
# Until 2026-08-14 the duration of the Director's plan call was recorded
# NOWHERE. Reconstructing it took subtracting two log timestamps by hand
# (`Director route:` → the `HTTP Request: POST` line), which is why a 2.4×
# latency climb over ten days — 87s on 2026-08-04, 135s on 2026-08-13, 205s on
# 2026-08-14 against the same route, same model and same registry max_tokens —
# was first noticed as a hard SF failure rather than as a trend.
#
# `DIRECTOR_PLAN_CEILING_S` is the wall. This is the line drawn well short of
# it. 0.6 × the ceiling is 204s: at the measured throughput of 99-109 tok/s
# that is the call already drawing ~21k completion tokens, i.e. exactly the
# regime the 2026-08-14 run was in the run BEFORE the one that failed. A run
# that crosses it has not failed and loses nothing — it says so, once, with
# the numbers attached, while there is still a week to act.
#
# It is a FRACTION, not a second literal, so raising the ceiling can never
# silently move the amber line up with it and re-hide the same regression.
DIRECTOR_PLAN_AMBER_FRACTION = 0.6


def _plan_amber_threshold_s(ceiling_s: float = None) -> float:
    return DIRECTOR_PLAN_AMBER_FRACTION * (
        DIRECTOR_PLAN_CEILING_S if ceiling_s is None else ceiling_s
    )


def _emit_plan_latency(
    *,
    elapsed_s: float,
    outcome: str,
    prompt_chars: int,
    carryover_items: int,
    usage=None,
) -> dict:
    """Emit one EMF record for one plan ATTEMPT, and return what was emitted.

    **Per attempt, not per invocation, and in a ``finally``.** The per-attempt
    ceiling is what the call is bounded by, and the attempt that matters most
    is the one that never returns — on 2026-08-14 the two censored 340s
    attempts are the whole event. An emitter that only ran after a successful
    parse would have published nothing on the day the Director hard-failed.

    **Emitted on the healthy path too**, with ``DirectorPlanLatencyAmber: 0``.
    `observability-policy` §9: a component emitting nothing is not healthy, it
    is unobserved, and "no data" is never rendered green. This mirrors
    ``DirectorRouteFallback`` above, whose alarm is only meaningful because the
    0 is published every run (alpha-engine-config-I6185).

    Never raises: a telemetry failure must not take down the weekly plan. It
    does log at ERROR, because a silent emitter is the failure mode the metric
    exists to prevent.
    """
    amber_s = _plan_amber_threshold_s()
    over_amber = elapsed_s >= amber_s
    record = {
        "DirectorPlanLatencySeconds": round(elapsed_s, 3),
        "DirectorPlanLatencyAmber": 1 if over_amber else 0,
        "DirectorPlanCeilingSeconds": DIRECTOR_PLAN_CEILING_S,
        "DirectorPlanAmberSeconds": round(amber_s, 1),
        "DirectorPlanPromptChars": int(prompt_chars),
        "DirectorPlanCarryoverItems": int(carryover_items),
        "DirectorPlanPromptTokens": int(getattr(usage, "input_tokens", 0) or 0),
        "DirectorPlanCompletionTokens": int(getattr(usage, "output_tokens", 0) or 0),
        "DirectorPlanReasoningTokens": int(getattr(usage, "reasoning_tokens", 0) or 0),
        "outcome": outcome,
        "Group": DIRECTOR_GROUP,
    }
    if over_amber:
        logger.warning(
            "Director plan call AMBER: %.1fs >= %.1fs (%.0f%% of the %.0fs "
            "ceiling) — outcome=%s prompt_chars=%d carryover_items=%d "
            "prompt_tokens=%d completion_tokens=%d (reasoning=%d). The call "
            "has not failed; it is measurably closer to the ceiling than it "
            "was. alpha-engine-config-I7311: the inputs and the output this "
            "call must produce both grow, so this number trends up on its own "
            "and the ceiling does not move.",
            elapsed_s, amber_s, DIRECTOR_PLAN_AMBER_FRACTION * 100,
            DIRECTOR_PLAN_CEILING_S, outcome, record["DirectorPlanPromptChars"],
            record["DirectorPlanCarryoverItems"],
            record["DirectorPlanPromptTokens"],
            record["DirectorPlanCompletionTokens"],
            record["DirectorPlanReasoningTokens"],
        )
    else:
        logger.info(
            "Director plan call %s in %.1fs (amber at %.1fs, ceiling %.0fs) — "
            "prompt_chars=%d carryover_items=%d prompt_tokens=%d "
            "completion_tokens=%d",
            outcome, elapsed_s, amber_s, DIRECTOR_PLAN_CEILING_S,
            record["DirectorPlanPromptChars"],
            record["DirectorPlanCarryoverItems"],
            record["DirectorPlanPromptTokens"],
            record["DirectorPlanCompletionTokens"],
        )
    try:
        print(json.dumps({
            "_aws": {
                "Timestamp": int(time.time() * 1000),
                "CloudWatchMetrics": [{
                    "Namespace": "AlphaEngine/Director",
                    "Dimensions": [["Group"]],
                    "Metrics": [
                        {"Name": "DirectorPlanLatencySeconds", "Unit": "Seconds"},
                        {"Name": "DirectorPlanLatencyAmber", "Unit": "Count"},
                        {"Name": "DirectorPlanPromptTokens", "Unit": "Count"},
                        {"Name": "DirectorPlanCompletionTokens", "Unit": "Count"},
                        {"Name": "DirectorPlanCarryoverItems", "Unit": "Count"},
                    ],
                }],
            },
            **record,
        }))
    except Exception:
        logger.exception(
            "Director: failed to emit DirectorPlanLatencySeconds — the latency "
            "trend is blind for this run, which is the condition "
            "alpha-engine-config-I7311 exists to end"
        )
    return record


def _load_system_prompt() -> str:
    """The tuned prompt (gitignored director/prompt.py) if present, else the
    committed generic template."""
    try:
        from director.prompt import SYSTEM_PROMPT  # type: ignore
        return SYSTEM_PROMPT
    except Exception:  # ImportError or anything — fall back to the template
        from director.prompt_example import SYSTEM_PROMPT
        return SYSTEM_PROMPT


def _split_messages(messages: list) -> tuple[str, str]:
    """``build_messages()``'s ``[("system", ...), ("human", ...)]`` shape ->
    krepis.llm's flat ``(system, user_content)`` call surface."""
    system = ""
    human_parts: list[str] = []
    for role, content in messages:
        if role == "system":
            system = content
        else:
            human_parts.append(content)
    return system, "\n\n".join(human_parts)


class _KrepisStructuredDirector:
    """Adapts a ``krepis.llm.LLMClient`` to the ``.invoke(messages) ->
    DirectorWeeklyActionPlan`` surface ``_invoke_with_retry`` expects."""

    #: Wall-clock one ``invoke()`` may consume, for the budget gate in
    #: ``_invoke_with_retry``. Deliberately the STATIC ceiling rather than the
    #: (possibly shorter) quoted timeout: a retry is funded only when the
    #: invocation could afford a full-ceiling attempt, so an already-late run
    #: declines instead of buying one more chance to be killed at the wall.
    attempt_cost_s = DIRECTOR_PLAN_CEILING_S

    def __init__(self, client, *, director_model: str):
        self._client = client
        self._director_model = director_model
        #: Token usage of the most recent completed attempt, read by
        #: ``_invoke_with_retry``'s latency emitter. ``None`` until a call
        #: RETURNS — a timed-out attempt reports no tokens because none were
        #: reported, and publishing a stale count for it would make the one
        #: attempt that matters look like the previous healthy one
        #: (alpha-engine-config-I7311).
        self.last_usage = None

    def invoke(self, messages: list) -> DirectorWeeklyActionPlan:
        self.last_usage = None
        system, user_content = _split_messages(messages)
        # No `max_tokens=` here. It carried a literal 8000 until 2026-08-04,
        # which SHADOWED the registry: `LLMClient.structured` takes the
        # caller's value when one is given and `spec.max_tokens` otherwise, so
        # the row's budget never reached the wire. GLM-5.2 is a reasoning model
        # and max_tokens bounds reasoning + content TOGETHER — the whole 8000
        # went to the reasoning trace and the completion came back with
        # `content: ''`, twice, fully billed (alpha-engine-config-I6396).
        #
        # It also defeated the remediation: raising the row to 65536
        # (alpha-engine-config-PR6390) changed nothing, because this line was
        # what the request actually carried, and the route log below printed
        # `spec.max_tokens` — the registry's value, not the one being sent.
        #
        # The budget is a registry-owned parameter (model-router-policy §2:
        # the registry decides model, endpoint, auth and params; the call site
        # decides only its capability tier and where it runs). Restating one
        # here is the layer-5 duplication `resolve_group_spec` was adopted to
        # end.
        result = self._client.structured(
            system=system,
            user_content=user_content,
            schema=DirectorWeeklyActionPlan,
            schema_name=_DIRECTOR_SCHEMA_NAME,
            # Explicit, not krepis' default of 2 — see `_STRUCTURED_ATTEMPTS`.
            # Inheriting the default silently doubled every quoted budget.
            attempts=_STRUCTURED_ATTEMPTS,
        )
        self.last_usage = getattr(result, "usage", None)
        plan: DirectorWeeklyActionPlan = result.parsed
        plan.director_model = self._director_model
        plan.resolved_model = result.model
        return plan


def _default_llm(budget=None) -> _KrepisStructuredDirector:
    """Construct the real structured-output Director client (lazy import).

    Resolves ``DIRECTOR_GROUP`` ("ultra") through
    ``krepis.router.resolve_group_structured()`` — krepis' documented public
    contract for programmatic callers — and builds the ``ModelSpec`` from the
    returned route: provider, deployment_id, api_base_url, and the credential
    NAME implied by auth_token_type. The registry decides model, endpoint and
    auth; this module decides only which capability tier it wants and **where
    it is running**.

    The registry is downloaded from S3 by ``handler.py::_ensure_registry``,
    which sets ``LLM_MODEL_REGISTRY_PATH`` — krepis Tier 1 (this Lambda runs
    in a public repo without ``private-docs/`` on disk). Both imports are lazy
    so tests + the grading path never pull krepis' provider SDKs or hit SSM.

    Migrated from ChatAnthropic(claude-opus-4-8) → krepis.llm(glm-5.2 via
    OpenRouter) 2026-07-24, then → krepis.router.resolve_group_structured
    ("ultra") 2026-08-02.
    """
    from krepis.llm import LLMClient
    from krepis.router import resolve_group_spec

    # `resolve_group_spec` is krepis' supported way to go from "I want the
    # ultra tier, and I am running in a Lambda" to a client. It returns a
    # ready ModelSpec plus the route it came from.
    #
    # This module used to hand-build the ModelSpec from
    # `resolve_group_structured`, and that reconstruction was the bug. The
    # resolver reports `provider: "litellm"` for the proxy route, and
    # `llm_config.PROVIDER_REGISTRY` binds that name to TRANSPORT_LITELLM —
    # `get_router()`, an **in-process LiteLLM Router** built from the registry
    # that calls each provider DIRECTLY from this Lambda, reading
    # OPENROUTER_API_KEY out of the environment as it goes.
    #
    # Measured 2026-08-04, on the first REAL invoke after the edge went live:
    # `ModuleNotFoundError: No module named 'litellm'`. Adding that package
    # would have made the Director "work" while egressing straight to
    # openrouter.ai, unscanned, bypassing the authenticated edge and every
    # per-consumer control on it — the exact shortcut this arc exists to end.
    # krepis 0.31.1 maps the proxy route to a plain OpenAI transport against
    # the edge URL, which is what the proxy actually speaks.
    #
    # A dry run cannot catch this: it stops before `.invoke()`. Only a real
    # completion reaches the transport.
    #
    # `exec_context` is the ONLY thing this module says about routing, and it
    # is a statement about where it is running, not about which routes it
    # wants (model-router-policy §2 layer 5, R29). `wire=openai` because this
    # call site speaks the OpenAI wire format.
    spec, route = resolve_group_spec(
        DIRECTOR_GROUP,
        exec_context=DIRECTOR_EXEC_CONTEXT,
        wire="openai",
    )
    _assert_routed_through_the_proxy(route)
    _warn_on_degraded_route(route)

    # The schema and auth-token checks that used to live here are inside
    # `resolve_group_spec`, which refuses rather than guesses on both counts.
    # Keeping local copies would be the same layer-5 duplication the
    # ModelSpec reconstruction was.
    api_key_env = spec.api_key_env
    auth_type = route["auth_token_type"]

    logger.info(
        "Director route: group=%s model=%s provider=%s route=%s "
        "(primary=%s, max_tokens=%s, transport=%s)",
        DIRECTOR_GROUP, route["deployment_id"], spec.provider,
        route.get("route"), route.get("primary_model"), spec.max_tokens,
        spec.transport,
    )
    # callsite_id is REQUIRED since krepis 0.23 (krepis/llm.py::LLMClient.__init__,
    # validated non-empty). It is the join key between this call's emitted cost
    # row and its LLM_CALLSITE_REGISTRY.yaml entry, so the literal must stay in
    # sync with that row's `id` (alpha-engine-config, id: director-plan).
    #
    # `auth_token_type == "placeholder"` maps to api_key_env=None and means
    # the egress proxy holds the real key — there is NOTHING to resolve, and
    # treating that as a missing credential would break every direct route
    # from a context that can reach one. The two cases are distinguished
    # rather than collapsed.
    #
    # Where a credential IS named and cannot be resolved, this RAISES. It
    # previously fell through with no key and surfaced as a 401 from the
    # provider — an auth failure that looks like a provider problem
    # (model-router-policy R20: fail closed, loudly).
    api_kwargs = {}
    if api_key_env is not None:
        from krepis.secrets import get_secret

        secret = get_secret(api_key_env, required=False)
        if not secret:
            raise RuntimeError(
                f"Director: no credential for auth_token_type {auth_type!r} — "
                f"{api_key_env!r} is absent from the environment and from SSM. "
                "Refusing to call the endpoint unauthenticated: the resulting "
                "401 would read as a provider fault rather than a missing key."
            )
        api_kwargs["api_key"] = secret
    # timeout/max_retries are sized for the ultra group's single-call latency,
    # not left at the krepis defaults (180s / 3): the plan call's measured
    # Duration 2026-06-01→08-03 is p90 ≈ 85s, p99 ≈ 285s, and the 2026-08-08
    # Saturday run died exactly this way — the 180s default aborted a healthy
    # in-flight call, the SDK retried, and the second attempt hit the Lambda's
    # 300s wall (config#6050). 340s keeps the deadline client-owned: the router
    # edge's proxy_read_timeout is 360s, so the client, not the edge, times out
    # first and the failure is attributable.
    #
    # Where 340 comes from — the derivation, which until config#7126 was
    # asserted and never written down. Note the p90/p99 quoted above are
    # CENSORED statistics: 3 of 14 observed runs were killed at a ceiling, so
    # those percentiles report the wall back to the reader rather than the
    # model's requirement. Five UNCENSORED plan calls, measured 2026-08-13
    # against this same router edge with the same registry max_tokens and a
    # Lambda-shaped prompt (11,212 prompt tokens, both backlog digests
    # included), every one returning `finish_reason: stop`:
    #
    #     170.1s · 183.0s · 194.9s · 199.5s · 228.0s
    #     draw 18.2k-23.3k completion tokens at 99-109 tok/s
    #
    # Plus one full `build_action_plan()` through this exact code path at
    # krepis 0.56.0 — 212.7s, 25 action items, resolved=glm-5.2 — which also
    # clears the served-model guard that failed the 2026-08-09 run
    # (alpha-engine-config-I6543).
    #
    # 340s = 1.49 × the slowest uncensored call — `sf-pipeline-policy.md` §4's
    # "observed p95 × 1.5" shape, now computed over samples that were allowed
    # to finish.
    #
    # It also cannot simply be raised, for a reason outside this repo: the
    # router edge's `proxy_read_timeout` is 360s, so any ceiling at or above it
    # hands the deadline to the EDGE and the failure stops being attributable
    # to this call site. 340 < 360 keeps the client owning it.
    #
    # `max_retries=0` is not a loss of resilience — the retry moved to
    # `_invoke_with_retry`, the only loop that can see the invocation deadline.
    # See `_CLIENT_MAX_RETRIES`.
    #
    # config#6904: 340 is now a CEILING, not the budget. The old comment here
    # claimed one retry "bounds the worst case to 2×340s inside the Lambda's
    # 900s budget" — but the Phase-G retro judge adds 2×120s in the same
    # invocation, so the worst case is 920s against a 900s function timeout,
    # before a single S3 write. The quote below derives the per-attempt budget
    # from the time actually remaining, so a late invocation declines the call
    # (raising BudgetExhausted, which the caller records) instead of starting
    # one the wall will kill mid-flight with no artifact and no cause.
    plan_budget = budget or UNBOUNDED
    client = LLMClient(
        spec,
        callsite_id="director-plan",
        timeout=plan_budget.quote("director-plan", DIRECTOR_PLAN_CEILING_S, attempts=2),
        max_retries=_CLIENT_MAX_RETRIES,
        **api_kwargs,
    )
    return _KrepisStructuredDirector(client, director_model=route["deployment_id"])


#: Marker the carry-over section header carries so the emitted latency record
#: can report how many ledger rows this prompt made the model dispose of —
#: the quantity that grew 19 → 41 while the call duration grew 87s → 205s
#: (alpha-engine-config-I7311). Parsed rather than threaded through
#: ``build_messages``' signature because ``_invoke_with_retry`` is handed
#: MESSAGES, not the artifacts they were built from, and widening that
#: boundary to carry telemetry would couple the retry loop to the card.
_CARRYOVER_COUNT_MARKER = "carry-over ledger, active items="


def _carryover_item_count(messages: list) -> int:
    """How many active ledger rows this prompt carries; 0 when there is no
    carry-over section (first cycle, empty ledger, or an injected test
    double's hand-built messages). Never raises."""
    for _, content in messages:
        idx = str(content).find(_CARRYOVER_COUNT_MARKER)
        if idx < 0:
            continue
        digits = ""
        for ch in str(content)[idx + len(_CARRYOVER_COUNT_MARKER):]:
            if not ch.isdigit():
                break
            digits += ch
        if digits:
            return int(digits)
    return 0


def _carryover_context(carryover: dict | None) -> str:
    if not carryover or not carryover.get("items"):
        return "No prior action plan on record (this is the first cycle or the ledger is empty)."
    items = carryover.get("items", [])
    lines = [
        f"Last week's open action items ({_CARRYOVER_COUNT_MARKER}{len(items)}):"
    ]
    for it in items:
        lines.append(
            f"  - [{it.get('id')}] {it.get('title')} "
            f"(status={it.get('status')}, owner={it.get('proposed_owner')}, priority={it.get('priority')})"
        )
    return "\n".join(lines)


def build_messages(report_card: dict, *, carryover: dict | None = None, roadmap_digest: str | None = None,
                   resolved_digest: str | None = None) -> list:
    """Assemble (system, human) messages for the Director call."""
    human = [
        summarize_report_card(report_card),
        "",
        _carryover_context(carryover),
    ]
    if roadmap_digest:
        human += ["", "Currently-tracked / in-flight work (open backlog — do NOT re-propose):", roadmap_digest]
    if resolved_digest:
        human += [
            "",
            "Recently INVESTIGATED & RESOLVED (director-proposals closed in the last ~8 weeks — "
            "do NOT re-propose these under a new name. If a metric tied to one of these still reads "
            "adverse, it is being MONITORED / accumulating to significance, NOT unexamined — note it "
            "in the existing item's terms rather than opening a fresh investigation):",
            resolved_digest,
        ]
    human += [
        "",
        "Produce the DirectorWeeklyActionPlan now. Ground every action item's "
        "rationale + evidence in the metrics above.",
    ]
    return [("system", _load_system_prompt()), ("human", "\n".join(human))]


def _invoke_with_retry(llm, messages, *, budget=None):
    """Retry transient LLM failures — but only while the invocation can pay.

    This is the fleet's single retry loop for both Director LLM calls (the plan
    here, the Phase-G judge in ``director/retro.py``), and since config#7126 it
    is the ONLY one: ``_CLIENT_MAX_RETRIES`` and ``_STRUCTURED_ATTEMPTS``
    collapse krepis' transport and body-level loops to one attempt each, so one
    ``llm.invoke()`` is exactly one timeout-bounded model call. That matters
    because retry loops **multiply**, and only this one can see the deadline.

    ``budget`` is an ``InvocationBudget``. Before every retry the loop requires
    the invocation to still afford a full attempt plus its backoff; when it
    cannot, the last error is raised **immediately** rather than after starting
    a call the function timeout will kill mid-flight. That distinction is the
    whole point: a Lambda killed at the wall writes no artifact and logs no
    cause, which is what made the 2026-08-08 failure — and the three silent
    weeks after it — invisible. An error raised here reaches the SF state as a
    named error with a message.

    The attempt's cost is read off the llm adapter's ``attempt_cost_s``.
    Adapters that do not declare one (every injected test double) are treated
    as unbounded, so the retry behaviour outside Lambda is unchanged.
    """
    budget = budget or UNBOUNDED
    cost = getattr(llm, "attempt_cost_s", None)
    last = None
    # Measured once for the whole loop — the prompt does not change between
    # attempts, and these are the two quantities that explain the duration
    # (alpha-engine-config-I7311).
    prompt_chars = sum(len(c) for _, c in messages)
    carryover_items = _carryover_item_count(messages)
    for attempt in range(1, _MAX_RETRIES + 1):
        started = time.monotonic()
        try:
            result = llm.invoke(messages)
        except Exception as e:  # noqa: BLE001 — classify + retry transient, raise the rest
            # Elapsed is read HERE, before the classify/backoff below, so the
            # published number is the model call and never the backoff the
            # loop adds after it. Same reason the success path emits before
            # returning rather than in a `finally`.
            _emit_plan_latency(
                elapsed_s=time.monotonic() - started,
                outcome=f"error:{type(e).__name__}",
                prompt_chars=prompt_chars,
                carryover_items=carryover_items,
                usage=None,
            )
            last = e
            msg = str(e).lower()
            if attempt >= _MAX_RETRIES or not any(t in msg for t in _RETRYABLE):
                raise
            delay = min(2 ** attempt, 30)
            if cost is not None and not budget.can_afford(cost + delay):
                logger.error(
                    "Director LLM transient error (attempt %d): %s — NOT retrying: "
                    "another attempt needs ~%.0fs (%.0fs ceiling + %.0fs backoff) "
                    "and only %.0fs remain after the write reserve. Raising now so "
                    "the failure carries a cause, instead of starting a call the "
                    "function timeout would kill mid-flight.",
                    attempt, e, cost + delay, cost, delay, budget.remaining(),
                )
                raise
            logger.warning("Director LLM transient error (attempt %d): %s — retrying in %ss", attempt, e, delay)
            time.sleep(delay)
        else:
            record = _emit_plan_latency(
                elapsed_s=time.monotonic() - started,
                outcome="ok",
                prompt_chars=prompt_chars,
                carryover_items=carryover_items,
                usage=getattr(llm, "last_usage", None),
            )
            # Also stamp it onto the plan, which is ARCHIVED to
            # ``director/{run_date}/action_plan.json``. CloudWatch metrics age
            # out; the artifact does not, and I7311's Closes-when asks for the
            # route, the resolved model and the token counts to be readable
            # per invocation from a durable surface. ``DirectorWeeklyActionPlan``
            # is ``extra="allow"``, so this needs no schema change — and it is
            # guarded because ``llm=`` is injectable and a test double may
            # return something that is not a pydantic model.
            try:
                result.plan_call_telemetry = record
            except Exception:  # noqa: BLE001 — telemetry never breaks the plan
                logger.debug(
                    "Director: could not stamp plan_call_telemetry onto %r; "
                    "the EMF record above is still published",
                    type(result).__name__,
                )
            return result
    raise RuntimeError(f"Director LLM failed after {_MAX_RETRIES} attempts") from last


def build_action_plan(
    report_card: dict,
    *,
    run_date: str | None = None,
    carryover: dict | None = None,
    roadmap_digest: str | None = None,
    resolved_digest: str | None = None,
    llm=None,
    budget=None,
) -> DirectorWeeklyActionPlan:
    """Run the Director: report card → DirectorWeeklyActionPlan.

    ``llm`` is injectable (a structured-output runnable returning a
    DirectorWeeklyActionPlan); defaults to the real krepis.llm client resolved
    through ``krepis.router.resolve_group_structured("ultra")`` — the registry
    decides model, endpoint and auth. ``run_date`` overrides the plan's
    run_date (else taken from the card provenance).

    ``budget`` is an ``InvocationBudget`` (config#6904): the plan call's
    per-attempt timeout is the smaller of its static ceiling and what the
    invocation can still afford. Omitted (or ``None``) means unbounded, which
    is the behaviour outside Lambda.
    """
    llm = llm or _default_llm(budget)
    messages = build_messages(report_card, carryover=carryover, roadmap_digest=roadmap_digest,
                              resolved_digest=resolved_digest)
    plan = _invoke_with_retry(llm, messages, budget=budget)
    # Stamp the run_date from the card if the model didn't echo one.
    rd = run_date or (report_card.get("_provenance", {}) or {}).get("run_date")
    if rd and not plan.run_date:
        plan.run_date = rd
    logger.info(
        "Director plan for %s: %d action items, %d top risks",
        plan.run_date, len(plan.action_items), len(plan.top_risks),
    )
    return plan
