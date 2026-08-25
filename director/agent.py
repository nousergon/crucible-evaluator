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
krepis Tier 1, and now the only path. The dead AppConfig fallback that used to
sit permanently shadowed behind it was deleted in PR263
(alpha-engine-config-I6187, closed); the S3 download fails loud rather than
degrading onto a second source.

The plan call is **streamed** (alpha-engine-config-I8164). See
``DIRECTOR_PLAN_IDLE_TIMEOUT_S`` for what that bounds and what it does not.
Migrated from claude-opus-4-8 direct Anthropic → OpenRouter 2026-07-24, then
→ krepis.router.resolve_group_structured("ultra") 2026-08-02.
"""

from __future__ import annotations

import json
import logging
import os
import time

from director.budget import RETRO_JUDGE_RESERVE_S, UNBOUNDED
from director.carryover import carry_count, is_p0, order_for_prompt
# Pure, stdlib-only helpers (no boto3, no GitHub call) — the import graph is
# acyclic: loop_verification -> issue_filer -> roadmap_pr -> schema, and none
# of those imports agent.
from director.loop_verification import (
    ADVERSE_STATUSES,
    component_status_map,
    resolve_cited_metrics,
)
from director.report_card_digest import summarize_report_card
from director.schema import DirectorWeeklyActionPlan

logger = logging.getLogger(__name__)

# The Director addresses a REGISTRY MODEL GROUP, never a model id. The group's
# ordered chain (LLM_MODEL_REGISTRY.yaml) is what provides redundancy.
#
# The chain, measured 2026-08-22 AFTER alpha-engine-config-I8165 landed
# (`krepis.router models`, `_parse_registry` fallbacks, against the registry at
# that commit) — two live entries over TWO providers:
#
#     ultra-glm-5.2-direct   -> glm-5.2         (zhipu)     primary
#     ultra-deepseek-v4-pro  -> deepseek-v4-pro (deepseek)  fallback
#
# The other three entries carrying `group: ultra` are not live in it: `kimi-k3`
# and `kimi-k3-direct` are `deprecated`, and `glm-5.2` (openrouter) is
# `unavailable` — and it would not have been a second PROVIDER anyway, being the
# same Zhipu model reached through an aggregator (config-I6561).
# `claude-fable-5` is excluded from every group by Brian's 2026-07-29 ruling and
# stays excluded; I8165 declined the option that would have re-admitted it.
#
# THIS IS THE THIRD VERSION OF THIS COMMENT, AND THE FIRST TWO WERE BOTH WRONG.
# It read "kimi-k3-direct -> glm-5.2-direct -> glm-5.2 -> deepseek-v4-pro-max,
# deliberately spanning providers" while the chain was one entry long
# (corrected in crucible-evaluator-PR248); the correction then said "a chain of
# ONE ... DirectorRouteFallback cannot be anything but 0", true for exactly the
# hours between PR248 and I8165. A chain comment is a claim about ANOTHER
# repo's file, so nothing in this repo's CI can keep it true — re-measure it
# against `krepis.router models` before trusting it, and re-write it here in the
# same change as any registry edit that moves the group.
#
# WHAT THE SECOND ARM IS AND IS NOT. `deepseek-v4-pro` is admitted for
# AVAILABILITY only (model-router-policy R33). It is a WEAKER model than
# GLM-5.2 for this call, so a plan it serves is a DEGRADED plan, never an
# equivalent one — which is why `_stamp_route_degradation` below marks the plan
# artifact and the Report Card with the model that actually served. Before
# I8165 there was nothing to fall back TO: `DirectorRouteFallback` was
# structurally pinned at 0 and a slow or unavailable zhipu was a full outage,
# which is what killed the 2026-08-22 weekly run and its rerun
# (alpha-engine-config-I8151).
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
# proxy_read_timeout, so the client owns the deadline and the failure is
# attributable. config#6904 made it a ceiling rather than the budget — see
# director/budget.py and the invariant test in
# tests/test_director_invocation_budget.py.
#
# 2026-08-22 (alpha-engine-config-I7311): raised 340 -> 600, and the quote below
# moved from attempts=2 to a single attempt with the retro judge's time named as
# an explicit downstream reservation. What forced it: the 2026-08-22 weekly run
# and its first rerun burned FOUR consecutive attempts censored at 340s (344.5s,
# 347.3s, 340.2s, 344.5s elapsed, zero completion tokens each) on an unchanged
# prompt of 48,240 chars / 32 carry-over items. 340 was never the model's
# requirement -- it was `available/2`, and that divisor was an unnamed
# reservation for the retro judge wearing a retry's clothes.
#
# The derivation, which now has one term per thing the invocation owes:
#
#     600 = 900s function timeout
#         - 240s retro-judge reservation (budget.RETRO_JUDGE_RESERVE_S)
#         -  45s write reserve            (budget.DEFAULT_RESERVE_S)
#         =  615s affordable, rounded down to 600
#
# so the ceiling stays individually affordable (the invariant
# tests/test_director_invocation_budget.py asserts) instead of becoming a loose
# bound the budget silently clamps. Against a measured uncensored need of
# 170-228s that is 2.6x headroom, where 340s was 1.5x and censoring.
#
# The edge must stay ABOVE it -- nous-ergon-ops raised the router's
# proxy_read_timeout 360s -> 900s in the same arc for exactly that reason; a
# client deadline at or above the edge's hands the timeout to nginx and it
# stops being attributable to this call site.
#
# ── WHAT STREAMING DOES TO THIS NUMBER (alpha-engine-config-I8164) ───────────
#
# It still binds, and it binds a DIFFERENT QUANTITY. Say so here rather than
# leaving the derivation above reading as though nothing moved.
#
# The quote below is still what the client is built with, and krepis still
# hands it to the SDK as `OpenAI(timeout=...)`. A bare float there becomes
# httpx's timeout on every phase — connect, read, write, pool — and a READ
# timeout bounds the gap between reads, not the life of the request. On a
# non-streamed call there is exactly one read, so read-bound and total-bound
# were the same number and nothing here had to distinguish them. On a streamed
# call there are thousands, so:
#
#   * 600s remains a hard bound on CONNECT and on any single silent read. It is
#     the outer backstop behind DIRECTOR_PLAN_IDLE_TIMEOUT_S, which fires first
#     at 90s and fires with evidence.
#   * 600s no longer bounds TOTAL duration. A stream that keeps producing
#     chunks resets the read clock with every one of them.
#
# The remaining total bound is the Lambda's 900s wall, minus what
# `budget.quote(..., downstream_s=RETRO_JUDGE_RESERVE_S)` reserves — and
# `_invoke_with_retry` enforces that only BEFORE a call, by declining to start
# one it cannot afford. Nothing interrupts a call already in flight. That is a
# real narrowing of what this ceiling guarantees, and it is stated rather than
# absorbed: the correct home for a total-duration bound on a streamed call is
# krepis (a `total_timeout` beside `idle_timeout`, bounding the accumulation
# loop that already tracks `elapsed`), not a second deadline hand-rolled at this
# call site next to the pump that owns one. Filed as
# alpha-engine-config-I8348.
#
# The exposure that narrowing buys is small and is bounded on the other side by
# measurement: the uncensored plan calls draw 18.2k-23.3k completion tokens at
# 99-109 tok/s, and streaming does not change generation rate. Reaching 900s
# would take a generation running ~4x slower than any yet observed while never
# once falling silent for 90s. Against that: the failure this replaces has
# already killed a weekly run and its rerun, four attempts, zero tokens.
DIRECTOR_PLAN_CEILING_S = 600.0

# ── The liveness bound (alpha-engine-config-I8164) ───────────────────────────
#
# The plan call streams. `DIRECTOR_PLAN_CEILING_S` above is a bound on WAITING;
# this is a bound on SILENCE, and the difference is the whole point of the
# change. Four consecutive 2026-08-22 attempts were censored at the then-340s
# ceiling with ZERO completion tokens each — after ~340s of a wholly opaque
# socket, the only fact the failure carried was its own duration. Under
# streaming the same failure carries the partial generation, the chunk count and
# the elapsed time on `krepis.llm.StreamIdleTimeoutError`, because the thing
# being bounded is now a measurable quantity rather than the absence of one.
#
# WHY 90 AND NOT `1.5 x measured`. The measured inter-chunk silence on both
# `ultra` arms, probed 2026-08-25 through each row's own egress-proxy path with
# `stream_options.include_usage`:
#
#     glm-5.2-direct   979 chunks / 83.2s   largest silence 3.72s
#     deepseek-v4-pro 5529 chunks / 90.0s   largest silence 1.64s
#
# In BOTH runs the largest silence in the whole call was the time to the FIRST
# chunk; every gap after it is sub-millisecond at the median, and neither
# reasoning model buffered its trace into a silent block. So the quantity this
# constant must cover is the first-chunk wait, and `3.72 x 1.5` would be a
# reckless reading of it: those probes used a 93- and a 166-token prompt, while
# the Director's is ~11,212 prompt tokens (the uncensored 2026-08-13 sample set
# below). Time-to-first-chunk is prefill-dominated, so 3.72s is a FLOOR on the
# real first-chunk wait on this call site, not an estimate of it.
#
# The asymmetry decides the number. A budget set too LOOSE costs a slower
# failure inside a bound that still exists. A budget set too TIGHT aborts a
# healthy plan call and looks exactly like the outage it replaced. With no
# Lambda-shaped first-chunk measurement in hand, the correct move is the loose
# side of the unmeasured quantity:
#
#     90s = ~24x the largest silence measured on the ultra primary
#         = ~1/6.7 of the 600s that is currently the effective liveness bound
#
# It is stated as a literal here rather than inherited from krepis'
# `DEFAULT_STREAM_IDLE_TIMEOUT_S` — which is also 90.0 today — deliberately, and
# for the same reason `_STRUCTURED_ATTEMPTS` is explicit: a library default is a
# figure chosen for an unknown call site, and letting it move this call site's
# liveness bound silently is how a knob stops meaning what its comment says. The
# coincidence is a coincidence.
#
# It must stay strictly BELOW the client's transport timeout or it can never
# fire: krepis warns on `idle_timeout >= LLMClient(timeout=...)` because the
# transport's read deadline would bind first. The quoted timeout is derived from
# `DIRECTOR_PLAN_CEILING_S` (600s) and floored by the budget, so 90 is safe by a
# wide margin; `tests/test_director_streaming_plan_call.py` asserts the ordering
# rather than trusting it.
#
# RE-ANCHOR TRIGGER: the first weekly run that completes a streamed plan call
# emits its own inter-chunk distribution. `alpha-engine-config-I8200` (gated on
# the 2026-08-29 weekly run) re-anchors `DIRECTOR_PLAN_MEASURED_MAX_S`; this
# constant should be re-derived in the same pass, against the measured
# Lambda-shaped first-chunk wait, and is expected to come DOWN.
DIRECTOR_PLAN_IDLE_TIMEOUT_S = 90.0

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


def _warn_on_degraded_route(
    route: dict,
    *,
    group: str | None = None,
    metric_name: str = "DirectorRouteFallback",
) -> None:
    """Alert when the group's declared primary is not what will serve.

    ``group`` and ``metric_name`` default to the Director's own plan call, so
    every existing caller is unchanged. They are parameters because the retro
    judge (``director/retro.py``) resolves a DIFFERENT group through the same
    router and needs the same R12 alert — and emitting its degradation under
    ``Group=ultra`` / ``DirectorRouteFallback`` would attribute the judge's
    fallback to the plan call, which is a worse signal than none: the alarm
    would fire for a component that was healthy. One metric per resolved group,
    named for the call site that resolved it.

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
    group = group or DIRECTOR_GROUP
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
    # BEFORE the call, the consumer genuinely cannot know which entry will serve
    # through the proxy: LiteLLM walks the fallback chain internally and the
    # resolution contract reports the group, by design. So at THIS point the
    # only degradation signal available is `skipped_entries` — entries the
    # resolver itself refused before handing over.
    #
    # CORRECTED 2026-08-22 (alpha-engine-config-I8165). This comment used to end
    # "the consumer genuinely cannot know which entry served" full stop, and
    # concluded that R12's alert was owed only by the router's own telemetry.
    # That is true of RESOLUTION time and false of COMPLETION time: krepis
    # resolves the response's model field back to the billable upstream id
    # (`krepis.llm._resolve_group_served_model`), so `result.model` names the
    # entry that actually served — which is exactly how `plan.resolved_model`
    # has been populated all along. The consumer had the fact and threw it away.
    #
    # That mattered little while `ultra` was a chain of one. I8165 gave it a
    # second arm that is deliberately WEAKER than the primary, so a silently
    # substituted fallback plan would now be a quality regression wearing a
    # champion plan's clothes. `_stamp_route_degradation` closes it, after the
    # call, where the answer exists. This function keeps its pre-call job: it is
    # the only signal available when the call never returns at all.
    if route.get("route") == "litellm_proxy":
        degraded = bool(skipped)
    else:
        degraded = bool(skipped) or (primary is not None and served != primary)

    if degraded:
        logger.warning(
            "Director route DEGRADED: group=%s primary=%s served=%s route=%s "
            "context=%s — skipped: %s",
            group, primary, served, route.get("route"),
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
                    "Metrics": [{"Name": metric_name, "Unit": "Count"}],
                }],
            },
            "Group": group,
            metric_name: 1 if degraded else 0,
            "served": served,
            "primary": primary,
            "route": route.get("route"),
            "exec_context": route.get("exec_context"),
        }))
    except Exception:
        logger.exception(
            "Director: failed to emit %s — the fallback alarm is blind for "
            "this run", metric_name
        )


# ── The served-model signal (alpha-engine-config-I8165) ──────────────────────
#
# Keys stamped onto `director/{run_date}/action_plan.json`. They are EXTRAS on
# `DirectorWeeklyActionPlan` (`extra="allow"`), deliberately not declared
# fields: a declared field becomes a field the LLM is ASKED to produce, and a
# plan cannot be trusted to report its own degradation — the same reasoning
# `director/verdict.py::stamp_plan_artifact` gives for keeping the correctness
# verdict off the schema. `resolved_model` has been stamped this way since
# config#1673; these three join it.
#
# `grading/tiles/director_quality.py` reads them back off
# `director/latest/action_plan.json` to render the Report Card component. The
# literals are duplicated there rather than imported: the Report Card Lambda
# does not package `director/`, and the tile already reads its other artifact's
# keys as literals. Change one, change both — the contract test
# `tests/test_director_route_degradation.py::TestArtifactContract` fails if the
# producer's keys and the consumer's keys drift apart.
PLAN_KEY_ROUTE_DEGRADED = "route_degraded"
PLAN_KEY_SERVED_MODEL = "served_model"
PLAN_KEY_ROUTE_PRIMARY_MODEL = "route_primary_model"
PLAN_KEY_DEGRADED_REASON = "route_degraded_reason"


def _stamp_route_degradation(plan, *, served_model, primary_model) -> bool | None:
    """Mark the plan artifact with whether a FALLBACK model produced it.

    Returns the value stamped as ``route_degraded``: ``True`` (a fallback
    served), ``False`` (the group's primary served), or ``None`` (unknowable —
    see below). Never raises: this is telemetry stamped onto a plan that has
    already been produced, and losing the stamp must not lose the plan.

    **Why this exists.** ``ultra`` gained a second arm on 2026-08-22
    (alpha-engine-config-I8165) and that arm — ``deepseek-v4-pro`` — is
    deliberately WEAKER than the primary for this call. It is admitted for
    availability (model-router-policy R33) and nothing else. So a fallback
    plan is a **degraded** plan, and substituting one for a champion-produced
    plan without saying so would launder a quality regression into next week's
    baseline: every downstream reader — the console Director page, the retro
    judge grading it, the Report Card trend — would compare a weaker model's
    output against a stronger model's history with nothing marking the seam.
    Recording the served model is the condition on which the second arm was
    ruled acceptable at all, not a nicety attached to it.

    **Why the comparison is against ``primary_model`` and not the deployment
    id.** ``result.model`` is the BILLABLE UPSTREAM id
    (``krepis.llm._resolve_group_served_model`` resolves ``ultra-{entry}`` back
    through the registry), and ``route["primary_model"]`` is the same shape —
    ``glm-5.2``, not ``ultra-glm-5.2-direct``. Comparing an upstream id against
    a deployment id is the category error that pinned ``DirectorRouteFallback``
    at 1 on every healthy run through the router (alpha-engine-config-I6185);
    it is not repeated here.

    **``None`` is a real answer and is not collapsed into ``False``.** When the
    route declared no primary, or the response reported no model, this cannot
    tell "the champion served" from "nobody looked" — and `principles.md` §2.7
    is that *no data* is never rendered as green. The Report Card component
    renders it N/A-MISSING-INPUT rather than a passing 0.
    """
    reason = None
    if not primary_model or not served_model:
        degraded = None
        reason = (
            "served-model unknown: "
            f"primary_model={primary_model!r} served_model={served_model!r} — "
            "cannot distinguish a champion-served plan from an unmeasured one"
        )
    else:
        degraded = served_model != primary_model
        if degraded:
            reason = (
                f"plan produced by FALLBACK model {served_model!r}, not the "
                f"{DIRECTOR_GROUP!r} group's primary {primary_model!r} — a "
                "weaker model served, so this plan is not comparable to a "
                "champion-produced one"
            )

    try:
        setattr(plan, PLAN_KEY_ROUTE_DEGRADED, degraded)
        setattr(plan, PLAN_KEY_SERVED_MODEL, served_model)
        setattr(plan, PLAN_KEY_ROUTE_PRIMARY_MODEL, primary_model)
        setattr(plan, PLAN_KEY_DEGRADED_REASON, reason)
    except Exception:
        logger.exception(
            "Director: failed to stamp route degradation onto the plan — the "
            "artifact and the Report Card cannot tell which model served this "
            "week (served=%s primary=%s)", served_model, primary_model,
        )
        return degraded

    if degraded:
        logger.warning("Director plan DEGRADED: %s", reason)
    elif degraded is None:
        logger.warning("Director plan route degradation UNKNOWN: %s", reason)

    return degraded


# ── The latency signal (alpha-engine-config-I7311) ───────────────────────────
#
# Until 2026-08-14 the duration of the Director's plan call was recorded
# NOWHERE. Reconstructing it took subtracting two log timestamps by hand
# (`Director route:` → the `HTTP Request: POST` line), which is why a 2.4×
# latency climb over ten days — 87s on 2026-08-04, 135s on 2026-08-13, 205s on
# 2026-08-14 against the same route, same model and same registry max_tokens —
# was first noticed as a hard SF failure rather than as a trend.
#
# `DIRECTOR_PLAN_CEILING_S` is the WALL — a resource fact about the Lambda.
# Amber is a TREND fact about the model: how close this call is to needing more
# time than it has ever needed. Those are different quantities, so amber is
# anchored on the measured requirement, not on the wall.
#
# It was 0.6 × the ceiling until 2026-08-22, which read as ceiling-independent
# and was not: raising the ceiling 340 -> 600 in the same change would have
# moved amber 204s -> 360s, above every duration the call has ever survived,
# and the trend signal would have gone dark exactly as the trend continued
# (alpha-engine-config-I7311). The original comment claimed a fraction "can
# never silently move the amber line up" — a fraction OF THE WALL is precisely
# what does.
#
# The anchor: five uncensored plan calls measured 2026-08-13 against this route
# ran 170.1 / 183.0 / 194.9 / 199.5 / 228.0s, every one `finish_reason: stop`.
# 228.0 is therefore the slowest duration this call is known to REQUIRE, and
# 0.9 × it is 205.2s — at 99-109 tok/s, the call already drawing ~21k completion
# tokens, which is the regime the 2026-08-14 run was in the run BEFORE the one
# that failed. A run that crosses it has not failed and loses nothing: it says
# so once, with the numbers attached, while there is still a week to act.
#
# ── Re-anchored 2026-08-22, 228.0 -> 356.9 (alpha-engine-config-I8163) ───────
#
# 228.0 was the slowest of five uncensored calls on 2026-08-13. Two later
# uncensored calls have since exceeded it, read from this module's own EMF
# records in /aws/lambda/alpha-engine-evaluator-director:
#
#   2026-08-15 16:43Z  231.4s  outcome=ok  41 carry-over items  22,724 completion tokens
#   2026-08-22 16:04Z  356.9s  outcome=ok  32 carry-over items  32,643 completion tokens
#
# so the old value no longer describes "the slowest duration this call is known
# to REQUIRE" — it describes a call that stopped existing a week ago. Left
# alone it also breaks the signal in the way this file already warns about for
# `DirectorRouteFallback`: amber at 0.9 x 228 = 205.2s is below EVERY duration
# the call has recorded since, so `DirectorPlanLatencyAmber` would pin at 1 on
# every run, and an alarm that is always on is indistinguishable from one that
# is stuck.
#
# **This is a PRE-cap anchor and is known to be one.** The re-measurement
# alpha-engine-config-I8163 asks for cannot be taken here: it needs uncensored
# calls made through this Lambda AFTER the carry-over cap below is deployed,
# and the first of those is the next weekly run. Guessing a lower number to
# stand in for it would be the stale-anchor failure with the sign flipped.
# Tracked for re-anchoring from post-cap runs: alpha-engine-config-I8200.
#
# One thing the two measurements above already establish, and which the next
# re-anchor should be read against: duration tracks COMPLETION tokens, not
# carry-over count. 356.9s at 32 items was slower than 231.4s at 41, and both
# sit at 91-98 tok/s. So the cap's contribution is bounded by how much of the
# output it removes (one `carryover_review` disposition line per elided row),
# not by the prompt characters it saves.
DIRECTOR_PLAN_MEASURED_MAX_S = 356.9
DIRECTOR_PLAN_AMBER_FRACTION = 0.9


def _plan_amber_threshold_s(measured_max_s: float = None) -> float:
    """Seconds at which the plan call is reported AMBER.

    Deliberately takes the measured requirement, not the ceiling: the ceiling
    is what the invocation can afford and moves with the Lambda's budget; this
    line moves only when the model's own measured need is re-measured.
    """
    return DIRECTOR_PLAN_AMBER_FRACTION * (
        DIRECTOR_PLAN_MEASURED_MAX_S if measured_max_s is None else measured_max_s
    )


def _emit_plan_latency(
    *,
    elapsed_s: float,
    outcome: str,
    prompt_chars: int,
    carryover_items: int,
    carryover_omitted: int = 0,
    usage=None,
    ceiling_s: float | None = None,
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
    # The bound this attempt actually ran under — the budget's quote when the
    # caller knows it (a late invocation is quoted less than the ceiling), the
    # static ceiling otherwise (tests, local runs). Published so the reader can
    # tell "the model got slower" from "this invocation had less to give".
    effective_ceiling_s = (
        DIRECTOR_PLAN_CEILING_S if ceiling_s is None else float(ceiling_s)
    )
    amber_s = _plan_amber_threshold_s()
    over_amber = elapsed_s >= amber_s
    record = {
        "DirectorPlanLatencySeconds": round(elapsed_s, 3),
        "DirectorPlanLatencyAmber": 1 if over_amber else 0,
        "DirectorPlanCeilingSeconds": effective_ceiling_s,
        "DirectorPlanAmberSeconds": round(amber_s, 1),
        "DirectorPlanPromptChars": int(prompt_chars),
        "DirectorPlanCarryoverItems": int(carryover_items),
        # The count ELIDED by the prompt cap (alpha-engine-config-I8163).
        # Published on every run, including the 0 of an uncapped run: an
        # omission metric that only appears when something was omitted is
        # indistinguishable from a dead emitter, which is the same
        # `principles.md` §2.7 rule that puts the 0 on DirectorRouteFallback.
        "DirectorPlanCarryoverOmitted": int(carryover_omitted),
        "DirectorPlanPromptTokens": int(getattr(usage, "input_tokens", 0) or 0),
        "DirectorPlanCompletionTokens": int(getattr(usage, "output_tokens", 0) or 0),
        "DirectorPlanReasoningTokens": int(getattr(usage, "reasoning_tokens", 0) or 0),
        "outcome": outcome,
        "Group": DIRECTOR_GROUP,
    }
    if over_amber:
        logger.warning(
            "Director plan call AMBER: %.1fs >= %.1fs (%.0f%% of the %.0fs "
            "slowest measured healthy call) — outcome=%s prompt_chars=%d carryover_items=%d "
            "prompt_tokens=%d completion_tokens=%d (reasoning=%d). The call "
            "has not failed; it is measurably closer to the ceiling than it "
            "was. alpha-engine-config-I7311: the inputs and the output this "
            "call must produce both grow, so this number trends up on its own "
            "while the measured baseline does not. carryover_omitted=%d "
            "(alpha-engine-config-I8163) — a non-zero omission means the cap "
            "is already binding, so the ledger, not the prompt, is where the "
            "next reduction has to come from.",
            elapsed_s, amber_s, DIRECTOR_PLAN_AMBER_FRACTION * 100,
            DIRECTOR_PLAN_MEASURED_MAX_S, outcome, record["DirectorPlanPromptChars"],
            record["DirectorPlanCarryoverItems"],
            record["DirectorPlanPromptTokens"],
            record["DirectorPlanCompletionTokens"],
            record["DirectorPlanReasoningTokens"],
            record["DirectorPlanCarryoverOmitted"],
        )
    else:
        logger.info(
            "Director plan call %s in %.1fs (amber at %.1fs, ceiling %.0fs) — "
            "prompt_chars=%d carryover_items=%d carryover_omitted=%d "
            "prompt_tokens=%d completion_tokens=%d",
            outcome, elapsed_s, amber_s, effective_ceiling_s,
            record["DirectorPlanPromptChars"],
            record["DirectorPlanCarryoverItems"],
            record["DirectorPlanCarryoverOmitted"],
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
                        {"Name": "DirectorPlanCarryoverOmitted", "Unit": "Count"},
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
    #: ``_invoke_with_retry``. This is the QUOTED per-attempt timeout the client
    #: was actually built with, not the static ceiling.
    #:
    #: It was the static ceiling until 2026-08-22, which was sound only while
    #: the ceiling WAS the per-attempt cost. Now that the ceiling is a loose
    #: upper bound (840s) and the budget quotes the real figure, using the
    #: ceiling here would make ``can_afford(cost + delay)`` false on the first
    #: retry check of every invocation — silently converting a documented
    #: two-attempt loop into one attempt for *fast* failures too, which is the
    #: case the retry exists for (a connection reset at t=3s must still get a
    #: second try; a call censored at its full quote must not).
    attempt_cost_s = DIRECTOR_PLAN_CEILING_S

    #: Inter-chunk silence one streamed attempt may contain — the LIVENESS
    #: bound, as distinct from ``attempt_cost_s`` above, which is how long the
    #: invocation budget is told to expect the attempt to take. Injectable so a
    #: test can drive the bound without reaching into the module constant, and
    #: defaulted so every existing caller and test double is unchanged.
    idle_timeout_s = DIRECTOR_PLAN_IDLE_TIMEOUT_S

    def __init__(self, client, *, director_model: str, primary_model: str | None = None,
                 attempt_cost_s: float | None = None, idle_timeout_s: float | None = None):
        if attempt_cost_s is not None:
            self.attempt_cost_s = attempt_cost_s
        if idle_timeout_s is not None:
            self.idle_timeout_s = idle_timeout_s
        self._client = client
        self._director_model = director_model
        #: The group's declared primary, as the BILLABLE UPSTREAM id
        #: (``route["primary_model"]`` — ``glm-5.2``, not
        #: ``ultra-glm-5.2-direct``). Compared against ``result.model`` after
        #: every call to decide whether a fallback produced the plan
        #: (alpha-engine-config-I8165). Defaults to ``None`` so injected test
        #: doubles and older callers keep working — and ``None`` stamps
        #: ``route_degraded: None``, never a falsely-reassuring ``False``.
        self._primary_model = primary_model
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
            # alpha-engine-config-I8164. krepis re-assembles the streamed
            # response into the same `ChatCompletion`-shaped object a
            # non-streamed call returns, so the empty-content diagnostics, the
            # budget-exhaustion guard, the served-model resolution and usage
            # extraction all run unchanged — `_stamp_route_degradation` below
            # reads `result.model` exactly as before. This is a change to how
            # the bytes arrive, not to what this method returns.
            #
            # It does NOT silently degrade: a route that does not declare
            # `capabilities.streaming` raises `StreamingUnsupportedError`
            # rather than sending a non-streamed request, because a non-streamed
            # request would come back as a perfectly valid completion carrying
            # the exact opaque-deadline failure envelope this change removes.
            # `_default_llm` therefore states the requirement at RESOLVE time
            # (`requires=("streaming",)`), so the failure, when there is one,
            # names the registry entry to fix instead of appearing 600s into a
            # weekly run.
            stream=True,
            idle_timeout=self.idle_timeout_s,
        )
        self.last_usage = getattr(result, "usage", None)
        plan: DirectorWeeklyActionPlan = result.parsed
        plan.director_model = self._director_model
        plan.resolved_model = result.model
        # I8165: the group has a second, deliberately weaker arm now, so which
        # model served is a QUALITY fact about this plan and not just a cost
        # attribution. `resolved_model` alone left every consumer to work that
        # out by comparing against a registry it does not have.
        _stamp_route_degradation(
            plan, served_model=result.model, primary_model=self._primary_model,
        )
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
    #
    # `requires=("streaming",)` states the REQUEST SHAPE, which is the only
    # other thing (besides where it runs) this module is allowed to say about
    # routing — `model-router-policy` R32: the consumer names what its shape
    # requires, the derivation drops members that do not declare it BEFORE a
    # primary is chosen, and a group with no such member fails at RESOLVE time
    # naming the group, the capability and each rejected member.
    #
    # Declaring it is not decoration. Without it the chain would be filtered
    # only on reachability and the first `structured(stream=True)` would raise
    # `StreamingUnsupportedError` deep inside the call instead — the same
    # outcome, reported later and with less to act on. With it, a registry in
    # which `ultra`'s primary has lost `capabilities.streaming` is a named
    # failure before any token is billed.
    #
    # Measured against the registry on 2026-08-25 (alpha-engine-config-I8164):
    # before the declaration landed this call raised
    # `CapabilityUnavailableError` from BOTH `laptop` and `lambda`; after it,
    # both resolve `litellm_proxy` / `ultra-glm-5.2-direct` with
    # `supports_streaming=True`. A group route declares its PRIMARY's flag.
    spec, route = resolve_group_spec(
        DIRECTOR_GROUP,
        exec_context=DIRECTOR_EXEC_CONTEXT,
        wire="openai",
        requires=("streaming",),
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
    # It also cannot be raised alone, for a reason outside this repo: the router
    # edge's `proxy_read_timeout` bounds every request through it, so any client
    # deadline at or above the edge's hands the timeout to nginx and the failure
    # stops being attributable to this call site. That edge was 360s, which is
    # why this ceiling sat at 340 — and why raising the ceiling was blocked on a
    # nous-ergon-ops change. Both moved in the 2026-08-22 arc: the edge is now
    # 900s and the quote below is what the client owns.
    #
    # `max_retries=0` is not a loss of resilience — the retry moved to
    # `_invoke_with_retry`, the only loop that can see the invocation deadline.
    # See `_CLIENT_MAX_RETRIES`.
    #
    # config#6904: the literal is a CEILING, not the budget. The old comment
    # here claimed one retry "bounds the worst case to 2×340s inside the
    # Lambda's 900s budget" — but the Phase-G retro judge adds 2×120s in the
    # same invocation, so the worst case was 920s against a 900s function
    # timeout, before a single S3 write. The quote below derives the per-attempt
    # budget from the time actually remaining, so a late invocation declines the
    # call (raising BudgetExhausted, which the caller records) instead of
    # starting one the wall will kill mid-flight with no artifact and no cause.
    #
    # I7311 (2026-08-22): `attempts=2` was how that overrun was avoided, and it
    # halved every plan call to pay for work it never named. The retro judge's
    # time is now an explicit `downstream_s` reservation, so the plan call is
    # quoted as ONE attempt against everything else the invocation can afford.
    # `_invoke_with_retry` still retries — funded by whatever is left, which is
    # enough after a fast failure and correctly nothing after a full-quote one.
    plan_budget = budget or UNBOUNDED
    quoted_timeout = plan_budget.quote(
        "director-plan",
        DIRECTOR_PLAN_CEILING_S,
        attempts=1,
        downstream_s=RETRO_JUDGE_RESERVE_S,
    )
    client = LLMClient(
        spec,
        callsite_id="director-plan",
        timeout=quoted_timeout,
        max_retries=_CLIENT_MAX_RETRIES,
        **api_kwargs,
    )
    return _KrepisStructuredDirector(
        client,
        director_model=route["deployment_id"],
        # The upstream id of the group's declared primary — the yardstick
        # `_stamp_route_degradation` measures `result.model` against.
        primary_model=route.get("primary_model"),
        attempt_cost_s=quoted_timeout,
    )


#: Marker the carry-over section header carries so the emitted latency record
#: can report how many ledger rows this prompt made the model dispose of —
#: the quantity that grew 19 → 41 while the call duration grew 87s → 205s
#: (alpha-engine-config-I7311). Parsed rather than threaded through
#: ``build_messages``' signature because ``_invoke_with_retry`` is handed
#: MESSAGES, not the artifacts they were built from, and widening that
#: boundary to carry telemetry would couple the retry loop to the card.
#:
#: Since alpha-engine-config-I8163 this counts the rows the prompt actually
#: CARRIES, not the rows the ledger holds — the two diverge the moment the cap
#: below bites, and a metric named ``CarryoverItems`` that reported the ledger
#: size would report a quantity this call no longer pays for.
_CARRYOVER_COUNT_MARKER = "carry-over ledger, active items="

#: Companion marker for the rows the cap ELIDED. Publishing the carried count
#: without it is silent truncation, which `sf-pipeline-policy.md` §2.3 forbids:
#: a plan produced from a truncated ledger has to say it was truncated, and it
#: has to say so on a durable surface (the EMF record is stamped onto the
#: archived plan by ``_invoke_with_retry``), not only in a log line that ages
#: out (alpha-engine-config-I8163).
_CARRYOVER_OMITTED_MARKER = "omitted from this prompt="


# ── The prompt-side bound on the carry-over ledger (alpha-engine-config-I8163) ─
#
# `DIRECTOR_PLAN_CEILING_S` is now derived from AWS Lambda's 900s function
# maximum, so **there is no third raise available**. The next time this call
# outgrows its budget the only remaining lever is less work per call, and this
# is that lever: the ledger is the one input to this prompt that grows on its
# own (19 → 41 active items over the ten days the call went 87s → 205s).
#
# `carryover.ACTIVE_LEDGER_MAX_ITEMS` (40) does not cover this. That bound is
# applied by `merge_plan_into_ledger` AFTER the plan is built, so the ledger the
# PROMPT reads is the pre-merge one and nothing in the read path bounds it at
# all — a growth-limiting bound that runs strictly downstream of the thing it is
# supposed to protect.
#
# The derivation:
#
#     20 = every P0 and P1 row the live ledger carries (6 + 12 = 18,
#          measured 2026-08-21 on s3://alpha-engine-research/director/
#          carryover_ledger.json), plus two rows of slack
#        = 0.5 × carryover.ACTIVE_LEDGER_MAX_ITEMS
#
# — i.e. the cap is set where today's whole above-P2 backlog fits inside it, so
# in the normal case it elides only P2/P3 tail and the model sees everything it
# would have seen anyway. It is a real ceiling only in the case it exists for.
#
# What a row costs, measured 2026-08-21 by rendering the live ledger against the
# live report card through this exact function: a rendered row block is 282
# chars on average and 457 at the worst, and the section as a whole was 8,320
# chars of a 33,878-char prompt at 28 rows. But the prompt cost is the smaller
# half. Every row also buys an OUTPUT obligation — `DirectorWeeklyActionPlan`'s
# `carryover_review` takes one disposition line per row — and output is what
# this call actually pays for: the 2026-08-22 run drew 32,643 completion tokens
# (25,590 of them reasoning) against 15,328 prompt tokens. A row costs ~300
# prompt chars and a paragraph of reasoning.
CARRYOVER_PROMPT_MAX_ITEMS = 20

# A second, independent bound, because the item cap alone cannot bound the
# section: `title` and `rationale` are model-authored free text and a single
# pathological row could be arbitrarily long. This is the one that makes the
# bound hold BY CONSTRUCTION rather than by an assumption about row size, and
# it is what `tests/test_director_carryover_prompt_bound.py` pins.
#
#     12,000 = 437 (the header + instruction block, measured)
#            + 20 × 457 (the largest rendered row block measured on the live
#                        ledger, applied to every row rather than the mean)
#            + ~300 for the elision summary
#            = 9,877, rounded up to 12,000 for headroom
#
# Deliberately NOT a truncation of individual rows: a row cut mid-sentence is a
# row the model can misread, and a half-rendered claim is worse than an elided
# one that is counted and named. When the budget binds, whole rows move to the
# omitted set and are reported there.
CARRYOVER_PROMPT_CHAR_BUDGET = 12_000

#: Charged against the budget before any row is, so the budget bounds the WHOLE
#: section and not just its body. It covers the header line, the live-card
#: instruction block (437 chars measured), the P0 carve-out notice, the
#: contradicted-row footer and the elision summary (~700 chars) — every part of
#: the section whose size does not depend on how many rows survive the cap.
#: Without it a section could pass the row loop at exactly the budget and then
#: exceed it by adding the line that reports the elision, which is the one line
#: that is guaranteed to be present whenever the budget bound in the first
#: place.
_SECTION_OVERHEAD_CHARS = 1_400

#: The P0 carve-out (alpha-engine-config-I8163). A P0 is never elided, even when
#: doing so breaches BOTH bounds above. If the active P0 set alone exceeds the
#: cap, that is a finding about the backlog rather than a prompt-sizing problem,
#: and the section says so in the prompt and in the metric — silently dropping
#: the Director's own highest-priority carried commitment to save tokens is the
#: one failure this whole change would not be worth causing.
_P0_NEVER_ELIDED = True


def _carryover_item_count(messages: list) -> int:
    """How many active ledger rows this prompt carries; 0 when there is no
    carry-over section (first cycle, empty ledger, or an injected test
    double's hand-built messages). Never raises."""
    return _marker_count(messages, _CARRYOVER_COUNT_MARKER)


def _carryover_omitted_count(messages: list) -> int:
    """How many active ledger rows the cap elided from this prompt. 0 when
    nothing was elided AND when there is no carry-over section at all — the
    two are distinguished by ``_carryover_item_count``, which is 0 only in the
    second case whenever the ledger is non-empty. Never raises."""
    return _marker_count(messages, _CARRYOVER_OMITTED_MARKER)


def _marker_count(messages: list, marker: str) -> int:
    """First run of digits following ``marker`` in any message; 0 if absent."""
    for _, content in messages:
        text = str(content)
        idx = text.find(marker)
        if idx < 0:
            continue
        digits = ""
        for ch in text[idx + len(marker):]:
            if not ch.isdigit():
                break
            digits += ch
        if digits:
            return int(digits)
    return 0


def select_carryover_rows(rows: list[dict]) -> tuple[list[dict], list[dict], bool]:
    """Split active ledger rows into ``(shown, omitted, p0_over_cap)``.

    Pure and total: no I/O, no exceptions on malformed rows, and the same
    ledger always splits the same way. Separated from the renderer so the
    SELECTION — which changes what the Director decides, and is therefore the
    design-bearing half of alpha-engine-config-I8163 — can be tested and argued
    about without constructing a report card.

    Ordering is ``carryover.order_for_prompt`` (priority, then weeks carried,
    then first-seen); its full rationale lives there, next to the retirement
    ordering it deliberately differs from.

    ``p0_over_cap`` is True when the active P0 set alone is at or above
    :data:`CARRYOVER_PROMPT_MAX_ITEMS`. Every P0 is still returned in ``shown``
    — see :data:`_P0_NEVER_ELIDED` — so the flag is a statement about the
    BACKLOG, not a fallback the caller has to handle.
    """
    ordered = order_for_prompt(rows)
    p0 = [r for r in ordered if is_p0(r)]
    rest = [r for r in ordered if not is_p0(r)]
    if _P0_NEVER_ELIDED and len(p0) >= CARRYOVER_PROMPT_MAX_ITEMS:
        return p0, rest, True
    room = CARRYOVER_PROMPT_MAX_ITEMS - len(p0)
    shown = order_for_prompt(p0 + rest[:room])
    return shown, rest[room:], False


def _row_block(it: dict, status_map: dict) -> tuple[list[str], bool]:
    """Render one ledger row (header + live-card annotation) -> (lines, contradicted)."""
    lines = [
        f"  - [{it.get('id')}] {it.get('title')} "
        f"(status={it.get('status')}, owner={it.get('proposed_owner')}, priority={it.get('priority')})"
    ]
    if not status_map:
        return lines, False
    # Wider than evidence_still_adverse's evidence-only scope on purpose:
    # the claim the model restates lives in the TITLE (that is where
    # "#7289" and "collapse" sat), so the title and rationale are read
    # too. Safe to widen precisely because this annotates a prompt and
    # reopens nothing.
    hits = resolve_cited_metrics(
        list(it.get("evidence") or []) + [it.get("title") or "", it.get("rationale") or ""],
        status_map,
    )
    if not hits:
        lines.append(
            "      live card: no metric named by this row appears on this week's card "
            "— its claim is UNVERIFIABLE against current data, so do not restate it as fact."
        )
        return lines, False
    rendered = ", ".join(f"{n}={s or 'UNKNOWN'}" for n, s in sorted(hits.items()))
    lines.append(f"      live card: {rendered}")
    if all(s not in ADVERSE_STATUSES for s in hits.values()):
        lines.append(
            "      ⚠ CONTRADICTED BY THIS WEEK'S CARD — every metric this row names "
            "is non-adverse now. Do NOT carry it forward on its existing wording."
        )
        return lines, True
    return lines, False


def _elision_summary(
    omitted: list[dict], *, shown: int, budget_bound: bool, cap_bound: bool
) -> str:
    """The single counted line that replaces the elided tail.

    A bare count is a weaker artifact than it looks: a reader who is told
    "17 items were omitted" cannot tell whether the Director dropped a month of
    unresolved P1s or a tail of P3 monitoring notes. So the line carries the
    priority distribution, the oldest first-seen date, and the longest
    carry-count in the omitted set — enough to answer "should I go and look?"
    without opening the ledger.
    """
    from director.carryover import _parse_run_date  # local: pure helper, no cycle

    dist = {}
    for r in omitted:
        dist[str(r.get("priority") or "unknown")] = dist.get(str(r.get("priority") or "unknown"), 0) + 1
    rendered_dist = ", ".join(f"{k}×{v}" for k, v in sorted(dist.items()))
    first_seen = sorted(
        d for d in (_parse_run_date(r.get("first_seen")) for r in omitted) if d
    )
    oldest = first_seen[0].isoformat() if first_seen else "unknown"
    longest = max((carry_count(r) for r in omitted), default=0)
    reasons = []
    if cap_bound:
        reasons.append(f"the {CARRYOVER_PROMPT_MAX_ITEMS}-item prompt cap")
    if budget_bound:
        reasons.append(
            f"the {CARRYOVER_PROMPT_CHAR_BUDGET:,}-character section budget"
        )
    why = " and ".join(reasons) or "the prompt bound"
    return (
        f"  + {len(omitted)} further active items are NOT shown above "
        f"({_CARRYOVER_OMITTED_MARKER}{len(omitted)}): {rendered_dist}; "
        f"oldest first seen {oldest}, longest carried {longest} consecutive runs. "
        f"They were elided by {why} — the {shown} shown are the highest-priority, "
        f"longest-carried rows (alpha-engine-config-I8163). Every omitted item "
        f"remains OPEN in the carry-over ledger and is untouched by this plan. "
        f"Their absence here is a PROMPT BOUND, not a disposition: do not mark "
        f"them resolved, dropped or complete, do not restate them, and scope "
        f"`carryover_review` to the items listed above."
    )


def _carryover_context(carryover: dict | None, report_card: dict | None = None) -> str:
    """Render the carry-over ledger for the prompt, EACH ROW RECONCILED
    AGAINST THIS WEEK'S CARD (alpha-engine-config-I8178), and BOUNDED
    (alpha-engine-config-I8163).

    Until 2026-08-22 this rendered ``id``/``title``/``status``/``owner``/
    ``priority`` straight out of the ledger and nothing else. Nothing on the
    path between the ledger and the model compared a row's claim to the
    current numbers, so a claim that entered the ledger once was restated
    every week thereafter on its own authority, and the restatement was
    written back to the ledger by ``merge_plan_into_ledger`` — a loop with no
    term that reads the world.

    ``loop_verification`` was NOT that missing check, and this is the part
    worth being precise about, because its existence is what made the gap
    look covered. It compares evidence to the card, but (a) it acts only on
    ``issue_number`` — null on all 28 live rows measured 2026-08-22, so it
    ``continue``s past every one of them — (b) it is inside the §2.3a
    ``actions_withheld`` gate and had been withheld for three consecutive
    cycles under contamination=UNKNOWN, and (c) decisively, even on a clean
    run it corrects the GITHUB ISSUE and never the ledger row or this
    string. All three had to hold for the failure; only (c) is structural,
    and (c) alone is sufficient. So the annotation below is deliberately NOT
    routed through that pass: it is computed here, from the card already in
    hand, with no GitHub call, no ``issue_number``, and no withholding gate
    — none of which it needs, because it mutates nothing.

    The measured case (2026-08-22): ``inference-coverage-critical`` carried
    the title "Fix inference_coverage collapse — predictor Lambda KeyError
    (#7289)". ``#7289`` had been closed 8 days; ``inference_coverage`` read
    1.0 / GREEN on the very card the Director was reading. It was published
    as P0 #4 anyway. Three of 28 rows cited only GREEN metrics; nine cited
    nothing worse than WATCH.

    Shown rows are annotated, never dropped or reordered *by the annotation*.
    A GREEN metric does not prove an item is finished — the item may name work
    the metric does not cover, and suppressing it here would be the
    close-and-look-away failure ``loop_verification``'s own docstring refuses.
    The reader is told what the card says and left to weigh it; the fix for
    reasoning from stale text is putting the current fact NEXT TO the text, not
    deleting the text.

    **The cap is a separate mechanism from the annotation and must stay that
    way.** The annotation says a row may be wrong; the cap says a row did not
    fit. Neither is allowed to become the other: eliding a row because the card
    contradicts it would be exactly the suppression the paragraph above
    refuses, so the selection in :func:`select_carryover_rows` reads only
    ``priority`` and ``carry_count`` and never the annotation's verdict. That
    matters here specifically because the ledger is known to contain stale rows
    (alpha-engine-config-I8178, open) — a selection rule that assumed every row
    were true would rank on a claim nothing has re-measured.

    The rendered section is bounded by :data:`CARRYOVER_PROMPT_MAX_ITEMS` and
    :data:`CARRYOVER_PROMPT_CHAR_BUDGET`; whatever does not fit is replaced by
    one counted, characterised summary line (:func:`_elision_summary`) rather
    than disappearing.
    """
    if not carryover or not carryover.get("items"):
        return "No prior action plan on record (this is the first cycle or the ledger is empty)."
    items = list(carryover.get("items") or [])
    status_map = component_status_map(report_card or {})
    shown, omitted, p0_over_cap = select_carryover_rows(items)

    header: list[str] = []
    if status_map:
        header.append(
            "  Each row is annotated with THIS WEEK'S status for every card metric it "
            "names ('live card:'). Those statuses come from the card above and OVERRIDE "
            "the row's own text, which may be weeks old and is not re-measured when it "
            "is carried. Where a row is marked CONTRADICTED, do not restate its claim: "
            "either re-ground the item in what the card now says, or mark it resolved."
        )
    if p0_over_cap:
        header.append(
            f"  ⚠ P0 SET EXCEEDS THE PROMPT CAP: {len(shown)} of {len(items)} active "
            f"items are priority P0, at or above the {CARRYOVER_PROMPT_MAX_ITEMS}-item cap. "
            "Every P0 is carried regardless of the cap — a P0 is never elided — so this "
            "section is deliberately over its budget. A P0 set this large is itself a "
            "finding about the backlog and belongs in top_risks."
        )

    # The char budget is applied to the ROW BLOCKS, in rank order, and a row
    # that does not fit moves whole into the omitted set. Applied after the P0
    # carve-out, and never to a P0: the budget yields to the carve-out, not the
    # other way round, or the carve-out would be advisory.
    budget_bound = False
    body: list[str] = []
    spilled: list[dict] = []
    n_contradicted = 0
    used = _SECTION_OVERHEAD_CHARS + sum(len(line) + 1 for line in header)
    for it in shown:
        block, contradicted = _row_block(it, status_map)
        cost = sum(len(line) + 1 for line in block)
        if spilled or (not is_p0(it) and used + cost > CARRYOVER_PROMPT_CHAR_BUDGET):
            # Once one row has spilled, every later (lower-ranked) row spills
            # too — otherwise a short P3 could leapfrog a long P1 and the
            # published ordering would not describe what was actually carried.
            spilled.append(it)
            continue
        used += cost
        body.extend(block)
        n_contradicted += 1 if contradicted else 0
    if spilled:
        budget_bound = True
        omitted = spilled + omitted
        # Identity, not equality: two ledger rows can compare equal (a dict
        # `==` over the same fields) without being the same row, and `in` over
        # a list of dicts would then drop the wrong one.
        _spilled_ids = {id(r) for r in spilled}
        shown = [it for it in shown if id(it) not in _spilled_ids]

    lines = [
        f"Last week's open action items ({_CARRYOVER_COUNT_MARKER}{len(shown)}"
        + (f" of {len(items)}" if omitted else "")
        + "):"
    ]
    lines.extend(header)
    lines.extend(body)
    if status_map and n_contradicted:
        lines.append(
            f"  ({n_contradicted} of {len(shown)} carry-over rows shown are contradicted by this "
            f"week's card. A carried row is a claim about the PAST; the card is the present.)"
        )
    if omitted:
        lines.append(
            _elision_summary(
                omitted,
                shown=len(shown),
                budget_bound=budget_bound,
                cap_bound=len(omitted) > len(spilled),
            )
        )
    return "\n".join(lines)


def build_messages(report_card: dict, *, carryover: dict | None = None, roadmap_digest: str | None = None,
                   resolved_digest: str | None = None) -> list:
    """Assemble (system, human) messages for the Director call."""
    human = [
        summarize_report_card(report_card),
        "",
        _carryover_context(carryover, report_card),
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
    carryover_omitted = _carryover_omitted_count(messages)
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
                carryover_omitted=carryover_omitted,
                usage=None,
                ceiling_s=cost,
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
                carryover_omitted=carryover_omitted,
                usage=getattr(llm, "last_usage", None),
                ceiling_s=cost,
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
