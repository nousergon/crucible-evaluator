"""
retro.py — the Director's Phase-G self-grading retro loop (Layer C).

Each weekly run judges LAST week's plan against THIS week's Report Card — the
realized-outcome feedback loop the in-call ``SelfGrade`` can't provide (it can't
see the future at emission). One structured judge call → a ``RetroGrade``
(grounding / calibration / actionability), reusing the LLM-as-judge rubric
pattern (mirrors research ``evals/judge.py`` ``RubricEvalLLMOutput``).

The LLM is injectable (``llm=``) so build/validate + tests run without a key or
krepis' provider SDKs; ``_default_llm()`` lazily constructs the real client.

**Judge tier: the ``high`` group, deliberately NOT the Director's ``ultra``.**
Grading a plan with the same model that generated it is self-grading bias
(config#1673, judge != generator) — see ``agent.py``'s ``DIRECTOR_GROUP``, and
``_assert_judge_is_not_the_generator``, which enforces it at call time rather
than trusting two constants to stay distinct.
(This paragraph said "Sonnet" / "Opus" until 2026-08-01; both were stale from
the 2026-07-24 migration off direct Anthropic.)

**The judge resolves its GROUP through ``krepis.router``, exactly as the plan
call does** — ``resolve_group_spec(group, exec_context=..., wire="openai")``,
then ``_assert_routed_through_the_proxy`` and ``_warn_on_degraded_route``. The
tier is config (``RETRO_JUDGE_GROUP`` env var, default ``"high"``); the model,
endpoint, provider and credential are the registry's business and appear
nowhere in this file.

Until 2026-08-14 they did appear here: the client was a hand-built
``ModelSpec(provider="openrouter", model="deepseek-v4-pro-max")`` with an
OpenRouter key read from SSM at this call site, egressing straight past the
authenticated LiteLLM edge. Because ``deepseek-v4-pro-max`` is a registry group
handle rather than an OpenRouter model id, every call returned ``400 — not a
valid model ID`` and ``director/{date}/retro.json`` went unwritten for 24 days
while the run summary recorded ``retro: error`` and nothing alarmed
(alpha-engine-config-I6562, -I7326).

Three provenance fields are stamped onto the persisted ``RetroGrade``
(``extra="allow"``) so the dashboard/audit trail can see exactly what ran:
``judge_group`` (the tier asked for), ``judge_model`` (what the registry
resolved it to) and ``resolved_model`` (what the API served).
"""

from __future__ import annotations

import logging
import os

from director.agent import (
    DIRECTOR_EXEC_CONTEXT,
    DIRECTOR_GROUP,
    _CLIENT_MAX_RETRIES,
    _STRUCTURED_ATTEMPTS,
    _assert_routed_through_the_proxy,
    _invoke_with_retry,
    _warn_on_degraded_route,
)
from director.budget import RETRO_JUDGE_RESERVE_S, UNBOUNDED
from director.report_card_digest import summarize_report_card
from director.schema import DirectorWeeklyActionPlan, RetroGrade

logger = logging.getLogger(__name__)

# Per-attempt ceiling for the retro judge (config#6050); a ceiling rather than
# the budget since config#6904 — see director/budget.py.
RETRO_JUDGE_CEILING_S = 120.0

# Lockstep guard: the plan call reserves `budget.RETRO_JUDGE_RESERVE_S` for this
# judge's two attempts. Raising the ceiling here without widening that
# reservation would let the plan call consume time this judge is owed, which is
# the 920s-against-900s arithmetic `director/budget.py` exists to prevent — so
# the two move together or import fails.
assert RETRO_JUDGE_RESERVE_S == RETRO_JUDGE_CEILING_S * 2, (
    f"retro judge reservation drift: director/budget.py reserves "
    f"{RETRO_JUDGE_RESERVE_S}s for this judge but its ceiling is now "
    f"{RETRO_JUDGE_CEILING_S}s × 2 attempts. Update RETRO_JUDGE_RESERVE_S in "
    f"director/budget.py in the same change."
)

# Retro judge capability tier — a GROUP HANDLE resolved through
# ``krepis.router``, never a model id (config#1673; direct Anthropic → direct
# OpenRouter 2026-07-24 → krepis.router.resolve_group_spec 2026-08-14).
#
# Intentionally a DIFFERENT group from ``agent.DIRECTOR_GROUP`` ("ultra") —
# grading a plan with the model that generated it is self-grading bias. That is
# the whole reason this constant exists and is not an import of DIRECTOR_GROUP.
#
# **This is a capability class, not a model.** `principles.md` §2.8: a call site
# addresses a registry group through the router and never a model id, a base
# url, a provider name, or an SDK client it constructs itself. The old
# `RETRO_JUDGE_MODEL_DEFAULT = "deepseek-v4-pro-max"` violated that twice over —
# it named an entry, and it handed that name to a hand-built
# `ModelSpec(provider="openrouter", ...)`. `deepseek-v4-pro-max` is a registry
# group handle that resolves only through the router, so OpenRouter answered
# `400 — not a valid model ID` on every call from 2026-07-21 to 2026-08-14 and
# `director/{date}/retro.json` was not written once in 24 days
# (alpha-engine-config-I6562, -I7326).
#
# There is deliberately no model-id env override. `RETRO_JUDGE_MODEL` used to
# exist and would have "unstuck" the outage with a literal OpenRouter id; that
# is precisely the shortcut that re-pins this call site off the registry and
# hides the router bypass again. The tier is overridable, the model is not.
RETRO_JUDGE_GROUP_DEFAULT = "high"

_RETRO_JUDGE_SCHEMA_NAME = "RetroGrade"


def _judge_group() -> str:
    """The configured retro-judge capability tier: ``RETRO_JUDGE_GROUP`` env
    override if set, else :data:`RETRO_JUDGE_GROUP_DEFAULT`. Read at call time
    (not frozen at import) so an operator/test override takes effect without a
    process restart.

    A GROUP, not a model. See the constant's note above."""
    return os.environ.get("RETRO_JUDGE_GROUP", RETRO_JUDGE_GROUP_DEFAULT)


def _assert_judge_is_not_the_generator(group: str) -> None:
    """The judge tier must not be the Director's own tier (config#1673).

    Enforced at call time rather than by the constant's distinctness alone,
    because ``RETRO_JUDGE_GROUP`` is env-overridable and an operator setting it
    to ``ultra`` would silently turn the retro into self-grading — a grade that
    still looks like a grade and is worth nothing.

    **What this can and cannot establish** (alpha-engine-config-I6052): it
    compares GROUPS. On the proxy route krepis reports ``registry_id`` as a
    group handle — LiteLLM walks the fallback chain internally by design — so
    the consumer cannot see which ENTRY served, and a fully-degraded ultra chain
    reaching the same entry as high's primary remains undetectable from here.
    That residual is owed by the router's own telemetry (model-router-policy
    R21), not by this module. Asserting the half that is knowable is not a
    weakened check; claiming the other half would be a false one.
    """
    if group == DIRECTOR_GROUP:
        raise RuntimeError(
            f"Retro judge group {group!r} is the Director's own generation "
            f"group (config#1673: judge != generator). Grading a plan with the "
            f"tier that wrote it is self-grading bias, and the resulting "
            f"RetroGrade would read as a real grade. Set RETRO_JUDGE_GROUP to "
            f"a different registry group, or unset it to use "
            f"{RETRO_JUDGE_GROUP_DEFAULT!r}."
        )


def _load_retro_prompt() -> str:
    """The tuned retro rubric (gitignored director/retro_prompt.py) if present,
    else the committed generic template."""
    try:
        from director.retro_prompt import RETRO_PROMPT  # type: ignore
        return RETRO_PROMPT
    except Exception:  # ImportError or anything — fall back to the template
        from director.retro_prompt_example import RETRO_PROMPT
        return RETRO_PROMPT


def _split_messages(messages: list) -> tuple[str, str]:
    """``build_messages()``'s ``[("system", ...), ("human", ...)]`` shape ->
    krepis.llm's flat ``(system, user_content)`` call surface. Any non-system
    entries are joined in order — robust to the exact tuple count even though
    ``build_messages`` currently emits exactly one of each."""
    system = ""
    human_parts: list[str] = []
    for role, content in messages:
        if role == "system":
            system = content
        else:
            human_parts.append(content)
    return system, "\n\n".join(human_parts)


class _KrepisStructuredJudge:
    """Adapts a ``krepis.llm.LLMClient`` to the ``.invoke(messages) ->
    RetroGrade`` surface ``director.agent._invoke_with_retry`` expects.

    Keeping this adapter shape (rather than reworking ``_invoke_with_retry``
    or ``grade_prior_plan``) means the retro's corrective-retry wiring and the
    Opus plan-generation path in ``agent.py`` are untouched by the judge-model
    swap — only ``_default_llm`` (below) changes which client backs the
    ``llm`` the retro invokes.
    """

    #: Wall-clock one ``invoke()`` may consume — see
    #: ``director.agent._KrepisStructuredDirector.attempt_cost_s``. The judge
    #: shares `_invoke_with_retry`, so it shares the budget gate.
    attempt_cost_s = RETRO_JUDGE_CEILING_S

    def __init__(self, client, *, judge_group: str, judge_model: str):
        self._client = client
        self._judge_group = judge_group
        self._judge_model = judge_model

    def invoke(self, messages: list) -> RetroGrade:
        system, user_content = _split_messages(messages)
        result = self._client.structured(
            system=system,
            user_content=user_content,
            schema=RetroGrade,
            schema_name=_RETRO_JUDGE_SCHEMA_NAME,
            # Explicit, not krepis' default of 2 — see
            # `director.agent._STRUCTURED_ATTEMPTS`. The retro runs LAST, on
            # whatever budget the plan left, so an inherited hidden doubling
            # lands here first.
            attempts=_STRUCTURED_ATTEMPTS,
        )
        grade: RetroGrade = result.parsed
        # Three provenance fields, each answering a different question, all
        # landing as extra fields — RetroGrade has extra="allow" — so they
        # persist through model_dump()/model_dump_json() unchanged.
        #
        #   judge_group    what this call site ASKED FOR (a capability tier)
        #   judge_model    what the registry RESOLVED that tier to
        #   resolved_model what the API actually served
        #
        # judge_group is new as of the router migration and is the field that
        # makes the audit trail honest: before it, `judge_model` carried a
        # hardcoded literal and read as a resolution when it was a hardcode.
        grade.judge_group = self._judge_group
        grade.judge_model = self._judge_model
        grade.resolved_model = result.model
        return grade


def _default_llm(budget=None) -> _KrepisStructuredJudge:
    """Construct the real structured-output judge client (lazy import).

    Resolves :func:`_judge_group` (default ``"high"``) through
    ``krepis.router.resolve_group_spec()`` — krepis' documented public contract
    for programmatic callers — and builds the client from the returned route:
    provider, deployment_id, api_base_url, and the credential NAME implied by
    ``auth_token_type``. **The registry decides model, endpoint and auth; this
    module decides only which capability tier it wants and where it is
    running.** Exactly the shape ``agent._default_llm`` already uses for the
    plan call; this is that pattern swept to its sibling, not a parallel one
    (`shared-code-policy`, engagement §5: a fix survives the class).

    Migrated: direct Anthropic → direct OpenRouter 2026-07-24 →
    ``krepis.router.resolve_group_spec`` 2026-08-14 (alpha-engine-config-I6562).

    **Why this had to change.** The previous body built
    ``ModelSpec(provider="openrouter", model="deepseek-v4-pro-max")`` by hand
    and fetched ``OPENROUTER_API_KEY`` from SSM itself, so this call site left
    the Lambda for openrouter.ai directly — bypassing the authenticated LiteLLM
    edge, its DLP scanning, and every per-consumer control on it. It had the
    same three defects ``agent._assert_routed_through_the_proxy``'s docstring
    enumerates, and none of that function's protection, because the guard was
    added to ``agent.py`` in this same package and never swept here. It failed
    closed only by accident: ``deepseek-v4-pro-max`` is a registry GROUP HANDLE,
    so OpenRouter returned ``400 — not a valid model ID`` rather than serving
    an unscanned paid call. A valid literal there would have "worked", silently,
    off-registry and off-proxy.

    Both imports stay lazy so tests and the grading path never pull krepis'
    provider SDKs or hit SSM.
    """
    from krepis.llm import LLMClient
    from krepis.router import resolve_group_spec

    judge_group = _judge_group()
    _assert_judge_is_not_the_generator(judge_group)

    # `wire="openai"` because this call site speaks the OpenAI wire format.
    # `exec_context` is the ONLY thing this module says about routing, and it is
    # a statement about WHERE IT IS RUNNING, not about which routes it wants
    # (model-router-policy §2 layer 5, R29). It is shared with the plan call
    # because both run in the same Lambda invocation — a judge that resolved
    # from a different execution context than the generator it grades would be
    # describing a process that does not exist.
    spec, route = resolve_group_spec(
        judge_group,
        exec_context=DIRECTOR_EXEC_CONTEXT,
        wire="openai",
    )
    _assert_routed_through_the_proxy(route)
    _warn_on_degraded_route(
        route, group=judge_group, metric_name="RetroJudgeRouteFallback"
    )

    judge_model = route["deployment_id"]

    # `max_tokens` comes from the ROUTE, not from here. The old body pinned it
    # to 2000 at the call site; that is the same layer-5 duplication as pinning
    # the model, one field over — the registry states each entry's ceiling and
    # this module has no basis to know better. `agent._default_llm` takes it
    # from the spec for exactly this reason. The judge's real bound is WALL
    # CLOCK (`RETRO_JUDGE_CEILING_S` quoted against the remaining invocation
    # budget), which is the resource that actually runs out — the Lambda's 900s
    # ceiling is AWS's service maximum, not a number we chose.

    # `auth_token_type == "placeholder"` maps to api_key_env=None and means the
    # egress proxy holds the real key — there is NOTHING to resolve, and
    # treating that as a missing credential would break every direct route from
    # a context that can reach one. Where a credential IS named and cannot be
    # resolved this RAISES rather than falling through to a 401 that reads as a
    # provider problem (model-router-policy R20: fail closed, loudly). Mirrors
    # `agent._default_llm` exactly.
    api_kwargs = {}
    if spec.api_key_env is not None:
        from krepis.secrets import get_secret

        secret = get_secret(spec.api_key_env, required=False)
        if not secret:
            raise RuntimeError(
                f"Retro judge route for group {judge_group!r} names credential "
                f"{spec.api_key_env!r} (auth_token_type="
                f"{route['auth_token_type']!r}) and it could not be resolved. "
                f"Refusing to call unauthenticated."
            )
        api_kwargs["api_key"] = secret

    logger.info(
        "Retro judge route: group=%s model=%s provider=%s route=%s "
        "(primary=%s, max_tokens=%s, transport=%s)",
        judge_group, judge_model, spec.provider, route.get("route"),
        route.get("primary_model"), spec.max_tokens, spec.transport,
    )
    # callsite_id is REQUIRED since krepis 0.23 (krepis/llm.py::LLMClient.__init__,
    # validated non-empty). It is the join key between this call's emitted cost
    # row and its LLM_CALLSITE_REGISTRY.yaml entry, so the literal must stay in
    # sync with that row's `id` (alpha-engine-config, id: director-retro-judge).
    # timeout/max_retries bound the judge's share of the single Director
    # invoke. The judge's 2k-token grade is a much smaller call than the plan;
    # 120s is generous against its observed latency.
    #
    # config#6904: the old note here claimed 2×340s (plan) plus 2×120s (judge)
    # kept "the whole invoke inside the Lambda's 900s budget" — that is 920s
    # against a 900s ceiling, before any S3 write, and it was never enforced by
    # anything but this comment. 120 is now a CEILING; the quote below derives
    # the per-attempt budget from the time actually left. The retro is
    # explicitly best-effort and runs AFTER the plan is persisted, so declining
    # it on an exhausted budget costs a grade, never the primary deliverable.
    judge_budget = budget or UNBOUNDED
    client = LLMClient(
        spec,
        callsite_id="director-retro-judge",
        timeout=judge_budget.quote(
            "director-retro-judge", RETRO_JUDGE_CEILING_S, attempts=2
        ),
        max_retries=_CLIENT_MAX_RETRIES,
        **api_kwargs,
    )
    return _KrepisStructuredJudge(
        client, judge_group=judge_group, judge_model=judge_model
    )


def _prior_plan_summary(prior_plan: dict) -> str:
    """Condense the prior plan into the judge's human message — the claims it
    made, so the judge can score them against the realized card."""
    lines = [
        f"PRIOR PLAN (run_date {prior_plan.get('run_date', '?')}):",
        f"  System summary: {prior_plan.get('system_summary', '(none)')}",
        "  Top risks flagged:",
    ]
    for r in prior_plan.get("top_risks", []) or ["(none)"]:
        lines.append(f"    - {r}")
    lines.append("  Action items proposed:")
    for it in prior_plan.get("action_items", []) or []:
        lines.append(
            f"    - [{it.get('priority')}] {it.get('title')} "
            f"(owner={it.get('proposed_owner')}, type={it.get('suggested_change_type')}) "
            f"— {it.get('rationale')} [evidence: {', '.join(it.get('evidence', []) or []) or 'none'}]"
        )
    if not (prior_plan.get("action_items") or []):
        lines.append("    - (no action items)")
    return "\n".join(lines)


def build_messages(prior_plan: dict, current_card: dict) -> list:
    """Assemble (system, human) messages for the retro judge call."""
    human = [
        _prior_plan_summary(prior_plan),
        "",
        "CURRENT REPORT CARD (the realized outcome ~1 week later):",
        summarize_report_card(current_card),
        "",
        "Grade the prior plan now (grounding / calibration / actionability). "
        "Set prior_run_date to the prior plan's run_date.",
    ]
    return [("system", _load_retro_prompt()), ("human", "\n".join(human))]


def grade_prior_plan(
    prior_plan: dict,
    current_card: dict,
    *,
    llm=None,
    budget=None,
) -> RetroGrade:
    """Judge the prior week's plan against the current Report Card → RetroGrade.

    ``llm`` is injectable (a structured-output runnable returning a RetroGrade);
    defaults to the real Sonnet judge.
    """
    llm = llm or _default_llm(budget)
    messages = build_messages(prior_plan, current_card)
    grade = _invoke_with_retry(llm, messages, budget=budget)
    # Stamp the prior run_date from the plan if the model didn't echo it.
    rd = prior_plan.get("run_date")
    if rd and not grade.prior_run_date:
        grade.prior_run_date = rd
    logger.info(
        "Director retro of plan %s: grounding=%d calibration=%d actionability=%d",
        grade.prior_run_date, grade.grounding, grade.calibration, grade.actionability,
    )
    return grade


def _is_plan(obj) -> bool:
    return isinstance(obj, DirectorWeeklyActionPlan)
