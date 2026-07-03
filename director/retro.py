"""
retro.py — the Director's Phase-G self-grading retro loop (Layer C).

Each weekly run judges LAST week's plan against THIS week's Report Card — the
realized-outcome feedback loop the in-call ``SelfGrade`` can't provide (it can't
see the future at emission). One structured judge call → a ``RetroGrade``
(grounding / calibration / actionability), reusing the LLM-as-judge rubric
pattern (mirrors research ``evals/judge.py`` ``RubricEvalLLMOutput``).

The LLM is injectable (``llm=``) so build/validate + tests run without a key or
krepis' provider SDKs; ``_default_llm()`` lazily constructs the real client.

**Judge tier: Sonnet, deliberately NOT the Director's Opus.** Grading a plan
with the same model that generated it is self-grading bias (config#1673,
judge != generator) — see ``agent.py``'s ``DIRECTOR_MODEL`` (Opus, locked for
plan generation, untouched here). The judge call is routed through
``krepis.llm``'s provider-agnostic adapter (krepis>=0.9.0) rather than
langchain's ``ChatAnthropic`` — a separate call surface from
``agent._default_llm``. The model is config, not code: ``RETRO_JUDGE_MODEL``
env var, default ``"claude-sonnet-4-6"``. That default is a floating alias
with no dated snapshot — the API resolves it to a live snapshot per call, and
both the alias (``judge_model``) and the API-resolved model (``resolved_model``)
are stamped onto the persisted ``RetroGrade`` (``extra="allow"``) so the
dashboard/audit trail can see exactly what ran.
"""

from __future__ import annotations

import logging
import os

from director.agent import _invoke_with_retry
from director.report_card_digest import summarize_report_card
from director.schema import DirectorWeeklyActionPlan, RetroGrade

logger = logging.getLogger(__name__)

# Sonnet judge tier (config#1673) — env-overridable; the default is a
# floating alias (no dated snapshot). Intentionally distinct from
# `agent.DIRECTOR_MODEL` (Opus, plan generation) — do not import/reuse that
# constant here; the whole point is judge != generator.
RETRO_JUDGE_MODEL_DEFAULT = "claude-sonnet-4-6"

_RETRO_JUDGE_SCHEMA_NAME = "RetroGrade"


def _judge_model() -> str:
    """The configured retro-judge model alias: ``RETRO_JUDGE_MODEL`` env
    override if set, else :data:`RETRO_JUDGE_MODEL_DEFAULT`. Read at call
    time (not frozen at import) so an operator/test override takes effect
    without a process restart."""
    return os.environ.get("RETRO_JUDGE_MODEL", RETRO_JUDGE_MODEL_DEFAULT)


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

    def __init__(self, client, *, judge_model: str):
        self._client = client
        self._judge_model = judge_model

    def invoke(self, messages: list) -> RetroGrade:
        system, user_content = _split_messages(messages)
        result = self._client.structured(
            system=system,
            user_content=user_content,
            schema=RetroGrade,
            schema_name=_RETRO_JUDGE_SCHEMA_NAME,
        )
        grade: RetroGrade = result.parsed
        # judge_model = the logical alias configured for this call (no dated
        # snapshot); resolved_model = what the API actually resolved that
        # alias to. Both land as extra fields — RetroGrade has extra="allow" —
        # so they persist through model_dump()/model_dump_json() unchanged.
        grade.judge_model = self._judge_model
        grade.resolved_model = result.model
        return grade


def _default_llm() -> _KrepisStructuredJudge:
    """Construct the real structured-output Sonnet judge client (lazy import).

    Same SSM ``ANTHROPIC_API_KEY`` secret path as ``agent._default_llm`` —
    kept here so the retro can be exercised independently — but routed
    through ``krepis.llm.LLMClient`` (krepis>=0.9.0) bound to ``RetroGrade``,
    not langchain's ``ChatAnthropic``. Both imports are lazy so tests + the
    grading path never pull krepis' provider SDKs or hit SSM.
    """
    from krepis.llm import LLMClient
    from krepis.llm_config import ModelSpec
    from krepis.secrets import get_secret

    judge_model = _judge_model()
    api_key = get_secret("ANTHROPIC_API_KEY")
    # No `temperature` — matches agent._default_llm's note: current-generation
    # Claude models reject sampling params. krepis.llm's anthropic transport
    # never sets one, so there's nothing to strip here.
    spec = ModelSpec(provider="anthropic", model=judge_model, max_tokens=2000)
    client = LLMClient(spec, api_key=api_key)
    return _KrepisStructuredJudge(client, judge_model=judge_model)


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
) -> RetroGrade:
    """Judge the prior week's plan against the current Report Card → RetroGrade.

    ``llm`` is injectable (a structured-output runnable returning a RetroGrade);
    defaults to the real Sonnet judge.
    """
    llm = llm or _default_llm()
    messages = build_messages(prior_plan, current_card)
    grade = _invoke_with_retry(llm, messages)
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
