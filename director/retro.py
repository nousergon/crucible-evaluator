"""
retro.py — the Director's Phase-G self-grading retro loop (Layer C).

Each weekly run judges LAST week's plan against THIS week's Report Card — the
realized-outcome feedback loop the in-call ``SelfGrade`` can't provide (it can't
see the future at emission). One structured judge call → a ``RetroGrade``
(grounding / calibration / actionability), reusing the LLM-as-judge rubric
pattern (mirrors research ``evals/judge.py`` ``RubricEvalLLMOutput``).

The LLM is injectable (``llm=``) so build/validate + tests run without a key or
langchain; ``_default_llm()`` lazily constructs the real client. Model: the same
Opus the Director uses (a judge of the Director should be at least as capable).
"""

from __future__ import annotations

import logging

from director.agent import DIRECTOR_MODEL, _invoke_with_retry
from director.report_card_digest import summarize_report_card
from director.schema import DirectorWeeklyActionPlan, RetroGrade

logger = logging.getLogger(__name__)


def _load_retro_prompt() -> str:
    """The tuned retro rubric (gitignored director/retro_prompt.py) if present,
    else the committed generic template."""
    try:
        from director.retro_prompt import RETRO_PROMPT  # type: ignore
        return RETRO_PROMPT
    except Exception:  # ImportError or anything — fall back to the template
        from director.retro_prompt_example import RETRO_PROMPT
        return RETRO_PROMPT


def _default_llm():
    """Construct the real structured-output Opus judge client (lazy import).

    Same SSM-key + langchain path as ``agent._default_llm`` — kept here so the
    retro can be exercised independently — but bound to ``RetroGrade``.
    """
    from krepis.secrets import get_secret
    from langchain_anthropic import ChatAnthropic  # lazy — not needed for tests

    api_key = get_secret("ANTHROPIC_API_KEY")
    # No `temperature` — claude-opus-4-8 removed the sampling params; passing one
    # 400s ("`temperature` is deprecated for this model"). Mirrors agent._default_llm.
    base = ChatAnthropic(
        model=DIRECTOR_MODEL, max_tokens=2000, anthropic_api_key=api_key,
    )
    return base.with_structured_output(RetroGrade)


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
    defaults to the real Opus judge.
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
