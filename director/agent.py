"""
agent.py — the Director agent (Layer C): one structured Opus call over the
Report Card v2 → a DirectorWeeklyActionPlan.

Not LangGraph — a single ``ChatAnthropic`` call with
``with_structured_output(DirectorWeeklyActionPlan)``, wrapped in a small
rate-limit retry. The report card is condensed to a digest (the issues +
trends), last week's plan is supplied as carry-over context, and the model
emits the structured plan directly (tool-use + Pydantic — no freeform parsing).

The LLM is injectable (``llm=``) so the build/validate + tests run without a key
or langchain installed; ``_default_llm()`` lazily constructs the real client.
Model: Opus (locked, director-plan §4.2).
"""

from __future__ import annotations

import logging
import time

from director.report_card_digest import summarize_report_card
from director.schema import DirectorWeeklyActionPlan

logger = logging.getLogger(__name__)

DIRECTOR_MODEL = "claude-opus-4-8"
_MAX_RETRIES = 3
_RETRYABLE = ("overloaded", "rate", "429", "529", "timeout", "connection")


def _load_system_prompt() -> str:
    """The tuned prompt (gitignored director/prompt.py) if present, else the
    committed generic template."""
    try:
        from director.prompt import SYSTEM_PROMPT  # type: ignore
        return SYSTEM_PROMPT
    except Exception:  # ImportError or anything — fall back to the template
        from director.prompt_example import SYSTEM_PROMPT
        return SYSTEM_PROMPT


def _default_llm():
    """Construct the real structured-output Opus client (lazy import)."""
    from langchain_anthropic import ChatAnthropic  # lazy — not needed for tests
    base = ChatAnthropic(model=DIRECTOR_MODEL, temperature=0, max_tokens=8000)
    return base.with_structured_output(DirectorWeeklyActionPlan)


def _carryover_context(carryover: dict | None) -> str:
    if not carryover or not carryover.get("items"):
        return "No prior action plan on record (this is the first cycle or the ledger is empty)."
    lines = ["Last week's open action items (carry-over ledger):"]
    for it in carryover.get("items", []):
        lines.append(
            f"  - [{it.get('id')}] {it.get('title')} "
            f"(status={it.get('status')}, owner={it.get('proposed_owner')}, priority={it.get('priority')})"
        )
    return "\n".join(lines)


def build_messages(report_card: dict, *, carryover: dict | None = None, roadmap_digest: str | None = None) -> list:
    """Assemble (system, human) messages for the Director call."""
    human = [
        summarize_report_card(report_card),
        "",
        _carryover_context(carryover),
    ]
    if roadmap_digest:
        human += ["", "Currently-tracked / in-flight work (ROADMAP digest — do NOT re-propose):", roadmap_digest]
    human += [
        "",
        "Produce the DirectorWeeklyActionPlan now. Ground every action item's "
        "rationale + evidence in the metrics above.",
    ]
    return [("system", _load_system_prompt()), ("human", "\n".join(human))]


def _invoke_with_retry(llm, messages) -> DirectorWeeklyActionPlan:
    last = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return llm.invoke(messages)
        except Exception as e:  # noqa: BLE001 — classify + retry transient, raise the rest
            last = e
            msg = str(e).lower()
            if attempt < _MAX_RETRIES and any(t in msg for t in _RETRYABLE):
                delay = min(2 ** attempt, 30)
                logger.warning("Director LLM transient error (attempt %d): %s — retrying in %ss", attempt, e, delay)
                time.sleep(delay)
                continue
            raise
    raise RuntimeError(f"Director LLM failed after {_MAX_RETRIES} attempts") from last


def build_action_plan(
    report_card: dict,
    *,
    run_date: str | None = None,
    carryover: dict | None = None,
    roadmap_digest: str | None = None,
    llm=None,
) -> DirectorWeeklyActionPlan:
    """Run the Director: report card → DirectorWeeklyActionPlan.

    ``llm`` is injectable (a structured-output runnable returning a
    DirectorWeeklyActionPlan); defaults to the real Opus client. ``run_date``
    overrides the plan's run_date (else taken from the card provenance).
    """
    llm = llm or _default_llm()
    messages = build_messages(report_card, carryover=carryover, roadmap_digest=roadmap_digest)
    plan = _invoke_with_retry(llm, messages)
    # Stamp the run_date from the card if the model didn't echo one.
    rd = run_date or (report_card.get("_provenance", {}) or {}).get("run_date")
    if rd and not plan.run_date:
        plan.run_date = rd
    logger.info(
        "Director plan for %s: %d action items, %d top risks",
        plan.run_date, len(plan.action_items), len(plan.top_risks),
    )
    return plan
