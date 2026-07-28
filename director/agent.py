"""
agent.py — the Director agent (Layer C): one structured LLM call over the
Report Card v2 → a DirectorWeeklyActionPlan.

Not LangGraph — a single ``krepis.llm.LLMClient`` structured call routed
through OpenRouter to the ultra-group primary (GLM 5.2), wrapped in a small
rate-limit retry. The report card is condensed to a digest (the issues +
trends), last week's plan is supplied as carry-over context, and the model
emits the structured plan directly (krepis structured-output + Pydantic — no
freeform parsing).

The LLM is injectable (``llm=``) so the build/validate + tests run without a
key or krepis installed; ``_default_llm()`` lazily constructs the real client.
Model: ultra group (LLM_MODEL_REGISTRY.yaml), primary = GLM 5.2 via OpenRouter.
Migrated from claude-opus-4-8 direct Anthropic → OpenRouter 2026-07-24.
"""

from __future__ import annotations

import logging
import time

from director.report_card_digest import summarize_report_card
from director.schema import DirectorWeeklyActionPlan

logger = logging.getLogger(__name__)

DIRECTOR_MODEL = "glm-5.2"
_DIRECTOR_SCHEMA_NAME = "DirectorWeeklyActionPlan"
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

    def __init__(self, client, *, director_model: str):
        self._client = client
        self._director_model = director_model

    def invoke(self, messages: list) -> DirectorWeeklyActionPlan:
        system, user_content = _split_messages(messages)
        result = self._client.structured(
            system=system,
            user_content=user_content,
            schema=DirectorWeeklyActionPlan,
            schema_name=_DIRECTOR_SCHEMA_NAME,
            max_tokens=8000,
        )
        plan: DirectorWeeklyActionPlan = result.parsed
        plan.director_model = self._director_model
        plan.resolved_model = result.model
        return plan


def _default_llm() -> _KrepisStructuredDirector:
    """Construct the real structured-output Director client (lazy import).

    The OpenRouter key is fetched from SSM (``/alpha-engine/OPENROUTER_API_KEY``)
    via ``krepis.secrets.get_secret``, routed through ``krepis.llm.LLMClient``
    bound to ``DirectorWeeklyActionPlan``. Uses the ultra group primary (GLM 5.2)
    from LLM_MODEL_REGISTRY.yaml. Both imports are lazy so tests + the grading
    path never pull krepis' provider SDKs or hit SSM.

    Migrated from ChatAnthropic(claude-opus-4-8) → krepis.llm(glm-5.2 via
    OpenRouter) 2026-07-24.
    """
    from krepis.llm import LLMClient
    from krepis.llm_config import ModelSpec
    from krepis.secrets import get_secret

    api_key = get_secret("OPENROUTER_API_KEY")
    spec = ModelSpec(provider="openrouter", model=DIRECTOR_MODEL, max_tokens=8000)
    # callsite_id is REQUIRED since krepis 0.23 (krepis/llm.py::LLMClient.__init__,
    # validated non-empty). It is the join key between this call's emitted cost
    # row and its LLM_CALLSITE_REGISTRY.yaml entry, so the literal must stay in
    # sync with that row's `id` (alpha-engine-config, id: director-plan).
    client = LLMClient(spec, api_key=api_key, callsite_id="director-plan")
    return _KrepisStructuredDirector(client, director_model=DIRECTOR_MODEL)


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
    resolved_digest: str | None = None,
    llm=None,
) -> DirectorWeeklyActionPlan:
    """Run the Director: report card → DirectorWeeklyActionPlan.

    ``llm`` is injectable (a structured-output runnable returning a
    DirectorWeeklyActionPlan); defaults to the real krepis.llm client using
    the ultra group primary (GLM 5.2 via OpenRouter). ``run_date`` overrides
    the plan's run_date (else taken from the card provenance).
    """
    llm = llm or _default_llm()
    messages = build_messages(report_card, carryover=carryover, roadmap_digest=roadmap_digest,
                              resolved_digest=resolved_digest)
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
