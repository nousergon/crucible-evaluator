"""
schema.py — the Director's structured output contract (Layer C).

``DirectorWeeklyActionPlan`` is what the single Opus director call emits: a
3-5 sentence whole-system read, the top risks, a list of grounded
``ActionItem``s, an explicit disposition of last week's items, and an optional
self-grade. The agent reasons over the Report Card v2 substrate; every action
item must cite the input sources it leaned on as typed ``citations`` (ids
rendered in the prompt as ``{cite=...}`` tags and checked by code after the
call — ``director/citations.py``, alpha-engine-config-I12312), so the plan is
grounded rather than plausible-but-ungrounded. ``evidence`` remains as optional
free text with no identity role; plans written before the citation contract
carry only ``evidence`` and still parse.

The plan PROPOSES; the operator DISPOSES. Nothing here writes live trading
config — the downstream writes are the archived artifact + (Phase H,
``director/roadmap_pr.py``) an approval-gated ROADMAP PR the operator merges.

This is defined here in the evaluator for the Phase-E build/validate; it lifts
to ``nousergon_lib.agent_schemas`` once a 2nd consumer (the dashboard / the
ROADMAP-PR renderer) appears (the lift-on-2nd-consumer rule). Authoritative
spec: ``director-implementation-plan-260604.md`` §4.3.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

OwnerLiteral = Literal[
    "research", "predictor", "executor", "backtester", "data", "substrate", "operator",
]
PriorityLiteral = Literal["P0", "P1", "P2", "P3"]
HorizonLiteral = Literal["this_week", "carryover", "watch"]
ChangeTypeLiteral = Literal[
    "param_tune", "structural", "investigation", "data_fix", "no_action_monitor",
]
ItemStatusLiteral = Literal["proposed", "carried_over", "resolved", "dropped"]

# alpha-engine-config-I12312 / I12101: the kinds of input-context source a
# citation may name. Each is rendered into the prompt with an explicit
# ``{cite=<kind>:<key>}`` tag by ``director.citations`` — the tag IS the id.
CitationKindLiteral = Literal[
    "metric", "tile", "card", "carryover", "issue", "artifact", "plan_item",
]
#: Deterministic post-call verdict on one citation (``director.citations``).
#: Never produced by the model: the field carrying it is excluded from the
#: structured-output schema and overwritten after every call.
CitationCheckLiteral = Literal["valid", "unknown_id", "kind_mismatch", "quote_not_found"]
RetroDimensionLiteral = Literal["grounding", "calibration", "actionability"]


class Citation(BaseModel):
    """One typed pointer from a claim to a piece of the input the model saw.

    ``id`` is copied VERBATIM from a ``{cite=...}`` tag in the prompt
    (``metric:momentum_l1_ic``, ``tile:predictor``, ``carryover:<slug>``,
    ``issue:alpha-engine-config-I123``, ``artifact:<s3 key>``, ...). It is
    checked by code against the exact set of ids the prompt offered; an id the
    prompt did not offer is recorded as invalid, never repaired or dropped.
    """

    id: str = Field(description="Copied verbatim from a {cite=...} tag in the input, e.g. 'metric:momentum_l1_ic'.")
    # A plain ``str`` that ADVERTISES the enum, not a ``Literal``: an
    # off-vocabulary kind from a non-native structured-output rung must land
    # as a recorded ``kind_mismatch`` on this one citation, not as a
    # ValidationError that discards the whole plan.
    kind: str = Field(
        description="The id's prefix: metric / tile / card / carryover / issue / artifact / plan_item.",
        json_schema_extra={"enum": list(CitationKindLiteral.__args__)},
    )
    quote: str | None = Field(
        default=None,
        description="Optional short verbatim fragment of the cited line (e.g. 'value 0.012'); checked against the input.",
    )
    # Stamped by director.citations.validate_*; absent from the LLM schema.
    check: SkipJsonSchema[CitationCheckLiteral | None] = None
    check_reason: SkipJsonSchema[str | None] = None


class ActionItem(BaseModel):
    """One proposed action, grounded in named report-card evidence."""

    id: str = Field(description="Stable slug, carries across weeks for carry-over matching.")
    title: str
    rationale: str = Field(description="Grounded in named metrics — cite the grade/CI/trend.")
    # alpha-engine-config-I12312: the item's typed, code-checked grounding.
    # ``evidence`` below stays as human-readable prose only — it is no longer
    # the identity of what the item cites (it was regex-tokenised after the
    # fact, and 26 of 28 ledger rows cited prose — I8178 / I12101).
    citations: list[Citation] = Field(
        default_factory=list,
        description=(
            "Typed citations: every input source this item rests on, each id copied verbatim "
            "from a {cite=...} tag in the input. At least one is required for a grounded item."
        ),
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Optional free-text evidence notes for humans. NOT used for citation identity — use `citations`.",
    )
    proposed_owner: OwnerLiteral
    priority: PriorityLiteral
    horizon: HorizonLiteral
    suggested_change_type: ChangeTypeLiteral
    confidence: int = Field(ge=0, le=100, description="Self-assessed 0-100.")
    status: ItemStatusLiteral = "proposed"
    #: ``"grounded"`` (>= 1 valid citation) / ``"ungrounded"``; stamped by
    #: ``director.citations.validate_plan_citations``, absent from the LLM schema.
    citation_grounding: SkipJsonSchema[Literal["grounded", "ungrounded"] | None] = None
    # alpha-engine-config-I11990: work identity, declared by the proposer and
    # judged by code. Additive with an empty default, so a plan or ledger row
    # written before this field still parses.
    targets: list[str] = Field(
        default_factory=list,
        description=(
            "Issue refs (alpha-engine-config-I<N>) of any CLOSED or RULED issue this item "
            "proposes to land, redo, reopen or re-confirm. Empty for genuinely new work. "
            "This is the item's work identity: an item naming a closed or ruled issue is "
            "filed only if that issue's declared reopen condition is met, and only an item "
            "naming a prior closed proposal is filed as its re-confirm."
        ),
    )


class SelfGrade(BaseModel):
    """The Director's self-assessment of its own plan (grounding / actionability).

    The realized-outcome retro (calibration vs what actually happened) is the
    Phase-G loop; this is the cheap in-call self-check.
    """

    grounding: int = Field(ge=0, le=100, description="Did every rationale cite real metrics?")
    actionability: int = Field(ge=0, le=100, description="Are the items concrete + owner-assignable?")
    notes: str = ""


class RetroCitation(Citation):
    """A retro-judge citation, tied to the rubric dimension it supports."""

    dimension: RetroDimensionLiteral | None = Field(
        default=None, description="Which score this citation supports: grounding / calibration / actionability.",
    )


class RetroGrade(BaseModel):
    """Phase-G realized-outcome retro: last week's plan judged against what
    actually happened (the current Report Card), a week later.

    Distinct from ``SelfGrade`` (the cheap same-call self-check, which cannot see
    outcomes): the retro adds **calibration** — did the risks the prior plan
    flagged actually materialize? — by grading the prior plan against the fresh
    card. Reuses the LLM-as-judge rubric pattern (mirrors research
    ``evals/judge.py`` ``RubricEvalLLMOutput``).
    """

    model_config = ConfigDict(extra="allow")

    prior_run_date: str = Field(description="run_date of the plan being graded.")
    grounding: int = Field(ge=0, le=100, description="Did the prior plan's rationale cite real metrics?")
    calibration: int = Field(ge=0, le=100, description="Did the risks the prior plan flagged actually materialize in this card?")
    actionability: int = Field(ge=0, le=100, description="Were the prior plan's items concrete + owner-assignable?")
    notes: str = Field(default="", description="One-paragraph justification, citing what moved vs what the plan expected.")
    # alpha-engine-config-I12312: what the grade relied on, typed and
    # code-checked — the prior plan's items (plan_item:<id>) and the current
    # card's metrics (metric:<name>), each tagged with the dimension it supports.
    citations: list[RetroCitation] = Field(
        default_factory=list,
        description=(
            "The prior-plan items and current-card sources this grade relied on, each id copied "
            "verbatim from a {cite=...} tag in the input, with the dimension it supports."
        ),
    )


class DirectorWeeklyActionPlan(BaseModel):
    """The weekly advisory plan — the single Opus call's structured output."""

    model_config = ConfigDict(extra="allow")  # forward-compat

    run_date: str
    system_summary: str = Field(description="3-5 sentence whole-system read.")
    top_risks: list[str]
    action_items: list[ActionItem]
    carryover_review: list[str] = Field(
        default_factory=list,
        description="Explicit disposition of last week's items (done / rolled / dropped).",
    )
    self_grade: SelfGrade | None = None
