"""
schema.py — the Director's structured output contract (Layer C).

``DirectorWeeklyActionPlan`` is what the single Opus director call emits: a
3-5 sentence whole-system read, the top risks, a list of grounded
``ActionItem``s, an explicit disposition of last week's items, and an optional
self-grade. The agent reasons over the Report Card v2 substrate; every action
item must cite the MetricRecord names / artifacts it leaned on (``evidence``),
so the plan is grounded rather than plausible-but-ungrounded.

The plan PROPOSES; the operator DISPOSES. Nothing here writes live trading
config — the only downstream write is the archived artifact + (later, Phase H,
flag-gated) an approval-gated ROADMAP PR the operator merges.

This is defined here in the evaluator for the Phase-E build/validate; it lifts
to ``alpha_engine_lib.agent_schemas`` once a 2nd consumer (the dashboard / the
ROADMAP-PR renderer) appears (the lift-on-2nd-consumer rule). Authoritative
spec: ``director-implementation-plan-260604.md`` §4.3.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

OwnerLiteral = Literal[
    "research", "predictor", "executor", "backtester", "data", "substrate", "operator",
]
PriorityLiteral = Literal["P0", "P1", "P2", "P3"]
HorizonLiteral = Literal["this_week", "carryover", "watch"]
ChangeTypeLiteral = Literal[
    "param_tune", "structural", "investigation", "data_fix", "no_action_monitor",
]
ItemStatusLiteral = Literal["proposed", "carried_over", "resolved", "dropped"]


class ActionItem(BaseModel):
    """One proposed action, grounded in named report-card evidence."""

    id: str = Field(description="Stable slug, carries across weeks for carry-over matching.")
    title: str
    rationale: str = Field(description="Grounded in named metrics — cite the grade/CI/trend.")
    evidence: list[str] = Field(
        default_factory=list,
        description="MetricRecord names / tile names / artifact paths the item leaned on.",
    )
    proposed_owner: OwnerLiteral
    priority: PriorityLiteral
    horizon: HorizonLiteral
    suggested_change_type: ChangeTypeLiteral
    confidence: int = Field(ge=0, le=100, description="Self-assessed 0-100.")
    status: ItemStatusLiteral = "proposed"


class SelfGrade(BaseModel):
    """The Director's self-assessment of its own plan (grounding / actionability).

    The realized-outcome retro (calibration vs what actually happened) is the
    Phase-G loop; this is the cheap in-call self-check.
    """

    grounding: int = Field(ge=0, le=100, description="Did every rationale cite real metrics?")
    actionability: int = Field(ge=0, le=100, description="Are the items concrete + owner-assignable?")
    notes: str = ""


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
