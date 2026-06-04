"""
agent.py — Tile 6: Behavioural / Agent Quality (RC v2).

Grades the LLM agents + their judge-derived calibration. None of the seven
components are sourceable by the evaluator today — the producers (judge
calibration κ artifacts, LangGraph validation/latency traces, cost-per-signal,
the signals' ``stance_source`` field) are not yet persisted in a form the
report card can read. So this tile is, honestly, an **all-N/A transparency
shell**: it surfaces that *agent quality is not yet measured* and each
component's reason names the exact producer to build — turning the
agent-measurement backlog into a first-class, visible part of the report card
(the RC v2 "show all modules, honest N/A" principle) rather than a silent gap.

``stance_source_provenance`` is N/A-MISSING-INPUT (verified 2026-06-04: the
``signals.json`` universe entries carry no ``stance_source`` field); the rest
are N/A-NOT-IMPL.

Spec: ``system-report-card-revamp-260522.md`` Tile 6.
"""

from __future__ import annotations

import logging

import boto3

from grading.metric_record import build_metric
from grading.module_agg import build_tile

logger = logging.getLogger(__name__)

MODULE = "agent"


def build_agent_tile(bucket: str, s3_client=None) -> dict:
    """Build the Behavioural / Agent-Quality tile (transparency shell today)."""
    _ = s3_client or boto3.client("s3")  # reserved for when producers land
    components = []

    components.append(build_metric(
        name="stance_source_provenance", module=MODULE, metric_type="pct", criticality="diagnostic",
        n_floor=1, target=0.95, red_line=0.50, source_path=f"s3://{bucket}/signals/latest.json",
        input_present=False,
        na_detail="stance_source_provenance: signals.json universe entries carry no `stance_source` field (verified 2026-06-04); add it per attractiveness-pillars Phase 5 to grade pick provenance coverage.",
    ))

    not_impl = [
        ("judge_calibration_cohen_kappa", "critical",
         "judge_calibration_cohen_kappa: needs the blind-Step-1-operator vs LLM-judge κ from decision_artifacts/_calibration/*.jsonl (L480) — not yet computed for the report card."),
        ("agent_validation_failure_rate", "critical",
         "agent_validation_failure_rate: needs the % of agent runs with Pydantic schema-validation failures from LangGraph traces — not yet persisted."),
        ("cost_per_signal", "supporting",
         "cost_per_signal: needs the LLM $ cost / finalized-signal count from the cost report — not yet exposed to the evaluator."),
        ("retry_storm_count", "supporting",
         "retry_storm_count: needs the count of agents exceeding max_retries from LLM-provider logs — not yet aggregated."),
        ("agent_latency_p95", "diagnostic",
         "agent_latency_p95: needs per-agent-type p95 latency from LangGraph traces — not yet persisted."),
        ("judge_rubric_distribution", "diagnostic",
         "judge_rubric_distribution: needs the distribution of judge grades across rubric dims (rubric-collapse flag) — not yet aggregated."),
    ]
    for name, crit, detail in not_impl:
        components.append(build_metric(
            name=name, module=MODULE, metric_type="pct", criticality=crit, n_floor=1,
            source_path=f"s3://{bucket}/", implemented=False, na_detail=detail,
        ))

    return build_tile(MODULE, components)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Build the Agent-Quality tile.")
    parser.add_argument("--bucket", default="alpha-engine-research")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(json.dumps(build_agent_tile(args.bucket), indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
