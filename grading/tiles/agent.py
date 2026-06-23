"""
agent.py — Tile 6: Behavioural / Agent Quality (RC v2).

Grades the LLM agents that drive research. Most components read a single
producer artifact, ``backtest/{run_date}/agent_quality.json``, emitted by the
research-side agent-quality aggregator (off-hot-path, backfillable — it reads
the existing ``decision_artifacts/_cost`` + ``_eval`` streams, agent telemetry
and ``signals.json``). The contract this tile reads is the authoritative one
the producer must satisfy (consumer-contract-first, config Batch A #1149):

    agent_quality.json = {
      "status": "ok",
      "agent_validation_failure_rate": {"value": <0-1>, "n": <invokes>},
      "cost_per_signal":               {"value": <usd>, "n": <signals>},
      "retry_storm_count":             {"value": <count>, "n": <agents>},
      "agent_latency_p95":             {"value": <ms>,  "n": <agents>},
      "judge_rubric_distribution":     {"value": <0-1 modal-concentration>, "n": <evals>},
      ...                              # judge_rubric_pass_rate / pillar_emit_coverage
      ...                              # / signal_volume_adequacy live on the Research tile
    }

Until the producer lands, every block is absent → each component grades a
precise **N/A-MISSING-INPUT** naming the producer (not a generic N/A-NOT-IMPL),
so the report card honestly shows "agent quality not measured yet" while the
contract is live and self-activates on the first ``agent_quality.json``.

``judge_calibration_cohen_kappa`` stays N/A-NOT-IMPL — it needs a human
blind-Step-1-operator labeling pass (config Batch E #1153), not just a
producer. ``stance_source_provenance`` is N/A-MISSING-INPUT until the
``signals.json`` ``stance_source`` field lands (crucible-research#297).

Spec: ``system-report-card-revamp-260522.md`` Tile 6.
"""

from __future__ import annotations

import json
import logging

import boto3
from botocore.exceptions import ClientError

from grading.artifacts import get_json_windowed
from grading.metric_record import build_metric
from grading.module_agg import build_tile

logger = logging.getLogger(__name__)

MODULE = "agent"


def _get_json(s3, bucket: str, key: str) -> dict | None:
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        logger.error("S3 read failed for s3://%s/%s: %s", bucket, key, e)
        raise
    return json.loads(resp["Body"].read())


def _block(aq: dict | None, key: str) -> dict | None:
    """Return the per-metric sub-block from agent_quality.json, or None.

    A block counts as present only when the artifact status is ok, the block
    exists, and it carries a non-null ``value`` — anything else degrades to a
    precise N/A-MISSING-INPUT (never a silent omission or a false GREEN).
    """
    if not aq or aq.get("status") != "ok":
        return None
    blk = aq.get(key)
    if isinstance(blk, dict) and blk.get("value") is not None:
        return blk
    return None


def build_agent_tile(bucket: str, run_date: str, s3_client=None) -> dict:
    """Build the Behavioural / Agent-Quality tile.

    ``run_date`` keys the ``backtest/{run_date}/agent_quality.json`` producer
    artifact (NYSE trading day, resolved by the handler).
    """
    s3 = s3_client or boto3.client("s3")
    # Windowed resolution (config#1190): freshest within the trailing window.
    aq, _, _, _aq_key = get_json_windowed(s3, bucket, "backtest/{date}/agent_quality.json", run_date)
    aq_src = f"s3://{bucket}/{_aq_key}" if _aq_key else f"s3://{bucket}/backtest/{run_date}/agent_quality.json"
    components = []

    # 1. agent_validation_failure_rate (critical) — % of agent .invoke()s that
    #    failed Pydantic schema validation (lower better). A rising rate means
    #    the agents are emitting malformed structured output the graph has to
    #    repair or drop — a direct agent-quality regression the Director acts on.
    blk = _block(aq, "agent_validation_failure_rate")
    if blk is not None:
        components.append(build_metric(
            name="agent_validation_failure_rate", module=MODULE, metric_type="pct",
            criticality="critical", estimator="wilson_failure_rate",
            measurement_horizon="per_run",
            value=blk["value"], n_samples=blk.get("n"), n_floor=50,
            target=0.02, red_line=0.10, higher_is_better=False, source_path=aq_src,
            reason=(f"agent_validation_failure_rate = {blk['value']:.1%} "
                    f"(N={blk.get('n')} invokes) vs target 2% / red-line 10%."),
        ))
    else:
        components.append(build_metric(
            name="agent_validation_failure_rate", module=MODULE, metric_type="pct",
            criticality="critical", estimator="wilson_failure_rate",
            measurement_horizon="per_run",
            n_floor=50, target=0.02, red_line=0.10, higher_is_better=False,
            source_path=aq_src, input_present=False,
            na_detail="agent_validation_failure_rate: agent_quality.json absent or no value this cycle (research agent-quality producer, config#1149).",
        ))

    # 2. cost_per_signal (supporting) — total run LLM $ / finalized signal (lower
    #    better). Efficiency of the research spend per usable output.
    blk = _block(aq, "cost_per_signal")
    if blk is not None:
        components.append(build_metric(
            name="cost_per_signal", module=MODULE, metric_type="ratio", criticality="supporting",
            value=blk["value"], n_samples=blk.get("n"), n_floor=5,
            target=1.0, red_line=5.0, higher_is_better=False, source_path=aq_src,
            reason=f"cost_per_signal = ${blk['value']:.2f} over {blk.get('n')} finalized signals vs target $1 / red-line $5.",
        ))
    else:
        components.append(build_metric(
            name="cost_per_signal", module=MODULE, metric_type="ratio", criticality="supporting",
            n_floor=5, target=1.0, red_line=5.0, higher_is_better=False,
            source_path=aq_src, input_present=False,
            na_detail="cost_per_signal: agent_quality.json absent or no value this cycle (research agent-quality producer, config#1149).",
        ))

    # 3. retry_storm_count (supporting) — # of agents that reached their per-type
    #    max_retries ceiling (lower better). A storm signals provider instability
    #    or a malformed-output loop burning budget.
    blk = _block(aq, "retry_storm_count")
    if blk is not None:
        components.append(build_metric(
            name="retry_storm_count", module=MODULE, metric_type="count", criticality="supporting",
            value=blk["value"], n_samples=blk.get("n"), n_floor=1,
            target=0.0, red_line=5.0, higher_is_better=False, source_path=aq_src,
            reason=f"retry_storm_count = {blk['value']:.0f} agents at retry ceiling (of {blk.get('n')}) vs target 0 / red-line 5.",
        ))
    else:
        components.append(build_metric(
            name="retry_storm_count", module=MODULE, metric_type="count", criticality="supporting",
            n_floor=1, target=0.0, red_line=5.0, higher_is_better=False,
            source_path=aq_src, input_present=False,
            na_detail="retry_storm_count: agent_quality.json absent or no value this cycle (research agent-quality producer, config#1149).",
        ))

    # 4. agent_latency_p95 (diagnostic) — per-agent-type p95 wall-clock (ms,
    #    lower better). A latency creep flags a slowing agent before it trips the
    #    Lambda ceiling.
    blk = _block(aq, "agent_latency_p95")
    if blk is not None:
        components.append(build_metric(
            name="agent_latency_p95", module=MODULE, metric_type="duration", criticality="diagnostic",
            value=blk["value"], n_samples=blk.get("n"), n_floor=1,
            target=15000.0, red_line=60000.0, higher_is_better=False, source_path=aq_src,
            reason=f"agent_latency_p95 = {blk['value']:.0f} ms (N={blk.get('n')}) vs target 15s / red-line 60s.",
        ))
    else:
        components.append(build_metric(
            name="agent_latency_p95", module=MODULE, metric_type="duration", criticality="diagnostic",
            n_floor=1, target=15000.0, red_line=60000.0, higher_is_better=False,
            source_path=aq_src, input_present=False,
            na_detail="agent_latency_p95: agent_quality.json absent or no value this cycle (research agent-quality producer, config#1149).",
        ))

    # 5. judge_rubric_distribution (diagnostic) — modal-score concentration of
    #    the LLM-judge grades across rubric dimensions (lower better). A high
    #    concentration is rubric collapse: the judge gives the same grade to
    #    everything, so its signal is degenerate.
    blk = _block(aq, "judge_rubric_distribution")
    if blk is not None:
        components.append(build_metric(
            name="judge_rubric_distribution", module=MODULE, metric_type="ratio", criticality="diagnostic",
            value=blk["value"], n_samples=blk.get("n"), n_floor=10,
            target=0.40, red_line=0.70, higher_is_better=False, source_path=aq_src,
            reason=f"judge_rubric_distribution modal-concentration = {blk['value']:.0%} (N={blk.get('n')} evals) vs target 40% / red-line 70% (collapse).",
        ))
    else:
        components.append(build_metric(
            name="judge_rubric_distribution", module=MODULE, metric_type="ratio", criticality="diagnostic",
            n_floor=10, target=0.40, red_line=0.70, higher_is_better=False,
            source_path=aq_src, input_present=False,
            na_detail="judge_rubric_distribution: agent_quality.json absent or no value this cycle (research agent-quality producer, config#1149).",
        ))

    # 6. stance_source_provenance (diagnostic) — pick-provenance coverage; lands
    #    when signals.json carries `stance_source` (crucible-research#297).
    components.append(build_metric(
        name="stance_source_provenance", module=MODULE, metric_type="pct", criticality="diagnostic",
        n_floor=1, target=0.95, red_line=0.50, source_path=f"s3://{bucket}/signals/latest.json",
        input_present=False,
        na_detail="stance_source_provenance: signals.json universe entries' `stance_source` field self-activates on the first run carrying it (crucible-research#297).",
    ))

    # 7. judge_calibration_cohen_kappa (critical) — needs a HUMAN blind-Step-1
    #    labeling pass vs the LLM judge; not a producer (config Batch E #1153).
    components.append(build_metric(
        name="judge_calibration_cohen_kappa", module=MODULE, metric_type="ratio", criticality="critical",
        n_floor=1, source_path=f"s3://{bucket}/decision_artifacts/_calibration/", implemented=False,
        na_detail="judge_calibration_cohen_kappa: needs the blind-Step-1-operator vs LLM-judge κ from decision_artifacts/_calibration/*.jsonl — human labeling, not a producer (config#1153).",
    ))

    return build_tile(MODULE, components)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Build the Agent-Quality tile.")
    parser.add_argument("--bucket", default="alpha-engine-research")
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(json.dumps(build_agent_tile(args.bucket, args.date), indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
