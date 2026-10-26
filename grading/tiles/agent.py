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
      "agent_latency_p95":             {"value": <ms>,  "n": <agents>,
                                        "n_samples": <samples>,       # crucible-research#871
                                        "by_class": {"short_form": {...}, "long_form": {...}}},
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
producer. ``stance_source_provenance`` reads the dated
``signals/{run_date}/signals.json`` (windowed) and grades the share of
``universe`` entries carrying a ``stance_source`` (crucible-research#297). It
is N/A-MISSING-INPUT only when no signals artifact resolves, it is stale, or
no entry carries the field at all — until alpha-engine-config-I12231 it was
hard-coded N/A even though the producer had been emitting the field.

The four ``groom_*`` components (config#2151) grade the backlog-groom
pipeline's LLM-agent runs from ``groom/{date}/*.json`` — computed in
``grading/tiles/groom.py`` and appended here (the groom pipeline is an
agent-quality surface, not a trading-behavior one).

Spec: ``system-report-card-revamp-260522.md`` Tile 6.
"""

from __future__ import annotations

import json
import logging
from collections import Counter

import boto3
from botocore.exceptions import ClientError

from grading.artifacts import (
    AGENT_ARTIFACT_MAX_AGE_DAYS,
    agent_quality_na_reason,
    artifact_is_stale,
    get_json_windowed,
)
from grading.metric_record import build_metric
from grading.module_agg import build_tile
from grading.tiles.groom import build_groom_components
from grading.units import COUNT_AGENTS, FRACTION, MILLISECONDS, USD

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


#: Windowed key of the dated signals artifact the provenance component reads.
#: Dated, not ``signals/latest.json``: the card grades the run it is for.
SIGNALS_KEY_TEMPLATE = "signals/{date}/signals.json"


def _stance_source_provenance(s3, bucket: str, run_date: str):
    """Grade ``stance_source_provenance`` from the dated ``signals.json``.

    value = entries with a non-empty string ``stance_source`` / all
    ``universe`` entries. An artifact in which NO entry carries the field is a
    producer that predates crucible-research#297 — that is MISSING-INPUT, not
    a measured 0%. The reason names the source-value distribution so a card
    reader can see when provenance is present but single-valued.
    """
    name = "stance_source_provenance"
    sig, _sig_date, sig_age, sig_key = get_json_windowed(s3, bucket, SIGNALS_KEY_TEMPLATE, run_date)
    src = f"s3://{bucket}/{sig_key}" if sig_key else f"s3://{bucket}/{SIGNALS_KEY_TEMPLATE.format(date=run_date)}"

    def _na(detail: str):
        return build_metric(
            name=name, module=MODULE, metric_type="pct", criticality="diagnostic",
            n_floor=1, source_path=src, input_present=False, na_detail=detail,
        )

    if sig is None:
        return _na(
            f"{name}: no signals.json resolved for {run_date} in the trailing "
            f"window — the field is emitted on universe entries by the signals "
            f"producer (crucible-research#297)."
        )
    if artifact_is_stale(sig_age, AGENT_ARTIFACT_MAX_AGE_DAYS):
        return _na(
            f"{name}: signals.json is {sig_age}d old "
            f"(> {AGENT_ARTIFACT_MAX_AGE_DAYS}d max) — stale input, cannot grade confidently."
        )
    universe = sig.get("universe") if isinstance(sig, dict) else None
    entries = [e for e in universe if isinstance(e, dict)] if isinstance(universe, list) else []
    if not entries:
        return _na(f"{name}: signals.json carries no universe entries this cycle.")
    if not any("stance_source" in e for e in entries):
        return _na(
            f"{name}: none of the {len(entries)} signals.json universe entries "
            f"carries a `stance_source` field — the producer predates "
            f"crucible-research#297."
        )
    sources = [e.get("stance_source") for e in entries]
    tagged = [v for v in sources if isinstance(v, str) and v.strip()]
    value = len(tagged) / len(entries)
    dist = Counter(tagged).most_common(4)
    dist_txt = ", ".join(f"{k}={n}" for k, n in dist) or "none"
    return build_metric(
        name=name, module=MODULE, metric_type="pct", criticality="diagnostic",
        value=value, unit=FRACTION, n_samples=len(entries), n_floor=1,
        source_path=src,
        reason=(
            f"{name} = {value:.1%} ({len(tagged)}/{len(entries)} universe entries "
            f"carry a stance_source; distribution: {dist_txt}) vs target 95% / red-line 50%."
        ),
    )


def _count(x) -> int | None:
    """A non-negative int sample count, or None (bool is not a count)."""
    if isinstance(x, bool) or not isinstance(x, int) or x < 0:
        return None
    return x


def _latency_class(by_class, cls: str) -> dict | None:
    """One ``by_class`` entry when it is well-formed (a numeric ``p95_ms``)."""
    if not isinstance(by_class, dict):
        return None
    entry = by_class.get(cls)
    if not isinstance(entry, dict):
        return None
    p95 = entry.get("p95_ms")
    if isinstance(p95, bool) or not isinstance(p95, (int, float)):
        return None
    return entry


def _latency_component(blk: dict, aq_src: str):
    """Grade ``agent_latency_p95`` from a present ``agent_quality.json`` block.

    alpha-engine-config-I12625. ``n`` in the block is the number of AGENTS with
    a duration sample, not a sample size: the 10-23 card printed ``N=6`` over
    107 pooled invocations and the Director read it as six samples. And one
    wall-clock bar does not fit every agent: the long-form writers
    (``analyst_thesis``, ``analyst_pillar``) emit ~6-11K output tokens at a flat
    ~190 tok/s, so their p95 measures output length, not a slow call.

    crucible-research#871 adds ``n_samples`` and a ``by_class`` split. When the
    split is present:

    * ``by_class.short_form`` is graded against the registry's 15s / 60s bars,
      with ``N`` = that class's pooled duration samples;
    * ``by_class.long_form`` is REPORTED in the reason and NOT graded. Its bar
      is the card owner's decision, still open on the issue; grading it on
      15s/60s would be the mis-specification the issue found, and grading it on
      any other number would be a bar nobody has ruled on;
    * no short-form agent sampled ⇒ N/A-MISSING-INPUT naming the long-form p95,
      never a GREEN by omission.

    An artifact written before #871 carries neither field: the headline (worst
    agent overall) is graded exactly as before, with ``N`` from ``n_samples``
    when present and from ``n`` otherwise.
    """
    name = "agent_latency_p95"
    common = {
        "name": name, "module": MODULE, "metric_type": "duration", "criticality": "diagnostic",
        "n_floor": 1, "higher_is_better": False, "source_path": aq_src,
    }
    by_class = blk.get("by_class")
    short = _latency_class(by_class, "short_form")
    long_ = _latency_class(by_class, "long_form")

    if short is None and long_ is None:
        # Pre-#871 artifact (or a malformed split): today's behaviour.
        n_samples = _count(blk.get("n_samples"))
        if n_samples is not None:
            n_txt = f"N={n_samples} samples over {blk.get('n')} agents"
            n_graded = n_samples
        else:
            n_txt = f"N={blk.get('n')}"
            n_graded = blk.get("n")
        return build_metric(
            **common, value=blk["value"], unit=MILLISECONDS, n_samples=n_graded,
            reason=f"{name} = {blk['value']:.0f} ms ({n_txt}) vs target 15s / red-line 60s.",
        )

    long_txt = ""
    if long_ is not None:
        long_txt = (
            f" long_form p95 = {long_['p95_ms']:.0f} ms (worst {long_.get('worst_agent')}, "
            f"N={long_.get('n_samples')} samples over {long_.get('n_agents')} agents) is "
            f"reported, not graded: its bar awaits the card owner (alpha-engine-config-I12625)."
        )

    if short is None:
        return build_metric(
            **common, input_present=False,
            na_detail=(
                f"{name}: no short_form agent was sampled this cycle, so nothing is "
                f"graded on 15s / red-line 60s.{long_txt}"
            ),
        )

    short_n = _count(short.get("n_samples"))
    total_n = _count(blk.get("n_samples"))
    total_txt = f"; all classes N={total_n} samples over {blk.get('n')} agents" if total_n is not None else ""
    return build_metric(
        **common, value=float(short["p95_ms"]), unit=MILLISECONDS, n_samples=short_n,
        reason=(
            f"{name} (short_form) = {short['p95_ms']:.0f} ms (worst {short.get('worst_agent')}, "
            f"N={short_n} samples over {short.get('n_agents')} agents{total_txt}) "
            f"vs target 15s / red-line 60s.{long_txt}"
        ),
    )


def build_agent_tile(bucket: str, run_date: str, s3_client=None) -> dict:
    """Build the Behavioural / Agent-Quality tile.

    ``run_date`` keys the ``backtest/{run_date}/agent_quality.json`` producer
    artifact (NYSE trading day, resolved by the handler).
    """
    s3 = s3_client or boto3.client("s3")
    # Windowed resolution (config#1190): freshest within the trailing window.
    aq, _aq_date, aq_age, _aq_key = get_json_windowed(s3, bucket, "backtest/{date}/agent_quality.json", run_date)
    aq_src = f"s3://{bucket}/{_aq_key}" if _aq_key else f"s3://{bucket}/backtest/{run_date}/agent_quality.json"
    aq_stale = artifact_is_stale(aq_age, AGENT_ARTIFACT_MAX_AGE_DAYS)

    def _stale_na(name: str) -> str:
        return (
            f"{name}: agent_quality.json is {aq_age}d old "
            f"(> {AGENT_ARTIFACT_MAX_AGE_DAYS}d max) — stale input, "
            f"cannot grade confidently."
        )

    components = []

    # 1. agent_validation_failure_rate (critical) — % of agent .invoke()s that
    #    failed Pydantic schema validation (lower better). A rising rate means
    #    the agents are emitting malformed structured output the graph has to
    #    repair or drop — a direct agent-quality regression the Director acts on.
    blk = _block(aq, "agent_validation_failure_rate")
    if blk is not None and not aq_stale:
        components.append(build_metric(
            name="agent_validation_failure_rate", module=MODULE, metric_type="pct",
            criticality="critical", estimator="wilson_failure_rate",
            measurement_horizon="per_run",
            value=blk["value"], unit=FRACTION, n_samples=blk.get("n"), n_floor=50,
            higher_is_better=False, source_path=aq_src,
            reason=(f"agent_validation_failure_rate = {blk['value']:.1%} "
                    f"(N={blk.get('n')} invokes) vs target 2% / red-line 10%."),
        ))
    else:
        components.append(build_metric(
            name="agent_validation_failure_rate", module=MODULE, metric_type="pct",
            criticality="critical", estimator="wilson_failure_rate",
            measurement_horizon="per_run",
            n_floor=50, higher_is_better=False,
            source_path=aq_src, input_present=False,
            na_detail=_stale_na("agent_validation_failure_rate") if aq_stale
            else agent_quality_na_reason(aq, "agent_validation_failure_rate"),
        ))

    # 2. cost_per_signal (supporting) — total run LLM $ / finalized signal (lower
    #    better). Efficiency of the research spend per usable output.
    blk = _block(aq, "cost_per_signal")
    if blk is not None and not aq_stale:
        components.append(build_metric(
            name="cost_per_signal", module=MODULE, metric_type="ratio", criticality="supporting",
            value=blk["value"], unit=USD, n_samples=blk.get("n"), n_floor=5,
            higher_is_better=False, source_path=aq_src,
            reason=f"cost_per_signal = ${blk['value']:.2f} over {blk.get('n')} finalized signals vs target $1 / red-line $5.",
        ))
    else:
        components.append(build_metric(
            name="cost_per_signal", module=MODULE, metric_type="ratio", criticality="supporting",
            n_floor=5, higher_is_better=False,
            source_path=aq_src, input_present=False,
            na_detail=_stale_na("cost_per_signal") if aq_stale
            else agent_quality_na_reason(aq, "cost_per_signal"),
        ))

    # 3. retry_storm_count (supporting) — # of agents that reached their per-type
    #    max_retries ceiling (lower better). A storm signals provider instability
    #    or a malformed-output loop burning budget.
    blk = _block(aq, "retry_storm_count")
    if blk is not None and not aq_stale:
        components.append(build_metric(
            name="retry_storm_count", module=MODULE, metric_type="count", criticality="supporting",
            value=blk["value"], unit=COUNT_AGENTS, n_samples=blk.get("n"), n_floor=1,
            higher_is_better=False, source_path=aq_src,
            reason=f"retry_storm_count = {blk['value']:.0f} agents at retry ceiling (of {blk.get('n')}) vs target 0 / red-line 5.",
        ))
    else:
        components.append(build_metric(
            name="retry_storm_count", module=MODULE, metric_type="count", criticality="supporting",
            n_floor=1, higher_is_better=False,
            source_path=aq_src, input_present=False,
            na_detail=_stale_na("retry_storm_count") if aq_stale
            else agent_quality_na_reason(aq, "retry_storm_count"),
        ))

    # 4. agent_latency_p95 (diagnostic) — per-agent-type p95 wall-clock (ms,
    #    lower better). A latency creep flags a slowing agent before it trips the
    #    Lambda ceiling.
    #    Since alpha-engine-config-I12625 the graded value is the SHORT-FORM
    #    class's p95 when the producer splits by class; see _latency_component.
    blk = _block(aq, "agent_latency_p95")
    if blk is not None and not aq_stale:
        components.append(_latency_component(blk, aq_src))
    else:
        components.append(build_metric(
            name="agent_latency_p95", module=MODULE, metric_type="duration", criticality="diagnostic",
            n_floor=1, higher_is_better=False,
            source_path=aq_src, input_present=False,
            na_detail=_stale_na("agent_latency_p95") if aq_stale
            else agent_quality_na_reason(aq, "agent_latency_p95"),
        ))

    # 5. judge_rubric_distribution (diagnostic) — modal-score concentration of
    #    the LLM-judge grades across rubric dimensions (lower better). A high
    #    concentration is rubric collapse: the judge gives the same grade to
    #    everything, so its signal is degenerate.
    blk = _block(aq, "judge_rubric_distribution")
    if blk is not None and not aq_stale:
        components.append(build_metric(
            name="judge_rubric_distribution", module=MODULE, metric_type="ratio", criticality="diagnostic",
            value=blk["value"], unit=FRACTION, n_samples=blk.get("n"), n_floor=10,
            higher_is_better=False, source_path=aq_src,
            reason=f"judge_rubric_distribution modal-concentration = {blk['value']:.0%} (N={blk.get('n')} evals) vs target 40% / red-line 70% (collapse).",
        ))
    else:
        components.append(build_metric(
            name="judge_rubric_distribution", module=MODULE, metric_type="ratio", criticality="diagnostic",
            n_floor=10, higher_is_better=False,
            source_path=aq_src, input_present=False,
            na_detail=_stale_na("judge_rubric_distribution") if aq_stale
            else agent_quality_na_reason(aq, "judge_rubric_distribution"),
        ))

    # 6. stance_source_provenance (diagnostic) — pick-provenance coverage: the
    #    share of signals.json ``universe`` entries carrying a non-empty
    #    ``stance_source`` (crucible-research#297). Until
    #    alpha-engine-config-I12231 this was hard-coded N/A-MISSING-INPUT and
    #    never read the artifact, so the producer's field (902/902 entries on
    #    2026-10-09) could not self-activate as the contract promised.
    components.append(_stance_source_provenance(s3, bucket, run_date))

    # 7. judge_calibration_cohen_kappa (critical) — needs a HUMAN blind-Step-1
    #    labeling pass vs the LLM judge; not a producer (config Batch E #1153).
    components.append(build_metric(
        name="judge_calibration_cohen_kappa", module=MODULE, metric_type="ratio", criticality="critical",
        n_floor=1, source_path=f"s3://{bucket}/decision_artifacts/_calibration/", implemented=False,
        na_detail="judge_calibration_cohen_kappa: needs the blind-Step-1-operator vs LLM-judge κ from decision_artifacts/_calibration/*.jsonl — human labeling, not a producer (config#1153).",
    ))

    # 8-11. groom_* (supporting/diagnostic, config#2151) — groom-pipeline
    #    throughput/efficiency from the trailing week of groom/{date}/*.json
    #    run artifacts. See grading/tiles/groom.py for definitions + posture.
    components.extend(build_groom_components(bucket, run_date, s3_client=s3))

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
