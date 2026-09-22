"""cost_pricing.py — producer for the Substrate-tile ``unpriced_cost_rows``.

**The gap this closes (alpha-engine-config-I11113, deliverable 4 of I11100).**
``krepis.cost.record_llm_call`` degrades rather than drops when it cannot
price a call: a ``PriceCardLookupError`` writes the row anyway with
``cost_usd: None`` / ``cost_source: "unpriced"``, and a provider that
reported no usage writes ``cost_source: "usage_unreported"`` with
``usage_unknown: true``. Both are honest and both are correct. Neither was
ever *read*: as of 2026-09-22 no consumer in ``krepis``,
``crucible-evaluator``, ``crucible-research`` or ``nousergon-data`` scanned
``_cost_raw/**/*.jsonl`` for ``cost_source`` at all, so a served model with
no active price card produced a week of rows whose spend nobody could state
and no surface said so. ``principles.md`` §2.7: a component emitting a
degraded value that no surface renders is unobserved, not healthy.

**This is NOT the fan-in coverage check.** ``crucible-research``
``scripts/cost_coverage.py`` asks "did every stage that ran emit a cost
record?" and raises ``CostCoverageError`` when one did not. That check
passes the moment the row exists — which is exactly what the degrade
guarantees. This module asks the separate, softer question the coverage
check structurally cannot: **the row exists, but can anybody say what it
cost?** A non-zero count here is a DEGRADATION, not a failure: the row is
present, coverage is satisfied, and the only thing lost is the dollar
figure. It therefore renders as one ``supporting`` MetricRecord on the
Report Card's Substrate tile, beside the other substrate-integrity counts,
rather than as a new hard gate.

**Why the whole trailing week and not just ``run_date``.** The card is
weekly and the weekday preopen/postclose pipelines emit cost rows every
trading day; a scan of the Saturday partition alone would report zero for a
Tuesday price-card gap. And ``krepis.cost_sink.S3JsonlCostSink`` partitions
on each record's OWN UTC ``ts``, not on the run's date — the weekly pipeline
starts 09:00 UTC on the day AFTER its ``run_date``, so 100% of its own rows
land one partition ahead (measured 2026-08-29, see ``cost_coverage.py``).
The window is therefore ``[run_date - 6, run_date + 1]``, inclusive: the
seven days the card covers, plus the one-day spill of the cycle doing the
reading.

**Unreadable is not zero.** Every failure to enumerate or read the window
raises :exc:`CostPricingUnmeasured`. "No degraded rows this week" and "I
could not read the partition" are different facts and the surface renders
them differently — collapsing them is the same defect class this module
exists to close. A truncated listing counts as unreadable: an under-count
in the reassuring direction is worse than an honest refusal.
"""

from __future__ import annotations

import json
import logging
from datetime import date as date_type
from datetime import timedelta

logger = logging.getLogger(__name__)

#: Prefix the krepis S3 cost sink partitions under, inside the research bucket.
COST_RAW_PREFIX = "decision_artifacts/_cost_raw"

#: The ``cost_source`` values that mean "the row exists and nobody can say
#: what it cost". BOTH are counted: registering only the newer ``unpriced``
#: would leave the older ``usage_unreported`` path — unmonitored since it was
#: added — reading as covered (alpha-engine-config-I11113 is explicit on this).
DEGRADED_COST_SOURCES: frozenset[str] = frozenset({"unpriced", "usage_unreported"})

#: Trailing days of partitions the weekly card covers, plus the +1 spill day.
WINDOW_DAYS = 7
SPILL_DAYS = 1

#: Hard ceiling on objects enumerated in one scan. Sized ~10x the observed
#: weekly object count (a full week of all three pipelines is O(100) objects,
#: one per callsite per flush). Hitting it means the prefix no longer looks
#: like what this module was built to read, so the scan REFUSES rather than
#: reporting the partial count it managed — see the module docstring.
MAX_OBJECTS = 5000


class CostPricingUnmeasured(RuntimeError):
    """The degraded-row count could not be established for the window.

    Deliberately distinct from a count of zero, and deliberately distinct from
    ``cost_coverage.CostCoverageError`` (a finding about the pipeline) and
    ``CostCoverageUnmeasured`` (a fault in *that* check). This one means the
    ``_cost_raw`` window could not be enumerated or read in full.
    """


def window_partitions(run_date: str, *, window_days: int = WINDOW_DAYS,
                      spill_days: int = SPILL_DAYS) -> list[str]:
    """The ISO dates whose ``_cost_raw`` partitions the scan reads.

    ``run_date`` is the cycle's trading day (``$.run_date``, the
    ``trading_day`` partition family). Raises ``ValueError`` on a date this
    module cannot parse — a caller passing garbage must not silently get an
    empty window, which would read as zero.
    """
    anchor = date_type.fromisoformat(run_date)
    first = anchor - timedelta(days=window_days - 1)
    last = anchor + timedelta(days=spill_days)
    out, d = [], first
    while d <= last:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _callsite_of(row: dict, key: str) -> str:
    """The callsite a degraded row belongs to.

    ``krepis`` stamps ``callsite_id`` on every ``LLMClient`` row and mirrors it
    to ``agent_id`` for the aggregator's schema. Pre-``callsite_id`` rows and
    rows written by a non-``LLMClient`` producer carry neither, so the object
    key's own basename is the fallback — the sink names each object for its
    callsite (``.../<callsite>.<n>.jsonl``). "(unknown)" is a last resort and
    is a real bucket, never an omission.
    """
    for field in ("callsite_id", "agent_id"):
        v = row.get(field)
        if isinstance(v, str) and v.strip():
            return v.strip()
    basename = key.rsplit("/", 1)[-1]
    stem = basename.split(".", 1)[0]
    return stem or "(unknown)"


def scan_degraded_cost_rows(
    s3,
    bucket: str,
    run_date: str,
    *,
    window_days: int = WINDOW_DAYS,
    spill_days: int = SPILL_DAYS,
    max_objects: int = MAX_OBJECTS,
) -> dict:
    """Count ``_cost_raw`` rows whose ``cost_source`` is a degraded value.

    Returns::

        {
          "run_date": "2026-09-19",
          "partitions": ["2026-09-13", ..., "2026-09-20"],
          "objects_scanned": 41,
          "rows_scanned": 1180,
          "degraded_total": 3,
          "by_callsite": {"director-plan": {"unpriced": 3, "usage_unreported": 0,
                                            "total": 3}},
          "by_cost_source": {"unpriced": 3, "usage_unreported": 0},
        }

    Raises :exc:`CostPricingUnmeasured` if any partition cannot be listed, any
    object cannot be fetched, any line is not JSON, or the object ceiling is
    hit. There is no partial-credit return value: an under-count would be
    indistinguishable from health, which is the defect being closed.
    """
    try:
        partitions = window_partitions(run_date, window_days=window_days,
                                       spill_days=spill_days)
    except (TypeError, ValueError) as exc:
        raise CostPricingUnmeasured(
            f"unparseable run_date {run_date!r} — cannot establish the "
            f"_cost_raw window to scan"
        ) from exc

    by_callsite: dict[str, dict[str, int]] = {}
    by_cost_source = {src: 0 for src in sorted(DEGRADED_COST_SOURCES)}
    objects_scanned = 0
    rows_scanned = 0

    paginator = s3.get_paginator("list_objects_v2")
    for part in partitions:
        prefix = f"{COST_RAW_PREFIX}/{part}/"
        try:
            pages = list(paginator.paginate(Bucket=bucket, Prefix=prefix))
        except Exception as exc:  # noqa: BLE001 — re-raised as unmeasured, never swallowed
            raise CostPricingUnmeasured(
                f"could not list s3://{bucket}/{prefix}: {type(exc).__name__}: {exc}"
            ) from exc
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".jsonl"):
                    continue
                objects_scanned += 1
                if objects_scanned > max_objects:
                    raise CostPricingUnmeasured(
                        f"more than {max_objects} .jsonl objects under "
                        f"{COST_RAW_PREFIX}/ for {partitions[0]}..{partitions[-1]} "
                        f"— refusing to report a truncated count"
                    )
                try:
                    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                except Exception as exc:  # noqa: BLE001 — re-raised as unmeasured
                    raise CostPricingUnmeasured(
                        f"could not read s3://{bucket}/{key}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                for lineno, line in enumerate(body.splitlines(), start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except (ValueError, UnicodeDecodeError) as exc:
                        raise CostPricingUnmeasured(
                            f"malformed JSONL at s3://{bucket}/{key}:{lineno} "
                            f"— the window cannot be counted: {exc}"
                        ) from exc
                    if not isinstance(row, dict):
                        raise CostPricingUnmeasured(
                            f"non-object JSONL record at s3://{bucket}/{key}:{lineno}"
                        )
                    rows_scanned += 1
                    src = row.get("cost_source")
                    if src not in DEGRADED_COST_SOURCES:
                        continue
                    by_cost_source[src] += 1
                    site = _callsite_of(row, key)
                    bucket_row = by_callsite.setdefault(
                        site, {s: 0 for s in sorted(DEGRADED_COST_SOURCES)} | {"total": 0}
                    )
                    bucket_row[src] += 1
                    bucket_row["total"] += 1

    return {
        "run_date": run_date,
        "partitions": partitions,
        "objects_scanned": objects_scanned,
        "rows_scanned": rows_scanned,
        "degraded_total": sum(by_cost_source.values()),
        "by_callsite": by_callsite,
        "by_cost_source": by_cost_source,
    }


def format_by_callsite(by_callsite: dict[str, dict[str, int]], *, limit: int = 5) -> str:
    """Compact per-callsite breakdown for a MetricRecord ``status_reason``.

    Ordered worst-first so the reason names the callsite an operator should
    look at, and truncated with an explicit "+N more" rather than silently.
    """
    if not by_callsite:
        return "none"
    ranked = sorted(by_callsite.items(), key=lambda kv: (-kv[1]["total"], kv[0]))
    shown = ranked[:limit]
    parts = []
    for site, counts in shown:
        detail = ", ".join(
            f"{src}={counts.get(src, 0)}"
            for src in sorted(DEGRADED_COST_SOURCES)
            if counts.get(src, 0)
        )
        parts.append(f"{site} ({detail})")
    if len(ranked) > limit:
        parts.append(f"+{len(ranked) - limit} more callsite(s)")
    return "; ".join(parts)
