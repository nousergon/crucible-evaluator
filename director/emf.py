"""emf.py — one transport for the Director's CloudWatch metrics on either host.

The Director publishes its metrics as Embedded Metric Format records: a JSON
line on stdout that CloudWatch Logs turns into datapoints. That works on
Lambda, which ships stdout to a log group that CloudWatch extracts EMF from.
It does NOT work on the weekly spot (``director/box_run.py``,
alpha-engine-config-I11936): there stdout goes to the SSM command output and
``krepis.ssm_log_capture``'s S3 copy, and nothing extracts a metric from
either. Measured 2026-10-07: the 2026-10-04 on-box run made a 332.6s plan
call, and ``AlphaEngine/Director`` has no datapoint after 2026-10-03, the
last Lambda run. ``DirectorPlanLatency*`` and ``RetroJudge*`` had gone dark,
which lets the plan-latency alarm fall back to OK on silence and its
no-datapoint deadman page about a Director that is running.

:func:`emit` prints the record exactly as before, so the Lambda path and
every log reader are unchanged. Once :func:`enable_direct_put` has been called
(``box_run.main`` calls it), it also publishes the same values with
``PutMetricData``; the spot's instance role already grants it for the
``AlphaEngine/*`` namespaces. Lambda never enables it, because there the EMF
line already counts and a second publish would double every SampleCount.

Never raises: a telemetry failure must not take down a Director run.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_DIRECT_PUT = False


def enable_direct_put() -> None:
    """Also publish every EMF record via PutMetricData (off-Lambda hosts only)."""
    global _DIRECT_PUT
    _DIRECT_PUT = True


def direct_put_enabled() -> bool:
    return _DIRECT_PUT


def metric_data(record: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    """``(namespace, MetricData)`` pairs equivalent to one EMF record.

    One datapoint per (directive, dimension set, metric), the same fan-out
    CloudWatch applies to the EMF line. A metric whose value is missing or not
    numeric is skipped, as EMF extraction would skip it.
    """
    meta = record.get("_aws") or {}
    ts_ms = meta.get("Timestamp")
    timestamp = (
        datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        if isinstance(ts_ms, (int, float)) else datetime.now(timezone.utc)
    )
    out: list[tuple[str, list[dict[str, Any]]]] = []
    for directive in meta.get("CloudWatchMetrics") or []:
        namespace = directive.get("Namespace")
        if not namespace:
            continue
        dim_sets = directive.get("Dimensions") or [[]]
        data: list[dict[str, Any]] = []
        for dim_names in dim_sets:
            dims = [
                {"Name": name, "Value": str(record[name])}
                for name in dim_names if record.get(name) is not None
            ]
            if len(dims) != len(dim_names):
                continue
            for metric in directive.get("Metrics") or []:
                name = metric.get("Name")
                value = record.get(name)
                if isinstance(value, bool):
                    value = int(value)
                if not isinstance(value, (int, float)):
                    continue
                datum: dict[str, Any] = {
                    "MetricName": name,
                    "Dimensions": dims,
                    "Timestamp": timestamp,
                    "Value": float(value),
                }
                if metric.get("Unit"):
                    datum["Unit"] = metric["Unit"]
                data.append(datum)
        if data:
            out.append((namespace, data))
    return out


def _put(record: dict[str, Any]) -> None:
    import boto3

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    client = boto3.client("cloudwatch", region_name=region)
    for namespace, data in metric_data(record):
        # PutMetricData takes at most 1000 datums per call; a Director record
        # carries at most eight.
        for start in range(0, len(data), 1000):
            client.put_metric_data(Namespace=namespace, MetricData=data[start:start + 1000])


def emit(record: dict[str, Any]) -> None:
    """Print ``record`` as one EMF line and, off Lambda, publish it directly.

    The print is not wrapped here: callers keep their own ``try`` and their
    own log line naming which alarm goes blind, as they always have.
    """
    print(json.dumps(record))
    if not _DIRECT_PUT or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        return
    try:
        _put(record)
    except Exception:  # noqa: BLE001 — telemetry must never fail the run
        names = [
            m.get("Name")
            for d in (record.get("_aws") or {}).get("CloudWatchMetrics") or []
            for m in d.get("Metrics") or []
        ]
        logger.exception(
            "Director: PutMetricData failed for %s. The EMF line above was "
            "printed, but on this host nothing extracts it, so these metrics "
            "have no datapoint for this run", names,
        )
