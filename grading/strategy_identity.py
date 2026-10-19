"""alpha-engine-config-I11089 — which DEPLOYED strategy each eod row was traded under.

``trades/eod_pnl.csv`` carries no strategy or config column, so a window's
``strategy_epoch`` is ``undeclared`` and a since-inception figure spans every
configuration that traded in it. The executor does record identity, one object
per session, at plan time:

    s3://{bucket}/order_books/{date}/summary.json

with ``champion`` (the candidate-source arm) and ``promotion_source``, and since
crucible-executor#597 also ``executor_sha``, ``config_sha256`` (SHA-256 of the
effective run config) and ``champion_pointer_version`` (the S3 VersionId of
``config/producer_champion.json`` the run read). This module JOINS each eod date
to that object and reports the result as run-length segments.

Read-only and additive. Nothing here writes to S3, changes how any figure is
computed, or raises: provenance that cannot be read says so (``read_errors``,
``dates_without_summary``) and the card still builds. Dates with no summary, and
summaries that predate a field, are reported as unstamped, never filled in.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

ORDER_BOOK_SUMMARY_KEY_TEMPLATE = "order_books/{date}/summary.json"

# The identity fields the executor stamps. `champion` / `promotion_source` since
# config#2364; the other three since crucible-executor#597 (2026-10-15 preopen).
IDENTITY_FIELDS = (
    "champion", "promotion_source", "executor_sha", "config_sha256",
    "champion_pointer_version",
)

# Under botocore's default max_pool_connections (10), same bound and reasoning
# as grading/regime_index.py. summary.json is small, so memory is not the limit.
_MAX_FETCH_WORKERS = 8

UNSTAMPED = "unstamped"

NOTE = (
    "joined from order_books/{date}/summary.json, not read from eod_pnl.csv. "
    "A segment is a run of consecutive eod dates with the same (champion, "
    "config_sha256). Dates with no summary, or a summary predating a field, are "
    "unstamped: they are not inferred from neighbouring dates."
)


def _fetch_one(s3, bucket: str, date_s: str) -> tuple[dict | None, str | None]:
    """(summary, None) on success; (None, None) when absent; (None, reason) on a fault."""
    key = ORDER_BOOK_SUMMARY_KEY_TEMPLATE.format(date=date_s)
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
        body = json.loads(resp["Body"].read())
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404"):
            return None, None
        return None, f"{date_s}: {code or type(e).__name__}"
    except Exception as e:  # noqa: BLE001 — provenance must never fail the card
        return None, f"{date_s}: {type(e).__name__}"
    if not isinstance(body, dict):
        return None, f"{date_s}: summary.json is not an object"
    return body, None


def _identity(summary: dict | None) -> dict:
    """The identity fields of one summary, ``None`` for any that is absent."""
    s = summary or {}
    return {f: (s.get(f) if isinstance(s.get(f), str) and s.get(f) else None) for f in IDENTITY_FIELDS}


def _strategy_key(ident: dict) -> tuple:
    return (ident["champion"], ident["config_sha256"])


def build_strategy_identity(bucket: str, dates: list[str], s3_client=None) -> dict:
    """Join ``dates`` (eod_pnl.csv trading dates) to the executor's order-book summaries.

    Never raises. Returns a JSON-serialisable dict; see the module docstring.
    """
    uniq = sorted(set(dates))
    source = f"s3://{bucket}/{ORDER_BOOK_SUMMARY_KEY_TEMPLATE}"
    out: dict = {
        "source": source,
        "note": NOTE,
        "dates_requested": len(uniq),
        "dates_stamped": 0,
        "dates_without_summary": 0,
        "read_errors": [],
        "segments": [],
    }
    if not uniq:
        return out
    try:
        s3 = s3_client or boto3.client("s3")
        workers = min(_MAX_FETCH_WORKERS, len(uniq))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="strategy-identity") as pool:
            fetched = list(pool.map(lambda d: (d, *_fetch_one(s3, bucket, d)), uniq))
    except Exception as e:  # noqa: BLE001 — e.g. no credentials; the card must still build
        logger.warning("strategy_identity: could not read %s: %s", source, e)
        out["read_errors"].append(f"all: {type(e).__name__}")
        out["dates_without_summary"] = len(uniq)
        out["segments"] = [_segment(uniq, [_identity(None)] * len(uniq))]
        return out

    idents: list[dict] = []
    for d, summary, err in fetched:
        if err:
            out["read_errors"].append(err)
        if summary is None:
            out["dates_without_summary"] += 1
        idents.append(_identity(summary))
    out["dates_stamped"] = sum(1 for i in idents if i["champion"] is not None)

    run_dates: list[str] = []
    run_idents: list[dict] = []
    for d, ident in zip(uniq, idents):
        if run_idents and _strategy_key(ident) != _strategy_key(run_idents[-1]):
            out["segments"].append(_segment(run_dates, run_idents))
            run_dates, run_idents = [], []
        run_dates.append(d)
        run_idents.append(ident)
    out["segments"].append(_segment(run_dates, run_idents))
    return out


def _distinct(idents: list[dict], field: str) -> list[str]:
    return sorted({i[field] for i in idents if i[field] is not None})


def _segment(dates: list[str], idents: list[dict]) -> dict:
    champion = idents[0]["champion"]
    config = idents[0]["config_sha256"]
    return {
        "start": dates[0],
        "end": dates[-1],
        "trading_days": len(dates),
        "champion": champion if champion is not None else UNSTAMPED,
        "config_sha256": config if config is not None else UNSTAMPED,
        # Vary within a segment without changing the strategy key: executor code
        # moves almost daily and the pointer VersionId every Saturday.
        "promotion_sources": _distinct(idents, "promotion_source"),
        "executor_shas": _distinct(idents, "executor_sha"),
        "champion_pointer_versions": _distinct(idents, "champion_pointer_version"),
    }
