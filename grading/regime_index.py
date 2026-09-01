"""regime_index.py — O(1) steady-state ``date -> market_regime`` lookup.

**The defect this closes (alpha-engine-config-I9702).** Tile 0's
``regime_weighted_alpha`` component joins every date in ``trades/eod_pnl.csv``
to the top-level ``market_regime`` field of ``signals/{date}/signals.json``.
Until this module existed that join was ONE SEQUENTIAL S3 ``get_object`` PER
DATE, unconditionally, over the full since-inception history — 121 GETs of a
~637KB artifact apiece, to read one short string each. The cost grows by one
GET per weekday, forever, and it is the dominant term in the
``alpha-engine-evaluator`` Lambda's measured duration regression: max duration
never exceeded ~68s through 2026-08-20, then 209.5s (08-21), 220.0s (08-22),
293.2s (08-28), 302.1s (08-30), with two ReportCard Step Functions states dying
at exactly 300.000s (``States.Timeout``) on 2026-08-30. Memory was never the
constraint — peak Max Memory Used is 755-759MB of a 1024MB allocation. The
bottleneck is sequential network I/O, and raising the function timeout (300s ->
660s, already deployed without a published profile) only moves the breach out
roughly four weeks.

**The fix.** One small S3 object, ``evaluator/indexes/market_regime.json``,
maps each already-resolved date to its regime. Each build reads that one
object, determines which of today's ``eod_pnl.csv`` dates are not in it, fetches
only those (through a bounded thread pool), merges, and hands the merged
document back for the handler to persist. Steady state is 1 GET plus a handful
of misses instead of N GETs, and the miss set is bounded by the number of
trading days added since the last successful build.

**Negative entries are recorded, not re-derived.** A date whose
``signals.json`` is absent, unparseable, or carries no ``market_regime`` field
is stored as ``null``. Without that, every permanently-unresolvable date would
be re-fetched on every build forever and the index would not actually be O(1) —
the exact defect, relocated. ``_MISS_REPROBE_DAYS`` re-probes only the most
recent handful of null entries, so a date whose producer wrote late is still
picked up, at a cost bounded by a constant rather than by history length.

**Failure posture (fail-loud, fleet rule).** An index that is present but
unreadable — non-JSON, not an object, or carrying a ``schema_version`` this
code does not implement — RAISES. It is a cache of immutable inputs, so the
tempting move is to discard it and rebuild silently; that would hide a real
corruption behind a run that merely got slow again, which is precisely the
class of silence this arc exists to remove. The operator remedy is one line and
is named in the exception. A genuinely ABSENT index (``NoSuchKey``) is not an
error — it is genesis, and it backfills on the next build. An unresolvable DATE
is likewise not an error: that is the pre-existing card contract
(``_regime_na_detail`` in ``grading/tiles/portfolio_outcome.py`` renders it as a
specific N/A), and this module preserves it while making the count of such
dates a persisted, countable field rather than an invisible skip.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# Under the "evaluator" prefix, which grading/iam_s3_contract.json already
# declares `readwrite` (the report card, the self test and the attribution
# artifact all live there) — so this needs no new IAM grant and no ops-repo
# change. `indexes/` rather than a dated directory because the object is
# cumulative state, not a per-cycle snapshot; it deliberately does NOT match
# grading/history.py::_CARD_KEY_RE, so it can never be mistaken for a card.
REGIME_INDEX_KEY = "evaluator/indexes/market_regime.json"
SCHEMA_VERSION = 1
ARTIFACT_NAME = "market_regime_index"
SIGNALS_KEY_TEMPLATE = "signals/{date}/signals.json"
REGIME_FIELD = "market_regime"

# Bounded concurrency for the miss path. Chosen at 8, not higher:
#   * botocore's default `max_pool_connections` is 10. Beyond that, urllib3
#     opens and discards connections outside the pool ("Connection pool is
#     full") — more threads buy latency and churn, not throughput. 8 sits
#     under the pool with headroom for the concurrent tile builds sharing the
#     same client.
#   * Memory: each in-flight signals.json is ~637KB raw plus its parsed dict,
#     call it ~3MB resident per worker. 8 workers is ~25MB against a measured
#     755-759MB peak in a 1024MB Lambda — ~265MB of headroom, so this cannot
#     be what pushes the function into an OOM.
#   * S3 sustains far more than 8 concurrent GETs on one prefix; the ceiling
#     here is the Lambda, not the service.
# Worst case (a cold index over the full 121-date history) is 121/8 ~= 16
# waves of a ~100ms GET, i.e. a couple of seconds instead of ~12s+ sequential.
_MAX_FETCH_WORKERS = 8

# How many of the most recent dates get their NEGATIVE index entry re-probed on
# every build. A `signals.json` written after the date first appeared in
# eod_pnl.csv would otherwise be tombstoned forever. Bounded by a constant, so
# the steady-state GET count does not grow with history length.
_MISS_REPROBE_DAYS = 7


class RegimeIndexError(RuntimeError):
    """The persisted regime index exists but cannot be trusted."""


@dataclass
class RegimeResolution:
    """What one ``resolve_regimes`` call produced.

    ``regimes`` is the caller-facing join (unresolvable dates simply absent —
    the pre-existing contract). ``index`` is the merged document to persist, or
    ``None`` when nothing changed and a write would be a no-op PUT. ``gets``
    counts signals.json fetches actually performed, which is what the
    regression test in ``tests/test_regime_index.py`` asserts does not grow
    with the number of sessions.
    """

    regimes: dict[str, str]
    index: dict | None
    gets: int = 0
    unresolved: list[str] = field(default_factory=list)


def _empty_index() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT_NAME,
        "source_key_template": SIGNALS_KEY_TEMPLATE,
        "field": REGIME_FIELD,
        "updated_at": None,
        "entries": {},
    }


def read_index(bucket: str, s3_client=None) -> dict:
    """Read the persisted index, or an empty one if it has never been written.

    Raises ``RegimeIndexError`` when the object is present but not a document
    this code can safely merge into, and re-raises any S3 fault that is not
    ``NoSuchKey`` — a throttle or an ``AccessDenied`` is not "no index".
    """
    s3 = s3_client or boto3.client("s3")
    uri = f"s3://{bucket}/{REGIME_INDEX_KEY}"
    try:
        resp = s3.get_object(Bucket=bucket, Key=REGIME_INDEX_KEY)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("NoSuchKey", "404", "NoSuchBucket"):
            logger.info(
                "regime index absent at %s — genesis build, backfilling the full "
                "eod_pnl.csv date range this cycle.", uri,
            )
            return _empty_index()
        logger.error("S3 read failed for %s: %s", uri, e)
        raise

    raw = resp["Body"].read()
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise RegimeIndexError(
            f"{uri} is not valid JSON ({exc}). This object is a derived cache of "
            f"immutable signals.json inputs; delete the key to force a clean "
            f"rebuild on the next grading cycle."
        ) from exc
    if not isinstance(payload, dict):
        raise RegimeIndexError(
            f"{uri} is not a JSON object (got {type(payload).__name__}). Delete "
            f"the key to force a clean rebuild on the next grading cycle."
        )
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise RegimeIndexError(
            f"{uri} declares schema_version={version!r}, but this code implements "
            f"v{SCHEMA_VERSION} only. Merging into an index whose shape is not "
            f"understood would corrupt it. If {version!r} is older, delete the key "
            f"to force a clean rebuild; if it is newer, this Lambda is behind its "
            f"own artifact and must be redeployed."
        )
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise RegimeIndexError(
            f"{uri} carries no `entries` object (got {type(entries).__name__}). "
            f"Delete the key to force a clean rebuild."
        )
    normalized = _empty_index()
    normalized.update(payload)
    normalized["entries"] = {
        str(k): (str(v) if isinstance(v, str) and v else None)
        for k, v in entries.items()
    }
    return normalized


def _fetch_one(s3, bucket: str, date_s: str) -> str | None:
    """Resolve one date's regime from its ``signals.json``, or ``None``.

    ``None`` means *probed and unresolvable* — the object is absent, or is not
    parseable JSON, or carries no non-empty top-level ``market_regime``. The
    caller records that as a negative index entry rather than discarding it, so
    the same date is not re-fetched on every future build. Any S3 fault other
    than a missing key RAISES: a throttle or an auth failure must not be read
    as "this date has no regime" and then tombstoned as one.
    """
    key = SIGNALS_KEY_TEMPLATE.format(date=date_s)
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        logger.error("S3 read failed for s3://%s/%s: %s", bucket, key, e)
        raise
    try:
        payload = json.loads(resp["Body"].read())
    except (json.JSONDecodeError, ValueError):
        # Swallowed failure mode: ONE corrupt/half-written signals.json. It is
        # swallowed because the producer owns that artifact's integrity and one
        # bad date must not fail the whole report card — the pre-existing
        # contract (config#857 C2-fu). Recording surface: this WARN in the
        # Lambda log, plus a persisted `null` entry for the date in
        # evaluator/indexes/market_regime.json and the date's inclusion in the
        # index's `n_unresolved` count and in the component's own N/A detail.
        logger.warning("Skipping corrupt signals.json at s3://%s/%s", bucket, key)
        return None
    if not isinstance(payload, dict):
        logger.warning(
            "signals.json at s3://%s/%s is not a JSON object — no regime.", bucket, key,
        )
        return None
    regime = payload.get(REGIME_FIELD)
    return regime if isinstance(regime, str) and regime else None


def _dates_to_fetch(dates: list[str], entries: dict[str, str | None]) -> list[str]:
    """Every date with no entry, plus the most recent ``_MISS_REPROBE_DAYS`` nulls.

    Order-preserving and de-duplicated. This is the whole O(1) claim: the first
    term is the dates added since the last build, the second is a constant.
    """
    recent = set(dates[-_MISS_REPROBE_DAYS:])
    missing: list[str] = []
    seen: set[str] = set()
    for d in dates:
        if d in seen:
            continue
        known = d in entries
        if not known or (entries.get(d) is None and d in recent):
            missing.append(d)
            seen.add(d)
    return missing


def resolve_regimes(bucket: str, dates: list[str], s3_client=None) -> RegimeResolution:
    """Join ``dates`` to their ``market_regime`` tags through the persisted index.

    Reads the index (1 GET), fetches only the dates it does not already answer
    — concurrently, bounded by ``_MAX_FETCH_WORKERS`` — merges, and returns the
    merged document for the caller to persist. Never writes: the tile builders
    in this repo are pure of S3 writes so the pre-promotion deploy canary
    (``{"write": false}``) genuinely writes nothing, and every derived artifact
    is persisted by ``grading/handler.py`` under its ``write`` flag.
    """
    s3 = s3_client or boto3.client("s3")
    index = read_index(bucket, s3_client=s3)
    entries: dict[str, str | None] = index["entries"]
    gets = 1  # the index read itself

    to_fetch = _dates_to_fetch(dates, entries)
    fetched: dict[str, str | None] = {}
    if to_fetch:
        workers = min(_MAX_FETCH_WORKERS, len(to_fetch))
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="regime-fetch",
        ) as pool:
            results = list(pool.map(lambda d: (d, _fetch_one(s3, bucket, d)), to_fetch))
        # `pool.map` re-raises the first exception in submission order, so a
        # real S3 fault on any date still fails the build loudly and
        # deterministically — same posture as the sequential reader it replaces.
        fetched = dict(results)
        gets += len(to_fetch)
        logger.info(
            "regime index: %d/%d dates already indexed, %d fetched with %d workers "
            "(%d resolved, %d unresolved).",
            len(dates) - len(to_fetch), len(dates), len(to_fetch), workers,
            sum(1 for v in fetched.values() if v), sum(1 for v in fetched.values() if not v),
        )
    else:
        logger.info("regime index: all %d dates served from the index (0 fetches).", len(dates))

    dirty = False
    for d, regime in fetched.items():
        if d not in entries or entries[d] != regime:
            # A null re-probe that is still null is NOT a change — writing the
            # index every cycle for a no-op would defeat the point of tracking
            # dirtiness at all.
            if d in entries and entries[d] is None and regime is None:
                continue
            entries[d] = regime
            dirty = True

    regimes = {d: entries[d] for d in dates if entries.get(d)}
    unresolved = [d for d in dates if not entries.get(d)]

    merged: dict | None = None
    if dirty:
        index["schema_version"] = SCHEMA_VERSION
        index["artifact"] = ARTIFACT_NAME
        index["source_key_template"] = SIGNALS_KEY_TEMPLATE
        index["field"] = REGIME_FIELD
        index["updated_at"] = datetime.now(timezone.utc).isoformat()
        index["n_entries"] = len(entries)
        index["n_resolved"] = sum(1 for v in entries.values() if v)
        index["n_unresolved"] = sum(1 for v in entries.values() if not v)
        index["entries"] = dict(sorted(entries.items()))
        merged = index

    return RegimeResolution(
        regimes=regimes, index=merged, gets=gets, unresolved=unresolved,
    )


def write_index(bucket: str, index: dict, s3_client=None) -> str:
    """Persist the merged index. Returns the key written."""
    s3 = s3_client or boto3.client("s3")
    s3.put_object(
        Bucket=bucket,
        Key=REGIME_INDEX_KEY,
        Body=json.dumps(index, indent=2, sort_keys=False).encode("utf-8"),
        ContentType="application/json",
    )
    return REGIME_INDEX_KEY
