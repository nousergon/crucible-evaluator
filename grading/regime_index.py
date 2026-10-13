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

**The join is AS-OF, not exact-date (alpha-engine-config-I12230).** The only
producer of ``signals/{date}/signals.json`` is the ``signals_envelope`` Lambda,
invoked once per WEEKLY Step Functions run — usually dated a Friday, sometimes a
Saturday or Sunday. ``eod_pnl.csv`` is DAILY. An exact-date join therefore
resolved about one trading day in five: measured 2026-10-12, 43 of 149 rows
(71.1% unresolved), and every one of the 106 misses was an absent file, not a
file missing the field. The regime a weekly run publishes is the regime in
force until the next run, so each row now takes the ``market_regime`` of the
LATEST ``signals.json`` dated on or before the row date, no more than
``ASOF_MAX_AGE_DAYS`` calendar days older. Over the same 2026-10-12 data that
resolves 149 of 149.

* **No look-ahead.** A row only ever reads signal dates ``<= row date`` — the
  same-day pairing the exact join already used, extended backwards only.
* **Weekend-dated runs count.** 11 of the 58 live ``signals.json`` files are
  dated on a Saturday or Sunday, which are never ``eod_pnl.csv`` rows, so the
  probe set is no longer just the row dates: a row with no same-day regime
  also probes the non-row calendar days behind it, back to the nearest day the
  index already resolves. Each such day is fetched once and tombstoned like
  any other, so the steady-state GET count stays bounded by a constant.
* **Bounded staleness.** Past ``ASOF_MAX_AGE_DAYS`` the tag is stale (the
  producer skipped a week), and the row stays UNRESOLVED and counted rather
  than inheriting an old regime.

The persisted index is unchanged in shape (still ``signals date -> regime or
null``, ``schema_version`` 1): it caches exact per-date probes, and the as-of
step is a pure in-memory pass over it. Which signal date each row inherited is
returned as ``RegimeResolution.sources``.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

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

# How many calendar days a published regime stays in force for the as-of join
# (alpha-engine-config-I12230). The producer is weekly, so 7 covers a run that
# slipped a day (Friday -> Saturday) without letting a SKIPPED week inherit a
# fortnight-old tag. Measured 2026-10-12 over the live 149-row book and all 58
# signals.json files: 3 days resolves 86 rows, 5 resolves 131, 7 resolves 149.
ASOF_MAX_AGE_DAYS = 7


class RegimeIndexError(RuntimeError):
    """The persisted regime index exists but cannot be trusted."""


@dataclass
class RegimeResolution:
    """What one ``resolve_regimes`` call produced.

    ``regimes`` is the caller-facing AS-OF join (unresolvable dates simply
    absent — the pre-existing contract). ``sources`` maps each resolved date to
    the ``signals.json`` date its regime came from (equal to the date itself
    for a same-day match). ``index`` is the merged document to persist, or
    ``None`` when nothing changed and a write would be a no-op PUT. ``gets``
    counts signals.json fetches actually performed, which is what the
    regression test in ``tests/test_regime_index.py`` asserts does not grow
    with the number of sessions.
    """

    regimes: dict[str, str]
    index: dict | None
    gets: int = 0
    unresolved: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)

    @property
    def n_carried(self) -> int:
        """Rows whose regime came from an EARLIER signals.json, not same-day."""
        return sum(1 for d, src in self.sources.items() if src != d)


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


def _parse_date(date_s: str) -> date | None:
    try:
        return date.fromisoformat(date_s)
    except (TypeError, ValueError):
        return None


def _asof_candidates(dates: list[str], entries: dict[str, str | None]) -> list[str]:
    """Non-row calendar days that could carry the regime in force on a row.

    Only rows the index does not resolve same-day need one. For each, walk
    back up to ``ASOF_MAX_AGE_DAYS`` and stop at the first day the index
    already resolves — nothing older can be the latest. Row dates on the way
    are skipped (``_dates_to_fetch`` owns them); a non-row day is returned when
    the index has never probed it, or when it is a null no older than the
    ``_MISS_REPROBE_DAYS``-th most recent row (the same late-writer re-probe
    the row dates get). Order-preserving and de-duplicated.
    """
    rows = set(dates)
    recent_floor = min(dates[-_MISS_REPROBE_DAYS:]) if dates else ""
    out: list[str] = []
    seen: set[str] = set()
    for d in dates:
        if entries.get(d):
            continue
        row_day = _parse_date(d)
        if row_day is None:
            continue
        for back in range(1, ASOF_MAX_AGE_DAYS + 1):
            c = (row_day - timedelta(days=back)).isoformat()
            if entries.get(c):
                break
            if c in rows or c in seen:
                continue
            if c not in entries or c >= recent_floor:
                out.append(c)
                seen.add(c)
    return out


def _asof_join(
    dates: list[str], entries: dict[str, str | None],
) -> tuple[dict[str, str], dict[str, str]]:
    """Each date's regime = the latest resolved entry on or before it, within
    ``ASOF_MAX_AGE_DAYS``. Returns ``(regimes, sources)``. Never looks forward."""
    regimes: dict[str, str] = {}
    sources: dict[str, str] = {}
    for d in dates:
        if entries.get(d):
            regimes[d], sources[d] = entries[d], d
            continue
        row_day = _parse_date(d)
        if row_day is None:
            continue
        for back in range(1, ASOF_MAX_AGE_DAYS + 1):
            c = (row_day - timedelta(days=back)).isoformat()
            if entries.get(c):
                regimes[d], sources[d] = entries[c], c
                break
    return regimes, sources


def _fetch_many(s3, bucket: str, to_fetch: list[str]) -> dict[str, str | None]:
    """Fetch ``to_fetch`` through the bounded pool. Empty in, empty out."""
    if not to_fetch:
        return {}
    workers = min(_MAX_FETCH_WORKERS, len(to_fetch))
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="regime-fetch",
    ) as pool:
        results = list(pool.map(lambda d: (d, _fetch_one(s3, bucket, d)), to_fetch))
    # `pool.map` re-raises the first exception in submission order, so a
    # real S3 fault on any date still fails the build loudly and
    # deterministically — same posture as the sequential reader it replaces.
    return dict(results)


def _merge(entries: dict[str, str | None], fetched: dict[str, str | None]) -> bool:
    """Merge ``fetched`` into ``entries``; True when anything changed."""
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
    return dirty


def resolve_regimes(bucket: str, dates: list[str], s3_client=None) -> RegimeResolution:
    """As-of join of ``dates`` to the ``market_regime`` in force on each date.

    Reads the index (1 GET), fetches only the dates it does not already answer
    — the row dates first, then any non-row calendar days an as-of lookup still
    needs (I12230) — concurrently, bounded by ``_MAX_FETCH_WORKERS``. Each date
    then takes the regime of the latest resolved signals date on or before it,
    within ``ASOF_MAX_AGE_DAYS``. Merges, and returns the
    merged document for the caller to persist. Never writes: the tile builders
    in this repo are pure of S3 writes so the pre-promotion deploy canary
    (``{"write": false}``) genuinely writes nothing, and every derived artifact
    is persisted by ``grading/handler.py`` under its ``write`` flag.
    """
    s3 = s3_client or boto3.client("s3")
    index = read_index(bucket, s3_client=s3)
    entries: dict[str, str | None] = index["entries"]
    gets = 1  # the index read itself

    # Pass 1: the row dates themselves (new dates + recent-null re-probes).
    to_fetch = _dates_to_fetch(dates, entries)
    fetched = _fetch_many(s3, bucket, to_fetch)
    dirty = _merge(entries, fetched)
    # Pass 2 (I12230): the non-row calendar days an as-of lookup still needs —
    # weekend-dated weekly runs live there. Computed AFTER pass 1 so a row that
    # just resolved same-day, or a nearer row that did, prunes the walk-back.
    asof_fetch = _asof_candidates(dates, entries)
    asof_fetched = _fetch_many(s3, bucket, asof_fetch)
    dirty = _merge(entries, asof_fetched) or dirty
    gets += len(to_fetch) + len(asof_fetch)

    regimes, sources = _asof_join(dates, entries)
    unresolved = [d for d in dates if d not in regimes]
    n_same_day = sum(1 for d, src in sources.items() if src == d)
    logger.info(
        "regime index: %d rows, %d row dates + %d non-row days fetched; as-of join "
        "resolved %d (%d same-day, %d carried from an earlier signals.json within "
        "%dd), %d unresolved.",
        len(dates), len(to_fetch), len(asof_fetch), len(regimes), n_same_day,
        len(regimes) - n_same_day, ASOF_MAX_AGE_DAYS, len(unresolved),
    )

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
        # Informational: the entries are exact per-date probes; consumers join
        # them as-of within this many days (I12230). Not a schema change.
        index["asof_max_age_days"] = ASOF_MAX_AGE_DAYS
        index["entries"] = dict(sorted(entries.items()))
        merged = index

    return RegimeResolution(
        regimes=regimes, index=merged, gets=gets, unresolved=unresolved,
        sources=sources,
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
