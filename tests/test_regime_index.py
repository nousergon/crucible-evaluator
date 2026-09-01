"""tests/test_regime_index.py — the S3-GET GROWTH guard (alpha-engine-config-I9702).

**Why this file exists.** ``regime_weighted_alpha`` read
``signals/{date}/signals.json`` once per date in ``trades/eod_pnl.csv``,
sequentially, over the full since-inception history — 121 GETs of a ~637KB
artifact to read 121 short strings, growing by one every weekday. Nothing in
this repo measured that. The defect was therefore invisible until it surfaced
as a Lambda duration regression (max ~68s through 2026-08-20, then 209.5s on
08-21, 293.2s on 08-28, and two ReportCard Step Functions states dying at
exactly 300.000s with ``States.Timeout`` on 08-30) — i.e. as an OUTAGE rather
than as a red test.

``TestGetCountDoesNotScale`` is the load-bearing test, and it is deliberately
NOT a unit test of the index. It counts the S3 GETs performed while building
the real Tile 0 over a SMALL session history and a LARGE one, and asserts the
steady-state count does not grow between them. That assertion fails against the
pre-I9702 code by construction, and it fails again for any future change that
reintroduces a per-session read on this path, whatever shape that change takes.
"""

from __future__ import annotations

import json
import threading
import time

import boto3
import pytest
from moto import mock_aws

from grading.regime_index import (
    REGIME_INDEX_KEY,
    SCHEMA_VERSION,
    SIGNALS_KEY_TEMPLATE,
    RegimeIndexError,
    _MAX_FETCH_WORKERS,
    _MISS_REPROBE_DAYS,
    read_index,
    resolve_regimes,
    write_index,
)
from grading.tiles.portfolio_outcome import EOD_PNL_KEY, build_portfolio_outcome_tile

BUCKET = "alpha-engine-research"

_HEADER = (
    "date,portfolio_nav,daily_return_pct,spy_return_pct,daily_alpha_pct,"
    "positions_snapshot,created_at"
)


def _dates(n: int) -> list[str]:
    """``n`` distinct ISO dates, sequential across 2024-2026 so large N fits."""
    out = []
    y, m, d = 2024, 1, 1
    for _ in range(n):
        out.append(f"{y:04d}-{m:02d}-{d:02d}")
        d += 1
        if d > 28:
            d, m = 1, m + 1
            if m > 12:
                m, y = 1, y + 1
    return out


def _eod_csv(dates: list[str]) -> str:
    rows = [_HEADER]
    nav = 1_000_000.0
    for i, day in enumerate(dates):
        alpha_pct = 1.0 if i % 2 == 0 else 2.0
        nav *= 1 + alpha_pct / 100.0
        rows.append(f"{day},{nav:.2f},{alpha_pct},0.0,{alpha_pct},{{}},2026-01-01T00:00:00+00:00")
    return "\n".join(rows) + "\n"


class _CountingS3:
    """Thread-safe counting/concurrency-observing proxy around a real client.

    Counts are per key-prefix so the assertions can talk about *signals.json
    fetches* specifically rather than about total S3 traffic, which other
    components on the tile also generate. ``max_inflight`` records peak
    concurrency, which is what proves the miss path is pooled rather than
    sequential.
    """

    def __init__(self, inner, *, delay: float = 0.0):
        self._inner = inner
        self._lock = threading.Lock()
        self._delay = delay
        self.gets: dict[str, int] = {}
        self.inflight = 0
        self.max_inflight = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _bump(self, key: str) -> None:
        prefix = key.split("/", 1)[0]
        with self._lock:
            self.gets[prefix] = self.gets.get(prefix, 0) + 1

    def get_object(self, **kwargs):
        key = kwargs.get("Key", "")
        self._bump(key)
        with self._lock:
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            if self._delay and key.startswith("signals/"):
                time.sleep(self._delay)
            return self._inner.get_object(**kwargs)
        finally:
            with self._lock:
                self.inflight -= 1

    @property
    def signals_gets(self) -> int:
        return self.gets.get("signals", 0)

    def reset(self) -> None:
        with self._lock:
            self.gets = {}
            self.inflight = 0
            self.max_inflight = 0


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _seed(s3, n: int, *, untagged: int = 0) -> list[str]:
    """Write an ``n``-session eod_pnl.csv plus a signals.json per date.

    The last ``untagged`` dates deliberately get NO signals.json, so the
    permanently-unresolvable path is exercised too.
    """
    dates = _dates(n)
    s3.put_object(Bucket=BUCKET, Key=EOD_PNL_KEY, Body=_eod_csv(dates).encode("utf-8"))
    tagged = dates if untagged == 0 else dates[:-untagged]
    for i, day in enumerate(tagged):
        s3.put_object(
            Bucket=BUCKET,
            Key=SIGNALS_KEY_TEMPLATE.format(date=day),
            # Padding stands in for the ~637KB live artifact: the point of the
            # index is that this body is never fetched to read one short string.
            Body=json.dumps(
                {"market_regime": "bull" if i % 2 == 0 else "bear", "_pad": "x" * 4096}
            ).encode("utf-8"),
        )
    return dates


def _build_and_persist(counting, *, persist: bool = True):
    """Build Tile 0 once and persist the index the way grading/handler.py does."""
    tile = build_portfolio_outcome_tile(BUCKET, s3_client=counting)
    index = tile.pop("_regime_index", None)
    if persist and index is not None:
        write_index(BUCKET, index, s3_client=counting)
    return tile, index


class TestGetCountDoesNotScale:
    """The regression guard. Fails against the pre-I9702 sequential reader."""

    @pytest.mark.parametrize("n_sessions", [40, 400])
    def test_cold_build_reads_every_date_exactly_once(self, s3, n_sessions):
        """Sanity on the instrument: a COLD index must cost one GET per date.

        Without this, a broken counter could make the steady-state assertion
        below pass vacuously — a guard that measures nothing looks identical to
        a guard that measures a healthy system.
        """
        _seed(s3, n_sessions)
        counting = _CountingS3(s3)
        _build_and_persist(counting)
        assert counting.signals_gets == n_sessions

    def test_steady_state_get_count_is_constant_in_session_count(self, s3):
        """THE assertion: warm-index GETs must not grow with history length.

        Builds the tile twice over a 40-session book and twice over a
        400-session book. The second build of each — the steady state every
        weekly run after the first is in — must perform the SAME small,
        bounded number of signals.json GETs. Ten times the history, identical
        cost. Under the pre-I9702 reader the two numbers were 40 and 400.
        """
        counts = {}
        for n_sessions in (40, 400):
            with mock_aws():
                client = boto3.client("s3", region_name="us-east-1")
                client.create_bucket(Bucket=BUCKET)
                _seed(client, n_sessions)
                counting = _CountingS3(client)
                _build_and_persist(counting)          # cold: backfills the index
                counting.reset()
                _build_and_persist(counting)          # warm: the steady state
                counts[n_sessions] = counting.signals_gets

        assert counts[40] == counts[400], (
            f"signals.json GET count scales with session history: "
            f"{counts[40]} at 40 sessions vs {counts[400]} at 400. The regime "
            f"join must be O(1) in the number of sessions (I9702)."
        )
        assert counts[400] <= _MISS_REPROBE_DAYS, (
            f"steady-state GET count {counts[400]} exceeds the bounded re-probe "
            f"window ({_MISS_REPROBE_DAYS}); the index is not converging."
        )

    def test_permanently_untagged_dates_are_not_refetched_forever(self, s3):
        """Negative entries are persisted, not re-derived every cycle.

        Without tombstones this is the original defect relocated: every date
        that will never have a signals.json is re-fetched on every build, and
        that set grows monotonically.
        """
        _seed(s3, 200, untagged=100)
        counting = _CountingS3(s3)
        _build_and_persist(counting)
        assert counting.signals_gets == 200
        counting.reset()
        _build_and_persist(counting)
        assert counting.signals_gets <= _MISS_REPROBE_DAYS

    def test_only_the_newly_appended_dates_are_fetched(self, s3):
        """One more weekday of book costs one more GET, not N more."""
        _seed(s3, 100)
        counting = _CountingS3(s3)
        _build_and_persist(counting)

        dates = _dates(105)
        s3.put_object(Bucket=BUCKET, Key=EOD_PNL_KEY, Body=_eod_csv(dates).encode("utf-8"))
        for i, day in enumerate(dates[100:], start=100):
            s3.put_object(
                Bucket=BUCKET,
                Key=SIGNALS_KEY_TEMPLATE.format(date=day),
                Body=json.dumps({"market_regime": "bull" if i % 2 == 0 else "bear"}).encode(),
            )
        counting.reset()
        _build_and_persist(counting)
        # The 5 new dates, plus at most the bounded null re-probe window.
        assert counting.signals_gets <= 5 + _MISS_REPROBE_DAYS
        assert counting.signals_gets >= 5


class TestMissPathIsBoundedConcurrent:
    """The miss path is pooled, not sequential, and the pool is bounded."""

    def test_fetches_overlap_and_never_exceed_the_worker_bound(self, s3):
        dates = _seed(s3, 60)
        # A per-GET delay makes overlap observable: with a sequential reader the
        # peak in-flight count is 1 no matter how long each call takes.
        counting = _CountingS3(s3, delay=0.02)
        resolution = resolve_regimes(BUCKET, dates, s3_client=counting)

        assert len(resolution.regimes) == 60
        assert counting.max_inflight > 1, (
            "the regime miss path ran sequentially — peak in-flight GETs was 1"
        )
        assert counting.max_inflight <= _MAX_FETCH_WORKERS, (
            f"unbounded concurrency: peak in-flight GETs {counting.max_inflight} "
            f"exceeds the declared bound {_MAX_FETCH_WORKERS}"
        )

    def test_pool_is_not_wider_than_the_work(self, s3):
        dates = _seed(s3, 3)
        counting = _CountingS3(s3, delay=0.02)
        resolve_regimes(BUCKET, dates, s3_client=counting)
        assert counting.max_inflight <= 3

    def test_a_real_s3_fault_on_one_date_still_raises(self, s3):
        """Fail loud: a throttle is not "this date has no regime"."""
        dates = _seed(s3, 20)

        class _Faulty(_CountingS3):
            def get_object(self, **kwargs):
                if kwargs.get("Key", "").startswith("signals/"):
                    from botocore.exceptions import ClientError
                    raise ClientError(
                        {"Error": {"Code": "SlowDown", "Message": "throttled"}}, "GetObject",
                    )
                return super().get_object(**kwargs)

        with pytest.raises(Exception) as exc:
            resolve_regimes(BUCKET, dates, s3_client=_Faulty(s3))
        assert "SlowDown" in str(exc.value) or "GetObject" in str(exc.value)


class TestIndexContract:
    """Schema, genesis, re-probe and the fail-loud posture on a bad index."""

    def test_absent_index_is_genesis_not_an_error(self, s3):
        index = read_index(BUCKET, s3_client=s3)
        assert index["schema_version"] == SCHEMA_VERSION
        assert index["entries"] == {}

    def test_written_index_shape(self, s3):
        dates = _seed(s3, 12, untagged=2)
        resolution = resolve_regimes(BUCKET, dates, s3_client=s3)
        idx = resolution.index
        assert idx is not None
        assert idx["schema_version"] == SCHEMA_VERSION
        assert idx["artifact"] == "market_regime_index"
        assert idx["source_key_template"] == SIGNALS_KEY_TEMPLATE
        assert idx["field"] == "market_regime"
        assert idx["updated_at"]
        assert idx["n_entries"] == 12
        assert idx["n_resolved"] == 10
        assert idx["n_unresolved"] == 2
        # Unresolvable dates are recorded as explicit nulls, not omitted.
        assert idx["entries"][dates[-1]] is None
        assert set(resolution.unresolved) == set(dates[-2:])

    def test_no_op_cycle_produces_no_index_to_write(self, s3):
        dates = _seed(s3, 30)
        first = resolve_regimes(BUCKET, dates, s3_client=s3)
        write_index(BUCKET, first.index, s3_client=s3)
        second = resolve_regimes(BUCKET, dates, s3_client=s3)
        assert second.index is None, "a converged cycle must not re-PUT the index"
        assert second.regimes == first.regimes

    def test_recent_miss_is_reprobed_when_the_producer_writes_late(self, s3):
        dates = _seed(s3, 30, untagged=1)
        late = dates[-1]
        first = resolve_regimes(BUCKET, dates, s3_client=s3)
        write_index(BUCKET, first.index, s3_client=s3)
        assert late not in first.regimes

        s3.put_object(
            Bucket=BUCKET,
            Key=SIGNALS_KEY_TEMPLATE.format(date=late),
            Body=json.dumps({"market_regime": "caution"}).encode(),
        )
        second = resolve_regimes(BUCKET, dates, s3_client=s3)
        assert second.regimes[late] == "caution"
        assert second.index is not None

    @pytest.mark.parametrize(
        "body",
        [b"{not json", b'["a", "list"]', b'{"schema_version": 99, "entries": {}}',
         b'{"schema_version": 1, "entries": "nope"}'],
    )
    def test_unreadable_index_raises_rather_than_silently_rebuilding(self, s3, body):
        s3.put_object(Bucket=BUCKET, Key=REGIME_INDEX_KEY, Body=body)
        with pytest.raises(RegimeIndexError):
            read_index(BUCKET, s3_client=s3)

    def test_index_read_fault_is_not_read_as_an_empty_index(self, s3):
        from botocore.exceptions import ClientError

        class _Denied:
            def get_object(self, **kwargs):
                raise ClientError(
                    {"Error": {"Code": "AccessDenied", "Message": "no"}}, "GetObject",
                )

        with pytest.raises(ClientError):
            read_index(BUCKET, s3_client=_Denied())
