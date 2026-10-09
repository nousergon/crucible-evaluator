"""SF cycle grading is point-in-time at ``as_of`` (alpha-engine-config-I12059).

Measured 2026-10-06 on crucible-evaluator ba013fe7: a read-only
``_sf_success_rate(sfn, 2026-10-04T20:00Z, 28)`` graded
``ne-postclose-trading-pipeline:2026-10-05`` — an execution that had not
started on the day the calculation describes. The Step Functions listing is
read NOW, so every current execution and every current terminal status leaked
into historical windows.

The rule these tests pin: eligible evidence lies in ``[as_of - window, as_of]``
by start AND stop. A later start is excluded and counted; a later stop was in
flight on the day and is excluded and named; a terminal execution with no stop
instant makes its cycle unevaluable, and the card renders N/A — never GREEN.
Scope semantics (declared skips, first-pass vs recovered, trading-day keys,
truncation) are unchanged. These run the real ``classify_work`` /
``build_cycle_shape``; only the AWS read is replaced.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import nousergon_lib.pipeline_status as ps
from grading.tiles.substrate import _sf_success_rate, build_substrate_tile

UTC = timezone.utc
WEEKLY = "ne-weekly-freshness-pipeline"
POSTCLOSE = "ne-postclose-trading-pipeline"
PARITY = ("ParityParallel", "PitParityCompare")
WITNESS = "MarkParityVerdictUnknownByCadence"
AS_OF = datetime(2026, 10, 4, 20, tzinfo=UTC)  # the I12059 measurement's cutoff
WINDOW = 28
CUTOFF = AS_OF - timedelta(days=WINDOW)

_n = iter(range(10_000))
_NO_END = object()


def _walk(sm, *, upto=None, terminal="WriteCompletionMarker", witness=False, without=()):
    spine = ps.stage_order_for(sm)
    states = ["InitializeInput"]
    for stage in spine:
        if stage == "ParityParallel" and witness:
            states.append(WITNESS)
        if stage in without:
            continue
        states.append(stage)
        if stage == upto:
            break
    states.append(terminal)
    return states


def _exec(sm, states, *, start, end=None, status="SUCCEEDED", role="weekly", run_date=None):
    """One listed execution plus its history read. ``end`` defaults to an hour
    after ``start``; pass ``_NO_END`` for a summary with no stop instant."""
    arn = f"arn:aws:states:us-east-1:1:execution:{sm}:e{next(_n)}"
    outcome = ps.classify_work(
        state_machine_name=sm, status=ps.RunStatus(status), entered_states=states,
        execution_arn=arn,
    )
    end_utc = None if end is _NO_END else (end or start + timedelta(hours=1))
    run = SimpleNamespace(
        name=arn.rsplit(":", 1)[-1], status=status, start_utc=start, end_utc=end_utc,
        pipeline_role=role, execution_arn=arn, run_date=run_date,
    )
    return sm, run, (outcome, tuple(states), {})


def _wire(monkeypatch, execs):
    by_sm: dict[str, list] = {}
    reads = {}
    for sm, run, read in execs:
        by_sm.setdefault(sm, []).append(run)
        reads[run.execution_arn] = read
    sms = sorted(by_sm) or [WEEKLY]
    monkeypatch.setattr(
        "nousergon_lib.pipeline_status.list_recent_pipeline_runs",
        lambda arn, **kw: list(by_sm.get(arn.rsplit(":", 1)[-1], [])),
    )
    monkeypatch.setattr("grading.tiles.substrate._read_execution", lambda arn, sfn: reads[arn])
    monkeypatch.setattr(
        "grading.tiles.substrate._discover_sf_arns",
        lambda sfn: [f"arn:aws:states:us-east-1:1:stateMachine:{sm}" for sm in sms],
    )


def _rate(monkeypatch, execs, as_of=AS_OF):
    _wire(monkeypatch, execs)
    return _sf_success_rate(object(), as_of, WINDOW)


def _full(sm=WEEKLY, **kw):
    return _exec(sm, _walk(sm), **kw)


# --- the measured leak ------------------------------------------------------

def test_the_measured_leak_a_10_05_postclose_run_is_not_graded_at_10_04(monkeypatch):
    """The exact I12059 reproduction: a 56.9s SUCCEEDED postclose run on
    2026-10-05 must not appear in the 2026-10-04 20:00Z calculation."""
    start = datetime(2026, 10, 5, 20, 30, tzinfo=UTC)
    leak = _full(POSTCLOSE, start=start, end=start + timedelta(seconds=56.9),
                 role="eod", run_date="2026-10-05")
    sf = _rate(monkeypatch, [leak])
    assert sf["n_cycles"] == 0 and sf["cycle_rate"] is None
    assert not any(k.startswith(f"{POSTCLOSE}:2026-10-05") for k in sf["scope_detail"])
    assert sf["n_executions_after_as_of"] == {POSTCLOSE: 1}
    assert sf["as_of"] == AS_OF.isoformat() and sf["window_start"] == CUTOFF.isoformat()


# --- the interval -----------------------------------------------------------

def test_an_execution_completed_inside_the_window_is_graded(monkeypatch):
    sf = _rate(monkeypatch, [_full(start=datetime(2026, 10, 3, 9, tzinfo=UTC))])
    assert sf["n_cycles"] == sf["n_cycles_clean_full_scope"] == 1
    assert sf["n_executions_after_as_of"] == {} and sf["in_flight_at_as_of"] == []
    assert sf["temporally_unevaluable"] == []


def test_an_execution_starting_after_as_of_is_excluded_and_counted(monkeypatch):
    inside = _full(start=datetime(2026, 9, 26, 9, tzinfo=UTC))
    later = _exec(WEEKLY, _walk(WEEKLY, upto="Backtester", terminal="FailExecution"),
                  status="FAILED", start=AS_OF + timedelta(days=6))
    sf = _rate(monkeypatch, [inside, later])
    # Only the 09-26 cycle exists at as_of; the 10-10 failure is not yet evidence.
    assert sf["n_cycles"] == 1 and sf["cycle_rate"] == 1.0
    assert sf["n_executions_after_as_of"] == {WEEKLY: 1}


def test_starting_before_but_completing_after_as_of_was_in_flight_on_the_day(monkeypatch):
    """Its current terminal status is not evidence about as_of, in either
    direction — a later SUCCEEDED is not credited, a later FAILED not charged."""
    ok_late = _full(start=AS_OF - timedelta(hours=1), end=AS_OF + timedelta(hours=1))
    bad_late = _exec(WEEKLY, _walk(WEEKLY, upto="Backtester", terminal="FailExecution"),
                     status="FAILED", start=AS_OF - timedelta(days=7, hours=1),
                     end=AS_OF + timedelta(seconds=1))
    sf = _rate(monkeypatch, [ok_late, bad_late])
    assert sf["n_cycles"] == 0 and sf["cycle_rate"] is None
    assert sorted(sf["in_flight_at_as_of"]) == sorted(
        f"{WEEKLY}:{e[1].name}" for e in (ok_late, bad_late)
    )


def test_exact_bounds_are_inclusive_at_both_ends(monkeypatch):
    """[as_of - window, as_of]: a start ON the window's opening and a stop ON
    as_of are both eligible; a second either side is not."""
    at_open = _full(start=CUTOFF, run_date="2026-09-05")
    stops_at_as_of = _full(start=AS_OF - timedelta(hours=2), end=AS_OF, run_date="2026-10-02")
    starts_at_as_of = _full(start=AS_OF, end=AS_OF, run_date="2026-09-25")
    before_open = _full(start=CUTOFF - timedelta(seconds=1), run_date="2026-08-29")
    after_close = _full(start=AS_OF + timedelta(seconds=1), run_date="2026-09-18")
    stops_late = _full(start=AS_OF - timedelta(hours=2), end=AS_OF + timedelta(seconds=1),
                       run_date="2026-09-11")
    sf = _rate(monkeypatch, [at_open, stops_at_as_of, starts_at_as_of,
                             before_open, after_close, stops_late])
    assert set(sf["scope_detail"]) == {
        f"{WEEKLY}:2026-09-04",  # 09-05 is a Saturday → its trading day, the Friday
        f"{WEEKLY}:2026-10-02",
        f"{WEEKLY}:2026-09-25",
    }
    assert sf["n_cycles"] == 3 and sf["n_cycles_clean"] == 3
    assert sf["n_executions_after_as_of"] == {WEEKLY: 1}
    assert sf["in_flight_at_as_of"] == [f"{WEEKLY}:{stops_late[1].name}"]


# --- recovery after the cutoff ----------------------------------------------

def _scheduled_failure():
    return _exec(WEEKLY, _walk(WEEKLY, witness=True, upto="Backtester", terminal="FailExecution"),
                 status="FAILED", start=datetime(2026, 10, 3, 9, tzinfo=UTC))


def _recovery(start, end=None):
    spine = ps.stage_order_for(WEEKLY)
    states = [WITNESS, *(s for s in spine[spine.index("Backtester"):] if s not in PARITY),
              "WriteCompletionMarker"]
    return _exec(WEEKLY, states, role="watch-rerun", run_date="2026-10-02", start=start, end=end)


@pytest.mark.parametrize("recovery_start,recovery_end", [
    (AS_OF + timedelta(hours=3), None),                       # started after the cutoff
    (AS_OF - timedelta(hours=1), AS_OF + timedelta(hours=2)),  # started before, finished after
])
def test_a_recovery_after_the_cutoff_does_not_repair_the_earlier_cycle(
    monkeypatch, recovery_start, recovery_end,
):
    """At as_of the cycle had one scheduled failure and no completed recovery.
    Graded later the same evidence is a recovered clean cycle — graded at
    as_of it must stay not-clean, and never unattended."""
    evidence = [_scheduled_failure(), _recovery(recovery_start, recovery_end)]
    then = _rate(monkeypatch, evidence)
    assert then["n_cycles"] == 1
    assert then["n_cycles_clean"] == 0 and then["n_cycles_clean_recovered"] == 0
    assert then["n_unattended"] == 1 and then["n_unattended_ok"] == 0
    assert "[not_clean]" in then["scope_detail"][f"{WEEKLY}:2026-10-02"]

    later = _rate(monkeypatch, evidence, as_of=AS_OF + timedelta(days=2))
    assert later["n_cycles_clean"] == 1 and later["n_cycles_clean_recovered"] == 1
    assert later["n_cycles_clean_first_pass"] == 0


# --- an absent end timestamp ------------------------------------------------

def test_a_terminal_execution_with_no_end_timestamp_makes_its_cycle_unevaluable(monkeypatch):
    clean = _full(start=datetime(2026, 9, 26, 9, tzinfo=UTC))
    sched = _scheduled_failure()
    no_end = _recovery(AS_OF - timedelta(hours=5), end=_NO_END)
    sf = _rate(monkeypatch, [clean, sched, no_end])
    assert sf["temporally_unevaluable"] == [f"{WEEKLY}:{no_end[1].name}"]
    # The whole cycle is withheld — folding the scheduled failure alone would
    # grade a guess, and so would crediting the recovery.
    assert sf["n_cycles"] == 1 and sf["n_unattended"] == 1
    assert sf["scope_detail"][f"{WEEKLY}:2026-10-02"].startswith("[unevaluable]")


def test_the_card_renders_an_unevaluable_window_na_never_green(monkeypatch):
    execs = [_full(start=datetime(2026, 9, d, 9, tzinfo=UTC)) for d in (12, 19, 26)]
    execs.append(_full(start=datetime(2026, 10, 3, 9, tzinfo=UTC), end=_NO_END))
    _wire(monkeypatch, execs)
    import boto3
    from moto import mock_aws

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        tile = build_substrate_tile("test-bucket", s3_client=s3, as_of=AS_OF, sfn_client=object())
    for name in ("sf_success_rate_4w", "unattended_first_pass_rate"):
        comp = next(c for c in tile["components"] if c["name"] == name)
        assert comp["status"].startswith("N/A"), comp["status"]
        assert comp.get("value") is None
        assert "I12059" in comp["status_reason"] or "unevaluable" in comp["status_reason"]


def test_the_card_states_the_interval_it_graded(monkeypatch):
    execs = [_full(start=datetime(2026, 9, d, 9, tzinfo=UTC)) for d in (12, 19, 26)]
    execs.append(_full(start=AS_OF + timedelta(days=1)))
    _wire(monkeypatch, execs)
    import boto3
    from moto import mock_aws

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        tile = build_substrate_tile("test-bucket", s3_client=s3, as_of=AS_OF, sfn_client=object())
    sf = next(c for c in tile["components"] if c["name"] == "sf_success_rate_4w")
    assert sf["n_samples"] == 3
    pit = sf["cycle_scope"]["point_in_time"]
    assert pit == {
        "as_of": AS_OF.isoformat(), "window_start": CUTOFF.isoformat(),
        "n_executions_after_as_of": {WEEKLY: 1}, "in_flight_at_as_of": [],
    }


# --- scan truncation --------------------------------------------------------

def test_future_executions_filling_the_scan_are_detected_as_truncation(monkeypatch):
    """The listing is count-bounded and most-recent-first, so executions after
    as_of consume it. When they push the scan's oldest start inside the window,
    the requested past window was not fully read: refused, not published."""
    monkeypatch.setattr("grading.tiles.substrate._SF_EXEC_SCAN_LIMIT", 5)
    future = [_full(start=AS_OF + timedelta(days=d)) for d in range(1, 5)]
    inside = [_full(start=datetime(2026, 10, 3, 9, tzinfo=UTC))]
    sf = _rate(monkeypatch, future + inside)
    assert sf["truncated"] == [WEEKLY]
    assert sf["n_executions_after_as_of"] == {WEEKLY: 4}


def test_future_executions_do_not_cry_truncation_when_the_scan_reaches_past_the_window(monkeypatch):
    monkeypatch.setattr("grading.tiles.substrate._SF_EXEC_SCAN_LIMIT", 5)
    future = [_full(start=AS_OF + timedelta(days=d)) for d in range(1, 4)]
    older = [_full(start=datetime(2026, 10, 3, 9, tzinfo=UTC)),
             _full(start=CUTOFF - timedelta(days=1))]
    sf = _rate(monkeypatch, future + older)
    assert sf["truncated"] == []
    assert sf["n_cycles"] == 1 and sf["n_executions_after_as_of"] == {WEEKLY: 3}
