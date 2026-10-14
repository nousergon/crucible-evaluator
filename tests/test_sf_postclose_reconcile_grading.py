"""The substrate tile grades ``ne-postclose-reconcile-pipeline`` (alpha-engine-config-I12279).

nousergon-data-PR1996 (2026-10-01) split the post-close Step Function:
``WaitForCollectionManifests``, ``EODReconcile`` and ``StopTradingInstance``
moved, names unchanged, to ``ne-postclose-reconcile-pipeline``, and
nousergon-lib (>= v0.124.171) cut ``ne-postclose-trading-pipeline``'s spine to
``("CaptureSnapshot",)``. Until ``_SF_NAMES`` named the new machine, the
reconcile, the collector-readiness wait and the box stop were graded nowhere.

These tests run the real ``classify_work`` / ``build_cycle_shape`` against
entered-state walks of the reconcile machine's own state names (from
nousergon-data ``infrastructure/step_function_eod_reconcile.json``); only the
AWS reads are replaced.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import nousergon_lib.pipeline_status as ps
from grading.tiles.substrate import _SCHEDULED_ROLES, _SF_NAMES, _discover_sf_arns, _sf_success_rate

UTC = timezone.utc
RECONCILE = "ne-postclose-reconcile-pipeline"
TRADING = "ne-postclose-trading-pipeline"
RECONCILE_STAGES = ("WaitForCollectionManifests", "EODReconcile", "StopTradingInstance")
AS_OF = datetime(2026, 10, 10, 12, tzinfo=UTC)


def _arn(kind: str, name: str, suffix: str = "") -> str:
    return f"arn:aws:states:us-east-1:1:{kind}:{name}{suffix}"


#: A normal reconcile run, in entry order, as the definition walks it.
_RECONCILE_WALK = (
    "MarketHoursGate", "MarketHoursGateChoice", "StampStartClockUtc", "CheckMutexRole",
    "AcquireMutex", "InitCollectionReadinessPoll", "SeedCollectionReadiness",
    "WaitForCollectionManifests", "CheckCollectionReadiness", "ProbeEODReconcilePrecondition",
    "CheckSkipEODReconcile", "EODReconcile", "WaitForEOD", "CheckEODStatus",
    "CheckBoxOwnedBeforeStop", "StopTradingInstance", "ReadExerciseCadence",
    "CheckExerciseCadence", "CheckDegradedOutcome", "WriteCompletionMarkerNormal",
    "NormalSucceeded",
)


def _upto(stage: str) -> tuple[str, ...]:
    """The walk of a run that died inside ``stage``."""
    return _RECONCILE_WALK[: _RECONCILE_WALK.index(stage) + 1] + ("HandleFailure", "FailExecution")


_n = iter(range(10_000))


def _exec(sm, states, *, status="SUCCEEDED", role="eod", run_date="2026-10-07",
          start=datetime(2026, 10, 7, 21, 30, tzinfo=UTC)):
    arn = _arn("execution", sm, f":e{next(_n)}")
    outcome = ps.classify_work(
        state_machine_name=sm, status=ps.RunStatus(status), entered_states=list(states),
        execution_arn=arn,
    )
    run = SimpleNamespace(
        name=arn.rsplit(":", 1)[-1], status=status, start_utc=start, pipeline_role=role,
        end_utc=start + timedelta(hours=1), execution_arn=arn, run_date=run_date,
    )
    return run, (outcome, tuple(states), {})


def _rate(monkeypatch, execs_by_sm):
    runs = {_arn("stateMachine", sm): [r for r, _ in ex] for sm, ex in execs_by_sm.items()}
    reads = {r.execution_arn: read for ex in execs_by_sm.values() for r, read in ex}
    monkeypatch.setattr(
        "nousergon_lib.pipeline_status.list_recent_pipeline_runs",
        lambda arn, **kw: runs[arn],
    )
    monkeypatch.setattr("grading.tiles.substrate._read_execution", lambda arn, sfn: reads[arn])
    monkeypatch.setattr("grading.tiles.substrate._discover_sf_arns", lambda sfn: list(runs))
    return _sf_success_rate(object(), AS_OF, 28)


# --- scope -------------------------------------------------------------------

def test_the_reconcile_machine_is_in_scope():
    assert RECONCILE in _SF_NAMES
    assert TRADING in _SF_NAMES


def test_the_lib_declares_the_reconcile_spine_this_suite_assumes():
    """If the lib moves these stages again, the walks below test nothing —
    pin the declared spine first. Requires nousergon-lib >= v0.124.171."""
    assert ps.stage_order_for(RECONCILE) == RECONCILE_STAGES
    assert not set(RECONCILE_STAGES) & set(ps.stage_order_for(TRADING))


def test_its_scheduled_role_counts_as_unattended():
    """Started by alpha-engine-eod-backstop with role ``eod``."""
    assert "eod" in _SCHEDULED_ROLES
    assert "eod" in ps.CADENCE_ROLES


def test_discovery_resolves_the_reconcile_machine_by_name(monkeypatch):
    monkeypatch.delenv("EVALUATOR_SF_ARNS", raising=False)
    machines = [{"name": n, "stateMachineArn": _arn("stateMachine", n)}
                for n in (*_SF_NAMES, "ne-some-other-pipeline")]

    class _Paginator:
        def paginate(self):
            return [{"stateMachines": machines}]

    sfn = SimpleNamespace(get_paginator=lambda op: _Paginator())
    arns = _discover_sf_arns(sfn)
    assert _arn("stateMachine", RECONCILE) in arns
    assert _arn("stateMachine", "ne-some-other-pipeline") not in arns
    assert len(arns) == len(_SF_NAMES)


# --- grading -----------------------------------------------------------------

def test_a_full_reconcile_walk_is_a_clean_first_pass_cycle(monkeypatch):
    sf = _rate(monkeypatch, {RECONCILE: [_exec(RECONCILE, _RECONCILE_WALK)]})
    assert sf["n_cycles"] == 1 and sf["n_cycles_clean"] == 1
    assert sf["n_cycles_clean_full_scope"] == 1
    assert sf["per_sf"] == {RECONCILE: "1/1"}
    assert sf["per_sf_unattended"] == {RECONCILE: "1/1"}
    assert sf["n_cycles_clean_first_pass"] == 1
    assert "[clean]" in sf["scope_detail"][f"{RECONCILE}:2026-10-07"]


def test_a_reconcile_run_that_died_in_eodreconcile_is_not_clean(monkeypatch):
    sf = _rate(monkeypatch, {RECONCILE: [_exec(RECONCILE, _upto("EODReconcile"), status="FAILED")]})
    assert sf["n_cycles"] == 1 and sf["n_cycles_clean"] == 0
    assert sf["per_sf_unattended"] == {RECONCILE: "0/1"}
    assert "[not_clean]" in sf["scope_detail"][f"{RECONCILE}:2026-10-07"]


def test_a_reconcile_run_that_never_stopped_the_box_is_not_clean(monkeypatch):
    """SUCCEEDED but short of StopTradingInstance: the work was not done."""
    walk = tuple(s for s in _RECONCILE_WALK if s != "StopTradingInstance")
    sf = _rate(monkeypatch, {RECONCILE: [_exec(RECONCILE, walk)]})
    assert sf["n_cycles"] == 1 and sf["n_cycles_clean"] == 0


def test_both_postclose_machines_grade_as_separate_cycles_on_one_day(monkeypatch):
    """The split day's two halves: a clean snapshot and a failed reconcile are
    two cycles, and the reconcile failure is no longer invisible."""
    trading = _exec(TRADING, ("CaptureSnapshot", "WaitForCaptureSnapshot", "CheckSnapshotStatus"))
    reconcile = _exec(RECONCILE, _upto("WaitForCollectionManifests"), status="FAILED")
    sf = _rate(monkeypatch, {TRADING: [trading], RECONCILE: [reconcile]})
    assert sf["n_cycles"] == 2 and sf["n_cycles_clean"] == 1
    assert sf["per_sf"] == {TRADING: "1/1", RECONCILE: "0/1"}
    assert f"{TRADING}:2026-10-07" in sf["scope_detail"]
    assert f"{RECONCILE}:2026-10-07" in sf["scope_detail"]
