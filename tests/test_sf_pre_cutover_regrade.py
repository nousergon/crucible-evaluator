"""Pre-cutover cycles in the trailing window are graded against the
definition they ran (alpha-engine-config-I12271).

The I11269 decoupled data cutover (nousergon-data-PR1930, 2026-09-28)
replaced the v1 data legs with ``WaitForCollectionManifests`` on the weekly
and preopen spines. Measured on this fold 2026-10-05, nousergon-lib .158-.171
graded every PRE-cutover cycle as short that stage (preopen 2/19), which is
why the pin sat on .157. The lib's ``SUPERSEDED_STAGES`` excuses the
successor ONLY for a walk that entered one of its predecessors — evidence,
never absence — so these walks pin both halves end to end through
``_sf_success_rate``. Also pinned: the post-split post-close spine
(nousergon-data-PR1996), which .157 graded short EODReconcile /
StopTradingInstance.

Remove the pre-cutover cases together with the lib's map (Re-exam:
2026-11-02, once no pre-cutover cycle can sit in the 28-day window).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import nousergon_lib.pipeline_status as ps
from grading.tiles.substrate import _sf_success_rate

UTC = timezone.utc
WEEKLY = "ne-weekly-freshness-pipeline"
PREOPEN = "ne-preopen-trading-pipeline"
POSTCLOSE = "ne-postclose-trading-pipeline"
SUCCESSOR = "WaitForCollectionManifests"
AS_OF = datetime(2026, 10, 10, 12, tzinfo=UTC)
PRE_CUTOVER = datetime(2026, 9, 25, 11, tzinfo=UTC)  # a Friday before 09-28

#: The v1 legs each pre-cutover definition ran where the successor now sits.
PREDECESSORS = {
    WEEKLY: ("MorningEnrich", "DataPhase1"),
    PREOPEN: ("LaunchMorningEnrichSpot", "LaunchMorningArcticAppendSpot"),
}
ROLE = {WEEKLY: "weekly", PREOPEN: "daily", POSTCLOSE: "eod"}


def _walk(sm, *, predecessors=None, successor=False):
    """Every spine stage in order; at the successor's slot, the predecessors
    (pre-cutover) or the successor itself (post-cutover)."""
    preds = PREDECESSORS.get(sm, ()) if predecessors is None else predecessors
    states = ["InitializeInput"]
    for stage in ps.stage_order_for(sm):
        if stage == SUCCESSOR:
            states.extend([SUCCESSOR] if successor else preds)
            continue
        states.append(stage)
    states.append("WriteCompletionMarker")
    return states


_n = iter(range(10_000))


def _exec(sm, states, *, start=PRE_CUTOVER):
    arn = f"arn:aws:states:us-east-1:1:execution:{sm}:e{next(_n)}"
    outcome = ps.classify_work(
        state_machine_name=sm, status=ps.RunStatus("SUCCEEDED"), entered_states=states,
        execution_arn=arn,
    )
    run = SimpleNamespace(
        name=arn.rsplit(":", 1)[-1], status="SUCCEEDED", start_utc=start,
        pipeline_role=ROLE[sm], end_utc=start + timedelta(hours=1), execution_arn=arn,
        run_date=None,
    )
    return sm, run, (outcome, tuple(states), {})


def _rate(monkeypatch, execs):
    by_sm: dict[str, list] = {}
    reads = {}
    for sm, run, read in execs:
        by_sm.setdefault(sm, []).append(run)
        reads[run.execution_arn] = read
    arns = [f"arn:aws:states:us-east-1:1:stateMachine:{sm}" for sm in by_sm]
    monkeypatch.setattr(
        "nousergon_lib.pipeline_status.list_recent_pipeline_runs",
        lambda arn, **kw: by_sm[arn.rsplit(":", 1)[-1]],
    )
    monkeypatch.setattr("grading.tiles.substrate._read_execution", lambda arn, sfn: reads[arn])
    monkeypatch.setattr("grading.tiles.substrate._discover_sf_arns", lambda sfn: arns)
    return _sf_success_rate(object(), AS_OF, 28)


def test_the_lib_declares_the_cutover_lineage_this_suite_assumes():
    """If the lib drops or renames the map, every case below would test
    nothing — pin it first."""
    for sm, preds in PREDECESSORS.items():
        assert ps.SUPERSEDED_STAGES[sm] == {SUCCESSOR: preds}
        assert SUCCESSOR in ps.stage_order_for(sm)


@pytest.mark.parametrize("sm", [WEEKLY, PREOPEN])
def test_a_pre_cutover_cycle_that_ran_the_v1_legs_is_clean(monkeypatch, sm):
    sf = _rate(monkeypatch, [_exec(sm, _walk(sm))])
    assert sf["n_cycles"] == 1
    assert sf["n_cycles_clean"] == 1 and sf["n_cycles_clean_full_scope"] == 1


@pytest.mark.parametrize("sm", [WEEKLY, PREOPEN])
def test_a_cycle_with_neither_the_v1_legs_nor_the_wait_is_still_short(monkeypatch, sm):
    """Absence is never an excuse: nothing in this walk says it ran the old
    definition, so it is short the stage the current one declares."""
    sf = _rate(monkeypatch, [_exec(sm, _walk(sm, predecessors=()))])
    assert sf["n_cycles"] == 1
    assert sf["n_cycles_clean"] == 0 and sf["n_cycles_clean_full_scope"] == 0
    key = next(iter(k for k in sf["scope_detail"] if k.startswith(sm)))
    assert f"missing {SUCCESSOR}" in sf["scope_detail"][key]


@pytest.mark.parametrize("sm", [WEEKLY, PREOPEN])
def test_a_post_cutover_cycle_that_entered_the_wait_is_clean(monkeypatch, sm):
    start = datetime(2026, 10, 9, 11, tzinfo=UTC)
    sf = _rate(monkeypatch, [_exec(sm, _walk(sm, successor=True), start=start)])
    assert sf["n_cycles_clean"] == 1 and sf["n_cycles_clean_full_scope"] == 1


def test_a_window_spanning_the_cutover_grades_every_preopen_cycle_clean(monkeypatch):
    """The 2/19 shape, end to end: pre- and post-cutover preopen cycles in
    one 28-day window, every one of them clean."""
    pre = [datetime(2026, 9, d, 11, tzinfo=UTC) for d in (22, 23, 24, 25, 26)]
    post = [datetime(2026, 10, d, 11, tzinfo=UTC) for d in (5, 6, 7, 8, 9)]
    execs = [_exec(PREOPEN, _walk(PREOPEN), start=d) for d in pre]
    execs += [_exec(PREOPEN, _walk(PREOPEN, successor=True), start=d) for d in post]
    sf = _rate(monkeypatch, execs)
    assert sf["n_cycles"] == 10 and sf["n_cycles_clean"] == 10
    assert sf["cycle_rate"] == pytest.approx(1.0)


def test_a_post_split_postclose_cycle_is_clean(monkeypatch):
    """nousergon-data-PR1996 moved WaitForCollectionManifests / EODReconcile /
    StopTradingInstance to ne-postclose-reconcile-pipeline; the post-close
    spine is what is left, and a run of it is a full cycle."""
    states = ["InitializeInput", *ps.stage_order_for(POSTCLOSE), "WriteCompletionMarker"]
    assert "EODReconcile" not in states
    start = datetime(2026, 10, 8, 21, tzinfo=UTC)
    sf = _rate(monkeypatch, [_exec(POSTCLOSE, states, start=start)])
    assert sf["n_cycles_clean"] == 1 and sf["n_cycles_clean_full_scope"] == 1
