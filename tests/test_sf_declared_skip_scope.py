"""Clean-with-declared-skips vs full-scope cycle grading (alpha-engine-config-I11987).

Brian's ruling, option A (2026-10-04): stages disabled by a REGISTERED standing
operator skip do not count against a clean cycle, the full-scope result is
published beside it, first-pass and recovered counts stay separate, and an
arbitrary skip flag never excuses missing work.

The registered witness is a STATE the cycle's own walk entered —
``MarkParityVerdictUnknownByCadence`` — mapped to the stages it excuses by
``nousergon_lib.pipeline_status.CADENCE_SKIP_MARKER_STAGES``. These tests run
the real ``classify_work`` / ``build_cycle_shape`` against synthetic entered-
state walks of the live weekly spine; only the AWS read is replaced.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import nousergon_lib.pipeline_status as ps
from grading.tiles.substrate import _sf_success_rate, build_substrate_tile

UTC = timezone.utc
SM = "ne-weekly-freshness-pipeline"
SM_ARN = f"arn:aws:states:us-east-1:1:stateMachine:{SM}"
SPINE = ps.stage_order_for(SM)
PARITY = ("ParityParallel", "PitParityCompare")
WITNESS = "MarkParityVerdictUnknownByCadence"
AS_OF = datetime(2026, 10, 10, 12, tzinfo=UTC)


def test_the_registered_witness_is_the_one_this_suite_assumes():
    """If the lib renames the marker or changes what it excuses, every test
    below would silently test nothing — pin the registry entry first."""
    assert ps.CADENCE_SKIP_MARKER_STAGES[SM] == {WITNESS: PARITY}
    assert set(PARITY) <= set(SPINE)


def _walk(*, without=(), witness=False, upto=None, terminal="WriteCompletionMarker"):
    """An entered-state walk over the spine in order, minus ``without``.

    ``upto`` stops the walk after that spine stage (a run that died there).
    """
    states = ["InitializeInput"]
    for stage in SPINE:
        if stage == "ParityParallel" and witness:
            states.append(WITNESS)
        if stage in without:
            continue
        states.append(stage)
        if stage == upto:
            break
    states.append(terminal)
    return states


_n = iter(range(10_000))


def _exec(states, *, status="SUCCEEDED", role="weekly", run_date=None,
          start=datetime(2026, 10, 3, 9, tzinfo=UTC), payload=None):
    arn = f"arn:aws:states:us-east-1:1:execution:{SM}:e{next(_n)}"
    outcome = ps.classify_work(
        state_machine_name=SM, status=ps.RunStatus(status), entered_states=states,
        execution_arn=arn,
    )
    run = SimpleNamespace(
        name=arn.rsplit(":", 1)[-1], status=status, start_utc=start, pipeline_role=role,
        execution_arn=arn, run_date=run_date,
    )
    return run, (outcome, tuple(states), payload or {})


def _wire(monkeypatch, execs):
    runs = [r for r, _ in execs]
    reads = {r.execution_arn: read for r, read in execs}
    monkeypatch.setattr(
        "nousergon_lib.pipeline_status.list_recent_pipeline_runs", lambda arn, **kw: runs,
    )
    monkeypatch.setattr(
        "grading.tiles.substrate._read_execution", lambda arn, sfn: reads[arn],
    )
    monkeypatch.setattr("grading.tiles.substrate._discover_sf_arns", lambda sfn: [SM_ARN])


def _rate(monkeypatch, execs):
    _wire(monkeypatch, execs)
    return _sf_success_rate(object(), AS_OF, 28)


# --- the declared-skip case -------------------------------------------------

def test_a_parity_only_skip_with_its_witness_is_clean_with_declared_skips(monkeypatch):
    """The Done-when case on #11987: a run that did everything except the
    parity stages, and whose walk entered the registered witness."""
    sf = _rate(monkeypatch, [_exec(_walk(without=PARITY, witness=True))])
    assert sf["n_cycles"] == 1
    assert sf["n_cycles_clean"] == 1 and sf["cycle_rate"] == 1.0
    # Full scope is published beside it and is NOT clean: parity did not run.
    assert sf["n_cycles_clean_full_scope"] == 0 and sf["cycle_rate_full_scope"] == 0.0
    assert sf["n_cycles_clean_with_declared_skips"] == 1
    assert sf["declared_skips"] == {SM: {"ParityParallel": 1, "PitParityCompare": 1}}
    # A scheduled run alone, no recovery: first pass in the declared scope only.
    assert sf["n_cycles_clean_first_pass"] == 1 and sf["n_cycles_clean_recovered"] == 0
    assert sf["unattended_rate"] == 1.0 and sf["unattended_rate_full_scope"] == 0.0
    assert "[clean_with_declared_skips]" in sf["scope_detail"][f"{SM}:2026-10-02"]


def test_a_full_spine_run_is_clean_in_both_scopes(monkeypatch):
    """Full scope is a subset of the declared scope, never the other way."""
    sf = _rate(monkeypatch, [_exec(_walk())])
    assert sf["n_cycles_clean"] == sf["n_cycles_clean_full_scope"] == 1
    assert sf["n_cycles_clean_with_declared_skips"] == 0
    assert sf["declared_skips"] == {}


# --- the undeclared-skip cases ----------------------------------------------

def test_the_same_run_without_the_witness_is_not_clean_in_either_scope(monkeypatch):
    """Absence is never a skip. ``skip_parity: true`` in the input is exactly
    what the live trigger carries, and it must excuse nothing on its own."""
    sf = _rate(monkeypatch, [_exec(_walk(without=PARITY), payload={"skip_parity": True})])
    assert sf["n_cycles"] == 1
    assert sf["n_cycles_clean"] == 0 and sf["n_cycles_clean_full_scope"] == 0
    assert sf["declared_skips"] == {}
    assert sf["unattended_rate"] == 0.0


def test_an_arbitrary_skip_flag_never_excuses_a_stage_the_witness_does_not_name(monkeypatch):
    """Measured 2026-10-05: the live scheduled trigger also carries
    ``skip_rag_ingestion`` and ``skip_data_phase2``. No registered witness
    names those stages, so the cycle stays NOT clean even though the parity
    witness is present and parity is excused."""
    flags = {"skip_parity": True, "skip_rag_ingestion": True, "skip_data_phase2": True}
    walk = _walk(without=(*PARITY, "RAGIngestion", "DataPhase2"), witness=True)
    sf = _rate(monkeypatch, [_exec(walk, payload=flags)])
    assert sf["n_cycles_clean"] == 0 and sf["n_cycles_clean_full_scope"] == 0
    assert "missing RAGIngestion, DataPhase2" in sf["scope_detail"][f"{SM}:2026-10-02"]


# --- recovery ---------------------------------------------------------------

def test_recovered_on_retry_is_clean_but_counted_recovered_not_first_pass(monkeypatch):
    """Scheduled run dies inside Backtester; a same-cycle rerun carrying the
    trading-day run_date picks up from Backtester and finishes. Clean with
    declared skips, but via recovery — never unattended."""
    scheduled = _exec(
        _walk(witness=True, upto="Backtester", terminal="FailExecution"), status="FAILED",
    )
    recovery = _exec(
        [WITNESS, *(s for s in SPINE[SPINE.index("Backtester"):] if s not in PARITY),
         "WriteCompletionMarker"],
        role="watch-rerun", run_date="2026-10-02",
        start=datetime(2026, 10, 4, 19, tzinfo=UTC),
    )
    sf = _rate(monkeypatch, [scheduled, recovery])
    assert sf["n_cycles"] == 1, "the Saturday scheduled run and the Friday-keyed rerun are ONE cycle"
    assert sf["n_cycles_clean"] == 1
    assert sf["n_cycles_clean_full_scope"] == 0
    assert sf["n_cycles_clean_first_pass"] == 0 and sf["n_cycles_clean_recovered"] == 1
    assert sf["n_unattended"] == 1 and sf["n_unattended_ok"] == 0


def test_a_rerun_that_skips_the_stage_the_scheduled_run_died_in_is_not_clean(monkeypatch):
    """The failed run ENTERED Backtester but did not leave it. A recovery that
    resumes after it must not be able to borrow that entry."""
    scheduled = _exec(
        _walk(witness=True, upto="Backtester", terminal="FailExecution"), status="FAILED",
    )
    after = SPINE[SPINE.index("Backtester") + 1:]
    recovery = _exec(
        [WITNESS, *(s for s in after if s not in PARITY), "WriteCompletionMarker"],
        role="watch-rerun", run_date="2026-10-02",
        start=datetime(2026, 10, 4, 19, tzinfo=UTC),
    )
    sf = _rate(monkeypatch, [scheduled, recovery])
    assert sf["n_cycles_clean"] == 0


# --- an actual FAILED execution never becomes clean -------------------------

def test_a_failed_run_does_not_become_clean_because_parity_was_skipped(monkeypatch):
    """The 2026-09-26 shape: every non-parity stage entered, the witness
    entered, then DegradedRun on a degraded Director. FAILED is FAILED."""
    walk = _walk(without=PARITY, witness=True, terminal="DegradedRun")
    sf = _rate(monkeypatch, [_exec(walk, status="FAILED")])
    assert sf["n_cycles"] == 1
    assert sf["n_cycles_clean"] == 0 and sf["n_cycles_clean_full_scope"] == 0
    assert sf["unattended_rate"] == 0.0


def test_a_vacuous_green_rerun_does_not_launder_a_failed_cycle(monkeypatch):
    """A SUCCEEDED rerun that entered no spine stage adds nothing, and the
    stage the failed run died in stays uncredited."""
    failed = _exec(_walk(without=PARITY, witness=True, terminal="DegradedRun"), status="FAILED")
    vacuous = _exec(
        ["CheckSkipEverything", "WriteCompletionMarker"], role="watch-rerun",
        run_date="2026-10-02", start=datetime(2026, 10, 4, 19, tzinfo=UTC),
    )
    sf = _rate(monkeypatch, [failed, vacuous])
    assert sf["n_cycles"] == 1
    assert sf["n_cycles_clean"] == 0


# --- rehearsals and keying --------------------------------------------------

def test_a_rehearsal_is_excluded_and_counted_never_folded(monkeypatch):
    """Rehearsals run under ``pipeline_role: watch-rerun`` and dry-run the
    graph; folded into a cycle their entered stages would read as work."""
    rehearsal = _exec(
        _walk(without=PARITY, witness=True), role="watch-rerun",
        payload={"rehearsal": "2026-09-25-agent-launched", "skip_parity": True},
    )
    sf = _rate(monkeypatch, [rehearsal])
    assert sf["n_cycles"] == 0
    assert sf["n_rehearsals_excluded"] == 1


# --- the card -------------------------------------------------------------

def test_the_card_publishes_both_scopes_and_the_split(monkeypatch):
    days = [datetime(2026, 9, d, 9, tzinfo=UTC) for d in (19, 26)] + [datetime(2026, 10, 3, 9, tzinfo=UTC)]
    execs = [_exec(_walk(without=PARITY, witness=True), start=d) for d in days]
    _wire(monkeypatch, execs)
    import boto3
    from moto import mock_aws

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        tile = build_substrate_tile("test-bucket", s3_client=s3, as_of=AS_OF, sfn_client=object())
    sf = next(c for c in tile["components"] if c["name"] == "sf_success_rate_4w")
    assert sf["value"] == pytest.approx(1.0) and sf["n_samples"] == 3
    reason = sf["status_reason"]
    assert "clean_with_declared_skips" in reason
    assert "FULL SCOPE (every declared stage ran): 0% (0/3" in reason
    assert "NOT_IN_SCOPE" in reason and "contamination" in reason
    scope = sf["cycle_scope"]
    assert scope["headline_scope"] == "clean_with_declared_skips"
    assert scope["full_scope"]["rate"] == 0.0
    assert scope["clean_with_declared_skips"]["rate"] == 1.0
    assert scope["declared_skips"]["label"] == "NOT_IN_SCOPE"
    assert scope["declared_skips"]["stages_by_sf"] == {SM: {"ParityParallel": 3, "PitParityCompare": 3}}
    assert scope["first_pass"]["n_clean"] == 3 and scope["recovered"]["n_clean"] == 0
    unatt = next(c for c in tile["components"] if c["name"] == "unattended_first_pass_rate")
    assert "FULL SCOPE: 0%" in unatt["status_reason"]


def test_a_cycle_whose_every_run_failed_is_never_clean_whatever_the_union_covers(monkeypatch):
    """Two failed reruns can, between them, have LEFT every stage (each is
    credited with what the other died in). Still not clean: no contributing
    execution succeeded, and success is what a clean cycle asserts."""
    a = _exec(_walk(without=PARITY, witness=True, terminal="DegradedRun"), status="FAILED")
    b = _exec(
        [WITNESS, "Director", "ReportCard", "DegradedRun"], status="FAILED",
        role="watch-rerun", run_date="2026-10-02", start=datetime(2026, 10, 4, 19, tzinfo=UTC),
    )
    sf = _rate(monkeypatch, [a, b])
    assert sf["n_cycles"] == 1
    assert sf["n_cycles_clean"] == 0 and sf["n_cycles_clean_full_scope"] == 0
