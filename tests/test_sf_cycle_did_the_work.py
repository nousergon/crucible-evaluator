"""A cycle whose only green run dispatched nothing is not a clean cycle, and
a declared self-skip is not a measured opportunity at all.

config-I7644 / alpha-engine-config-I8069. ``_sf_success_rate`` now classifies
every terminal, role-carrying, in-window execution with
``nousergon_lib.pipeline_status.read_work_outcome`` (``classify_work`` against
the execution's own entered-state history) instead of comparing the raw
terminal status string. Built from the REAL captured execution histories at
``nousergon-lib/tests/fixtures/pipeline_status/`` (alpha-engine-config-I8045):

* ``weekly_gateout_2026_08_20.json`` / ``weekly_gateout_2026_08_21.json`` —
  the THU-SAT ``WeeklyRunDayGate`` self-skip, terminal state
  ``WeeklyRunDaySkip``, SUCCEEDED in 3.5s having entered nothing of the
  declared spine. Verdict ``SKIPPED`` — excluded from every denominator.
* ``weekly_vacuous_success_watch_rerun_2026_08_16_4.json`` — the scheduled run
  died at ``PredictorBacktest`` after 3h53m; five recovery reruns failed; the
  sixth (``watch-rerun-2026-08-16-4``) carried 22 ``skip_*`` flags all true and
  terminated SUCCEEDED in under 9 minutes having entered ZERO declared
  substantive stages. Verdict ``INCOMPLETE`` (``vacuous_success``) — counts
  toward the denominator, never toward the numerator. Under the old
  ``clean iff any SUCCEEDED`` rule this cycle graded clean on a day that
  produced no report card, no Director plan and no ``backtest/2026-08-15/``
  prefix at all.
* ``weekly_failed_2026_08_15.json`` — the scheduled run itself, FAILED.
  Verdict ``INCOMPLETE`` (``execution_failed``).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from grading.tiles.substrate import _sf_success_rate

import nousergon_lib.pipeline_status as ps

UTC = timezone.utc

#: Copied verbatim from ``nousergon-lib/tests/fixtures/pipeline_status/`` — see
#: each file's own ``_provenance`` field (captured live via
#: ``AWS_PROFILE=ne-admin aws stepfunctions describe-execution`` +
#: ``get-execution-history``, alpha-engine-config-I8045). Copied rather than
#: read cross-repo so this suite is self-contained in CI, where a sibling
#: checkout of nousergon-lib is not guaranteed to exist.
_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "pipeline_status"
_SM_NAME = "ne-weekly-freshness-pipeline"
_STAGE_SPINE = ps.stage_order_for(_SM_NAME)


def _outcome_from_fixture(name: str) -> ps.WorkOutcome:
    """A real captured execution, classified with the live library predicate."""
    doc = json.loads((_FIXTURE_DIR / name).read_text())
    return ps.classify_work(
        state_machine_name=_SM_NAME,
        status=ps.RunStatus(doc["status"]),
        entered_states=doc["entered_states"],
        duration_sec=doc.get("duration_sec"),
        execution_arn=doc["execution_arn"],
        execution_name=doc.get("name"),
    )


def _outcome_full_run(execution_arn: str) -> ps.WorkOutcome:
    """A synthetic execution that entered the ENTIRE declared spine — COMPLETED."""
    entered = ["InitializeInput", *_STAGE_SPINE, "WriteCompletionMarker"]
    return ps.classify_work(
        state_machine_name=_SM_NAME,
        status=ps.RunStatus.SUCCEEDED,
        entered_states=entered,
        execution_arn=execution_arn,
    )


def _outcome_failed(execution_arn: str) -> ps.WorkOutcome:
    return ps.classify_work(
        state_machine_name=_SM_NAME,
        status=ps.RunStatus.FAILED,
        entered_states=["InitializeInput", "WeeklyPreflight"],
        execution_arn=execution_arn,
    )


class _Run:
    def __init__(self, name, status, start, role, execution_arn, run_date=None):
        self.name = name
        self.start_utc = start
        self.pipeline_role = role
        self.status = status
        self.execution_arn = execution_arn
        self.run_date = run_date


def _rate(monkeypatch, runs, outcomes_by_arn):
    monkeypatch.setattr(
        "nousergon_lib.pipeline_status.list_recent_pipeline_runs",
        lambda arn, limit=50, client=None: runs,
    )
    monkeypatch.setattr(
        "nousergon_lib.pipeline_status.read_work_outcome",
        lambda execution_arn, client=None: outcomes_by_arn[execution_arn],
    )
    monkeypatch.setattr(
        "grading.tiles.substrate._discover_sf_arns",
        lambda sfn: [f"arn:aws:states:us-east-1:1:stateMachine:{_SM_NAME}"],
    )
    return _sf_success_rate(object(), datetime(2026, 8, 22, tzinfo=UTC), 28)


# --------------------------------------------------------------------------
# SKIPPED — a declared self-skip is excluded from the denominator entirely.
# --------------------------------------------------------------------------

def test_a_weekly_run_day_skip_is_excluded_from_the_denominator(monkeypatch):
    """The THU-SAT self-skip is the pipeline correctly declining to run — not
    an unmeasured opportunity, and not a clean cycle either. Its whole cycle
    must not appear in n_cycles at all."""
    skip = _outcome_from_fixture("weekly_gateout_2026_08_20.json")
    assert skip.verdict is ps.WorkVerdict.SKIPPED
    run = _Run(
        "gateout", "SUCCEEDED", datetime(2026, 8, 20, 9, tzinfo=UTC),
        "weekly", skip.execution_arn, "2026-08-20",
    )
    sf = _rate(monkeypatch, [run], {skip.execution_arn: skip})
    assert sf["n_cycles"] == 0
    assert sf["cycle_rate"] is None
    assert sf["n_unattended"] == 0


def test_two_skip_days_and_one_real_cycle_only_the_real_one_counts(monkeypatch):
    gateout_20 = _outcome_from_fixture("weekly_gateout_2026_08_20.json")
    gateout_21 = _outcome_from_fixture("weekly_gateout_2026_08_21.json")
    full = _outcome_full_run("arn:aws:states:us-east-1:1:execution:x:full-run")
    runs = [
        _Run("g20", "SUCCEEDED", datetime(2026, 8, 20, 9, tzinfo=UTC), "weekly",
             gateout_20.execution_arn, "2026-08-20"),
        _Run("g21", "SUCCEEDED", datetime(2026, 8, 21, 9, tzinfo=UTC), "weekly",
             gateout_21.execution_arn, "2026-08-21"),
        _Run("real", "SUCCEEDED", datetime(2026, 8, 22, 9, tzinfo=UTC), "weekly",
             full.execution_arn, "2026-08-22"),
    ]
    sf = _rate(monkeypatch, runs, {
        gateout_20.execution_arn: gateout_20,
        gateout_21.execution_arn: gateout_21,
        full.execution_arn: full,
    })
    assert sf["n_cycles"] == 1
    assert sf["n_cycles_clean"] == 1
    assert sf["cycle_rate"] == 1.0


# --------------------------------------------------------------------------
# INCOMPLETE (vacuous_success / execution_failed) — counts toward the
# denominator, never the numerator.
# --------------------------------------------------------------------------

def test_the_vacuous_success_watch_rerun_counts_as_a_non_clean_cycle(monkeypatch):
    """watch-rerun-2026-08-16-4: terminal SUCCEEDED, zero declared substantive
    stages entered. Must count toward n_cycles and NOT toward n_cycles_clean."""
    vacuous = _outcome_from_fixture("weekly_vacuous_success_watch_rerun_2026_08_16_4.json")
    assert vacuous.verdict is ps.WorkVerdict.INCOMPLETE
    assert vacuous.reason == "vacuous_success"
    run = _Run(
        "watch-rerun-2026-08-16-4", "SUCCEEDED",
        datetime(2026, 8, 16, 3, 21, tzinfo=UTC), "watch-rerun",
        vacuous.execution_arn, "2026-08-15",
    )
    sf = _rate(monkeypatch, [run], {vacuous.execution_arn: vacuous})
    assert sf["n_cycles"] == 1
    assert sf["n_cycles_clean"] == 0
    assert sf["cycle_rate"] == 0.0
    # scope_detail is keyed "<sf-name>:<day>" since alpha-engine-config-I8183:
    # the dict is shared across all three SFs, so keyed on the day alone a
    # second SF running that day overwrote the first last-write-wins, and the
    # card showed one pipeline while claiming to describe the day.
    assert "vacuous_success" in sf["scope_detail"]["ne-weekly-freshness-pipeline:2026-08-15"]


def test_the_no_op_recovery_does_not_make_the_real_2026_08_15_cycle_clean(monkeypatch):
    """The full real cycle: a FAILED scheduled run plus four FAILED reruns plus
    the vacuous-success sixth rerun. None of the six did the declared work —
    the cycle must grade dirty, not clean."""
    scheduled = _outcome_failed("arn:aws:states:us-east-1:1:execution:x:scheduled")
    rerun1 = _outcome_failed("arn:aws:states:us-east-1:1:execution:x:rerun1")
    vacuous = _outcome_from_fixture("weekly_vacuous_success_watch_rerun_2026_08_16_4.json")
    base = datetime(2026, 8, 15, 9, 0, tzinfo=UTC)
    runs = [
        _Run("scheduled", "FAILED", base, "weekly", scheduled.execution_arn, "2026-08-15"),
        _Run("watch-rerun-1", "FAILED", base + timedelta(hours=6), "watch-rerun",
             rerun1.execution_arn, "2026-08-15"),
        _Run("watch-rerun-2026-08-16-4", "SUCCEEDED", datetime(2026, 8, 16, 3, 21, tzinfo=UTC),
             "watch-rerun", vacuous.execution_arn, "2026-08-15"),
    ]
    sf = _rate(monkeypatch, runs, {
        scheduled.execution_arn: scheduled,
        rerun1.execution_arn: rerun1,
        vacuous.execution_arn: vacuous,
    })
    assert sf["n_cycles"] == 1, "run_date must key one cycle, not several"
    assert sf["n_cycles_clean"] == 0
    assert sf["cycle_rate"] == 0.0


def test_a_recovered_cycle_that_did_the_work_still_counts_clean(monkeypatch):
    """config#1059 must not regress: recovery is not failure. A scheduled run
    that FAILED plus a same-cycle recovery run that entered the FULL declared
    spine (COMPLETED) grades the cycle clean."""
    scheduled = _outcome_failed("arn:aws:states:us-east-1:1:execution:x:scheduled")
    recovery = _outcome_full_run("arn:aws:states:us-east-1:1:execution:x:recovery")
    base = datetime(2026, 8, 15, 9, 0, tzinfo=UTC)
    runs = [
        _Run("scheduled", "FAILED", base, "weekly", scheduled.execution_arn, "2026-08-15"),
        _Run("recovery", "SUCCEEDED", base + timedelta(hours=6), "watch-rerun",
             recovery.execution_arn, "2026-08-15"),
    ]
    sf = _rate(monkeypatch, runs, {
        scheduled.execution_arn: scheduled,
        recovery.execution_arn: recovery,
    })
    assert sf["n_cycles_clean"] == 1
    assert sf["cycle_rate"] == 1.0
    # Unattended (first-pass, no recovery) must NOT credit this cycle — a
    # recovery role appeared alongside the scheduled run.
    assert sf["n_unattended_ok"] == 0
    assert sf["n_unattended"] == 1


def test_the_same_cycle_keyed_on_start_date_would_have_split_in_two(monkeypatch):
    """The old key, reproduced by dropping run_date. Two cycles, and the one
    holding the (still non-clean) vacuous-success run has no failed scheduled
    run beside it — which is how a dead week could read partially clean."""
    scheduled = _outcome_failed("arn:aws:states:us-east-1:1:execution:x:scheduled")
    vacuous = _outcome_from_fixture("weekly_vacuous_success_watch_rerun_2026_08_16_4.json")
    runs = [
        _Run("scheduled", "FAILED", datetime(2026, 8, 15, 9, 0, tzinfo=UTC), "weekly",
             scheduled.execution_arn, None),
        _Run("watch-rerun-2026-08-16-4", "SUCCEEDED", datetime(2026, 8, 16, 3, 21, tzinfo=UTC),
             "watch-rerun", vacuous.execution_arn, None),
    ]
    sf = _rate(monkeypatch, runs, {
        scheduled.execution_arn: scheduled,
        vacuous.execution_arn: vacuous,
    })
    assert sf["n_cycles"] == 2
    assert sf["n_cycles_clean"] == 0


def test_a_missing_run_date_warns_rather_than_going_quietly_inert(monkeypatch, caplog):
    """Below the nousergon-lib version supplying run_date, the fallback is the
    old (wrong) key. It must say so — a fix that goes silent on a stale pin is
    the guard-that-only-works-when-cooperating class."""
    full = _outcome_full_run("arn:aws:states:us-east-1:1:execution:x:x")
    run = _Run("x", "SUCCEEDED", datetime(2026, 8, 15, 9, tzinfo=UTC), "weekly",
                full.execution_arn, None)
    with caplog.at_level("WARNING"):
        _rate(monkeypatch, [run], {full.execution_arn: full})
    assert any("no input.run_date" in r.message for r in caplog.records)


def test_no_role_ad_hoc_runs_are_excluded_entirely(monkeypatch):
    """Role-less smoke/legacy runs never even reach read_work_outcome — they
    must not appear in outcomes_by_arn, or the test double raises KeyError."""
    run = _Run("adhoc", "SUCCEEDED", datetime(2026, 8, 15, 9, tzinfo=UTC), None,
                "arn:aws:states:us-east-1:1:execution:x:adhoc", "2026-08-15")
    sf = _rate(monkeypatch, [run], {})
    assert sf["n_cycles"] == 0
