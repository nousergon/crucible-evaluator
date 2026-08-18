"""A cycle whose only green run dispatched nothing is not a clean cycle.

config-I7644. Built from the REAL 2026-08-15/16 `ne-weekly-freshness-pipeline`
execution set, because the shape that defeated the old rule is not one anybody
would invent: the scheduled run died at `PredictorBacktest` after 3h53m, five
recovery reruns failed, and the sixth — carrying 22 `skip_*` flags all true —
terminated SUCCEEDED in eight minutes having entered one real task. Under
`clean iff any SUCCEEDED`, that cycle graded clean, on a day that produced no
report card, no Director plan and no `backtest/2026-08-15/` prefix at all.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from grading.tiles.substrate import (
    _cycle_did_the_work,
    _cycle_run_scope,
    _sf_success_rate,
)

UTC = timezone.utc
STAGES = [
    "DataPhase1", "Scanner", "PredictorTraining", "PredictorBacktest",
    "Evaluator", "ReportCard", "Director",
]


def _scope(dispositions: dict) -> dict:
    return {
        "schema": "run_scope-1.0.0",
        "run_date": "2026-08-15",
        "stages": {n: {"disposition": d} for n, d in dispositions.items()},
    }


# --------------------------------------------------------------------------
# _cycle_did_the_work
# --------------------------------------------------------------------------

def test_the_2026_08_15_no_op_cycle_is_not_clean():
    """One stage dispatched, the rest never reached — the measured shape."""
    clean, detail = _cycle_did_the_work(_scope({
        "ChallengerShadow": "ENABLED_COMPLETED",
        **{n: "NOT_REACHED" for n in STAGES},
    }))
    assert clean is False
    assert "never reached" in detail
    assert "did NOT complete its work" in detail


def test_a_legitimate_partial_rerun_is_still_clean():
    """DISABLED is a decision. A rerun that dispatches three stages and
    completes them did its job; counting it dirty makes the metric permanently
    red and therefore unread."""
    clean, detail = _cycle_did_the_work(_scope({
        "DataPhase1": "DISABLED",
        "Scanner": "DISABLED",
        "PredictorTraining": "DISABLED",
        "PredictorBacktest": "ENABLED_COMPLETED",
        "Evaluator": "ENABLED_COMPLETED",
        "ReportCard": "ENABLED_COMPLETED",
        "Director": "DISABLED",
    }))
    assert clean is True
    assert "none left unreached" in detail


def test_a_full_run_is_clean():
    clean, _ = _cycle_did_the_work(_scope({n: "ENABLED_COMPLETED" for n in STAGES}))
    assert clean is True


def test_enabled_failed_alone_does_not_make_the_cycle_dirty():
    """A stage that ran and failed already fails the execution; this rule is
    about stages the run never got to."""
    clean, _ = _cycle_did_the_work(_scope({
        **{n: "ENABLED_COMPLETED" for n in STAGES},
        "PredictorBacktest": "ENABLED_FAILED",
    }))
    assert clean is True


@pytest.mark.parametrize("doc", [None, {}, {"stages": {}}, {"degraded": True}])
def test_absent_or_unmeasured_scope_grades_exactly_as_before(doc):
    """The two trading pipelines write no run_scope, and neither did any run
    before 2026-08-18. Silence declines to sharpen the verdict; it never
    flips one."""
    clean, _ = _cycle_did_the_work(doc)
    assert clean is True


def test_cycle_run_scope_returns_none_without_a_run_date():
    assert _cycle_run_scope(object(), "bucket", None) is None


# --------------------------------------------------------------------------
# _sf_success_rate — the end-to-end shape
# --------------------------------------------------------------------------

class _Run:
    def __init__(self, name, status, start, role, run_date=None):
        self.name, self.start_utc, self.pipeline_role = name, start, role
        self.status, self.run_date = status, run_date


class _S3:
    """Serves run_scope.json by key; anything else is absent."""
    def __init__(self, by_key):
        self.by_key = by_key
        self.asked = []

    def get_object(self, Bucket, Key):  # noqa: N803 — boto3 signature
        self.asked.append(Key)
        import io, json
        if Key not in self.by_key:
            from botocore.exceptions import ClientError
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(json.dumps(self.by_key[Key]).encode())}


def _the_real_2026_08_15_cycle(run_date="2026-08-15"):
    """Six executions. The scheduled run and four reruns FAILED; the last
    SUCCEEDED. The green one started 2026-08-16 03:21 UTC — a DIFFERENT UTC
    calendar date from the scheduled run, which is why the cycle key must be
    run_date and not start_utc.date()."""
    base = datetime(2026, 8, 15, 9, 0, tzinfo=UTC)
    return [
        _Run("scheduled", "FAILED", base, "weekly", run_date),
        _Run("watch-rerun-1", "FAILED", base + timedelta(hours=6), "watch-rerun", run_date),
        _Run("watch-rerun-2", "FAILED", base + timedelta(hours=14), "watch-rerun", run_date),
        _Run("watch-rerun-3", "FAILED", base + timedelta(hours=17), "watch-rerun", run_date),
        # 2026-08-16 in UTC, still run_date 2026-08-15.
        _Run("watch-rerun-4", "SUCCEEDED", datetime(2026, 8, 16, 3, 21, tzinfo=UTC),
             "watch-rerun", run_date),
    ]


def _rate(monkeypatch, runs, s3=None, bucket="alpha-engine-research"):
    monkeypatch.setattr(
        "nousergon_lib.pipeline_status.list_recent_pipeline_runs",
        lambda arn, limit=50, client=None: runs,
    )
    monkeypatch.setattr(
        "grading.tiles.substrate._discover_sf_arns",
        lambda sfn: ["arn:aws:states:us-east-1:1:stateMachine:ne-weekly-freshness-pipeline"],
    )
    return _sf_success_rate(
        object(), datetime(2026, 8, 18, tzinfo=UTC), 28, s3=s3, bucket=bucket,
    )


def test_the_no_op_recovery_does_not_make_the_cycle_clean(monkeypatch):
    s3 = _S3({"backtest/2026-08-15/run_scope.json": _scope({
        "ChallengerShadow": "ENABLED_COMPLETED",
        **{n: "NOT_REACHED" for n in STAGES},
    })})
    sf = _rate(monkeypatch, _the_real_2026_08_15_cycle(), s3=s3)
    assert sf["n_cycles"] == 1, "run_date must key one cycle, not two"
    assert sf["n_cycles_clean"] == 0
    assert sf["cycle_rate"] == 0.0
    assert "never reached" in str(sf["scope_detail"])


def test_the_same_cycle_keyed_on_start_date_would_have_split_in_two(monkeypatch):
    """The old key, reproduced by dropping run_date. Two cycles, and the one
    holding the green run has no failed scheduled run beside it — which is how
    a dead week came to read 50% instead of 0%."""
    runs = [
        _Run(r.name, r.status, r.start_utc, r.pipeline_role, None)
        for r in _the_real_2026_08_15_cycle()
    ]
    sf = _rate(monkeypatch, runs, s3=_S3({}))
    assert sf["n_cycles"] == 2
    assert sf["n_cycles_clean"] == 1


def test_a_recovered_cycle_that_did_the_work_still_counts_clean(monkeypatch):
    """config#1059 must not regress: recovery is not failure."""
    s3 = _S3({"backtest/2026-08-15/run_scope.json": _scope(
        {n: "ENABLED_COMPLETED" for n in STAGES}
    )})
    sf = _rate(monkeypatch, _the_real_2026_08_15_cycle(), s3=s3)
    assert sf["n_cycles_clean"] == 1
    assert sf["cycle_rate"] == 1.0


def test_no_run_scope_artifact_grades_on_status_alone(monkeypatch):
    """The two trading pipelines. Behaviour identical to before this change."""
    sf = _rate(monkeypatch, _the_real_2026_08_15_cycle(), s3=_S3({}))
    assert sf["n_cycles"] == 1
    assert sf["n_cycles_clean"] == 1


def test_the_scope_read_is_skipped_when_no_execution_succeeded(monkeypatch):
    """No point asking what an all-failed cycle dispatched — and it keeps the
    S3 call budget at one GET per GREEN cycle, not per cycle."""
    runs = [r for r in _the_real_2026_08_15_cycle() if r.status == "FAILED"]
    s3 = _S3({})
    sf = _rate(monkeypatch, runs, s3=s3)
    assert sf["n_cycles_clean"] == 0
    assert s3.asked == []


def test_a_missing_run_date_warns_rather_than_going_quietly_inert(monkeypatch, caplog):
    """Below the nousergon-lib version supplying run_date, the fallback is the
    old (wrong) key. It must say so — a fix that goes silent on a stale pin is
    the guard-that-only-works-when-cooperating class."""
    runs = [_Run("x", "SUCCEEDED", datetime(2026, 8, 15, 9, tzinfo=UTC), "weekly", None)]
    with caplog.at_level("WARNING"):
        _rate(monkeypatch, runs, s3=_S3({}))
    assert any("no input.run_date" in r.message for r in caplog.records)
