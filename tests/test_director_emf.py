"""director/emf.py: the Director's metrics reach CloudWatch from the weekly spot.

The on-box Director (alpha-engine-config-I11936) prints the same EMF lines as
the Lambda, but nothing on the spot extracts them. AlphaEngine/Director had no
datapoint after the last Lambda run (2026-10-03), although the 2026-10-04
on-box run made a plan call.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from director import box_run, emf

RECORD = {
    "_aws": {
        "Timestamp": 1791041722338,
        "CloudWatchMetrics": [{
            "Namespace": "AlphaEngine/Director",
            "Dimensions": [["Group"]],
            "Metrics": [
                {"Name": "DirectorPlanLatencySeconds", "Unit": "Seconds"},
                {"Name": "DirectorPlanLatencyAmber", "Unit": "Count"},
                {"Name": "DirectorPlanPromptTokens", "Unit": "Count"},
            ],
        }],
    },
    "Group": "ultra",
    "DirectorPlanLatencySeconds": 332.6,
    "DirectorPlanLatencyAmber": 1,
    "DirectorPlanPromptTokens": None,
    "outcome": "ok",
}


class _FakeCW:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def put_metric_data(self, **kwargs):
        if self.fail:
            raise RuntimeError("AccessDenied")
        self.calls.append(kwargs)


@pytest.fixture
def fake_cw(monkeypatch):
    import boto3

    cw = _FakeCW()
    monkeypatch.setattr(boto3, "client", lambda service, **kw: cw)
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
    return cw


def test_metric_data_matches_the_emf_fan_out():
    [(namespace, data)] = emf.metric_data(RECORD)
    assert namespace == "AlphaEngine/Director"
    # The None-valued metric is skipped, as EMF extraction skips it.
    assert [d["MetricName"] for d in data] == [
        "DirectorPlanLatencySeconds", "DirectorPlanLatencyAmber",
    ]
    assert all(d["Dimensions"] == [{"Name": "Group", "Value": "ultra"}] for d in data)
    assert data[0]["Value"] == 332.6 and data[0]["Unit"] == "Seconds"
    assert data[0]["Timestamp"] == datetime.fromtimestamp(1791041722.338, tz=timezone.utc)


def test_a_missing_dimension_value_publishes_nothing_for_that_set():
    record = {k: v for k, v in RECORD.items() if k != "Group"}
    assert emf.metric_data(record) == []


def test_off_by_default_prints_only(fake_cw, capsys):
    emf.emit(RECORD)
    assert json.loads(capsys.readouterr().out.strip()) == RECORD
    assert fake_cw.calls == []


def test_enabled_prints_and_publishes(fake_cw, capsys):
    emf.enable_direct_put()
    emf.emit(RECORD)
    assert json.loads(capsys.readouterr().out.strip()) == RECORD
    [call] = fake_cw.calls
    assert call["Namespace"] == "AlphaEngine/Director"
    assert len(call["MetricData"]) == 2


def test_never_publishes_twice_on_lambda(fake_cw, monkeypatch):
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "alpha-engine-evaluator-director")
    emf.enable_direct_put()
    emf.emit(RECORD)
    assert fake_cw.calls == []


def test_a_put_failure_never_raises(monkeypatch, capsys, caplog):
    import boto3

    monkeypatch.setattr(boto3, "client", lambda service, **kw: _FakeCW(fail=True))
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
    emf.enable_direct_put()
    emf.emit(RECORD)  # must not raise
    assert json.loads(capsys.readouterr().out.strip()) == RECORD
    assert "PutMetricData failed" in caplog.text


def test_box_run_turns_direct_put_on(monkeypatch):
    from director import handler

    monkeypatch.setattr(handler, "run_director", lambda *a, **k: {"status": "not_degraded"})
    monkeypatch.delenv("DIRECTOR_REGISTRY_DEST", raising=False)
    assert not emf.direct_put_enabled()
    box_run.main(["--date", "2026-10-02"])
    assert emf.direct_put_enabled()


def test_the_plan_latency_emitter_publishes_on_the_box(fake_cw):
    from director import agent

    emf.enable_direct_put()
    agent._emit_plan_latency(
        elapsed_s=332.6, outcome="ok", prompt_chars=63779, carryover_items=20,
    )
    [call] = fake_cw.calls
    names = {d["MetricName"] for d in call["MetricData"]}
    assert {"DirectorPlanLatencySeconds", "DirectorPlanLatencyAmber"} <= names
