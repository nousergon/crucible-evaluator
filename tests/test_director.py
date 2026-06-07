"""Tests for the Director agent (Layer C) — Phase E. No LLM key / langchain
required: the LLM is injected, and the handler's plan-build is monkeypatched."""

import json

import boto3
import pytest
from moto import mock_aws

from director.agent import build_action_plan, build_messages
from director.carryover import load_ledger, merge_plan_into_ledger
from director.report_card_digest import summarize_report_card
from director.schema import ActionItem, DirectorWeeklyActionPlan

BUCKET = "alpha-engine-research"
# A TRADING day (Fri) so _resolve_run_date's trading-day normalization is a
# no-op (was "2026-05-30", a Saturday → would normalize to Fri 05-29 and break
# the seed/lookup match). Calendar→trading normalization is covered in
# tests/test_handler.py::TestResolveRunDate.
RUN_DATE = "2026-05-29"

_CARD = {
    "tiles_overall_status": "RED",
    "_provenance": {"run_date": RUN_DATE, "artifacts": {"n_read": 5, "n_missing": 12}},
    "tiles": {
        "portfolio_outcome": {"status": "RED", "letter": "F", "numeric_grade": 49.6, "components": [
            {"name": "information_ratio", "criticality": "critical", "status": "RED",
             "value": -4.1, "target": 0.5, "red_line": 0.0, "trend_decoration": "→",
             "status_reason": "IR = -4.1, deeply negative."},
            {"name": "sharpe_ratio", "criticality": "critical", "status": "GREEN", "value": 1.2},
            {"name": "dsr", "criticality": "supporting", "status": "N/A-NOT-IMPL", "value": None},
        ]},
        "predictor": {"status": "RED", "letter": "F", "numeric_grade": 70.0, "components": [
            {"name": "momentum_l1_ic", "criticality": "critical", "status": "RED",
             "value": -0.0015, "target": 0.03, "red_line": 0.0, "status_reason": "dead L1."},
        ]},
    },
}


def _plan() -> DirectorWeeklyActionPlan:
    return DirectorWeeklyActionPlan(
        run_date=RUN_DATE,
        system_summary="System underperforming SPY.",
        top_risks=["IR deeply negative", "momentum L1 dead"],
        action_items=[ActionItem(
            id="revive-momentum-l1", title="Revive momentum L1",
            rationale="momentum_l1_ic = -0.0015 (RED, below target 0.03).",
            evidence=["predictor.momentum_l1_ic"], proposed_owner="predictor",
            priority="P0", horizon="this_week", suggested_change_type="structural", confidence=70,
        )],
        carryover_review=[],
    )


class _FakeLLM:
    """A structured-output runnable stand-in: .invoke(messages) → a plan."""
    def __init__(self, plan, *, fail_times=0, exc=None):
        self.plan = plan
        self.fail_times = fail_times
        self.exc = exc or RuntimeError("overloaded")
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        return self.plan


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


class TestSchema:
    def test_action_item_confidence_bounds(self):
        with pytest.raises(Exception):
            ActionItem(id="x", title="t", rationale="r", proposed_owner="predictor",
                       priority="P1", horizon="this_week", suggested_change_type="param_tune", confidence=150)

    def test_plan_extra_allowed(self):
        p = DirectorWeeklyActionPlan(run_date="2026-01-01", system_summary="s", top_risks=[], action_items=[],
                                     extra_field="ok")
        assert p.run_date == "2026-01-01"


class TestDigest:
    def test_summarize_includes_overall_and_adverse(self):
        text = summarize_report_card(_CARD)
        assert "OVERALL: RED" in text
        assert "information_ratio" in text and "-4.1" in text
        assert "momentum_l1_ic" in text
        # N/A rolled up, not expanded.
        assert "N/A-NOT-IMPL×1" in text
        # GREEN sharpe not expanded into a reason line.
        assert "sharpe_ratio" not in text or "1.2" not in text

    def test_empty_card(self):
        assert "No Report Card" in summarize_report_card({})


class TestAgent:
    def test_build_messages_has_digest_and_carryover(self):
        msgs = build_messages(_CARD, carryover={"items": [{"id": "old-1", "title": "Old", "status": "carried_over"}]})
        system, human = msgs[0][1], msgs[1][1]
        assert "Director" in system
        assert "OVERALL: RED" in human
        assert "old-1" in human

    def test_build_action_plan_injected_llm(self):
        plan = build_action_plan(_CARD, llm=_FakeLLM(_plan()))
        assert plan.run_date == RUN_DATE
        assert plan.action_items[0].id == "revive-momentum-l1"

    def test_run_date_stamped_from_card(self):
        p = _plan(); p.run_date = ""
        plan = build_action_plan(_CARD, llm=_FakeLLM(p))
        assert plan.run_date == RUN_DATE  # stamped from provenance

    def test_retry_then_succeed(self):
        llm = _FakeLLM(_plan(), fail_times=1, exc=RuntimeError("overloaded_error"))
        import director.agent as A
        A.time.sleep = lambda *_: None  # no real sleep
        plan = build_action_plan(_CARD, llm=llm)
        assert llm.calls == 2 and plan.run_date == RUN_DATE

    def test_non_transient_raises(self):
        llm = _FakeLLM(_plan(), fail_times=5, exc=ValueError("bad schema"))
        with pytest.raises(ValueError):
            build_action_plan(_CARD, llm=llm)


class TestCarryover:
    def test_load_absent_empty(self, s3):
        assert load_ledger(BUCKET, s3_client=s3) == {"items": []}

    def test_merge_upsert_by_id_preserves_first_seen(self, s3):
        ledger = {"items": [{"id": "revive-momentum-l1", "title": "old", "status": "carried_over",
                             "first_seen": "2026-05-23"}]}
        merged = merge_plan_into_ledger(ledger, _plan(), RUN_DATE)
        row = next(r for r in merged["items"] if r["id"] == "revive-momentum-l1")
        assert row["first_seen"] == "2026-05-23"  # preserved
        assert row["last_seen"] == RUN_DATE
        assert row["status"] == "proposed"  # updated from the new plan

    def test_load_error_raises(self, s3):
        with pytest.raises(Exception):
            load_ledger("nonexistent-bucket-xyz", s3_client=s3)


class TestHandler:
    def test_disabled_is_noop(self, s3, monkeypatch):
        monkeypatch.delenv("DIRECTOR_ENABLED", raising=False)
        from director import handler as H
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET})
        assert out["status"] == "disabled"

    def test_enabled_writes_plan_and_ledger(self, s3, monkeypatch):
        monkeypatch.setenv("DIRECTOR_ENABLED", "1")
        s3.put_object(Bucket=BUCKET, Key=f"evaluator/{RUN_DATE}/report_card.json",
                      Body=json.dumps(_CARD).encode())
        from director import handler as H
        monkeypatch.setattr(H, "build_action_plan",
                            lambda card, **kw: _plan())
        # handler builds its own boto3 client → moto intercepts globally.
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET})
        assert out["status"] == "ok"
        assert out["n_action_items"] == 1
        assert out["action_plan_key"] == f"director/{RUN_DATE}/action_plan.json"
        written = json.loads(s3.get_object(Bucket=BUCKET, Key=out["action_plan_key"])["Body"].read())
        assert written["run_date"] == RUN_DATE
        assert out["ledger_size"] == 1

    def test_enabled_missing_card_raises(self, s3, monkeypatch):
        monkeypatch.setenv("DIRECTOR_ENABLED", "1")
        from director import handler as H
        with pytest.raises(RuntimeError):
            H.handler({"date": RUN_DATE, "bucket": BUCKET})

    def test_dry_run_probes_infra_without_invoke_or_write(self, s3, monkeypatch):
        # Friday-PM preflight (ROADMAP L4504): dry_run constructs the LLM client
        # (the langchain import + SSM key-fetch IAM check) and builds the digest,
        # but makes NO Opus call and NO write — and must NOT mutate the shared
        # carry-over ledger. Stub _default_llm so the test needs no key/langchain.
        monkeypatch.setenv("DIRECTOR_ENABLED", "1")
        import director.agent as A
        constructed = {"n": 0}
        monkeypatch.setattr(A, "_default_llm", lambda: constructed.__setitem__("n", constructed["n"] + 1))
        s3.put_object(Bucket=BUCKET, Key=f"evaluator/{RUN_DATE}/report_card.json",
                      Body=json.dumps(_CARD).encode())
        from director import handler as H
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET, "dry_run": True})
        assert out["status"] == "dry_run"
        assert out["card_present"] is True
        assert out["llm_constructed"] is True
        assert out["digest_built"] is True
        assert constructed["n"] == 1  # the client (key-fetch + import) WAS exercised
        # no action plan written, no ledger created.
        assert s3.list_objects_v2(Bucket=BUCKET, Prefix=f"director/{RUN_DATE}/").get("KeyCount", 0) == 0
        assert s3.list_objects_v2(Bucket=BUCKET, Prefix="director/carryover_ledger.json").get("KeyCount", 0) == 0

    def test_dry_run_tolerates_missing_card(self, s3, monkeypatch):
        # On a real preflight the upstream dry ReportCard didn't write a card, so
        # the Director's card read misses — dry_run must still exercise the client
        # (the key/import infra check) and return cleanly, NOT raise like live mode.
        monkeypatch.setenv("DIRECTOR_ENABLED", "1")
        import director.agent as A
        monkeypatch.setattr(A, "_default_llm", lambda: object())
        from director import handler as H
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET, "dry_run": True})
        assert out["status"] == "dry_run"
        assert out["card_present"] is False
        assert out["digest_built"] is False  # no card → digest skipped, but client still built
        assert out["llm_constructed"] is True

    def test_dry_run_respects_disabled_flag(self, s3, monkeypatch):
        # Pre-flip (DIRECTOR_ENABLED off) the Director no-ops regardless of dry_run.
        monkeypatch.delenv("DIRECTOR_ENABLED", raising=False)
        from director import handler as H
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET, "dry_run": True})
        assert out["status"] == "disabled"


def _retro() -> "object":
    from director.schema import RetroGrade
    return RetroGrade(prior_run_date="2026-05-23", grounding=80, calibration=55,
                      actionability=70, notes="Flagged risks mostly held.")


class TestRetro:
    """Phase G — self-grading retro loop."""

    def test_grade_prior_plan_injected_llm_stamps_prior_date(self):
        from director.retro import grade_prior_plan
        from director.schema import RetroGrade
        g = RetroGrade(prior_run_date="", grounding=80, calibration=55, actionability=70)
        out = grade_prior_plan({"run_date": "2026-05-23"}, _CARD, llm=_FakeLLM(g))
        assert isinstance(out, RetroGrade)
        assert out.prior_run_date == "2026-05-23"  # stamped from the plan
        assert out.calibration == 55

    def test_build_messages_has_prior_plan_and_current_card(self):
        from director.retro import build_messages
        msgs = build_messages(_plan().model_dump(), _CARD)
        human = msgs[-1][1]
        assert "PRIOR PLAN" in human and "CURRENT REPORT CARD" in human
        assert "Revive momentum L1" in human  # the prior plan's action item

    def test_handler_runs_retro_when_prior_plan_exists(self, s3, monkeypatch):
        monkeypatch.setenv("DIRECTOR_ENABLED", "1")
        # current card + a PRIOR plan (older date) seeded.
        s3.put_object(Bucket=BUCKET, Key=f"evaluator/{RUN_DATE}/report_card.json",
                      Body=json.dumps(_CARD).encode())
        s3.put_object(Bucket=BUCKET, Key="director/2026-05-23/action_plan.json",
                      Body=_plan().model_dump_json().encode())
        from director import handler as H
        monkeypatch.setattr(H, "build_action_plan", lambda card, **kw: _plan())
        import director.retro as R
        monkeypatch.setattr(R, "grade_prior_plan", lambda prior, card, **kw: _retro())
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET})
        assert out["status"] == "ok"
        assert out["retro"] == "ok"
        assert out["retro_prior_run_date"] == "2026-05-23"
        assert out["retro_calibration"] == 55
        # retro.json + trend persisted.
        retro = json.loads(s3.get_object(Bucket=BUCKET, Key=f"director/{RUN_DATE}/retro.json")["Body"].read())
        assert retro["calibration"] == 55
        trend = json.loads(s3.get_object(Bucket=BUCKET, Key="director/retro_trend.json")["Body"].read())
        assert len(trend["grades"]) == 1 and trend["grades"][0]["prior_run_date"] == "2026-05-23"

    def test_handler_skips_retro_on_first_cycle(self, s3, monkeypatch):
        monkeypatch.setenv("DIRECTOR_ENABLED", "1")
        s3.put_object(Bucket=BUCKET, Key=f"evaluator/{RUN_DATE}/report_card.json",
                      Body=json.dumps(_CARD).encode())  # no prior plan seeded
        from director import handler as H
        monkeypatch.setattr(H, "build_action_plan", lambda card, **kw: _plan())
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET})
        assert out["status"] == "ok"
        assert out["retro"] == "skipped"

    def test_handler_retro_failure_is_best_effort(self, s3, monkeypatch):
        # A retro failure must NOT lose the plan (primary deliverable) — status
        # stays ok, the plan is written, the retro records its error.
        monkeypatch.setenv("DIRECTOR_ENABLED", "1")
        s3.put_object(Bucket=BUCKET, Key=f"evaluator/{RUN_DATE}/report_card.json",
                      Body=json.dumps(_CARD).encode())
        s3.put_object(Bucket=BUCKET, Key="director/2026-05-23/action_plan.json",
                      Body=_plan().model_dump_json().encode())
        from director import handler as H
        monkeypatch.setattr(H, "build_action_plan", lambda card, **kw: _plan())
        import director.retro as R
        def _boom(prior, card, **kw):
            raise RuntimeError("judge overloaded")
        monkeypatch.setattr(R, "grade_prior_plan", _boom)
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET})
        assert out["status"] == "ok"  # plan still shipped
        assert out["retro"] == "error" and "judge overloaded" in out["retro_error"]
        assert json.loads(s3.get_object(Bucket=BUCKET, Key=out["action_plan_key"])["Body"].read())
