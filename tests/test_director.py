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

    def test_digest_surfaces_low_reliability_and_horizon(self):
        # L4562 — an adverse component flagged reliability=low / non-canonical
        # horizon must surface a validity warning so the Director hedges.
        card = {
            "tiles_overall_status": "RED", "_provenance": {"run_date": RUN_DATE},
            "tiles": {"research": {"status": "RED", "letter": "F", "components": [
                {"name": "cio", "criticality": "critical", "status": "RED",
                 "value": -0.038, "target": 0.05, "red_line": -0.02,
                 "measurement_horizon": "5d", "reliability": "low",
                 "status_reason": "[5d] cio edge -3.8%."},
            ]}},
        }
        text = summarize_report_card(card)
        assert "reliability LOW" in text
        assert "horizon 5d" in text


class TestAgent:
    def test_build_messages_has_digest_and_carryover(self):
        msgs = build_messages(_CARD, carryover={"items": [{"id": "old-1", "title": "Old", "status": "carried_over"}]})
        system, human = msgs[0][1], msgs[1][1]
        assert "Director" in system
        assert "OVERALL: RED" in human
        assert "old-1" in human

    def test_build_messages_includes_resolved_digest_section(self):
        msgs = build_messages(
            _CARD,
            roadmap_digest="#10 [P1] open thing",
            resolved_digest="#1168 (closed 2026-06-23) [director] Validate CIO skill (id=validate-cio-selection-skill-metric)",
        )
        human = msgs[1][1]
        assert "Recently INVESTIGATED & RESOLVED" in human
        assert "do NOT re-propose these under a new name" in human
        assert "validate-cio-selection-skill-metric" in human
        assert "open backlog — do NOT re-propose" in human  # both sections present

    def test_build_messages_omits_resolved_section_when_absent(self):
        msgs = build_messages(_CARD)
        assert "Recently INVESTIGATED & RESOLVED" not in msgs[1][1]

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

    def test_new_item_starts_carry_count_zero(self, s3):
        merged = merge_plan_into_ledger({"items": []}, _plan(), RUN_DATE)
        row = merged["items"][0]
        assert row["carry_count"] == 0
        assert row["escalated"] is False
        assert row["issue_number"] is None

    def test_carry_count_increments_while_not_resolved(self, s3):
        ledger = {"items": [{"id": "revive-momentum-l1", "status": "carried_over",
                             "first_seen": "2026-05-23", "carry_count": 1, "escalated": False,
                             "issue_number": 501}]}
        merged = merge_plan_into_ledger(ledger, _plan(), RUN_DATE)  # _plan()'s item defaults to "proposed"
        row = next(r for r in merged["items"] if r["id"] == "revive-momentum-l1")
        assert row["carry_count"] == 2
        assert row["issue_number"] == 501  # preserved — not derivable from the plan

    def test_carry_count_resets_on_resolved(self, s3):
        ledger = {"items": [{"id": "revive-momentum-l1", "status": "carried_over",
                             "first_seen": "2026-05-23", "carry_count": 3, "escalated": True,
                             "issue_number": 501}]}
        plan = _plan()
        plan.action_items[0].status = "resolved"
        merged = merge_plan_into_ledger(ledger, plan, RUN_DATE)
        row = next(r for r in merged["items"] if r["id"] == "revive-momentum-l1")
        assert row["carry_count"] == 0
        assert row["escalated"] is False


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

    def test_reads_dated_snapshot_not_latest_pointer(self, s3, monkeypatch):
        # config-I2556: the grading Lambda now ALSO maintains a continuously
        # updated evaluator/latest/report_card.json. Seed it with DIFFERENT
        # content than the dated snapshot and confirm the Director's plan is
        # built from the dated (frozen) card, never the moving latest one.
        monkeypatch.setenv("DIRECTOR_ENABLED", "1")
        s3.put_object(Bucket=BUCKET, Key=f"evaluator/{RUN_DATE}/report_card.json",
                      Body=json.dumps(_CARD).encode())
        stale_latest = {**_CARD, "tiles_overall_status": "GREEN"}
        s3.put_object(Bucket=BUCKET, Key="evaluator/latest/report_card.json",
                      Body=json.dumps(stale_latest).encode())
        from director import handler as H
        captured = {}

        def _fake_build(card, **kw):
            captured["card"] = card
            return _plan()

        monkeypatch.setattr(H, "build_action_plan", _fake_build)
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET})
        assert out["status"] == "ok"
        assert captured["card"]["tiles_overall_status"] == "RED"  # the dated card, not "GREEN"

    def test_issue_filing_skipped_when_no_token(self, s3, monkeypatch):
        # Phase H (repointed): no PAT configured → the issue channel records a
        # skip (no silent swallow) and the plan still writes. Non-fatal advisory.
        monkeypatch.setenv("DIRECTOR_ENABLED", "1")
        s3.put_object(Bucket=BUCKET, Key=f"evaluator/{RUN_DATE}/report_card.json",
                      Body=json.dumps(_CARD).encode())
        from director import handler as H
        monkeypatch.setattr(H, "build_action_plan", lambda card, **kw: _plan())
        monkeypatch.setattr(H, "_director_github_token", lambda: None)
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET})
        assert out["status"] == "ok"
        assert out["director_issues"] == "skipped"
        assert out["director_issues_reason"] == "no token configured"

    def test_check_deploy_drift_dispatches_before_enabled_flag(self, monkeypatch):
        # config#2348: the drift probe must run even when DIRECTOR_ENABLED is
        # off (the default) — a dormant-but-stale image is still stale.
        monkeypatch.delenv("DIRECTOR_ENABLED", raising=False)
        from director import handler as H
        import grading.deploy_drift as dd
        monkeypatch.setattr(
            dd, "check_deploy_drift",
            lambda *, function_name: {"has_drift": True, "function_name": function_name},
        )

        class _Ctx:
            function_name = "alpha-engine-evaluator-director"

        out = H.handler({"action": "check_deploy_drift"}, context=_Ctx())
        assert out == {"has_drift": True, "function_name": "alpha-engine-evaluator-director"}

    def test_issues_filed_when_token_present(self, s3, monkeypatch):
        # Phase H (repointed): with a PAT, the handler files area:director-proposals
        # issues and threads the live open-issue backlog digest into the plan build.
        monkeypatch.setenv("DIRECTOR_ENABLED", "1")
        s3.put_object(Bucket=BUCKET, Key=f"evaluator/{RUN_DATE}/report_card.json",
                      Body=json.dumps(_CARD).encode())
        from director import handler as H
        seen = {}
        monkeypatch.setattr(H, "build_action_plan",
                            lambda card, **kw: seen.update(kw) or _plan())
        monkeypatch.setattr(H, "_director_github_token", lambda: "tok")
        monkeypatch.setattr(H, "_fetch_backlog_digest_best_effort", lambda tok: "DIGEST")
        monkeypatch.setattr(H, "file_director_issues",
                            lambda plan, run_date, token: {
                                "status": "ok", "n_filed": 1,
                                "issues": [{"slug": "s", "number": 7,
                                            "url": "https://x/issues/7"}]})
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET})
        assert out["director_issues"] == "ok"
        assert out["director_issues_urls"] == ["https://x/issues/7"]
        assert seen.get("roadmap_digest") == "DIGEST"  # backlog digest threaded into the build

    def test_issue_filing_disabled_kill_switch(self, s3, monkeypatch):
        # Preserved kill-switch: DIRECTOR_ROADMAP_PR_ENABLED=false disables filing.
        monkeypatch.setenv("DIRECTOR_ENABLED", "1")
        monkeypatch.setenv("DIRECTOR_ROADMAP_PR_ENABLED", "false")
        s3.put_object(Bucket=BUCKET, Key=f"evaluator/{RUN_DATE}/report_card.json",
                      Body=json.dumps(_CARD).encode())
        from director import handler as H
        monkeypatch.setattr(H, "build_action_plan", lambda card, **kw: _plan())
        monkeypatch.setattr(H, "_director_github_token", lambda: "tok")
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET})
        assert out["director_issues"] == "disabled"

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

    # ── config#1673: cross-model judge (judge != generator) ─────────

    def test_judge_model_defaults_and_respects_env_override(self, monkeypatch):
        """RETRO_JUDGE_MODEL is config, not code: env override wins; the
        default is the high-group primary (DeepSeek V4 Pro Max), deliberately
        NOT the Director's ``ultra`` group — grading a plan with the model
        that wrote it is self-grading bias."""
        from director.agent import DIRECTOR_GROUP
        from director.retro import RETRO_JUDGE_MODEL_DEFAULT, _judge_model

        monkeypatch.delenv("RETRO_JUDGE_MODEL", raising=False)
        assert _judge_model() == RETRO_JUDGE_MODEL_DEFAULT == "deepseek-v4-pro-max"

        # The Director now addresses the `ultra` GROUP rather than pinning a
        # model, so judge != generator can no longer be asserted against a
        # single constant — the served model depends on which chain entry is
        # healthy. What IS still assertable: the judge is not the group itself,
        # and the two are configured independently.
        assert DIRECTOR_GROUP == "ultra"
        assert _judge_model() != DIRECTOR_GROUP

        # KNOWN GAP, tracked as alpha-engine-config-I6052 and deliberately NOT
        # asserted here: `deepseek-v4-pro-max` is BOTH this judge default (the
        # `high` primary) and `ultra`'s LAST fallback. If ultra exhausts its
        # first three entries the plan is graded by the model that wrote it.
        # Enforcing that needs the SERVED model at call time, which this unit
        # test has no access to — asserting a weaker proxy here would make the
        # invariant look enforced when it is not.

        monkeypatch.setenv("RETRO_JUDGE_MODEL", "claude-sonnet-5")
        assert _judge_model() == "claude-sonnet-5"

    def test_grade_prior_plan_injected_llm_never_touches_real_secrets(self, monkeypatch):
        """Test-hygiene guard (bit a previous integration): the llm= injection
        point must short-circuit _default_llm() entirely, so an ambient
        OPENROUTER_API_KEY sitting in the CI environment (as this repo's own
        runner may have) can never leak into a hermetic test path. Simulates
        the leak by setting a fake key AND making krepis.secrets.get_secret
        raise if it's ever called."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "leaked-ambient-key")
        import krepis.secrets as ks

        def _must_not_be_called(*a, **kw):
            raise AssertionError(
                "krepis.secrets.get_secret must not be called when llm= is injected"
            )
        monkeypatch.setattr(ks, "get_secret", _must_not_be_called)

        from director.retro import grade_prior_plan
        from director.schema import RetroGrade
        g = RetroGrade(prior_run_date="", grounding=80, calibration=55, actionability=70)
        out = grade_prior_plan({"run_date": "2026-05-23"}, _CARD, llm=_FakeLLM(g))
        assert out.calibration == 55

    def test_default_llm_routes_through_krepis_for_configured_judge_model(self, monkeypatch):
        """_default_llm() must build a krepis.llm.LLMClient (not langchain's
        ChatAnthropic) targeting the OpenRouter provider + the configured
        judge alias — never agent.DIRECTOR_MODEL."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("RETRO_JUDGE_MODEL", "claude-sonnet-4-6")
        import krepis.secrets as ks
        monkeypatch.setattr(ks, "get_secret", lambda name, **kw: "test-key")

        from director.retro import _KrepisStructuredJudge, _default_llm
        llm = _default_llm()
        assert isinstance(llm, _KrepisStructuredJudge)
        assert llm._judge_model == "claude-sonnet-4-6"
        assert llm._client.spec.provider == "openrouter"
        assert llm._client.spec.model == "claude-sonnet-4-6"

    def test_krepis_judge_stamps_judge_model_and_resolved_model(self):
        """End-to-end (hermetic, no network / no anthropic SDK): drives
        _KrepisStructuredJudge.invoke() through a krepis.llm.LLMClient whose
        transport is a fake injected via client_factory (krepis' own test
        seam), simulating the API resolving the floating Sonnet alias to a
        dated snapshot. Verifies structured-output validation against
        RetroGrade AND that judge_model (the alias) / resolved_model (the
        API-reported model) both land as extra fields and survive
        model_dump()."""
        from krepis.llm import LLMClient
        from krepis.llm_config import ModelSpec
        from director.retro import RETRO_JUDGE_MODEL_DEFAULT, _KrepisStructuredJudge, build_messages

        class _FakeToolUseBlock:
            def __init__(self, name, input_):
                self.type = "tool_use"
                self.name = name
                self.input = input_

        class _FakeAnthropicMessage:
            def __init__(self, content, model):
                self.content = content
                self.model = model
                self.usage = None

        class _FakeAnthropicClient:
            def __init__(self):
                self.messages = self

            def create(self, **payload):
                assert payload["model"] == RETRO_JUDGE_MODEL_DEFAULT
                tool_input = {
                    "prior_run_date": "2026-05-23",
                    "grounding": 80,
                    "calibration": 55,
                    "actionability": 70,
                    "notes": "Flagged risks mostly materialized.",
                }
                block = _FakeToolUseBlock("RetroGrade", tool_input)
                # The API resolves the floating alias to a dated snapshot —
                # distinct from the request's `model=` alias string.
                return _FakeAnthropicMessage([block], model="claude-sonnet-4-6-20260115")

        def _client_factory(spec, api_key):
            assert api_key == "test-key"
            return _FakeAnthropicClient()

        spec = ModelSpec(provider="anthropic", model=RETRO_JUDGE_MODEL_DEFAULT, max_tokens=2000)
        client = LLMClient(
            spec,
            api_key="test-key",
            client_factory=_client_factory,
            callsite_id="director-retro-judge",
        )
        judge = _KrepisStructuredJudge(client, judge_model=RETRO_JUDGE_MODEL_DEFAULT)

        messages = build_messages(_plan().model_dump(), _CARD)
        grade = judge.invoke(messages)

        assert grade.calibration == 55
        assert grade.judge_model == RETRO_JUDGE_MODEL_DEFAULT
        assert grade.resolved_model == "claude-sonnet-4-6-20260115"
        assert grade.judge_model != grade.resolved_model  # alias vs API-resolved snapshot
        dumped = grade.model_dump()
        assert dumped["judge_model"] == RETRO_JUDGE_MODEL_DEFAULT
        assert dumped["resolved_model"] == "claude-sonnet-4-6-20260115"

    def test_handler_persists_judge_model_and_resolved_model(self, s3, monkeypatch):
        """The judge_model/resolved_model extras (RetroGrade has
        extra="allow") must survive persistence into both the per-run
        retro.json and the retro_trend.json ledger row — handler.py's
        _persist_retro is unchanged; this is a regression guard on
        model_dump()/model_dump_json() carrying the new fields through."""
        monkeypatch.setenv("DIRECTOR_ENABLED", "1")
        s3.put_object(Bucket=BUCKET, Key=f"evaluator/{RUN_DATE}/report_card.json",
                      Body=json.dumps(_CARD).encode())
        s3.put_object(Bucket=BUCKET, Key="director/2026-05-23/action_plan.json",
                      Body=_plan().model_dump_json().encode())
        from director import handler as H
        monkeypatch.setattr(H, "build_action_plan", lambda card, **kw: _plan())
        import director.retro as R

        def _fake_grade(prior, card, **kw):
            g = _retro()
            g.judge_model = "claude-sonnet-4-6"
            g.resolved_model = "claude-sonnet-4-6-20260115"
            return g
        monkeypatch.setattr(R, "grade_prior_plan", _fake_grade)

        out = H.handler({"date": RUN_DATE, "bucket": BUCKET})
        assert out["retro"] == "ok"
        retro = json.loads(s3.get_object(Bucket=BUCKET, Key=f"director/{RUN_DATE}/retro.json")["Body"].read())
        assert retro["judge_model"] == "claude-sonnet-4-6"
        assert retro["resolved_model"] == "claude-sonnet-4-6-20260115"
        trend = json.loads(s3.get_object(Bucket=BUCKET, Key="director/retro_trend.json")["Body"].read())
        assert trend["grades"][0]["judge_model"] == "claude-sonnet-4-6"
        assert trend["grades"][0]["resolved_model"] == "claude-sonnet-4-6-20260115"


# ── execution-context routing (model-router-policy R28/R29) ──────────────
#
# alpha-engine-config-I6183: this module passed
# `exclude_route="litellm_proxy"` and resolution fell through to `glm-5.2` at
# openrouter.ai, DLP-unscanned, while `Director route:` logged a healthy
# route. Nothing failed when the proxy path was never restored — which is
# what made it a shortcut that produced a passing result.

class TestDirectorExecutionContext:
    def test_no_resolve_call_narrows_the_chain(self):
        """The consumer declares WHERE it runs and nothing else about routing
        (model-router-policy §2 layer 5). An exclusion argument is a routing
        table held at layer 5, whatever the mechanism.

        Asserted over the AST rather than the source text: prose in a
        docstring explaining why the argument was removed must not fail the
        test, and a grep-shaped check would either do that or be defeated by
        reformatting.
        """
        import ast
        import inspect
        from director import agent as A

        tree = ast.parse(inspect.getsource(A))
        offenders = [
            kw.arg
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", getattr(node.func, "attr", None))
            == "resolve_group_structured"
            for kw in node.keywords
            if kw.arg in {"exclude_route", "exclude_provider"}
        ]
        assert not offenders, (
            f"director/agent.py narrows the fallback chain at the call site "
            f"via {offenders} — that is a routing table held at layer 5"
        )

    def test_declares_lambda_vpc_by_default(self):
        from director import agent as A
        assert A.DIRECTOR_EXEC_CONTEXT == "lambda_vpc"

    def test_resolution_passes_context_and_openai_wire(self, monkeypatch):
        """`wire=openai` matters: this call site builds a krepis LLMClient on
        the openai transport, and the resolver's default is the Claude CLI's
        Anthropic wire. Without it a DeepSeek egress fallback would hand this
        module an 8971 URL its transport cannot speak."""
        from director import agent as A

        seen = {}

        def _fake_resolve(group, **kw):
            seen["group"] = group
            seen.update(kw)
            return {
                "schema_version": A._EXPECTED_ROUTE_SCHEMA,
                "provider": "litellm", "deployment_id": "ultra",
                "api_base_url": "http://172.31.73.124:8980",
                "auth_token_type": "litellm_master_key",
                "registry_id": "litellm:group:ultra",
                "primary_registry_id": "litellm:group:ultra",
                "route": "litellm_proxy", "exec_context": "lambda_vpc",
                "params": {}, "skipped_entries": [],
            }

        import sys, types
        krepis_router = types.ModuleType("krepis.router")
        krepis_router.resolve_group_structured = _fake_resolve
        krepis_llm = types.ModuleType("krepis.llm")
        krepis_llm.LLMClient = lambda spec, **kw: object()
        krepis_cfg = types.ModuleType("krepis.llm_config")

        class _Spec:
            def __init__(self, **kw):
                self.__dict__.update(kw)
        krepis_cfg.ModelSpec = _Spec
        pkg = types.ModuleType("krepis")
        for name, mod in (("krepis", pkg), ("krepis.router", krepis_router),
                          ("krepis.llm", krepis_llm),
                          ("krepis.llm_config", krepis_cfg)):
            monkeypatch.setitem(sys.modules, name, mod)
        monkeypatch.setattr(A, "_warn_on_degraded_route", lambda r: None)
        monkeypatch.setenv("LITELLM_MASTER_KEY", "test-key")

        A._default_llm()

        assert seen["group"] == "ultra"
        assert seen["exec_context"] == "lambda_vpc"
        assert seen["wire"] == "openai"
        assert "exclude_route" not in seen


class TestDirectorRouteDegradationIsAlerted:
    """alpha-engine-config-I6185 — krepis returns skipped_entries and
    primary_registry_id; this module discarded both, so a third-fallback route
    logged identically to the primary."""

    def _route(self, **over):
        r = {
            "registry_id": "kimi-k3-direct",
            "primary_registry_id": "kimi-k3-direct",
            "route": "litellm_proxy",
            "exec_context": "lambda_vpc",
            "skipped_entries": [],
        }
        r.update(over)
        return r

    def test_healthy_route_emits_metric_value_zero(self, monkeypatch):
        """A metric that only appears on failure is indistinguishable from a
        dead emitter (principles.md §2.7)."""
        from director import agent as A
        emitted = {}
        monkeypatch.setattr(A, "_cw_put", lambda v: emitted.setdefault("v", v),
                            raising=False)
        self._patch_boto(monkeypatch, emitted)
        A._warn_on_degraded_route(self._route())
        assert emitted["value"] == 0.0

    def test_fallback_emits_one_and_warns(self, monkeypatch, caplog):
        from director import agent as A
        emitted = {}
        self._patch_boto(monkeypatch, emitted)
        route = self._route(
            registry_id="glm-5.2",
            skipped_entries=[{"registry_id": "kimi-k3-direct",
                              "reason": "Not reachable from execution context 'lambda_vpc'"}],
        )
        with caplog.at_level("WARNING"):
            A._warn_on_degraded_route(route)
        assert emitted["value"] == 1.0
        assert "DEGRADED" in caplog.text
        assert "kimi-k3-direct" in caplog.text
        assert "lambda_vpc" in caplog.text

    def test_telemetry_failure_does_not_break_the_run(self, monkeypatch, caplog):
        """A blind alarm is bad; a weekly plan that does not run because
        CloudWatch was unavailable is worse. Logged as an exception, not
        swallowed."""
        from director import agent as A
        import sys, types
        boto3 = types.ModuleType("boto3")

        def _client(_name):
            raise RuntimeError("no creds")
        boto3.client = _client
        monkeypatch.setitem(sys.modules, "boto3", boto3)
        with caplog.at_level("ERROR"):
            A._warn_on_degraded_route(self._route())
        assert "alarm is blind" in caplog.text

    @staticmethod
    def _patch_boto(monkeypatch, emitted):
        import sys, types
        boto3 = types.ModuleType("boto3")

        class _CW:
            def put_metric_data(self, Namespace, MetricData):
                emitted["namespace"] = Namespace
                emitted["name"] = MetricData[0]["MetricName"]
                emitted["value"] = MetricData[0]["Value"]
        boto3.client = lambda _n: _CW()
        monkeypatch.setitem(sys.modules, "boto3", boto3)
