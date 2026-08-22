"""The Director's §2.3a consumer half (``director/verdict.py`` + its wiring).

`sf-pipeline-policy.md` §2.3a rule 1 — *the verdict is consumed by every stage
whose output depends on it being true.* The Director is the fleet's strongest
instance: it does not merely render the week's numbers, it files GitHub issues,
reopens issues, escalates reserved matters to Brian's Decision Queue, and emails
the advisory Brian reads. Every one of those outlives the cycle.

The tests are organised around the failure mode the clause exists to remove, not
around the functions:

``TestAbsenceIsNeverAPass``
    the card carries no verdict, an unreadable one, or an unrecognised string.
    Each must WITHHOLD. This is the defect the issue's own gotcha names as the
    most likely one to introduce.
``TestWithholding``
    what actually stops when the guarantee is withheld — proven by tripwires on
    the mutating calls, not by reading a status string — and, since Brian's
    2026-08-22 ruling (``alpha-engine-config-I8187``), what deliberately does
    NOT stop.
``TestMutationClassSplit``
    the gate's subject: the AUTHORITY an action exercises, not a global
    run-quality flag. A read that informs a human or repairs the Director's own
    ledger is not the same act as a write that reopens someone's issue, and one
    verdict flag must not govern both.
``TestPassRestoresAuthority``
    the gate is not merely always-closed.
``TestSurfaces``
    rule 3 — the artifact and the digest carry the state.
"""
from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws

from director import verdict as V
from director.emailer import build_director_digest

RUN_DATE = "2026-08-15"
BUCKET = "test-bucket"

_PASS_BLOCK = {
    "schema": "report_card_attestation-1.0.0",
    "run_date": RUN_DATE,
    "verdict": "PASS",
    "as_of": {"backtester": "2026-08-15T09:41:02Z",
              "evaluator_stage": "2026-08-15T10:02:55Z"},
    "evaluator": {"verdict": "PASS", "n_checks": 6},
    "backtester": {"verdict": "PASS", "n_checks": 6},
    "evaluator_stage": {"verdict": "PASS", "n_checks": 4},
    "promotion_withheld": False,
    "reason": "All three halves attested.",
}


def _card(attestation=...):
    card = {"tiles_overall_status": "GREEN", "tiles": {}}
    if attestation is not ...:
        card["attestation"] = attestation
    return card


# ════════════════════════════════════════════════════════════════════════════
class TestAbsenceIsNeverAPass:
    """Every way the verdict can fail to arrive resolves to UNKNOWN.

    The shared property: a consumer written with ``card.get("attestation", {})
    .get("verdict") != "FAIL"`` — or any truthiness test — passes all of these.
    That is the "old artifact, proceed" bug, and it reproduces exactly the
    blindness §2.3a exists to remove.
    """

    def test_card_without_attestation_key_withholds(self):
        # A card written before the verdict producer existed. Indistinguishable
        # from a clean run to any tolerant reader.
        block = V.read_card_verdict(_card())
        assert block["verdict"] == "UNKNOWN"
        assert block["present"] is False
        assert V.actions_withheld(block) is True
        assert "never read as a pass" in block["reason"]

    def test_null_attestation_withholds(self):
        block = V.read_card_verdict(_card(None))
        assert block["verdict"] == "UNKNOWN"
        assert V.actions_withheld(block) is True

    @pytest.mark.parametrize("bad", ["ok", "", "pass", "PASSED", True, 1, {}, []])
    def test_unrecognised_verdict_string_withholds(self, bad):
        # A producer that starts writing "ok" must not silently be believed. A
        # verdict vocabulary that accepts new truthy strings is not a verdict.
        block = V.read_card_verdict(_card({"verdict": bad}))
        assert block["verdict"] == "UNKNOWN", f"{bad!r} was accepted as a verdict"
        assert V.actions_withheld(block) is True

    def test_attestation_not_a_mapping_withholds(self):
        assert V.actions_withheld(V.read_card_verdict(_card("PASS"))) is True

    def test_no_card_at_all_withholds(self):
        assert V.actions_withheld(V.read_card_verdict(None)) is True

    def test_explicit_unknown_withholds_as_hard_as_fail(self):
        # They differ in what they say about the numbers, never in what the
        # Director may do next.
        unknown = V.read_card_verdict(_card({**_PASS_BLOCK, "verdict": "UNKNOWN"}))
        failed = V.read_card_verdict(_card({**_PASS_BLOCK, "verdict": "FAIL"}))
        assert V.actions_withheld(unknown) is V.actions_withheld(failed) is True
        # ...but they remain distinguishable on the surface.
        assert unknown["verdict"] != failed["verdict"]


# ════════════════════════════════════════════════════════════════════════════
class _Tripwire:
    """Records a call it must never receive."""

    def __init__(self):
        self.calls = 0

    def __call__(self, *a, **kw):
        self.calls += 1
        raise AssertionError("a gated mutation ran while the guarantee was withheld")


class TestWithholding:
    """What stops — proven by tripwires on the mutating calls."""

    @pytest.mark.parametrize("block", [
        None, {}, {"verdict": "UNKNOWN"}, {"verdict": "FAIL"}, {"verdict": "ok"},
    ])
    def test_issue_filing_never_runs(self, block, monkeypatch):
        from director import handler as H
        tw = _Tripwire()
        monkeypatch.setattr(H, "file_director_issues", tw)
        monkeypatch.setattr(H, "_issue_filing_enabled", lambda: True)

        out = H._file_issues_best_effort(object(), RUN_DATE, "tok", verdict_block=block)

        assert out["director_issues"] == "withheld"
        assert tw.calls == 0
        # Recorded, not silent: a skip with no reason is indistinguishable from
        # the feature being switched off.
        assert "§2.3a" in out["director_issues_reason"]

    @pytest.mark.parametrize("block", [None, {}, {"verdict": "UNKNOWN"}, {"verdict": "FAIL"}])
    def test_loop_verification_never_reopens_or_escalates(self, block, monkeypatch):
        from director import handler as H
        tw = _Tripwire()
        backfilled: list[list] = []
        monkeypatch.setattr(
            H, "backfill_issue_numbers",
            lambda items, repo=None, token=None: (backfilled.append(items), 2)[1],
        )
        monkeypatch.setattr(H, "verify_and_correct", tw)
        ledger = {"items": [{"id": "a", "status": "open"}]}

        out = H._verify_loop_best_effort(ledger, _card(), "tok", verdict_block=block)

        # The MUTATING half stops — a reopen is an accusation with no evidence.
        assert out["director_loop"] == "mutations_withheld"
        assert tw.calls == 0
        assert "§2.3a" in out["director_loop_reason"]
        # ...and the NON-mutating half runs anyway (alpha-engine-config-I8187).
        # Withholding an own-ledger repair buys no safety: `verify_and_correct`
        # skips every row with no `issue_number`, so gating the thing that FILLS
        # `issue_number` dismantles the state the corrections need. Three
        # consecutive UNKNOWN cycles left all 28 live rows null (I8179).
        assert out["director_loop_backfilled"] == 2
        assert backfilled == [ledger["items"]]

    @pytest.mark.parametrize("block", [None, {}, {"verdict": "UNKNOWN"}, {"verdict": "FAIL"}])
    def test_backfill_runs_before_the_gate_is_evaluated(self, block, monkeypatch):
        """The ordering, not just the fact. The backfill must complete BEFORE
        the withholding decision returns, so the cycle the attestation clears
        already sees the numbers recovered while it was UNKNOWN."""
        from director import handler as H
        order: list[str] = []
        monkeypatch.setattr(
            H, "backfill_issue_numbers",
            lambda items, repo=None, token=None: (order.append("backfill"), 1)[1],
        )
        monkeypatch.setattr(
            H, "verify_and_correct",
            lambda items, card, repo=None, token=None: (order.append("verify"), {})[1],
        )
        H._verify_loop_best_effort({"items": []}, _card(), "tok", verdict_block=block)
        assert order == ["backfill"], "the gate must not sit upstream of the backfill"

    def test_a_failing_backfill_does_not_sink_the_gated_half(self, monkeypatch):
        """The two halves fail independently — fail-loud, no silent swallow, and
        no shared try block that lets an own-ledger error suppress the
        corrections a PASS verdict authorized."""
        from director import handler as H

        def _boom(items, repo=None, token=None):
            raise RuntimeError("github 502")

        monkeypatch.setattr(H, "backfill_issue_numbers", _boom)
        monkeypatch.setattr(
            H, "verify_and_correct",
            lambda items, card, repo=None, token=None: {"open": 3},
        )
        out = H._verify_loop_best_effort(
            {"items": []}, _card(), "tok", verdict_block=V.read_card_verdict(_card(_PASS_BLOCK))
        )
        assert out["director_loop"] == "ok"
        assert out["director_loop_open"] == 3
        assert "github 502" in out["director_loop_backfill_error"]

    def test_gate_precedes_the_enable_flag(self, monkeypatch):
        # An unverified cycle must read as WITHHELD, never as "disabled" — the
        # two are different findings and only one of them is a problem.
        from director import handler as H
        monkeypatch.setattr(H, "_issue_filing_enabled", lambda: False)
        out = H._file_issues_best_effort(object(), RUN_DATE, "tok", verdict_block={})
        assert out["director_issues"] == "withheld"

    def test_withheld_summary_names_what_stopped(self):
        s = V.withheld_summary(V.read_card_verdict(_card()))
        assert s["correctness_verdict"] == "UNKNOWN"
        assert s["director_actions_withheld"] is True
        assert set(s["director_actions_withheld_list"]) == set(V.GATED_ACTIONS)


# ════════════════════════════════════════════════════════════════════════════
class TestMutationClassSplit:
    """alpha-engine-config-I8187 — the gate's subject is the AUTHORITY an action
    exercises, never a global run-quality flag."""

    def test_the_two_sets_are_disjoint_and_loop_verification_is_in_neither(self):
        assert not set(V.MUTATING_ACTIONS) & set(V.UNGATED_ACTIONS)
        # The atom this ruling dissolved. A gate named after a code path grows
        # to cover whatever that path later does; one named after an authority
        # does not.
        assert "loop_verification" not in V.MUTATING_ACTIONS + V.UNGATED_ACTIONS
        assert V.GATED_ACTIONS == V.MUTATING_ACTIONS

    @pytest.mark.parametrize("verdict", ["UNKNOWN", "FAIL", "PARTIAL", "PASS"])
    @pytest.mark.parametrize("action", ["issue_number_backfill", "carryover_reconciliation"])
    def test_an_ungated_action_is_never_withheld_under_any_verdict(self, verdict, action):
        block = V.read_card_verdict(_card({**_PASS_BLOCK, "verdict": verdict}))
        assert V.actions_withheld(block, action=action) is False

    @pytest.mark.parametrize("action", ["issue_filing", "issue_reopen", "carryover_escalation"])
    def test_a_mutating_action_is_withheld_on_every_non_pass_verdict(self, action):
        for verdict in ("UNKNOWN", "FAIL", "PARTIAL"):
            block = V.read_card_verdict(_card({**_PASS_BLOCK, "verdict": verdict}))
            assert V.actions_withheld(block, action=action) is True
        clean = V.read_card_verdict(_card(_PASS_BLOCK))
        assert V.actions_withheld(clean, action=action) is False

    def test_an_undeclared_action_raises_rather_than_guessing(self):
        # Both silent defaults are wrong: False ships a new mutating action
        # ungated on a typo, True reinstates the over-broad withholding this
        # split removed. Fail loud — the fleet's default is RAISE.
        with pytest.raises(ValueError, match="unknown Director action"):
            V.actions_withheld(V.read_card_verdict(_card()), action="loop_verification")
        with pytest.raises(ValueError):
            V.is_gated_action("issue_filng")

    def test_the_class_question_is_unchanged_by_the_split(self):
        assert V.actions_withheld(V.read_card_verdict(_card())) is True
        assert V.actions_withheld(V.read_card_verdict(_card(_PASS_BLOCK))) is False

    def test_both_sides_of_the_split_are_named_on_every_surface(self):
        # A list appearing only when something stopped cannot be distinguished
        # from a producer that stopped emitting — so the ran-regardless set is
        # emitted in BOTH verdict states.
        for block in (_card(), _card(_PASS_BLOCK)):
            s = V.withheld_summary(V.read_card_verdict(block))
            assert set(s["director_actions_ungated_list"]) == set(V.UNGATED_ACTIONS)

    def test_the_plan_artifact_names_what_still_ran(self):
        block = V.read_card_verdict(_card())
        body = json.loads(V.stamp_plan_artifact({"action_items": []}, block))
        assert set(body["actions_withheld"]) == set(V.MUTATING_ACTIONS)
        assert set(body["actions_ran_regardless"]) == set(V.UNGATED_ACTIONS)
        assert "loop_verification" not in body["actions_withheld"]


# ════════════════════════════════════════════════════════════════════════════
class TestPassRestoresAuthority:
    """The gate is default-deny, not always-closed."""

    def test_pass_lets_issue_filing_run(self, monkeypatch):
        from director import handler as H
        calls = []
        monkeypatch.setattr(H, "_issue_filing_enabled", lambda: True)
        monkeypatch.setattr(
            H, "file_director_issues",
            lambda plan, rd, token=None: calls.append(rd) or {"status": "ok", "n_filed": 2,
                                                              "issues": []},
        )
        block = V.read_card_verdict(_card(_PASS_BLOCK))
        out = H._file_issues_best_effort(object(), RUN_DATE, "tok", verdict_block=block)
        assert out["director_issues"] == "ok"
        assert calls == [RUN_DATE]

    def test_pass_summary_carries_the_as_of(self):
        s = V.withheld_summary(V.read_card_verdict(_card(_PASS_BLOCK)))
        assert s["correctness_verdict"] == "PASS"
        assert s["director_actions_withheld"] is False
        # A verdict with no timestamp cannot read as stale, which is the failure
        # mode one layer up from reading absence as green.
        assert s["correctness_as_of"]["backtester"] == "2026-08-15T09:41:02Z"
        assert s["correctness_as_of"]["evaluator_stage"] == "2026-08-15T10:02:55Z"


# ════════════════════════════════════════════════════════════════════════════
class TestSurfaces:
    """§2.3a rule 3 — every surface presenting the run's results carries it."""

    def test_plan_artifact_carries_the_verdict(self):
        block = V.read_card_verdict(_card(_PASS_BLOCK))
        body = json.loads(V.stamp_plan_artifact({"run_date": RUN_DATE, "top_risks": []}, block))
        assert body["attestation"]["verdict"] == "PASS"
        assert body["advisory_unverified"] is False
        assert body["run_date"] == RUN_DATE  # the plan itself is preserved

    def test_plan_artifact_marks_an_unverified_advisory(self):
        block = V.read_card_verdict(_card())
        body = json.loads(V.stamp_plan_artifact({"run_date": RUN_DATE}, block))
        assert body["advisory_unverified"] is True
        assert "§2.3a" in body["advisory_unverified_reason"]
        assert set(body["actions_withheld"]) == set(V.GATED_ACTIONS)

    def test_plan_model_is_not_asked_to_produce_its_own_verdict(self):
        # The plan model is the LLM's structured-output schema. A correctness
        # verdict generated by the thing being verified is not a verdict.
        from director.schema import DirectorWeeklyActionPlan
        assert "attestation" not in DirectorWeeklyActionPlan.model_fields

    def test_digest_leads_with_the_verdict_when_withheld(self):
        block = V.read_card_verdict(_card())
        subject, plain, html = build_director_digest(
            {"run_date": RUN_DATE, "system_summary": "s", "top_risks": [], "action_items": []},
            RUN_DATE, verdict_block=block,
        )
        assert subject.startswith("[UNVERIFIED] ")
        # Above the console link: a reader who clicks straight through must
        # have seen it first.
        assert plain.index("CORRECTNESS ATTESTATION") < plain.index("View the full")
        assert "did NOT file issues" in plain
        assert "CORRECTNESS ATTESTATION" in html

    def test_digest_subject_distinguishes_fail_from_unknown(self):
        block = V.read_card_verdict(_card({**_PASS_BLOCK, "verdict": "FAIL"}))
        subject, _, _ = build_director_digest(
            {"run_date": RUN_DATE, "action_items": []}, RUN_DATE, verdict_block=block,
        )
        assert subject.startswith("[NUMBERS WRONG] ")

    def test_digest_states_the_verdict_even_on_pass(self):
        # A surface that states the verdict only when it is bad is a surface
        # where silence means pass — and silence is what absence produces.
        block = V.read_card_verdict(_card(_PASS_BLOCK))
        subject, plain, html = build_director_digest(
            {"run_date": RUN_DATE, "action_items": []}, RUN_DATE, verdict_block=block,
        )
        assert not subject.startswith("[")
        assert "Correctness attestation: PASS" in plain
        assert "2026-08-15T09:41:02Z" in plain
        assert "Correctness attestation: PASS" in html


# ════════════════════════════════════════════════════════════════════════════
class TestEndToEnd:
    """The whole handler path, through moto — the property as it actually ships."""

    @pytest.fixture
    def s3(self):
        with mock_aws():
            c = boto3.client("s3", region_name="us-east-1")
            c.create_bucket(Bucket=BUCKET)
            yield c

    def _run(self, s3, monkeypatch, card, *, mutators=None):
        """Drive the real handler. ``mutators`` replaces the three GitHub-mutating
        calls; the default is a tripwire set, so any test that does not opt in
        FAILS if a mutation runs."""
        from director import handler as H
        from director.schema import ActionItem, DirectorWeeklyActionPlan
        monkeypatch.setenv("DIRECTOR_ENABLED", "1")
        monkeypatch.setattr(H, "_ensure_registry", lambda *a, **kw: None)
        monkeypatch.setattr(H, "_ensure_litellm_credential", lambda: None)
        monkeypatch.setattr(H, "_director_github_token", lambda: "tok")
        monkeypatch.setattr(H, "_issue_filing_enabled", lambda: True)
        for name, fn in (mutators or {}).items():
            monkeypatch.setattr(H, name, fn)
        for name in ("file_director_issues", "backfill_issue_numbers", "verify_and_correct"):
            if name not in (mutators or {}):
                monkeypatch.setattr(H, name, _Tripwire())
        monkeypatch.setattr(
            H, "build_action_plan",
            lambda card, **kw: DirectorWeeklyActionPlan(
                run_date=RUN_DATE, system_summary="s", top_risks=[],
                action_items=[ActionItem(
                    id="a", title="t", rationale="r", evidence=["e"],
                    priority="P1", horizon="this_week", proposed_owner="predictor",
                    suggested_change_type="investigation", confidence=50,
                )],
            ),
        )
        # The handler resolves its own run_date (last closed trading day), so the
        # card must be seeded at the key IT will read — deriving it rather than
        # assuming, which is how the same test would have silently passed on a
        # Saturday and failed on a Monday.
        self.resolved = H._resolve_run_date({"date": RUN_DATE})
        s3.put_object(Bucket=BUCKET, Key=f"evaluator/{self.resolved}/report_card.json",
                      Body=json.dumps(card).encode())
        return H.handler({"date": RUN_DATE, "bucket": BUCKET})

    def test_unattested_run_still_produces_a_plan_but_acts_on_nothing(self, s3, monkeypatch):
        # §2.3a: withholding the guarantee is NOT failing the pipeline. An
        # UNKNOWN here is frequently just a spot reclaim on the box that ran the
        # backtest; killing the weekly advisory on that trades one blindness for
        # an outage. The plan ships — marked — and the mutations stop.
        out = self._run(s3, monkeypatch, _card())

        assert out["status"] == "ok"
        assert out["correctness_verdict"] == "UNKNOWN"
        assert out["director_actions_withheld"] is True
        assert out["director_issues"] == "withheld"
        # I8187: the pass no longer goes dark wholesale — only its mutations do.
        assert out["director_loop"] == "mutations_withheld"

        body = json.loads(
            s3.get_object(Bucket=BUCKET,
                          Key=f"director/{self.resolved}/action_plan.json")["Body"].read()
        )
        assert body["advisory_unverified"] is True
        assert body["attestation"]["verdict"] == "UNKNOWN"
        assert len(body["action_items"]) == 1, "the advisory itself must survive"

    def test_attested_run_keeps_its_authority(self, s3, monkeypatch):
        filed: list[str] = []
        out = self._run(s3, monkeypatch, _card(_PASS_BLOCK), mutators={
            "file_director_issues": (
                lambda plan, rd, token=None: (filed.append(rd),
                                              {"status": "ok", "n_filed": 1, "issues": []})[1]
            ),
            "backfill_issue_numbers": lambda items, repo=None, token=None: 0,
            "verify_and_correct": lambda items, card, repo=None, token=None: {"open": 0},
        })
        assert out["correctness_verdict"] == "PASS"
        assert out["director_actions_withheld"] is False
        assert out["director_issues"] == "ok"
        assert filed == [self.resolved], "an attested cycle must retain its acting authority"

        body = json.loads(
            s3.get_object(Bucket=BUCKET,
                          Key=f"director/{self.resolved}/action_plan.json")["Body"].read()
        )
        assert body["advisory_unverified"] is False
        assert body["attestation"]["as_of"]["backtester"] == "2026-08-15T09:41:02Z"
