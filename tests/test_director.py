"""Tests for the Director agent (Layer C) — Phase E. No LLM key / langchain
required: the LLM is injected, and the handler's plan-build is monkeypatched."""

import hashlib
import json
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from director.agent import _carryover_context, build_action_plan, build_messages
from director.loop_verification import evidence_still_adverse, resolve_cited_metrics
from director.carryover import load_ledger, merge_plan_into_ledger
from director.report_card_digest import summarize_report_card
from director.schema import ActionItem, DirectorWeeklyActionPlan

BUCKET = "alpha-engine-research"
# A TRADING day (Fri) so _resolve_run_date's trading-day normalization is a
# no-op (was "2026-05-30", a Saturday → would normalize to Fri 05-29 and break
# the seed/lookup match). Calendar→trading normalization is covered in
# tests/test_handler.py::TestResolveRunDate.
RUN_DATE = "2026-05-29"

#: A PASSING §2.3a correctness verdict, in the shape
#: ``grading/attestation.py::build_run_attestation`` emits.
#:
#: This is DELIBERATELY explicit on the happy-path card rather than defaulted in
#: the handler (config-I7039). The Director's acting authority is gated on the
#: verdict, and the gate is DEFAULT-DENY: a card with no attestation block reads
#: as UNKNOWN and withholds. Every test below that exercises issue filing or the
#: reopen/escalate loop therefore has to state that this cycle was attested —
#: which is the property under test made structural. A future change that drops
#: the verdict read would not quietly re-enable those paths; it would have to
#: delete this block from the fixture first.
_ATTESTATION_PASS = {
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

_CARD = {
    "attestation": _ATTESTATION_PASS,
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
        # alpha-engine-config-I6187: S3 is the sole registry delivery path,
        # and `_ensure_registry` now raises when it is absent — seed a
        # placeholder registry object so every test using this fixture
        # reflects a healthy bucket by default. Tests exercising the failure
        # path itself (TestEnsureRegistryFailsLoud) delete this object first.
        client.put_object(Bucket=BUCKET, Key="director/LLM_MODEL_REGISTRY.yaml",
                          Body=b"groups: {}\n")
        yield client


#: A minimal card carrying the one component the reconciliation tests assert
#: on. Kept separate from ``_CARD`` so widening it never perturbs the digest
#: assertions that read ``_CARD`` verbatim.
_CARD_FOR_RECONCILE = {
    "_provenance": {"run_date": RUN_DATE},
    "tiles_overall_status": "RED",
    "tiles": {"predictor": {"components": [
        {"name": "inference_coverage", "criticality": "critical", "status": "GREEN", "value": 1.0},
    ]}},
}


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

    def test_unpinned_green_is_counted_not_named(self):
        # Baseline: with no pinned_components, sharpe_ratio (GREEN) is only
        # counted in the tile head, never named/expanded — the defect
        # alpha-engine-config-I8380 is about.
        text = summarize_report_card(_CARD)
        assert "1 GREEN" in text  # portfolio_outcome tile head count
        assert "sharpe_ratio" not in text

    def test_pinned_green_component_is_named_with_value(self):
        # alpha-engine-config-I8380 closes-when: a GREEN component named by an
        # open carry-over item must appear, by name and value, in the digest.
        text = summarize_report_card(_CARD, pinned_components={"sharpe_ratio"})
        assert "sharpe_ratio" in text
        assert "value 1.2" in text
        assert "carryover-pinned" in text
        # Still counted in the head alongside being named.
        assert "1 GREEN" in text

    def test_pinned_na_component_is_named(self):
        text = summarize_report_card(_CARD, pinned_components={"dsr"})
        assert "  - dsr " in text
        assert "carryover-pinned" in text
        # No longer folded into the bare N/A×1 rollup line.
        assert "N/A-NOT-IMPL×1" not in text

    def test_pinned_adverse_component_not_duplicated(self):
        # A pinned name that is already RED/WATCH is expanded once (by the
        # adverse loop), not twice.
        text = summarize_report_card(_CARD, pinned_components={"information_ratio"})
        assert text.count("information_ratio") == 1

    def test_absent_from_card_note_present(self):
        text = summarize_report_card(_CARD)
        assert "'Not visible on this card' is true ONLY for a metric absent" in text


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

    def test_build_messages_pins_carryover_cited_green_component(self):
        # alpha-engine-config-I8380 — a carry-over item citing a component
        # that has since recovered to GREEN must have that component named
        # (not just counted) in the digest half of the human message.
        msgs = build_messages(
            _CARD,
            carryover={"items": [{
                "id": "watch-sharpe", "title": "Watch sharpe_ratio recovery",
                "rationale": "sharpe_ratio was RED, monitor for recovery.",
                "evidence": ["sharpe_ratio"], "status": "carried_over",
            }]},
        )
        human = msgs[1][1]
        assert "sharpe_ratio" in human
        assert "value 1.2" in human
        assert "carryover-pinned" in human

    def test_build_action_plan_injected_llm(self):
        plan = build_action_plan(_CARD, llm=_FakeLLM(_plan()))
        assert plan.run_date == RUN_DATE
        assert plan.action_items[0].id == "revive-momentum-l1"

    def test_run_date_stamped_from_card(self):
        p = _plan(); p.run_date = ""
        plan = build_action_plan(_CARD, llm=_FakeLLM(p))
        assert plan.run_date == RUN_DATE  # stamped from provenance

    def test_retry_then_succeed(self, monkeypatch):
        llm = _FakeLLM(_plan(), fail_times=1, exc=RuntimeError("overloaded_error"))
        import director.agent as A
        # `A.time` IS the stdlib `time` module, so a bare assignment here
        # disabled `time.sleep` PROCESS-WIDE for every test that ran afterwards
        # and never restored it — a cross-file leak that made any later
        # timing-dependent assertion silently vacuous. `monkeypatch` restores it
        # at teardown; the speed-up is unchanged.
        monkeypatch.setattr(A.time, "sleep", lambda *_: None)
        plan = build_action_plan(_CARD, llm=llm)
        assert llm.calls == 2 and plan.run_date == RUN_DATE

    def test_non_transient_raises(self):
        llm = _FakeLLM(_plan(), fail_times=5, exc=ValueError("bad schema"))
        with pytest.raises(ValueError):
            build_action_plan(_CARD, llm=llm)


#: The per-row marker, distinct from the section header's mention of the word
#: (asserting on the bare word matched the instruction line and passed
#: vacuously — caught by these tests failing on first run).
_MARKER = "\u26a0 CONTRADICTED BY THIS WEEK'S CARD"


class TestCarryoverLiveCardReconciliation:
    """alpha-engine-config-I8178 — a carry-over row is reconciled against the
    CURRENT card before the Director reasons from it.

    The 2026-08-22 case these tests pin: the ledger row
    ``inference-coverage-critical`` was titled "Fix inference_coverage
    collapse — predictor Lambda KeyError (#7289)". On the card the Director
    read, ``inference_coverage`` was 1.0 / GREEN; ``#7289`` had been closed 8
    days. It shipped as P0 #4 anyway, because nothing on the path between the
    ledger and the model compared the row to the numbers."""

    _CARD_WITH = {
        "tiles": {
            "predictor": {"components": [
                {"name": "inference_coverage", "status": "GREEN"},
                {"name": "momentum_l1_ic", "status": "RED"},
            ]},
            "substrate": {"components": [{"name": "sf_success_rate_4w", "status": "WATCH"}]},
        }
    }

    def _row(self, **kw):
        row = {"id": "x", "title": "t", "status": "carried_over",
               "proposed_owner": "predictor", "priority": "P0",
               "evidence": [], "rationale": ""}
        row.update(kw)
        return row

    def test_metric_named_only_in_the_title_is_still_reconciled(self):
        """The measured miss. The row's ``evidence`` was
        ``["predictor tile N/A entries (...)"]`` — no metric name anywhere in
        it. The refuted claim lived in the TITLE, which is also the only part
        of the row the prompt had ever rendered."""
        led = {"items": [self._row(
            id="inference-coverage-critical",
            title="Fix inference_coverage collapse \u2014 predictor Lambda KeyError (#7289)",
            evidence=["predictor tile N/A entries (N/A-MISSING-INPUT\u00d72, N/A-LOW-N\u00d71)"],
        )]}
        out = _carryover_context(led, self._CARD_WITH)
        assert "inference_coverage=GREEN" in out
        assert _MARKER in out

    def test_prose_citation_resolves_to_the_component(self):
        """26 of 28 live rows cited their metric as ``"<tile> tile
        <metric>"``. A whole-string lookup matched none of them."""
        led = {"items": [self._row(id="m", title="hold", priority="P2",
                                   evidence=["predictor tile momentum_l1_ic"])]}
        out = _carryover_context(led, self._CARD_WITH)
        assert "momentum_l1_ic=RED" in out
        assert _MARKER not in out

    def test_watch_is_adverse_so_a_watch_row_is_not_contradicted(self):
        """``ADVERSE_STATUSES`` is ``{RED, WATCH}``. A row citing a WATCH
        metric is still live work; only all-non-adverse contradicts."""
        led = {"items": [self._row(id="s", evidence=["substrate tile sf_success_rate_4w"])]}
        out = _carryover_context(led, self._CARD_WITH)
        assert "sf_success_rate_4w=WATCH" in out
        assert _MARKER not in out

    def test_row_naming_no_card_metric_is_marked_unverifiable_not_contradicted(self):
        """Absence of evidence is not evidence of recovery — the same
        asymmetry ``evidence_still_adverse`` already refuses to cross."""
        led = {"items": [self._row(id="nothing", title="something untraceable")]}
        out = _carryover_context(led, self._CARD_WITH)
        assert "UNVERIFIABLE" in out
        assert _MARKER not in out

    def test_contradicted_rows_are_annotated_never_dropped(self):
        """A GREEN metric does not prove the item is finished — the row may
        name work the metric does not cover. Suppressing it would be the
        close-and-look-away failure loop_verification exists to stop."""
        led = {"items": [self._row(id="inference-coverage-critical",
                                   title="Fix inference_coverage collapse")]}
        out = _carryover_context(led, self._CARD_WITH)
        assert "inference-coverage-critical" in out
        assert "carry-over ledger, active items=1" in out

    def test_no_card_renders_exactly_as_before(self):
        """Back-compat: every existing caller that passes no card gets the
        unannotated rendering, and no annotation is invented from nothing."""
        led = {"items": [self._row(id="a", title="Fix inference_coverage collapse")]}
        assert "live card:" not in _carryover_context(led)
        assert _MARKER not in _carryover_context(led)

    def test_build_messages_threads_the_card_into_the_annotation(self):
        """The wiring, not just the helper — the defect was that the card and
        the ledger were both in hand at this exact call site and were never
        compared."""
        led = {"items": [self._row(id="inference-coverage-critical",
                                   title="Fix inference_coverage collapse")]}
        human = build_messages(_CARD_FOR_RECONCILE, carryover=led)[1][1]
        assert "inference_coverage=GREEN" in human
        assert _MARKER in human

    def test_counts_contradicted_rows_in_the_footer(self):
        led = {"items": [
            self._row(id="a", title="Fix inference_coverage collapse"),
            self._row(id="b", title="hold momentum_l1_ic"),
        ]}
        out = _carryover_context(led, self._CARD_WITH)
        assert "1 of 2 carry-over rows shown are contradicted" in out


class TestCitedMetricResolution:
    """The detection blindness itself (alpha-engine-config-I8178): the check
    ran, returned an answer, and was structurally incapable of returning
    anything but "can't tell" on 93% of the ledger."""

    _MAP = {"momentum_l1_ic": "RED", "scanner": "WATCH",
            "scanner_basket_return": "GREEN", "inference_coverage": "GREEN"}

    def test_prose_citation_resolves(self):
        assert resolve_cited_metrics(["predictor tile momentum_l1_ic"], self._MAP) == {
            "momentum_l1_ic": "RED"}

    def test_substring_does_not_match_a_sibling(self):
        """``scanner`` IS a live component name. Substring containment would
        fire it on every ``scanner_*`` sibling; exact token equality does not."""
        assert resolve_cited_metrics(["research tile scanner_basket_return"], self._MAP) == {
            "scanner_basket_return": "GREEN"}

    def test_bare_name_still_resolves(self):
        """The 2 of 28 rows that DID cite bare names must not regress."""
        assert resolve_cited_metrics(["scanner"], self._MAP) == {"scanner": "WATCH"}

    def test_unrelated_prose_resolves_to_nothing(self):
        assert resolve_cited_metrics(["CORRECTNESS ATTESTATION tile"], self._MAP) == {}

    def test_evidence_still_adverse_now_sees_prose(self):
        assert evidence_still_adverse(["predictor tile momentum_l1_ic"], self._MAP) == "adverse"
        assert evidence_still_adverse(["predictor tile inference_coverage"], self._MAP) == "recovered"
        assert evidence_still_adverse(["predictor tile N/A entries"], self._MAP) == "unverifiable"


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

    def test_writes_latest_pointer_alongside_dated_key(self, s3, monkeypatch):
        # alpha-engine-config-I7157 option (a): every enabled run also
        # overwrites the standing `director/latest/action_plan.json` pointer,
        # mirroring `grading/aggregate.py::write_report_card`'s
        # `evaluator/latest/report_card.json` — a fixed key the console's
        # document-fields binding can read without templating on run_date.
        monkeypatch.setenv("DIRECTOR_ENABLED", "1")
        s3.put_object(Bucket=BUCKET, Key=f"evaluator/{RUN_DATE}/report_card.json",
                      Body=json.dumps(_CARD).encode())
        from director import handler as H
        monkeypatch.setattr(H, "build_action_plan", lambda card, **kw: _plan())
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET})
        assert out["status"] == "ok"
        assert out["latest_action_plan_key"] == "director/latest/action_plan.json"
        assert H.latest_action_plan_key() == "director/latest/action_plan.json"
        dated = json.loads(s3.get_object(Bucket=BUCKET, Key=out["action_plan_key"])["Body"].read())
        latest = json.loads(s3.get_object(Bucket=BUCKET, Key=out["latest_action_plan_key"])["Body"].read())
        # Same stamped body (including the §2.3a verdict block) on both keys.
        assert latest == dated

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

    def test_check_deploy_drift_dispatches_before_enabled_flag(self, s3, monkeypatch):
        # config#2348: the drift probe must run even when DIRECTOR_ENABLED is
        # off (the default) — a dormant-but-stale image is still stale.
        # `s3` (mock_aws) so the stage-coverage call this dispatch also makes
        # (config-I7214/I7334) resolves against moto rather than reaching
        # real AWS — krepis.stage_coverage is genuinely importable now, so
        # this exercises the REAL library call, not a fake.
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
        assert out["has_drift"] is True
        assert out["function_name"] == "alpha-engine-evaluator-director"
        assert out["stage_coverage"]["stage"] == "EvaluatorDirectorDeployDriftCheck"

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
        monkeypatch.setattr(H, "_fetch_backlog_digest_best_effort", lambda tok, **kw: "DIGEST")
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


class TestEnsureRegistryFailsLoud:
    """alpha-engine-config-I6187: with the AppConfig Tier 2 fallback removed,
    S3 is the ONLY registry delivery path — a failed download must raise, not
    warn-and-continue. Regression guard for the prior best-effort behavior
    (`_ensure_registry` used to log a WARNING and return, leaving
    `LLM_MODEL_REGISTRY_PATH` unset and the run to silently resolve against
    whatever a lower tier found)."""

    def test_missing_registry_object_raises(self, s3, monkeypatch):
        monkeypatch.delenv("LLM_MODEL_REGISTRY_PATH", raising=False)
        s3.delete_object(Bucket=BUCKET, Key="director/LLM_MODEL_REGISTRY.yaml")
        # `_ensure_registry` skips the S3 round-trip entirely if
        # /tmp/LLM_MODEL_REGISTRY.yaml already exists on disk — clear any
        # leftover from an earlier test in this same process so the missing
        # S3 object is actually exercised.
        dest = __import__("pathlib").Path("/tmp/LLM_MODEL_REGISTRY.yaml")
        dest.unlink(missing_ok=True)
        from director import handler as H
        # No `director/LLM_MODEL_REGISTRY.yaml` object in the bucket — the
        # download must fail with a real ClientError (NoSuchKey), and that
        # must now RAISE rather than log a WARNING and return.
        with pytest.raises(RuntimeError, match="LLM_MODEL_REGISTRY.yaml"):
            H._ensure_registry(BUCKET, s3)
        assert "LLM_MODEL_REGISTRY_PATH" not in __import__("os").environ

    def test_handler_propagates_the_registry_failure(self, s3, monkeypatch):
        # End-to-end: a live-enabled run with a report card present but no
        # registry object must fail the whole invocation loudly (the repo's
        # fail-loud-by-design convention — an SF Catch, not a swallow here,
        # is what makes this non-fatal to the weekly run).
        monkeypatch.setenv("DIRECTOR_ENABLED", "1")
        monkeypatch.delenv("LLM_MODEL_REGISTRY_PATH", raising=False)
        s3.delete_object(Bucket=BUCKET, Key="director/LLM_MODEL_REGISTRY.yaml")
        dest = __import__("pathlib").Path("/tmp/LLM_MODEL_REGISTRY.yaml")
        dest.unlink(missing_ok=True)
        s3.put_object(Bucket=BUCKET, Key=f"evaluator/{RUN_DATE}/report_card.json",
                      Body=json.dumps(_CARD).encode())
        from director import handler as H
        with pytest.raises(RuntimeError, match="LLM_MODEL_REGISTRY.yaml"):
            H.handler({"date": RUN_DATE, "bucket": BUCKET})

    def test_present_registry_object_sets_the_env_var(self, s3, monkeypatch):
        monkeypatch.delenv("LLM_MODEL_REGISTRY_PATH", raising=False)
        from director import handler as H
        # /tmp/LLM_MODEL_REGISTRY.yaml persists across tests in the same
        # process — clear it so this test proves the download path, not a
        # leftover file from an earlier test/run.
        dest = __import__("pathlib").Path("/tmp/LLM_MODEL_REGISTRY.yaml")
        dest.unlink(missing_ok=True)
        try:
            H._ensure_registry(BUCKET, s3)
            assert __import__("os").environ["LLM_MODEL_REGISTRY_PATH"] == str(dest)
        finally:
            dest.unlink(missing_ok=True)


class TestRegistryProvenance:
    """alpha-engine-config-I8332: the action plan stamps WHICH
    LLM_MODEL_REGISTRY.yaml it was built against, alongside `attestation`."""

    _REG_BODY = b"groups: {fast: {model: sonnet}}\n"

    def _reset(self, monkeypatch):
        # `_REGISTRY_FINGERPRINT` is a module global (deliberately — it survives
        # a warm Lambda container the same way LLM_MODEL_REGISTRY_PATH does), so
        # each test resets both it and the env var + /tmp copy to avoid leaking
        # state across tests in this same pytest process.
        from director import handler as H
        monkeypatch.delenv("LLM_MODEL_REGISTRY_PATH", raising=False)
        monkeypatch.setattr(H, "_REGISTRY_FINGERPRINT", None)
        Path("/tmp/LLM_MODEL_REGISTRY.yaml").unlink(missing_ok=True)
        return H

    def test_ensure_registry_captures_sha256(self, s3, monkeypatch):
        H = self._reset(monkeypatch)
        s3.put_object(Bucket=BUCKET, Key="director/LLM_MODEL_REGISTRY.yaml",
                      Body=self._REG_BODY)
        H._ensure_registry(BUCKET, s3)
        fp = H.registry_fingerprint()
        assert fp["sha256"] == hashlib.sha256(self._REG_BODY).hexdigest()
        assert fp["source_path"] == f"s3://{BUCKET}/director/LLM_MODEL_REGISTRY.yaml"
        # moto's default bucket has no versioning enabled — VersionId is
        # legitimately absent; LastModified is always present.
        assert fp["as_of"] is not None

    def test_fingerprint_absent_before_ensure_registry_runs(self, monkeypatch):
        H = self._reset(monkeypatch)
        fp = H.registry_fingerprint()
        assert fp == {"source_path": None, "sha256": None, "as_of": None, "version_id": None}

    def test_stamp_reflects_bytes_loaded_not_a_later_reread(self, s3, monkeypatch):
        # The sha256 on the artifact must be the hash of the bytes actually
        # used to resolve model groups this run — not whatever /tmp holds at
        # serialization time. Mutate /tmp AND the S3 object after
        # `_ensure_registry` returns; the captured fingerprint must not move.
        H = self._reset(monkeypatch)
        s3.put_object(Bucket=BUCKET, Key="director/LLM_MODEL_REGISTRY.yaml",
                      Body=self._REG_BODY)
        H._ensure_registry(BUCKET, s3)
        original_sha = H.registry_fingerprint()["sha256"]

        # Mutate the local copy krepis will have already read, and the S3
        # object, to a different body.
        Path("/tmp/LLM_MODEL_REGISTRY.yaml").write_bytes(b"groups: {tampered: true}\n")
        s3.put_object(Bucket=BUCKET, Key="director/LLM_MODEL_REGISTRY.yaml",
                      Body=b"groups: {tampered: true}\n")

        fp_after = H.registry_fingerprint()
        assert fp_after["sha256"] == original_sha
        assert fp_after["sha256"] != hashlib.sha256(b"groups: {tampered: true}\n").hexdigest()

    def test_action_plan_carries_registry_provenance_on_both_keys(self, s3, monkeypatch):
        H = self._reset(monkeypatch)
        # Override the fixture's placeholder registry body with a known one so
        # the sha256 assertion below is exact.
        s3.put_object(Bucket=BUCKET, Key="director/LLM_MODEL_REGISTRY.yaml",
                      Body=self._REG_BODY)
        monkeypatch.setenv("DIRECTOR_ENABLED", "1")
        s3.put_object(Bucket=BUCKET, Key=f"evaluator/{RUN_DATE}/report_card.json",
                      Body=json.dumps(_CARD).encode())
        monkeypatch.setattr(H, "build_action_plan", lambda card, **kw: _plan())

        out = H.handler({"date": RUN_DATE, "bucket": BUCKET})
        assert out["status"] == "ok"

        expected_sha = hashlib.sha256(self._REG_BODY).hexdigest()
        dated = json.loads(s3.get_object(Bucket=BUCKET, Key=out["action_plan_key"])["Body"].read())
        latest = json.loads(s3.get_object(Bucket=BUCKET, Key=out["latest_action_plan_key"])["Body"].read())

        for body in (dated, latest):
            assert "registry_provenance" in body
            assert body["registry_provenance"]["sha256"] == expected_sha
            assert body["registry_provenance"]["source_path"] == \
                f"s3://{BUCKET}/director/LLM_MODEL_REGISTRY.yaml"
            # Sits beside `attestation`, not in place of it.
            assert "attestation" in body

        # Both keys share the exact same stamped body (registry_provenance
        # included) — same assertion style as
        # test_writes_latest_pointer_alongside_dated_key.
        assert dated == latest


# ── Stage-output coverage (config-I7214, config-I7334) ────────────────────────
#
# The single shared implementation is `krepis.stage_coverage` (relocated from
# `nousergon_lib.stage_coverage`, which never shipped on any released
# nousergon-lib version — config-I7334). These tests inject a fake module into
# sys.modules so the call sites can be exercised without a real krepis
# resolve; `TestStageCoverageImportDegrades` below separately proves the
# import-failure degrade path (the REAL live behavior before I7334's pin bump).

def _install_fake_stage_coverage(monkeypatch, calls=None):
    """Inject a stand-in `krepis.stage_coverage` module so the handler's
    `from krepis.stage_coverage import assert_stage_coverage` resolves
    to a controllable fake (Python's import system checks sys.modules before
    any finder/loader lookup, so this needs no real submodule on disk)."""
    import types
    calls = calls if calls is not None else []
    fake_mod = types.ModuleType("krepis.stage_coverage")

    def _fake_assert_stage_coverage(stage, *, run_date, window_start):
        calls.append({"stage": stage, "run_date": run_date, "window_start": window_start})
        return {"stage": stage, "status": "COVERED", "run_date": run_date}

    fake_mod.assert_stage_coverage = _fake_assert_stage_coverage
    monkeypatch.setitem(__import__("sys").modules, "krepis.stage_coverage", fake_mod)
    return calls


def _block_stage_coverage_import(monkeypatch):
    """Force `from krepis.stage_coverage import assert_stage_coverage` to
    raise ImportError regardless of whether the real module is actually
    resolvable in this test environment — the module being absent from
    sys.modules is not sufficient to prove an import failure once krepis
    provides it (I7334 fixed exactly that gap)."""
    import builtins
    import sys

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "krepis.stage_coverage":
            raise ImportError("No module named 'krepis.stage_coverage'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "krepis.stage_coverage", raising=False)
    monkeypatch.setattr(builtins, "__import__", _fake_import)


class TestStageCoverageDirector:
    """(director Lambda, work dispatch) — the Director SF stage. Covers all
    three internal branches (disabled / dry_run / enabled-real) since they
    all map to the SAME SF Task, never hardcoding a stage name."""

    def test_disabled_branch_verdict_lands_with_correct_stage_name(self, s3, monkeypatch):
        monkeypatch.delenv("DIRECTOR_ENABLED", raising=False)
        calls = _install_fake_stage_coverage(monkeypatch)
        from director import handler as H
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET})
        assert out["status"] == "disabled"
        assert out["stage_coverage"] == {"stage": "Director", "status": "COVERED", "run_date": RUN_DATE}
        assert [c["stage"] for c in calls] == ["Director"]

    def test_dry_run_branch_verdict_lands_with_correct_stage_name(self, s3, monkeypatch):
        monkeypatch.setenv("DIRECTOR_ENABLED", "1")
        import director.agent as A
        monkeypatch.setattr(A, "_default_llm", lambda: object())
        calls = _install_fake_stage_coverage(monkeypatch)
        from director import handler as H
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET, "dry_run": True})
        assert out["status"] == "dry_run"
        assert out["stage_coverage"]["stage"] == "Director"
        assert [c["stage"] for c in calls] == ["Director"]

    def test_enabled_real_branch_verdict_lands_with_correct_stage_name(self, s3, monkeypatch):
        monkeypatch.setenv("DIRECTOR_ENABLED", "1")
        s3.put_object(Bucket=BUCKET, Key=f"evaluator/{RUN_DATE}/report_card.json",
                      Body=json.dumps(_CARD).encode())
        from director import handler as H
        monkeypatch.setattr(H, "build_action_plan", lambda card, **kw: _plan())
        calls = _install_fake_stage_coverage(monkeypatch)
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET})
        assert out["status"] == "ok"
        assert out["stage_coverage"] == {"stage": "Director", "status": "COVERED", "run_date": RUN_DATE}
        assert [c["stage"] for c in calls] == ["Director"]


class TestStageCoverageEvaluatorDirectorDeployDriftCheck:
    """(director Lambda, check_deploy_drift dispatch) — the
    EvaluatorDirectorDeployDriftCheck SF stage. An infrastructure/gate stage:
    declares no durable artifact, but must still record that it declared
    nothing rather than being silently un-considered."""

    def test_verdict_lands_under_stage_coverage_with_correct_stage_name(self, monkeypatch):
        monkeypatch.delenv("DIRECTOR_ENABLED", raising=False)
        from director import handler as H
        import grading.deploy_drift as dd
        monkeypatch.setattr(dd, "check_deploy_drift", lambda *, function_name: {"has_drift": False})
        calls = _install_fake_stage_coverage(monkeypatch)

        out = H.handler({"action": "check_deploy_drift"}, context=None)

        assert out["stage_coverage"]["stage"] == "EvaluatorDirectorDeployDriftCheck"
        assert [c["stage"] for c in calls] == ["EvaluatorDirectorDeployDriftCheck"]
        # never confused with the real work stage this Lambda also backs.
        assert out["stage_coverage"]["stage"] != "Director"


class TestStageCoverageImportDegrades:
    """Observe mode cannot break the stage it observes: a ModuleNotFoundError
    from the lib must never change the handler's own outcome — logged loud
    AND the `stage_coverage` key is ALWAYS present (never absent, per I7334):
    status UNMEASURED with a reason, distinguishable from a real COVERED
    verdict, plus an alarmable CloudWatch metric."""

    def test_director_outcome_unchanged_when_module_absent(self, s3, monkeypatch):
        _block_stage_coverage_import(monkeypatch)
        monkeypatch.delenv("DIRECTOR_ENABLED", raising=False)
        from director import handler as H
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET})
        assert out["status"] == "disabled"
        assert out["stage_coverage"]["status"] == "UNMEASURED"
        assert "krepis.stage_coverage" in out["stage_coverage"]["reason"]
        assert out["stage_coverage"]["stage"] == "Director"

    def test_deploy_drift_check_outcome_unchanged_when_module_absent(self, monkeypatch):
        _block_stage_coverage_import(monkeypatch)
        from director import handler as H
        import grading.deploy_drift as dd
        monkeypatch.setattr(dd, "check_deploy_drift", lambda *, function_name: {"has_drift": False})
        out = H.handler({"action": "check_deploy_drift"}, context=None)
        assert out["has_drift"] is False
        assert out["stage_coverage"]["status"] == "UNMEASURED"
        assert out["stage_coverage"]["stage"] == "EvaluatorDirectorDeployDriftCheck"

    def test_unavailable_verdict_never_shares_shape_with_a_real_pass(self, s3, monkeypatch):
        _block_stage_coverage_import(monkeypatch)
        monkeypatch.delenv("DIRECTOR_ENABLED", raising=False)
        from director import handler as H
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET})
        assert out["stage_coverage"]["status"] != "COVERED"
        assert out["stage_coverage"]["status"] != "COVERED_NO_OUTPUT"

    def test_unavailable_publishes_alarmable_cloudwatch_metric(self, s3, monkeypatch):
        _block_stage_coverage_import(monkeypatch)
        monkeypatch.delenv("DIRECTOR_ENABLED", raising=False)
        from director import handler as H
        H.handler({"date": RUN_DATE, "bucket": BUCKET})

        cw = boto3.client("cloudwatch", region_name="us-east-1")
        stats = cw.list_metrics(Namespace="AlphaEngine", MetricName="StageCoverage")
        matches = [
            m for m in stats["Metrics"]
            if {"Name": "Stage", "Value": "Director"} in m["Dimensions"]
            and {"Name": "Status", "Value": "UNMEASURED"} in m["Dimensions"]
        ]
        assert matches, f"no UNMEASURED/Director StageCoverage metric published: {stats['Metrics']}"


class TestStageCoverageNeverEnablesEnforcement:
    """OBSERVE MODE ONLY (config-I7214): no shipped call site may pass an
    enforcement-enabling argument. `assert_stage_coverage` takes exactly
    (stage, run_date, window_start) from every Director call site — any
    extra kwarg would be how an enforcement flag could sneak in."""

    def test_call_sites_pass_only_the_observe_mode_signature(self, s3, monkeypatch):
        import sys
        import types
        seen_kwargs = []

        def _capture(stage, **kwargs):
            seen_kwargs.append(set(kwargs))
            return {"stage": stage, "status": "COVERED"}

        fake_mod = types.ModuleType("krepis.stage_coverage")
        fake_mod.assert_stage_coverage = _capture
        monkeypatch.setitem(sys.modules, "krepis.stage_coverage", fake_mod)

        monkeypatch.setenv("DIRECTOR_ENABLED", "1")
        s3.put_object(Bucket=BUCKET, Key=f"evaluator/{RUN_DATE}/report_card.json",
                      Body=json.dumps(_CARD).encode())
        from director import handler as H
        monkeypatch.setattr(H, "build_action_plan", lambda card, **kw: _plan())
        H.handler({"date": RUN_DATE, "bucket": BUCKET})

        import grading.deploy_drift as dd
        monkeypatch.setattr(dd, "check_deploy_drift", lambda *, function_name: {"has_drift": False})
        H.handler({"action": "check_deploy_drift"}, context=None)

        assert seen_kwargs
        for kwargs in seen_kwargs:
            assert kwargs == {"run_date", "window_start"}


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

    def test_judge_group_defaults_and_respects_env_override(self, monkeypatch):
        """The judge tier is config, not code — and it is a GROUP, never a
        model id. RETRO_JUDGE_GROUP overrides; the default is `high`,
        deliberately not the Director's `ultra` — grading a plan with the tier
        that wrote it is self-grading bias (config#1673)."""
        from director.agent import DIRECTOR_GROUP
        from director.retro import RETRO_JUDGE_GROUP_DEFAULT, _judge_group

        monkeypatch.delenv("RETRO_JUDGE_GROUP", raising=False)
        assert _judge_group() == RETRO_JUDGE_GROUP_DEFAULT == "high"
        assert DIRECTOR_GROUP == "ultra"
        assert _judge_group() != DIRECTOR_GROUP

        monkeypatch.setenv("RETRO_JUDGE_GROUP", "mid")
        assert _judge_group() == "mid"

    def test_no_model_id_override_survives_the_router_migration(self, monkeypatch):
        """RETRO_JUDGE_MODEL is GONE and must stay gone (I6562).

        Setting a literal model id was the reachable way to "unstick" the
        24-day retro outage, and it is exactly the shortcut that re-pins this
        call site off the registry and re-hides the router bypass. The tier is
        overridable; the model is not."""
        import director.retro as R

        assert not hasattr(R, "RETRO_JUDGE_MODEL_DEFAULT")
        assert not hasattr(R, "_judge_model")

        # An operator setting the old variable changes nothing.
        monkeypatch.delenv("RETRO_JUDGE_GROUP", raising=False)
        monkeypatch.setenv("RETRO_JUDGE_MODEL", "gpt-4o")
        assert R._judge_group() == "high"

    def test_judge_group_may_not_be_the_generators_group(self, monkeypatch):
        """config#1673 enforced at CALL TIME, not by two constants being
        distinct — RETRO_JUDGE_GROUP is env-overridable, and an operator
        setting it to `ultra` would silently turn the retro into self-grading
        while still producing something shaped like a grade."""
        import pytest
        from director.agent import DIRECTOR_GROUP
        from director.retro import _assert_judge_is_not_the_generator

        _assert_judge_is_not_the_generator("high")  # no raise

        with pytest.raises(RuntimeError, match="judge != generator"):
            _assert_judge_is_not_the_generator(DIRECTOR_GROUP)

        monkeypatch.setenv("RETRO_JUDGE_GROUP", DIRECTOR_GROUP)
        from director.retro import _default_llm
        with pytest.raises(RuntimeError, match="judge != generator"):
            _default_llm()

    # ── I6052: the SERVED-model half of judge != generator ─────────

    @staticmethod
    def _graded(judge_model):
        """A RetroGrade shaped like one the real judge returns — the client
        stamps `resolved_model` with what the API actually served."""
        from director.schema import RetroGrade
        g = RetroGrade(prior_run_date="2026-05-23", grounding=80,
                       calibration=55, actionability=70)
        if judge_model is not None:
            g.resolved_model = judge_model
        return g

    def test_a_judge_served_the_graded_plans_own_model_refuses_to_publish(self):
        """The live conditional the GROUP check cannot see
        (alpha-engine-config-I6052).

        `high` and `ultra` are distinct groups and the registry's disjointness
        invariant compares ENTRY IDS, so `high`'s `deepseek-v4-pro-max` and
        `ultra`'s `deepseek-v4-pro` pass it while resolving to the SAME
        upstream model. When `ultra` falls through to that arm, the plan is
        graded by the model that wrote it — and nothing before the call can
        know, because LiteLLM walks the chain internally.

        No grade is returned at all: `handler._persist_retro` never runs, so a
        self-graded verdict cannot reach `director/{date}/retro.json`, the
        retro trend ledger, or the Report Card.
        """
        import pytest
        from director.retro import SelfGradedRetroError, grade_prior_plan

        prior = {"run_date": "2026-05-23", "served_model": "deepseek-v4-pro"}
        llm = _FakeLLM(self._graded("deepseek-v4-pro"))

        with pytest.raises(SelfGradedRetroError) as exc:
            grade_prior_plan(prior, _CARD, llm=llm)

        assert "deepseek-v4-pro" in str(exc.value)
        assert "2026-05-23" in str(exc.value)
        assert "self-grading" in str(exc.value)

    def test_the_same_model_through_a_different_reseller_is_still_a_collision(self):
        """`high` carries `deepseek-v4-pro-openrouter-max`
        (`deepseek/deepseek-v4-pro`) beside the direct entry. Same weights, one
        aggregator apart — comparing the raw upstream ids would read them as
        distinct models and publish the grade."""
        import pytest
        from director.retro import SelfGradedRetroError, grade_prior_plan

        prior = {"run_date": "2026-05-23", "served_model": "deepseek-v4-pro"}
        llm = _FakeLLM(self._graded("deepseek/deepseek-v4-pro"))

        with pytest.raises(SelfGradedRetroError):
            grade_prior_plan(prior, _CARD, llm=llm)

    def test_distinct_served_models_publish_and_are_stamped(self):
        """The healthy path still returns a grade, and says so on the
        artifact — a check whose only output is a failure cannot be watched."""
        from director.retro import (
            JUDGE_DISTINCT, RETRO_KEY_GRADED_PLAN_MODEL,
            RETRO_KEY_JUDGE_VS_GENERATOR, grade_prior_plan,
        )

        prior = {"run_date": "2026-05-23", "served_model": "glm-5.2"}
        out = grade_prior_plan(prior, _CARD,
                               llm=_FakeLLM(self._graded("deepseek-v4-pro")))

        assert out.calibration == 55
        dumped = out.model_dump()
        assert dumped[RETRO_KEY_JUDGE_VS_GENERATOR] == JUDGE_DISTINCT
        assert dumped[RETRO_KEY_GRADED_PLAN_MODEL] == "glm-5.2"

    def test_an_unprovable_grade_is_published_as_unknown_never_as_clean(self):
        """A plan with no served-model stamp cannot be shown free of bias.
        `unknown` is a real answer and is never collapsed into `distinct` —
        principles.md §2.7, no data is never rendered as green."""
        from director.retro import (
            JUDGE_UNKNOWN, RETRO_KEY_JUDGE_VS_GENERATOR, grade_prior_plan,
        )

        out = grade_prior_plan({"run_date": "2026-05-23"}, _CARD,
                               llm=_FakeLLM(self._graded("deepseek-v4-pro")))
        assert out.model_dump()[RETRO_KEY_JUDGE_VS_GENERATOR] == JUDGE_UNKNOWN

        # ... and symmetrically when the JUDGE side is the unstamped one.
        out2 = grade_prior_plan(
            {"run_date": "2026-05-23", "served_model": "glm-5.2"}, _CARD,
            llm=_FakeLLM(self._graded(None)),
        )
        assert out2.model_dump()[RETRO_KEY_JUDGE_VS_GENERATOR] == JUDGE_UNKNOWN

    def test_a_pre_i8165_plan_is_still_comparable_via_resolved_model(self):
        """`served_model` only exists on plans written since I8165;
        `resolved_model` has carried the same value since config#1673. Reading
        only the newer key would render every older plan `unknown`."""
        import pytest
        from director.retro import SelfGradedRetroError, grade_prior_plan

        prior = {"run_date": "2026-05-23", "resolved_model": "deepseek-v4-pro"}
        with pytest.raises(SelfGradedRetroError):
            grade_prior_plan(prior, _CARD,
                             llm=_FakeLLM(self._graded("deepseek-v4-pro")))

    def test_every_retro_emits_the_self_grading_metric(self, capsys):
        """Both metrics on every run, 0 on the healthy path. A metric that only
        appears when something is wrong is indistinguishable from a dead
        emitter (principles.md §2.7)."""
        from director.retro import grade_prior_plan

        grade_prior_plan({"run_date": "2026-05-23", "served_model": "glm-5.2"},
                         _CARD, llm=_FakeLLM(self._graded("deepseek-v4-pro")))

        emitted = [
            json.loads(line)
            for line in capsys.readouterr().out.splitlines()
            if '"RetroJudgeSelfGraded"' in line and line.startswith("{")
        ]
        assert len(emitted) == 1
        row = emitted[0]
        assert row["RetroJudgeSelfGraded"] == 0
        assert row["RetroJudgeSelfGradeUnknown"] == 0
        assert row["Group"] == "high"
        assert row["_aws"]["CloudWatchMetrics"][0]["Namespace"] == "AlphaEngine/Director"

    def test_the_metric_fires_on_the_collision_before_the_refusal(self, capsys):
        """The refusal raises, so the metric is the only surface that survives
        the run — it must be emitted before the raise, not after it."""
        import pytest
        from director.retro import SelfGradedRetroError, grade_prior_plan

        with pytest.raises(SelfGradedRetroError):
            grade_prior_plan(
                {"run_date": "2026-05-23", "served_model": "deepseek-v4-pro"},
                _CARD, llm=_FakeLLM(self._graded("deepseek-v4-pro")),
            )

        row = next(
            json.loads(line)
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("{") and '"RetroJudgeSelfGraded"' in line
        )
        assert row["RetroJudgeSelfGraded"] == 1
        assert row["RetroJudgeSelfGradeUnknown"] == 0

    def test_the_group_check_is_not_sufficient_and_says_so(self):
        """Guard against the fix being reverted to the group comparison alone:
        two distinct groups pass `_assert_judge_is_not_the_generator` and the
        served-model check still refuses."""
        import pytest
        from director.retro import (
            SelfGradedRetroError, _assert_judge_is_not_the_generator,
            _assert_judge_did_not_grade_its_own_plan,
        )

        _assert_judge_is_not_the_generator("high")  # distinct groups: fine

        with pytest.raises(SelfGradedRetroError):
            _assert_judge_did_not_grade_its_own_plan(
                self._graded("deepseek-v4-pro"),
                {"run_date": "2026-05-23", "served_model": "deepseek-v4-pro"},
                judge_group="high",
            )

    def test_handler_records_a_self_graded_refusal_without_losing_the_plan(
        self, s3, monkeypatch
    ):
        """End to end: the refusal reaches the handler's best-effort wrapper as
        `retro: error`, the plan (primary deliverable) still ships, and
        `director/{date}/retro.json` is NOT written — there is no self-graded
        verdict for the trend ledger or the Report Card to read."""
        import botocore.exceptions

        monkeypatch.setenv("DIRECTOR_ENABLED", "1")
        s3.put_object(Bucket=BUCKET, Key=f"evaluator/{RUN_DATE}/report_card.json",
                      Body=json.dumps(_CARD).encode())
        prior = _plan().model_dump()
        prior["run_date"] = "2026-05-23"
        prior["served_model"] = "deepseek-v4-pro"
        s3.put_object(Bucket=BUCKET, Key="director/2026-05-23/action_plan.json",
                      Body=json.dumps(prior).encode())

        from director import handler as H
        monkeypatch.setattr(H, "build_action_plan", lambda card, **kw: _plan())
        import director.retro as R
        monkeypatch.setattr(
            R, "_default_llm",
            lambda budget=None: _FakeLLM(self._graded("deepseek-v4-pro")),
        )

        out = H.handler({"date": RUN_DATE, "bucket": BUCKET})

        assert out["status"] == "ok"  # the plan still shipped
        assert out["retro"] == "error"
        assert "self-grading" in out["retro_error"]
        assert json.loads(
            s3.get_object(Bucket=BUCKET, Key=out["action_plan_key"])["Body"].read()
        )
        with pytest.raises(botocore.exceptions.ClientError):
            s3.get_object(Bucket=BUCKET, Key=f"director/{RUN_DATE}/retro.json")

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

    def test_default_llm_resolves_the_group_through_the_router_not_a_pinned_model(self, monkeypatch):
        """_default_llm() resolves a GROUP through krepis.router and takes the
        model/provider/endpoint from the returned route.

        The regression this guards (I6562): it used to build
        `ModelSpec(provider="openrouter", model=<literal>)` at this call site
        and egress straight past the authenticated edge. `provider` must now be
        whatever the registry says for the proxy route — never the string
        "openrouter" chosen here."""
        monkeypatch.delenv("RETRO_JUDGE_GROUP", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        import krepis.secrets as ks
        monkeypatch.setattr(ks, "get_secret", lambda name, **kw: "test-key")

        # The resolver is faked rather than driven against the real registry:
        # `LLM_MODEL_REGISTRY.yaml` lives in the PRIVATE alpha-engine-config
        # repo, so a test that resolves for real passes on a laptop that
        # happens to have that checkout and fails in CI and in the built image
        # — an ambient-environment dependency, not a test. What is under test
        # here is this module's CONSUMPTION of the route, which is where the
        # bug was.
        seen = {}

        def _fake_resolve(group, **kw):
            from krepis.llm_config import ModelSpec
            seen["group"] = group
            seen["kw"] = kw
            spec = ModelSpec(
                provider="litellm_proxy",
                model="high-deepseek-v4-pro-max",
                max_tokens=8192,
                base_url="http://127.0.0.1:8980",
            )
            return spec, {
                "route": "litellm_proxy",
                "deployment_id": "high-deepseek-v4-pro-max",
                "provider": "litellm",
                "registry_id": "litellm:group:high",
                "primary_model": "deepseek-v4-pro-max",
                "primary_registry_id": "deepseek-v4-pro-max",
                "api_base_url": "http://127.0.0.1:8980",
                "auth_token_type": "placeholder",
                "skipped_entries": [],
                "exec_context": "lambda",
            }

        import krepis.router as KR
        monkeypatch.setattr(KR, "resolve_group_spec", _fake_resolve)

        from director.retro import _KrepisStructuredJudge, _default_llm
        llm = _default_llm()
        assert isinstance(llm, _KrepisStructuredJudge)

        # It asks for a GROUP, and says only where it runs — nothing else about
        # routing (model-router-policy §2 layer 5, R29).
        assert seen["group"] == "high"
        assert seen["kw"]["wire"] == "openai"
        assert seen["kw"]["exec_context"] == "lambda"

        assert llm._judge_group == "high"
        # Taken from the ROUTE, not hardcoded at this call site.
        assert llm._judge_model == "high-deepseek-v4-pro-max"
        assert llm._client.spec.provider != "openrouter"
        # max_tokens comes from the registry entry, not pinned here (it was 2000).
        assert llm._client.spec.max_tokens == 8192

    def test_default_llm_refuses_a_route_that_skips_the_proxy(self, monkeypatch):
        """From a Lambda the only conformant route is `litellm_proxy`
        (model-router-policy R26). The judge shares the plan call's guard —
        this is the assertion that existed in agent.py and never reached
        retro.py, which is how the bypass survived (engagement §5)."""
        import pytest
        import director.retro as R

        monkeypatch.delenv("RETRO_JUDGE_GROUP", raising=False)
        monkeypatch.setattr(R, "DIRECTOR_EXEC_CONTEXT", "lambda")

        def _fake_resolve(group, **kw):
            from krepis.llm_config import ModelSpec
            spec = ModelSpec(provider="openrouter", model="whatever", max_tokens=2000)
            return spec, {"route": "direct", "deployment_id": "whatever",
                          "provider": "openrouter", "api_base_url": "https://openrouter.ai",
                          "auth_token_type": "api_key", "skipped_entries": []}

        import krepis.router as KR
        monkeypatch.setattr(KR, "resolve_group_spec", _fake_resolve)
        with pytest.raises(RuntimeError, match="model-router-policy R26"):
            R._default_llm()

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
        from director.retro import _KrepisStructuredJudge, build_messages

        _JUDGE_MODEL = "high-deepseek-v4-pro-max"  # as the registry would resolve `high`

        class _FakeToolUseBlock:
            def __init__(self, name, input_):
                self.type = "tool_use"
                self.name = name
                self.input = input_

        class _FakeUsage:
            input_tokens = 4200
            output_tokens = 380

        class _FakeAnthropicMessage:
            def __init__(self, content, model):
                self.content = content
                self.model = model
                self.usage = _FakeUsage()

        class _FakeAnthropicClient:
            def __init__(self):
                self.messages = self

            def create(self, **payload):
                assert payload["model"] == _JUDGE_MODEL
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

        spec = ModelSpec(provider="anthropic", model=_JUDGE_MODEL, max_tokens=2000)
        client = LLMClient(
            spec,
            api_key="test-key",
            client_factory=_client_factory,
            callsite_id="director-retro-judge",
        )
        judge = _KrepisStructuredJudge(client, judge_group="high", judge_model=_JUDGE_MODEL)
        assert judge.last_usage is None  # unset until a call RETURNS

        messages = build_messages(_plan().model_dump(), _CARD)
        grade = judge.invoke(messages)

        # Measured regression: this adapter never assigned `self.last_usage`,
        # so `_invoke_with_retry`'s `getattr(llm, "last_usage", None)` always
        # read None for the retro/judge call — every judge-call
        # plan_call_telemetry record published DirectorPlanPromptTokens=0 /
        # DirectorPlanCompletionTokens=0 / DirectorPlanReasoningTokens=0
        # regardless of what krepis returned (alpha-engine-config-I8164-followon).
        assert judge.last_usage is not None
        assert judge.last_usage.input_tokens == 4200
        assert judge.last_usage.output_tokens == 380

        assert grade.calibration == 55
        assert grade.judge_group == "high"
        assert grade.judge_model == _JUDGE_MODEL
        assert grade.resolved_model == "claude-sonnet-4-6-20260115"
        assert grade.judge_model != grade.resolved_model  # registry-resolved vs API-served
        dumped = grade.model_dump()
        assert dumped["judge_group"] == "high"
        assert dumped["judge_model"] == _JUDGE_MODEL
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

    def test_declares_lambda_by_default(self):
        from director import agent as A
        assert A.DIRECTOR_EXEC_CONTEXT == "lambda"

    def test_context_names_no_network_attachment(self):
        """§3.4a R27a — reaching the router may not depend on network
        position, so the context this module declares must not assert one.
        `lambda_vpc` was the original value in this arc, and a name asserting
        an attachment is what makes "attach the consumer" read as the fix for
        an unreachable router — the reasoning behind nous-ergon-ops-I417."""
        from director import agent as A
        assert not any(t in A.DIRECTOR_EXEC_CONTEXT
                       for t in ("vpc", "subnet", "sg", "peering")), (
            f"{A.DIRECTOR_EXEC_CONTEXT!r} names a network attachment, not a "
            "place code runs"
        )

    def test_resolution_passes_context_and_openai_wire(self, monkeypatch):
        """`wire=openai` matters: this call site builds a krepis LLMClient on
        the openai transport, and the resolver's default is the Claude CLI's
        Anthropic wire. Without it a DeepSeek egress fallback would hand this
        module an 8971 URL its transport cannot speak."""
        from director import agent as A

        seen = {}

        import sys, types

        class _Spec:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        def _fake_resolve(group, **kw):
            seen["group"] = group
            seen.update(kw)
            spec = _Spec(
                provider="litellm_proxy", model="ultra",
                base_url="https://router.nousergon.ai:8443",
                api_key_env="LITELLM_MASTER_KEY", max_tokens=16384,
                structured_outputs=True, reasoning=None, transport="openai",
            )
            route = {
                "schema_version": 2,
                "provider": "litellm", "deployment_id": "ultra",
                "api_base_url": "https://router.nousergon.ai:8443",
                "auth_token_type": "litellm_master_key",
                "registry_id": "litellm:group:ultra",
                "primary_registry_id": "litellm:group:ultra",
                "route": "litellm_proxy", "exec_context": "lambda",
                "params": {}, "skipped_entries": [],
            }
            return spec, route

        krepis_router = types.ModuleType("krepis.router")
        krepis_router.resolve_group_spec = _fake_resolve
        krepis_llm = types.ModuleType("krepis.llm")
        krepis_llm.LLMClient = lambda spec, **kw: object()
        krepis_cfg = types.ModuleType("krepis.llm_config")
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
        assert seen["exec_context"] == "lambda"
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
            "exec_context": "lambda",
            "skipped_entries": [],
        }
        r.update(over)
        return r

    @staticmethod
    def _emf(capsys):
        """The EMF line is the metric. Parse it back out of stdout."""
        import json as _json
        for line in capsys.readouterr().out.splitlines():
            line = line.strip()
            if line.startswith("{") and "_aws" in line:
                return _json.loads(line)
        return None

    def test_healthy_route_emits_metric_value_zero(self, capsys):
        """A metric that only appears on failure is indistinguishable from a
        dead emitter (principles.md §2.7)."""
        from director import agent as A
        A._warn_on_degraded_route(self._route())
        emf = self._emf(capsys)
        assert emf is not None, "no EMF line emitted on the healthy path"
        assert emf["DirectorRouteFallback"] == 0

    def test_fallback_emits_one_and_warns(self, capsys, caplog):
        from director import agent as A
        route = self._route(
            registry_id="glm-5.2",
            skipped_entries=[{"registry_id": "kimi-k3-direct",
                              "reason": "Not reachable from execution context 'lambda'"}],
        )
        with caplog.at_level("WARNING"):
            A._warn_on_degraded_route(route)
        emf = self._emf(capsys)
        assert emf["DirectorRouteFallback"] == 1
        assert emf["served"] == "glm-5.2"
        assert "DEGRADED" in caplog.text
        assert "kimi-k3-direct" in caplog.text
        assert "lambda" in caplog.text

    def test_emf_carries_the_metric_directive(self, capsys):
        """Without the `_aws` block CloudWatch treats the line as plain log
        text and no metric is ever created — a silent dead emitter with a
        healthy-looking log."""
        from director import agent as A
        A._warn_on_degraded_route(self._route())
        emf = self._emf(capsys)
        directive = emf["_aws"]["CloudWatchMetrics"][0]
        assert directive["Namespace"] == "AlphaEngine/Director"
        assert directive["Dimensions"] == [["Group"]]
        assert {"Name": "DirectorRouteFallback", "Unit": "Count"} in directive["Metrics"]
        assert emf["Group"] == "ultra"

    def test_emits_no_network_call(self, monkeypatch, capsys):
        """The Lambda is VPC-attached with no internet route, so a boto3
        CloudWatch call would hang until timeout on EVERY run and then be
        swallowed. Importing boto3 here at all is the regression."""
        import sys, types
        from director import agent as A

        exploding = types.ModuleType("boto3")

        def _boom(*a, **k):
            raise AssertionError(
                "_warn_on_degraded_route made a boto3 call; EMF must need none"
            )
        exploding.client = _boom
        monkeypatch.setitem(sys.modules, "boto3", exploding)
        A._warn_on_degraded_route(self._route())
        assert self._emf(capsys)["DirectorRouteFallback"] == 0


class TestDirectorCredentialResolution:
    """config-I6056 / model-router-policy R20 — fail closed, loudly.

    Supersedes crucible-evaluator-PR169, which resolved the credential
    unconditionally and would therefore have raised on a `placeholder` route
    (egress_proxy entries carry no key of their own — the proxy holds it).
    That never bit in the Lambda, where only the litellm_proxy route can
    serve, but it would have broken every direct route from `laptop`.
    """

    def _patch_krepis(self, monkeypatch, *, auth_token_type, secret):
        import sys, types
        from director import agent as A

        captured = {}

        class _Spec:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        # krepis 0.31.1 owns the auth_token_type -> credential-name mapping and
        # applies it inside resolve_group_spec, so the spec arrives with
        # api_key_env already decided. `placeholder` means the egress proxy
        # holds the real key and there is nothing for the consumer to resolve.
        _spec = _Spec(
            provider="litellm_proxy", model="ultra",
            base_url="https://router.nousergon.ai:8443",
            api_key_env=None if auth_token_type == "placeholder" else "LITELLM_MASTER_KEY",
            max_tokens=16384, structured_outputs=True, reasoning=None,
            transport="openai",
        )
        _route = {
            "schema_version": 2,
            "provider": "litellm", "deployment_id": "ultra",
            "api_base_url": "https://router.nousergon.ai:8443",
            "auth_token_type": auth_token_type,
            "registry_id": "r", "primary_registry_id": "r",
            "route": "litellm_proxy", "exec_context": "lambda",
            "params": {}, "skipped_entries": [],
        }
        krepis_router = types.ModuleType("krepis.router")
        krepis_router.resolve_group_spec = lambda group, **kw: (_spec, _route)
        krepis_llm = types.ModuleType("krepis.llm")

        def _client(spec, **kw):
            captured.update(kw)
            return object()
        krepis_llm.LLMClient = _client
        krepis_cfg = types.ModuleType("krepis.llm_config")
        krepis_cfg.ModelSpec = _Spec
        krepis_secrets = types.ModuleType("krepis.secrets")
        krepis_secrets.get_secret = lambda name, **kw: secret
        pkg = types.ModuleType("krepis")
        for name, mod in (("krepis", pkg), ("krepis.router", krepis_router),
                          ("krepis.llm", krepis_llm),
                          ("krepis.llm_config", krepis_cfg),
                          ("krepis.secrets", krepis_secrets)):
            monkeypatch.setitem(sys.modules, name, mod)
        monkeypatch.setattr(A, "_warn_on_degraded_route", lambda r: None)
        return A, captured

    def test_placeholder_route_resolves_no_credential(self, monkeypatch):
        """The egress proxy holds the real key. Demanding one here would break
        every direct route reachable from `laptop`."""
        A, captured = self._patch_krepis(
            monkeypatch, auth_token_type="placeholder", secret=None)
        A._default_llm()
        assert "api_key" not in captured

    def test_named_credential_is_passed_through(self, monkeypatch):
        A, captured = self._patch_krepis(
            monkeypatch, auth_token_type="litellm_master_key", secret="sk-x")
        A._default_llm()
        assert captured["api_key"] == "sk-x"

    def test_unresolvable_named_credential_raises(self, monkeypatch):
        """It used to fall through with no key; the 401 that followed read as
        a provider fault rather than a missing credential."""
        A, _ = self._patch_krepis(
            monkeypatch, auth_token_type="litellm_master_key", secret=None)
        with pytest.raises(RuntimeError, match="no credential for auth_token_type"):
            A._default_llm()


class TestDirectorRoutesThroughTheProxyNoExceptions:
    """From a Lambda the Director routes through the LiteLLM proxy or it does not run.

    Brian, 2026-08-03: "director should be using the krepis router, ultra
    complexity. no exceptions."

    The reason this is a RUNTIME guard and not only a CI assertion: every time
    this has failed in production, the code was correct and the artifact it read
    was not. `exclude_route="litellm_proxy"` in this module; then a hand-published
    S3 registry copy with no `reachable_from`, which krepis read as universal
    reachability; then a krepis health probe that spoke plain HTTP at the
    router's TLS edge and declared it unreachable. Each produced the same
    symptom — a paid `ultra` call to openrouter.ai, DLP-unscanned, logging a
    healthy route (alpha-engine-config-I6183, model-router-policy R26).
    """

    @staticmethod
    def _route(route_name, **over):
        base = {
            "schema_version": 2,
            "route": route_name,
            "provider": "openrouter" if route_name == "openrouter" else "litellm",
            "deployment_id": "z-ai/glm-5.2" if route_name == "openrouter" else "ultra",
            "api_base_url": ("https://openrouter.ai/api" if route_name == "openrouter"
                             else "https://router.nousergon.ai:8443"),
            "auth_token_type": ("openrouter_key" if route_name == "openrouter"
                                else "litellm_master_key"),
            "params": {"max_tokens": 16384},
            "skipped_entries": [],
        }
        base.update(over)
        return base

    def test_openrouter_from_a_lambda_raises(self):
        from director import agent as A
        with pytest.raises(RuntimeError) as exc:
            A._assert_routed_through_the_proxy(self._route("openrouter"))
        msg = str(exc.value)
        # The message has to be diagnosable on its own: which route, and enough
        # to tell "stale registry" from "resolver skipped the proxy".
        assert "openrouter" in msg
        assert "litellm_proxy" in msg
        assert "openrouter.ai" in msg

    def test_egress_proxy_from_a_lambda_also_raises(self):
        """Not an openrouter-specific block — 127.0.0.1:8990 does not exist here."""
        from director import agent as A
        with pytest.raises(RuntimeError):
            A._assert_routed_through_the_proxy(
                self._route("egress_proxy", api_base_url="http://127.0.0.1:8990"))

    def test_litellm_proxy_from_a_lambda_passes(self):
        from director import agent as A
        A._assert_routed_through_the_proxy(self._route("litellm_proxy"))

    def test_a_direct_route_is_legitimate_off_lambda(self, monkeypatch):
        """On the laptop or the box the egress proxy is on loopback (R27d)."""
        from director import agent as A
        monkeypatch.setattr(A, "DIRECTOR_EXEC_CONTEXT", "laptop")
        A._assert_routed_through_the_proxy(self._route("openrouter"))

    def test_the_guard_is_wired_into_default_llm(self, monkeypatch):
        """The load-bearing case: assert the SHIPPED path, not the helper.

        A guard that exists and is never called is the failure mode this whole
        arc is made of.
        """
        from director import agent as A
        import krepis.router as kr
        monkeypatch.setattr(
            kr, "resolve_group_structured",
            lambda *a, **kw: TestDirectorRoutesThroughTheProxyNoExceptions._route("openrouter"))
        with pytest.raises(RuntimeError, match="litellm_proxy"):
            A._default_llm()


class TestDegradedRouteIsNotAlwaysTrueOnTheProxyPath:
    """A fallback alarm that is always on is as useless as one that never fires.

    On the proxy path krepis returns `registry_id = "litellm:group:ultra"` — a
    GROUP HANDLE — while `primary_registry_id` is an entry id like
    `kimi-k3-direct`. The old comparison could never match, so the very first
    healthy run through the router (2026-08-03 23:15:45Z, live) logged
    `Director route DEGRADED` and emitted `DirectorRouteFallback=1`:

        Director route DEGRADED: group=ultra primary=kimi-k3-direct
        served=litellm:group:ultra route=litellm_proxy — skipped: (none recorded)

    The consumer cannot know which entry served through the proxy; LiteLLM
    walks the chain internally. `skipped_entries` — what the resolver itself
    refused — is the only degradation signal this layer legitimately has.
    """

    @staticmethod
    def _proxy_route(**over):
        base = {
            "route": "litellm_proxy",
            "registry_id": "litellm:group:ultra",
            "primary_registry_id": "kimi-k3-direct",
            "primary_model": "kimi-k3",
            "exec_context": "lambda",
            "skipped_entries": [],
        }
        base.update(over)
        return base

    def test_healthy_proxy_route_is_not_degraded(self, capsys, caplog):
        from director import agent as A
        with caplog.at_level("WARNING"):
            A._warn_on_degraded_route(self._proxy_route())
        assert "DEGRADED" not in caplog.text
        emitted = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert emitted["DirectorRouteFallback"] == 0, (
            "a healthy proxy route emitted the fallback metric — the alarm is "
            "pinned on and cannot signal the week it matters"
        )

    def test_proxy_route_with_skipped_entries_is_still_degraded(self, capsys, caplog):
        """The signal that survives: entries the resolver itself refused."""
        from director import agent as A
        with caplog.at_level("WARNING"):
            A._warn_on_degraded_route(self._proxy_route(
                skipped_entries=[{"registry_id": "kimi-k3-direct",
                                  "reason": "not reachable from 'lambda'"}]))
        assert "DEGRADED" in caplog.text
        emitted = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert emitted["DirectorRouteFallback"] == 1

    def test_direct_route_still_compares_served_against_primary(self, capsys, caplog):
        """Off the proxy path the comparison is meaningful and must not regress."""
        from director import agent as A
        with caplog.at_level("WARNING"):
            A._warn_on_degraded_route({
                "route": "openrouter",
                "registry_id": "glm-5.2",
                "primary_registry_id": "kimi-k3-direct",
                "exec_context": "laptop",
                "skipped_entries": [],
            })
        assert "DEGRADED" in caplog.text
        emitted = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert emitted["DirectorRouteFallback"] == 1


class TestDirectorNeverUsesTheInProcessRouter:
    """The proxy route is an HTTP endpoint, not an in-process LiteLLM Router.

    `resolve_group_structured` reports `provider: "litellm"` for the proxy
    route, and `krepis.llm_config.PROVIDER_REGISTRY` binds that name to
    TRANSPORT_LITELLM — `get_router()`, an in-process Router built from the
    registry that calls each provider DIRECTLY from this Lambda, reading
    OPENROUTER_API_KEY out of the environment as it goes.

    Found 2026-08-04 on the FIRST REAL invoke after the authenticated edge went
    live: `ModuleNotFoundError: No module named 'litellm'`. The tempting fix —
    add the package — would have made the Director "work" while egressing
    straight to openrouter.ai, unscanned, bypassing the edge and every
    per-consumer control on it.

    A dry run cannot catch it. `_dry_run_probe` stops before `.invoke()`, so
    the transport is never exercised; the probe was green throughout.
    """

    def test_the_spec_transport_is_never_the_in_process_router(self, monkeypatch):
        import sys, types
        from director import agent as A

        built = {}

        class _Spec:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        spec = _Spec(
            provider="litellm_proxy", model="ultra",
            base_url="https://router.nousergon.ai:8443",
            api_key_env="LITELLM_MASTER_KEY", max_tokens=16384,
            structured_outputs=True, reasoning=None, transport="openai",
        )
        route = {
            "schema_version": 2, "provider": "litellm", "deployment_id": "ultra",
            "api_base_url": "https://router.nousergon.ai:8443",
            "auth_token_type": "litellm_master_key",
            "registry_id": "r", "primary_registry_id": "r",
            "route": "litellm_proxy", "exec_context": "lambda",
            "params": {}, "skipped_entries": [],
        }

        krepis_router = types.ModuleType("krepis.router")
        krepis_router.resolve_group_spec = lambda group, **kw: (spec, route)
        krepis_llm = types.ModuleType("krepis.llm")

        def _client(s, **kw):
            built["spec"] = s
            return object()
        krepis_llm.LLMClient = _client
        krepis_cfg = types.ModuleType("krepis.llm_config")
        krepis_cfg.ModelSpec = _Spec
        krepis_secrets = types.ModuleType("krepis.secrets")
        krepis_secrets.get_secret = lambda name, **kw: "test-key"
        pkg = types.ModuleType("krepis")
        for name, mod in (("krepis", pkg), ("krepis.router", krepis_router),
                          ("krepis.llm", krepis_llm),
                          ("krepis.llm_config", krepis_cfg),
                          ("krepis.secrets", krepis_secrets)):
            monkeypatch.setitem(sys.modules, name, mod)
        monkeypatch.setattr(A, "_warn_on_degraded_route", lambda r: None)

        A._default_llm()

        assert built["spec"].transport != "litellm", (
            "the Director's ModelSpec resolves to TRANSPORT_LITELLM — the "
            "IN-PROCESS LiteLLM Router, which calls providers directly from "
            "this Lambda and bypasses the authenticated edge entirely"
        )
        assert built["spec"].base_url, (
            "the proxy route must carry an explicit base_url; without one the "
            "OpenAI transport has no edge to address"
        )

    def test_agent_does_not_rebuild_the_spec_itself(self):
        """Consume the contract; do not reconstruct it (layer-5 rule).

        Hand-building a ModelSpec from the route dict is what re-derived
        `provider="litellm"` and selected the in-process router. AST, not grep,
        so the docstrings explaining the history do not fail the test.
        """
        import ast
        import inspect
        from director import agent as A

        tree = ast.parse(inspect.getsource(A))
        offenders = [
            node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", getattr(node.func, "attr", None)) == "ModelSpec"
        ]
        assert not offenders, (
            f"director/agent.py constructs a ModelSpec at line(s) {offenders}. "
            "krepis' resolve_group_spec returns one already adapted to the "
            "route — rebuilding it re-derives the provider and reselects the "
            "in-process router."
        )


class TestDirectorDoesNotShadowTheRegistryBudget:
    """`max_tokens` is a registry-owned parameter. A literal here silently wins.

    `krepis.llm.LLMClient.structured` resolves the budget as
    `max_tokens if max_tokens is not None else self.spec.max_tokens`, so a
    caller-supplied literal takes precedence over the registry row — with no
    warning, and no trace in any log the Lambda emits.

    Live 2026-08-04 (alpha-engine-config-I6396): this module passed
    `max_tokens=8000`. GLM-5.2 is a reasoning model and `max_tokens` bounds
    reasoning + content TOGETHER, so the entire budget went to the reasoning
    trace and both attempts returned `content: ''` after ~100s each, fully
    billed. nginx logged those responses at 30,689 and 30,710 bytes — the
    reasoning trace, not an empty body.

    It also made the remediation inert: raising the registry row 16384 → 65536
    (alpha-engine-config-PR6390) changed nothing, because this literal was what
    the request carried, while the route log printed `spec.max_tokens` — the
    registry's value, which was never sent. Three diagnostic cycles were spent
    on hypotheses that assumed the logged number was the wire number.
    """

    def test_the_structured_call_passes_no_max_tokens(self):
        from director.agent import _KrepisStructuredDirector
        from director.schema import DirectorWeeklyActionPlan

        seen = {}

        class _Client:
            def structured(self, **kwargs):
                seen.update(kwargs)

                class _R:
                    parsed = DirectorWeeklyActionPlan(
                        run_date="2026-07-31",
                        system_summary="s",
                        top_risks=[],
                        action_items=[],
                    )
                    model = "z-ai/glm-5.2"

                return _R()

        _KrepisStructuredDirector(
            _Client(), director_model="ultra"
        ).invoke([("system", "sys"), ("human", "body")])

        assert "max_tokens" not in seen, (
            f"the Director passes max_tokens={seen['max_tokens']!r} to krepis, "
            "shadowing the registry row's budget. The registry decides the "
            "budget (model-router-policy §2); this call site decides only its "
            "capability tier and where it runs."
        )

    def test_no_literal_max_tokens_anywhere_in_the_module(self):
        """AST, not grep — the docstrings above explain the history in prose.

        Guards the CLASS, not the one call: any krepis call surface in this
        module that restates a registry parameter reintroduces the same
        silent shadowing.
        """
        import ast
        import inspect
        from director import agent as A

        offenders = [
            node.lineno
            for node in ast.walk(ast.parse(inspect.getsource(A)))
            if isinstance(node, ast.Call)
            for kw in node.keywords
            if kw.arg == "max_tokens" and isinstance(kw.value, ast.Constant)
        ]
        assert not offenders, (
            f"director/agent.py passes a literal max_tokens at line(s) "
            f"{offenders}. The registry owns the budget; a literal here wins "
            "over it silently and makes any registry-side change inert."
        )
