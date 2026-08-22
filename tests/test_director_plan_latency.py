"""alpha-engine-config-I7311 — the Director's plan call is bounded and measured.

**What was measured, on the live system, 2026-08-14.**

`watch-rerun-2026-08-13-5` hard-failed the weekly SF: two consecutive plan
attempts each hit the 340s per-attempt ceiling and the invocation budget
correctly refused a third. Nothing in the retry path was broken. What was
broken is that the call's duration was a monotonically increasing function of
two artifacts that only grow, bounded by nothing and measured by nothing:

* Plan-call wall time, reconstructed from CloudWatch by subtracting the
  `Director route:` line from the `HTTP Request: POST` line, same route, same
  model (`ultra` → `ultra-glm-5.2-direct` via `litellm_proxy`), same registry
  `max_tokens=65536`, `degraded=False` throughout:
  89s/87s/87s/100s/107s/107s (2026-08-04) → 135s (2026-08-13) → 205s
  (2026-08-14 01:55, succeeded) → two censored attempts >340s (2026-08-14
  01:37 and 01:43).
* Carry-over ledger: 19 → 22 → 26 → 29 → 31 → 35 → 37 → 41 rows. All 41 with
  `status="carried_over"`; not one had ever reached `resolved`; 17 had a
  `last_seen` between 2026-06-18 and 2026-07-17. Every one still rendered into
  the prompt and still requiring a `carryover_review` line — that list had
  exactly 41 entries in the 2026-08-13 plan.
* Report card: 64.9 KB (2026-06-05) → 120.0 KB (2026-07-17) → 145.0 KB
  (2026-08-13). Reconstructed prompt: 7,589 → 12,458 → 21,991 chars.
* Archived plan: 13.4 KiB → 18.0 KiB → 27.6 KiB.

The 340s ceiling was not wrong. It was a fixed line under a rising curve, and
no surface anywhere published the curve.

These tests pin the two things that fixes that: the active ledger is bounded
by construction, and every plan ATTEMPT — including the one that times out —
publishes its duration against a threshold well short of the ceiling.

They deliberately do NOT pin any retry count or ceiling value. Brian's ruling
of 2026-08-14 puts both out of scope as fixes: they make a latency regression
survivable instead of visible.
"""

from __future__ import annotations

import json

import pytest

from director.agent import (
    DIRECTOR_PLAN_AMBER_FRACTION,
    DIRECTOR_PLAN_CEILING_S,
    DIRECTOR_PLAN_MEASURED_MAX_S,
    _carryover_context,
    _carryover_item_count,
    _emit_plan_latency,
    _invoke_with_retry,
    _plan_amber_threshold_s,
    build_messages,
)
from director.carryover import (
    ACTIVE_LEDGER_MAX_ITEMS,
    RETIREMENT_STALE_DAYS,
    merge_plan_into_ledger,
    partition_ledger,
)
from director.schema import ActionItem, DirectorWeeklyActionPlan

RUN_DATE = "2026-08-13"


def _row(rid: str, last_seen: str, **kw) -> dict:
    row = {
        "id": rid, "title": rid, "status": "carried_over",
        "first_seen": "2026-06-05", "last_seen": last_seen,
        "priority": "P2", "proposed_owner": "predictor",
    }
    row.update(kw)
    return row


def _plan_with(ids: list[str]) -> DirectorWeeklyActionPlan:
    return DirectorWeeklyActionPlan(
        run_date=RUN_DATE,
        system_summary="s",
        top_risks=[],
        action_items=[
            ActionItem(
                id=i, title=i, rationale="r", evidence=[],
                proposed_owner="predictor", priority="P1",
                horizon="this_week", suggested_change_type="structural",
                confidence=50,
            )
            for i in ids
        ],
        carryover_review=[],
    )


# ── The ledger is bounded ────────────────────────────────────────────────────

class TestLedgerIsBounded:
    def test_stale_row_retires_and_is_kept_not_deleted(self):
        """The 17 dead rows measured on the live artifact are exactly this
        shape: never re-proposed, no carry_count, no issue number, and — before
        this change — permanent."""
        rows = [_row("stale", "2026-06-18"), _row("fresh", RUN_DATE)]
        active, retiring = partition_ledger(rows, RUN_DATE)
        assert [r["id"] for r in active] == ["fresh"]
        assert [r["id"] for r in retiring] == ["stale"]
        # Retired, not deleted: the row survives with its reason and the run
        # that retired it, so the commitment stays reconstructible from the
        # artifact alone.
        assert retiring[0]["retired_on"] == RUN_DATE
        assert retiring[0]["retired_reason"].startswith("stale:")
        assert retiring[0]["title"] == "stale"

    def test_row_inside_the_staleness_window_survives(self):
        """Exactly at the horizon is NOT stale — the boundary is stated so a
        later reader does not have to infer it from a comparison operator."""
        rows = [_row("edge", "2026-07-16")]  # 28 days before 2026-08-13
        active, retiring = partition_ledger(rows, RUN_DATE)
        assert (len(active), len(retiring)) == (1, 0)
        assert RETIREMENT_STALE_DAYS == 28

    def test_size_cap_retires_the_tail_and_keeps_the_most_recent(self):
        """Without this the active set is bounded only by (items per plan × 4
        weeks), which is not a bound anyone chose."""
        rows = [_row(f"i{n:03d}", RUN_DATE) for n in range(ACTIVE_LEDGER_MAX_ITEMS + 5)]
        active, retiring = partition_ledger(rows, RUN_DATE)
        assert len(active) == ACTIVE_LEDGER_MAX_ITEMS
        assert len(retiring) == 5
        assert all(r["retired_reason"].startswith("over_cap:") for r in retiring)

    def test_size_cap_prefers_higher_priority_at_equal_recency(self):
        rows = [_row("p3", RUN_DATE, priority="P3")]
        rows += [_row(f"p0-{n}", RUN_DATE, priority="P0")
                 for n in range(ACTIVE_LEDGER_MAX_ITEMS)]
        active, retiring = partition_ledger(rows, RUN_DATE)
        assert [r["id"] for r in retiring] == ["p3"]

    def test_partition_is_deterministic(self):
        rows = [_row(f"i{n:03d}", RUN_DATE) for n in range(ACTIVE_LEDGER_MAX_ITEMS + 3)]
        first = [r["id"] for r in partition_ledger(list(rows), RUN_DATE)[0]]
        second = [r["id"] for r in partition_ledger(list(rows), RUN_DATE)[0]]
        assert first == second

    def test_unparseable_last_seen_is_kept_not_retired(self):
        """An age that cannot be established keeps the row visible. The
        conservative direction: a commitment must not vanish on a formatting
        accident."""
        active, retiring = partition_ledger([_row("odd", "not-a-date")], RUN_DATE)
        assert (len(active), len(retiring)) == (1, 0)

    def test_resolved_is_not_a_retirement_trigger(self):
        """Deliberate. Retiring on the resolution would hide the row from
        loop_verification's reopen-if-unrecovered pass, which exists to dispute
        exactly that claim."""
        active, retiring = partition_ledger(
            [_row("done", RUN_DATE, status="resolved")], RUN_DATE
        )
        assert (len(active), len(retiring)) == (1, 0)


class TestMergeRetiresAfterTheUpsert:
    def test_reproposed_row_is_renewed_before_retirement_is_judged(self):
        """A row the Director re-proposed THIS run must not be retired by the
        same call that renewed it — the order of the two operations is the
        whole guarantee."""
        ledger = {"items": [_row("revived", "2026-06-01")]}
        merged = merge_plan_into_ledger(ledger, _plan_with(["revived"]), RUN_DATE)
        assert [r["id"] for r in merged["items"]] == ["revived"]
        assert merged["retired_items"] == []

    def test_merge_moves_stale_rows_to_retired_items(self):
        ledger = {"items": [_row("stale", "2026-06-01")]}
        merged = merge_plan_into_ledger(ledger, _plan_with(["new"]), RUN_DATE)
        assert [r["id"] for r in merged["items"]] == ["new"]
        assert [r["id"] for r in merged["retired_items"]] == ["stale"]

    def test_previously_retired_rows_accumulate_and_are_never_dropped(self):
        ledger = {
            "items": [_row("stale", "2026-06-01")],
            "retired_items": [_row("older", "2026-05-01", retired_on="2026-07-01")],
        }
        merged = merge_plan_into_ledger(ledger, _plan_with(["new"]), RUN_DATE)
        assert {r["id"] for r in merged["retired_items"]} == {"older", "stale"}

    def test_retired_rows_do_not_reach_the_prompt(self):
        """The bound only does anything if `items` is what `build_messages`
        renders. This is the assertion that ties the two files together."""
        ledger = {"items": [_row("stale", "2026-06-01")]}
        merged = merge_plan_into_ledger(ledger, _plan_with(["new"]), RUN_DATE)
        text = _carryover_context(merged)
        assert "stale" not in text
        assert "new" in text


# ── The latency signal ───────────────────────────────────────────────────────

class TestPlanLatencySignal:
    def test_amber_is_a_fraction_of_the_measured_need_not_of_the_ceiling(self):
        """Raising the ceiling must never move the amber line up with it.

        The line was 0.6 x the CEILING until 2026-08-22, which read as
        ceiling-independent and was the opposite: raising the ceiling
        340 -> 600 would have moved amber 204s -> 360s, above every duration
        this call has ever survived, and the trend signal would have gone dark
        exactly as the trend it watches continued (alpha-engine-config-I7311).
        The anchor is now the slowest UNCENSORED call ever measured, so the
        line moves only when the model's own requirement is re-measured.
        """
        assert _plan_amber_threshold_s() == pytest.approx(
            DIRECTOR_PLAN_AMBER_FRACTION * DIRECTOR_PLAN_MEASURED_MAX_S
        )
        assert 0 < DIRECTOR_PLAN_AMBER_FRACTION < 1

    def test_amber_does_not_move_when_the_ceiling_does(self):
        """The regression this file exists to catch, stated as a test."""
        import director.agent as agent

        before = _plan_amber_threshold_s()
        original = agent.DIRECTOR_PLAN_CEILING_S
        try:
            agent.DIRECTOR_PLAN_CEILING_S = original * 3
            assert _plan_amber_threshold_s() == pytest.approx(before), (
                "the amber line tracked a change to DIRECTOR_PLAN_CEILING_S — it "
                "is a trend signal about the model's requirement, not a fraction "
                "of whatever wall the invocation happens to have"
            )
        finally:
            agent.DIRECTOR_PLAN_CEILING_S = original

    def test_amber_fires_below_the_ceiling_on_the_measured_regression(self):
        """205.3s is the 2026-08-14 01:55 call that SUCCEEDED, one invocation
        after two that did not. Amber must already be lit there — a signal that
        only fires once the call has failed is the failure it is replacing."""
        assert _plan_amber_threshold_s() < 205.3 < DIRECTOR_PLAN_CEILING_S
        assert _plan_amber_threshold_s() < DIRECTOR_PLAN_MEASURED_MAX_S

    def test_healthy_call_still_publishes_a_zero(self):
        """observability-policy §9: absence of a signal is never rendered
        healthy. The alarm on this metric is only meaningful because the 0 is
        published every run — the DirectorRouteFallback precedent
        (alpha-engine-config-I6185)."""
        rec = _emit_plan_latency(
            elapsed_s=90.0, outcome="ok", prompt_chars=7589, carryover_items=19,
        )
        assert rec["DirectorPlanLatencyAmber"] == 0
        assert rec["DirectorPlanLatencySeconds"] == 90.0

    def test_amber_record_carries_the_quantities_that_explain_it(self):
        rec = _emit_plan_latency(
            elapsed_s=205.3, outcome="ok", prompt_chars=21991, carryover_items=41,
        )
        assert rec["DirectorPlanLatencyAmber"] == 1
        assert rec["DirectorPlanPromptChars"] == 21991
        assert rec["DirectorPlanCarryoverItems"] == 41

    def test_emf_envelope_declares_every_metric_it_publishes(self, capsys):
        _emit_plan_latency(
            elapsed_s=205.3, outcome="ok", prompt_chars=1, carryover_items=1,
        )
        payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        declared = {
            m["Name"]
            for m in payload["_aws"]["CloudWatchMetrics"][0]["Metrics"]
        }
        assert "DirectorPlanLatencySeconds" in declared
        assert "DirectorPlanLatencyAmber" in declared
        assert payload["_aws"]["CloudWatchMetrics"][0]["Namespace"] == "AlphaEngine/Director"
        for name in declared:
            assert name in payload, f"{name} declared but not present in the record"

    def test_token_counts_are_read_off_the_usage_when_present(self):
        class _Usage:
            input_tokens = 11212
            output_tokens = 23300
            reasoning_tokens = 16000

        rec = _emit_plan_latency(
            elapsed_s=228.0, outcome="ok", prompt_chars=1, carryover_items=1,
            usage=_Usage(),
        )
        assert rec["DirectorPlanPromptTokens"] == 11212
        assert rec["DirectorPlanCompletionTokens"] == 23300
        assert rec["DirectorPlanReasoningTokens"] == 16000

    def test_missing_usage_reports_zero_rather_than_raising(self):
        rec = _emit_plan_latency(
            elapsed_s=1.0, outcome="error:APITimeoutError", prompt_chars=1,
            carryover_items=1, usage=None,
        )
        assert rec["DirectorPlanCompletionTokens"] == 0

    def test_emitter_never_raises_even_when_the_record_cannot_be_encoded(self, monkeypatch):
        """A telemetry failure must not take down the weekly plan — but it is
        logged at ERROR, because a silent emitter is what this exists to end."""
        import director.agent as agent

        def _boom(*a, **kw):
            raise RuntimeError("no json for you")

        monkeypatch.setattr(agent.json, "dumps", _boom)
        rec = agent._emit_plan_latency(
            elapsed_s=1.0, outcome="ok", prompt_chars=1, carryover_items=1,
        )
        assert rec["DirectorPlanLatencySeconds"] == 1.0


class TestEveryAttemptIsMeasured:
    """The attempt that matters most is the one that never returns."""

    def _messages(self, n_items: int) -> list:
        ledger = {"items": [_row(f"i{n}", RUN_DATE) for n in range(n_items)]}
        return build_messages({}, carryover=ledger)

    def test_timed_out_attempt_publishes_its_duration(self, capsys):
        class _AlwaysTimesOut:
            def invoke(self, messages):
                raise RuntimeError("Request timed out.")

        with pytest.raises(RuntimeError):
            _invoke_with_retry(_AlwaysTimesOut(), self._messages(3))
        records = [
            json.loads(line)
            for line in capsys.readouterr().out.strip().splitlines()
            if "DirectorPlanLatencySeconds" in line
        ]
        assert records, (
            "no latency record published for a failing attempt — on 2026-08-14 "
            "the two censored attempts ARE the event"
        )
        assert all(r["outcome"].startswith("error:") for r in records)

    def test_successful_attempt_publishes_the_prompt_size(self, capsys):
        class _Ok:
            last_usage = None

            def invoke(self, messages):
                return _plan_with(["x"])

        _invoke_with_retry(_Ok(), self._messages(7))
        record = [
            json.loads(line)
            for line in capsys.readouterr().out.strip().splitlines()
            if "DirectorPlanLatencySeconds" in line
        ][-1]
        assert record["outcome"] == "ok"
        assert record["DirectorPlanCarryoverItems"] == 7
        assert record["DirectorPlanPromptChars"] > 0

    def test_telemetry_is_stamped_onto_the_archived_plan(self):
        """CloudWatch metrics age out; `director/{date}/action_plan.json` does
        not, and I7311's Closes-when asks for the numbers to be readable per
        invocation from a durable surface."""
        class _Ok:
            last_usage = None

            def invoke(self, messages):
                return _plan_with(["x"])

        plan = _invoke_with_retry(_Ok(), self._messages(2))
        assert plan.plan_call_telemetry["DirectorPlanCarryoverItems"] == 2
        # It survives the round-trip to the archived artifact.
        assert "plan_call_telemetry" in json.loads(plan.model_dump_json())


class TestCarryoverCount:
    def test_count_is_recoverable_from_the_prompt_text(self):
        ledger = {"items": [_row(f"i{n}", RUN_DATE) for n in range(41)]}
        assert _carryover_item_count(build_messages({}, carryover=ledger)) == 41

    def test_absent_carryover_section_counts_zero(self):
        assert _carryover_item_count(build_messages({})) == 0

    def test_hand_built_messages_count_zero_rather_than_raising(self):
        assert _carryover_item_count([("system", "hi"), ("human", "there")]) == 0
