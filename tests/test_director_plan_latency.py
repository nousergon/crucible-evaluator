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
    CARRYOVER_PROMPT_MAX_ITEMS,
    _carryover_omitted_count,
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

    def test_amber_fires_below_the_ceiling_and_below_the_measured_max(self):
        """Amber must be lit BEFORE the call fails, and must not be the wall.

        Re-anchored 2026-08-22, 228.0 -> 356.9 (alpha-engine-config-I8163):
        two uncensored calls (231.4s on 08-15, 356.9s on 08-22, both
        `outcome=ok`) exceeded the old anchor, so 0.9 x 228 = 205.2s sat below
        every duration the call had recorded since and `DirectorPlanLatencyAmber`
        would have pinned at 1 forever. An alarm that is always on is
        indistinguishable from one that is stuck — the same failure this file's
        `DirectorRouteFallback` precedent exists to prevent.

        What is pinned here is the RELATIONSHIP, not either literal: amber sits
        strictly below the slowest call known to succeed and strictly below the
        ceiling, so a future re-anchor (alpha-engine-config-I8200) cannot
        silently move it above the wall or onto it.
        """
        amber = _plan_amber_threshold_s()
        assert amber < DIRECTOR_PLAN_MEASURED_MAX_S < DIRECTOR_PLAN_CEILING_S
        assert amber == pytest.approx(0.9 * DIRECTOR_PLAN_MEASURED_MAX_S)
        # 356.9s, 2026-08-22 16:04Z, outcome=ok, 32,643 completion tokens — the
        # slowest call the Director is known to have needed. Amber is lit there.
        assert amber < 356.9

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
        # 356.9s / 48,240 chars / 32 carry-over items — the 2026-08-22 16:04Z
        # call, read from this module's own EMF records. At the re-anchored
        # 356.9s max this is exactly ON the measured requirement and therefore
        # above amber.
        rec = _emit_plan_latency(
            elapsed_s=356.9, outcome="ok", prompt_chars=48240, carryover_items=32,
            carryover_omitted=0,
        )
        assert rec["DirectorPlanLatencyAmber"] == 1
        assert rec["DirectorPlanPromptChars"] == 48240
        assert rec["DirectorPlanCarryoverItems"] == 32
        assert rec["DirectorPlanCarryoverOmitted"] == 0

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

    def test_amber_cannot_exceed_a_tight_ceiling_quote(self):
        """Measured 2026-08-21 (`director/2026-08-21/action_plan.json`,
        `plan_call_telemetry`): DirectorPlanAmberSeconds=205.2 published above
        DirectorPlanCeilingSeconds=120.0 for that attempt — amber structurally
        could never fire, because a call quoted 120s either finishes under
        120s or is killed by the ceiling before reaching 205.2s. The amber
        line must be clamped to the effective (quoted) ceiling, not just the
        static one, and the clamp must be visible on the record."""
        rec = _emit_plan_latency(
            elapsed_s=100.0, outcome="ok", prompt_chars=1, carryover_items=1,
            ceiling_s=120.0,
        )
        assert rec["DirectorPlanAmberSeconds"] < rec["DirectorPlanCeilingSeconds"]
        assert rec["DirectorPlanAmberClampedToCeiling"] == 1

    def test_amber_clamp_is_not_set_under_a_generous_ceiling(self):
        rec = _emit_plan_latency(
            elapsed_s=100.0, outcome="ok", prompt_chars=1, carryover_items=1,
            ceiling_s=DIRECTOR_PLAN_CEILING_S,
        )
        assert rec["DirectorPlanAmberClampedToCeiling"] == 0
        assert rec["DirectorPlanAmberSeconds"] == pytest.approx(
            round(_plan_amber_threshold_s(), 1)
        )

    def test_missing_usage_is_flagged_unmeasured_not_a_silent_zero(self):
        """`principles.md` §2.7 / observability-policy: a successful call
        reporting 0 tokens is indistinguishable from a free call unless
        something else says the 0 is unmeasured rather than real."""
        rec = _emit_plan_latency(
            elapsed_s=90.0, outcome="ok", prompt_chars=1, carryover_items=1,
            usage=None,
        )
        assert rec["DirectorPlanCompletionTokens"] == 0
        assert rec["DirectorPlanTokensUnmeasured"] == 1

    def test_krepis_usage_unknown_sentinel_is_flagged_unmeasured(self):
        """krepis's own sentinel for "the provider returned no usage block for
        at least one attempt" (LLMUsage.usage_unknown, alpha-engine-config-
        I8164) — zero token counts alongside it must not read as measured."""
        class _UnknownUsage:
            input_tokens = 0
            output_tokens = 0
            reasoning_tokens = 0
            usage_unknown = True

        rec = _emit_plan_latency(
            elapsed_s=90.0, outcome="ok", prompt_chars=1, carryover_items=1,
            usage=_UnknownUsage(),
        )
        assert rec["DirectorPlanTokensUnmeasured"] == 1

    def test_real_usage_is_not_flagged_unmeasured(self):
        class _Usage:
            input_tokens = 11212
            output_tokens = 23300
            reasoning_tokens = 16000
            usage_unknown = False

        rec = _emit_plan_latency(
            elapsed_s=228.0, outcome="ok", prompt_chars=1, carryover_items=1,
            usage=_Usage(),
        )
        assert rec["DirectorPlanTokensUnmeasured"] == 0

    def test_emf_envelope_declares_the_new_metrics_too(self, capsys):
        _emit_plan_latency(
            elapsed_s=90.0, outcome="ok", prompt_chars=1, carryover_items=1,
        )
        payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        declared = {
            m["Name"]
            for m in payload["_aws"]["CloudWatchMetrics"][0]["Metrics"]
        }
        assert "DirectorPlanTokensUnmeasured" in declared
        assert "DirectorPlanAmberClampedToCeiling" in declared

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
        """The count is what the prompt CARRIES, not what the ledger holds.

        Changed by alpha-engine-config-I8163: a 41-row ledger now renders 20
        rows plus one counted elision line, so `DirectorPlanCarryoverItems`
        reports 20 and `DirectorPlanCarryoverOmitted` reports 21. Reporting 41
        here would name a quantity the call no longer pays for — the metric
        exists to explain the duration.
        """
        ledger = {"items": [_row(f"i{n}", RUN_DATE) for n in range(41)]}
        messages = build_messages({}, carryover=ledger)
        assert _carryover_item_count(messages) == CARRYOVER_PROMPT_MAX_ITEMS
        assert _carryover_omitted_count(messages) == 41 - CARRYOVER_PROMPT_MAX_ITEMS

    def test_absent_carryover_section_counts_zero(self):
        assert _carryover_item_count(build_messages({})) == 0

    def test_hand_built_messages_count_zero_rather_than_raising(self):
        assert _carryover_item_count([("system", "hi"), ("human", "there")]) == 0
