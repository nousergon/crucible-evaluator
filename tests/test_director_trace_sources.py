"""The Director's ``{cite=...}`` lines as ``llm_trace.v1`` sources
(alpha-engine-config-I12312, audit §6.3 item 4a).

krepis writes one ``llm_trace.v1`` row per call when a trace sink is configured,
and that row's ``citations[]`` lists the sources the caller declared. These
tests pin the Director's declaration:

- every id the prompt offered becomes one source whose ``source_id`` is the
  typed id (``metric:<name>``, ``issue:alpha-engine-config-I<N>``, ...);
- each source's recorded text, sha256 and offsets are of characters actually
  sent, and declaring them leaves the prompt byte-identical;
- both calls pass them, and stamp the call's ``trace_id`` on the artifact so
  the archived ``citation_check`` joins the trace row;
- a fault building them never blocks the call;
- the handler flushes the trace sink in the same ``finally`` as the cost sink.
"""

from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("krepis.llm_trace")

from krepis import llm_trace as lt  # noqa: E402
from krepis.llm import LLMResult, LLMUsage  # noqa: E402

from director import agent, retro  # noqa: E402
from director.agent import _KrepisStructuredDirector  # noqa: E402
from director.citations import (  # noqa: E402
    CITATION_CONTRACT_VERSION,
    TRACE_IDS_KEY,
    TRACE_SOURCE_KINDS,
    KINDS,
    stamp_trace_ids,
    trace_sources,
    trace_sources_or_none,
)
from director.retro import _KrepisStructuredJudge  # noqa: E402
from director.schema import DirectorWeeklyActionPlan, RetroGrade  # noqa: E402

RUN_DATE = "2026-10-10"
TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
SPAN_ID = "00f067aa0ba902b7"

CARD = {
    "_provenance": {"run_date": RUN_DATE},
    "tiles_overall_status": "RED",
    "attestation": {"verdict": "PASS", "as_of": {}},
    "degraded_component_census": False,
    "tiles": {
        "predictor": {"status": "RED", "letter": "F", "numeric_grade": 40.0, "components": [
            {"name": "momentum_l1_ic", "criticality": "critical", "status": "RED",
             "value": -0.0015, "target": 0.03, "red_line": 0.0, "status_reason": "dead L1."},
        ]},
        "portfolio_outcome": {"status": "WATCH", "letter": "C", "components": [
            {"name": "excess_return", "criticality": "critical", "status": "WATCH", "value": -0.01,
             "window": {"label": "13w", "sources": [
                 "s3://alpha-engine-research/trades/eod_pnl.csv",
                 "s3://alpha-engine-research/trades/fills.csv",
             ]}},
        ]},
    },
}
LEDGER = {"items": [{"id": "fix-pnl", "title": "Fix PnL", "status": "carried_over",
                     "priority": "P1", "proposed_owner": "data",
                     "evidence": ["excess_return"]}]}


def _plan_prompt():
    messages, reg = agent.build_messages_with_sources(
        CARD, carryover=LEDGER,
        roadmap_digest="#123 [P1] some open work\nnot an issue line",
        resolved_digest="#99 (closed 2026-10-01) prior fix",
    )
    _, user_content = agent._split_messages(messages)
    return messages, user_content, reg


# ── the declaration ─────────────────────────────────────────────────────────


class TestTraceSources:
    def test_every_offered_id_is_a_source_with_its_typed_id(self):
        _, user, reg = _plan_prompt()
        rs = trace_sources(user, reg)
        assert isinstance(rs, lt.RenderedSources)
        assert rs.text == user
        assert {s.block.source_id for s in rs.sources} == set(reg.ids())
        assert {"metric:momentum_l1_ic", "carryover:fix-pnl",
                "issue:alpha-engine-config-I123"} <= set(reg.ids())

    def test_text_hash_and_offsets_are_of_the_characters_sent(self):
        _, user, _ = _plan_prompt()
        for s in trace_sources(user).sources:
            text = user[s.char_start:s.char_end]
            assert text == s.block.text
            assert s.content_sha256 == lt.sha256_hex(text)
            # Each span ends with its own tag, so the hashed text names its id.
            assert text.endswith("{cite=" + s.block.source_id + "}")
            assert "\n" not in text

    def test_two_tags_on_one_line_get_disjoint_spans(self):
        _, user, _ = _plan_prompt()
        arts = [s for s in trace_sources(user).sources
                if s.block.source_id.startswith("artifact:s3://")]
        assert len(arts) == 2
        a, b = sorted(arts, key=lambda s: s.char_start)
        assert a.char_end <= b.char_start
        assert "{cite=" + a.block.source_id + "}" not in b.block.text

    def test_kinds_cids_and_meta(self):
        _, user, _ = _plan_prompt()
        sources = trace_sources(user).sources
        assert set(TRACE_SOURCE_KINDS) == set(KINDS)
        assert set(TRACE_SOURCE_KINDS.values()) <= lt.SOURCE_KINDS
        cids = [s.block.cid for s in sources]
        assert len(cids) == len(set(cids))
        for s in sources:
            kind = s.block.source_id.split(":", 1)[0]
            assert s.block.kind == TRACE_SOURCE_KINDS[kind]
            assert s.block.meta["director_kind"] == kind
            assert s.block.meta["contract"] == CITATION_CONTRACT_VERSION

    def test_a_repeated_tag_is_one_id_with_numbered_occurrences(self):
        user = "a {cite=metric:x}\nb {cite=tile:p}\nc {cite=metric:x}"
        got = [(s.block.cid, s.block.source_id, s.block.meta["occurrence"])
               for s in trace_sources(user).sources]
        assert got == [("s1", "metric:x", 0), ("s2", "tile:p", 0), ("s1.1", "metric:x", 1)]

    def test_deterministic(self):
        _, user, reg = _plan_prompt()
        assert trace_sources(user, reg) == trace_sources(user, reg)

    def test_registry_filters_tags_it_did_not_offer(self):
        user = "row quoting {cite=metric:ghost}\nreal {cite=metric:x}"
        from director.citations import SourceRegistry

        reg = SourceRegistry()
        reg.offer("metric", "x", "real")
        assert [s.block.source_id for s in trace_sources(user, reg).sources] == ["metric:x"]
        assert len(trace_sources(user).sources) == 2

    def test_the_instruction_placeholder_is_not_a_source(self):
        assert trace_sources(agent.PLAN_CITATION_INSTRUCTIONS) is None
        assert trace_sources("") is None

    def test_krepis_binds_it_without_changing_the_prompt(self):
        _, user, _ = _plan_prompt()
        rs = trace_sources(user)
        bound, rendered = lt.bind_sources(user, rs)
        assert bound == user
        assert rendered == rs.sources

    def test_the_trace_row_carries_the_typed_ids(self):
        """End to end through krepis' own row builder: the ``citations[]``
        a Director call's ``llm_trace.v1`` row would carry."""
        system, user = agent._split_messages(_plan_prompt()[0])
        _, rendered = lt.bind_sources(user, trace_sources(user))
        result = LLMResult(
            text="{}", model="glm-5.2", provider="litellm_proxy", usage=LLMUsage(),
            raw_request={"messages": [{"role": "system", "content": system},
                                      {"role": "user", "content": user}]},
            dlp_verdict="ok", trace_id=TRACE_ID, span_id=SPAN_ID,
        )
        t0 = datetime(2026, 10, 16, tzinfo=timezone.utc)
        row, _ = lt.build_trace_record(
            result, callsite_id="director-plan",
            context=lt.TraceContext(trace_id=TRACE_ID), span_id=SPAN_ID,
            started_at=t0, ended_at=t0, duration_ms=1.0, request_model="high",
            sources=rendered, system=system,
        )
        cits = row["krepis"]["citations"]
        assert {c["source_id"] for c in cits} >= {
            "metric:momentum_l1_ic", "carryover:fix-pnl", "issue:alpha-engine-config-I99"}
        metric = next(c for c in cits if c["source_id"] == "metric:momentum_l1_ic")
        assert metric["kind"] == "metric"
        span = metric["span"]
        assert user[span["char_start"]:span["char_end"]].endswith("{cite=metric:momentum_l1_ic}")
        json.dumps(row)  # the row is JSON as written


# ── the two calls ───────────────────────────────────────────────────────────


class _Result:
    def __init__(self, parsed, *, trace_id=TRACE_ID, span_id=SPAN_ID):
        self.parsed = parsed
        self.model = "glm-5.2"
        self.usage = None
        self.trace_id = trace_id
        self.span_id = span_id


class _Client:
    def __init__(self, parsed):
        self._parsed = parsed
        self.kwargs = None

    def structured(self, **kwargs):
        self.kwargs = kwargs
        return _Result(self._parsed)


def _empty_plan():
    return DirectorWeeklyActionPlan(run_date=RUN_DATE, system_summary="s",
                                    top_risks=["r"], action_items=[])


class TestCalls:
    def test_plan_call_declares_sources_and_stamps_the_trace(self):
        messages, user, reg = _plan_prompt()
        client = _Client(_empty_plan())
        plan = _KrepisStructuredDirector(client, director_model="ultra-glm-5.2-direct").invoke(messages)
        sent = client.kwargs
        assert sent["user_content"] == user
        assert {s.block.source_id for s in sent["sources"].sources} == set(reg.ids())
        assert getattr(plan, TRACE_IDS_KEY) == {"trace_id": TRACE_ID, "span_id": SPAN_ID}
        assert json.loads(plan.model_dump_json())[TRACE_IDS_KEY]["trace_id"] == TRACE_ID

    def test_retro_call_declares_sources_and_stamps_the_trace(self):
        prior = {"run_date": "2026-10-03", "action_items": [
            {"id": "revive-momentum-l1", "title": "Revive", "priority": "P1",
             "rationale": "r", "proposed_owner": "predictor", "confidence": 60}]}
        messages, reg = retro.build_messages_with_sources(prior, CARD)
        grade = RetroGrade(prior_run_date="2026-10-03", grounding=70, calibration=60,
                           actionability=80)
        client = _Client(grade)
        judge = _KrepisStructuredJudge(client, judge_group="high", judge_model="m")
        out = judge.invoke(messages)
        ids = {s.block.source_id for s in client.kwargs["sources"].sources}
        assert ids == set(reg.ids())
        assert any(i.startswith("plan_item:") for i in ids)
        assert getattr(out, TRACE_IDS_KEY)["trace_id"] == TRACE_ID

    def test_a_prompt_with_no_tags_sends_no_sources(self):
        client = _Client(_empty_plan())
        _KrepisStructuredDirector(client, director_model="m").invoke(
            [("system", "s"), ("human", "no citable lines here")])
        assert client.kwargs["sources"] is None

    def test_a_fault_building_sources_never_blocks_the_call(self, monkeypatch):
        import director.citations as c

        def boom(*a, **k):
            raise RuntimeError("bad source")

        monkeypatch.setattr(c, "trace_sources", boom)
        assert trace_sources_or_none("x {cite=metric:a}") is None

    def test_a_result_without_trace_ids_stamps_nothing(self):
        plan = _empty_plan()

        class Bare:
            pass

        stamp_trace_ids(plan, Bare())
        assert TRACE_IDS_KEY not in plan.model_dump()


# ── the flush ───────────────────────────────────────────────────────────────


def test_run_director_flushes_the_trace_sink_in_its_finally():
    tree = ast.parse((Path(__file__).resolve().parents[1] / "director" / "handler.py").read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run_director")
    finals = [n for n in ast.walk(fn) if isinstance(n, ast.Try) and n.finalbody]
    assert finals
    guarded = [
        n for t in finals for stmt in t.finalbody for n in ast.walk(stmt)
        if isinstance(n, ast.Try) and n.handlers
        and any(isinstance(a, ast.alias) and a.name == "flush_default_trace_sink"
                for a in ast.walk(n))
    ]
    assert guarded, "run_director's finally must flush krepis.trace_sink, exception-guarded"
