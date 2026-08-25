"""The Director's plan call streams — alpha-engine-config-I8164.

The failure this removes: on 2026-08-22 four consecutive plan attempts were
censored at the then-340s per-attempt ceiling with ZERO completion tokens each,
taking the weekly run and its rerun down (alpha-engine-config-I8151). After
~340s of a wholly opaque socket the only fact each failure carried was its own
duration — which is why raising the ceiling to 600s moved the cliff instead of
removing it. Streaming replaces a bound on WAITING with a bound on SILENCE, and
a bound on silence fails with the partial generation, the chunk count and the
elapsed time attached.

Every assertion here is about a property that is invisible from the outside
until the weekly run fires. `stream=True` reaching the wire, the idle budget
being the one this module derived rather than a library default, and the
requirement being declared at RESOLVE time are each things that would look
completely healthy in CI while being wrong in production — the plan call runs
once a week, so the first honest signal is seven days out.

The ordering test at the bottom is the one that survives a refactor: krepis
warns and then proceeds when `idle_timeout >= LLMClient(timeout=...)`, because
the transport's read deadline binds first and the inter-chunk budget can never
fire. A warning in a Lambda log is not a signal anyone reads, so the ordering is
asserted here instead of trusted.
"""

import pytest

from director import agent as agent_mod
from director.agent import (
    DIRECTOR_PLAN_CEILING_S,
    DIRECTOR_PLAN_IDLE_TIMEOUT_S,
    _KrepisStructuredDirector,
)
from director.schema import DirectorWeeklyActionPlan

RUN_DATE = "2026-08-29"
PRIMARY = "glm-5.2"

MESSAGES = [
    {"role": "system", "content": "s"},
    {"role": "user", "content": "u"},
]


def _plan() -> DirectorWeeklyActionPlan:
    return DirectorWeeklyActionPlan(
        run_date=RUN_DATE,
        system_summary="A summary.",
        top_risks=["a risk"],
        action_items=[],
    )


class _Result:
    """Stand-in for the object krepis returns — streamed or not, it is the same
    shape, which is the point of re-assembling the stream inside the library."""

    def __init__(self, model: str):
        self.model = model
        self.parsed = _plan()
        self.usage = None


class _RecordingClient:
    """Captures the kwargs `structured()` was called with."""

    def __init__(self, served_model: str = PRIMARY):
        self._served = served_model
        self.calls: list[dict] = []

    def structured(self, **kwargs):
        self.calls.append(kwargs)
        return _Result(self._served)


# ── stream=True reaches the wire ─────────────────────────────────────────────


class TestStreamingIsRequested:
    def test_structured_is_called_with_stream_true(self):
        client = _RecordingClient()
        llm = _KrepisStructuredDirector(
            client, director_model="ultra-glm-5.2-direct", primary_model=PRIMARY,
        )
        llm.invoke(MESSAGES)

        assert len(client.calls) == 1
        assert client.calls[0]["stream"] is True, (
            "the plan call must stream. Without it the per-attempt ceiling is "
            "again a bound on an opaque socket, and a censored attempt carries "
            "nothing but its own duration."
        )

    def test_an_idle_timeout_is_always_passed_explicitly(self):
        """Never inherited from krepis' DEFAULT_STREAM_IDLE_TIMEOUT_S.

        The two are the same number today. That is a coincidence, and letting a
        library default chosen for an unknown call site silently move this call
        site's liveness bound is how a knob stops meaning what its comment says
        — the same reason `attempts=_STRUCTURED_ATTEMPTS` is explicit here.
        """
        client = _RecordingClient()
        llm = _KrepisStructuredDirector(
            client, director_model="ultra-glm-5.2-direct", primary_model=PRIMARY,
        )
        llm.invoke(MESSAGES)

        assert client.calls[0]["idle_timeout"] == DIRECTOR_PLAN_IDLE_TIMEOUT_S

    def test_the_idle_budget_is_injectable_without_touching_the_constant(self):
        client = _RecordingClient()
        llm = _KrepisStructuredDirector(
            client, director_model="ultra-glm-5.2-direct",
            primary_model=PRIMARY, idle_timeout_s=12.5,
        )
        llm.invoke(MESSAGES)

        assert client.calls[0]["idle_timeout"] == 12.5

    def test_streaming_does_not_change_what_invoke_returns(self):
        """krepis re-assembles a streamed response into the same shape.

        The route-degradation stamp (I8165) reads `result.model` and must keep
        working across this change — a streamed call that quietly stopped
        stamping degradation would let a weaker arm's plan enter the record as
        if the champion wrote it, which is the condition Brian's ruling
        admitting that arm was made under.
        """
        client = _RecordingClient(served_model="deepseek-v4-pro")
        llm = _KrepisStructuredDirector(
            client, director_model="ultra-glm-5.2-direct", primary_model=PRIMARY,
        )
        plan = llm.invoke(MESSAGES)

        assert plan.resolved_model == "deepseek-v4-pro"
        assert getattr(plan, agent_mod.PLAN_KEY_ROUTE_DEGRADED) is True

    def test_the_single_attempt_multiplier_survives(self):
        """`attempts` stays at 1 alongside the new kwargs.

        Adding parameters to this call is exactly where `attempts=2` would creep
        back in by accident, and it silently doubled every quoted budget the
        last time it was implicit (config#7126).
        """
        client = _RecordingClient()
        llm = _KrepisStructuredDirector(
            client, director_model="ultra-glm-5.2-direct", primary_model=PRIMARY,
        )
        llm.invoke(MESSAGES)

        assert client.calls[0]["attempts"] == agent_mod._STRUCTURED_ATTEMPTS == 1


# ── the idle budget must be able to fire ─────────────────────────────────────


class TestIdleBudgetOrdering:
    def test_idle_timeout_is_strictly_below_the_per_attempt_ceiling(self):
        """Otherwise the transport's read deadline binds first and this is moot.

        krepis logs a warning and proceeds in that case. A warning inside a
        weekly Lambda is not a signal anyone reads, so the ordering is asserted
        here — the config knob that quietly has no effect is precisely the
        failure mode this change exists to remove, and reintroducing it one
        layer up would be worse than not streaming at all.
        """
        assert 0 < DIRECTOR_PLAN_IDLE_TIMEOUT_S < DIRECTOR_PLAN_CEILING_S

    def test_the_idle_budget_leaves_room_for_the_measured_first_chunk(self):
        """Measured 2026-08-25 through each ultra arm's egress-proxy path:

            glm-5.2-direct    979 chunks / 83.2s   largest silence 3.72s
            deepseek-v4-pro  5529 chunks / 90.0s   largest silence 1.64s

        In both runs the largest silence in the whole call WAS the time to the
        first chunk. Those probes used ~100-token prompts and the Director's is
        ~11k, and time-to-first-chunk is prefill-dominated, so 3.72s is a floor
        on the real first-chunk wait rather than an estimate of it. This asserts
        the headroom is generous by design: a budget tightened toward the
        measured figure would abort a healthy plan call and look exactly like
        the outage it replaced.
        """
        measured_largest_silence_s = 3.72
        assert DIRECTOR_PLAN_IDLE_TIMEOUT_S >= 20 * measured_largest_silence_s


# ── the requirement is declared at resolve time ──────────────────────────────


class TestStreamingIsRequiredAtResolveTime:
    def test_default_llm_resolves_with_requires_streaming(self, monkeypatch):
        """R32: the consumer names the shape it requires; the derivation drops
        members that do not declare it BEFORE a primary is chosen.

        Without this, the chain is filtered only on reachability and the first
        `structured(stream=True)` raises `StreamingUnsupportedError` deep inside
        the call — the same outcome, reported 600s later, with less to act on
        and a paid connection already open.
        """
        captured: dict = {}

        def _fake_resolve(group, **kwargs):
            captured["group"] = group
            captured.update(kwargs)
            return (
                _FakeSpec(),
                {
                    "route": "litellm_proxy",
                    "deployment_id": "ultra-glm-5.2-direct",
                    "auth_token_type": "litellm_master_key",
                    "primary_model": PRIMARY,
                    "primary_registry_id": "glm-5.2-direct",
                    "api_base_url": "https://router.invalid",
                    "provider": "litellm",
                    "skipped_entries": [],
                },
            )

        _install_fake_krepis(monkeypatch, _fake_resolve)

        agent_mod._default_llm()

        assert captured["group"] == agent_mod.DIRECTOR_GROUP == "ultra"
        assert captured["requires"] == ("streaming",), (
            "the plan call streams, so `streaming` is part of the request shape "
            "the registry must be filtered on — not an assumption made after "
            "a primary has already been chosen."
        )


class _FakeSpec:
    provider = "litellm"
    model = "glm-5.2"
    max_tokens = 65536
    transport = "openai"
    api_key_env = None
    supports_streaming = True


def _install_fake_krepis(monkeypatch, resolve_fn):
    """Stand in for `krepis.router` / `krepis.llm`, which `_default_llm` imports
    lazily so the grading path and most of this suite need neither a key nor the
    provider SDKs."""
    import sys
    import types

    router = types.ModuleType("krepis.router")
    router.resolve_group_spec = resolve_fn

    llm = types.ModuleType("krepis.llm")

    class _LLMClient:
        def __init__(self, spec, **kwargs):
            self.spec = spec
            self.kwargs = kwargs

    llm.LLMClient = _LLMClient

    krepis_pkg = sys.modules.get("krepis") or types.ModuleType("krepis")
    monkeypatch.setitem(sys.modules, "krepis", krepis_pkg)
    monkeypatch.setitem(sys.modules, "krepis.router", router)
    monkeypatch.setitem(sys.modules, "krepis.llm", llm)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
