"""The `director-plan` leg emits a cost row even when the model is UNPRICED.

``alpha-engine-config-I11298``. The 2026-09-19 SCHEDULED weekly run
(``3c2fe2f8-…_d0e3eae8-…``, 09:00:49Z → 14:25:00Z) FAILED at its LAST state,
``AggregateCosts``, at 19,450 s of 19,451 s, with::

    CostCoverageError: 1 stage(s) ran and emitted no cost record: director-plan
    observed:  ['director-retro-judge', 'evaljudge-sync', 'single-agent-quant']
    missing:   ['director-plan']

5 h 24 m of work was discarded. The Director itself had run fine — ``status:
ok``, 11 action items, plan written, digest sent.

**Root cause, measured** — ``/aws/lambda/alpha-engine-evaluator-director``,
2026-09-19T14:19:56.215Z::

    ERROR [evaluator-director] cost emission failed for
    callsite_id=director-plan model=glm-5.3:
    No price card for model 'glm-5.3' active on 2026-09-19

``glm-5.3`` became the ``ultra`` group primary on 2026-09-12 and carried no
price card. ``krepis.cost.recompute_cost`` raised ``PriceCardLookupError``, it
propagated out of ``record_llm_call``, and ``LLMClient._emit_cost_record``'s
deliberate non-raising ``except`` swallowed THE WHOLE RECORD — token counts
included, which do not depend on the price table at all. Nothing was buffered,
so the handler's ``finally: flush_default_sink()`` had nothing under
``director-plan`` to write, and fan-in coverage — which is keyed on OBJECT
EXISTENCE (``cost_coverage.observed_producers`` derives the producer from the
S3 KEY, never from a row field) — correctly reported the stage as silent.

The retro judge leg of the SAME invocation (``run_id krepis-e1e86b1ea07b``) was
served ``deepseek-v4-pro``, which HAS a price card, and flushed normally at
14:23:47Z. That asymmetry is what rules out every process-level hypothesis: the
sink was configured, the flush ran, and both legs shared one process and one
run id.

**The fix** is ``krepis`` PR226 / v0.59.64 (``alpha-engine-config-I11100``):
``record_llm_call`` degrades a ``PriceCardLookupError`` to ``cost_source:
"unpriced"`` with ``cost_usd: None`` and the row STILL EMITTED, so the gap is
visible in the ledger instead of absent from it. This repo's pin was raised to
``0.59.64`` in PR #323.

These are therefore **lockstep guard tests on the pin**, in the sense
``~/Development/CLAUDE.md`` means it: they fail if the pin is ever lowered
below the release that carries the degradation, or if the behaviour regresses
upstream. The alternative — trusting a requirements comment — is what the
2026-09-19 run cost.
"""

from __future__ import annotations

import pytest

pytest.importorskip("krepis.cost", reason="krepis is a deploy-time dependency")

from krepis.cost import PriceCardLookupError, record_llm_call  # noqa: E402
from krepis.llm import LLMClient, LLMResult, LLMUsage  # noqa: E402
from krepis.llm_config import ModelSpec  # noqa: E402

#: A model id no price card can plausibly carry. Deliberately NOT ``glm-5.3``:
#: that one is priced now, so asserting on it would test the price table rather
#: than the degradation path, and would go quiet the next time a newly-promoted
#: primary arrives unpriced — which is the whole failure mode.
UNPRICED_MODEL = "no-such-model-unpriced-fixture-v0"

#: The production plan call's callsite id, and the join key between the emitted
#: cost row and the weekly SF's ``coverage.required_producers`` entry
#: ``{"Director": ["director-plan"]}``. Kept as a literal here on purpose: a
#: test that imported the constant would still pass if both moved together,
#: and the SF's declaration is a hand-maintained JSON string that cannot.
DIRECTOR_PLAN_CALLSITE = "director-plan"


def _result(model: str = UNPRICED_MODEL) -> LLMResult:
    """An LLMResult shaped like a completed streamed structured plan call."""
    return LLMResult(
        text="{}",
        model=model,
        provider="litellm_proxy",
        usage=LLMUsage(
            input_tokens=19248,
            output_tokens=40401,
            reasoning_tokens=33279,
            reasoning_tokens_max_attempt=33279,
            attempts=1,
        ),
        raw_request={},
        raw_response=None,
        streamed=True,
        finish_reason="stop",
    )


def test_record_llm_call_degrades_an_unpriced_model_instead_of_raising():
    """The upstream half of the fix: the row is built, not dropped."""
    record = record_llm_call(_result(), extra_fields={"callsite_id": DIRECTOR_PLAN_CALLSITE})

    assert record["cost_source"] == "unpriced", (
        "a served model with no price card must degrade the row, not drop it — "
        "krepis PR226 / v0.59.64 (alpha-engine-config-I11100). A raise here "
        "means the pin has been lowered below that release and the "
        "2026-09-19 AggregateCosts breach can recur."
    )
    assert record["cost_usd"] is None
    # The counts do not depend on the price table and must survive the gap:
    # losing them is what made the 2026-09-19 spend unattributable rather than
    # merely unpriced.
    assert record["input_tokens"] == 19248
    assert record["output_tokens"] == 40401
    assert record["reasoning_tokens"] == 33279
    assert record["callsite_id"] == DIRECTOR_PLAN_CALLSITE


def test_a_priced_model_is_still_priced():
    """The degradation must not swallow the ordinary path with it."""
    record = record_llm_call(
        _result(model="deepseek-v4-pro"),
        extra_fields={"callsite_id": "director-retro-judge"},
    )
    assert record["cost_source"] == "price_card"
    assert record["cost_usd"] is not None and record["cost_usd"] > 0


def test_a_director_plan_call_on_an_unpriced_model_reaches_the_sink():
    """The failure as the weekly SF sees it: a row under ``director-plan``.

    ``AggregateCosts`` does not read ``cost_usd`` — it reads whether an object
    named for the callsite exists. So the assertion that matters is that the
    sink is HANDED a record for this callsite, whatever the pricing outcome.
    """
    captured: list[dict] = []
    client = LLMClient(
        ModelSpec(provider="openai", model=UNPRICED_MODEL, max_tokens=1024),
        callsite_id=DIRECTOR_PLAN_CALLSITE,
        api_key="not-used-no-call-is-made",
        cost_sink=captured.append,
    )

    client._emit_cost_record(_result())

    assert len(captured) == 1, (
        "a Director invocation that made an LLM call ended with NO "
        "director-plan cost record — exactly the 2026-09-19 breach "
        "(alpha-engine-config-I11298)"
    )
    assert captured[0]["callsite_id"] == DIRECTOR_PLAN_CALLSITE
    assert captured[0]["cost_source"] == "unpriced"


def test_price_card_lookup_error_is_still_a_real_exception():
    """Guard the guard: the degradation is a CAUGHT raise, not a removed one.

    If ``PriceCardLookupError`` ever stops existing, the tests above would keep
    passing while meaning something else entirely.
    """
    assert issubclass(PriceCardLookupError, Exception)
