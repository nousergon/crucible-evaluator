"""alpha-engine-config-I7179 — the Director's LLM spend is attributed to nothing.

`director/agent.py::_default_llm` builds `krepis.llm.LLMClient(...)` with no
`cost_sink`, for the pipeline's single most expensive LLM call (`ultra`
group). The fix is NOT a per-call-site `cost_sink=` — that reproduces the gap
for the next call site added. krepis 0.57.0 (krepis-PR140) makes `LLMClient`
resolve a default cost sink from the environment
(`KREPIS_COST_SINK_BUCKET` / `KREPIS_COST_SINK_PREFIX`) when the call site
doesn't pass one, so the fix is to land those two variables on the Director
Lambda's environment — never on the Report Card Lambda, which makes zero LLM
calls (measured: `grading/` has no `LLMClient` construction anywhere).

A bare `update-function-configuration --environment` REPLACES the whole
variable map, which is exactly what `test_deploy_drift.py`'s sibling,
`test_deploy_sh_authority.py::test_director_update_does_not_overwrite_the_live_environment`,
already guards against for `DIRECTOR_ENABLED` / `KREPIS_LITELLM_PROXY_URL` —
operator-set values invisible to this repo. `krepis.aws merge-lambda-env`
(also krepis 0.57.0) is a read-modify-write merge, so it can add these two
variables with that same safety property.

These tests pin: both cost-sink literals land on the Director function via a
merge (not a replace), the Report Card function is untouched, and
`director/agent.py` does not construct an `S3JsonlCostSink` itself — the sink
is an environment fact resolved inside krepis, and a later "helpful"
call-site retrofit is the regression this guards against.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

DEPLOY_SH = Path(__file__).resolve().parents[1] / "infrastructure" / "deploy.sh"
AGENT_PY = Path(__file__).resolve().parents[1] / "director" / "agent.py"
RETRO_PY = Path(__file__).resolve().parents[1] / "director" / "retro.py"

EXPECTED_BUCKET = "alpha-engine-research"
EXPECTED_PREFIX = "decision_artifacts/_cost_raw"


def _script() -> str:
    return DEPLOY_SH.read_text(encoding="utf-8")


def _executable_lines(text: str) -> list[str]:
    """Lines with comments stripped — a rule about behaviour, not prose."""
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        out.append(line.split("#", 1)[0])
    return out


def _merge_lambda_env_lines(text: str) -> list[str]:
    return [ln for ln in _executable_lines(text) if "merge-lambda-env" in ln]


def test_deploy_sh_merges_cost_sink_onto_the_director_function():
    hits = _merge_lambda_env_lines(_script())
    assert hits, (
        "deploy.sh no longer invokes `krepis.aws merge-lambda-env` — the "
        "Director's LLM spend has no cost_sink again."
    )
    director_hits = [ln for ln in hits if '"$DIRECTOR_FUNCTION"' in ln or "alpha-engine-evaluator-director" in ln]
    assert director_hits, (
        "no `merge-lambda-env` call names the Director function "
        "($DIRECTOR_FUNCTION) — the cost-sink env vars are not being wired "
        "onto the function that actually makes the LLM call."
    )


def test_deploy_sh_cost_sink_bucket_literal_is_exact():
    hits = _merge_lambda_env_lines(_script())
    bucket_hits = [ln for ln in hits if f"KREPIS_COST_SINK_BUCKET={EXPECTED_BUCKET}" in ln]
    assert bucket_hits, (
        f"deploy.sh does not set KREPIS_COST_SINK_BUCKET={EXPECTED_BUCKET} "
        "via merge-lambda-env. A different bucket makes the cost rows land "
        "somewhere the aggregator never reads."
    )


def test_deploy_sh_cost_sink_prefix_literal_is_exact():
    hits = _merge_lambda_env_lines(_script())
    prefix_hits = [ln for ln in hits if f"KREPIS_COST_SINK_PREFIX={EXPECTED_PREFIX}" in ln]
    assert prefix_hits, (
        f"deploy.sh does not set KREPIS_COST_SINK_PREFIX={EXPECTED_PREFIX} "
        "via merge-lambda-env. A value that LOOKS right but differs from "
        "this exact prefix makes the cost rows invisible to the aggregator "
        "— the failure class this test exists to catch."
    )


def test_deploy_sh_uses_merge_not_replace_for_cost_sink():
    """The merge-lambda-env call must not be a disguised `--environment` write.

    `--environment` replaces the whole map and would wipe operator-set
    DIRECTOR_ENABLED / KREPIS_LITELLM_PROXY_URL — see
    test_deploy_sh_authority.py's sibling test for that guard. This test pins
    that the cost-sink wiring uses the merge CLI, not that flag.
    """
    hits = _merge_lambda_env_lines(_script())
    assert hits, "no merge-lambda-env invocation found"
    for ln in hits:
        assert "--environment" not in ln, (
            "the merge-lambda-env call also passes --environment, which is "
            "a replace, not a merge — this defeats the whole point of using "
            "the merge CLI."
        )


def test_report_card_function_does_not_get_the_cost_sink_env():
    """The grading Lambda makes zero LLM calls (measured: no LLMClient in
    grading/) — it must not receive KREPIS_COST_SINK_* either."""
    hits = _merge_lambda_env_lines(_script())
    non_director = [
        ln for ln in hits
        if '"$FUNCTION"' in ln and '"$DIRECTOR_FUNCTION"' not in ln
    ]
    assert not non_director, (
        "a merge-lambda-env call targets the Report Card function "
        "($FUNCTION) with the cost-sink env — that Lambda makes no LLM "
        "calls (grading/ has no LLMClient construction) and should not "
        "receive it."
    )
    # Belt-and-braces: no bare merge-lambda-env call omits an explicit
    # --function-name naming the Director function.
    for ln in hits:
        assert "--function-name" in ln and (
            '"$DIRECTOR_FUNCTION"' in ln or "alpha-engine-evaluator-director" in ln
        ), f"merge-lambda-env call does not explicitly target the Director function: {ln!r}"


def test_no_grading_llm_calls_confirms_report_card_exclusion_is_correct():
    """Guards the premise of the test above: if grading/ ever grows an
    LLMClient construction, this repo's own claim (`grading/ makes zero LLM
    calls`) goes stale and the Report Card exclusion needs re-examination."""
    grading_dir = Path(__file__).resolve().parents[1] / "grading"
    hits = []
    for path in grading_dir.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if re.search(r"\bLLMClient\s*\(", text):
            hits.append(str(path))
    assert not hits, (
        f"grading/ now constructs LLMClient in {hits} — the Report Card "
        "Lambda is no longer LLM-free, and the merge-lambda-env exclusion "
        "in deploy.sh (and the test above) needs to be revisited."
    )


def _has_call_to(tree: ast.Module, name: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            called = getattr(func, "attr", None) or getattr(func, "id", None)
            if called == name:
                return True
    return False


def test_director_agent_does_not_construct_a_cost_sink_itself():
    """The sink is an environment fact resolved inside krepis' LLMClient, not
    a call-site construction. A later "helpful" retrofit that instantiates
    `S3JsonlCostSink` (or passes `cost_sink=`) directly in director/agent.py
    reproduces the exact per-call-site gap this fix eliminates for the next
    call site added."""
    tree = ast.parse(AGENT_PY.read_text(encoding="utf-8"))
    assert not _has_call_to(tree, "S3JsonlCostSink"), (
        "director/agent.py constructs S3JsonlCostSink directly — the fix is "
        "an environment-resolved default inside krepis.llm.LLMClient "
        "(KREPIS_COST_SINK_BUCKET/PREFIX), not a call-site cost_sink. "
        "Remove the direct construction and let the client resolve it."
    )
    assert "cost_sink=" not in AGENT_PY.read_text(encoding="utf-8"), (
        "director/agent.py passes cost_sink= explicitly at a call site. "
        "That reproduces the per-call-site gap for the next call site added "
        "— the environment variables on the Lambda are the single source of "
        "truth."
    )


def test_director_retro_does_not_construct_a_cost_sink_itself():
    tree = ast.parse(RETRO_PY.read_text(encoding="utf-8"))
    assert not _has_call_to(tree, "S3JsonlCostSink"), (
        "director/retro.py constructs S3JsonlCostSink directly — see "
        "test_director_agent_does_not_construct_a_cost_sink_itself for why "
        "this must stay environment-resolved."
    )
    assert "cost_sink=" not in RETRO_PY.read_text(encoding="utf-8"), (
        "director/retro.py passes cost_sink= explicitly at a call site."
    )
