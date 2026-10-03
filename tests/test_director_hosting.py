"""The Director's two homes and their bounds (alpha-engine-config-I11936).

Pins that moving the Director off Lambda changed WHICH hard bound applies, never
WHETHER one applies: the weekly-spot plan call stays capped, the wall covers
everything an invocation owes at that cap, and the Lambda path is unchanged.
"""
from __future__ import annotations

import pathlib
import re
import sys
import types

import pytest

from director import agent as agent_mod
from director import hosting
from director.budget import InvocationBudget

REPO = pathlib.Path(__file__).resolve().parent.parent


# ── the numbers ──────────────────────────────────────────────────────────────


def test_lambda_home_is_unchanged():
    assert hosting.LAMBDA.wall_s == 900.0
    assert hosting.LAMBDA.plan_ceiling_s == agent_mod.DIRECTOR_PLAN_CEILING_S == 600.0


def test_weekly_spot_plan_ceiling_is_bounded_and_covers_the_measured_worst_case():
    ceiling = hosting.WEEKLY_SPOT.plan_ceiling_s
    # ~1,500s is the extrapolated full-answer worst case; 596s the slowest
    # completed call. The ceiling must clear both — and stay a finite bound.
    assert 1500.0 < ceiling <= 1800.0
    assert ceiling >= 3 * 596.0
    assert ceiling > agent_mod.DIRECTOR_PLAN_IDLE_TIMEOUT_S


def test_weekly_spot_wall_covers_everything_owed_at_the_full_ceiling():
    owed = hosting.owed_seconds(hosting.WEEKLY_SPOT.plan_ceiling_s)
    assert hosting.WEEKLY_SPOT.wall_s >= owed
    # Rounded up, not padded open-ended: within 10 minutes of the sum.
    assert hosting.WEEKLY_SPOT.wall_s - owed <= 600.0


def test_ssm_timeout_is_wall_plus_provisioning_and_stays_finite():
    assert hosting.WEEKLY_SPOT_SSM_EXECUTION_TIMEOUT_S == (
        hosting.WEEKLY_SPOT_WALL_S + hosting.WEEKLY_SPOT_PROVISION_ALLOWANCE_S
    )
    # nousergon-data tests/test_sf_director_on_spot_wiring.py pins the SF side.
    assert hosting.WEEKLY_SPOT_SSM_EXECUTION_TIMEOUT_S == 3600.0


def test_the_budget_never_quotes_above_the_home_ceiling():
    class _Ctx:
        def get_remaining_time_in_millis(self):
            return int(hosting.WEEKLY_SPOT.wall_s * 1000)

    budget = InvocationBudget.from_context(_Ctx())
    q = budget.quote("director-plan", hosting.WEEKLY_SPOT.plan_ceiling_s, attempts=1,
                     downstream_s=agent_mod.RETRO_JUDGE_RESERVE_S)
    assert 0 < q <= hosting.WEEKLY_SPOT.plan_ceiling_s


# ── the exit-code contract ───────────────────────────────────────────────────


@pytest.mark.parametrize("summary,code", [
    ({"status": "not_degraded", "retro_refused": False}, 0),
    ({"status": "degraded", "retro_refused": False}, 20),
    ({"status": "not_degraded", "retro_refused": True}, 21),
    ({"status": "degraded", "retro_refused": True}, 22),
    ({"status": "skipped"}, 0),
])
def test_exit_code_for(summary, code):
    assert hosting.exit_code_for(summary) == code
    assert code in hosting.COMPLETED_EXIT_CODES


def test_completed_codes_never_collide_with_a_failure():
    from director.box_run import EXIT_FAILED

    assert EXIT_FAILED not in hosting.COMPLETED_EXIT_CODES


# ── the ceiling reaches the streamed call as its total_timeout ───────────────


def _install_fake_krepis(monkeypatch):
    router = types.ModuleType("krepis.router")

    class _Spec:
        provider = "litellm"
        model = "m"
        max_tokens = 1024
        transport = "openai"
        api_key_env = None
        supports_streaming = True

    router.resolve_group_spec = lambda group, **kw: (_Spec(), {
        "route": "litellm_proxy", "deployment_id": "d", "auth_token_type": "litellm_master_key",
        "primary_model": "m", "primary_registry_id": "r",
        "api_base_url": "https://router.invalid", "provider": "litellm", "skipped_entries": [],
    })
    llm = types.ModuleType("krepis.llm")

    class _LLMClient:
        def __init__(self, spec, **kwargs):
            self.spec = spec

    llm.LLMClient = _LLMClient
    monkeypatch.setitem(sys.modules, "krepis", sys.modules.get("krepis") or types.ModuleType("krepis"))
    monkeypatch.setitem(sys.modules, "krepis.router", router)
    monkeypatch.setitem(sys.modules, "krepis.llm", llm)


def test_default_llm_without_a_home_keeps_the_lambda_total_timeout(monkeypatch):
    _install_fake_krepis(monkeypatch)
    client = agent_mod._default_llm()
    assert client.total_timeout_s == agent_mod.DIRECTOR_PLAN_TOTAL_TIMEOUT_S == 600.0


def test_default_llm_arms_the_spot_ceiling_as_the_total_timeout(monkeypatch):
    _install_fake_krepis(monkeypatch)
    client = agent_mod._default_llm(plan_ceiling_s=hosting.WEEKLY_SPOT.plan_ceiling_s)
    assert client.total_timeout_s == 1800.0


def test_a_ceiling_at_or_below_the_idle_bound_is_refused(monkeypatch):
    _install_fake_krepis(monkeypatch)
    with pytest.raises(ValueError):
        agent_mod._default_llm(plan_ceiling_s=agent_mod.DIRECTOR_PLAN_IDLE_TIMEOUT_S)


# ── the box script stays in lockstep with the Lambda's packaging ─────────────


def _script() -> str:
    return (REPO / "infrastructure" / "director_on_box.sh").read_text()


@pytest.mark.repo_tree
def test_box_gitleaks_pin_matches_the_dockerfile():
    docker = (REPO / "Dockerfile").read_text()
    for var in ("GITLEAKS_VERSION", "GITLEAKS_SHA256"):
        box = re.search(rf'^{var}="([^"]+)"', _script(), re.M).group(1)
        img = re.search(rf'{var}="([^"]+)"', docker).group(1)
        assert box == img, var


@pytest.mark.repo_tree
def test_box_cost_sink_matches_the_lambda_deploy():
    deploy = (REPO / "infrastructure" / "deploy.sh").read_text()
    for var in ("KREPIS_COST_SINK_BUCKET", "KREPIS_COST_SINK_PREFIX"):
        box = re.search(rf"^export {var}=(\S+)", _script(), re.M).group(1)
        lam = re.search(rf"--set {var}=(\S+)", deploy).group(1)
        assert box == lam, var


@pytest.mark.repo_tree
def test_box_script_runs_the_dlp_gate_then_box_run():
    s = _script()
    assert s.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in s
    assert s.index("-m krepis.session_dlp preflight") < s.index("exec \"${PY}\" -m director.box_run")
    assert "KREPIS_EXEC_CONTEXT=lambda" in s


@pytest.mark.repo_tree
def test_box_script_is_executable():
    import os

    assert os.access(REPO / "infrastructure" / "director_on_box.sh", os.X_OK)
