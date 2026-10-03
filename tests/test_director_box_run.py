"""``director.box_run`` — the Director's off-Lambda entrypoint
(alpha-engine-config-I11936)."""
from __future__ import annotations

import json

import pytest

from director import box_run, hosting


def test_wall_context_counts_down_like_a_lambda_context():
    t = [100.0]
    ctx = box_run.WallContext(10.0, clock=lambda: t[0])
    assert ctx.get_remaining_time_in_millis() == 10_000
    t[0] = 104.5
    assert ctx.get_remaining_time_in_millis() == 5_500
    t[0] = 200.0
    assert ctx.get_remaining_time_in_millis() == 0


def test_wall_context_refuses_a_non_positive_wall():
    with pytest.raises(ValueError):
        box_run.WallContext(0)


def test_effective_wall_without_a_deadline_is_the_configured_wall():
    assert box_run.effective_wall_s(2700, None) == 2700


def test_effective_wall_is_clamped_to_the_ssm_deadline():
    # 1,000s left on the SSM clock, less the 60s margin.
    assert box_run.effective_wall_s(2700, 11_000, now=10_000) == 1000 - box_run.DEADLINE_MARGIN_S


def test_effective_wall_never_grows_past_the_configured_wall():
    assert box_run.effective_wall_s(2700, 10_000 + 99_999, now=10_000) == 2700


def test_effective_wall_refuses_to_start_with_nothing_left():
    with pytest.raises(RuntimeError):
        box_run.effective_wall_s(2700, 10_030, now=10_000)


def test_load_gate_state(tmp_path):
    assert box_run.load_gate_state(None) is None
    good = tmp_path / "g.json"
    good.write_text(json.dumps({"schema_version": 1}))
    assert box_run.load_gate_state(str(good)) == {"schema_version": 1}
    bad = tmp_path / "b.json"
    bad.write_text("[1]")
    with pytest.raises(ValueError):
        box_run.load_gate_state(str(bad))
    broken = tmp_path / "x.json"
    broken.write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        box_run.load_gate_state(str(broken))


@pytest.mark.parametrize("summary,code", [
    ({"status": "not_degraded", "retro_refused": False}, 0),
    ({"status": "degraded", "retro_refused": False}, 20),
    ({"status": "not_degraded", "retro_refused": True}, 21),
    ({"status": "degraded", "retro_refused": True}, 22),
])
def test_main_runs_the_shared_entrypoint_on_the_spot_profile(monkeypatch, tmp_path, capsys,
                                                               summary, code):
    from director import handler

    seen = {}

    def _fake_run_director(event, context, *, host):
        seen.update(event=event, host=host, remaining=context.get_remaining_time_in_millis())
        return summary

    monkeypatch.setattr(handler, "run_director", _fake_run_director)
    monkeypatch.delenv("DIRECTOR_REGISTRY_DEST", raising=False)
    gs = tmp_path / "gs.json"
    gs.write_text(json.dumps({"schema_version": 1, "gate_degraded": True}))

    rc = box_run.main(["--date", "2026-10-02", "--dry-run", "true",
                       "--gate-state-file", str(gs), "--execution-name", "e"])

    assert rc == code
    assert seen["event"] == {"date": "2026-10-02", "dry_run": True,
                             "gate_state": {"schema_version": 1, "gate_degraded": True}}
    assert seen["host"].name == "weekly-spot"
    assert seen["host"].plan_ceiling_s == hosting.WEEKLY_SPOT.plan_ceiling_s
    assert seen["host"].wall_s == hosting.WEEKLY_SPOT.wall_s
    assert 0 < seen["remaining"] <= hosting.WEEKLY_SPOT.wall_s * 1000
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert json.loads(out) == summary


def test_main_exits_one_when_the_director_raises(monkeypatch):
    from director import handler

    def _boom(event, context, *, host):
        raise RuntimeError("boom")

    monkeypatch.setattr(handler, "run_director", _boom)
    rc = box_run.main(["--date", "2026-10-02"])
    assert rc == box_run.EXIT_FAILED
    assert rc not in hosting.COMPLETED_EXIT_CODES


def test_main_exits_one_when_the_deadline_has_already_passed(monkeypatch):
    from director import handler

    monkeypatch.setattr(handler, "run_director", lambda *a, **k: pytest.fail("must not run"))
    rc = box_run.main(["--date", "2026-10-02", "--deadline-epoch", "1"])
    assert rc == box_run.EXIT_FAILED
