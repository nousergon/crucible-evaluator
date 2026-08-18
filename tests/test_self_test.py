"""Tests for `grading/self_test.py` — the published known-answer self-test.

Four layers, and the last two are the load-bearing ones:

1. **The battery agrees on THIS runner.** Every case passes here too, so a CI
   failure and an in-Lambda failure mean the same thing and can be compared.
2. **The expectations are re-derived here, independently.** Each closed form is
   recomputed in the test from the metric's definition. If the module's own
   arithmetic were ever quietly changed to match the implementation, this layer
   is what notices.
3. **The runner's outcome taxonomy holds.** Disagreed => FAIL, could-not-run =>
   UNKNOWN, over-budget => FAIL (Brian ruling 2026-08-13). This is the part that
   decides whether a harness fault gets reported as a correctness regression.
4. **The battery can actually FAIL** (alpha-engine-config-I7262's acceptance
   criterion, generalised to this repo since it applies cleanly here too — see
   `test_a_perturbed_sharpe_annualization_is_caught` below). A self-test never
   shown to fail is not evidence.
"""

from __future__ import annotations

import json
import math

import pytest
from nousergon_lib.quant.selftest_perturbation import assert_perturbation_caught

from grading import self_test as st


# ── layer 1: the real battery ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def body():
    return st.run_self_test(run_date="2026-08-15")


def test_every_case_passes_on_this_runner(body):
    failures = [c for c in body["cases"] if c["verdict"] != st.PASS]
    assert not failures, json.dumps(failures, indent=2, default=str)
    assert body["verdict"] == st.PASS
    assert body["n_cases"] == len(st.build_cases())


def test_the_five_named_metrics_are_all_covered(body):
    """Sharpe, Sortino, Calmar, CVaR(95) and max drawdown — the five asked for.

    A case silently dropped in a refactor is a coverage regression nothing else
    would notice: the artifact would still say PASS, on fewer questions.
    """
    names = {c["case"] for c in body["cases"]}
    assert {
        "sharpe_closed_form",
        "sortino_closed_form",
        "calmar_closed_form",
        "cvar_95_closed_form",
        "max_drawdown_closed_form",
    } <= names


def test_every_closed_form_case_asserts_to_1e_9(body):
    for case in body["cases"]:
        if case["case"].endswith("_closed_form"):
            assert case["tolerance"] == 1e-9
            assert case["abs_error"] <= 1e-9


# ── layer 2: the expectations, re-derived from first principles ─────────────

_UP, _DOWN, _N_UP, _N_DOWN = 0.01, -0.01, 60, 40
_N = _N_UP + _N_DOWN
_MEAN = (_N_UP * _UP + _N_DOWN * _DOWN) / _N


def test_expected_sharpe_is_the_definition():
    variance = (_N_UP * (_UP - _MEAN) ** 2 + _N_DOWN * (_DOWN - _MEAN) ** 2) / (_N - 1)
    assert st._expected_sharpe() == pytest.approx(
        _MEAN / math.sqrt(variance) * math.sqrt(252), rel=0, abs=1e-15)


def test_expected_sortino_uses_the_full_N_denominator():
    """The load-bearing convention: RMS of min(0, r) over ALL N, not over the
    negatives. The wrong denominator differs by sqrt(100/40) = 1.58x, so this
    assertion is what stops the expectation drifting to match a changed lib."""
    dd_over_n = math.sqrt(_N_DOWN * _DOWN**2 / _N)
    dd_over_negatives = math.sqrt(_N_DOWN * _DOWN**2 / _N_DOWN)
    assert st._expected_sortino() == pytest.approx(
        _MEAN / dd_over_n * math.sqrt(252), rel=0, abs=1e-15)
    assert st._expected_sortino() != pytest.approx(
        _MEAN / dd_over_negatives * math.sqrt(252), rel=1e-6)


def test_expected_max_drawdown_is_the_peak_to_trough_ratio():
    assert st._expected_max_drawdown() == pytest.approx(0.99**40 - 1, rel=0, abs=1e-15)


def test_expected_calmar_is_annualised_return_over_abs_drawdown():
    growth = 1.01**59 * 0.99**40
    expected = (growth ** (252 / 100) - 1.0) / abs(0.99**40 - 1)
    assert st._expected_calmar() == pytest.approx(expected, rel=0, abs=1e-15)


def test_expected_cvar_is_the_tail_mean_on_the_return_scale():
    assert st._expected_cvar_95() == -0.01


def test_the_frozen_fixture_is_the_series_the_expectations_assume():
    """A fixture drifting away from the derivation silently makes every closed
    form a coincidence. Asserted against the rendered CSV, not the constants."""
    text = st._eod_pnl_csv(st._returns())
    rows = text.strip().splitlines()
    assert rows[0] == "date,portfolio_nav,daily_return_pct,spy_return_pct"
    body_rows = rows[1:]
    assert len(body_rows) == _N
    ups = [r for r in body_rows if r.split(",")[2] == repr(1.0)]
    downs = [r for r in body_rows if r.split(",")[2] == repr(-1.0)]
    assert len(ups) == _N_UP and len(downs) == _N_DOWN
    # Percent, not fraction — the tile divides by 100, so the fixture must
    # exercise that conversion rather than bypass it.
    assert body_rows[0].split(",")[2] == repr(1.0)


# ── layer 3: the artifact and the taxonomy ──────────────────────────────────

def test_artifact_carries_the_provenance_header(body):
    """The library versions ARE the deliverable — this is an instrument check."""
    assert body["schema"] == "evaluator_self_test-1.0.0"
    assert body["component"] == "evaluator"
    assert body["run_date"] == "2026-08-15"
    assert body["python"]
    assert "code_sha" in body
    for dist in ("nousergon-lib", "numpy", "pandas"):
        assert body["libraries"][dist], f"{dist} version is empty"


def test_every_case_row_carries_the_full_shape(body):
    for case in body["cases"]:
        assert set(case) >= {
            "case", "description", "inputs", "expected", "actual",
            "abs_error", "tolerance", "verdict",
        }
        assert case["inputs"], f"{case['case']} publishes no inputs to re-derive from"
        assert case["verdict"] in (st.PASS, st.FAIL, st.UNKNOWN)


def test_artifact_is_strict_json(body):
    """``allow_nan=False`` RAISES on a non-finite float anywhere in the body."""
    text = json.dumps(body, allow_nan=False, default=str)
    assert json.loads(text)["verdict"] == body["verdict"]


def test_battery_is_cheap_enough_to_run_every_cycle(body):
    assert body["wall_clock_seconds"] < 30.0


def _case(name="c", expected=1.0, compute=lambda: 1.0, tolerance=0.0):
    return st.Case(name=name, description="d", inputs={"k": 1},
                   expected=expected, compute=compute, tolerance=tolerance)


def test_disagreement_is_FAIL_not_UNKNOWN():
    out = st.run_self_test(case_provider=lambda: [_case(expected=1.0, compute=lambda: 2.0)])
    assert out["cases"][0]["verdict"] == st.FAIL
    assert out["verdict"] == st.FAIL


def test_a_case_that_could_not_run_is_UNKNOWN_not_FAIL():
    def _boom():
        raise RuntimeError("import blew up")

    out = st.run_self_test(case_provider=lambda: [_case(compute=_boom)])
    assert out["cases"][0]["verdict"] == st.UNKNOWN
    assert out["verdict"] == st.UNKNOWN


def test_a_timeout_is_FAIL_never_UNKNOWN():
    """Brian ruling 2026-08-13.

    Asserted on the raised exception rather than on elapsed wall-clock, so the
    branch is exercised deterministically — a timing-based version of this test
    is only as trustworthy as the process's clock, and this suite has already
    had a test leak a no-op ``time.sleep`` across files.
    """
    def _too_slow():
        raise st._CaseTimeout("case exceeded its budget")

    out = st.run_self_test(case_provider=lambda: [_case(compute=_too_slow)])
    assert out["cases"][0]["verdict"] == st.FAIL
    assert out["cases"][0]["timed_out"] is True
    assert out["verdict"] == st.FAIL


def test_an_over_budget_call_really_does_raise_case_timeout():
    """The other half: the budget mechanism itself fires on a real overrun.

    Busy-waits on ``time.monotonic`` rather than sleeping, so it holds whether
    or not a SIGALRM budget could be installed and regardless of what any other
    test did to ``time.sleep``.
    """
    import time as _time

    def _busy():
        deadline = _time.monotonic() + 0.5
        while _time.monotonic() < deadline:
            pass
        return 1.0

    with pytest.raises(st._CaseTimeout):
        st._call_with_timeout(_busy, 0.1)


def test_a_battery_that_could_not_be_built_is_UNKNOWN_and_does_not_raise():
    def _boom():
        raise ImportError("no lib")

    out = st.run_self_test(case_provider=_boom)
    assert out["verdict"] == st.UNKNOWN
    assert out["status"] == "error"


def test_an_empty_battery_is_UNKNOWN_never_PASS():
    assert st.run_self_test(case_provider=lambda: [])["verdict"] == st.UNKNOWN


def test_fail_beats_unknown_in_the_overall_verdict():
    def _boom():
        raise RuntimeError("x")

    out = st.run_self_test(case_provider=lambda: [
        _case(name="a", expected=1.0, compute=lambda: 2.0),
        _case(name="b", compute=_boom),
    ])
    assert out["verdict"] == st.FAIL


def test_verdict_is_pass_only_on_the_literal_PASS():
    assert st.verdict_is_pass("PASS")
    for other in (None, "", "ok", "pass", "UNKNOWN", "FAIL", True):
        assert not st.verdict_is_pass(other)


def test_missing_distribution_is_recorded_explicitly_never_omitted():
    resolved = st.resolved_library_versions(("nousergon-lib", "definitely-not-a-package"))
    assert resolved["definitely-not-a-package"] == "<not installed>"


def test_self_test_key_is_the_declared_artifact_path():
    assert st.self_test_key("2026-08-15") == "evaluator/2026-08-15/self_test.json"


def test_frozen_s3_serves_only_the_fixture_and_never_reaches_the_network():
    from botocore.exceptions import ClientError

    from grading.tiles.portfolio_outcome import EOD_PNL_KEY

    fake = st._FrozenS3("x", EOD_PNL_KEY)
    assert fake.get_object(Bucket="b", Key=EOD_PNL_KEY)["Body"].read() == b"x"
    with pytest.raises(ClientError):
        fake.get_object(Bucket="b", Key="signals/2024-01-01/signals.json")


def test_write_self_test_puts_the_declared_key():
    written = {}

    class _S3:
        def put_object(self, **kw):
            written.update(kw)

    key = st.write_self_test("bkt", "2026-08-15", {"verdict": "PASS"}, s3_client=_S3())
    assert key == "evaluator/2026-08-15/self_test.json"
    assert written["Bucket"] == "bkt"
    assert json.loads(written["Body"])["verdict"] == "PASS"


# ── wiring: the handler runs it, publishes it, and reports the verdict ──────

def test_handler_wires_the_self_test():
    """String-level wiring assertion: without these the artifact silently stops
    being published and nothing fails."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "grading" / "handler.py").read_text()
    assert "from grading.self_test import run_self_test, verdict_is_pass, write_self_test" in source
    assert "self_test = run_self_test(run_date=run_date)" in source
    assert "write_self_test(bucket, run_date, self_test)" in source
    assert '"self_test_verdict": self_test.get("verdict")' in source
    assert '"degraded_self_test": not self_test_pass' in source
    # alpha-engine-config-I7282 added a second threaded block (`gate_state`) to
    # the same call; the invariant this line protects is that the self-test is
    # threaded IN rather than re-run inside the builder, so it pins the kwarg
    # rather than the full argument list.
    assert "build_report_card(bucket, run_date, self_test=self_test," in source


class TestReportCardCarriesTheVerdict:
    """§2.3a rule 3 — every surface presenting the run's results carries the
    verdict state. The card is the surface Brian and the Director read off."""

    def test_aggregate_sets_both_keys_and_falls_back_rather_than_dropping_them(self):
        """A caller that forgets to thread the verdict must not silently produce
        a card with nothing to declare — that is indistinguishable from a card
        nobody checked."""
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "grading" / "aggregate.py").read_text()
        assert 'scorecard["self_test"] = self_test_body' in source
        assert ('scorecard["degraded_self_test"] = '
                'not self_test_is_pass(self_test_body.get("verdict"))') in source
        assert "self_test if self_test is not None else run_self_test(run_date)" in source

    @pytest.mark.parametrize("verdict,expected_degraded", [
        ("PASS", False), ("FAIL", True), ("UNKNOWN", True), (None, True), ("ok", True),
    ])
    def test_degraded_flag_is_derived_so_it_can_never_disagree(self, verdict, expected_degraded):
        """An absent or unrecognised verdict reads as degraded, never as a pass."""
        assert (not st.verdict_is_pass(verdict)) is expected_degraded


# ── layer 4: the battery can FAIL — generalising I7262's acceptance criterion ──

def test_a_perturbed_sharpe_annualization_is_caught(monkeypatch):
    """Recreates the EXACT defect class that started this arc
    (alpha-engine-config-I7236: sqrt(365) vs sqrt(252), a 20.3% divergence)
    and asserts the battery catches it.

    Perturbed on `grading.tiles.portfolio_outcome.sharpe_ratio` — that module
    imports the NAME directly (`from nousergon_lib.quant.riskstats import
    sharpe_ratio`), so it is bound into the TILE's own namespace and patching
    `nousergon_lib.quant.riskstats.sharpe_ratio` would not reach it.

    REAL FINDING while building this test, filed as `alpha-engine-config#7621`
    and fixed by that issue: `self_test.py`'s own `_TILE_CACHE` docstring
    claimed "the cache cannot mask a change — a redeploy is a new process".
    That was false within one warm PROCESS — this test's perturbation was
    invisible until the cache was cleared here, because an earlier case/test in
    this same session had already populated `_TILE_CACHE["base"]`. The
    evaluator's grading Lambda (`alpha-engine-evaluator`) reuses warm
    containers across invocations (measured: `aws lambda list-functions`
    against the live function, `PackageType: "Image"`), so the same staleness
    would occur there — the second invocation's self-test would silently grade
    the first invocation's tile, not its own. `run_self_test` now clears
    `_TILE_CACHE` at its own entry (I7621's fix), so the manual clear below is
    now redundant defense-in-depth for this test's isolation, not the thing
    that makes the perturbation visible — `test_a_warm_container_reuse_does_not_serve_a_stale_tile`
    below tests the fix itself, at the `run_self_test` boundary, without any
    manual clear.
    """
    from grading import self_test as _self_test_module
    from grading.tiles import portfolio_outcome as tile

    _self_test_module._TILE_CACHE.clear()
    original_sharpe_ratio = tile.sharpe_ratio

    def sqrt_365(returns, *, risk_free_rate=0.0, periods_per_year=365):
        return original_sharpe_ratio(
            returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year,
        )

    try:
        out = assert_perturbation_caught(
            monkeypatch,
            module_path="grading.tiles.portfolio_outcome",
            attr="sharpe_ratio",
            perturbed=sqrt_365,
            run=lambda: st.run_self_test(run_date="2026-08-15"),
            case_name="sharpe_closed_form",
        )
        assert out["n_failed"] >= 1
    finally:
        # The perturbed run wrote a POISONED "base" tile (built with sqrt_365)
        # into the module cache — clear it so any test after this one that
        # calls the `body` fixture for the first time does not silently grade
        # against this test's perturbation instead of the real implementation.
        _self_test_module._TILE_CACHE.clear()


def test_a_warm_container_reuse_does_not_serve_a_stale_tile(monkeypatch):
    """`alpha-engine-config#7621` — the same-process regression test the issue
    requires: two consecutive `run_self_test()` calls in ONE process, with
    different inputs between them, and the second call must not observe the
    first's cached tile.

    Simulates exactly what a warm Lambda container does: invocation N runs
    clean and PASSes; invocation N+1 in the SAME process runs against a
    perturbed `sharpe_ratio` and must FAIL on its own inputs — not silently
    reuse invocation N's cached "base" tile values (which is what happened
    before `run_self_test` cleared `_TILE_CACHE` at its own entry). No manual
    cache clear happens in this test — if the fix regresses, this reproduces
    the exact failure mode the issue reported: `n_failed == 0` on a run whose
    inputs actually disagree, because the case read invocation N's stale
    "base" tile.
    """
    from grading.tiles import portfolio_outcome as tile

    original_sharpe_ratio = tile.sharpe_ratio

    def sqrt_365(returns, *, risk_free_rate=0.0, periods_per_year=365):
        return original_sharpe_ratio(
            returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year,
        )

    # Invocation N: clean run, populates _TILE_CACHE["base"] with the CORRECT
    # sharpe_ratio value.
    first = st.run_self_test(run_date="2026-08-15")
    first_sharpe_case = next(c for c in first["cases"] if c["case"] == "sharpe_closed_form")
    assert first_sharpe_case["verdict"] == st.PASS

    # Invocation N+1: SAME process, no manual cache clear, perturbed lib call.
    # A container reused warm across invocations is exactly this — a second
    # call into the same module state with different underlying behavior.
    monkeypatch.setattr(tile, "sharpe_ratio", sqrt_365)
    second = st.run_self_test(run_date="2026-08-16")
    second_sharpe_case = next(c for c in second["cases"] if c["case"] == "sharpe_closed_form")

    assert second_sharpe_case["actual"] != first_sharpe_case["actual"], (
        "invocation N+1 returned invocation N's cached tile value — the "
        "warm-container cache masked a change between invocations"
    )
    assert second_sharpe_case["verdict"] == st.FAIL
    assert second["n_failed"] >= 1
