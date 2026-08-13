"""Tests for the Report Card's runtime correctness attestation (`grading/attestation.py`).

Two halves, both required by `sf-pipeline-policy.md` §2.3a:

**Producer** — the evaluator recomputes the headline portfolio numbers (Sharpe,
Sortino, max drawdown, CVaR, cumulative log-alpha) from raw daily rows using
`nousergon_lib.quant.*` primitives resolved at container-build time. The lib pin
moves independently of this repo, and no existing test pins any of those numbers
to a value (`test_portfolio_outcome.py`'s Sharpe assertion is `sharpe > 0`). A
changed ddof, annualization factor or downside-deviation denominator in the lib
would shift every risk-adjusted tile on the card, plausibly and invisibly. The
battery pins each primitive to a closed-form expectation written out from its
definition, computed here with `math` alone — never by calling the code under test.

**Consumer** — two upstream stages emit their own verdicts under
`backtest/{run_date}/`: the simulation engine at `attestation.json`, and the
Evaluator stage (the one that grades the ranking metrics and promotes
`config/executor_params.json` + `config/producer_champion.json`) at
`evaluator_attestation.json`. The card must carry both, with their `as_of`
stamps, and a missing artifact must propagate as UNKNOWN, never as a pass —
including when the other two halves passed.
"""
from __future__ import annotations

import json
import math

import boto3
import pytest
from moto import mock_aws

from grading import attestation

BUCKET = "test-bucket"
RUN_DATE = "2026-08-15"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _put_backtester_attestation(s3, body: dict, run_date: str = RUN_DATE):
    s3.put_object(
        Bucket=BUCKET,
        Key=f"backtest/{run_date}/attestation.json",
        Body=json.dumps(body).encode(),
    )


def _put_evaluator_stage_attestation(s3, body: dict, run_date: str = RUN_DATE):
    s3.put_object(
        Bucket=BUCKET,
        Key=f"backtest/{run_date}/evaluator_attestation.json",
        Body=json.dumps(body).encode(),
    )


def _stage_body(verdict: str = "PASS", run_date: str = RUN_DATE, **extra):
    """A minimal `evaluator_attestation-1.0.0` body as the producer emits it."""
    body = {
        "schema": "evaluator_attestation-1.0.0",
        "component": "evaluator",
        "run_date": run_date,
        "verdict": verdict,
        "promotion_withheld": verdict != "PASS",
        "own": {
            "schema": "evaluator_attestation-1.0.0", "component": "evaluator",
            "status": "ok", "verdict": verdict, "n_checks": 11, "n_failed": 0,
            "n_errored": 0, "checks": [], "engine": {"python": "3.11.9"},
        },
        "upstream": {
            "component": "backtester", "verdict": verdict, "reason": "verdict read",
            "as_of": "2026-08-08T06:13:29Z",
            "key": f"backtest/{run_date}/attestation.json",
        },
    }
    body.update(extra)
    return body


def _put_contamination(s3, body=None, run_date: str = RUN_DATE):
    """config#7199 — the look-ahead-contamination verdict half."""
    import json as _json
    doc = body if body is not None else {
        "schema": "pit_parity-1.0.0", "run_date": run_date, "status": "ok",
        "verdict": "PASS", "verdict_reason": "PASS.",
        "coverage": {"coverage_fraction": 1.0, "budget_stopped": False,
                     "complete": True, "measured": True},
        "materiality": {"material": False},
    }
    s3.put_object(
        Bucket=BUCKET, Key=f"backtest/{run_date}/pit_parity.json",
        Body=_json.dumps(doc).encode(),
    )


def _put_both_passing(s3, run_date: str = RUN_DATE):
    """All FOUR halves passing (config#7199 added the contamination half; the
    name predates it and is kept so every existing call site still means "the
    fully-attested case")."""
    _put_backtester_attestation(s3, {
        "schema": "backtest_attestation-1.0.0", "run_date": run_date,
        "status": "ok", "verdict": "PASS", "n_checks": 5, "n_failed": 0,
    }, run_date=run_date)
    _put_evaluator_stage_attestation(s3, _stage_body(run_date=run_date), run_date=run_date)
    _put_contamination(s3, run_date=run_date)


# ════════════════════════════════════════════════════════════════════════════
# Producer — the evaluator's own known-answer battery
# ════════════════════════════════════════════════════════════════════════════

class TestEvaluatorBattery:
    def test_passes_on_the_deployed_lib(self):
        result = attestation.run_evaluator_attestation()
        assert result["verdict"] == attestation.PASS, result["checks"]
        assert result["status"] == "ok"

    def test_covers_every_headline_primitive(self):
        names = {c["name"] for c in attestation.run_evaluator_attestation()["checks"]}
        assert {
            "sharpe_annualization",
            "sortino_downside_denominator",
            "max_drawdown_running_peak",
            "cvar_95_tail_mean",
            "cumulative_log_alpha",
            "eod_pnl_percent_to_fraction",
        } <= names

    def test_expectations_are_independent_of_the_lib(self):
        """The battery's expected values must be derivable with `math` alone.

        If an expectation were produced by calling the same primitive it checks,
        the check would agree with any behaviour the lib ever adopts.
        """
        # Every `_expected_*` helper must be lib-free: derived from the metric's
        # definition with `math` alone. A helper that called the primitive it
        # checks would agree with any behaviour the lib ever adopts.
        import inspect

        for name in ("_expected_sharpe", "_expected_sortino", "_expected_cvar_95",
                     "_expected_cumulative_log_alpha"):
            src = inspect.getsource(getattr(attestation, name))
            assert "nousergon_lib" not in src, f"{name} leans on the code it attests"
            assert "grading." not in src, f"{name} leans on the code it attests"

        # And re-derived independently here, in this module, from the definition.
        checks = {c["name"]: c for c in attestation.run_evaluator_attestation()["checks"]}

        r = list(attestation.FROZEN_RETURNS)
        n = len(r)
        mean = sum(r) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in r) / (n - 1))
        assert checks["sharpe_annualization"]["expected"] == pytest.approx(
            (mean / sd) * math.sqrt(252), rel=1e-12,
        )

        dd = math.sqrt(sum(min(0.0, x) ** 2 for x in r) / n)
        assert checks["sortino_downside_denominator"]["expected"] == pytest.approx(
            (mean / dd) * math.sqrt(252), rel=1e-12,
        )

        # NAV 100 → 120 → 90 → 130 → 104: worst is 90/120 - 1 = -0.25.
        assert checks["max_drawdown_running_peak"]["expected"] == pytest.approx(-0.25, rel=1e-12)

    def test_records_the_lib_it_attested(self):
        env = attestation.run_evaluator_attestation()["engine"]
        assert env["nousergon_lib"]
        assert env["python"]

    def test_serializes_to_json(self):
        json.dumps(attestation.run_evaluator_attestation())


class TestTeeth:
    def test_a_wrong_expectation_flips_the_verdict(self, monkeypatch):
        original = attestation._EVALUATOR_CHECKS

        def _poisoned():
            checks = list(original())
            checks[0] = checks[0]._replace(expected=checks[0].expected * 1.0001)
            return checks

        monkeypatch.setattr(attestation, "_EVALUATOR_CHECKS", _poisoned)
        result = attestation.run_evaluator_attestation()
        assert result["verdict"] == attestation.FAIL

    def test_a_check_that_cannot_run_is_unknown_not_fail(self, monkeypatch):
        original = attestation._EVALUATOR_CHECKS

        def _boom():
            raise ImportError("nousergon_lib.quant is gone")

        def _exploding():
            checks = list(original())
            return [checks[0]._replace(compute=_boom)] + checks[1:]

        monkeypatch.setattr(attestation, "_EVALUATOR_CHECKS", _exploding)
        result = attestation.run_evaluator_attestation()
        assert result["verdict"] == attestation.UNKNOWN
        assert result["n_errored"] == 1

    def test_battery_construction_failure_is_unknown_not_an_exception(self, monkeypatch):
        def _boom():
            raise RuntimeError("battery exploded")

        monkeypatch.setattr(attestation, "_EVALUATOR_CHECKS", _boom)
        result = attestation.run_evaluator_attestation()
        assert result["verdict"] == attestation.UNKNOWN
        assert result["error_class"] == "RuntimeError"


# ════════════════════════════════════════════════════════════════════════════
# Consumer — the backtester's verdict must reach the card
# ════════════════════════════════════════════════════════════════════════════

class TestBacktesterVerdictConsumption:
    def test_reads_a_passing_verdict(self, s3):
        _put_backtester_attestation(s3, {
            "schema": "backtest_attestation-1.0.0", "run_date": RUN_DATE,
            "status": "ok", "verdict": "PASS", "n_checks": 5, "n_failed": 0,
            "engine": {"vectorbt": "0.28.5"},
        })
        block = attestation.read_backtester_attestation(BUCKET, RUN_DATE, s3_client=s3)
        assert block["verdict"] == attestation.PASS
        assert block["source_path"].endswith(f"backtest/{RUN_DATE}/attestation.json")

    def test_absent_artifact_is_unknown_never_pass(self, s3):
        block = attestation.read_backtester_attestation(BUCKET, RUN_DATE, s3_client=s3)
        assert block["verdict"] == attestation.UNKNOWN
        assert "absent" in block["reason"].lower()
        assert attestation.verdict_is_pass(block["verdict"]) is False

    def test_a_failing_verdict_is_carried_verbatim(self, s3):
        _put_backtester_attestation(s3, {
            "schema": "backtest_attestation-1.0.0", "run_date": RUN_DATE,
            "status": "ok", "verdict": "FAIL", "n_checks": 5, "n_failed": 2,
            "checks": [{"name": "pnl_no_fees", "passed": False}],
        })
        block = attestation.read_backtester_attestation(BUCKET, RUN_DATE, s3_client=s3)
        assert block["verdict"] == attestation.FAIL
        assert block["n_failed"] == 2

    def test_an_unrecognised_verdict_string_is_unknown(self, s3):
        """A producer that starts writing `"ok"` must not silently read as a pass."""
        _put_backtester_attestation(s3, {"schema": "x", "verdict": "ok"})
        block = attestation.read_backtester_attestation(BUCKET, RUN_DATE, s3_client=s3)
        assert block["verdict"] == attestation.UNKNOWN

    def test_a_verdict_for_the_wrong_run_date_is_unknown(self, s3):
        """§2.3a rule 1: a stale verdict from another cycle says nothing about
        THIS cycle's numbers. It must not be inherited."""
        _put_backtester_attestation(s3, {
            "schema": "backtest_attestation-1.0.0", "run_date": "2026-08-01",
            "status": "ok", "verdict": "PASS",
        }, run_date=RUN_DATE)
        block = attestation.read_backtester_attestation(BUCKET, RUN_DATE, s3_client=s3)
        assert block["verdict"] == attestation.UNKNOWN
        assert "run_date" in block["reason"]


class TestCombinedVerdict:
    def test_pass_requires_all_four_halves(self, s3):
        _put_both_passing(s3)
        block = attestation.build_run_attestation(BUCKET, RUN_DATE, s3_client=s3)
        assert block["verdict"] == attestation.PASS
        assert block["evaluator"]["verdict"] == attestation.PASS
        assert block["backtester"]["verdict"] == attestation.PASS
        assert block["evaluator_stage"]["verdict"] == attestation.PASS
        assert block["contamination"]["verdict"] == attestation.PASS
        # config#7199 — the two claims stay two.
        assert block["arithmetic_verdict"] == attestation.PASS
        assert block["contamination_verdict"] == attestation.PASS

    def test_three_clean_arithmetic_halves_do_not_carry_a_missing_contamination_half(self, s3):
        """The 2026-08-07 configuration. Every arithmetic half attests, the
        contamination check never answered, and the card must NOT read verified
        — that is the whole of config#7199."""
        _put_backtester_attestation(s3, {
            "schema": "backtest_attestation-1.0.0", "run_date": RUN_DATE,
            "status": "ok", "verdict": "PASS", "n_checks": 5, "n_failed": 0,
        }, run_date=RUN_DATE)
        _put_evaluator_stage_attestation(s3, _stage_body(run_date=RUN_DATE),
                                         run_date=RUN_DATE)
        block = attestation.build_run_attestation(BUCKET, RUN_DATE, s3_client=s3)
        assert block["arithmetic_verdict"] == attestation.PASS
        assert block["contamination_verdict"] == attestation.UNKNOWN
        assert block["verdict"] == attestation.UNKNOWN

    def test_a_missing_evaluator_stage_verdict_degrades_the_card(self, s3):
        # The engine attested; the stage that grades and PROMOTES off its numbers
        # did not. Two of three halves is not a verified card.
        _put_backtester_attestation(s3, {
            "schema": "backtest_attestation-1.0.0", "run_date": RUN_DATE,
            "status": "ok", "verdict": "PASS",
        })
        block = attestation.build_run_attestation(BUCKET, RUN_DATE, s3_client=s3)
        assert block["verdict"] == attestation.UNKNOWN
        assert block["evaluator_stage"]["verdict"] == attestation.UNKNOWN
        assert "absent" in block["evaluator_stage"]["reason"]

    def test_a_failing_evaluator_stage_outranks_a_passing_engine(self, s3):
        _put_backtester_attestation(s3, {
            "schema": "backtest_attestation-1.0.0", "run_date": RUN_DATE,
            "status": "ok", "verdict": "PASS",
        })
        _put_evaluator_stage_attestation(s3, _stage_body(verdict="FAIL"))
        block = attestation.build_run_attestation(BUCKET, RUN_DATE, s3_client=s3)
        assert block["verdict"] == attestation.FAIL

    def test_stage_verdict_from_another_cycle_is_not_inherited(self, s3):
        _put_backtester_attestation(s3, {
            "schema": "backtest_attestation-1.0.0", "run_date": RUN_DATE,
            "status": "ok", "verdict": "PASS",
        })
        # A PASS body written under THIS cycle's key but stamped with last week's
        # run_date — S3 will serve it happily.
        _put_evaluator_stage_attestation(s3, _stage_body(run_date="2026-08-08"))
        block = attestation.build_run_attestation(BUCKET, RUN_DATE, s3_client=s3)
        assert block["verdict"] == attestation.UNKNOWN
        assert "another cycle" in block["evaluator_stage"]["reason"]

    def test_promotion_withheld_surfaces_on_the_block(self, s3):
        _put_backtester_attestation(s3, {
            "schema": "backtest_attestation-1.0.0", "run_date": RUN_DATE,
            "status": "ok", "verdict": "PASS",
        })
        _put_evaluator_stage_attestation(s3, _stage_body(verdict="UNKNOWN"))
        block = attestation.build_run_attestation(BUCKET, RUN_DATE, s3_client=s3)
        assert block["promotion_withheld"] is True
        assert "WITHHELD" in block["evaluator_stage"]["reason"]

    def test_as_of_is_carried_per_half(self, s3):
        _put_both_passing(s3)
        block = attestation.build_run_attestation(BUCKET, RUN_DATE, s3_client=s3)
        # A state with no timestamp cannot read as stale — §2.3a rule 3's
        # surface requirement is state AND as-of.
        assert block["as_of"]["backtester"] is not None
        assert block["as_of"]["evaluator_stage"] is not None

    def test_as_of_is_none_when_the_verdict_is_absent(self, s3):
        block = attestation.build_run_attestation(BUCKET, RUN_DATE, s3_client=s3)
        assert block["as_of"]["backtester"] is None
        assert block["as_of"]["evaluator_stage"] is None

    def test_a_missing_backtester_verdict_degrades_the_card(self, s3):
        block = attestation.build_run_attestation(BUCKET, RUN_DATE, s3_client=s3)
        assert block["verdict"] == attestation.UNKNOWN

    def test_a_failing_half_outranks_an_unknown_half(self, s3):
        _put_backtester_attestation(s3, {
            "schema": "backtest_attestation-1.0.0", "run_date": RUN_DATE,
            "status": "ok", "verdict": "FAIL",
        })
        block = attestation.build_run_attestation(BUCKET, RUN_DATE, s3_client=s3)
        assert block["verdict"] == attestation.FAIL

    def test_never_raises_on_an_s3_failure(self):
        class _Boom:
            def get_object(self, **kwargs):
                raise RuntimeError("s3 down")

        block = attestation.build_run_attestation(BUCKET, RUN_DATE, s3_client=_Boom())
        assert block["verdict"] == attestation.UNKNOWN


# ════════════════════════════════════════════════════════════════════════════
# Propagation — §2.3a rule 3: every surface carries the verdict state
# ════════════════════════════════════════════════════════════════════════════

class TestBacktesterTileCarriesTheVerdict:
    def _component(self, s3):
        from grading.tiles.backtester import build_backtester_tile

        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        return next(c for c in tile["components"] if c["name"] == "numeric_attestation")

    def test_is_critical(self, s3):
        # A supporting component cannot fail the module; a wrong number must.
        assert self._component(s3)["criticality"] == "critical"

    def test_missing_verdict_is_na_never_green(self, s3):
        comp = self._component(s3)
        assert comp["status"].startswith("N/A")
        assert comp["status"] != "GREEN"
        assert "§2.3a" in comp["status_reason"]

    def test_failing_verdict_is_red(self, s3):
        _put_backtester_attestation(s3, {
            "schema": "backtest_attestation-1.0.0", "run_date": RUN_DATE,
            "status": "ok", "verdict": "FAIL", "n_checks": 5, "n_failed": 1,
            "checks": [{"name": "fee_charged_both_sides", "passed": False}],
        })
        comp = self._component(s3)
        assert comp["status"] == "RED"
        assert "fee_charged_both_sides" in comp["status_reason"]

    def test_passing_verdict_is_green(self, s3):
        _put_backtester_attestation(s3, {
            "schema": "backtest_attestation-1.0.0", "run_date": RUN_DATE,
            "status": "ok", "verdict": "PASS", "n_checks": 5, "n_failed": 0,
        })
        assert self._component(s3)["status"] == "GREEN"


class TestDigestCarriesTheVerdict:
    def test_a_non_pass_verdict_is_rendered_before_the_tiles(self):
        from director.report_card_digest import summarize_report_card

        text = summarize_report_card({
            "_provenance": {"run_date": RUN_DATE},
            "tiles_overall_status": "GREEN",
            "tiles": {"backtester": {"status": "GREEN", "components": []}},
            "attestation": {"verdict": "FAIL", "reason": "backtester attestation FAILED."},
        })
        assert "CORRECTNESS ATTESTATION: FAIL" in text
        assert text.index("CORRECTNESS ATTESTATION") < text.index("## backtester")

    def test_a_card_with_no_attestation_block_reads_as_unknown(self):
        from director.report_card_digest import summarize_report_card

        text = summarize_report_card({
            "_provenance": {"run_date": RUN_DATE},
            "tiles_overall_status": "GREEN", "tiles": {},
        })
        assert "CORRECTNESS ATTESTATION: UNKNOWN" in text

    def test_staleness_degradation_also_reaches_the_prompt(self):
        """aggregate.py has claimed since config#2885 that the Director's prompt
        MUST check degraded_staleness; until now the flag never left the JSON."""
        from director.report_card_digest import summarize_report_card

        text = summarize_report_card({
            "_provenance": {"run_date": RUN_DATE},
            "tiles_overall_status": "GREEN", "tiles": {},
            "attestation": {"verdict": "PASS"},
            "degraded_staleness": True, "stale_tiles": ["research"],
        })
        assert "DEGRADED (staleness)" in text
        assert "research" in text


class TestReportCardCarriesTheVerdict:
    def test_top_level_block_and_degraded_flag(self, s3, monkeypatch):
        import grading.aggregate as aggregate

        monkeypatch.setattr(aggregate, "assert_input_freshness", lambda *a, **k: {})
        monkeypatch.setattr(aggregate, "read_scorecard_inputs",
                            lambda *a, **k: ({}, _StubReport()))
        monkeypatch.setattr(aggregate, "compute_scorecard",
                            lambda **k: {"status": "ok", "overall": {"grade": None, "letter": "N/A"}})
        monkeypatch.setattr(aggregate, "load_card_history", lambda *a, **k: None)
        for name in ("build_portfolio_outcome_tile", "build_predictor_tile", "build_research_tile",
                     "build_executor_tile", "build_backtester_tile", "build_substrate_tile",
                     "build_agent_tile", "build_behavioral_tile", "build_director_quality_tile"):
            monkeypatch.setattr(aggregate, name,
                                lambda *a, **k: {"status": "GREEN", "components": []})

        card = aggregate.build_report_card(BUCKET, RUN_DATE, s3_client=s3)
        assert card["attestation"]["schema"] == attestation.SCHEMA
        # No backtester attestation in the bucket → the card must say so.
        assert card["attestation"]["verdict"] == attestation.UNKNOWN
        assert card["degraded_attestation"] is True

    def test_degraded_flag_clears_only_on_a_full_pass(self, s3, monkeypatch):
        import grading.aggregate as aggregate

        _put_both_passing(s3)
        monkeypatch.setattr(aggregate, "assert_input_freshness", lambda *a, **k: {})
        monkeypatch.setattr(aggregate, "read_scorecard_inputs",
                            lambda *a, **k: ({}, _StubReport()))
        monkeypatch.setattr(aggregate, "compute_scorecard",
                            lambda **k: {"status": "ok", "overall": {"grade": None, "letter": "N/A"}})
        monkeypatch.setattr(aggregate, "load_card_history", lambda *a, **k: None)
        for name in ("build_portfolio_outcome_tile", "build_predictor_tile", "build_research_tile",
                     "build_executor_tile", "build_backtester_tile", "build_substrate_tile",
                     "build_agent_tile", "build_behavioral_tile", "build_director_quality_tile"):
            monkeypatch.setattr(aggregate, name,
                                lambda *a, **k: {"status": "GREEN", "components": []})

        card = aggregate.build_report_card(BUCKET, RUN_DATE, s3_client=s3)
        assert card["attestation"]["verdict"] == attestation.PASS
        assert card["degraded_attestation"] is False


class TestProducerConsumerContract:
    """M0 contract discipline — the consumer half of
    `crucible-backtester contracts/backtest_attestation.schema.json`.

    The producer half (a real `run_attestation()` body validating against the
    schema) lives in that repo; this half pins the fields THIS consumer reads, so
    a producer-side rename cannot silently degrade every card to UNKNOWN.
    """

    #: A verbatim v1 body as the producer emits it.
    PRODUCER_BODY = {
        "schema": "backtest_attestation-1.0.0",
        "run_date": RUN_DATE,
        "status": "ok",
        "verdict": "PASS",
        "checks": [
            {"name": "pnl_no_fees", "description": "…", "expected": 0.0003,
             "observed": 0.0003, "abs_error": 0.0, "rtol": 1e-9, "atol": 1e-12,
             "passed": True},
        ],
        "n_checks": 5,
        "n_failed": 0,
        "n_errored": 0,
        "engine": {"python": "3.11.9", "vectorbt": "0.28.5", "numpy": "2.4.6",
                   "pandas": "2.2.3", "numba": "0.61.0"},
        "wall_clock_seconds": 1.42,
    }

    #: Exactly the fields this consumer reads. A producer change to any of these
    #: is a breaking change and needs a coordinated major schema bump.
    CONSUMED_FIELDS = ("schema", "run_date", "verdict", "n_checks", "n_failed",
                       "n_errored", "engine", "checks")

    def test_consumes_a_verbatim_producer_body(self, s3):
        _put_backtester_attestation(s3, self.PRODUCER_BODY)
        block = attestation.read_backtester_attestation(BUCKET, RUN_DATE, s3_client=s3)
        assert block["verdict"] == attestation.PASS
        assert block["n_checks"] == 5
        assert block["engine"]["vectorbt"] == "0.28.5"

    @pytest.mark.parametrize("field", CONSUMED_FIELDS)
    def test_each_consumed_field_can_go_missing_without_an_exception(self, s3, field):
        """Robustness, not permission: the consumer degrades rather than crashing
        the whole Report Card build when a producer field is absent."""
        body = {k: v for k, v in self.PRODUCER_BODY.items() if k != field}
        _put_backtester_attestation(s3, body)
        block = attestation.read_backtester_attestation(BUCKET, RUN_DATE, s3_client=s3)
        assert block["verdict"] in (attestation.PASS, attestation.UNKNOWN)
        # Losing run_date or verdict must cost the guarantee, not be shrugged off.
        if field in ("run_date", "verdict"):
            assert block["verdict"] == attestation.UNKNOWN


class _StubReport:
    def as_dict(self):
        return {"n_read": 0, "n_missing": 0}


class TestEvaluatorStageProducerConsumerContract:
    """M0 contract discipline — the consumer half of
    `crucible-backtester contracts/evaluator_attestation.schema.json`.

    The producer half (a real `build_stage_attestation()` body validated against
    that schema) lives in that repo; this half pins the fields THIS consumer
    reads, so a producer-side rename cannot silently degrade every card to
    UNKNOWN without a test here going red first.
    """

    #: A body copied from the producer's emitted shape.
    PRODUCER_BODY = {
        "schema": "evaluator_attestation-1.0.0",
        "component": "evaluator",
        "run_date": RUN_DATE,
        "verdict": "PASS",
        "promotion_withheld": False,
        "own": {
            "schema": "evaluator_attestation-1.0.0",
            "component": "evaluator",
            "run_date": RUN_DATE,
            "status": "ok",
            "verdict": "PASS",
            "checks": [
                {"name": "ic_single_adjacent_transposition",
                 "description": "rho = 1 - 6*sum(d^2)/(n(n^2-1)) = 4493/4495",
                 "expected": 0.9995550611790879, "observed": 0.9995550611790878,
                 "abs_error": 1.1102230246251565e-16, "rtol": 1e-9, "atol": 1e-12,
                 "passed": True},
            ],
            "n_checks": 11,
            "n_failed": 0,
            "n_errored": 0,
            "engine": {"python": "3.11.9", "numpy": "2.4.6", "pandas": "2.2.3"},
            "wall_clock_seconds": 0.31,
        },
        "upstream": {
            "component": "backtester",
            "verdict": "PASS",
            "reason": "verdict read",
            "as_of": "2026-08-08T06:13:29Z",
            "key": f"backtest/{RUN_DATE}/attestation.json",
        },
    }

    #: Exactly the fields this consumer reads.
    CONSUMED_FIELDS = ("schema", "run_date", "verdict", "promotion_withheld", "own")

    def test_consumes_a_verbatim_producer_body(self, s3):
        _put_evaluator_stage_attestation(s3, self.PRODUCER_BODY)
        block = attestation.read_evaluator_stage_attestation(BUCKET, RUN_DATE, s3_client=s3)
        assert block["verdict"] == attestation.PASS
        assert block["n_checks"] == 11
        assert block["n_failed"] == 0
        assert block["engine"]["python"] == "3.11.9"
        assert block["promotion_withheld"] is False

    @pytest.mark.parametrize("field", CONSUMED_FIELDS)
    def test_each_consumed_field_can_go_missing_without_an_exception(self, s3, field):
        body = {k: v for k, v in self.PRODUCER_BODY.items() if k != field}
        _put_evaluator_stage_attestation(s3, body)
        block = attestation.read_evaluator_stage_attestation(BUCKET, RUN_DATE, s3_client=s3)
        assert block["verdict"] in (attestation.PASS, attestation.UNKNOWN)
        if field in ("run_date", "verdict"):
            assert block["verdict"] == attestation.UNKNOWN

    def test_named_failing_checks_reach_the_reason(self, s3):
        body = json.loads(json.dumps(self.PRODUCER_BODY))
        body["verdict"] = "FAIL"
        body["own"]["verdict"] = "FAIL"
        body["own"]["n_failed"] = 1
        body["own"]["checks"][0]["passed"] = False
        _put_evaluator_stage_attestation(s3, body)
        block = attestation.read_evaluator_stage_attestation(BUCKET, RUN_DATE, s3_client=s3)
        assert block["verdict"] == attestation.FAIL
        assert "ic_single_adjacent_transposition" in block["reason"]


class TestBacktesterTileCarriesTheStageVerdict:
    """The tile component must be CRITICAL and must never default to GREEN."""

    def _component(self, s3, monkeypatch):
        import grading.tiles.backtester as tile_mod

        tile = tile_mod.build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        by_name = {c["name"]: c for c in tile["components"]}
        return by_name["evaluator_stage_attestation"]

    def test_absent_verdict_is_not_green(self, s3, monkeypatch):
        comp = self._component(s3, monkeypatch)
        assert comp["criticality"] == "critical"
        assert comp["status"] == "N/A-MISSING-INPUT"
        assert comp["value"] is None
        assert "absence is never a pass" in comp["status_reason"]

    def test_pass_is_green(self, s3, monkeypatch):
        _put_evaluator_stage_attestation(s3, _stage_body())
        comp = self._component(s3, monkeypatch)
        assert comp["status"] == "GREEN"
        assert comp["value"] == 1.0

    def test_fail_is_red(self, s3, monkeypatch):
        _put_evaluator_stage_attestation(s3, _stage_body(verdict="FAIL"))
        comp = self._component(s3, monkeypatch)
        assert comp["status"] == "RED"
        assert comp["value"] == 0.0


class TestDirectorDigestCarriesTheState:
    def _digest(self, card):
        from director.report_card_digest import summarize_report_card

        return summarize_report_card(card)

    def test_absent_attestation_block_warns(self):
        text = self._digest({"tiles": {}, "tiles_overall_status": "GREEN"})
        assert "CORRECTNESS ATTESTATION: UNKNOWN" in text

    def test_as_of_is_rendered_beside_the_state(self):
        text = self._digest({
            "tiles": {}, "tiles_overall_status": "GREEN",
            "attestation": {
                "verdict": "PASS", "reason": "ok",
                "as_of": {"backtester": "2026-08-08T06:13:29Z",
                          "evaluator_stage": "2026-08-08T06:14:02Z"},
            },
        })
        assert "2026-08-08T06:13:29Z" in text
        assert "2026-08-08T06:14:02Z" in text

    def test_a_never_written_half_renders_as_never_not_as_blank(self):
        text = self._digest({
            "tiles": {}, "tiles_overall_status": "GREEN",
            "attestation": {
                "verdict": "UNKNOWN", "reason": "absent",
                "as_of": {"backtester": "2026-08-08T06:13:29Z",
                          "evaluator_stage": None},
            },
        })
        assert "evaluator_stage as-of never" in text

    def test_withheld_promotion_is_stated_in_terms_of_the_live_system(self):
        text = self._digest({
            "tiles": {}, "tiles_overall_status": "GREEN",
            "attestation": {"verdict": "UNKNOWN", "reason": "absent",
                            "promotion_withheld": True},
        })
        assert "PROMOTION WITHHELD" in text
        assert "PREVIOUS cycle's" in text
