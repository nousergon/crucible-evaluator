"""Tests for grading/tiles/backtester.py — Tile 4 (self-grade)."""

import json
from datetime import UTC, datetime, timedelta

import boto3
import pytest
from moto import mock_aws

from grading.history import CardHistory
from grading.tiles.backtester import (
    REGISTRY_MANIFEST_KEY_FMT,
    _coverage,
    _fdr_surface_status,
    build_backtester_tile,
)
from grading.tiles.predictor import LATEST_KEY, MANIFEST_KEY

BUCKET = "alpha-engine-research"
RUN_DATE = "2026-06-07"
SERVED = "v3.0-meta-2026-08-14-119e069b"
REGISTRY_KEY = REGISTRY_MANIFEST_KEY_FMT.format(served_version=SERVED)


def _put_served(s3, mean_ic, status="ok", standing_mean_ic=None):
    """Write the standing manifest as a served_version POINTER and the served
    model's frozen registry bundle carrying the CPCV (alpha-engine-config-I12123).
    ``standing_mean_ic`` puts a DIFFERENT (challenger) CPCV in the standing key,
    which the tile must ignore."""
    standing = {"served_version": SERVED}
    if standing_mean_ic is not None:
        standing["meta_model_oos_ic_cpcv"] = {"status": "ok", "mean_ic": standing_mean_ic, "n_combos": 12}
    s3.put_object(Bucket=BUCKET, Key=MANIFEST_KEY, Body=json.dumps(standing).encode())
    s3.put_object(Bucket=BUCKET, Key=REGISTRY_KEY, Body=json.dumps(
        {"meta_model_oos_ic_cpcv": {"status": status, "mean_ic": mean_ic, "n_combos": 44}}).encode())


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _put(s3, key, data, raw=False):
    body = data.encode() if raw else json.dumps(data).encode()
    s3.put_object(Bucket=BUCKET, Key=key, Body=body)


def _comp(tile, name):
    return next(c for c in tile["components"] if c["name"] == name)


# A grading.json mirroring the live v1 shape: lots of N/A in executor/predictor.
# RC v3 T1 (config-I7474): the v1 "letter" field is retired; a leaf's
# gradedness is now keyed off "grade" (numeric, None == N/A) instead.
_GRADING = {
    "status": "partial",
    "research": {"components": {
        "scanner": {"grade": 35.0},
        "sector_teams": [{"grade": 45.0}, {"grade": None}, {"grade": 65.0}],
        "sector_teams_avg": {"grade": 20.0},  # rollup — excluded
        "macro_agent": {"grade": 38.0},
    }},
    "predictor": {"components": {"meta_model": {"grade": 45.0}, "veto_gate": {"grade": None}}},
    "executor": {"components": {
        "entry_triggers": {"grade": None}, "risk_guard": {"grade": None},
        "exit_rules": {"grade": None}, "portfolio": {"grade": 78.0},
    }},
}


class TestCoverageHelper:
    def test_counts_leaves_excludes_avg(self):
        cov, graded, total = _coverage(_GRADING)
        # leaves: scanner, 3 sector_teams items, macro_agent (5 research, avg excluded)
        #         + meta_model, veto_gate (2) + entry/risk/exit/portfolio (4) = 11 total
        # graded (non-N/A): scanner, 2 teams, macro_agent (4) + meta_model (1)
        #         + portfolio (1) = 6
        assert total == 11
        assert graded == 6
        assert cov == pytest.approx(6 / 11)


class TestEvaluatorCoverage:
    def test_low_coverage_red(self, s3):
        _put(s3, f"backtest/{RUN_DATE}/grading.json", _GRADING)
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        cov = _comp(tile, "evaluator_coverage")
        assert cov["value"] == pytest.approx(6 / 11)
        assert cov["status"] == "RED"  # 0.55 < red-line 0.80

    def test_absent_grading_missing_input(self, s3):
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert _comp(tile, "evaluator_coverage")["status"] == "N/A-MISSING-INPUT"


class TestFreshness:
    def test_fresh_grading_green(self, s3):
        _put(s3, f"backtest/{RUN_DATE}/grading.json", _GRADING)
        # moto stamps LastModified ≈ now → age ~0h → GREEN.
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3, as_of=datetime.now(UTC))
        fr = _comp(tile, "grading_freshness")
        assert fr["status"] == "GREEN"

    def test_stale_grading_red(self, s3):
        _put(s3, f"backtest/{RUN_DATE}/grading.json", _GRADING)
        future = datetime.now(UTC) + timedelta(hours=200)  # > 192h red-line
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3, as_of=future)
        assert _comp(tile, "grading_freshness")["status"] == "RED"


class TestParity:
    def test_clean_parity_green(self, s3):
        _put(s3, f"backtest/{RUN_DATE}/parity_report.json",
             {"data_state": "ok", "trade_count_divergence": {}, "ticker_set_divergence": {}})
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert _comp(tile, "vectorized_vs_consolidated_parity")["status"] == "GREEN"

    # ── config-I7222: a declared failure is RED, never N/A ────────────────
    #
    # The assertion that used to live here — `status == "N/A-MISSING-INPUT"`
    # for a `backtester_replay_error` report — PINNED the detection failure.
    # `backtest/{date}/parity_report.json` is the backtester's only real
    # grade, and between 2026-05-29 and 2026-07-17 nine consecutive weekly
    # reports were written carrying `data_state: backtester_replay_error`,
    # every metric 0.0, `n_backtester_orders_total: 0`. Each one rendered
    # here as N/A — byte-identical in effect to "no parity_report.json this
    # cycle". The pipeline said, in writing, nine times, that its own grade
    # was broken, and the surface built to read it rendered that statement
    # as no-data. Root cause of the replay abort itself:
    # `crucible-backtester-PR<N>` (config-I7222) — both replay paths called
    # `_simulate_single_date` with no ATR map.
    #
    # `no data` is never rendered as green, and it is not rendered as
    # absent-input either when the artifact is present and says FAIL
    # (principles §2.7).

    def test_a_declared_replay_error_is_red_not_na(self, s3):
        """The exact payload shape of the nine reports that hid for eleven
        weeks. This is the assertion whose absence WAS the outage."""
        _put(s3, f"backtest/{RUN_DATE}/parity_report.json", {
            "data_state": "backtester_replay_error",
            "data_state_note": (
                "Backtester replay raised RuntimeError: atr_map missing EOG at "
                "decide_entries — load_atr_14_pct contract violated."
            ),
            "metrics": {"capture_rate": 0.0, "ticker_jaccard_avg": 0.0,
                        "data_state": "backtester_replay_error"},
            "n_backtester_orders_total": 0,
            "n_backtester_enters": 0,
        })
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        p = _comp(tile, "vectorized_vs_consolidated_parity")
        assert p["status"] == "RED", p
        assert p["value"] == 0.0
        assert "backtester_replay_error" in p["status_reason"]
        assert "UNPROVEN" in p["status_reason"]

    def test_the_verdict_fields_are_carried_into_the_reason(self, s3):
        """The producer now stamps schema/status/verdict on every path. The
        reason an operator reads must quote them — a RED tile that does not
        say WHY sends the reader back to S3."""
        _put(s3, f"backtest/{RUN_DATE}/parity_report.json", {
            "schema": "parity_report-0.0.0",
            "data_state": "backtester_replay_error",
            "status": "failed",
            "verdict": "FAIL",
            "verdict_reason": "the parity replay produced no usable comparison this run",
        })
        p = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3),
                  "vectorized_vs_consolidated_parity")
        assert p["status"] == "RED"
        assert "FAIL" in p["status_reason"]
        assert "no usable comparison" in p["status_reason"]

    @pytest.mark.parametrize(
        "data_state",
        ["backtester_replay_error", "empty_trades_db", "insufficient_cohort_dates",
         "some_state_nobody_has_written_yet"],
    )
    def test_every_non_ok_data_state_is_red(self, s3, data_state):
        """Deliberately including an unknown value. The split is on `is this a
        clean comparison`, not on an allowlist of known-bad strings — an
        allowlist is how the NEXT degraded state goes quiet."""
        _put(s3, f"backtest/{RUN_DATE}/parity_report.json", {"data_state": data_state})
        p = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3),
                  "vectorized_vs_consolidated_parity")
        assert p["status"] == "RED", data_state

    def test_the_launcher_absence_marker_is_red(self, s3):
        """`spot_parity_replay.sh` records a missing artifact as a PRESENT
        artifact saying it is absent. It carries no `data_state` at all — it
        must still be red, not N/A."""
        _put(s3, f"backtest/{RUN_DATE}/parity_report.json", {
            "schema": "parity_report-0.0.0", "run_date": RUN_DATE,
            "status": "failed", "verdict": "FAIL",
            "verdict_reason": "the parity replay produced no report this run",
            "producer_exit": 1,
        })
        p = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3),
                  "vectorized_vs_consolidated_parity")
        assert p["status"] == "RED"

    def test_a_genuinely_absent_report_stays_na(self, s3):
        """Absence is still an absence — the artifact's own non-arrival is
        owned by the launcher's absence marker and the freshness detector, not
        by this metric inventing a failure it did not observe."""
        p = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3),
                  "vectorized_vs_consolidated_parity")
        assert p["status"] == "N/A-MISSING-INPUT"
        assert "absent this cycle" in p["status_reason"]

    def test_a_stale_clean_report_stays_na_not_red(self, s3):
        """A clean comparison from an older cycle is a missing measurement for
        THIS cycle, which the staleness detector owns. Turning it RED here
        would double-count one fact and make the stale signal unreadable."""
        old = "2026-01-05"
        _put(s3, f"backtest/{old}/parity_report.json",
             {"data_state": "ok", "trade_count_divergence": {}, "ticker_set_divergence": {}})
        p = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3),
                  "vectorized_vs_consolidated_parity")
        assert p["status"] == "N/A-MISSING-INPUT"


def _scope_stage(disposition, flag, disabled_by=None):
    row = {"disposition": disposition, "flag": flag, "gate": f"CheckSkip{flag}"}
    if disposition == "DISABLED":
        row["disabled_by"] = disabled_by or flag
    return row


def _run_scope(parity="DISABLED", replay=None, parity_flag="skip_parity", replay_by=None):
    """The raw ``run_scope.json`` shape the live weekly writes
    (``backtest/2026-10-16/run_scope.json``: ``Parity`` DISABLED by
    ``skip_parity`` at its own gate, ``ParityReplay`` DISABLED by the same flag
    through its parent gate)."""
    replay = replay or parity
    return {
        "schema": "run_scope-1.0.0", "run_date": RUN_DATE,
        "graded_stages": ["Backtester"],
        "stages": {
            "Backtester": _scope_stage("ENABLED_COMPLETED", "skip_backtester"),
            "Parity": _scope_stage(parity, parity_flag),
            "ParityReplay": _scope_stage(replay, "skip_parity_replay",
                                         disabled_by=replay_by or parity_flag),
        },
    }


class TestParityDisabledByInput:
    """alpha-engine-config-I12389: a parity stage switched off by input is
    ``N/A, disabled``, not "absent". ``skip_parity`` has been on since
    2026-08-13 by a standing ruling (I7309), and "parity_report.json absent
    this cycle" made the weekly Director file that decision as a missing
    producer. The absence reading must survive every case where the stage
    was dispatched, or where the scope cannot say."""

    def _parity(self, s3, **kw):
        return _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3, **kw),
                     "vectorized_vs_consolidated_parity")

    def test_skip_parity_reads_disabled_not_absent(self, s3):
        _put(s3, f"backtest/{RUN_DATE}/run_scope.json", _run_scope())
        p = self._parity(s3)
        assert p["status"] == "N/A-NOT-RUN", p
        assert "N/A, disabled (skip_parity, I7309)" in p["status_reason"]
        assert "absent" not in p["status_reason"]
        assert "not a pass" in p["status_reason"]

    def test_in_band_scope_is_used_without_an_s3_artifact(self, s3):
        """The card builder hands the tile the scope it already resolved
        (in-band payload first). No S3 artifact is needed."""
        p = self._parity(s3, run_scope=_run_scope())
        assert p["status"] == "N/A-NOT-RUN"
        assert "skip_parity, I7309" in p["status_reason"]

    def test_an_explicit_none_scope_is_absent_and_not_re_read(self, s3):
        """``None`` from the card builder means both delivery paths were
        empty. An S3 object that appears afterwards is not consulted, and the
        reading is absent."""
        _put(s3, f"backtest/{RUN_DATE}/run_scope.json", _run_scope())
        p = self._parity(s3, run_scope=None)
        assert p["status"] == "N/A-MISSING-INPUT"
        assert "absent this cycle" in p["status_reason"]

    def test_replay_alone_disabled_names_its_own_flag(self, s3):
        p = self._parity(s3, run_scope=_run_scope(
            parity="ENABLED_COMPLETED", replay="DISABLED", replay_by="skip_parity_replay"))
        assert p["status"] == "N/A-NOT-RUN"
        assert "N/A, disabled (skip_parity_replay)" in p["status_reason"]
        assert "I7309" not in p["status_reason"]

    @pytest.mark.parametrize("disposition", ["ENABLED_COMPLETED", "ENABLED_FAILED", "NOT_REACHED"])
    def test_a_stage_that_was_not_disabled_still_flags_absence(self, s3, disposition):
        """Dispatched (or never reached) and no report: that is the gap
        #12389 is about, and it must still read as absent."""
        p = self._parity(s3, run_scope=_run_scope(parity=disposition))
        assert p["status"] == "N/A-MISSING-INPUT", disposition
        assert "absent this cycle" in p["status_reason"]

    @pytest.mark.parametrize("scope", [
        None,
        {},
        {"degraded": True, "degraded_reason": "history unreadable"},
        {"stages": {}},
        {"stages": {"Parity": {"disposition": "PROBABLY_FINE"},
                    "ParityReplay": {"disposition": "PROBABLY_FINE"}}},
    ])
    def test_an_unknown_scope_never_excuses_the_absence(self, s3, scope):
        p = self._parity(s3, run_scope=scope)
        assert p["status"] == "N/A-MISSING-INPUT", scope
        assert "absent this cycle" in p["status_reason"]

    def test_no_scope_artifact_at_all_is_absent(self, s3):
        p = self._parity(s3)
        assert p["status"] == "N/A-MISSING-INPUT"

    def test_a_present_report_is_graded_whatever_the_scope_says(self, s3):
        """The scope only explains a MISSING report. A report that exists is
        graded on its own content; a declared failure stays RED."""
        _put(s3, f"backtest/{RUN_DATE}/parity_report.json",
             {"data_state": "backtester_replay_error"})
        p = self._parity(s3, run_scope=_run_scope())
        assert p["status"] == "RED"

    def test_disabled_parity_does_not_block_a_green_tile(self, s3):
        """Supporting N/A, as before: the decision to stop measuring parity
        must not move the tile's status any more than the old absent N/A did."""
        _put(s3, f"backtest/{RUN_DATE}/run_scope.json", _run_scope())
        tile_disabled = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        tile_absent = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3, run_scope=None)
        assert tile_disabled["status"] == tile_absent["status"]


class TestFdrSurface:
    """config#2305: the band scales with n_fdr_tests (the actual size of the
    shared BH-FDR correction pool), not a fixed absolute count — the fixed
    3-15 band was calibrated for the pre-config#1456 ~8-9-test surface and is
    unreachable/misleading on the post-cutover 4-5-test surface. Legacy
    artifacts without n_fdr_tests still get the old fixed-band behavior."""

    def test_healthy_band_green_with_n_fdr_tests(self, s3):
        # 3 of 10 tests significant = 0.30 fraction, within [0.15, 0.60].
        corr = {f"f{i}": {"r10_fdr_significant": i < 3} for i in range(10)}
        _put(s3, f"backtest/{RUN_DATE}/attribution.json",
             {"status": "ok", "rows_analyzed": 500, "n_fdr_tests": 10, "correlations": corr})
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        fdr = _comp(tile, "fdr_surface_health")
        assert fdr["value"] == 3.0
        assert fdr["status"] == "GREEN"

    def test_zero_significant_on_small_surface_is_watch_not_red(self, s3):
        """The live 2026-07-10 scenario: 4 tests (2 sub-scores x 2 canonical
        horizons post-config#1456), 0 survive BH correction — this is the
        statistically EXPECTED outcome at this sample size (best raw p=0.034
        still fails the n=4 rank-1 threshold of 0.0125), not evidence the
        surface went flat. Must be WATCH, not RED (the config#2305 bug)."""
        corr = {
            "quant": {"beat_spy_21d_fdr_significant": False, "return_21d_fdr_significant": False},
            "qual": {"beat_spy_21d_fdr_significant": False, "return_21d_fdr_significant": False},
        }
        _put(s3, f"backtest/{RUN_DATE}/attribution.json",
             {"status": "ok", "rows_analyzed": 402, "n_fdr_tests": 4, "correlations": corr})
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        fdr = _comp(tile, "fdr_surface_health")
        assert fdr["value"] == 0.0
        assert fdr["status"] == "WATCH"

    def test_zero_significant_on_large_surface_is_red(self, s3):
        """0 survivors IS informative once the correction pool is large
        enough for the test to have real discriminating power."""
        corr = {f"f{i}": {"r_fdr_significant": False} for i in range(12)}
        _put(s3, f"backtest/{RUN_DATE}/attribution.json",
             {"status": "ok", "rows_analyzed": 500, "n_fdr_tests": 12, "correlations": corr})
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert _comp(tile, "fdr_surface_health")["status"] == "RED"

    def test_all_significant_is_watch_not_green(self, s3):
        """Every test surviving correction reads as overfit-suspicious, not
        healthy — fraction 1.0 is outside the [0.15, 0.60] GREEN band."""
        corr = {f"f{i}": {"r_fdr_significant": True} for i in range(4)}
        _put(s3, f"backtest/{RUN_DATE}/attribution.json",
             {"status": "ok", "rows_analyzed": 500, "n_fdr_tests": 4, "correlations": corr})
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert _comp(tile, "fdr_surface_health")["status"] == "WATCH"

    def test_legacy_artifact_without_n_fdr_tests_uses_fixed_band(self, s3):
        """Older attribution.json artifacts (written before config#2305)
        lack n_fdr_tests entirely — must fall back to the original fixed
        3-15 band rather than erroring or silently reinterpreting."""
        corr = {f"f{i}": {"r10_fdr_significant": True} for i in range(5)}
        _put(s3, f"backtest/{RUN_DATE}/attribution.json",
             {"status": "ok", "rows_analyzed": 500, "correlations": corr})  # no n_fdr_tests key
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        fdr = _comp(tile, "fdr_surface_health")
        assert fdr["value"] == 5.0
        assert fdr["status"] == "GREEN"  # 3 <= 5 <= 15, legacy band

    def test_legacy_zero_significant_red(self, s3):
        _put(s3, f"backtest/{RUN_DATE}/attribution.json",
             {"status": "ok", "rows_analyzed": 500, "correlations": {"f1": {"r10_fdr_significant": False}}})
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert _comp(tile, "fdr_surface_health")["status"] == "RED"

    def test_empty_attribution_tolerated(self, s3):
        # corrupt/empty artifact → N/A, not a crash.
        _put(s3, f"backtest/{RUN_DATE}/attribution.json", "", raw=True)
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert _comp(tile, "fdr_surface_health")["status"] == "N/A-MISSING-INPUT"


class TestFdrSurfaceStatusHelper:
    """Direct unit tests for `_fdr_surface_status` — pins the calibration
    contract independent of the S3/tile plumbing."""

    def test_none_n_fdr_tests_uses_legacy_band(self):
        assert _fdr_surface_status(5, None)[0] == "GREEN"
        assert _fdr_surface_status(0, None)[0] == "RED"
        assert _fdr_surface_status(2, None)[0] == "WATCH"
        assert _fdr_surface_status(20, None)[0] == "WATCH"
        assert _fdr_surface_status(40, None)[0] == "RED"

    def test_zero_tests_edge_case_falls_back_to_legacy(self):
        # n_fdr_tests=0 shouldn't happen when status=="ok", but must not crash.
        status, _ = _fdr_surface_status(0, 0)
        assert status == "RED"  # legacy rule: n_sig=0 -> RED

    def test_small_surface_zero_sig_is_watch(self):
        for n in (1, 2, 3, 4, 5, 9):
            assert _fdr_surface_status(0, n)[0] == "WATCH", f"n_fdr_tests={n}"

    def test_large_surface_zero_sig_is_red(self):
        for n in (10, 15, 50):
            assert _fdr_surface_status(0, n)[0] == "RED", f"n_fdr_tests={n}"

    def test_proportional_green_band(self):
        # 0.15 <= frac <= 0.60
        assert _fdr_surface_status(2, 10)[0] == "GREEN"   # 0.20
        assert _fdr_surface_status(6, 10)[0] == "GREEN"   # 0.60 (inclusive)
        assert _fdr_surface_status(1, 10)[0] == "WATCH"   # 0.10, below floor
        assert _fdr_surface_status(7, 10)[0] == "WATCH"   # 0.70, above ceiling


class TestRollbackAndNA:
    def test_rollback_count(self, s3):
        _put(s3, "config/rollback_audit/2026-05-01.json", {"x": 1})
        _put(s3, "config/rollback_audit/2026-05-08.json", {"x": 2})
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        rb = _comp(tile, "auto_apply_rollback_count")
        assert rb["value"] == 2.0

    def test_no_more_stub_components(self, s3):
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3)
        # sample_size_adequacy, optimizer_churn, walk_forward_stability
        # (config#1151) and backtest_vs_live_parity (config#1153) are all now
        # WIRED — they read producer artifacts, so absent input ⇒
        # N/A-MISSING-INPUT, never N/A-NOT-IMPL. No genuine stub remains on
        # this tile.
        for name in ("backtest_vs_live_parity", "optimizer_churn", "walk_forward_stability"):
            assert _comp(tile, name)["status"] == "N/A-MISSING-INPUT"


class TestOptimizerChurn:
    """optimizer_churn reads backtest/{date}/optimizer_churn.json (config#1151)."""

    def _put(self, s3, churn_ratio, within):
        _put(s3, f"backtest/{RUN_DATE}/optimizer_churn.json", {
            "status": "ok", "churn_ratio": churn_ratio, "max_abs_change": 0.06,
            "max_change_param": "momentum", "guardrail_cap": 0.10,
            "within_guardrails": within, "n_params_changed": 3})

    def test_missing_input(self, s3):
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "optimizer_churn")
        assert m["status"] == "N/A-MISSING-INPUT"
        assert "config#1151" in m["status_reason"]
        assert m["criticality"] == "critical"

    def test_within_guardrails_green(self, s3):
        self._put(s3, 0.6, True)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "optimizer_churn")
        assert m["status"] == "GREEN"
        assert m["value"] == 0.6
        assert "within guardrails" in m["status_reason"]

    def test_approaching_cap_watch(self, s3):
        self._put(s3, 0.9, True)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "optimizer_churn")
        assert m["status"] == "WATCH"

    def test_over_cap_red(self, s3):
        self._put(s3, 1.1, False)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "optimizer_churn")
        assert m["status"] == "RED"
        assert "AT/OVER" in m["status_reason"]

    def test_insufficient_data_na(self, s3):
        _put(s3, f"backtest/{RUN_DATE}/optimizer_churn.json",
             {"status": "insufficient_data", "reason": "no usable recommendation"})
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "optimizer_churn")
        assert m["status"] == "N/A-MISSING-INPUT"


class TestWalkForwardStability:
    """walk_forward_stability reads backtest/{date}/walk_forward_stability.json (config#1151)."""

    def _put(self, s3, stability_ratio, n_reversals, weeks=4, stable=True):
        _put(s3, f"backtest/{RUN_DATE}/walk_forward_stability.json", {
            "status": "ok", "stability_ratio": stability_ratio, "n_reversals": n_reversals,
            "max_possible_reversals": 10, "weeks_loaded": weeks, "stable": stable, "reversals": []})

    def test_missing_input(self, s3):
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "walk_forward_stability")
        assert m["status"] == "N/A-MISSING-INPUT"
        assert m["criticality"] == "supporting"

    def test_stable_green(self, s3):
        self._put(s3, 0.9, 1)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "walk_forward_stability")
        assert m["status"] == "GREEN"
        assert m["value"] == 0.9
        assert m["n_samples"] == 4

    def test_drifting_watch(self, s3):
        self._put(s3, 0.6, 4)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "walk_forward_stability")
        assert m["status"] == "WATCH"

    def test_oscillating_red(self, s3):
        self._put(s3, 0.3, 7, stable=False)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "walk_forward_stability")
        assert m["status"] == "RED"

    def test_insufficient_history_na(self, s3):
        _put(s3, f"backtest/{RUN_DATE}/walk_forward_stability.json",
             {"status": "insufficient_data", "reason": "only 1 prior week", "weeks_loaded": 1})
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "walk_forward_stability")
        assert m["status"] == "N/A-MISSING-INPUT"


class TestSampleSizeAdequacy:
    """sample_size_adequacy reads backtest/{date}/sample_size.json (config#1151)."""

    def _put_ss(self, s3, ratio, weakest="signal_quality", per=None):
        per = per or {"signal_quality": {"n": int(ratio * 60), "floor": 60, "adequacy_ratio": ratio}}
        _put(s3, f"backtest/{RUN_DATE}/sample_size.json", {
            "status": "ok", "adequacy_ratio": ratio, "adequate": ratio >= 1.0,
            "weakest_analysis": weakest, "per_analysis": per})

    def test_missing_input(self, s3):
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "sample_size_adequacy")
        assert m["status"] == "N/A-MISSING-INPUT"
        assert "config#1151" in m["status_reason"]
        assert m["criticality"] == "critical"

    def test_well_powered_green(self, s3):
        self._put_ss(s3, 1.5)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "sample_size_adequacy")
        assert m["status"] == "GREEN"
        assert m["value"] == 1.5
        assert "well-powered" in m["status_reason"]

    def test_building_watch(self, s3):
        self._put_ss(s3, 0.7)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "sample_size_adequacy")
        assert m["status"] == "WATCH"

    def test_severely_underpowered_red(self, s3):
        self._put_ss(s3, 0.3, weakest="attribution",
                     per={"attribution": {"n": 30, "floor": 100, "adequacy_ratio": 0.3}})
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "sample_size_adequacy")
        assert m["status"] == "RED"
        assert "under-powered" in m["status_reason"]
        assert m["n_samples"] == 30

    def test_windowed_grades_off_earlier_date(self, s3):
        # Producer ran 2 days before run_date (partial/off-cycle) → still grades.
        _put(s3, "backtest/2026-06-05/sample_size.json", {
            "status": "ok", "adequacy_ratio": 1.2, "adequate": True,
            "weakest_analysis": "signal_quality",
            "per_analysis": {"signal_quality": {"n": 72, "floor": 60, "adequacy_ratio": 1.2}}})
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "sample_size_adequacy")
        assert m["status"] == "GREEN" and m["value"] == 1.2


class TestRiskRatioCI:
    """risk_ratio_ci reads backtest/{date}/risk_ratio_ci.json (config#976,
    Director L4558's magnitude-certainty no-action monitor)."""

    def _put_rr(self, s3, all_certain, n_samples=200, sample_floor=126, ratios=None):
        ratios = ratios if ratios is not None else {
            "sharpe_ratio": {"point": 1.1, "ci_95": [0.4, 1.8], "straddles_zero": False,
                              "magnitude_certain": all_certain},
            "sortino_ratio": {"point": 1.4, "ci_95": [0.6, 2.2], "straddles_zero": False,
                               "magnitude_certain": all_certain},
            "information_ratio": {"point": 0.3, "ci_95": [-0.1, 0.7], "straddles_zero": not all_certain,
                                   "magnitude_certain": all_certain},
        }
        _put(s3, f"backtest/{RUN_DATE}/risk_ratio_ci.json", {
            "status": "ok", "n_samples": n_samples, "sample_floor": sample_floor,
            "n_adequate": n_samples >= sample_floor, "ratios": ratios,
            "all_magnitude_certain": all_certain,
            "note": "Director L4558: size remediation by direction, not magnitude, until certain.",
        })

    def test_missing_input(self, s3):
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "risk_ratio_ci")
        assert m["status"] == "N/A-MISSING-INPUT"
        assert "config#976" in m["status_reason"]
        assert m["criticality"] == "critical"

    def test_all_certain_green(self, s3):
        self._put_rr(s3, all_certain=True, n_samples=200)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "risk_ratio_ci")
        assert m["status"] == "GREEN"
        assert m["value"] == 1.0
        assert m["n_samples"] == 200

    def test_magnitude_uncertain_watch_not_red(self, s3):
        # A no-action monitor: CI-straddles-zero is a data state to wait out,
        # not a regression — must grade WATCH, never RED.
        self._put_rr(s3, all_certain=False, n_samples=63)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "risk_ratio_ci")
        assert m["status"] == "WATCH"
        assert m["value"] == 0.0
        assert "information_ratio" in m["status_reason"]
        assert "DIRECTION" in m["status_reason"]

    def test_insufficient_data_status_grades_na(self, s3):
        _put(s3, f"backtest/{RUN_DATE}/risk_ratio_ci.json", {
            "status": "insufficient_data", "n_samples": 1, "sample_floor": 126,
            "n_adequate": False, "ratios": {}, "all_magnitude_certain": False,
            "note": "Fewer than 2 aligned daily returns — no ratio estimable.",
        })
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "risk_ratio_ci")
        assert m["status"].startswith("N/A")

    def test_windowed_grades_off_earlier_date(self, s3):
        _put(s3, "backtest/2026-06-05/risk_ratio_ci.json", {
            "status": "ok", "n_samples": 300, "sample_floor": 126, "n_adequate": True,
            "ratios": {
                "sharpe_ratio": {"point": 0.9, "ci_95": [0.2, 1.6], "straddles_zero": False,
                                  "magnitude_certain": True},
            },
            "all_magnitude_certain": True,
            "note": "x",
        })
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "risk_ratio_ci")
        assert m["status"] == "GREEN" and m["value"] == 1.0


class TestBacktestVsLiveParity:
    """backtest_vs_live_parity (config#1153 critical-3) — diffs the predictor's
    live 30d rolling IC (predictor/metrics/latest.json::ic_30d) against its own
    leak-free CPCV backtest-expected IC (predictor/weights/meta/manifest.json::
    meta_model_oos_ic_cpcv.mean_ic). Sign convention: drift = live − backtest-
    promised; drift <= red_line (-0.05) is the actionable/bad direction (live
    underperforming what the backtest promised)."""

    def _put_manifest(self, s3, mean_ic=0.18, status="ok"):
        _put_served(s3, mean_ic, status=status)

    def _put_latest(self, s3, ic_30d=0.15, rolling_n=40, n_effective=8):
        # The producer's current shape (crucible-backtester pipeline_common,
        # alpha-engine-config-I8701; preserved by the daily write since
        # crucible-predictor#657): `realized_ic.n_effective` is the N the
        # tile grades on (alpha-engine-config-I10065).
        _put(s3, LATEST_KEY, {
            "ic_30d": ic_30d, "rolling_n": rolling_n, "forward_days": 21,
            "realized_ic": {"value": ic_30d, "horizon_days": 21, "n_rows": rolling_n,
                            "n_prediction_dates": n_effective * 21, "n_effective": n_effective},
        })

    def test_both_present_computes_drift(self, s3):
        self._put_manifest(s3, mean_ic=0.18)
        self._put_latest(s3, ic_30d=0.15, rolling_n=40)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "backtest_vs_live_parity")
        assert m["value"] == pytest.approx(0.15 - 0.18)
        assert m["n_samples"] == 8  # realized_ic.n_effective, not rolling_n=40
        assert m["criticality"] == "critical"
        assert m["estimator"] == "ic_delta"

    def test_reliability_is_declared_low_not_defaulted_high(self, s3):
        """alpha-engine-config-I8180 — the instrument carries two documented
        caveats (an AMBIGUOUS cross-estimator delta and a self-declared
        UNVALIDATED first-cut band), so the call site must say ``low``.

        This is a REGRESSION test on a silence: ``build_metric`` defaults a
        value-bearing critical to ``reliability="high"`` when the call site
        omits it, so deleting the keyword does not fail loudly anywhere else —
        it just quietly re-publishes a critical RED/F, with a trend arrow, as
        a high-confidence read. Asserting the value here is what makes that
        deletion a test failure."""
        self._put_manifest(s3, mean_ic=0.18)
        self._put_latest(s3, ic_30d=0.15, rolling_n=40)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "backtest_vs_live_parity")
        assert m["reliability"] == "low"

    def test_reliability_low_survives_every_status_band(self, s3):
        """The caveat is a property of the INSTRUMENT, not of this week's
        reading — a GREEN parity number is exactly as ambiguous as a RED one,
        so ``low`` must not be conditional on the value."""
        for mean_ic, ic_30d in ((0.18, 0.18), (0.18, 0.15), (0.18, 0.08)):
            self._put_manifest(s3, mean_ic=mean_ic)
            self._put_latest(s3, ic_30d=ic_30d, rolling_n=40)
            m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "backtest_vs_live_parity")
            assert m["reliability"] == "low", (mean_ic, ic_30d, m["status"])

    def test_status_reason_states_neither_leg_measures_execution(self, s3):
        """The component's NAME asserts backtest-vs-live parity; measured
        2026-08-22, neither leg touches an order or a fill (``ic_30d`` comes
        from ``predictor_outcomes``). A reader who only sees the name and the
        RED cannot tell that, so the reason string has to say it."""
        self._put_manifest(s3, mean_ic=0.18)
        self._put_latest(s3, ic_30d=0.08, rolling_n=40)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "backtest_vs_live_parity")
        reason = m["status_reason"]
        assert "reliability LOW" in reason
        assert "UNVALIDATED" in reason
        assert "EXECUTION" in reason

    def test_live_matching_backtest_is_green(self, s3):
        # drift == target (0.0) exactly → GREEN.
        self._put_manifest(s3, mean_ic=0.18)
        self._put_latest(s3, ic_30d=0.18, rolling_n=40)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "backtest_vs_live_parity")
        assert m["value"] == pytest.approx(0.0)
        assert m["status"] == "GREEN"

    def test_live_beating_backtest_is_green(self, s3):
        self._put_manifest(s3, mean_ic=0.10)
        self._put_latest(s3, ic_30d=0.20, rolling_n=40)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "backtest_vs_live_parity")
        assert m["value"] == pytest.approx(0.10)
        assert m["status"] == "GREEN"

    def test_mild_underperformance_is_watch(self, s3):
        # drift = -0.03: below target 0.0, above red-line -0.05 → WATCH.
        self._put_manifest(s3, mean_ic=0.18)
        self._put_latest(s3, ic_30d=0.15, rolling_n=40)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "backtest_vs_live_parity")
        assert m["value"] == pytest.approx(-0.03)
        assert m["status"] == "WATCH"

    def test_severe_underperformance_is_red(self, s3):
        # drift = -0.10: at/below red-line -0.05 → RED — the actionable direction.
        self._put_manifest(s3, mean_ic=0.18)
        self._put_latest(s3, ic_30d=0.08, rolling_n=40)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "backtest_vs_live_parity")
        assert m["value"] == pytest.approx(-0.10)
        assert m["status"] == "RED"
        assert "underperforming" in m["status_reason"].lower()

    def test_missing_manifest_is_na(self, s3):
        self._put_latest(s3, ic_30d=0.15, rolling_n=40)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "backtest_vs_live_parity")
        assert m["status"] == "N/A-MISSING-INPUT"
        assert "manifest.json absent" in m["status_reason"]
        assert m["value"] is None

    def test_missing_latest_is_na(self, s3):
        self._put_manifest(s3, mean_ic=0.18)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "backtest_vs_live_parity")
        assert m["status"] == "N/A-MISSING-INPUT"
        assert "latest.json absent" in m["status_reason"]

    def test_both_absent_is_na(self, s3):
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "backtest_vs_live_parity")
        assert m["status"] == "N/A-MISSING-INPUT"
        assert "manifest.json absent" in m["status_reason"]
        assert "latest.json absent" in m["status_reason"]

    def test_null_ic_30d_below_producer_floor_is_na(self, s3):
        # Producer leaves ic_30d null when <10 valid pairs in the trailing 30d
        # (crucible-backtester pipeline_common.py _MIN_IC_SAMPLES) — must NOT
        # silently default to 0.0 or any placeholder (fail-loud doctrine).
        self._put_manifest(s3, mean_ic=0.18)
        self._put_latest(s3, ic_30d=None, rolling_n=6)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "backtest_vs_live_parity")
        assert m["status"] == "N/A-MISSING-INPUT"
        assert "ic_30d null" in m["status_reason"]
        assert m["value"] is None

    def test_cpcv_not_ok_is_na(self, s3):
        self._put_manifest(s3, mean_ic=0.18, status="insufficient_data")
        self._put_latest(s3, ic_30d=0.15, rolling_n=40)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "backtest_vs_live_parity")
        assert m["status"] == "N/A-MISSING-INPUT"
        assert "meta_model_oos_ic_cpcv" in m["status_reason"]

    def test_source_path_names_both_artifacts(self, s3):
        self._put_manifest(s3, mean_ic=0.18)
        self._put_latest(s3, ic_30d=0.15, rolling_n=40)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "backtest_vs_live_parity")
        assert MANIFEST_KEY in m["source_path"]
        assert LATEST_KEY in m["source_path"]

    # ── alpha-engine-config-I12123: the CPCV leg is pinned to the served model ──

    def test_challenger_cpcv_in_standing_key_is_ignored(self, s3):
        """The 2026-09-04 RED graded against a NON-promoted challenger's 0.082
        sitting in the standing key while the 08-14 champion was served. With a
        challenger value in the standing key and the champion's frozen bundle
        present, the tile must use the champion's value."""
        _put_served(s3, 0.131924, standing_mean_ic=0.082003)
        self._put_latest(s3, ic_30d=0.10, rolling_n=40)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "backtest_vs_live_parity")
        assert m["value"] == pytest.approx(0.10 - 0.131924)

    def test_card_names_served_version_registry_key_and_etag(self, s3):
        _put_served(s3, 0.131924)
        self._put_latest(s3, ic_30d=0.10, rolling_n=40)
        etag = s3.head_object(Bucket=BUCKET, Key=REGISTRY_KEY)["ETag"].strip('"')
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "backtest_vs_live_parity")
        for needle in (SERVED, REGISTRY_KEY, etag, "0.131924"):
            assert needle in m["status_reason"], needle
        assert REGISTRY_KEY in m["source_path"] and etag in m["source_path"]

    def test_missing_served_version_is_na_not_standing_key(self, s3):
        _put(s3, MANIFEST_KEY, {"meta_model_oos_ic_cpcv": {"status": "ok", "mean_ic": 0.18, "n_combos": 12}})
        self._put_latest(s3, ic_30d=0.15, rolling_n=40)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "backtest_vs_live_parity")
        assert m["status"] == "N/A-MISSING-INPUT"
        assert m["value"] is None
        assert "served_version absent" in m["status_reason"]

    def test_missing_registry_bundle_is_na_not_standing_key(self, s3):
        _put(s3, MANIFEST_KEY, {"served_version": SERVED,
                                "meta_model_oos_ic_cpcv": {"status": "ok", "mean_ic": 0.18, "n_combos": 12}})
        self._put_latest(s3, ic_30d=0.15, rolling_n=40)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "backtest_vs_live_parity")
        assert m["status"] == "N/A-MISSING-INPUT"
        assert m["value"] is None
        assert "frozen registry manifest" in m["status_reason"]
        assert REGISTRY_KEY in m["status_reason"]

    def test_history_threads_trend(self, s3):
        # Mirrors predictor.py's meta_l2_ic trend-threading (config#1836).
        self._put_manifest(s3, mean_ic=0.18)
        self._put_latest(s3, ic_30d=0.15, rolling_n=40)
        history = CardHistory({("backtester", "backtest_vs_live_parity"): [-0.01, -0.02]}, 2)
        tile = build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3, history=history)
        m = _comp(tile, "backtest_vs_live_parity")
        assert m["trend_4w"] == [-0.01, -0.02, pytest.approx(-0.03)]

    def test_no_history_keeps_default_trends(self, s3):
        self._put_manifest(s3, mean_ic=0.18)
        self._put_latest(s3, ic_30d=0.15, rolling_n=40)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "backtest_vs_live_parity")
        assert m["trend_4w"] is None
        assert m["trend_decoration"] == "→"


class TestBacktestVsLiveParityEffectiveN:
    """alpha-engine-config-I10065 item (a): grade on ``realized_ic.n_effective``;
    below 2 the component reads insufficient-N, never GREEN or RED.

    The 2026-10-23 card graded this critical RED at N=689 (``rolling_n``) while
    the artifact it read declared ``n_effective: 1, se_effective: 1.0``. These
    tests fail on that code: it handed ``rolling_n`` to the floor check."""

    def _latest(self, s3, ic_30d, **block):
        _put(s3, LATEST_KEY, {"ic_30d": ic_30d, "rolling_n": 689, "forward_days": 21,
                              **({"realized_ic": block} if block else {})})

    def test_live_2026_10_24_shape_is_insufficient_n_not_red(self, s3):
        # Values from predictor/metrics/latest.json @ e6lETbzQR3Uj819Xwqh3pBqsaGijRF5g
        # and the served model's frozen CPCV (the 10-23 card's RED inputs).
        _put_served(s3, 0.131924)
        self._latest(s3, 0.0657, value=0.0657, horizon_days=21, n_rows=689,
                     n_prediction_dates=22, n_effective=1, se_effective=1.0)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "backtest_vs_live_parity")
        assert m["status"] == "N/A-LOW-N"
        assert m["n_samples"] == 1
        # The level is still published, with the rows beside it — only the grade is withheld.
        assert m["value"] == pytest.approx(0.0657 - 0.131924)
        assert "N_eff=1" in m["status_reason"] and "rows=689" in m["status_reason"]

    def test_effective_n_of_one_cannot_read_green_either(self, s3):
        _put_served(s3, 0.10)
        self._latest(s3, 0.20, horizon_days=21, n_effective=1)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "backtest_vs_live_parity")
        assert m["status"] == "N/A-LOW-N"

    def test_effective_n_of_two_grades_red_on_the_estimate(self, s3):
        _put_served(s3, 0.18)
        self._latest(s3, 0.08, horizon_days=21, n_effective=2)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "backtest_vs_live_parity")
        assert m["status"] == "RED"

    def test_thin_effective_n_withholds_green(self, s3):
        # 2 <= N_eff < 4: below the floor, so derive_status will not claim GREEN.
        _put_served(s3, 0.10)
        self._latest(s3, 0.20, horizon_days=21, n_effective=3)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "backtest_vs_live_parity")
        assert m["status"] == "WATCH"

    def test_effective_n_derived_from_dates_with_the_producers_formula(self, s3):
        # n_effective absent, n_prediction_dates declared: 22 // 21 = 1.
        _put_served(s3, 0.10)
        self._latest(s3, 0.20, horizon_days=21, n_prediction_dates=22)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "backtest_vs_live_parity")
        assert m["n_samples"] == 1
        assert m["status"] == "N/A-LOW-N"

    def test_undeclared_effective_n_never_falls_back_to_rows(self, s3):
        _put_served(s3, 0.18)
        self._latest(s3, 0.08)  # no realized_ic block at all
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "backtest_vs_live_parity")
        assert m["status"] == "N/A-LOW-N"
        assert m["n_samples"] is None
        assert "EFFECTIVE N UNDECLARED" in m["status_reason"]


class TestBacktestVsLiveParitySourceVerification:
    """Source-verification style (crucible-evaluator#100/#106 trust-battery
    harness, tests/test_source_verification.py): re-derive the graded value
    independently of the tile's own arithmetic and assert exact agreement,
    rather than asserting against a hand-copied constant."""

    def test_drift_matches_independent_subtraction(self, s3):
        live_ic, cpcv_ic = 0.1234567891, 0.0987654321
        self._put_manifest_and_latest(s3, cpcv_ic, live_ic)
        m = _comp(build_backtester_tile(BUCKET, RUN_DATE, s3_client=s3), "backtest_vs_live_parity")
        # Independent re-derivation: plain subtraction, no tile helper reuse.
        expected = live_ic - cpcv_ic
        assert m["value"] == pytest.approx(expected, abs=1e-9)

    def _put_manifest_and_latest(self, s3, cpcv_ic, live_ic):
        _put_served(s3, cpcv_ic)
        _put(s3, LATEST_KEY, {"ic_30d": live_ic, "rolling_n": 40,
                              "realized_ic": {"horizon_days": 21, "n_effective": 8}})
