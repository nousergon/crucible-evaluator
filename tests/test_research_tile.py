"""Tests for grading/tiles/research.py — Tile 1, precision-from-e2e sourcing."""

import json

import boto3
import pytest
from moto import mock_aws

from grading.tiles.research import build_research_tile

BUCKET = "alpha-engine-research"
RUN_DATE = "2026-06-07"


def _clf(precision, tp, fp, fn=0, tn=0):
    return {"precision": precision, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": tp + fp + fn + tn}


# Realistic classification blocks (with fn/tn) so the base-rate edge is
# meaningful: edge = precision − base_rate, base_rate = (tp+fn)/n_pop.
_E2E = {
    "status": "ok",
    # precision 0.50, base 200/5000=0.04 → edge +0.46
    "scanner_lift": {"lift": -0.0016, "n_passing": 320,
                     "classification": _clf(0.50, 60, 60, fn=140, tn=4740)},
    "team_lift": [
        {"team_id": "tech", "lift": 0.01, "classification": _clf(0.6, 18, 12, fn=12, tn=58)},
        {"team_id": "health", "lift": 0.0, "classification": _clf(0.5, 10, 10, fn=10, tn=70)},
    ],
    # precision 0.55, base 58/200=0.29 → edge +0.26
    "cio_lift": {"lift": -0.0067, "n_advance": 69,
                 "classification": _clf(0.55, 38, 31, fn=20, tn=111)},
    "cio_vs_ranking": {"lift": 0.0003},
}


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _put(s3, name, data):
    s3.put_object(Bucket=BUCKET, Key=f"backtest/{RUN_DATE}/{name}", Body=json.dumps(data).encode())


def _comp(tile, name):
    return next(c for c in tile["components"] if c["name"] == name)


class TestMissingArtifacts:
    def test_all_missing_inputs_loud(self, s3):
        tile = build_research_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert tile["module"] == "research"
        scanner = _comp(tile, "scanner")
        assert scanner["status"] == "N/A-MISSING-INPUT"
        # critical components N/A → tile WATCH (never false GREEN).
        assert tile["status"] == "WATCH"


class TestPrecisionComponents:
    def test_scanner_edge_over_base_rate(self, s3):
        _put(s3, "e2e_lift.json", _E2E)
        tile = build_research_tile(BUCKET, RUN_DATE, s3_client=s3)
        scanner = _comp(tile, "scanner")
        # edge = precision 0.50 − base_rate (200/5000=0.04) = +0.46
        assert scanner["value"] == pytest.approx(0.50 - 0.04)
        assert scanner["ci_method"] == "wilson"
        assert scanner["n_samples"] == 120  # tp+fp (selected)
        assert "base-rate" in scanner["status_reason"]
        assert "return-lift" in scanner["status_reason"]

    def test_sector_teams_pooled_edge(self, s3):
        _put(s3, "e2e_lift.json", _E2E)
        tile = build_research_tile(BUCKET, RUN_DATE, s3_client=s3)
        teams = _comp(tile, "sector_teams_avg")
        # pooled tp=28 fp=22 fn=22 tn=128 → precision 28/50=0.56,
        # base_rate (28+22)/200=0.25 → edge +0.31
        assert teams["value"] == pytest.approx(28 / 50 - 50 / 200)
        assert teams["n_samples"] == 50
        assert teams["criticality"] == "critical"

    def test_cio_edge(self, s3):
        _put(s3, "e2e_lift.json", _E2E)
        tile = build_research_tile(BUCKET, RUN_DATE, s3_client=s3)
        cio = _comp(tile, "cio")
        # precision 0.55 − base_rate (58/200=0.29) = +0.26
        assert cio["value"] == pytest.approx(0.55 - 0.29)
        assert "vs-ranking-lift" in cio["status_reason"]


class TestCioSelectionSkill:
    """L4561 — the CIO selection-skill instrument, graded under the L4562
    reliability contract (insignificant gap → WATCH + reliability=low)."""

    def _e2e_with_sel(self, gap, p, **extra):
        e = json.loads(json.dumps(_E2E))
        e["cio_lift"]["selection_skill_21d"] = {
            "selection_gap_21d": gap, "selection_gap_p": p,
            "advance_alpha_21d": -0.05, "reject_alpha_21d": -0.05 - gap,
            "n_advance": 48, "n_reject": 40, "conviction_ic_21d": 0.05, **extra}
        return e

    def test_insignificant_gap_watch_low_reliability(self, s3):
        # The live case: negative gap but p=0.12 → WATCH, NOT a confident RED.
        _put(s3, "e2e_lift.json", self._e2e_with_sel(-0.039, 0.123))
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "cio_selection_skill")
        assert m["status"] == "WATCH"
        assert m["reliability"] == "low"
        assert m["value"] == pytest.approx(-0.039)

    def test_significant_negative_gap_red(self, s3):
        # If the anti-selection were significant, it WOULD grade RED.
        _put(s3, "e2e_lift.json", self._e2e_with_sel(-0.04, 0.01))
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "cio_selection_skill")
        assert m["status"] == "RED"
        assert m["reliability"] == "high"

    def test_significant_positive_gap_green(self, s3):
        _put(s3, "e2e_lift.json", self._e2e_with_sel(0.03, 0.01))
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "cio_selection_skill")
        assert m["status"] == "GREEN"

    def test_absent_missing_input(self, s3):
        _put(s3, "e2e_lift.json", _E2E)  # no selection_skill_21d block
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "cio_selection_skill")
        assert m["status"] == "N/A-MISSING-INPUT"


class TestResearchCompositeIC:
    """L4561 — does the blended research score predict 21d alpha? Graded under
    the L4562 contract (insignificant → WATCH + reliability=low)."""

    def _e2e_with_attr(self, fic, p, *, date_ic=None, date_p=None, n_dates=None):
        e = json.loads(json.dumps(_E2E))
        attr = {
            "n": 89, "final_score_ic": fic, "final_score_ic_p": p,
            "combined_score_ic": -0.016, "macro_shift_ic": -0.044, "cio_conviction_ic": 0.013}
        if date_ic is not None:
            attr.update({
                "n_eval_dates": n_dates, "final_score_date_ic": date_ic,
                "final_score_date_ic_p": date_p,
                "combined_score_date_ic": -0.024, "macro_shift_date_ic": -0.185,
                "cio_conviction_date_ic": 0.067})
        e["cio_lift"]["layer_attribution_21d"] = attr
        return e

    def test_insignificant_ic_watch_low(self, s3):
        # Pre-config#1164 artifact (no date-clustered fields) → pooled fallback.
        _put(s3, "e2e_lift.json", self._e2e_with_attr(-0.045, 0.67))
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "research_composite_ic")
        assert m["status"] == "WATCH"
        assert m["reliability"] == "low"
        assert "per-layer" in m["status_reason"]

    def test_significant_positive_ic_green(self, s3):
        _put(s3, "e2e_lift.json", self._e2e_with_attr(0.08, 0.01))
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "research_composite_ic")
        assert m["status"] == "GREEN"

    def test_date_clustered_dissolves_pseudo_replicated_flag(self, s3):
        # config#1164 — the LIVE case: pooled p=0.02 (N=145 names) looks significant
        # negative, but the date-clustered estimator (N=9 weeks) is p=0.18 → the
        # honest read is WATCH/underpowered, NOT a high-reliability negative.
        _put(s3, "e2e_lift.json",
             self._e2e_with_attr(-0.193, 0.02, date_ic=-0.147, date_p=0.18, n_dates=9))
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "research_composite_ic")
        assert m["status"] == "WATCH"
        assert m["reliability"] == "low"
        assert m["value"] == -0.147          # graded on the date-clustered IC
        assert m["n_samples"] == 9            # effective N = weeks, not 145 names
        assert m["estimator"] == "date_clustered_rank_ic_vs_21d_alpha"
        assert "pseudo-replication" in m["status_reason"]

    def test_date_clustered_significant_positive_green(self, s3):
        _put(s3, "e2e_lift.json",
             self._e2e_with_attr(0.05, 0.30, date_ic=0.09, date_p=0.02, n_dates=12))
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "research_composite_ic")
        assert m["status"] == "GREEN"
        assert m["value"] == 0.09
        assert m["n_samples"] == 12

    def test_absent_missing_input(self, s3):
        _put(s3, "e2e_lift.json", _E2E)
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "research_composite_ic")
        assert m["status"] == "N/A-MISSING-INPUT"


class TestHorizonPreference:
    """Selectors grade on the canonical 21d block when present (ROADMAP L4551),
    falling back to legacy 5d for older artifacts."""

    def _e2e_with_21d(self):
        e = json.loads(json.dumps(_E2E))  # deep copy
        # 21d precision DIFFERS from 5d so we can prove which one was graded.
        e["scanner_lift"]["classification_21d"] = _clf(0.80, 96, 24, fn=104, tn=4776)
        e["scanner_lift"]["lift_21d_log"] = {"lift": 0.012, "selected_avg": 0.03, "baseline_avg": 0.018}
        e["cio_lift"]["classification_21d"] = _clf(0.75, 52, 17, fn=20, tn=111)
        e["cio_lift"]["lift_21d_log"] = {"lift": 0.02}
        for t in e["team_lift"]:
            t["classification_21d"] = _clf(0.9, 27, 3, fn=3, tn=67)
        return e

    def test_scanner_grades_21d_when_present(self, s3):
        _put(s3, "e2e_lift.json", self._e2e_with_21d())
        tile = build_research_tile(BUCKET, RUN_DATE, s3_client=s3)
        scanner = _comp(tile, "scanner")
        # 21d precision 0.80, base (96+104)/5000=0.04 → edge +0.76 (not the 5d +0.46)
        assert scanner["value"] == pytest.approx(0.80 - 0.04)
        assert scanner["n_samples"] == 120  # tp+fp of the 21d block
        assert "[21d]" in scanner["status_reason"]
        assert "21d-alpha-lift" in scanner["status_reason"]

    def test_cio_grades_21d_when_present(self, s3):
        _put(s3, "e2e_lift.json", self._e2e_with_21d())
        cio = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "cio")
        assert cio["value"] == pytest.approx(0.75 - (52 + 20) / 200)
        assert "[21d]" in cio["status_reason"]

    def test_teams_pool_21d_when_present(self, s3):
        _put(s3, "e2e_lift.json", self._e2e_with_21d())
        teams = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "sector_teams_avg")
        assert "[21d]" in teams["status_reason"]
        # pooled 21d precision 54/60 = 0.90
        assert teams["n_samples"] == 60

    def test_falls_back_to_5d_without_21d(self, s3):
        # The unmodified _E2E (5d only) must still grade on 5d.
        _put(s3, "e2e_lift.json", _E2E)
        scanner = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "scanner")
        assert "[5d]" in scanner["status_reason"]
        assert scanner["value"] == pytest.approx(0.50 - 0.04)


class TestCompositeScoring:
    def test_spearman_positive_green(self, s3):
        # Significant positive rank correlation → GREEN.
        _put(s3, "e2e_lift.json", _E2E)
        _put(s3, "score_calibration.json", {
            "status": "ok", "horizon": "10d", "monotonic": False, "spearman_rho": 0.28,
            "spearman_p": 0.001, "spearman_n": 200, "calibration_assessment": "positive",
            "beat_spy_pct": 0.56,
        })
        tile = build_research_tile(BUCKET, RUN_DATE, s3_client=s3)
        cs = _comp(tile, "composite_scoring")
        assert cs["status"] == "GREEN"
        assert cs["value"] == pytest.approx(0.28)
        assert "rho=+0.280" in cs["status_reason"]

    def test_horizon_reported_honestly_not_hardcoded_21d(self, s3):
        # config#1063: the artifact is computed at 10d (score_performance has no
        # 21d column) — the tile must report the TRUE horizon, NOT assert 21d,
        # grade it SUPPORTING (sub-horizon proxy), and point at the canonical-21d
        # research_composite_ic.
        _put(s3, "e2e_lift.json", _E2E)
        _put(s3, "score_calibration.json", {
            "status": "ok", "horizon": "10d", "monotonic": False, "spearman_rho": 0.12,
            "spearman_p": 0.02, "spearman_n": 150, "calibration_assessment": "positive",
        })
        cs = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "composite_scoring")
        assert cs["measurement_horizon"] == "10d"
        assert cs["criticality"] == "supporting"
        assert "[10d]" in cs["status_reason"]
        assert "research_composite_ic" in cs["status_reason"]

    def test_horizon_defaults_to_10d_when_artifact_omits_it(self, s3):
        # Legacy artifacts without a horizon field default to the producer's
        # actual default (10d) rather than a mislabeled 21d.
        _put(s3, "e2e_lift.json", _E2E)
        _put(s3, "score_calibration.json", {
            "status": "ok", "monotonic": False, "spearman_rho": 0.12, "spearman_p": 0.02,
            "spearman_n": 150, "calibration_assessment": "positive",
        })
        cs = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "composite_scoring")
        assert cs["measurement_horizon"] == "10d"

    def test_spearman_negative_red(self, s3):
        # Significant negative rank correlation (rho <= red-line 0.0) → RED.
        _put(s3, "e2e_lift.json", _E2E)
        _put(s3, "score_calibration.json", {
            "status": "ok", "monotonic": True, "spearman_rho": -0.22, "spearman_p": 0.004,
            "spearman_n": 200, "calibration_assessment": "negative", "beat_spy_pct": 0.42,
        })
        tile = build_research_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert _comp(tile, "composite_scoring")["status"] == "RED"

    def test_spearman_flat_watch_not_red(self, s3):
        # Insignificant (flat) calibration must NOT grade RED — the core fix:
        # a single noisy bucket no longer forces critical RED.
        _put(s3, "e2e_lift.json", _E2E)
        _put(s3, "score_calibration.json", {
            "status": "ok", "monotonic": False, "spearman_rho": -0.04, "spearman_p": 0.61,
            "spearman_n": 200, "calibration_assessment": "flat", "beat_spy_pct": 0.50,
        })
        tile = build_research_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert _comp(tile, "composite_scoring")["status"] == "WATCH"

    def test_legacy_monotonic_fallback_neutralized_to_watch(self, s3):
        # Pre-2026-06-07 artifact without spearman fields → legacy brittle binary.
        # Per the L4562 contract it must NOT drive a confident GREEN/RED: it is
        # neutralized to WATCH + reliability=low (the binary is context only).
        _put(s3, "e2e_lift.json", _E2E)
        _put(s3, "score_calibration.json", {"status": "ok", "monotonic": True, "beat_spy_pct": 0.56, "n": 200})
        tile = build_research_tile(BUCKET, RUN_DATE, s3_client=s3)
        cs = _comp(tile, "composite_scoring")
        assert cs["status"] == "WATCH"
        assert cs["reliability"] == "low"
        assert cs["estimator"] == "legacy_monotonic_binary_deprecated"
        assert "DEPRECATED" in cs["status_reason"]

    def test_absent_missing_input(self, s3):
        _put(s3, "e2e_lift.json", _E2E)
        tile = build_research_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert _comp(tile, "composite_scoring")["status"] == "N/A-MISSING-INPUT"


class TestMacroAndCalibration:
    def test_macro_accuracy_lift(self, s3):
        _put(s3, "e2e_lift.json", _E2E)
        _put(s3, "macro_eval.json", {"status": "ok", "accuracy_lift": 2.0, "assessment": "helpful", "n_evaluated": 40})
        tile = build_research_tile(BUCKET, RUN_DATE, s3_client=s3)
        macro = _comp(tile, "macro_agent")
        assert macro["value"] == pytest.approx(2.0)
        assert macro["status"] == "GREEN"  # +2pp > target 0, N=40 > floor 20

    def test_calibration_ece_lower_is_better(self, s3):
        _put(s3, "e2e_lift.json", _E2E)
        _put(s3, "portfolio_calibration.json", {"status": "ok", "ece": 0.03, "n": 200})
        tile = build_research_tile(BUCKET, RUN_DATE, s3_client=s3)
        cal = _comp(tile, "calibration_diagnostics")
        assert cal["value"] == pytest.approx(0.03)
        assert cal["status"] == "GREEN"  # 0.03 < target 0.05


class TestAgentQualityComponents:
    """judge_rubric_pass_rate / pillar_emit_coverage / signal_volume_adequacy
    read backtest/{date}/agent_quality.json (config Batch A #1149)."""

    def test_missing_input_when_no_producer(self, s3):
        _put(s3, "e2e_lift.json", _E2E)
        tile = build_research_tile(BUCKET, RUN_DATE, s3_client=s3)
        for name in ("judge_rubric_pass_rate", "pillar_emit_coverage", "signal_volume_adequacy"):
            c = _comp(tile, name)
            assert c["status"] == "N/A-MISSING-INPUT", name
            assert "agent_quality.json" in c["status_reason"], name

    def test_wired_grades_from_agent_quality(self, s3):
        _put(s3, "e2e_lift.json", _E2E)
        _put(s3, "agent_quality.json", {
            "status": "ok",
            "judge_rubric_pass_rate": {"value": 0.90, "n": 60},
            "pillar_emit_coverage": {"value": 0.95, "n": 25},
            "signal_volume_adequacy": {"value": 24, "n": 24},
        })
        tile = build_research_tile(BUCKET, RUN_DATE, s3_client=s3)
        prr = _comp(tile, "judge_rubric_pass_rate")
        assert prr["value"] == 0.90 and prr["status"] == "GREEN"
        assert _comp(tile, "pillar_emit_coverage")["value"] == 0.95
        assert _comp(tile, "signal_volume_adequacy")["value"] == 24


class TestMomentumRegimeIC:
    def _e2e(self, mri):
        d = dict(_E2E)
        d["momentum_regime_ic"] = mri
        return d

    def test_present_grades_on_low_breadth_ic(self, s3):
        _put(s3, "e2e_lift.json", self._e2e({
            "status": "ok", "n_weeks": 8, "low_breadth_ic": -0.115,
            "high_breadth_ic": 0.030, "breadth_ic_corr": 0.58, "horizon": "21d",
        }))
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "momentum_regime_ic")
        assert m["value"] == pytest.approx(-0.115)
        # low_breadth_ic below the -0.05 red-line → RED (momentum actively hurts).
        assert m["status"] == "RED"
        assert "breadth" in (m.get("status_reason") or "").lower()

    def test_present_neutral_low_breadth_not_red(self, s3):
        _put(s3, "e2e_lift.json", self._e2e({
            "status": "ok", "n_weeks": 8, "low_breadth_ic": 0.01,
            "high_breadth_ic": 0.04, "breadth_ic_corr": 0.3, "horizon": "21d",
        }))
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "momentum_regime_ic")
        assert m["status"] != "RED"

    def test_missing_block_na_missing_input(self, s3):
        _put(s3, "e2e_lift.json", _E2E)  # no momentum_regime_ic key
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "momentum_regime_ic")
        assert m["status"] == "N/A-MISSING-INPUT"

    def test_insufficient_status_na(self, s3):
        _put(s3, "e2e_lift.json", self._e2e({"status": "insufficient_data", "n_weeks": 2}))
        m = _comp(build_research_tile(BUCKET, RUN_DATE, s3_client=s3), "momentum_regime_ic")
        assert m["status"] == "N/A-MISSING-INPUT"
