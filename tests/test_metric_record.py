"""Tests for grading/metric_record.py — the MetricRecord factory."""

import pytest

from grading.metric_record import MetricContractError, build_metric


class TestMetricReliabilityContract:
    """L4562 / ARCHITECTURE §18 — a value-bearing critical metric must declare a
    robust estimator; the proven-bad classes are rejected at construction so a
    brittle metric never reaches the Director."""

    def _mk(self, **kw):
        base = dict(name="x", module="m", metric_type="ratio", n_floor=10, value=1.0,
                    n_samples=50, target=0.5, red_line=0.0, source_path="s3://b/x",
                    criticality="critical")
        base.update(kw)
        return build_metric(**base)

    def test_value_bearing_critical_requires_estimator(self):
        with pytest.raises(MetricContractError, match="must declare an estimator"):
            self._mk()  # no estimator

    def test_forbidden_estimators_rejected(self):
        for bad in ("strict_binary", "sub_horizon_proxy", "unbounded_ratio_mean"):
            with pytest.raises(MetricContractError, match="forbidden estimator"):
                self._mk(estimator=bad)

    def test_robust_estimator_accepted_and_defaults_high_reliability(self):
        m = self._mk(estimator="spearman_calibration", measurement_horizon="21d")
        assert m.estimator == "spearman_calibration"
        assert m.measurement_horizon == "21d"
        assert m.reliability == "high"  # auto-defaulted for a declared critical

    def test_explicit_low_reliability_preserved(self):
        m = self._mk(estimator="rank_ic", reliability="low")
        assert m.reliability == "low"

    def test_na_critical_is_exempt(self):
        # value None (N/A-*) carries no graded signal → no estimator required.
        m = self._mk(value=None, input_present=False, estimator=None)
        assert str(m.status).startswith("N/A")

    def test_supporting_metric_not_constrained(self):
        # Only criticals drive Director prescriptions; supporting is unconstrained.
        m = self._mk(criticality="supporting", estimator=None)
        assert m.estimator is None


class TestBuildMetricStatus:
    def test_green_above_target(self):
        m = build_metric(
            name="sharpe_ratio", module="portfolio_outcome", metric_type="sharpe",
            value=1.4, n_samples=120, n_floor=60, target=1.0, red_line=0.0,
            ci_low=0.8, ci_high=2.0, ci_method="bootstrap", criticality="critical", estimator="test_robust",
            source_path="s3://b/trades/eod_pnl.csv",
        )
        assert m.status == "GREEN"
        assert m.derived_letter == "A"
        assert "sharpe_ratio" in m.status_reason
        assert "1.4" in m.status_reason

    def test_watch_below_target_above_redline(self):
        m = build_metric(
            name="information_ratio", module="portfolio_outcome", metric_type="ratio",
            value=0.3, n_samples=120, n_floor=60, target=0.5, red_line=0.0,
            source_path="s3://b/x",
        )
        assert m.status == "WATCH"
        assert m.derived_letter == "C"

    def test_red_at_or_below_redline(self):
        m = build_metric(
            name="information_ratio", module="portfolio_outcome", metric_type="ratio",
            value=-0.1, n_samples=120, n_floor=60, target=0.5, red_line=0.0,
            source_path="s3://b/x",
        )
        assert m.status == "RED"
        assert m.derived_letter == "F"

    def test_lower_is_better_drawdown_green(self):
        # max drawdown: -0.08 is better than target -0.15 (numeric ordering
        # target>=red_line so higher-is-better holds: -0.08 >= -0.15 → GREEN).
        m = build_metric(
            name="max_drawdown", module="portfolio_outcome", metric_type="ratio",
            value=-0.08, n_samples=80, n_floor=2, target=-0.15, red_line=-0.25,
            source_path="s3://b/x", criticality="critical", estimator="test_robust",
        )
        assert m.status == "GREEN"

    def test_drawdown_breaches_redline_red(self):
        m = build_metric(
            name="max_drawdown", module="portfolio_outcome", metric_type="ratio",
            value=-0.30, n_samples=80, n_floor=2, target=-0.15, red_line=-0.25,
            source_path="s3://b/x", criticality="critical", estimator="test_robust",
        )
        assert m.status == "RED"


class TestBuildMetricNA:
    def test_not_impl(self):
        m = build_metric(
            name="dsr", module="portfolio_outcome", metric_type="pct", n_floor=60,
            target=0.95, red_line=0.50, source_path="s3://b/x", implemented=False,
            na_detail="dsr: needs a trial count (L4469 W1.3b).",
        )
        assert m.status == "N/A-NOT-IMPL"
        assert m.derived_letter == "N/A"
        assert "L4469" in m.status_reason
        assert m.is_na

    def test_missing_input(self):
        m = build_metric(
            name="regime_weighted_alpha", module="portfolio_outcome", metric_type="log_return",
            n_floor=30, target=0.0, red_line=0.0, source_path="s3://b/x", input_present=False,
            na_detail="regime tags absent.",
        )
        assert m.status == "N/A-MISSING-INPUT"
        assert "regime tags absent" in m.status_reason

    def test_low_n(self):
        m = build_metric(
            name="sharpe_ratio", module="portfolio_outcome", metric_type="sharpe",
            value=1.5, n_samples=10, n_floor=60, target=1.0, red_line=0.0,
            source_path="s3://b/x",
        )
        assert m.status == "N/A-LOW-N"
        assert "below 0.5×floor" in m.status_reason

    def test_watch_between_half_floor_and_floor(self):
        # N between 30 and 60 → WATCH regardless of value (Principle 6).
        m = build_metric(
            name="sharpe_ratio", module="portfolio_outcome", metric_type="sharpe",
            value=2.0, n_samples=45, n_floor=60, target=1.0, red_line=0.0,
            source_path="s3://b/x",
        )
        assert m.status == "WATCH"


class TestPermanentNA:
    """config#1153 (operator Option A 2026-07-11) — accepted-permanent honest-N/A:
    a metric deliberately NOT built, distinguished from a transient not-impl gap."""

    def _mk(self, **kw):
        base = dict(
            name="iam_drift", module="substrate", metric_type="pct", n_floor=1,
            source_path="s3://b/", criticality="diagnostic",
            permanent_na_reason="iam_drift: not building the CFN detect-drift producer (config#1153 Option A).",
        )
        base.update(kw)
        return build_metric(**base)

    def test_renders_as_na_not_impl(self):
        m = self._mk()
        assert m.status == "N/A-NOT-IMPL"
        assert m.derived_letter == "N/A"
        assert m.is_na

    def test_reason_prefixed_accepted_permanent(self):
        m = self._mk()
        assert m.status_reason.startswith("Accepted permanent N/A — ")
        assert "config#1153" in m.status_reason

    def test_carries_permanent_na_flag(self):
        m = self._mk()
        assert m.permanent_na is True
        assert "config#1153" in m.permanent_na_reason

    def test_ordinary_na_not_marked_permanent(self):
        # A transient not-impl (no permanent_na_reason) must NOT be flagged.
        m = build_metric(
            name="x", module="substrate", metric_type="pct", n_floor=1,
            source_path="s3://b/", implemented=False, na_detail="producer not yet wired.",
        )
        assert m.permanent_na is False
        assert not m.status_reason.startswith("Accepted permanent N/A")

    def test_explicit_reason_still_wins(self):
        m = self._mk(reason="custom override")
        assert m.status_reason == "custom override"
        assert m.permanent_na is True  # flag still set regardless of reason source


class TestMeasurementArm:
    """config#2318 — optional ``arm`` label distinguishing which measurement
    arm (e.g. a retired baseline vs a live/champion feed) a metric was
    computed from. MetricRecord allows extra fields (same low-risk pattern as
    estimator/reliability/permanent_na, config#1153 / L4562)."""

    def test_arm_defaults_to_none(self):
        m = build_metric(
            name="x", module="research", metric_type="pct", n_floor=10,
            source_path="s3://b/x",
        )
        assert m.arm is None

    def test_arm_passes_through(self):
        m = build_metric(
            name="scanner", module="research", metric_type="pct", n_floor=10,
            value=0.1, n_samples=50, source_path="s3://b/x",
            arm="tech_score_baseline (retired from live feed 2026-06-29)",
        )
        assert m.arm == "tech_score_baseline (retired from live feed 2026-06-29)"


class TestBuildMetricExtras:
    def test_status_override_for_band_metric(self):
        m = build_metric(
            name="beta_vs_spy", module="portfolio_outcome", metric_type="ratio",
            value=0.95, n_samples=80, n_floor=60, source_path="s3://b/x",
            criticality="diagnostic", status="GREEN", reason="beta in band.",
        )
        assert m.status == "GREEN"
        assert m.status_reason == "beta in band."

    def test_trend_decoration_up(self):
        m = build_metric(
            name="sharpe_ratio", module="portfolio_outcome", metric_type="sharpe",
            value=1.4, n_samples=120, n_floor=60, target=1.0, red_line=0.0,
            source_path="s3://b/x", trend_4w=[0.9, 1.0, 1.2, 1.4],
        )
        assert m.trend_decoration == "↑↑"

    def test_trend_decoration_down_for_lower_is_better(self):
        # For a lower-is-better metric, a rising series is degradation (↓↓).
        m = build_metric(
            name="max_dd_duration_days", module="portfolio_outcome", metric_type="duration",
            value=40.0, n_samples=80, n_floor=2, source_path="s3://b/x",
            higher_is_better=False, trend_4w=[10.0, 20.0, 30.0, 40.0],
        )
        assert m.trend_decoration == "↓↓"
