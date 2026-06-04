"""Tests for grading/metric_record.py — the MetricRecord factory."""

from grading.metric_record import build_metric


class TestBuildMetricStatus:
    def test_green_above_target(self):
        m = build_metric(
            name="sharpe_ratio", module="portfolio_outcome", metric_type="sharpe",
            value=1.4, n_samples=120, n_floor=60, target=1.0, red_line=0.0,
            ci_low=0.8, ci_high=2.0, ci_method="bootstrap", criticality="critical",
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
            source_path="s3://b/x", criticality="critical",
        )
        assert m.status == "GREEN"

    def test_drawdown_breaches_redline_red(self):
        m = build_metric(
            name="max_drawdown", module="portfolio_outcome", metric_type="ratio",
            value=-0.30, n_samples=80, n_floor=2, target=-0.15, red_line=-0.25,
            source_path="s3://b/x", criticality="critical",
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
