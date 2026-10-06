"""Statistical power beside every risk-adjusted metric (I8188 deliverable 8)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from krepis.metrics import MetricRecord, derive_status

from grading.power import (
    annotate_power,
    annotate_power_all,
    observed_half_width,
    required_n,
    target_inside_ci,
)


def _rec(**over) -> MetricRecord:
    kw = dict(
        name="sharpe_ratio", module="portfolio", metric_type="sharpe",
        value=0.204, unit="annualized_ratio", ci_low=-2.772, ci_high=2.948,
        ci_method="bootstrap", n_samples=119, n_floor=60,
        target=1.0, red_line=0.0, source_path="s3://x/y",
        last_updated_utc=datetime.now(UTC),
    )
    kw.update(over)
    kw["status"] = kw.get("status") or derive_status(
        value=kw["value"], n_samples=kw["n_samples"], n_floor=kw["n_floor"],
        target=kw["target"], red_line=kw["red_line"],
        ci_low=kw["ci_low"], ci_high=kw["ci_high"],
    )
    kw.setdefault("status_reason", "x")
    return MetricRecord(**kw)


class TestRequiredN:
    def test_scales_with_the_square_of_the_width_ratio(self):
        # h0 = 2.86, decision precision h = |1.0-0.0|/2 = 0.5 ⇒ (5.72)^2 × 119
        assert required_n(
            ci_low=-2.772, ci_high=2.948, n_samples=119, target=1.0, red_line=0.0,
        ) == 3894

    def test_matches_the_hand_figure_in_the_issue(self):
        """I8188 computed ~3,875 sessions by hand for CI [-2.36, +3.47]."""
        n = required_n(ci_low=-2.36, ci_high=3.47, n_samples=119, target=1.0, red_line=0.0)
        assert 3_700 <= n <= 4_200

    def test_already_precise_enough_returns_the_current_n(self):
        assert required_n(
            ci_low=0.9, ci_high=1.1, n_samples=500, target=1.0, red_line=0.0,
        ) == 500

    @pytest.mark.parametrize("kw", [
        {"ci_low": None, "ci_high": 1.0},
        {"ci_high": None},
        {"n_samples": 0},
        {"n_samples": None},
        {"target": None},
        {"red_line": None},
        {"target": 1.0, "red_line": 1.0},        # no gap to resolve
        {"ci_low": 2.0, "ci_high": 1.0},         # inverted
        {"ci_low": float("nan"), "ci_high": 1.0},
    ])
    def test_absent_rather_than_zero_when_uncomputable(self, kw):
        base = dict(ci_low=-1.0, ci_high=1.0, n_samples=100, target=1.0, red_line=0.0)
        base.update(kw)
        assert required_n(**base) is None, "0 would read as 'no more data needed'"

    def test_half_width(self):
        assert observed_half_width(-1.0, 3.0) == 2.0
        assert observed_half_width(None, 3.0) is None
        assert observed_half_width(3.0, -1.0) is None


class TestTargetInsideCi:
    def test_inside(self):
        assert target_inside_ci(target=1.0, ci_low=-2.7, ci_high=2.9) is True

    def test_outside(self):
        assert target_inside_ci(target=1.0, ci_low=-5.7, ci_high=0.24) is False

    def test_absent_ci_is_not_inside(self):
        assert target_inside_ci(target=1.0, ci_low=None, ci_high=2.9) is False


class TestSuppression:
    def test_the_live_sharpe_red_is_produced_by_the_ci_width(self):
        """THE DEFECT: value 0.204 is ABOVE the 0.0 red line; the RED comes
        entirely from ci_low = -2.772."""
        rec = _rec()
        assert rec.status == "RED"
        assert rec.value > rec.red_line

    def test_it_is_downgraded_to_watch_and_says_why(self):
        rec = annotate_power(_rec())
        assert rec.status == "WATCH"
        assert rec.status_before_power == "RED"
        assert rec.n_required == 3894
        assert rec.target_inside_ci is True
        assert "Power-limited" in rec.status_reason
        assert "3,894" in rec.status_reason
        assert "15.5y" in rec.status_reason or "~15" in rec.status_reason

    def test_a_red_earned_by_the_point_estimate_is_never_suppressed(self):
        """Live sortino: value 0.313 is BELOW its 0.5 red line, and its CI also
        contains the target. The estimate earned the RED — it stands."""
        rec = annotate_power(_rec(
            name="sortino_ratio", value=0.313, ci_low=-3.538, ci_high=5.552,
            target=1.5, red_line=0.5,
        ))
        assert rec.status == "RED"
        assert not hasattr(rec, "status_before_power") or rec.status_before_power is None
        assert rec.target_inside_ci is True  # still published

    def test_a_red_whose_ci_excludes_the_target_is_never_suppressed(self):
        """Live information_ratio: CI [-5.69, +0.24] does not reach target 0.5,
        so the data DOES rule the target out. RED stands."""
        rec = annotate_power(_rec(
            name="information_ratio", value=-2.651, ci_low=-5.686, ci_high=0.241,
            target=0.5, red_line=0.0,
        ))
        assert rec.status == "RED"
        assert rec.target_inside_ci is False

    def test_it_never_upgrades_a_watch_or_touches_a_green(self):
        for status in ("WATCH", "GREEN", "N/A-LOW-N"):
            rec = annotate_power(_rec(status=status))
            assert rec.status == status

    def test_lower_is_better_metrics_use_the_correct_bad_side(self):
        """max_drawdown: target -0.15, red_line -0.25, lower-is-better, so the
        bad-side bound is ci_HIGH. A CI-driven RED there is suppressible too."""
        rec = annotate_power(_rec(
            name="max_drawdown", metric_type="ratio", value=-0.20,
            unit="fraction", ci_low=-0.40, ci_high=-0.05,
            target=-0.15, red_line=-0.25,
        ))
        # value -0.20 is not at/beyond -0.25 for lower-is-better; ci_low -0.40 is.
        assert rec.status_before_power == "RED"
        assert rec.status == "WATCH"

    def test_every_record_carries_the_power_fields_even_when_untouched(self):
        rec = annotate_power(_rec(status="GREEN"))
        assert hasattr(rec, "n_required")
        assert hasattr(rec, "target_inside_ci")

    def test_annotate_all_is_in_place(self):
        recs = [_rec(), _rec(name="sortino_ratio", value=0.313, target=1.5, red_line=0.5)]
        out = annotate_power_all(recs)
        assert out is recs
        assert recs[0].status == "WATCH"


# ── alpha-engine-config-I11169 / -I11988: the letter follows the FINAL status ──
#
# The 2026-10-02 card carried power-downgraded WATCH rows still lettered "F":
# build_metric stamps derived_letter from the status a record is born with, and
# the downgrade moved the status without moving the letter. These cases build
# records the way production does (build_metric, so the letter is stamped at
# birth) and pin the letter against the status at every surface that emits one.

from krepis.metrics import derive_letter  # noqa: E402

from grading.metric_record import build_metric  # noqa: E402
from grading.module_agg import build_tile  # noqa: E402
from grading.power import (  # noqa: E402
    DISPOSITION_DOWNGRADED,
    DISPOSITION_RED_EARNED_BY_ESTIMATE,
)


def _built(**over):
    kw = dict(
        name="sharpe_ratio", module="portfolio_outcome", metric_type="sharpe",
        criticality="critical", estimator="sharpe_with_bootstrap_ci",
        measurement_horizon="since_inception", value=0.204,
        unit="annualized_ratio", n_samples=119, n_floor=60, target=1.0,
        red_line=0.0, ci_low=-2.772, ci_high=2.948, ci_method="bootstrap",
        source_path="s3://x/trades/eod_pnl.csv",
    )
    kw.update(over)
    return build_metric(**kw)


#: information_ratio exactly as the frozen 2026-10-02 card read it (I11988
#: audit comment on I11169): value -1.793, CI [-4.421, +0.8849], N 144,
#: target 0.5, red-line 0.0, target_inside_ci true.
_OCT2_IR = dict(
    name="information_ratio", metric_type="ratio",
    estimator="info_ratio_bootstrap_ci", value=-1.7929880167239172,
    n_samples=144, target=0.5, red_line=0.0, ci_low=-4.421, ci_high=0.8849,
)


class TestLetterFollowsFinalStatus:
    def test_a_born_red_record_is_lettered_f(self):
        """Precondition: the record really is born RED/F, so the next test is
        exercising the downgrade, not a record that was never F."""
        rec = _built()
        assert (rec.status, rec.derived_letter) == ("RED", "F")

    def test_a_power_downgraded_watch_is_never_lettered_f(self):
        """THE DEFECT: status WATCH with derived_letter F."""
        rec = annotate_power(_built())
        assert rec.status == "WATCH"
        assert rec.status_before_power == "RED"
        assert rec.derived_letter == derive_letter("WATCH") == "C"
        assert rec.power_disposition == DISPOSITION_DOWNGRADED

    def test_the_tile_serialises_every_letter_from_the_final_status(self):
        """The chokepoint: whatever moved a status after construction, the
        component the tile publishes carries the letter of the status the
        rollup graded."""
        stale = _built()
        stale.status = "WATCH"            # a mutation that forgot the letter
        assert stale.derived_letter == "F"
        tile = build_tile("portfolio_outcome", [stale], roster=[stale.name])
        (comp,) = tile["components"]
        assert comp["status"] == "WATCH"
        assert comp["derived_letter"] == "C"

    def test_no_published_component_letter_disagrees_with_its_status(self):
        recs = annotate_power_all([
            _built(),                                       # downgraded
            _built(**_OCT2_IR),                             # retained RED
            _built(name="psr", metric_type="pct",           # untouched WATCH
                   estimator="probabilistic_sharpe", value=0.646,
                   unit="probability", target=0.95, red_line=0.5,
                   ci_low=None, ci_high=None, n_samples=144),
        ])
        tile = build_tile("portfolio_outcome", recs, roster=[r.name for r in recs])
        for comp in tile["components"]:
            assert comp["derived_letter"] == derive_letter(comp["status"]), comp["name"]
        assert tile["letter"] == derive_letter(tile["status"])


class TestPointEstimateRedIsPublishedAsSuch:
    """I11169's premise: information_ratio's CI contains BOTH target and
    red-line yet it stays RED while Sharpe is downgraded. The rule (I8188)
    downgrades only a RED produced by the CI's bad-side bound; the Oct 2 IR's
    point estimate is itself below its red-line, so it is RED by the estimate.
    These pin that the RED stands AND that the record says why."""

    def test_oct2_information_ratio_stays_red_and_lettered_f(self):
        rec = annotate_power(_built(**_OCT2_IR))
        assert rec.target_inside_ci is True
        assert rec.status == "RED"
        assert rec.derived_letter == "F"
        assert getattr(rec, "status_before_power", None) is None

    def test_it_names_the_branch_it_took(self):
        rec = annotate_power(_built(**_OCT2_IR))
        assert rec.power_disposition == DISPOSITION_RED_EARNED_BY_ESTIMATE
        assert "earned by the estimate" in rec.status_reason
        assert "-1.79299" in rec.status_reason

    def test_the_note_is_not_appended_twice(self):
        rec = annotate_power(annotate_power(_built(**_OCT2_IR)))
        assert rec.status_reason.count("earned by the estimate") == 1

    def test_untouched_records_carry_no_disposition(self):
        rec = annotate_power(_built(value=1.2, ci_low=0.4, ci_high=2.0))
        assert rec.status == "GREEN"
        assert rec.power_disposition is None
