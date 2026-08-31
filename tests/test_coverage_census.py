"""``evaluator_coverage`` must measure THIS card, not a 14-leaf legacy artifact.

alpha-engine-config-I8177. The meta-metric that exists to close the
"insufficient data" cliff was computed over ``backtest/{date}/grading.json``
— the v1 grading artifact, 14 leaf components across research/predictor/
executor only. It could not see agent, substrate, behavioral,
contribution_lift, portfolio_outcome, director_quality or backtester: seven of
the ten tiles.

Measured 2026-08-22: the card rendered ``evaluator_coverage = 0.857`` (12/14,
WATCH, comfortably above the 0.8 red-line) while its own surface stood at
78/125 = 0.624, below the red-line. The ``agent`` tile was 11 N/A of 11 and
the number moved not at all.

These tests pin the census rule, the exclusions, and — most importantly — that
the denominator is DERIVED from the rendered records rather than hand-listed
(``observability-policy.md`` §2.2).
"""

from __future__ import annotations

import pytest

from grading.coverage import (
    card_component_census,
    coverage_reason,
    replace_evaluator_coverage,
)


def _c(name, status="GREEN", *, permanent_na=False):
    return {"name": name, "status": status, "permanent_na": permanent_na}


def _tiles(**tiles):
    return {k: {"components": v} for k, v in tiles.items()}


def _roster(tiles):
    """The declared roster for a SYNTHETIC card: exactly what it renders.

    Production reads ``grading/thresholds/registry.yaml`` (alpha-engine-config
    -I8193); those rules are pinned in ``test_coverage_registry_denominator``.
    These tests pin the COUNTING rules, so the roster is injected as the
    fixture's own component set — nothing unreported, nothing unregistered.
    """
    return {
        c["name"]
        for tile in tiles.values()
        for c in (tile.get("components") or [])
    }


def _census(tiles):
    return card_component_census(tiles, declared=_roster(tiles), declared_modules={})


def _replace(tiles):
    return replace_evaluator_coverage(
        tiles, declared=_roster(tiles), declared_modules={},
    )


# --------------------------------------------------------------------------
# The census rule
# --------------------------------------------------------------------------

def test_counts_every_tile_not_just_three() -> None:
    """The whole point: seven tiles used to be invisible to this metric."""
    tiles = _tiles(
        research=[_c("a"), _c("b", "N/A-MISSING-INPUT")],
        agent=[_c("c", "N/A-MISSING-INPUT"), _c("d", "N/A-MISSING-INPUT")],
        substrate=[_c("e")],
        contribution_lift=[_c("f", "N/A-NOT-IMPL")],
    )
    census = _census(tiles)
    assert census["total"] == 6
    assert census["graded"] == 2
    assert census["coverage"] == pytest.approx(2 / 6)


def test_every_na_flavour_counts_as_ungraded() -> None:
    tiles = _tiles(t=[
        _c("ok"),
        _c("a", "N/A-MISSING-INPUT"),
        _c("b", "N/A-NOT-IMPL"),
        _c("c", "N/A-LOW-N"),
    ])
    census = _census(tiles)
    assert (census["graded"], census["total"]) == (1, 4)


def test_low_n_stays_in_the_denominator() -> None:
    """A component still accumulating is not yet measured.

    Excluding it would render 'no data' as neither pass nor fail — exactly the
    failure this metric exists to catch (``principles.md`` §2.7).
    """
    tiles = _tiles(t=[_c("ok"), _c("acc", "N/A-LOW-N")])
    census = _census(tiles)
    assert census["total"] == 2
    assert "t.acc [N/A-LOW-N]" in census["ungraded"]


def test_red_and_watch_count_as_graded() -> None:
    """Coverage measures whether we MEASURED it, not whether it passed."""
    tiles = _tiles(t=[_c("a", "RED"), _c("b", "WATCH"), _c("c", "GREEN")])
    assert _census(tiles)["coverage"] == 1.0


def test_coverage_record_never_counts_itself() -> None:
    tiles = _tiles(backtester=[_c("evaluator_coverage"), _c("other")])
    assert _census(tiles)["total"] == 1


# --------------------------------------------------------------------------
# Exclusions are DECLARED on the record, never hand-listed here
# --------------------------------------------------------------------------

def test_declared_permanent_na_leaves_the_denominator() -> None:
    tiles = _tiles(t=[
        _c("live"),
        _c("retired", "N/A-NOT-IMPL", permanent_na=True),
    ])
    census = _census(tiles)
    assert census["total"] == 1
    assert census["declared_out"] == ["t.retired"]


def test_module_carries_no_component_list() -> None:
    """The exclusion set is derived from the card, never enumerated in code.

    A hand-maintained list drifts, and its drift is invisible because the
    missing rows produce no signal (``observability-policy.md`` §2.2). If a
    future edit adds a literal component name to this module, this fails.
    """
    from pathlib import Path

    import grading.coverage as mod

    source = Path(mod.__file__).read_text()
    body = source.split('"""', 2)[-1]  # skip the module docstring
    for known_component in (
        "position_sizing", "iam_drift", "alert_noise_ratio",
        "changelog_coverage", "sector_teams_avg", "judge_outcome_ic",
    ):
        assert known_component not in body, (
            f"coverage.py names {known_component!r} — the denominator must be "
            "derived from the rendered records, not hand-listed"
        )


def test_an_undeclared_na_is_never_silently_excluded() -> None:
    """Only `permanent_na` excuses a row. A bare N/A always counts against us."""
    tiles = _tiles(t=[_c("a", "N/A-NOT-IMPL", permanent_na=False)])
    census = _census(tiles)
    assert census["total"] == 1
    assert census["coverage"] == 0.0
    assert census["declared_out"] == []


# --------------------------------------------------------------------------
# Attribution — a bare fraction is how the old number went unquestioned
# --------------------------------------------------------------------------

def test_reason_names_the_worst_tile() -> None:
    tiles = _tiles(
        good=[_c("a"), _c("b")],
        agent=[_c("c", "N/A-MISSING-INPUT"), _c("d", "N/A-MISSING-INPUT")],
    )
    reason = coverage_reason(_census(tiles))
    assert "agent" in reason
    assert "0/2" in reason


def test_reason_notes_declared_exclusions() -> None:
    tiles = _tiles(t=[_c("a"), _c("r", "N/A-NOT-IMPL", permanent_na=True)])
    assert "excluded as declared-permanent-N/A" in coverage_reason(
        _census(tiles)
    )


def test_census_lists_which_components_are_ungraded() -> None:
    tiles = _tiles(agent=[_c("cost_per_signal", "N/A-MISSING-INPUT")])
    census = _census(tiles)
    assert census["ungraded"] == ["agent.cost_per_signal [N/A-MISSING-INPUT]"]


def test_per_tile_breakdown_is_populated() -> None:
    tiles = _tiles(a=[_c("x"), _c("y", "RED")], b=[_c("z", "N/A-LOW-N")])
    per_tile = _census(tiles)["per_tile"]
    assert per_tile == {"a": {"graded": 2, "total": 2}, "b": {"graded": 0, "total": 1}}


# --------------------------------------------------------------------------
# Substitution into the assembled card
# --------------------------------------------------------------------------

def _legacy_coverage_record():
    """The record as the backtester tile builds it from grading.json today."""
    from grading.metric_record import build_metric
    from grading.units import FRACTION

    return build_metric(
        name="evaluator_coverage", module="backtester", metric_type="pct",
        criticality="critical", estimator="coverage_proportion",
        measurement_horizon="trailing_4w", value=0.857, unit=FRACTION,
        n_samples=14, n_floor=1, target=0.95, red_line=0.80,
        source_path="s3://b/backtest/2026-08-21/grading.json",
        reason="evaluator_coverage = 86% (12/14 leaf components graded, non-N/A).",
    ).model_dump(mode="json")


def _sibling_record(name: str):
    from grading.metric_record import build_metric
    from grading.units import FRACTION

    return build_metric(
        name=name, module="backtester", metric_type="pct", criticality="supporting",
        value=1.0, unit=FRACTION, n_samples=5, n_floor=1, target=0.9, red_line=0.5,
        source_path="s3://b/backtest/2026-08-21/grading.json",
    ).model_dump(mode="json")


def _card_with_legacy_coverage():
    return {
        "backtester": {
            "components": [_legacy_coverage_record(), _sibling_record("grading_freshness")],
            "status": "GREEN",
        },
        "agent": {"components": [
            _c(f"agent_{i}", "N/A-MISSING-INPUT") for i in range(10)
        ]},
    }


def test_replacement_reports_the_card_not_the_legacy_artifact() -> None:
    tiles = _card_with_legacy_coverage()
    census = _replace(tiles)
    record = tiles["backtester"]["components"][0]
    assert record["name"] == "evaluator_coverage"
    # 1 graded (grading_freshness) of 11 gradable — NOT 12/14.
    assert record["n_samples"] == 11 != 14
    assert record["value"] == pytest.approx(1 / 11)
    assert census["graded"] == 1


def test_replacement_preserves_the_legacy_number_for_audit() -> None:
    """Never silently drop the number a prior card reported."""
    tiles = _card_with_legacy_coverage()
    _replace(tiles)
    audit = tiles["backtester"]["components"][0]["coverage_census"]
    assert audit["legacy_grading_json_value"] == 0.857
    assert audit["legacy_grading_json_n"] == 14


def test_replacement_repoints_the_source_path() -> None:
    tiles = _card_with_legacy_coverage()
    _replace(tiles)
    assert "grading.json" not in tiles["backtester"]["components"][0]["source_path"]


def test_replacement_redrives_the_tile_status() -> None:
    """`evaluator_coverage` is critical — a collapse must move the tile."""
    tiles = _card_with_legacy_coverage()
    _replace(tiles)
    assert tiles["backtester"]["status"] in {"RED", "WATCH"}


def test_replacement_is_a_noop_without_the_record() -> None:
    tiles = {"backtester": {"components": [_c("other")]}}
    assert _replace(tiles) is None


def test_replacement_never_raises_on_a_malformed_card(caplog) -> None:
    """A defect in the meter must not take down the thing it measures."""
    tiles = _card_with_legacy_coverage()
    # A sibling record the rollup rebuild cannot revalidate.
    tiles["backtester"]["components"].append({"name": "broken", "status": "GREEN"})
    assert _replace(tiles) is None
    assert any("evaluator_coverage recomputation" in r.message for r in caplog.records)
    # The card survives — but it does NOT keep rendering the legacy number.
    assert tiles["backtester"]["components"][0]["value"] is None


def test_a_failed_census_renders_na_not_the_legacy_number(caplog) -> None:
    """The failure path must not fail OPEN (alpha-engine-config-I8193 sweep).

    Keeping the 14-leaf ``grading.json`` value on the card when the census
    could not be computed IS this issue's original defect, re-entered through
    the error path: a true number about a smaller world, on a card that no
    longer says which world. A missing measurement is NULL and visible.
    """
    from grading.coverage import CENSUS_UNKNOWN_MARKER

    tiles = _card_with_legacy_coverage()
    tiles["backtester"]["components"].append({"name": "broken", "status": "GREEN"})
    assert _replace(tiles) is None

    record = tiles["backtester"]["components"][0]
    assert record["name"] == "evaluator_coverage"
    assert record["value"] is None
    assert record["n_samples"] is None
    assert record["status"].startswith("N/A")
    assert "UNMEASURED" in record["status_reason"]
    # The number it replaced is preserved for audit, labelled as the legacy
    # surface it measures — never silently dropped, never silently rendered.
    audit = record["coverage_census"]
    assert audit["legacy_grading_json_value"] == 0.857
    assert audit["legacy_grading_json_n"] == 14
    assert CENSUS_UNKNOWN_MARKER in audit["error"]
    assert any(CENSUS_UNKNOWN_MARKER in r.getMessage() for r in caplog.records)


def test_empty_card_grades_a_transparent_na() -> None:
    tiles = {"backtester": {"components": [_legacy_coverage_record()]}}
    _replace(tiles)
    record = tiles["backtester"]["components"][0]
    assert record["status"].startswith("N/A")
    assert record["value"] is None
