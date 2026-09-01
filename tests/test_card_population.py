"""The report card declares its population ONCE — alpha-engine-config-I9734.

Until this landed the card declared who was on it four times and nothing
reconciled them:

  1. ``grading/thresholds/registry.yaml#metrics`` — 9 modules over 130 rows,
     surfacing on 10 tiles once each row's ``surface_tile`` is applied.
  2. ``grading/aggregate.py`` — a hardcoded ten-key ``_tile_builders`` list
     whose own comment called itself "the membership source of truth".
  3. ``grading/module_agg.py::_CASCADE_MODULES`` — a four-name tuple.
  4. ``grading/scorecard.py`` — the keys of ``OVERALL_WEIGHTS`` /
     ``PROCESS_WEIGHTS`` / the per-module component tables.

Adding a fleet module took six coordinated hand-edits, and missing one graded
it on one path while making it invisible on the other. These tests are what
makes the populations unable to drift: (1) is now the only DECLARATION,
(2)–(4) are derived from it, and a disagreement raises at registry LOAD.

The load-bearing test is ``TestAModuleAddedToTheRegistryReachesTheCard`` — it
fails on the pre-I9734 code, where a module added to the registry appeared in
the coverage denominator and on no tile.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from grading import aggregate
from grading.module_agg import _CASCADE_MODULES
from grading.scorecard import (
    GRADE_BANDS,
    OVERALL_WEIGHTS,
    PROCESS_WEIGHTS,
    WEIGHT_TABLE_VERSION,
)
from grading.thresholds import registry as reg
from grading.thresholds.registry import (
    REGISTRY_PATH,
    ThresholdRegistrySchemaError,
    card_spec,
    declared_tiles,
    load_registry,
    parse_registry,
)


def _doc() -> dict:
    with open(REGISTRY_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _derived_tiles(doc: dict) -> set[str]:
    """The card's real population: every distinct ``surface_tile`` in the rows."""
    return {
        (row or {}).get("surface_tile") or module
        for module, rows in doc["metrics"].items()
        for _, row in rows.items()
    }


class TestTheFourPopulationsAreOne:
    """A module in one population and absent from another must FAIL."""

    def test_declared_tiles_equal_the_tiles_derived_from_the_rows(self):
        assert set(declared_tiles()) == _derived_tiles(_doc())

    def test_the_committed_registry_declares_ten_tiles_over_nine_modules(self):
        # Not a target to hit — a statement of what is committed today, so a
        # membership change has to be a deliberate edit to this line.
        assert len(_doc()["metrics"]) == 9
        assert len(declared_tiles()) == 10
        assert "contribution_lift" in declared_tiles()
        assert "contribution_lift" not in _doc()["metrics"], (
            "contribution_lift is cross-cutting: its rows are filed under their "
            "OWNING module and reach its tile via surface_tile, which is the "
            "relationship being declared in the registry rather than special-cased "
            "in Python"
        )

    def test_a_row_surfacing_on_an_undeclared_tile_raises(self):
        doc = copy.deepcopy(_doc())
        first_module = next(iter(doc["metrics"]))
        first_row = next(iter(doc["metrics"][first_module]))
        doc["metrics"][first_module][first_row]["surface_tile"] = "fleet_widget"
        with pytest.raises(ThresholdRegistrySchemaError, match="fleet_widget"):
            parse_registry(doc)

    def test_a_declared_tile_no_row_surfaces_on_raises(self):
        doc = copy.deepcopy(_doc())
        doc["card"]["tiles"]["fleet_widget"] = {}
        with pytest.raises(ThresholdRegistrySchemaError, match="fleet_widget"):
            parse_registry(doc)

    def test_dropping_a_declared_tile_raises(self):
        doc = copy.deepcopy(_doc())
        doc["card"]["tiles"].pop("substrate")
        with pytest.raises(ThresholdRegistrySchemaError, match="substrate"):
            parse_registry(doc)

    def test_aggregate_builds_exactly_the_declared_tiles_in_declared_order(self):
        # aggregate.py contributes BUILDERS; the registry decides membership.
        assert [name for name, _ in aggregate.resolve_tile_builders({})] == list(
            declared_tiles()
        )

    def test_a_builder_for_an_undeclared_tile_raises(self):
        with pytest.raises(ValueError, match="fleet_widget"):
            aggregate.resolve_tile_builders({"fleet_widget": lambda: {}})

    def test_cascade_modules_are_declared_tiles(self):
        assert set(_CASCADE_MODULES) <= set(declared_tiles())
        assert set(_CASCADE_MODULES) == {
            tile for tile, spec in _doc()["card"]["tiles"].items()
            if (spec or {}).get("cascades")
        }

    def test_weight_table_keys_are_declared_tiles(self):
        assert set(OVERALL_WEIGHTS) <= set(declared_tiles())
        assert set(PROCESS_WEIGHTS) <= set(declared_tiles())
        assert set(card_spec().component_weights) <= set(declared_tiles())


class TestAModuleAddedToTheRegistryReachesTheCard:
    """The closes-when: registry edit only, no Python edit.

    On the pre-I9734 code a module added to ``registry.yaml`` entered the
    coverage denominator (``grading/coverage.py`` already read the registry)
    and reached NO tile — graded on one path, invisible on the other. That is
    the drift these assert away.
    """

    @pytest.fixture
    def registry_with_a_new_module(self, monkeypatch):
        doc = copy.deepcopy(_doc())
        doc["metrics"]["fleet_widget"] = {
            "widget_uptime_ratio": {"target": 0.99, "red_line": 0.95,
                                    "higher_is_better": True},
        }
        doc["card"]["tiles"]["fleet_widget"] = {}
        patched = parse_registry(doc)
        monkeypatch.setattr(reg, "load_registry", lambda path=None: patched)
        return patched

    def test_the_new_module_becomes_a_card_tile(self, registry_with_a_new_module):
        assert "fleet_widget" in reg.declared_tiles()
        assert [name for name, _ in aggregate.resolve_tile_builders({})][-1] == (
            "fleet_widget"
        )

    def test_the_new_tile_renders_its_declared_roster_as_unreported(
        self, registry_with_a_new_module,
    ):
        # A declared tile with no grading/tiles/<name>.py yet renders as its
        # full roster, every component UNREPORTED — the honest state, and one
        # the coverage census can see. It is never silently absent.
        builders = dict(aggregate.resolve_tile_builders({}))
        tile = builders["fleet_widget"]()
        assert tile["module"] == "fleet_widget"
        assert tile["unreported"] == ["widget_uptime_ratio"]
        assert tile["n_components"] == 1
        assert tile["n_graded"] == 0


class TestTheGradingTablesMovedWithoutChanging:
    """Deliverable 2 is a MOVE. The published grade must not move with it."""

    def test_grade_bands_are_the_eleven_committed_cutoffs(self):
        assert GRADE_BANDS == [
            (90, "A"), (80, "A-"), (73, "B+"), (65, "B"), (58, "B-"),
            (50, "C+"), (42, "C"), (35, "C-"), (28, "D+"), (20, "D"), (0, "F"),
        ]

    def test_the_headline_composite_still_has_exactly_four_voters(self):
        # Widening this to all nine modules CHANGES the published grade and is
        # alpha-engine-config-I9690's decision, not this file's.
        assert OVERALL_WEIGHTS == {
            "portfolio_outcome": 0.50,
            "research": pytest.approx(0.20),
            "predictor": pytest.approx(0.125),
            "executor": pytest.approx(0.175),
        }
        assert PROCESS_WEIGHTS == {
            "research": pytest.approx(0.40),
            "predictor": pytest.approx(0.25),
            "executor": pytest.approx(0.35),
        }

    def test_weight_table_version_comes_from_the_registry(self):
        assert WEIGHT_TABLE_VERSION == _doc()["card"]["weight_table_version"]

    def test_every_weight_table_sums_to_one(self):
        for tile, table in card_spec().component_weights.items():
            assert sum(table.values()) == pytest.approx(1.0, abs=1e-9), tile
        assert sum(OVERALL_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-9)


class TestTheRegistryRefusesAnIncoherentCard:
    def test_a_weight_table_that_does_not_sum_to_one_raises(self):
        doc = copy.deepcopy(_doc())
        doc["card"]["component_weights"]["predictor"]["meta_model"] = 0.90
        with pytest.raises(ThresholdRegistrySchemaError, match="sums to"):
            parse_registry(doc)

    def test_process_weights_that_do_not_sum_to_one_raise(self):
        doc = copy.deepcopy(_doc())
        doc["card"]["tiles"]["research"]["process_weight"] = 0.90
        with pytest.raises(ThresholdRegistrySchemaError, match="sums to"):
            parse_registry(doc)

    def test_an_out_of_order_grade_ladder_raises(self):
        doc = copy.deepcopy(_doc())
        doc["card"]["grade_bands"][0], doc["card"]["grade_bands"][1] = (
            doc["card"]["grade_bands"][1], doc["card"]["grade_bands"][0],
        )
        with pytest.raises(ThresholdRegistrySchemaError, match="high to low"):
            parse_registry(doc)

    def test_a_ladder_that_does_not_reach_zero_raises(self):
        doc = copy.deepcopy(_doc())
        doc["card"]["grade_bands"][-1]["min"] = 5
        with pytest.raises(ThresholdRegistrySchemaError, match="min: 0"):
            parse_registry(doc)

    def test_two_headline_weights_raise(self):
        doc = copy.deepcopy(_doc())
        doc["card"]["tiles"]["backtester"] = {"headline_weight": 0.1}
        with pytest.raises(ThresholdRegistrySchemaError, match="more than one"):
            parse_registry(doc)

    def test_a_tile_voting_twice_raises(self):
        doc = copy.deepcopy(_doc())
        doc["card"]["tiles"]["research"]["headline_weight"] = 0.1
        with pytest.raises(ThresholdRegistrySchemaError, match="votes in the composite once"):
            parse_registry(doc)

    def test_a_missing_card_block_raises(self):
        doc = copy.deepcopy(_doc())
        doc.pop("card")
        with pytest.raises(ThresholdRegistrySchemaError, match="registry.card"):
            parse_registry(doc)

    def test_the_committed_registry_parses(self):
        assert load_registry().card.tiles == declared_tiles()
