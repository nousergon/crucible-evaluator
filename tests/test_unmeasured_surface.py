"""A surface with nothing measured on it must not render as a measured one.

alpha-engine-config-I8177 — the class behind ``evaluator_coverage`` measuring
a 14-leaf legacy artifact, swept over the two remaining call sites:

1. a TILE in which nothing graded rendered ``WATCH`` / ``C``;
2. the v1 ``overall`` composite reported ``qualifier: COMPLETE`` over three
   declared modules on a card carrying ten tiles.

`principles.md` §2.7: *no data* is never rendered as a measurement.
"""

import pytest

from grading.coverage import PARTIAL_SCOPE, stamp_composite_scope


def _roster(tiles):
    """The declared roster for a SYNTHETIC card: exactly what it renders."""
    return {
        c["name"]
        for tile in tiles.values()
        for c in (tile.get("components") or [])
    }
from grading.module_agg import build_tile, module_status, overall_status, unmeasured_status
from grading.metric_record import build_metric
from grading.scorecard import _display
from grading.units import RATIO

SRC = "s3://b/x"


def _na(name, status, *, criticality="critical"):
    kwargs = {}
    if status == "N/A-MISSING-INPUT":
        kwargs["input_present"] = False
    elif status == "N/A-LOW-N":
        kwargs["n_samples"] = 1
    return build_metric(
        name=name, module="agent", metric_type="ratio", n_floor=60,
        criticality=criticality, estimator="test_robust", source_path=SRC,
        status=status, na_detail="nothing this cycle", band=None, **kwargs,
    )


def _graded(name, status="GREEN", *, criticality="critical"):
    return build_metric(
        name=name, module="agent", metric_type="ratio", n_floor=1, value=1.0,
        unit=RATIO, n_samples=120, target=0.5, red_line=0.0,
        criticality=criticality, estimator="test_robust", source_path=SRC,
        status=status, reason="x",
    )


class TestUnmeasuredTile:
    def test_all_na_tile_is_not_watch(self):
        """The live 2026-08-22 `agent` tile: 11 N/A of 11, rendered WATCH/C."""
        components = [_na(f"c{i}", "N/A-MISSING-INPUT") for i in range(10)]
        components.append(_na("c10", "N/A-NOT-IMPL"))
        assert module_status(components) == "N/A-MISSING-INPUT"
        tile = build_tile("agent", components)
        assert tile["letter"] == "N/A"
        assert tile["n_components"] == 11
        assert tile["n_graded"] == 0

    def test_one_graded_component_keeps_the_tile_measured(self):
        """The rule fires on NOTHING graded, never on 'mostly N/A'."""
        components = [_na(f"c{i}", "N/A-MISSING-INPUT") for i in range(10)]
        components.append(_graded("c10"))
        assert module_status(components) == "WATCH"
        assert build_tile("agent", components)["n_graded"] == 1

    def test_status_names_the_reason_by_plurality(self):
        assert unmeasured_status(
            [_na("a", "N/A-NOT-IMPL"), _na("b", "N/A-NOT-IMPL"),
             _na("c", "N/A-MISSING-INPUT")],
        ) == "N/A-NOT-IMPL"

    def test_ties_break_to_the_most_actionable_class(self):
        assert unmeasured_status(
            [_na("a", "N/A-NOT-IMPL"), _na("b", "N/A-MISSING-INPUT")],
        ) == "N/A-MISSING-INPUT"

    def test_measured_tile_returns_none(self):
        assert unmeasured_status([_graded("a"), _na("b", "N/A-LOW-N")]) is None

    def test_empty_is_not_this_rule(self):
        assert unmeasured_status([]) is None
        assert module_status([]) == "N/A-NOT-RUN"


class TestUnmeasuredTileCannotGreenTheCard:
    def _tiles(self, **over):
        base = {
            "portfolio_outcome": "GREEN", "research": "GREEN",
            "predictor": "GREEN", "executor": "GREEN", "substrate": "GREEN",
        }
        base.update(over)
        return base

    def test_all_green_is_green(self):
        assert overall_status(self._tiles()) == "GREEN"

    @pytest.mark.parametrize("module", ["research", "predictor", "executor", "substrate"])
    def test_unmeasured_cascade_module_holds_at_watch(self, module):
        """Going dark must never make the card greener than going wrong."""
        assert overall_status(self._tiles(**{module: "N/A-MISSING-INPUT"})) == "WATCH"

    def test_unmeasured_lead_tile_holds_at_watch(self):
        assert overall_status(
            self._tiles(portfolio_outcome="N/A-MISSING-INPUT"),
        ) == "WATCH"


def _card(qualifier="COMPLETE"):
    return {
        "overall": {
            "grade": 55.3, "letter": "C+", "display": "C+",
            "coverage": {"qualifier": qualifier, "components_declared": 3,
                         "components_present": 3,
                         "weight_present_effective": 1.0},
        },
        "grading_weights": {"overall": {"research": 0.4, "predictor": 0.25,
                                        "executor": 0.35}},
    }


def _tile_dict(name, statuses):
    components = [
        _graded(f"{name}{i}") if s == "GREEN" else _na(f"{name}{i}", s)
        for i, s in enumerate(statuses)
    ]
    return build_tile(name, components)


class TestCompositeScope:
    def _tiles(self):
        return {
            "research": _tile_dict("research", ["GREEN", "N/A-MISSING-INPUT"]),
            "predictor": _tile_dict("predictor", ["GREEN"]),
            "executor": _tile_dict("executor", ["GREEN"]),
            "agent": _tile_dict("agent", ["N/A-MISSING-INPUT"]),
            "portfolio_outcome": _tile_dict("portfolio_outcome", ["GREEN"]),
        }

    def test_complete_is_demoted_while_tiles_sit_outside_the_composite(self):
        card = _card()
        scope = stamp_composite_scope(card, self._tiles(), declared=_roster(self._tiles()), declared_modules={})
        assert card["overall"]["coverage"]["qualifier"] == PARTIAL_SCOPE
        assert scope["tiles_in_scope"] == ["executor", "predictor", "research"]
        assert set(scope["tiles_out_of_scope"]) == {"agent", "portfolio_outcome"}
        # The out-of-scope tiles carry their VERDICT, not just their name.
        assert scope["tiles_out_of_scope"]["agent"].startswith("N/A")

    def test_leaf_counts_name_both_worlds(self):
        card = _card()
        tiles = self._tiles()
        # The roster is injected as the fixture's own component set: this test
        # pins the SCOPE arithmetic, not the live threshold registry's contents
        # (the registry-backed denominator is alpha-engine-config-I8193, pinned
        # in test_coverage_registry_denominator.py).
        scope = stamp_composite_scope(card, tiles, declared=_roster(tiles),
                                      declared_modules={})
        assert scope["leaf_components_in_scope"] == 4
        assert scope["leaf_components_on_card"] == 6
        assert scope["card_leaf_graded"] == 4

    def test_display_never_renders_the_bare_letter(self):
        card = _card()
        stamp_composite_scope(card, self._tiles(), declared=_roster(self._tiles()), declared_modules={})
        display = card["overall"]["display"]
        assert display != "C+"
        assert "PARTIAL SCOPE" in display
        assert "agent" in display

    def test_a_composite_covering_every_tile_stays_complete(self):
        """The demotion is a MEASURED fact about scope, not a blanket downgrade."""
        card = _card()
        tiles = {k: v for k, v in self._tiles().items()
                 if k in ("research", "predictor", "executor")}
        stamp_composite_scope(card, tiles, declared=_roster(tiles), declared_modules={})
        assert card["overall"]["coverage"]["qualifier"] == "COMPLETE"
        assert card["overall"]["display"] == "C+"

    def test_an_already_partial_qualifier_is_left_alone(self):
        card = _card(qualifier="PARTIAL-FAILURE-SCORED-ZERO")
        stamp_composite_scope(card, self._tiles(), declared=_roster(self._tiles()), declared_modules={})
        assert card["overall"]["coverage"]["qualifier"] == "PARTIAL-FAILURE-SCORED-ZERO"
        assert card["overall"]["coverage"]["census_scope"]["tiles_on_card"] == 5

    def test_never_raises_on_a_malformed_card(self):
        assert stamp_composite_scope({}, self._tiles(), declared=frozenset(), declared_modules={}) is None
        assert stamp_composite_scope({"overall": {}}, self._tiles(), declared=frozenset(), declared_modules={}) is None

    def test_carries_no_hardcoded_tile_list(self):
        """Scope is DERIVED from grading_weights + the tiles actually present.

        `observability-policy.md` §2.2 — a hand-listed scope drifts, and its
        drift is invisible because the missing rows produce no signal.
        """
        import pathlib
        source = (
            pathlib.Path(__file__).resolve().parents[1] / "grading" / "coverage.py"
        ).read_text()
        body = source.split("def _stamp_composite_scope")[1]
        for tile in ("portfolio_outcome", "behavioral", "director_quality",
                     "contribution_lift", "substrate"):
            assert tile not in body, f"{tile} hardcoded in the scope derivation"


class TestDisplay:
    def test_partial_scope_string_is_self_explaining(self):
        out = _display("C+", {
            "qualifier": PARTIAL_SCOPE,
            "census_scope": {
                "tiles_in_scope": ["executor", "predictor", "research"],
                "tiles_on_card": 10,
                "tiles_out_of_scope": {"agent": "N/A-MISSING-INPUT"},
                "card_leaf_coverage": 0.624,
            },
        })
        assert "3 of 10 tiles" in out
        assert "62%" in out
        assert "tiles_overall_status" in out
