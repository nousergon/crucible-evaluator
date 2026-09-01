"""A tile grades against its DECLARED roster, not against what it was handed.

``alpha-engine-config-I9612``. Parent class ``alpha-engine-config-I8193``,
which fixed the same "denominator is whatever reported" shape one level up, at
the CARD census (``grading/coverage.py``). This file pins the TILE level:

* ``build_tile`` derives ``status`` / ``letter`` / ``numeric_grade`` /
  ``n_components`` from the tile's declared roster, so a builder that silently
  omits a record cannot shrink its own denominator;
* a dropped component holds the tile at WATCH and can never leave it GREEN;
* the roster is the ONE committed registry
  (``grading/thresholds/registry.yaml``, partitioned by ``surface_tile``) — no
  second list;
* the card census (I8193) reports the identical fact identically, once.
"""
from __future__ import annotations

import pytest

from grading.coverage import UNREPORTED_STATUS, card_component_census
from grading.metric_record import build_metric
from grading.module_agg import build_tile, module_status
from grading.thresholds.registry import (
    ThresholdRegistryError, load_registry, tile_roster,
)
from grading.units import RATIO

SRC = "s3://bucket/x/2026-08-28/x.json"


def _owning_module(name: str) -> str:
    for (module, row_name) in load_registry().rows:
        if row_name == name:
            return module
    raise AssertionError(f"{name} is not a registry row")


def _green(name: str, *, criticality: str = "supporting"):
    """One GREEN record for a real roster member, bands bypassed.

    ``band=None`` because this fixture asserts the AGGREGATION rule, not the
    committed bands (those are ``grading/thresholds/`` territory).
    """
    return build_metric(
        name=name, module=_owning_module(name), metric_type="ratio", n_floor=1,
        value=1.0, unit=RATIO, n_samples=120, target=0.5, red_line=0.0,
        band=None, criticality=criticality, estimator="test_robust",
        source_path=SRC, status="GREEN", reason="x",
    )


def _healthy(tile: str, *, critical: set[str] | None = None):
    critical = critical or set()
    return [
        _green(n, criticality="critical" if n in critical else "supporting")
        for n in sorted(tile_roster(tile))
    ]


# --------------------------------------------------------------------------
# The roster itself: derived from the one registry, and unambiguous.
# --------------------------------------------------------------------------

def test_the_tile_rosters_partition_the_registry_exactly():
    """Every registry row surfaces on exactly one tile; no row is stranded."""
    rosters = load_registry().tile_rosters()
    all_rows = {name for _module, name in load_registry().rows}
    union = set().union(*rosters.values())
    assert union == all_rows

    seen: dict[str, str] = {}
    for tile, names in rosters.items():
        for name in names:
            assert name not in seen, (
                f"{name} is on both {seen[name]!r} and {tile!r} — a flat "
                f"per-tile roster is only unambiguous while names are unique"
            )
            seen[name] = tile


def test_the_contribution_lift_family_surfaces_off_its_owning_module():
    """The case that made a surface_tile field necessary at all.

    A ``*_contribution_lift`` row is registered under the module whose
    contribution it measures, and rendered on the ``contribution_lift`` tile.
    Declared on the row, never inferred from the name.
    """
    roster = tile_roster("contribution_lift")
    assert roster, "the contribution_lift tile must declare a roster"
    assert all(n.endswith("_contribution_lift") for n in roster)
    # ...and none of them is counted a second time under its owning module.
    for name in roster:
        assert name not in tile_roster(_owning_module(name))


def test_the_contribution_lift_builders_own_roster_matches_the_registry():
    """The tile builder's ``KNOWN_COMPONENTS`` is not a second registry.

    It predates this rule and is still used on the producer-error path, so it
    is pinned here rather than deleted: two rosters that agree by test are
    fine, two that can silently disagree are the I8193 defect.
    """
    from grading.tiles.contribution_lift import KNOWN_COMPONENTS

    # Its keys are the PRODUCER's bare names; the builder appends the suffix
    # when it mints the record, so the comparison is made in rendered form.
    rendered = {f"{n}_contribution_lift" for n in KNOWN_COMPONENTS}
    assert rendered == tile_roster("contribution_lift")
    for name, module in KNOWN_COMPONENTS.items():
        assert _owning_module(f"{name}_contribution_lift") == module


def test_an_unknown_tile_raises_rather_than_grading_against_nothing():
    with pytest.raises(ThresholdRegistryError, match="no threshold registry rows"):
        build_tile("a_tile_that_does_not_exist", [])


# --------------------------------------------------------------------------
# I9612 closes-when.
# --------------------------------------------------------------------------

def test_a_healthy_tile_counts_its_whole_declared_roster():
    tile = build_tile("director_quality", _healthy("director_quality"))
    assert tile["n_components"] == len(tile_roster("director_quality"))
    assert tile["n_graded"] == tile["n_components"]
    assert tile["unreported"] == []
    assert tile["status"] == "GREEN"


@pytest.mark.parametrize("tile_name", ["director_quality", "behavioral", "agent"])
def test_dropping_any_component_holds_the_tile_at_watch(tile_name):
    """Closes-when 1: delete a component from a builder's output → not GREEN."""
    full = _healthy(tile_name)
    assert build_tile(tile_name, full)["status"] == "GREEN"

    dropped = full[0].name
    tile = build_tile(tile_name, full[1:])
    assert tile["status"] == "WATCH", (
        "a builder that omits a record must not be able to leave its own tile "
        "GREEN over the survivors"
    )
    assert tile["letter"] == "C"
    # Closes-when 3: the denominator does not shrink, and the gap is NAMED.
    assert tile["n_components"] == len(tile_roster(tile_name))
    assert tile["unreported"] == [dropped]
    stub = next(c for c in tile["components"] if c["name"] == dropped)
    assert stub["status"] == UNREPORTED_STATUS
    assert stub["unreported"] is True
    assert dropped in stub["status_reason"]


def test_dropping_a_critical_component_is_never_green():
    """Closes-when 2: the critical gate sees the absence it could not see."""
    tile_name = "director_quality"
    names = sorted(tile_roster(tile_name))
    full = _healthy(tile_name, critical={names[0]})
    assert build_tile(tile_name, full)["status"] == "GREEN"

    tile = build_tile(tile_name, [c for c in full if c.name != names[0]])
    assert tile["status"] == "WATCH"
    assert tile["letter"] != "A"


def test_an_unreported_member_grades_as_an_unmeasured_critical():
    """Its criticality is unknown, so it is read the way that cannot flatter.

    The registry declares bands, not criticality — criticality lives at the
    tile call site, which is exactly the code that failed to run.
    """
    from grading.module_agg import UnreportedComponent

    stub = UnreportedComponent(name="x", module="agent")
    assert stub.criticality == "critical"
    assert stub.is_na
    assert module_status([_green("groom_lost_chunks"), stub]) == "WATCH"


def test_a_wholly_unreported_tile_says_so_rather_than_not_run():
    tile = build_tile("agent", [])
    assert tile["status"] == UNREPORTED_STATUS
    assert tile["letter"] == "N/A"
    assert tile["n_components"] == len(tile_roster("agent"))
    assert tile["n_graded"] == 0
    assert tile["numeric_grade"] is None


def test_an_unreported_member_does_not_outvote_a_declared_na_class():
    """A builder that short-circuits still gets to name the reason it knows.

    The predictor tile's both-inputs-absent path emits ONE honest
    ``N/A-MISSING-INPUT`` sentinel. Without the abstain rule the 14 roster
    members its own short circuit produced would outvote it by plurality, and
    the tile would stop reporting the cause it had already established.
    """
    sentinel = build_metric(
        name="meta_l2_ic", module="predictor", metric_type="ratio", n_floor=1,
        band=None, criticality="critical", estimator="test_robust",
        source_path=SRC, status="N/A-MISSING-INPUT",
        na_detail="predictor metrics absent this cycle",
    )
    tile = build_tile("predictor", [sentinel])
    assert tile["status"] == "N/A-MISSING-INPUT"
    assert len(tile["unreported"]) == len(tile_roster("predictor")) - 1


# --------------------------------------------------------------------------
# The card census (I8193) and the tile line (I9612) report ONE fact.
# --------------------------------------------------------------------------

def test_the_card_census_is_unmoved_by_the_tile_level_stubs():
    """A stub is rendered so the TILE cannot ignore it, and is still counted
    by the card census as declared-but-not-rendered — where I8193 put it —
    rather than migrating into the rendered set and being reported twice."""
    tile_name = "director_quality"
    full = _healthy(tile_name)
    dropped = full[0].name
    roster = tile_roster(tile_name)

    with_stubs = build_tile(tile_name, full[1:])
    census = card_component_census(
        {tile_name: with_stubs}, declared=roster, declared_modules={},
    )
    assert census["total"] == len(roster)
    assert census["graded"] == len(roster) - 1
    # `declared_modules` is injected empty here (this test pins the counting,
    # not the registry's module attribution), so the qualifier renders `?`.
    assert census["unreported"] == [f"?.{dropped}"]
    assert census["rendered_total"] == len(roster) - 1
    assert census["unregistered"] == []
