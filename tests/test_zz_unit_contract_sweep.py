"""test_zz_unit_contract_sweep.py — config#7485 sweep.

The ``zz`` prefix is deliberate: pytest's default collection walks the
``tests/`` directory in filename order, so this file collects and runs after
every other ``test_*.py`` module — meaning by the time this test body runs,
every tile builder has already been exercised against its own real fixtures
(``test_research_tile.py``, ``test_backtester_tile.py``, ``test_predictor_tile.py``,
``test_substrate_agent_tiles.py``, ``test_executor_tile.py``,
``test_behavioral_tile.py``, ``test_director_quality_tile.py``,
``test_portfolio_outcome.py``, ``test_groom_metrics.py``, plus
``test_metric_record.py`` / ``test_module_agg.py`` / ``test_threshold_registry.py``
and the self-test / handler / aggregate integration tests), and
``tests/conftest.py``'s wrap has recorded every ``MetricRecord`` constructed
along the way.

``grading/metric_record.py::build_metric`` already raises
``MetricContractError`` synchronously the instant a value-bearing record is
built without a unit — so if the suite got this far, the invariant already
held everywhere. This test is the second, independent check the deliverable
asks for: it names every violation (rather than stopping at the first) and
proves the sweep actually observed real call sites, not an empty patch.
"""

# Bare "conftest" (not "tests.conftest"): pytest's default rootless import
# mode loads tests/conftest.py as the top-level module "conftest" (tests/
# has no __init__.py, so it is not a regular package). Importing it via
# "tests.conftest" instead would create a SECOND, distinct module object —
# with its own empty ``_seen`` — disconnected from the one pytest actually
# patched build_metric through (measured 2026-08-16: the sweep silently saw
# zero calls under that form despite the wrapper firing 1000+ times).
from conftest import unit_contract_sweep_results


def test_every_value_bearing_metric_declared_a_unit():
    seen = unit_contract_sweep_results()
    violations = [(module, name, value, unit) for (module, name, value, unit) in seen
                  if value is not None and not unit]
    assert not violations, (
        f"{len(violations)} value-bearing MetricRecord(s) built during the test "
        f"session with no unit (config#7485): {violations[:20]}"
    )


def test_sweep_observed_real_call_sites():
    # A guard with zero coverage is not a passing test — this fails loud if
    # the conftest wrap never engaged (e.g. import-order change).
    seen = unit_contract_sweep_results()
    assert len(seen) > 100, (
        f"unit-contract sweep only observed {len(seen)} build_metric call(s) — "
        f"expected 100+ from the tile-builder test suite; the wrap likely never engaged."
    )


def test_sweep_covers_every_tile_module():
    seen = unit_contract_sweep_results()
    modules_seen = {module for (module, _name, _value, _unit) in seen}
    expected = {
        "research", "backtester", "predictor", "substrate", "agent",
        "executor", "behavioral", "portfolio_outcome", "director_quality",
    }
    missing = expected - modules_seen
    assert not missing, f"no build_metric call observed for module(s): {sorted(missing)}"
