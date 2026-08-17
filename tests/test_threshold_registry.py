"""The threshold registry is the ONLY place a report-card band may live.

alpha-engine-config#7476 / epic #7473: "a threshold that is not a registry row
is a defect". These tests are the enforcement surface for that sentence —
without them the literals grow back one call site at a time.

champion-challenger §7.4 — a guard must be verified to fail without the fix:
``test_literal_scanner_flags_a_planted_literal`` runs the scanner against a
synthetic offending source and asserts it goes red, so the green result above
means something.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from grading.thresholds.registry import (
    DEFAULT_BAND,
    ThresholdRegistryError,
    ThresholdRegistrySchemaError,
    load_registry,
    parse_registry,
    resolve,
)

TILES_DIR = Path(__file__).resolve().parents[1] / "grading" / "tiles"


def _tile_sources() -> list[Path]:
    return sorted(p for p in TILES_DIR.glob("*.py") if p.name != "__init__.py")


def _build_metric_calls(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = getattr(node.func, "id", getattr(node.func, "attr", None))
            if fn == "build_metric":
                yield node


def _kwargs(call: ast.Call) -> dict[str, ast.expr]:
    return {kw.arg: kw.value for kw in call.keywords if kw.arg}


def scan_threshold_literals(source: str) -> list[tuple[int, str, str | None]]:
    """Every ``build_metric`` call passing ``target``/``red_line``.

    Returns ``(lineno, kwarg, metric_name_literal_or_None)``. Shared by the
    real scan and by the negative control below — one implementation, so the
    control proves the real scanner.
    """
    hits: list[tuple[int, str, str | None]] = []
    tree = ast.parse(source)
    for call in _build_metric_calls(tree):
        kw = _kwargs(call)
        name_node = kw.get("name")
        name = name_node.value if isinstance(name_node, ast.Constant) else None
        for arg in ("target", "red_line"):
            if arg in kw:
                hits.append((call.lineno, arg, name))
    return hits


def _emitted_metric_names(source: str) -> set[tuple[str | None, str]]:
    """``(module_literal, name_literal)`` pairs a tile constructs.

    ``module=`` is the file's ``MODULE`` constant at every call site, so it is
    resolved from the module namespace rather than guessed.
    """
    tree = ast.parse(source)
    consts: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name) and isinstance(node.value, ast.Constant):
                consts[tgt.id] = node.value.value
    out: set[tuple[str | None, str]] = set()
    for call in _build_metric_calls(tree):
        kw = _kwargs(call)
        mod_node, name_node = kw.get("module"), kw.get("name")
        module = None
        if isinstance(mod_node, ast.Constant):
            module = mod_node.value
        elif isinstance(mod_node, ast.Name):
            module = consts.get(mod_node.id)
        if isinstance(name_node, ast.Constant):
            out.add((module, name_node.value))
    return out


class TestNoThresholdLiteralsInTiles:
    """The load-bearing guard: bands live in the registry, not in tile source."""

    def test_no_tile_passes_target_or_red_line(self):
        registry = load_registry()
        offenders: list[str] = []
        for path in _tile_sources():
            for lineno, arg, name in scan_threshold_literals(path.read_text()):
                # The only legitimate pass-through: a row whose bands the DATA
                # publishes (registry `dynamic: true`), which the call site
                # must therefore supply.
                if name is not None and registry.is_dynamic(_module_of(path), name):
                    continue
                offenders.append(f"{path.name}:{lineno} {arg}= (metric={name})")
        assert not offenders, (
            "build_metric may not be handed a threshold from tile source — move it to "
            "grading/thresholds/registry.yaml (alpha-engine-config#7476):\n  "
            + "\n  ".join(offenders)
        )

    def test_literal_scanner_flags_a_planted_literal(self):
        """§7.4 — the guard is shown red against offending source."""
        planted = textwrap.dedent(
            '''
            def f():
                return build_metric(
                    name="planted_metric", module=MODULE, metric_type="pct",
                    n_floor=1, target=0.42, red_line=0.1, source_path="s3://b/x",
                )
            '''
        )
        hits = scan_threshold_literals(planted)
        assert sorted(a for _, a, _ in hits) == ["red_line", "target"]
        assert {n for _, _, n in hits} == {"planted_metric"}

    def test_no_module_level_threshold_constants(self):
        """The literals must not simply move up the file into a constant."""
        banned = {"_TARGET", "_RED_LINE", "_WATCHDOG_TARGET", "_WATCHDOG_RED_LINE"}
        for path in _tile_sources():
            tree = ast.parse(path.read_text())
            names = {
                t.id
                for node in tree.body
                if isinstance(node, ast.Assign)
                for t in node.targets
                if isinstance(t, ast.Name)
            }
            assert not (names & banned), f"{path.name} re-declares a threshold constant"


def _module_of(path: Path) -> str:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name) and tgt.id == "MODULE":
                return node.value.value
    raise AssertionError(f"{path.name} declares no MODULE constant")


class TestRegistryCompleteness:
    def test_every_emitted_metric_has_a_row(self):
        registry = load_registry()
        missing = []
        for path in _tile_sources():
            for module, name in sorted(_emitted_metric_names(path.read_text())):
                if module is None:
                    continue
                if (module, name) not in registry.rows:
                    missing.append(f"{module}.{name} ({path.name})")
        assert not missing, (
            "these metrics grade against bands nobody declared — add a row to "
            "grading/thresholds/registry.yaml in the SAME PR:\n  " + "\n  ".join(missing)
        )

    def test_unknown_metric_raises_rather_than_grading_unbounded(self):
        with pytest.raises(ThresholdRegistryError, match="no threshold registry row"):
            resolve("substrate", "a_metric_nobody_declared")

    def test_unknown_band_raises(self):
        with pytest.raises(ThresholdRegistryError, match="unknown band"):
            resolve("substrate", "price_cache_freshness", "not_a_band")


class TestChampionBandsPinned:
    """The champion arm reproduces the pre-registry literals byte for byte.

    Spot-pins on the metrics the epic names, so a careless registry edit that
    silently re-grades the headline components fails here rather than on the
    console. Source of each value: the tile literal at commit 9712bc7.
    """

    PINNED = {
        ("portfolio_outcome", "sharpe_ratio"): (1.0, 0.0),
        ("portfolio_outcome", "information_ratio"): (0.5, 0.0),
        ("portfolio_outcome", "psr"): (0.95, 0.50),
        ("portfolio_outcome", "alpha_vs_spy"): (0.0, -0.05),
        ("portfolio_outcome", "max_drawdown"): (-0.15, -0.25),
        ("director_quality", "director_grounding"): (75, 40),
        ("director_quality", "director_calibration"): (75, 40),
        ("director_quality", "director_actionability"): (75, 40),
        ("substrate", "watchdog_firings"): (0.0, 2.0),
    }

    @pytest.mark.parametrize("key", sorted(PINNED))
    def test_champion_band(self, key):
        module, name = key
        band = resolve(module, name)
        assert (band.target, band.red_line) == self.PINNED[key]

    def test_shared_raw_precision_band(self):
        band = resolve("research", "sector_teams_avg", "raw_precision")
        assert (band.target, band.red_line) == (0.45, 0.35)

    def test_unbanded_shared_band_imposes_no_bar(self):
        band = resolve("research", "cio", "unbanded")
        assert band.target is None and band.red_line is None
        assert not band.graded


class TestSlotMetadata:
    """champion-challenger §10 — the registry names the slot's own terms."""

    def test_champion_is_an_arm(self):
        slot = load_registry().slot
        assert slot.champion in slot.arms

    def test_slot_declares_metric_horizon_benchmark_matching_hysteresis(self):
        slot = load_registry().slot
        assert slot.objective["benchmark"] == "SPY"
        assert slot.objective["horizon_cycles"] >= 1
        assert slot.scoring["metric"] == "brier"
        assert slot.count_matching["rule"]
        assert slot.hysteresis["promotion_margin_brier"] > 0
        assert slot.hysteresis["demotion_margin_brier"] > 0
        assert slot.hysteresis["cooldown_cycles"] >= 1

    def test_horizon_cannot_exceed_retention(self):
        """§7.1 asserted at LOAD, with a named error — not left to a reviewer."""
        doc = _minimal_doc()
        doc["slot"]["retention"] = {"source_prefix": "evaluator/", "cycle_days": 7,
                                    "expiry_days": 30}
        with pytest.raises(ThresholdRegistrySchemaError, match="structurally impossible"):
            parse_registry(doc)

    def test_cohort_cannot_be_shorter_than_its_own_floor(self):
        doc = _minimal_doc()
        doc["slot"]["scoring"]["cohort_max_cards"] = 5
        with pytest.raises(ThresholdRegistrySchemaError, match="can never reach its own floor"):
            parse_registry(doc)

    def test_vacant_champion_is_rejected(self):
        doc = _minimal_doc()
        doc["slot"]["champion"] = "nobody"
        with pytest.raises(ThresholdRegistrySchemaError, match="not in arms"):
            parse_registry(doc)

    def test_dynamic_row_may_not_also_declare_bands(self):
        doc = _minimal_doc()
        doc["metrics"]["m"]["x"] = {"target": 1.0, "red_line": 0.0, "dynamic": True,
                                    "note": "n"}
        with pytest.raises(ThresholdRegistrySchemaError, match="must not also be declared"):
            parse_registry(doc)

    def test_unknown_row_key_is_rejected(self):
        doc = _minimal_doc()
        doc["metrics"]["m"]["x"] = {"target": 1.0, "red_line": 0.0, "targt": 2.0}
        with pytest.raises(ThresholdRegistrySchemaError, match="unknown keys"):
            parse_registry(doc)


def _minimal_doc() -> dict:
    return {
        "version": 1,
        "slot": {
            "id": "s",
            "champion": "a",
            "arms": ["a", "b"],
            "objective": {"name": "o", "source": "s", "derivation": "cumulative_delta",
                          "unit": "log_return", "horizon_cycles": 4,
                          "horizon_trading_days": 20, "benchmark": "SPY"},
            "scoring": {"metric": "brier", "label": "l", "estimator": "e",
                        "cohort_max_cards": 104, "n_floor_cards": 26,
                        "n_floor_per_status": 5, "statuses_scored": ["GREEN"]},
            "hysteresis": {"promotion_margin_brier": 0.02, "demotion_margin_brier": 0.02,
                           "cooldown_cycles": 4},
            "count_matching": {"rule": "shared_cohort_intersection"},
            "retention": {"source_prefix": "evaluator/", "cycle_days": 7, "expiry_days": None},
        },
        "shared_bands": {},
        "metrics": {"m": {"y": {"target": 1.0, "red_line": 0.0}}},
    }


def test_default_band_name_is_reserved():
    assert DEFAULT_BAND not in load_registry().shared_bands
