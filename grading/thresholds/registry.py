"""registry.py — the committed threshold registry for the Report Card v2 bands.

Every ``target``/``red_line`` a ``MetricRecord`` is graded against used to be a
literal in tile source. That made the single most consequential decision the
card makes — where GREEN stops and RED starts — invisible to measurement: no
arm, no history, no way to ask whether a GREEN at week *t* predicted anything
about week *t+1* (champion-challenger-policy §2, alpha-engine-config#7476).

This module makes that decision a **slot** with one champion arm
(``declared_v2`` — the literals as they stood on 2026-08-16, moved here byte
for byte) and challenger arms scored beside it every cycle
(``grading/thresholds/scoring.py``).

Binding rules, all enforced here rather than by convention:

  * **Every ``(module, metric)`` the card emits has a row.** ``resolve`` raises
    ``ThresholdRegistryError`` on an unknown key — a metric whose bands are not
    a registry row is a defect (epic #7473 constraint), and a silent ``None``
    would grade it GREEN-by-default.
  * **Rows may declare no bands** (``target: null``, ``red_line: null``). That
    is a real, documented state — an ungraded diagnostic — and is distinct from
    an absent row.
  * **Shared bands** (``unbanded``, ``raw_precision``) express a rule that
    belongs to a *branch* rather than to a metric: the same fallback applied to
    whichever metric took that path. The row must still exist; the shared band
    overrides the numbers.
  * **``dynamic: true`` rows** carry bands derived from the data itself (the
    L4515 turnover tripwire publishes its own band), so the call site passes
    them explicitly. Only such rows may.

The registry is loaded once and cached. It is data, not code: a promotion is a
YAML edit (see ``grading/thresholds/promote.py``), never a source change.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REGISTRY_PATH = Path(__file__).with_name("registry.yaml")

#: The band name used when a call site does not ask for one — the champion's
#: declared bands for that metric.
DEFAULT_BAND = "champion"


class ThresholdRegistryError(KeyError):
    """A metric was graded against bands that are not in the registry.

    Raised loud at ``build_metric`` construction (the chokepoint every tile
    passes through) so an unregistered metric can never reach the report card
    grading against ``None``/``None`` — which ``derive_status`` reads as
    "no bar", i.e. GREEN above the floor.
    """


class ThresholdRegistrySchemaError(ValueError):
    """The committed registry.yaml violates its own schema."""


@dataclass(frozen=True)
class Band:
    """One resolved (target, red_line) pair plus the row's declared metadata."""

    module: str
    name: str
    band: str
    target: float | None
    red_line: float | None
    higher_is_better: bool | None
    n_floor_declared: int | None
    dynamic: bool
    note: str | None
    surface_tile: str

    @property
    def graded(self) -> bool:
        """True when this band actually imposes a bar."""
        return self.target is not None or self.red_line is not None


@dataclass(frozen=True)
class SlotSpec:
    """The champion/challenger slot metadata champion-challenger §10 requires."""

    id: str
    champion: str
    arms: tuple[str, ...]
    objective: dict[str, Any]
    scoring: dict[str, Any]
    hysteresis: dict[str, Any]
    count_matching: dict[str, Any]
    retention: dict[str, Any]

    @property
    def horizon_cycles(self) -> int:
        return int(self.objective["horizon_cycles"])

    @property
    def n_floor_cards(self) -> int:
        return int(self.scoring["n_floor_cards"])

    @property
    def cohort_max_cards(self) -> int:
        return int(self.scoring["cohort_max_cards"])


@dataclass(frozen=True)
class CardSpec:
    """The report card's DECLARED population and the tables it grades with.

    ``alpha-engine-config-I9734``. The card used to declare its membership in
    four places that nothing reconciled — this file's ``metrics`` modules, a
    hardcoded tile list in ``grading/aggregate.py``, ``module_agg``'s cascade
    tuple, and the keys of the Python weight tables in ``grading/scorecard.py``.
    They are one declaration now, and ``parse_registry`` raises when the
    declared tile order and the tile set DERIVED from the rows' ``surface_tile``
    disagree, so the two can no longer drift apart silently.

    ``tiles`` declares ORDER and per-tile ROLE. It cannot declare membership:
    that comes from the rows.
    """

    weight_table_version: str
    grade_bands: tuple[tuple[float, str], ...]
    tiles: tuple[str, ...]
    cascade_modules: tuple[str, ...]
    portfolio_outcome_weight: float
    process_weights: dict[str, float]
    component_weights: dict[str, dict[str, float]]

    @property
    def overall_weights(self) -> dict[str, float]:
        """The headline composite's declared voters and their weights.

        The outcome tile at its declared ``headline_weight``, and each process
        module at its share of the remaining half — the same arithmetic
        ``scorecard.OVERALL_WEIGHTS`` performed as a Python literal expression.
        """
        rest = 1.0 - self.portfolio_outcome_weight
        return {
            "portfolio_outcome": self.portfolio_outcome_weight,
            **{name: w * rest for name, w in self.process_weights.items()},
        }


@dataclass(frozen=True)
class ThresholdRegistry:
    slot: SlotSpec
    shared_bands: dict[str, tuple[float | None, float | None]]
    rows: dict[tuple[str, str], dict[str, Any]]
    card: CardSpec

    def resolve(self, module: str, name: str, band: str = DEFAULT_BAND) -> Band:
        try:
            row = self.rows[(module, name)]
        except KeyError:
            raise ThresholdRegistryError(
                f"no threshold registry row for ({module!r}, {name!r}) — every graded "
                f"MetricRecord needs one (alpha-engine-config#7476). Add it to "
                f"{REGISTRY_PATH.name} in the same PR as the metric; a row with "
                f"target: null / red_line: null is the correct entry for an ungraded "
                f"diagnostic."
            ) from None
        if band == DEFAULT_BAND:
            target, red_line = row.get("target"), row.get("red_line")
        elif band in self.shared_bands:
            target, red_line = self.shared_bands[band]
        else:
            raise ThresholdRegistryError(
                f"unknown band {band!r} for ({module!r}, {name!r}) — known shared bands: "
                f"{sorted(self.shared_bands)}"
            )
        return Band(
            module=module,
            name=name,
            band=band,
            target=target,
            red_line=red_line,
            higher_is_better=row.get("higher_is_better"),
            n_floor_declared=row.get("n_floor_declared"),
            dynamic=bool(row.get("dynamic", False)),
            note=row.get("note"),
            surface_tile=row.get("surface_tile") or module,
        )

    def is_dynamic(self, module: str, name: str) -> bool:
        return bool(self.rows.get((module, name), {}).get("dynamic", False))

    def surface_tile(self, module: str, name: str) -> str:
        """The card tile this row renders on — its module unless declared otherwise."""
        row = self.rows.get((module, name))
        if row is None:
            raise ThresholdRegistryError(
                f"no threshold registry row for ({module!r}, {name!r})"
            )
        return row.get("surface_tile") or module

    def tile_rosters(self) -> dict[str, frozenset[str]]:
        """``surface tile -> the component names that tile is obliged to render``.

        The per-tile half of ``coverage.declared_component_names``'s roster, and
        derived from the SAME rows (``alpha-engine-config-I9612``). A tile's
        status is graded against this set rather than against the list its
        builder happened to hand ``build_tile``, so a builder that silently
        drops a component cannot shrink its own denominator.

        Names are unique across the whole registry (pinned by a test), so a
        flat per-tile set is unambiguous.
        """
        out: dict[str, set[str]] = {}
        for (module, name), row in self.rows.items():
            out.setdefault(row.get("surface_tile") or module, set()).add(name)
        return {tile: frozenset(names) for tile, names in out.items()}

    def graded_keys(self) -> list[tuple[str, str]]:
        """Every ``(module, metric)`` whose champion band imposes a bar.

        The scored population for both arms — an ungraded diagnostic has no
        status to be right or wrong about.
        """
        return sorted(
            k for k, row in self.rows.items()
            if row.get("target") is not None or row.get("red_line") is not None
        )


_REQUIRED_SLOT_KEYS = ("id", "champion", "arms", "objective", "scoring", "hysteresis",
                       "count_matching", "retention")
_REQUIRED_OBJECTIVE_KEYS = ("name", "source", "derivation", "unit", "horizon_cycles",
                            "horizon_trading_days", "benchmark")
_REQUIRED_SCORING_KEYS = ("metric", "label", "estimator", "cohort_max_cards",
                          "n_floor_cards", "n_floor_per_status", "statuses_scored")
_ALLOWED_ROW_KEYS = frozenset({"target", "red_line", "higher_is_better",
                               "n_floor_declared", "dynamic", "note",
                               "surface_tile"})


_ALLOWED_TILE_KEYS = frozenset({"cascades", "process_weight", "headline_weight"})


def _check_sums_to_one(table: dict[str, float], what: str) -> None:
    total = sum(table.values())
    if abs(total - 1.0) > 1e-9:
        raise ThresholdRegistrySchemaError(
            f"{what} sums to {total!r}, not 1.0 — a weight table that does not "
            f"sum to 1 publishes a grade nobody can reproduce from the card"
        )


def parse_card(doc: dict[str, Any], derived_tiles: frozenset[str]) -> CardSpec:
    """Validate + structure ``registry.card`` against the DERIVED tile set.

    ``derived_tiles`` is every distinct ``surface_tile`` across the rows — the
    card's real population. ``card.tiles`` may order and annotate it; a
    disagreement in either direction raises here, at load, rather than
    surfacing as a module graded on one path and invisible on another
    (``alpha-engine-config-I9734``).
    """
    card_doc = doc.get("card")
    if not isinstance(card_doc, dict):
        raise ThresholdRegistrySchemaError("registry.card must be a mapping")

    version = card_doc.get("weight_table_version")
    if not isinstance(version, str) or not version:
        raise ThresholdRegistrySchemaError(
            "registry.card.weight_table_version must be a non-empty string — it is "
            "stamped onto every published card as grading_weights.version"
        )

    bands_doc = card_doc.get("grade_bands")
    if not isinstance(bands_doc, list) or not bands_doc:
        raise ThresholdRegistrySchemaError("registry.card.grade_bands must be a non-empty list")
    bands: list[tuple[float, str]] = []
    for entry in bands_doc:
        if (not isinstance(entry, dict) or set(entry) != {"min", "letter"}
                or not isinstance(entry["letter"], str)):
            raise ThresholdRegistrySchemaError(
                f"registry.card.grade_bands entry {entry!r} must be {{min, letter}}"
            )
        bands.append((float(entry["min"]), entry["letter"]))
    if [b[0] for b in bands] != sorted((b[0] for b in bands), reverse=True):
        raise ThresholdRegistrySchemaError(
            "registry.card.grade_bands must be ordered high to low — the ladder is "
            "read first-match-wins, so an out-of-order band silently swallows the "
            "ones below it"
        )
    if bands[-1][0] != 0:
        raise ThresholdRegistrySchemaError(
            "registry.card.grade_bands must end at min: 0 — a ladder with a floor "
            "above 0 has scores it maps to no letter at all"
        )

    tiles_doc = card_doc.get("tiles")
    if not isinstance(tiles_doc, dict) or not tiles_doc:
        raise ThresholdRegistrySchemaError("registry.card.tiles must be a non-empty mapping")

    declared = tuple(tiles_doc)
    # THE reconciliation, enforced at load rather than left to a convention.
    missing = sorted(derived_tiles - set(declared))
    extra = sorted(set(declared) - derived_tiles)
    if missing or extra:
        raise ThresholdRegistrySchemaError(
            f"registry.card.tiles disagrees with the tile set derived from the metric "
            f"rows (alpha-engine-config-I9734): rows surface on {missing!r} which "
            f"card.tiles does not declare; card.tiles declares {extra!r} which no row "
            f"surfaces on. card.tiles orders and annotates the card's population — the "
            f"rows (and their `surface_tile`) define it."
        )

    cascade: list[str] = []
    process: dict[str, float] = {}
    outcome_weight: float | None = None
    for tile, spec in tiles_doc.items():
        spec = spec or {}
        if not isinstance(spec, dict):
            raise ThresholdRegistrySchemaError(
                f"registry.card.tiles.{tile} must be a mapping (use {{}} for a tile "
                f"that only renders)"
            )
        unknown = set(spec) - _ALLOWED_TILE_KEYS
        if unknown:
            raise ThresholdRegistrySchemaError(
                f"registry.card.tiles.{tile} has unknown keys {sorted(unknown)}"
            )
        if "process_weight" in spec and "headline_weight" in spec:
            raise ThresholdRegistrySchemaError(
                f"registry.card.tiles.{tile} declares both process_weight and "
                f"headline_weight — a tile votes in the composite once"
            )
        if spec.get("cascades"):
            cascade.append(tile)
        if "process_weight" in spec:
            process[tile] = float(spec["process_weight"])
        if "headline_weight" in spec:
            if outcome_weight is not None:
                raise ThresholdRegistrySchemaError(
                    "registry.card.tiles declares headline_weight on more than one "
                    "tile — the process half is shared via process_weight"
                )
            outcome_weight = float(spec["headline_weight"])

    if outcome_weight is None:
        raise ThresholdRegistrySchemaError(
            "registry.card.tiles declares no headline_weight — the composite needs "
            "the product-outcome voter's share (alpha-engine-config-I9005)"
        )
    if not process:
        raise ThresholdRegistrySchemaError(
            "registry.card.tiles declares no process_weight on any tile"
        )
    _check_sums_to_one(process, "registry.card process_weight over the process tiles")

    weights_doc = card_doc.get("component_weights")
    if not isinstance(weights_doc, dict) or not weights_doc:
        raise ThresholdRegistrySchemaError(
            "registry.card.component_weights must be a non-empty mapping"
        )
    component_weights: dict[str, dict[str, float]] = {}
    for tile, table in weights_doc.items():
        if tile not in declared:
            raise ThresholdRegistrySchemaError(
                f"registry.card.component_weights.{tile} is not a declared card tile "
                f"— known tiles: {sorted(declared)}"
            )
        if not isinstance(table, dict) or not table:
            raise ThresholdRegistrySchemaError(
                f"registry.card.component_weights.{tile} must be a non-empty mapping"
            )
        parsed = {name: float(w) for name, w in table.items()}
        _check_sums_to_one(parsed, f"registry.card.component_weights.{tile}")
        component_weights[tile] = parsed

    return CardSpec(
        weight_table_version=version,
        grade_bands=tuple(bands),
        tiles=declared,
        cascade_modules=tuple(cascade),
        portfolio_outcome_weight=outcome_weight,
        process_weights=process,
        component_weights=component_weights,
    )


def parse_registry(doc: dict[str, Any]) -> ThresholdRegistry:
    """Validate + structure a parsed registry document. Raises on any violation."""
    if not isinstance(doc, dict):
        raise ThresholdRegistrySchemaError("registry root must be a mapping")
    if doc.get("version") != 1:
        raise ThresholdRegistrySchemaError(f"unsupported registry version {doc.get('version')!r}")

    slot_doc = doc.get("slot")
    if not isinstance(slot_doc, dict):
        raise ThresholdRegistrySchemaError("registry.slot must be a mapping")
    missing = [k for k in _REQUIRED_SLOT_KEYS if k not in slot_doc]
    if missing:
        raise ThresholdRegistrySchemaError(f"registry.slot missing keys: {missing}")
    for key, required in (("objective", _REQUIRED_OBJECTIVE_KEYS),
                          ("scoring", _REQUIRED_SCORING_KEYS)):
        gap = [k for k in required if k not in slot_doc[key]]
        if gap:
            raise ThresholdRegistrySchemaError(f"registry.slot.{key} missing keys: {gap}")
    if slot_doc["champion"] not in slot_doc["arms"]:
        raise ThresholdRegistrySchemaError(
            f"champion {slot_doc['champion']!r} is not in arms {slot_doc['arms']!r} — "
            f"champion-challenger §3: a leaderboard with a vacant champion is broken"
        )

    slot = SlotSpec(
        id=slot_doc["id"],
        champion=slot_doc["champion"],
        arms=tuple(slot_doc["arms"]),
        objective=dict(slot_doc["objective"]),
        scoring=dict(slot_doc["scoring"]),
        hysteresis=dict(slot_doc["hysteresis"]),
        count_matching=dict(slot_doc["count_matching"]),
        retention=dict(slot_doc["retention"]),
    )

    # champion-challenger §7.1 — a measurement whose horizon exceeds the
    # retention of the data it reads is structurally impossible and writes
    # empty artifacts forever. Asserted here, at load, with a named error.
    horizon_cards = slot.horizon_cycles + slot.n_floor_cards
    if slot.cohort_max_cards < horizon_cards:
        raise ThresholdRegistrySchemaError(
            f"scoring.cohort_max_cards ({slot.cohort_max_cards}) < horizon_cycles + "
            f"n_floor_cards ({horizon_cards}) — the cohort can never reach its own floor "
            f"(champion-challenger §7.1)"
        )
    expiry_days = slot.retention.get("expiry_days")
    if expiry_days is not None:
        span_days = (slot.cohort_max_cards + slot.horizon_cycles) * int(
            slot.retention.get("cycle_days", 7)
        )
        if span_days > int(expiry_days):
            raise ThresholdRegistrySchemaError(
                f"scoring window spans {span_days}d of {slot.retention['source_prefix']!r} "
                f"history but that prefix expires at {expiry_days}d — the measurement is "
                f"structurally impossible (champion-challenger §7.1)"
            )

    shared_raw = doc.get("shared_bands") or {}
    if not isinstance(shared_raw, dict):
        raise ThresholdRegistrySchemaError("registry.shared_bands must be a mapping")
    if DEFAULT_BAND in shared_raw:
        raise ThresholdRegistrySchemaError(
            f"shared_bands may not redefine the reserved band name {DEFAULT_BAND!r}"
        )
    shared: dict[str, tuple[float | None, float | None]] = {}
    for band_name, spec in shared_raw.items():
        if not isinstance(spec, dict) or set(spec) - {"target", "red_line", "note"}:
            raise ThresholdRegistrySchemaError(
                f"shared_bands.{band_name} must carry only target/red_line/note"
            )
        shared[band_name] = (spec.get("target"), spec.get("red_line"))

    metrics = doc.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ThresholdRegistrySchemaError("registry.metrics must be a non-empty mapping")
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for module, module_rows in metrics.items():
        if not isinstance(module_rows, dict) or not module_rows:
            raise ThresholdRegistrySchemaError(f"registry.metrics.{module} must be a mapping")
        for name, row in module_rows.items():
            if not isinstance(row, dict):
                raise ThresholdRegistrySchemaError(
                    f"registry.metrics.{module}.{name} must be a mapping"
                )
            unknown = set(row) - _ALLOWED_ROW_KEYS
            if unknown:
                raise ThresholdRegistrySchemaError(
                    f"registry.metrics.{module}.{name} has unknown keys {sorted(unknown)}"
                )
            if row.get("dynamic") and (row.get("target") is not None
                                       or row.get("red_line") is not None):
                raise ThresholdRegistrySchemaError(
                    f"registry.metrics.{module}.{name} is dynamic — its bands come from "
                    f"the data and must not also be declared here"
                )
            surface = row.get("surface_tile")
            if surface is not None and (not isinstance(surface, str) or not surface):
                raise ThresholdRegistrySchemaError(
                    f"registry.metrics.{module}.{name}.surface_tile must be a "
                    f"non-empty string naming the card tile this row renders on"
                )
            if row.get("dynamic") and not row.get("note"):
                raise ThresholdRegistrySchemaError(
                    f"registry.metrics.{module}.{name} is dynamic and must carry a note "
                    f"naming where its band comes from"
                )
            rows[(module, name)] = row

    derived_tiles = frozenset(
        row.get("surface_tile") or module for (module, _), row in rows.items()
    )
    card = parse_card(doc, derived_tiles)

    # NOTE: `card.component_weights` keys are the v1 composite's SECTION names
    # (`meta_model`, `scanner`, ...), a different namespace from a v2 tile's
    # component roster — so they are deliberately NOT checked against the rows.
    # The reconciliation that matters is on the TILES, above.

    return ThresholdRegistry(slot=slot, shared_bands=shared, rows=rows, card=card)


@functools.cache
def load_registry(path: Path | None = None) -> ThresholdRegistry:
    """Load + validate the committed registry (cached)."""
    p = path or REGISTRY_PATH
    with open(p, encoding="utf-8") as fh:
        return parse_registry(yaml.safe_load(fh))


def resolve(module: str, name: str, band: str = DEFAULT_BAND) -> Band:
    """Resolve one metric's champion bands. Raises on an unregistered metric."""
    return load_registry().resolve(module, name, band)


def tile_roster(tile: str) -> frozenset[str]:
    """Every component name the card tile ``tile`` is obliged to render.

    Raises ``ThresholdRegistryError`` on a tile the registry does not know.
    A tile whose roster cannot be named cannot be graded against one, and
    falling back to "whatever was handed in" is the exact defect
    ``alpha-engine-config-I9612`` removes — so this is loud, never empty.
    """
    rosters = load_registry().tile_rosters()
    try:
        return rosters[tile]
    except KeyError:
        raise ThresholdRegistryError(
            f"no threshold registry rows surface on tile {tile!r} — known tiles: "
            f"{sorted(rosters)}. Every card tile grades against its DECLARED "
            f"roster (alpha-engine-config-I9612); add the tile's rows to "
            f"{REGISTRY_PATH.name}, or declare `surface_tile: {tile}` on the "
            f"rows that render there."
        ) from None


def card_spec() -> CardSpec:
    """The committed card declaration — population, order, bands and weights."""
    return load_registry().card


def declared_tiles() -> tuple[str, ...]:
    """Every card tile, in declared build/publication order.

    ``grading/aggregate.py`` builds exactly these, and
    ``grading/module_agg.py``/``grading/scorecard.py`` draw their cascade set
    and weight tables from the same declaration (alpha-engine-config-I9734).
    """
    return load_registry().card.tiles
