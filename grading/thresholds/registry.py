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
class ThresholdRegistry:
    slot: SlotSpec
    shared_bands: dict[str, tuple[float | None, float | None]]
    rows: dict[tuple[str, str], dict[str, Any]]

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
        )

    def is_dynamic(self, module: str, name: str) -> bool:
        return bool(self.rows.get((module, name), {}).get("dynamic", False))

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
                               "n_floor_declared", "dynamic", "note"})


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
            if row.get("dynamic") and not row.get("note"):
                raise ThresholdRegistrySchemaError(
                    f"registry.metrics.{module}.{name} is dynamic and must carry a note "
                    f"naming where its band comes from"
                )
            rows[(module, name)] = row

    return ThresholdRegistry(slot=slot, shared_bands=shared, rows=rows)


@functools.cache
def load_registry(path: Path | None = None) -> ThresholdRegistry:
    """Load + validate the committed registry (cached)."""
    p = path or REGISTRY_PATH
    with open(p, encoding="utf-8") as fh:
        return parse_registry(yaml.safe_load(fh))


def resolve(module: str, name: str, band: str = DEFAULT_BAND) -> Band:
    """Resolve one metric's champion bands. Raises on an unregistered metric."""
    return load_registry().resolve(module, name, band)
