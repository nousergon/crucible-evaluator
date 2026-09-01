"""
module_agg.py — hierarchical aggregation for the System Report Card v2.

Pure functions over ``list[MetricRecord]`` implementing RC v2 Principles 3
(aggregation respects critical gates) + 4 (BH-FDR at the module layer):

  component statuses → module_status   (critical-gate rule, not weighted avg)
  module statuses    → overall_status  (worst-of, portfolio outcome leads)
  components         → numeric_grade   (legacy 0-100 compat)

The grade is NEVER a plain weighted average of letters: a single RED critical
component fails the module regardless of how green everything else is, and a
module cannot claim GREEN while a critical component is unimplemented. This is
the institutional rule the v1 surface lacked (which is how it floated a C+
overall while critical executor tiles were N/A).

Authoritative: ``system-report-card-revamp-260522.md`` §"Aggregation methodology".
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from krepis.metrics import MetricRecord, StatusLiteral
from nousergon_lib.quant.stats.multiple_testing import benjamini_hochberg

# Same literal, same meaning as the card census's — imported rather than
# re-declared so the tile line and `grading/coverage.py` can never disagree
# about what an unreported component is called.
from grading.coverage import UNREPORTED_STATUS
from grading.thresholds.registry import card_spec, tile_roster

# Modules whose RED cascades to an overall RED (RC v2 module→overall rule).
#
# DECLARED at ``grading/thresholds/registry.yaml#card.tiles.<tile>.cascades``
# since ``alpha-engine-config-I9734``, not listed here. Same four modules; what
# changes is that this can no longer name a module the card does not have, or
# miss one it does — ``parse_registry`` reconciles the declared tiles against
# the tile set derived from the metric rows at load.
_CASCADE_MODULES: tuple[str, ...] = card_spec().cascade_modules

# Per-tile freshness stamps (config-I2556). Every component already threads a
# real S3 `source_path` (and a `last_updated_utc` construction timestamp)
# through `metric_record.build_metric` — this is genuine per-artifact
# provenance, not a guess. `_ARTIFACT_DATE_RE` mines any embedded
# ``YYYY-MM-DD`` segment out of a component's `source_path` (e.g.
# ``s3://bucket/backtest/2026-07-10/e2e_lift.json`` — and, via
# ``grading.artifacts.get_json_windowed``'s backward-walk, the REAL date the
# artifact was last written, not necessarily the run_date, when a tile fell
# back to an older instance). A `source_path` that points at a non-dated
# pointer artifact (e.g. ``predictor/metrics/latest.json``,
# ``signals/latest.json``) contributes no date — that is an honest gap in
# per-artifact dating, not an omission on our part.
_ARTIFACT_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

@dataclass(frozen=True)
class UnreportedComponent:
    """A roster member whose tile builder produced no record for it.

    ``alpha-engine-config-I9612``. ``build_tile`` used to derive a tile's
    status, letter, grade and ``n_components`` from the list its builder
    HANDED it — the "denominator is whatever reported" shape ``I8193`` removed
    one level up, at the card census. A builder that silently omitted a record
    (a conditional append, an artifact read that returned nothing) produced a
    tile with fewer components, and because the critical-gate ladder is
    worst-of over the records PRESENT, a dropped **critical** component could
    not turn its tile RED: the tile rendered GREEN/WATCH over the survivors,
    with ``n_components`` quietly one smaller and nothing saying so.

    A missing record is now materialised as one of these, from the tile's
    declared roster (``grading/thresholds/registry.py::tile_roster``). It is
    deliberately NOT a ``MetricRecord``: that contract requires a
    ``source_path``, a ``status_reason``, a ``last_updated_utc`` and a status
    from the closed ``StatusLiteral`` taxonomy — all things a component that
    produced nothing has not got. Inventing them would be the fabrication the
    record contract exists to prevent. It duck-types the attributes the
    aggregation functions in this module read, and nothing else.

    **Criticality is ``critical``, always.** The registry declares bands, not
    criticality — that lives at the tile call site, which is exactly the code
    that failed to run. So the criticality of an absent component is
    genuinely unknown, and the fail-loud reading of an unknown is the one that
    cannot flatter the tile: an unmeasured critical holds the tile at WATCH
    and can never let it claim GREEN (``principles.md`` §2.7 — *no data* is
    never rendered as green).
    """

    name: str
    module: str
    status: str = UNREPORTED_STATUS
    criticality: str = "critical"
    value: None = None
    target: None = None
    red_line: None = None
    bh_fdr_adjusted_p: None = None

    @property
    def is_na(self) -> bool:
        return True

    def model_dump(self, mode: str = "json") -> dict:
        """Render on the tile the way every other component does.

        ``unreported: True`` is the flag ``grading/coverage.py`` reads to keep
        the CARD census counting this component as declared-but-not-rendered
        (where ``I8193`` put it) rather than as a rendered record — the two
        surfaces must report the same fact once, not twice.
        """
        return {
            "name": self.name,
            "module": self.module,
            "status": self.status,
            "criticality": self.criticality,
            # Explicit nulls, not omissions: every renderer of a tile's
            # component list gets the same shape, and "there is no value" is
            # said rather than left to a missing key.
            "value": None,
            "unit": None,
            "n_samples": None,
            "n_floor": None,
            "ci_low": None,
            "ci_high": None,
            "target": None,
            "red_line": None,
            "trend_4w": None,
            "trend_13w": None,
            "derived_letter": "N/A",
            "unreported": True,
            "status_reason": (
                f"{self.name}: declared on the {self.module!r} tile's roster "
                f"(grading/thresholds/registry.yaml) but its tile builder "
                f"produced no record this cycle — the component did not report, "
                f"which is not the same as having nothing to report "
                f"(alpha-engine-config-I9612)."
            ),
            "na_detail": "declared component produced no record this cycle",
            "source_path": None,
            "last_updated_utc": None,
        }


def bh_fdr_significant(p_values: list[float], alpha: float = 0.05) -> bool:
    """True if BH-FDR finds any significant test among ``p_values`` at ``alpha``.

    Empty / all-None input → False (no evidence of joint underperformance).
    """
    ps = [p for p in p_values if p is not None]
    if not ps:
        return False
    return any(benjamini_hochberg(ps, alpha=alpha))


#: Which N/A class a wholly-unmeasured tile reports, when its components do not
#: all agree. Plurality first, this order as the tie-break: the reader is asking
#: "why is there nothing here", and a missing input is the most actionable
#: answer, an unbuilt component the next, and a tile that simply never produced
#: anything the least specific. Every member is from the card's closed N/A
#: taxonomy (``krepis.metrics.StatusLiteral``) — this introduces no new state.
_UNMEASURED_PRECEDENCE = (
    "N/A-MISSING-INPUT",
    "N/A-NOT-IMPL",
    "N/A-LOW-N",
    "N/A-NOT-RUN",
)


def unmeasured_status(components: list[MetricRecord]) -> StatusLiteral | None:
    """The N/A class for a tile in which NOTHING graded — else ``None``.

    ``observability-policy.md``: *a component emitting nothing is not healthy,
    it is unobserved*, and `principles.md` §2.7 forbids rendering *no data* as
    a measurement of any kind. ``module_status``'s critical-gate ladder was
    written for a tile with SOME measurement in it: a tile whose criticals are
    all N/A landed on ``WATCH`` (letter ``C``), which is the same status a tile
    that ran, measured every component and came out borderline gets. Measured
    2026-08-22: the ``agent`` tile was **11 N/A of 11** — an entire tile with
    not one number in it — and it rendered ``WATCH`` / ``C`` on the console,
    in the Director digest, and as a ``WATCH`` vote inside ``overall_status``,
    where two of them are enough to hold the whole card at WATCH. An
    unmeasured tile was manufacturing a measured verdict.

    The status returned is the class the components themselves declare, so a
    reader learns *why* the tile is empty from the tile line alone rather than
    having to open its components. It is derived from the records, never
    hand-listed per tile.
    """
    if not components:
        return None
    if any(not c.is_na for c in components):
        return None
    # An unreported roster member is not a DECLARATION by anyone — nothing ran
    # to say why it is absent. So it does not vote on which N/A class the tile
    # reports; the components that did declare a class own that answer, and
    # UNREPORTED is the residual when there are no such components at all.
    # Without this, one honest `N/A-MISSING-INPUT` sentinel on a tile whose
    # builder short-circuits (the predictor tile's both-inputs-absent path)
    # would be outvoted by the roster members its own short circuit produced,
    # and the tile would stop naming the reason it already knew.
    declared = [c for c in components if c.status != UNREPORTED_STATUS]
    if not declared:
        return UNREPORTED_STATUS  # type: ignore[return-value]
    counts = {s: 0 for s in _UNMEASURED_PRECEDENCE}
    for c in declared:
        if c.status in counts:
            counts[c.status] += 1
    top = max(counts.values())
    if not top:
        # Every component is N/A under a class this taxonomy does not name.
        # Loud rather than guessed: a fall-through is the thing the closed
        # vocabulary exists to remove (`observability-policy.md` §8.3).
        raise ValueError(
            "unmeasured tile carries N/A statuses outside the closed taxonomy: "
            + ", ".join(sorted({str(c.status) for c in declared})),
        )
    for status in _UNMEASURED_PRECEDENCE:
        if counts[status] == top:
            return status  # type: ignore[return-value]
    raise AssertionError("unreachable")  # pragma: no cover


def module_status(components: list[MetricRecord], *, alpha: float = 0.05) -> StatusLiteral:
    """Roll a tile's components up to a module status (RC v2 Principle 3).

    Order of precedence:
      RED   if any critical component is RED.
      RED   if ≥2 critical components are WATCH AND BH-FDR finds their joint
            underperformance significant.
      WATCH if any critical component is N/A-NOT-IMPL (can't claim GREEN with an
            unimplemented critical).
      WATCH if ≥2 critical WATCH (not BH-significant), or any critical WATCH, or
            any supporting RED.
      WATCH if any critical component is N/A-* (transparency); GREEN if only
            supporting/diagnostic are N/A.
      GREEN otherwise.

    Ahead of all of that: a tile in which NOTHING graded does not get a
    measured status at all (alpha-engine-config-I8177). See
    ``unmeasured_status``.
    """
    if not components:
        return "N/A-NOT-RUN"
    if (unmeasured := unmeasured_status(components)) is not None:
        return unmeasured

    critical = [c for c in components if c.criticality == "critical"]
    supporting = [c for c in components if c.criticality == "supporting"]

    crit_red = [c for c in critical if c.status == "RED"]
    crit_watch = [c for c in critical if c.status == "WATCH"]
    crit_not_impl = [c for c in critical if c.status == "N/A-NOT-IMPL"]
    crit_na = [c for c in critical if c.is_na]
    sup_red = [c for c in supporting if c.status == "RED"]

    if crit_red:
        return "RED"
    if len(crit_watch) >= 2 and bh_fdr_significant(
        [c.bh_fdr_adjusted_p for c in crit_watch], alpha=alpha
    ):
        return "RED"
    if crit_not_impl:
        return "WATCH"
    if crit_watch or sup_red:
        return "WATCH"
    if crit_na:
        return "WATCH"
    if any(c.is_na for c in components):
        # Only supporting/diagnostic N/A remain — doesn't block GREEN.
        return "GREEN"
    return "GREEN"


def overall_status(tiles: dict[str, StatusLiteral]) -> StatusLiteral:
    """Roll module statuses to an overall status (RC v2 module→overall).

    Portfolio outcome leads (the system exists to produce alpha); a RED in any
    cascade module (research/predictor/executor/substrate) also fails overall.
    The lead tile being N/A holds the overall at WATCH — the same
    never-a-false-GREEN rule ``module_status`` applies to critical components
    (trust-battery fix, config#1958: previously an ungraded portfolio_outcome
    let the overall claim GREEN off the remaining tiles alone).
    """
    if not tiles:
        return "N/A-NOT-RUN"
    if all(s.startswith("N/A") for s in tiles.values()):
        return "N/A-NOT-RUN"
    if tiles.get("portfolio_outcome") == "RED":
        return "RED"
    if any(tiles.get(m) == "RED" for m in _CASCADE_MODULES):
        return "RED"
    n_watch = sum(1 for s in tiles.values() if s == "WATCH")
    if tiles.get("portfolio_outcome") == "WATCH" or n_watch >= 2:
        return "WATCH"
    if (tiles.get("portfolio_outcome") or "N/A").startswith("N/A"):
        return "WATCH"
    # A cascade module that is wholly UNMEASURED cannot let the card claim
    # GREEN either (alpha-engine-config-I8177). Before `unmeasured_status`, a
    # tile with nothing in it voted WATCH and was counted above; now it votes
    # N/A, and without this clause the card would get GREENER the more of
    # itself went dark — the exact inversion `principles.md` §2.7 forbids.
    # Only modules PRESENT in the mapping are judged: a key absent from the
    # roll-up is a tile that is not on this card at all, which is a census
    # question (`grading/coverage.py`), not a status one. Conflating the two
    # would make this function's verdict depend on how complete its caller's
    # dict happened to be.
    if any(
        str(tiles[m]).startswith("N/A") for m in _CASCADE_MODULES if m in tiles
    ):
        return "WATCH"
    return "GREEN"


def _component_score(c: MetricRecord) -> float | None:
    """Map one component to a 0-100 score for the legacy numeric grade.

    Not a metric-specific calibration (that lived in v1's ``_*_to_grade``) — a
    uniform status+position mapping so the 0-100 stays comparable across tiles:
      - N/A-* or diagnostic            → excluded (None)
      - RED                            → 15 (capped at the red-line band)
      - position of value within [red_line, target] → [40, 90], clamped [0,100];
        GREEN beyond target can reach 100, WATCH below target floors at 40.
    Excluding N/A-NOT-IMPL (rather than averaging a neutral score) is the fix for
    the v1 inflation where unimplemented criticals propped the overall up.
    """
    if c.is_na or c.criticality == "diagnostic":
        return None
    if c.status == "RED":
        return 15.0
    if c.value is None:
        return None
    if c.target is None or c.red_line is None or c.target == c.red_line:
        return 90.0 if c.status == "GREEN" else 55.0

    higher_is_better = c.target >= c.red_line
    # Normalize value position from red_line(0.0) → target(1.0).
    span = c.target - c.red_line
    frac = (c.value - c.red_line) / span if higher_is_better else (c.red_line - c.value) / (-span)
    score = 40.0 + frac * 50.0
    return max(0.0, min(100.0, score))


def numeric_grade(components: list[MetricRecord]) -> float | None:
    """Legacy 0-100 grade: mean of per-component scores (RC v2 numeric-compat).

    N/A and diagnostic components are excluded; RED criticals drag via their
    capped 15. None when no component is scorable.
    """
    scores = [s for c in components if (s := _component_score(c)) is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)


def build_tile(
    module: str, components: list[MetricRecord], *,
    alpha: float = 0.05,
    staleness: dict | None = None,
    roster: Iterable[str] | None = None,
) -> dict:
    """Assemble a tile summary, graded against the tile's DECLARED roster.

    ``alpha-engine-config-I9612``. The roster is
    ``grading/thresholds/registry.py::tile_roster(module)`` — the same
    committed rows ``grading/coverage.py`` uses as the card-level denominator,
    partitioned by each row's ``surface_tile`` (its owning module unless the
    row declares otherwise, which is how the ``*_contribution_lift`` family
    surfaces on its own tile). No second registry, and nothing hand-listed
    per tile: a component the card can grade must have a row, because
    ``build_metric`` raises without one.

    Every roster member with no record among ``components`` is materialised as
    an ``UnreportedComponent`` — counted in ``n_components``, named on the
    tile, and treated by the critical gate as an unmeasured critical. So a
    builder that drops a record can no longer shrink its own denominator, and
    a dropped critical can no longer leave the tile GREEN over the survivors.

    ``roster`` overrides the registry lookup. It exists for tests that pin the
    grading rules against synthetic components without also pinning the live
    registry's contents; production passes nothing. Passing an explicit empty
    roster is the one way to get the pre-I9612 "grade whatever was handed in"
    behaviour, and it is deliberately something a caller has to write down.

    Adds the RC v2 per-tile freshness stamps (config-I2556):
      - ``as_of``: ISO UTC time this tile finished computing — the max of its
        components' ``last_updated_utc`` (each stamped ``datetime.now(UTC)`` at
        ``build_metric`` construction time), so tiles that legitimately update
        on different cadences each carry their own honest timestamp rather than
        one card-global build time.
      - ``source_artifact_dates``: the distinct ``YYYY-MM-DD`` dates mined from
        each component's real ``source_path`` (see ``_ARTIFACT_DATE_RE`` above)
        — genuine per-tile attribution derived from data every tile builder
        already threads through ``build_metric``, not a guessed/global rollup.

    ``staleness`` (config#2885): optional per-tile staleness summary from the
    tile builder's ``grading.artifacts.StalenessRegistry.summary()``. When
    provided, adds ``stale_artifact_count`` and ``max_artifact_age_days`` to
    the tile dict so the report card's top-level ``degraded_staleness`` flag
    can be derived.
    """
    from krepis.metrics import derive_letter

    declared = frozenset(roster) if roster is not None else tile_roster(module)
    present = {c.name for c in components}
    unreported = [
        UnreportedComponent(name=name, module=module)
        for name in sorted(declared - present)
    ]
    graded_against = [*components, *unreported]

    status = module_status(graded_against, alpha=alpha)
    dumped = [c.model_dump(mode="json") for c in graded_against]

    stamps = [d for c in dumped if (d := c.get("last_updated_utc"))]
    as_of = max(stamps) if stamps else datetime.now(UTC).isoformat()

    source_artifact_dates = sorted({
        m.group(0)
        for c in dumped
        if (m := _ARTIFACT_DATE_RE.search(c.get("source_path") or ""))
    })

    tile: dict = {
        "module": module,
        "status": status,
        "letter": derive_letter(status),
        "numeric_grade": numeric_grade(graded_against),
        # The tile's DECLARED roster size, not the length of the list its
        # builder handed in (alpha-engine-config-I9612). These differ exactly
        # when a component went unreported, and `unreported` below names which.
        "n_components": len(graded_against),
        # The denominator behind this tile's status, on the tile line itself.
        # A reader (and the card census) can see "11 components, 0 graded"
        # without opening the component list (alpha-engine-config-I8177).
        "n_graded": sum(1 for c in graded_against if not c.is_na),
        # Named, never merely counted: a denominator a reader cannot
        # reconstruct is how the previous number survived unquestioned.
        "unreported": [c.name for c in unreported],
        "as_of": as_of,
        "source_artifact_dates": source_artifact_dates,
        "components": dumped,
    }
    if staleness is not None:
        tile["stale_artifact_count"] = staleness.get("stale_artifact_count", 0)
        tile["max_artifact_age_days"] = staleness.get("max_artifact_age_days")
        tile["any_stale"] = staleness.get("any_stale", False)
    return tile
