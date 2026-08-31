"""``evaluator_coverage`` over the card's OWN component census.

alpha-engine-config-I8177.

``evaluator_coverage`` is the anti-"insufficient data" meta-metric: it exists
to measure the very cliff the report card was built to close. It was computed
by ``grading/tiles/backtester.py::_coverage`` over
``backtest/{date}/grading.json`` — the **legacy v1 grading artifact**, whose
leaf set is 14 components across ``research`` / ``predictor`` / ``executor``
only.

Measured 2026-08-22:

* rendered: ``evaluator_coverage = 0.857`` (12/14) → WATCH, comfortably above
  the 0.8 red-line.
* actual RC v3 surface: **78 of 125 leaf components graded = 0.624** — below
  the red-line.

The v1 denominator cannot see ``agent`` (11 components, 11 N/A),
``substrate``, ``behavioral``, ``contribution_lift``, ``portfolio_outcome``,
``director_quality`` or ``backtester`` at all — seven of the ten tiles. A true
number about a smaller world than its name implied: the metric named for the
coverage gap was structurally incapable of seeing it.

DESIGN — the denominator is DERIVED, never hand-listed
------------------------------------------------------
``observability-policy.md`` §2.2: *"Coverage is derived from a registry, never
hand-listed. A hand-maintained monitored-things list drifts, and its drift is
invisible because the missing rows produce no signal."*

So this module carries no list of components of its own. The denominator is
``grading/thresholds/registry.yaml`` — the registry that already holds one row
per ``(module, metric)`` the card emits, and which ``build_metric`` RAISES
against when a component has no row. Because nothing can be graded without a
row, the registry is by construction a superset of what the tiles can emit,
which is the one property a coverage denominator needs and the reason no
second registry was created (``alpha-engine-config-I8193``).

Counting the components that HAPPENED TO RENDER — what this module did until
I8193 — is the same defect one level out: a tile that fails, empties, or is
dropped from ``aggregate.py``'s ``tiles`` dict left the denominator with it,
so coverage went UP when a tile disappeared. A registered component that
rendered nothing is now counted as ``N/A-UNREPORTED`` and named.

Against that roster, one exclusion rule applies:

    A component leaves the denominator iff it is DECLARED out —
    ``permanent_na is True``, i.e. it already carries a written
    ``permanent_na_reason`` on the card itself.

That is the same field ``build_metric`` sets when a caller passes
``permanent_na_reason``, and it is already rendered on every card, so the
denominator a reader can reconstruct from the artifact equals the one used to
compute the number. Nothing is retired by editing this file; a component is
retired by its own tile declaring why, in the record, where a reader sees it.

The consequences of that one rule, on the 2026-08-22 card:

* the five research components retired with the six-team/CIO graph
  (2026-07-12, ``config#1580`` / ``alpha-engine-config-I2993``, Brian ruling
  ``alpha-engine-config-I7210`` decision 1), ``executor.position_sizing``, and
  the three substrate metrics declared not-building
  (``alert_noise_ratio`` / ``changelog_coverage`` / ``iam_drift``) — nine
  components already carrying ``permanent_na: true`` — leave the denominator.
* the nine ``contribution_lift`` components whose producer declares them
  ``N/A-RETIRED`` or ``N/A-NOT-LIFT-SHAPED`` leave it too, once
  ``grading/tiles/contribution_lift.py`` stops remapping those declared
  lifecycle states onto a bare ``N/A-NOT-IMPL`` (same change set).

Neither is a row deleted to flatter the number. Both are states a producer or
a prior ruling had already declared, which the grading layer was reporting as
undeclared gaps — assertions that could never pass, permanently holding
coverage below 1.0 no matter what was built.

``N/A-LOW-N`` deliberately STAYS in the denominator: a component that is
legitimately accumulating samples is not yet measured, and pretending
otherwise is exactly the "no data rendered as neither pass nor fail" failure
this metric exists to catch (``principles.md`` §2.7).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: The coverage record cannot count itself — it is always present and always
#: graded, so including it would inflate every card by one.
SELF_COMPONENT = "evaluator_coverage"

#: Statuses that mean "this component has no real grade this cycle".
_NA_PREFIX = "N/A"


def _is_declared_out(component: dict) -> bool:
    """True when the card itself declares this component permanently N/A.

    Reads the rendered record, not a side table, so the exclusion is visible
    to any reader of the artifact.
    """
    return bool(component.get("permanent_na"))


def _is_graded(component: dict) -> bool:
    status = component.get("status") or ""
    return not str(status).startswith(_NA_PREFIX)


#: The status a REGISTERED component that produced no record on the card
#: carries in the census. `observability-policy.md` §8.3's loud fall-through:
#: a thing that was supposed to report and did not is UNREPORTED — never
#: absent, and never quietly out of the denominator.
UNREPORTED_STATUS = "N/A-UNREPORTED"


def declared_component_names() -> frozenset[str]:
    """Every component the card is CONTRACTUALLY obliged to render.

    The denominator's registry (``alpha-engine-config-I8193``). Not a second
    registry: ``grading/thresholds/registry.yaml`` already carries one row per
    ``(module, metric)`` the card emits, and ``metric_record.build_metric``
    RAISES ``ThresholdRegistryError`` on a component without one — so the
    registry is, by construction, a superset of what any tile can emit. That
    makes it the only list in this repo that cannot silently be missing a row
    for something the card grades, which is exactly the property a coverage
    denominator needs.

    Names are flat, not ``(module, name)``, on purpose: the registry keys a
    row by its OWNING module (a ``*_contribution_lift`` metric is registered
    under ``research``/``predictor``/``executor``/``behavioral`` because that
    is whose contribution it measures), while the card renders those same
    records on the ``contribution_lift`` tile. Component names are globally
    unique across the registry — verified, and pinned by a test — so a flat
    set compares cleanly against either partition.

    Raises if the registry cannot be loaded. The caller
    (``replace_evaluator_coverage``) turns that into a VISIBLE N/A record
    rather than a number over an unknown population.
    """
    from grading.thresholds.registry import load_registry

    return frozenset(name for _module, name in load_registry().rows)


def _declared_modules() -> dict[str, str]:
    """``name -> owning module``, for attributing an UNREPORTED component.

    An unreported component has no record, so the card cannot say which tile
    it belongs to; the registry can.
    """
    from grading.thresholds.registry import load_registry

    out: dict[str, str] = {}
    for module, name in sorted(load_registry().rows):
        out.setdefault(name, module)
    return out


def card_component_census(
    tiles: dict[str, Any],
    *,
    declared: frozenset[str] | set[str] | None = None,
    declared_modules: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Count graded vs gradable leaf components against the DECLARED roster.

    ``alpha-engine-config-I8193``. The first version of this function counted
    the components that were *present* on the card. That is the same defect it
    was written to fix, one level out: a tile builder that fails, returns
    empty, or is dropped from ``aggregate.py``'s ``tiles`` dict SHRANK the
    denominator, so the coverage number went **up** when a tile disappeared. A
    coverage metric whose denominator is "whatever reported" cannot see the
    only failure it exists to detect.

    The denominator is now the registry roster (``declared_component_names``),
    unioned with anything actually rendered so nothing can fall out either
    way:

    * **rendered and registered** — graded / ungraded by its own status, and
      excluded only if the record itself declares ``permanent_na``.
    * **registered but not rendered** — counted in the denominator as
      ``N/A-UNREPORTED`` and named in ``unreported``. It cannot be excluded as
      declared-out, because a component that produced no record produced no
      declaration either: a vanished tile takes its permanent-N/A exclusions
      with it and coverage falls further, which is the correct direction.
    * **rendered but not registered** — impossible today (``build_metric``
      raises first) and therefore counted, not dropped, and named in
      ``unregistered`` so the impossible case is loud if it ever happens.

    ``declared`` / ``declared_modules`` are injectable for tests; production
    always reads the registry.

    Returns the numbers AND the attribution a reader needs to reconstruct
    them. A bare fraction over an unnamed denominator is how the previous
    number survived unquestioned.
    """
    if declared is None:
        declared = declared_component_names()
    if declared_modules is None:
        declared_modules = _declared_modules() if declared else {}

    rendered: dict[str, tuple[str, dict]] = {}
    duplicated: list[str] = []
    for tile_name, tile in sorted(tiles.items()):
        if not isinstance(tile, dict):
            continue
        for component in tile.get("components") or []:
            if not isinstance(component, dict):
                continue
            name = component.get("name") or "?"
            if name in rendered:
                duplicated.append(f"{rendered[name][0]}.{name} / {tile_name}.{name}")
                continue
            rendered[name] = (tile_name, component)

    declared_names = set(declared) - {SELF_COMPONENT}
    rendered_names = set(rendered) - {SELF_COMPONENT}

    total = 0
    graded = 0
    declared_out: list[str] = []
    declared_out_detail: list[dict[str, Any]] = []
    ungraded: list[str] = []
    unreported: list[str] = []
    per_tile: dict[str, dict[str, int]] = {}

    def _bump(tile_name: str, *, is_graded: bool) -> None:
        counts = per_tile.setdefault(tile_name, {"graded": 0, "total": 0})
        counts["total"] += 1
        if is_graded:
            counts["graded"] += 1

    for name in sorted(declared_names | rendered_names):
        entry = rendered.get(name)
        if entry is None:
            # Registered, nothing rendered it. In the denominator, ungraded.
            module = declared_modules.get(name, "?")
            total += 1
            unreported.append(f"{module}.{name}")
            ungraded.append(f"{module}.{name} [{UNREPORTED_STATUS}]")
            _bump(module, is_graded=False)
            continue
        tile_name, component = entry
        qualified = f"{tile_name}.{name}"
        if _is_declared_out(component):
            declared_out.append(qualified)
            # Every exclusion carries its WRITTEN reason on the artifact, not
            # just its name. A denominator that shrinks by 23 rows owes the
            # reader 23 reasons in the place the number points at, or the
            # exclusion set is a claim nobody can check.
            declared_out_detail.append({
                "component": qualified,
                "status": component.get("status"),
                "reason": component.get("permanent_na_reason"),
            })
            continue
        total += 1
        is_graded = _is_graded(component)
        if is_graded:
            graded += 1
        else:
            ungraded.append(f"{qualified} [{component.get('status')}]")
        _bump(tile_name, is_graded=is_graded)

    return {
        "coverage": (graded / total) if total else None,
        "graded": graded,
        "total": total,
        "declared_out": sorted(declared_out),
        "declared_out_detail": sorted(
            declared_out_detail, key=lambda d: d["component"],
        ),
        "ungraded": sorted(ungraded),
        "per_tile": per_tile,
        # I8193 attribution. `declared_total` is the roster size the
        # denominator is built from; `rendered_total` is what the card
        # actually carried. They differ exactly when something is unreported
        # or unregistered, and both lists are named rather than counted.
        "declared_total": len(declared_names),
        "rendered_total": len(rendered_names),
        "unreported": sorted(unreported),
        "unregistered": sorted(rendered_names - declared_names),
        "duplicated": sorted(duplicated),
    }


#: Greppable recording surface for a census that could not be computed. Mirrors
#: ``grading/scorecard.py``'s ``COVERAGE_UNKNOWN_MARKER``: the same deviation
#: from fail-loud, the same three declarations. (a) The failure mode swallowed
#: is a defect in this reporting code; (b) the primary deliverable — every
#: graded component and the tile rollups — is produced before it runs and is
#: unaffected; (c) the recording surface is this string in the log AND an N/A
#: ``evaluator_coverage`` record on the artifact, so no reader sees a number.
CENSUS_UNKNOWN_MARKER = "evaluator_coverage_census_failed"


def _mark_coverage_unmeasured(tiles: dict[str, Any], exc: BaseException) -> None:
    """Render ``evaluator_coverage`` as a visible N/A after a census failure.

    Mutates the record in place rather than rebuilding it through
    ``build_metric``: we are already on the path where something in the normal
    construction raised, so the recovery must not depend on it.

    Best-effort by necessity — this is the handler of last resort — but every
    swallow here is one where the alternative is losing the whole card. The
    failure mode swallowed is "the card is malformed in a way this function
    also cannot navigate"; the recording surface is the ``logger.exception``
    above, which has already fired.
    """
    try:
        backtester = tiles.get("backtester")
        if not isinstance(backtester, dict):
            return
        for component in backtester.get("components") or []:
            if not isinstance(component, dict) or component.get("name") != SELF_COMPONENT:
                continue
            legacy_value = component.get("value")
            legacy_n = component.get("n_samples")
            reason = (
                f"evaluator_coverage: UNMEASURED — the card-wide component "
                f"census could not be computed "
                f"({type(exc).__name__}: {exc}). The legacy "
                f"backtest/{{date}}/grading.json value ({legacy_value}, n="
                f"{legacy_n}) is preserved under coverage_census for audit and "
                f"is NOT this card's coverage: it measures 14 leaves across "
                f"three of ten tiles (alpha-engine-config-I8177/I8193)."
            )
            component["value"] = None
            component["n_samples"] = None
            component["ci_low"] = None
            component["ci_high"] = None
            component["status"] = "N/A-MISSING-INPUT"
            component["status_reason"] = reason
            component["na_detail"] = reason
            component["derived_letter"] = None
            component["estimator"] = "card_component_census"
            component["source_path"] = "report_card.json#tiles[].components[]"
            component["coverage_census"] = {
                "error": f"{CENSUS_UNKNOWN_MARKER}: {type(exc).__name__}: {exc}",
                "legacy_grading_json_value": legacy_value,
                "legacy_grading_json_n": legacy_n,
            }
            break
        # An N/A critical component must move the tile it sits on. Re-derive
        # where possible; where the card is too malformed to revalidate, say
        # so on the tile rather than leaving a rollup that still reflects the
        # value we just withdrew.
        try:
            from grading.metric_record import MetricRecord
            from grading.module_agg import build_tile

            records = [
                MetricRecord.model_validate(c)
                for c in (backtester.get("components") or [])
            ]
            rebuilt = build_tile("backtester", records)
            for key in ("status", "letter", "numeric_grade", "n_components"):
                if key in rebuilt:
                    backtester[key] = rebuilt[key]
        except Exception:  # noqa: BLE001 — see docstring
            logger.exception(
                "%s: backtester tile rollup could not be re-derived after the "
                "census failure; the tile's status is stamped UNVERIFIED",
                CENSUS_UNKNOWN_MARKER,
            )
            backtester["rollup_unverified"] = CENSUS_UNKNOWN_MARKER
    except Exception:  # noqa: BLE001 — handler of last resort, see docstring
        logger.exception(
            "%s: could not even stamp the coverage record N/A", CENSUS_UNKNOWN_MARKER,
        )


def replace_evaluator_coverage(
    tiles: dict[str, Any],
    *,
    declared: frozenset[str] | set[str] | None = None,
    declared_modules: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Substitute the ``evaluator_coverage`` record with the real card census.

    ``declared`` / ``declared_modules`` override the registry roster; production
    passes neither. They exist so a test can pin the counting rules against a
    synthetic card without also pinning the live registry's contents.

    Mutates ``tiles`` in place and returns the census (or ``None`` when the
    record is absent — e.g. a tile builder that failed). Re-derives the
    backtester tile's rollup status afterwards, since ``evaluator_coverage``
    is a ``critical`` component and its status drives the tile.

    Never raises: a coverage-computation defect must not take down the card it
    is measuring. But it does not fail OPEN either (alpha-engine-config-I8193
    sweep): on failure the record is stamped N/A with the exception named, and
    the legacy 14-leaf value is preserved under ``coverage_census`` for audit
    rather than rendered as this card's coverage. Keeping the legacy number in
    place was the original defect wearing a different hat — a true number
    about a smaller world, on a card that no longer says so. A missing
    measurement is NULL and visible.
    """
    try:
        return _replace_evaluator_coverage(
            tiles, declared=declared, declared_modules=declared_modules,
        )
    except Exception as exc:  # noqa: BLE001 — see docstring
        logger.exception(
            "%s: evaluator_coverage recomputation over the card census "
            "failed; the record is rendered N/A rather than keeping its "
            "legacy grading.json value, which measures a DIFFERENT (14-leaf) "
            "surface (alpha-engine-config-I8177)",
            CENSUS_UNKNOWN_MARKER,
        )
        _mark_coverage_unmeasured(tiles, exc)
        return None


def _replace_evaluator_coverage(
    tiles: dict[str, Any],
    *,
    declared: frozenset[str] | set[str] | None = None,
    declared_modules: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    from grading.metric_record import build_metric
    from grading.module_agg import build_tile
    from grading.units import FRACTION

    backtester = tiles.get("backtester")
    if not isinstance(backtester, dict):
        return None
    components = backtester.get("components") or []
    index = next(
        (i for i, c in enumerate(components)
         if isinstance(c, dict) and c.get("name") == SELF_COMPONENT),
        None,
    )
    if index is None:
        return None

    census = card_component_census(
        tiles, declared=declared, declared_modules=declared_modules,
    )
    previous = components[index]

    if not census["total"]:
        record = build_metric(
            name=SELF_COMPONENT, module="backtester", metric_type="pct",
            criticality="critical", estimator="card_component_census",
            measurement_horizon="this_cycle", n_floor=1,
            source_path="report_card.json#tiles[].components[]",
            input_present=False,
            na_detail=(
                "evaluator_coverage: this card carries no gradable leaf "
                "components — every tile builder failed or returned empty."
            ),
        )
    else:
        record = build_metric(
            name=SELF_COMPONENT, module="backtester", metric_type="pct",
            criticality="critical", estimator="card_component_census",
            measurement_horizon="this_cycle",
            value=census["coverage"], unit=FRACTION,
            n_samples=census["total"], n_floor=1,
            target=0.95, red_line=0.80,
            source_path="report_card.json#tiles[].components[]",
            reason=coverage_reason(census),
            trend_4w=previous.get("trend_4w"),
            trend_13w=previous.get("trend_13w"),
        )

    # Validate the REBUILD before mutating anything. A tile carrying a record
    # this module cannot revalidate must leave the card exactly as it was —
    # a half-applied substitution is worse than a stale one, because the
    # tile rollup and the record it summarizes would then disagree.
    from grading.metric_record import MetricRecord

    records = [
        record if i == index else MetricRecord.model_validate(c)
        for i, c in enumerate(components)
    ]
    rebuilt = build_tile("backtester", records)

    dumped = record.model_dump(mode="json")
    # Attribution a reader can reconstruct the fraction from, per
    # `observability-policy.md` §2.2 — the previous number was a bare fraction
    # over an unnamed denominator, which is how it went unquestioned.
    dumped["coverage_census"] = {
        "graded": census["graded"],
        "total": census["total"],
        "per_tile": census["per_tile"],
        "declared_out": census["declared_out"],
        # alpha-engine-config-I8177 closes-when, in its DERIVED form: every
        # excluded component with the reason its own record declares.
        # `grading_weights.retired_components` is a hand-listed register of
        # components removed from a WEIGHT TABLE — three rows, a narrower
        # thing — and `coverage_reason` used to point readers at it for all 23
        # exclusions. Two readers of one namespace, disagreeing.
        "declared_out_detail": census["declared_out_detail"],
        "ungraded": census["ungraded"],
        # alpha-engine-config-I8193 — the denominator's provenance. `total` is
        # built from `declared_total` (the threshold registry roster), not from
        # `rendered_total`, so a tile that vanishes lowers this number instead
        # of raising it. `unreported` names exactly what went missing.
        "denominator_source": "grading/thresholds/registry.yaml#metrics",
        "declared_total": census["declared_total"],
        "rendered_total": census["rendered_total"],
        "unreported": census["unreported"],
        "unregistered": census["unregistered"],
        "duplicated": census["duplicated"],
        "legacy_grading_json_value": previous.get("value"),
        "legacy_grading_json_n": previous.get("n_samples"),
    }
    components[index] = dumped

    for key in ("status", "letter", "numeric_grade", "n_components"):
        if key in rebuilt:
            backtester[key] = rebuilt[key]
    backtester["components"] = components
    return census


def coverage_reason(census: dict[str, Any]) -> str:
    """The human-readable status_reason for the coverage record.

    Names the worst tile, because a card-level fraction hides a tile that is
    entirely unmeasured — the ``agent`` tile was 11 N/A of 11 while the card
    read 86%.
    """
    cov = census["coverage"]
    graded, total = census["graded"], census["total"]
    per_tile = census.get("per_tile") or {}
    worst = ""
    if per_tile:
        name, counts = min(
            per_tile.items(),
            key=lambda kv: (kv[1]["graded"] / kv[1]["total"], -kv[1]["total"]),
        )
        if counts["graded"] < counts["total"]:
            worst = (
                f" Worst tile: {name} "
                f"({counts['graded']}/{counts['total']} graded)."
            )
    n_out = len(census.get("declared_out") or [])
    out_note = (
        f" {n_out} component(s) excluded as declared-permanent-N/A, each "
        f"with its written reason in coverage_census.declared_out_detail."
        if n_out else ""
    )
    # A registered component that rendered nothing is the failure this
    # denominator exists to make visible — name it first, and name the members
    # (`observability-policy.md` §7.2a), never a bare count.
    unreported = census.get("unreported") or []
    shown = ", ".join(unreported[:10])
    if len(unreported) > 10:
        shown += f", +{len(unreported) - 10} more"
    unreported_note = (
        f" {len(unreported)} REGISTERED component(s) produced no record on "
        f"this card and are counted as {UNREPORTED_STATUS}: {shown}."
        if unreported else ""
    )
    unregistered = census.get("unregistered") or []
    unregistered_note = (
        f" {len(unregistered)} component(s) graded with no threshold-registry "
        f"row: {', '.join(unregistered[:10])}." if unregistered else ""
    )
    return (
        f"evaluator_coverage = {cov:.0%} ({graded}/{total} leaf components "
        f"graded, non-N/A, of the components declared in "
        f"grading/thresholds/registry.yaml) vs target 95% / red-line 80%."
        f"{worst}{out_note}{unreported_note}{unregistered_note}"
    )


# ---------------------------------------------------------------------------
# The same defect, one level UP: the v1 composite's own coverage block
# ---------------------------------------------------------------------------
#
# `evaluator_coverage` measured a 14-leaf legacy artifact and called it the
# card's coverage. The card's headline `overall` block does the identical
# thing with the identical shape, and had not been swept: it is the v1/v2
# composite over `research` / `predictor` / `executor` ONLY, and it reported
#
#     overall.coverage = {components_declared: 3, components_present: 3,
#                         qualifier: "COMPLETE"}
#
# on the 2026-08-22 card — the same card carrying `tiles_overall_status: RED`
# and 47 N/A across 125 leaf components. "COMPLETE" was TRUE of the three
# modules it declared and FALSE of the card it shipped on: seven of the ten
# tiles, portfolio_outcome (the product-outcome tile) among them, carry no
# weight in that composite at all.
#
# `engagement-protocol-policy.md` §5 — a fix survives the CLASS, not the
# instance. Fixing `evaluator_coverage` and leaving this is fixing one call
# site of a systemic defect.
#
# The composite's ARITHMETIC is not touched here. Its weights are a scoring
# rule Brian ruled on (`alpha-engine-config-I7210`), and re-weighting the
# headline grade over ten tiles is a scoring decision, not a reporting one —
# filed, not assumed. What is fixed is the claim the block makes ABOUT ITSELF:
# a coverage qualifier may not read COMPLETE while the artifact it is stamped
# on carries a component surface the composite cannot see.

#: Coverage qualifier for a composite that is complete over its own declared
#: scope while that scope is a strict subset of the card. Distinct from
#: ``PARTIAL`` on purpose: ``PARTIAL`` means declared weight went missing this
#: cycle (a run-quality fact that varies week to week), whereas this is a
#: standing structural fact about what the composite covers at all. Collapsing
#: them would make a scope gap look like a bad week and disappear the moment
#: every declared module happened to report.
PARTIAL_SCOPE = "PARTIAL-SCOPE"


#: Qualifier for a composite whose scope could not be established. Distinct
#: from ``PARTIAL-SCOPE`` (a MEASURED subset) — this is "we do not know what
#: this grade covers", and it must never be renderable as a bare letter.
SCOPE_UNKNOWN = "SCOPE-UNKNOWN"


def _mark_scope_unknown(scorecard: dict[str, Any], exc: BaseException) -> None:
    """Stamp ``overall.coverage`` unknown-scope after the scope stamp failed.

    Handler of last resort, same contract as ``_mark_coverage_unmeasured``:
    the log line has already fired, and the alternative to this swallow is
    losing the card.
    """
    try:
        overall = scorecard.get("overall")
        if not isinstance(overall, dict):
            return
        coverage = overall.get("coverage")
        if not isinstance(coverage, dict):
            return
        coverage["census_scope"] = {
            "error": f"{CENSUS_UNKNOWN_MARKER}: {type(exc).__name__}: {exc}",
            "note": (
                "The scope of the v1 composite could not be established on "
                "this card. Its coverage fields describe the three v1 modules "
                "only; the card-wide verdict is `tiles_overall_status` "
                "(alpha-engine-config-I8177)."
            ),
        }
        coverage["qualifier"] = SCOPE_UNKNOWN
        letter = overall.get("letter", "?")
        overall["display"] = (
            f"{letter} (SCOPE UNKNOWN — this grade covers the three v1 "
            f"modules; what it omits could not be determined this cycle. "
            f"Card-wide verdict is tiles_overall_status)"
        )
    except Exception:  # noqa: BLE001 — handler of last resort, see docstring
        logger.exception(
            "%s: could not even stamp overall.coverage %s",
            CENSUS_UNKNOWN_MARKER, SCOPE_UNKNOWN,
        )


def stamp_composite_scope(
    scorecard: dict[str, Any],
    tiles: dict[str, Any],
    *,
    declared: frozenset[str] | set[str] | None = None,
    declared_modules: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Declare, on the v1 ``overall`` block, what it does and does not cover.

    Adds ``overall.coverage.census_scope`` — the tiles inside the composite,
    the tiles outside it **with their statuses**, and the card-wide leaf
    census — and demotes a ``COMPLETE`` qualifier to ``PARTIAL-SCOPE`` while
    any tile sits outside. Rewrites ``overall.display`` so no surface renders
    the bare letter.

    Everything is DERIVED: in-scope is read from ``grading_weights.overall``
    (the composite's own declaration) and out-of-scope from the tiles actually
    on the card, so a tile added tomorrow lands in the out-of-scope list
    without anyone editing this file. There is no list of tile names here, for
    the same reason ``card_component_census`` carries no list of components.

    Never raises, for the reason in ``replace_evaluator_coverage``: a defect in
    a block that DESCRIBES the grade must not destroy the card carrying it.
    It does not fail OPEN either — on failure the qualifier is stamped
    ``SCOPE-UNKNOWN`` and ``display`` says so, because the state being
    withheld is precisely the one whose absence reads as "complete".
    Returns the scope block, or ``None`` when it could not be built.
    """
    try:
        return _stamp_composite_scope(
            scorecard, tiles, declared=declared, declared_modules=declared_modules,
        )
    except Exception as exc:  # noqa: BLE001 — see docstring
        logger.exception(
            "%s: composite scope stamp failed; overall.coverage is stamped "
            "%s rather than being left to render a COMPLETE that describes "
            "the three v1 modules and NOT this card's ten tiles "
            "(alpha-engine-config-I8177)",
            CENSUS_UNKNOWN_MARKER, SCOPE_UNKNOWN,
        )
        _mark_scope_unknown(scorecard, exc)
        return None


def _stamp_composite_scope(
    scorecard: dict[str, Any],
    tiles: dict[str, Any],
    *,
    declared: frozenset[str] | set[str] | None = None,
    declared_modules: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    from grading.scorecard import _display

    overall = scorecard.get("overall")
    if not isinstance(overall, dict):
        return None
    coverage = overall.get("coverage")
    if not isinstance(coverage, dict):
        return None

    # The composite's own declared TILE weights — deliberately not named
    # `declared`, which is the component roster this function forwards to the
    # census. Two different declarations, one of tiles and one of components;
    # collapsing their names once made the census read the weight table as its
    # roster and report a denominator of nine over a six-component card.
    declared_weights = (scorecard.get("grading_weights") or {}).get("overall") or {}
    in_scope = sorted(name for name in tiles if name in declared_weights)
    out_of_scope = sorted(name for name in tiles if name not in declared_weights)

    census = card_component_census(
        tiles, declared=declared, declared_modules=declared_modules,
    )
    in_leaves = sum(
        counts["total"] for name, counts in (census["per_tile"] or {}).items()
        if name in declared_weights
    )

    scope = {
        # What the composite grades.
        "tiles_in_scope": in_scope,
        # What it does not — with each one's verdict, so the omission is
        # readable as the finding it is rather than as a list of names
        # (`observability-policy.md` §7.2a: name the members).
        "tiles_out_of_scope": {
            name: (tiles[name] or {}).get("status") for name in out_of_scope
        },
        "tiles_on_card": len(tiles),
        # Leaf counts, so the "3 declared components" in this block cannot be
        # mistaken for the card's component surface again.
        "leaf_components_in_scope": in_leaves,
        "leaf_components_on_card": census["total"],
        "card_leaf_coverage": census["coverage"],
        "card_leaf_graded": census["graded"],
        "note": (
            "This composite grades "
            f"{len(in_scope)} of {len(tiles)} tiles on this card. Its "
            "coverage fields describe that scope ONLY; the card-wide "
            "component census is `tiles.backtester.components[] "
            "evaluator_coverage.coverage_census`, and the card-wide verdict "
            "is `tiles_overall_status` (alpha-engine-config-I8177)."
        ),
    }
    coverage["census_scope"] = scope

    if out_of_scope and coverage.get("qualifier") == "COMPLETE":
        coverage["qualifier"] = PARTIAL_SCOPE
    overall["display"] = _display(overall.get("letter", "?"), coverage)
    return scope
