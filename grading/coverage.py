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

So this module does not carry a list of components. It walks the assembled
card and applies one exclusion rule:

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


def card_component_census(tiles: dict[str, Any]) -> dict[str, Any]:
    """Count graded vs gradable leaf components across every tile on the card.

    Returns a dict carrying the numbers AND the attribution a reader needs to
    reconstruct them — which components were excluded as declared-out, and
    which remain N/A inside the denominator. A bare fraction with no
    attribution is how the previous number survived unquestioned.
    """
    total = 0
    graded = 0
    declared_out: list[str] = []
    ungraded: list[str] = []
    per_tile: dict[str, dict[str, int]] = {}

    for tile_name, tile in sorted(tiles.items()):
        if not isinstance(tile, dict):
            continue
        t_total = t_graded = 0
        for component in tile.get("components") or []:
            if not isinstance(component, dict):
                continue
            name = component.get("name") or "?"
            if name == SELF_COMPONENT:
                continue
            qualified = f"{tile_name}.{name}"
            if _is_declared_out(component):
                declared_out.append(qualified)
                continue
            total += 1
            t_total += 1
            if _is_graded(component):
                graded += 1
                t_graded += 1
            else:
                ungraded.append(f"{qualified} [{component.get('status')}]")
        if t_total:
            per_tile[tile_name] = {"graded": t_graded, "total": t_total}

    return {
        "coverage": (graded / total) if total else None,
        "graded": graded,
        "total": total,
        "declared_out": sorted(declared_out),
        "ungraded": sorted(ungraded),
        "per_tile": per_tile,
    }


def replace_evaluator_coverage(tiles: dict[str, Any]) -> dict[str, Any] | None:
    """Substitute the ``evaluator_coverage`` record with the real card census.

    Mutates ``tiles`` in place and returns the census (or ``None`` when the
    record is absent — e.g. a tile builder that failed). Re-derives the
    backtester tile's rollup status afterwards, since ``evaluator_coverage``
    is a ``critical`` component and its status drives the tile.

    Never raises: a coverage-computation defect must not take down the card it
    is measuring. On failure the tile keeps the legacy record and the reason is
    logged loudly — an honest stale number beats a missing card.
    """
    try:
        return _replace_evaluator_coverage(tiles)
    except Exception:  # noqa: BLE001 — see docstring
        logger.exception(
            "evaluator_coverage recomputation over the card census failed; "
            "the record retains its legacy grading.json value, which is a "
            "measurement of a DIFFERENT (14-leaf) surface — treat it as "
            "unverified (alpha-engine-config-I8177)",
        )
        return None


def _replace_evaluator_coverage(tiles: dict[str, Any]) -> dict[str, Any] | None:
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

    census = card_component_census(tiles)
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
        "ungraded": census["ungraded"],
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
        f" {n_out} component(s) excluded as declared-permanent-N/A "
        f"(see grading_weights.retired_components)." if n_out else ""
    )
    return (
        f"evaluator_coverage = {cov:.0%} ({graded}/{total} leaf components "
        f"graded, non-N/A, across every tile on this card) vs target 95% / "
        f"red-line 80%.{worst}{out_note}"
    )
