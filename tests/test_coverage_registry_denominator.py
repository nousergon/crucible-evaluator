"""The coverage denominator is the DECLARED roster, not what happened to render.

alpha-engine-config-I8193, the written delta of `crucible-evaluator-PR254`.

`evaluator_coverage` was repointed off the 14-leaf legacy `grading.json` onto a
census of the card's own components. That fixed the surface but kept the shape:
the census counted the components **present on the card**, so a tile builder
that failed, emptied, or was dropped from `aggregate.py`'s `tiles` dict took its
components out of the DENOMINATOR with it — and the coverage number went **up**
when a tile disappeared. The one failure a coverage metric exists to detect was
the one failure it could not see.

The denominator is now `grading/thresholds/registry.yaml`. Not a second
registry: it already carries one row per `(module, metric)` the card emits, and
`build_metric` RAISES on a component without one, so it is by construction a
superset of what any tile can emit — the only list in this repo that cannot
silently be missing a row for something the card grades
(`observability-policy.md` §2.2).

These tests pin the three properties I8193's closes-when names.
"""

from __future__ import annotations

import pytest

from grading.coverage import (
    SELF_COMPONENT,
    UNREPORTED_STATUS,
    card_component_census,
    coverage_reason,
    declared_component_names,
)
from grading.thresholds.registry import load_registry


def _c(name, status="GREEN", *, permanent_na=False):
    return {"name": name, "status": status, "permanent_na": permanent_na}


# ---------------------------------------------------------------------------
# The roster itself
# ---------------------------------------------------------------------------

def test_the_roster_is_the_threshold_registry() -> None:
    """No second registry. The denominator IS the file that already raises."""
    rows = load_registry().rows
    assert declared_component_names() == {name for _module, name in rows}
    assert len(declared_component_names()) > 100


def test_component_names_are_globally_unique_across_modules() -> None:
    """The flat roster is only sound while a name belongs to ONE module.

    The registry keys a row by its OWNING module — a `*_contribution_lift`
    metric sits under research/predictor/executor/behavioral — while the card
    renders those same records on the `contribution_lift` tile. A flat set
    reconciles the two partitions, and only if names do not collide. If a
    future row reuses a name under a second module, this fails loudly here
    rather than quietly double-counting a denominator.
    """
    rows = load_registry().rows
    seen: dict[str, str] = {}
    collisions = []
    for module, name in sorted(rows):
        if name in seen:
            collisions.append(f"{name}: {seen[name]} and {module}")
        seen[name] = module
    assert not collisions, collisions


# ---------------------------------------------------------------------------
# Closes-when 1 — deleting a tile makes coverage go DOWN
# ---------------------------------------------------------------------------

def _full_card() -> dict:
    """Every registered component rendered GREEN, split across two tiles."""
    names = sorted(declared_component_names() - {SELF_COMPONENT})
    half = len(names) // 2
    return {
        "alpha": {"components": [_c(n) for n in names[:half]]},
        "beta": {"components": [_c(n) for n in names[half:]]},
    }


def test_a_complete_card_reads_full_coverage() -> None:
    census = card_component_census(_full_card())
    assert census["coverage"] == 1.0
    assert census["unreported"] == []
    assert census["unregistered"] == []
    assert census["total"] == census["declared_total"] == census["rendered_total"]


def test_deleting_a_tile_LOWERS_coverage() -> None:
    """The regression I8193 exists for, proven by doing exactly that.

    Under the rendered-components census this went to 1.0 — a deleted tile
    left the numerator and the denominator together.
    """
    tiles = _full_card()
    before = card_component_census(tiles)["coverage"]
    dropped = len(tiles["beta"]["components"])
    del tiles["beta"]
    after = card_component_census(tiles)
    assert before == 1.0
    assert after["coverage"] < before
    assert len(after["unreported"]) == dropped
    # The denominator did not move. That is the whole property.
    assert after["total"] == card_component_census(_full_card())["total"]


def test_an_empty_tile_LOWERS_coverage() -> None:
    """A builder that returns no components is the same defect, softer."""
    tiles = _full_card()
    dropped = len(tiles["beta"]["components"])
    tiles["beta"]["components"] = []
    census = card_component_census(tiles)
    assert census["coverage"] < 1.0
    assert len(census["unreported"]) == dropped


# ---------------------------------------------------------------------------
# Closes-when 2 — a component that did not report is UNREPORTED, not absent
# ---------------------------------------------------------------------------

def test_a_missing_component_is_unreported_and_ungraded() -> None:
    tiles = _full_card()
    gone = tiles["alpha"]["components"].pop()["name"]
    census = card_component_census(tiles)
    assert gone in census["unreported"][0] or any(
        c.endswith(f".{gone}") for c in census["unreported"]
    )
    assert any(f"{gone} [{UNREPORTED_STATUS}]" in u for u in census["ungraded"])


def test_an_unreported_component_cannot_be_excluded_as_declared_out() -> None:
    """A vanished tile takes its permanent-N/A exclusions with it.

    A record that was never rendered carries no `permanent_na` declaration, so
    it cannot leave the denominator — coverage falls FURTHER, which is the
    correct direction. Excluding it would let a tile improve the number by
    disappearing, which is the defect wearing the exclusion rule as a disguise.
    """
    tiles = _full_card()
    retired = tiles["alpha"]["components"][0]["name"]
    tiles["alpha"]["components"][0]["permanent_na"] = True
    with_record = card_component_census(tiles)
    assert with_record["declared_out"] == [f"alpha.{retired}"]

    tiles["alpha"]["components"].pop(0)
    without = card_component_census(tiles)
    assert without["declared_out"] == []
    assert without["total"] == with_record["total"] + 1
    assert f"research.{retired}" in without["unreported"] or any(
        u.endswith(f".{retired}") for u in without["unreported"]
    )


def test_the_reason_names_the_unreported_members() -> None:
    tiles = _full_card()
    gone = tiles["alpha"]["components"].pop()["name"]
    reason = coverage_reason(card_component_census(tiles))
    assert "REGISTERED" in reason
    assert UNREPORTED_STATUS in reason
    assert gone in reason


# ---------------------------------------------------------------------------
# Closes-when 3 — a rendered component with no registry row is never dropped
# ---------------------------------------------------------------------------

def test_an_unregistered_component_is_counted_and_named() -> None:
    """Impossible today (`build_metric` raises first) — loud if it happens."""
    tiles = _full_card()
    tiles["alpha"]["components"].append(_c("a_metric_with_no_registry_row"))
    census = card_component_census(tiles)
    assert census["unregistered"] == ["a_metric_with_no_registry_row"]
    assert census["total"] == card_component_census(_full_card())["total"] + 1


def test_the_self_component_is_still_never_counted() -> None:
    tiles = _full_card()
    tiles["alpha"]["components"].append(_c(SELF_COMPONENT))
    assert card_component_census(tiles)["coverage"] == 1.0


# ---------------------------------------------------------------------------
# The denominator is derived, not hand-listed
# ---------------------------------------------------------------------------

def test_no_component_name_is_hard_coded_in_the_census_module() -> None:
    """Same guard as before, restated against the new source.

    The roster moved from "the rendered records" to "the registry", and both
    are DERIVED. What must never appear is a literal component name in this
    module — that list drifts, and its drift is invisible because the missing
    rows produce no signal.
    """
    from pathlib import Path

    import grading.coverage as mod

    body = Path(mod.__file__).read_text().split('"""', 2)[-1]
    for name in sorted(declared_component_names()):
        if name == SELF_COMPONENT:
            continue  # the one deliberate literal: the metric cannot count itself
        assert name not in body, (
            f"grading/coverage.py names {name!r} — the denominator is the "
            "registry, never a list in this file"
        )


@pytest.mark.parametrize("bad", [None, "", 0])
def test_a_malformed_tile_never_shrinks_the_denominator(bad) -> None:
    tiles = _full_card()
    tiles["beta"] = bad
    census = card_component_census(tiles)
    assert census["total"] == card_component_census(_full_card())["total"]
    assert census["coverage"] < 1.0


# ---------------------------------------------------------------------------
# End to end — the rendered figure and the audited figure are one number
# ---------------------------------------------------------------------------

class TestOnAnAssembledCard:
    """The card the Lambda actually writes.

    The fleet has been bitten repeatedly by two readers of one namespace
    disagreeing, so these assert the identity rather than the arithmetic: the
    value `evaluator_coverage` renders IS `graded / total` of the census
    published beside it, and the census's denominator IS the registry roster.
    """

    def _card(self):
        import boto3
        from moto import mock_aws

        from grading.aggregate import build_report_card
        from test_aggregate import BUCKET, RUN_DATE, _put, _seed_freshness_inputs

        with mock_aws():
            s3 = boto3.client("s3", region_name="us-east-1")
            s3.create_bucket(Bucket=BUCKET)
            # The freshness preflight hard-fails a bucket missing its declared
            # inputs (by design) — the same minimum seed the producer-contract
            # test uses, reused rather than copied.
            _seed_freshness_inputs(s3)
            _put(s3, "metrics.json", {"run_date": RUN_DATE, "status": "ok"})
            _put(s3, "e2e_lift.json", {"status": "ok"})
            return build_report_card(BUCKET, RUN_DATE, s3_client=s3)

    def _record(self, card):
        return next(
            c for c in card["tiles"]["backtester"]["components"]
            if c["name"] == SELF_COMPONENT
        )

    def test_the_rendered_value_is_the_published_census(self):
        record = self._record(self._card())
        census = record["coverage_census"]
        if record["value"] is None:
            assert census["total"] == 0
            return
        assert record["value"] == pytest.approx(census["graded"] / census["total"])
        assert record["n_samples"] == census["total"]

    def test_the_denominator_names_its_source(self):
        census = self._record(self._card())["coverage_census"]
        assert census["denominator_source"] == "grading/thresholds/registry.yaml#metrics"
        assert census["declared_total"] == len(
            declared_component_names() - {SELF_COMPONENT}
        )

    def test_the_card_carries_the_census_health_at_the_top(self):
        """A component that did not report is a top-level fact, not a footnote.

        `false` is a VALUE: it asserts the roster and the card agreed. On this
        fixture almost every producer artifact is absent, so the tiles emit
        honest N/A records and the census is complete — what is asserted is
        that the flag EXISTS and agrees with the census beside it.
        """
        card = self._card()
        assert "degraded_component_census" in card
        census = self._record(card)["coverage_census"]
        expected = bool(census["unreported"] or census["unregistered"])
        assert card["degraded_component_census"] is expected
        if expected:
            assert card["component_census_unreported"] == census["unreported"]


# ---------------------------------------------------------------------------
# Every exclusion carries its written reason, where the number points
# ---------------------------------------------------------------------------

def test_every_exclusion_publishes_its_written_reason() -> None:
    """alpha-engine-config-I8177's closes-when, in its DERIVED form.

    The issue asked for `grading_weights.retired_components` to carry all 18
    retirements with a reason each. That register is hand-listed and means
    something narrower — components removed from a WEIGHT TABLE, three rows —
    while the live card excludes 23 components from the coverage denominator.
    `coverage_reason` pointed readers at the three-row register for all 23:
    two readers of one namespace, disagreeing.

    The register is now DERIVED from the records themselves, so it cannot
    drift from the exclusion set it describes: one entry per excluded
    component, carrying the `permanent_na_reason` that record declares.
    """
    tiles = _full_card()
    retired = tiles["alpha"]["components"][0]
    retired["permanent_na"] = True
    retired["status"] = "N/A-NOT-IMPL"
    retired["permanent_na_reason"] = "producer retired 2026-07-12 (config#1580)"

    census = card_component_census(tiles)
    assert census["declared_out"] == [f"alpha.{retired['name']}"]
    assert census["declared_out_detail"] == [{
        "component": f"alpha.{retired['name']}",
        "status": "N/A-NOT-IMPL",
        "reason": "producer retired 2026-07-12 (config#1580)",
    }]


def test_the_reason_points_at_the_register_it_publishes() -> None:
    tiles = _full_card()
    tiles["alpha"]["components"][0]["permanent_na"] = True
    reason = coverage_reason(card_component_census(tiles))
    assert "coverage_census.declared_out_detail" in reason
    assert "retired_components" not in reason


def test_an_exclusion_with_no_written_reason_is_visible_as_such() -> None:
    """A NULL reason is rendered, never omitted — the row still owes one."""
    tiles = _full_card()
    tiles["alpha"]["components"][0]["permanent_na"] = True
    detail = card_component_census(tiles)["declared_out_detail"]
    assert len(detail) == 1
    assert detail[0]["reason"] is None
