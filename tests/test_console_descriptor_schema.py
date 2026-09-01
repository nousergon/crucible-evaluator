"""Schema-contract test for `console.descriptor.yaml` (alpha-engine-config-I7477).

`console.descriptor.yaml` at the repo root is this component's onboarding onto
the fleet console (console-policy.md §2.6). This test is the producer-side
half of that contract: it loads the descriptor and validates it against a
vendored copy of nousergon-console's own schema, exactly as
`test_experiment_record_producer_contract.py` validates against
`nousergon_lib.contracts` — crucible-evaluator has no runtime dependency on
nousergon-console, so the schema is a fixture, not an import.

`tests/fixtures/component_descriptor.schema.json` is a copy of
`nousergon-console/console/schemas/component_descriptor.schema.json` as of
commit e0aa0e6c83740a9cab219914f5cdefbf6740e97f (2026-08-10, unchanged as of
this PR's re-check against nousergon-console main 2026-08-17). It is a copy,
not a symlink, because this is a public AGPL repo and nousergon-console's
working tree is not guaranteed present at test time; re-sync it by hand if
the upstream schema changes.

`tests/fixtures/console_known_drivers.json` is a vendored copy of
`nousergon-console/console/drivers/__init__.py::KNOWN_DRIVERS` as of the
`s3-records` (PR100) and `state-machine` (PR102) driver merges, 2026-08-17 —
this PR's whole point is consuming those two.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DESCRIPTOR_PATH = REPO_ROOT / "console.descriptor.yaml"
SCHEMA_PATH = REPO_ROOT / "tests" / "fixtures" / "component_descriptor.schema.json"
KNOWN_DRIVERS_PATH = REPO_ROOT / "tests" / "fixtures" / "console_known_drivers.json"


def _load_descriptor() -> dict:
    with open(DESCRIPTOR_PATH) as fh:
        return yaml.safe_load(fh)


def _load_schema() -> dict:
    with open(SCHEMA_PATH) as fh:
        return json.load(fh)


def _load_known_drivers() -> set[str]:
    with open(KNOWN_DRIVERS_PATH) as fh:
        return set(json.load(fh)["drivers"])


def _bindings(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def test_descriptor_validates_against_schema():
    descriptor = _load_descriptor()
    schema = _load_schema()
    jsonschema.validate(instance=descriptor, schema=schema)


def test_descriptor_declares_the_registry_component_id():
    descriptor = _load_descriptor()
    # Must match the id already declared in
    # nous-ergon-ops/governance/observability.d/alpha-engine-evaluator.yaml —
    # component_id is the one name every signal, log key, alert and console
    # row uses (console-policy.md §3.6), assigned by the component owner and
    # never re-minted by the console.
    assert descriptor["component_id"] == "alpha-engine-evaluator"


def test_descriptor_only_names_registered_drivers():
    """A descriptor naming a driver that does not exist FAILS THE BUILD
    (console-policy.md §2.7) rather than rendering the component absent — a
    typo and a genuinely-gone component must not look the same. This test
    pins every driver this descriptor names against a vendored copy of the
    console's own registry, so a driver rename/removal upstream is caught
    here rather than at the next live console build.
    """
    descriptor = _load_descriptor()
    known_drivers = _load_known_drivers()

    for key in ("runs", "artifacts", "metrics"):
        for binding in _bindings(descriptor.get(key)):
            assert binding["driver"] in known_drivers, (
                f"{key} binding names driver {binding.get('driver')!r}, "
                "which is not in the vendored console driver registry."
            )


def _s3_records_binding(descriptor: dict) -> dict:
    """The `s3-records` fan-out binding out of `metrics`.

    `metrics` became a LIST of two bindings at the full-row conversion
    (alpha-engine-config-I7477 deliverable 2): `document-fields` for the
    §2.3a correctness verdict (ported from the retired
    `nous-ergon-ops/governance/observability.d/alpha-engine-evaluator.yaml`
    row) plus the `s3-records` fan-out this branch already shipped. Selected
    by driver rather than by list position, so a reorder of the two bindings
    cannot silently break this test.
    """
    bindings = _bindings(descriptor["metrics"])
    matches = [b for b in bindings if b.get("driver") == "s3-records"]
    assert len(matches) == 1, f"expected exactly one s3-records metrics binding, found {len(matches)}"
    return matches[0]


def _document_fields_binding(descriptor: dict) -> dict:
    bindings = _bindings(descriptor["metrics"])
    matches = [b for b in bindings if b.get("driver") == "document-fields"]
    assert len(matches) == 1, f"expected exactly one document-fields metrics binding, found {len(matches)}"
    return matches[0]


def test_metrics_binding_uses_s3_records_fanout():
    """T4 deliverable 1: one Signal per (tile, component) report-card row,
    via the `s3-records` driver's fan-out grammar (nousergon-console#98,
    shipped PR100). Pins the shape so a future edit cannot silently drop the
    fan-out back to a single-entity binding.
    """
    descriptor = _load_descriptor()
    metrics = _s3_records_binding(descriptor)
    assert metrics["kind"] == "signal"
    assert metrics["records_path"] == "tiles.*.components"
    assert metrics["group_field"] == "tile"
    assert metrics["id_template"] == "{tile}:{name}"
    assert metrics["state_field"] == "status"
    assert "cadence_minutes" in metrics


def test_metrics_binding_carries_the_correctness_verdict():
    """Full-row conversion (I7477 deliverable 2): the §2.3a correctness
    verdict binding, previously declared only on the now-retired
    `nous-ergon-ops` registry row, is now carried by this descriptor —
    ported verbatim so the fact does not go dark between the two PRs'
    merges.
    """
    descriptor = _load_descriptor()
    doc_fields = _document_fields_binding(descriptor)
    docs = doc_fields["documents"]
    assert len(docs) == 1
    fields = docs[0]["fields"]
    assert fields["correctness_verdict"]["path"] == "attestation.verdict"
    assert fields["tiles_overall_status"]["path"] == "tiles_overall_status"
    assert "cadence_minutes" in doc_fields


def test_metrics_value_field_has_no_hardcoded_unit_or_baseline():
    """Every krepis.metrics.MetricRecord in the fan-out carries its OWN
    `unit` (krepis 0.59.11 / PR158) and its own `target` as the natural
    baseline — both per-record facts. `console/records_shape.py::build_fields`
    only supports a LITERAL `unit`/`baseline` on the binding's `fields` spec,
    with no per-record indirection (`unit_field`/`baseline_field`) as of the
    PR100/PR102 driver vintage this branch consumes. A literal here would be
    wrong for most rows of a fan-out whose whole point is heterogeneous
    records, so this test pins the deliberate omission — see
    console.descriptor.yaml's header comment and the nousergon-console issue
    filed alongside this PR for the missing capability.
    """
    descriptor = _load_descriptor()
    metrics = _s3_records_binding(descriptor)
    value_field = metrics["fields"]["value"]
    assert "unit" not in value_field
    assert "baseline" not in value_field
    # The record's own unit/target are still surfaced, just not wired into
    # `value`'s own descriptor metadata yet.
    assert metrics["fields"]["unit"]["path"] == "unit"
    assert metrics["fields"]["target"]["path"] == "target"


def test_runs_binding_is_deliberately_absent():
    """The evaluator's grading run is a Task state inside the SHARED weekly
    pipeline (`ne-weekly-freshness-pipeline`), not its own state machine.
    `state-machine` (driver) binds one whole machine's execution history with
    no stage filter — binding it here would make every stage of that shared
    pipeline (MorningEnrich, ResearchPredictorParallel, PredictorTraining)
    claim the evaluator's own run history as its own. See
    console.descriptor.yaml's header comment for the full reasoning; this
    test pins the resulting omission as deliberate, not an oversight.
    """
    descriptor = _load_descriptor()
    assert "runs" not in descriptor


# ---------------------------------------------------------------------------
# The economic surface (alpha-engine-config-I9005)
# ---------------------------------------------------------------------------
#
# `console-policy.md` §2.6 onboards a module by pointing at where its facts
# already live — which means a declared `path` that resolves to nothing renders
# the field as absent, silently, on the one surface a reader trusts. These
# tests are the producer-side half: every path the descriptor declares must
# actually exist on output THIS REPO EMITS.
#
# Deliberately built from the producers rather than a fixture. A fixture would
# drift from the artifact exactly the way a hand-maintained monitored-things
# list drifts (`observability-policy.md` §2.2), and its drift would be
# invisible for the same reason.


def _get_path(doc, dotted: str):
    """`console/records_shape.py::get_path`, reimplemented, not imported.

    Same reason the schema is vendored: this repo has no dependency on
    nousergon-console. Returns a `_MISSING` sentinel rather than None so a
    field whose real value IS None cannot pass as "present" or fail as
    "absent" — that conflation is the defect this test exists to catch.
    """
    cur = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


_MISSING = object()


def _document_fields_paths(descriptor: dict) -> dict[str, str]:
    docs = _document_fields_binding(descriptor)["documents"]
    assert len(docs) == 1
    return {
        name: spec["path"] for name, spec in docs[0]["fields"].items()
    }


def test_every_declared_economic_path_resolves_on_a_real_card():
    """The binding may not name a path the producer does not emit.

    Covers both halves of the I9005 surface: the outcome TILE's own roll-up
    (built here by the real Tile 0 builder) and the HEADLINE the outcome now
    votes in (built here by the real composite).
    """
    import boto3
    from moto import mock_aws

    from grading.scorecard import compute_scorecard
    from grading.tiles.portfolio_outcome import build_portfolio_outcome_tile

    # An EMPTY bucket, on purpose — the tile's own N/A-MISSING-INPUT path is
    # the HARDER case: a tile that measured nothing must still emit every key
    # the console binds, or an absent producer renders as an absent FIELD
    # instead of a loud N/A (`observability-policy.md` §8.3 — `no data` is
    # never rendered green, and it is never rendered as nothing either).
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="alpha-engine-research")
        tile = build_portfolio_outcome_tile("alpha-engine-research", s3_client=s3)

    card = compute_scorecard()
    card["tiles"] = {"portfolio_outcome": tile}
    assert str(tile["status"]).startswith("N/A"), tile["status"]

    declared = _document_fields_paths(_load_descriptor())
    economic = {
        name: path for name, path in declared.items()
        if path.startswith(("tiles.portfolio_outcome.", "overall.", "grading_weights."))
    }
    # Guards the filter itself: a rename that emptied this set would make the
    # loop below vacuously pass.
    assert len(economic) >= 9, economic

    missing = {
        name: path for name, path in economic.items()
        if _get_path(card, path) is _MISSING
    }
    assert not missing, (
        "console.descriptor.yaml declares paths this repo's producers do not "
        f"emit — the console would render these fields absent: {missing}"
    )


def test_the_headline_binding_takes_display_not_the_bare_letter():
    """`grading/scorecard.py::_display` exists so a partial or partial-scope
    grade never renders as a bare letter. A console binding on
    `overall.letter` would undo that on the surface most people read
    (config-I7202 deliverable 3)."""
    declared = _document_fields_paths(_load_descriptor())
    assert declared["system_grade_display"] == "overall.display"
    assert "overall.letter" not in declared.values()


def test_the_outcome_verdict_carries_its_denominator():
    """A tile status with no component count is a dot that cannot say how much
    it measured (alpha-engine-config-I8177)."""
    declared = _document_fields_paths(_load_descriptor())
    assert declared["portfolio_outcome_status"] == "tiles.portfolio_outcome.status"
    assert declared["portfolio_outcome_components"] == "tiles.portfolio_outcome.n_components"
    assert declared["portfolio_outcome_graded"] == "tiles.portfolio_outcome.n_graded"
    assert declared["portfolio_outcome_as_of"] == "tiles.portfolio_outcome.as_of"


def test_the_economic_components_are_not_bound_twice():
    """§2.5 — two same-rank claims on one entity is a conflict, not coverage.

    The per-component economic Signals (alpha_vs_spy, sharpe_ratio, …) are
    already claimed by the `s3-records` fan-out below. The `document-fields`
    binding must add the ROLL-UP and nothing that duplicates a row the fan-out
    already produces.
    """
    descriptor = _load_descriptor()
    assert _s3_records_binding(descriptor)["records_path"] == "tiles.*.components"
    for path in _document_fields_paths(descriptor).values():
        assert ".components" not in path, (
            f"{path} re-claims a record the s3-records fan-out already owns"
        )
