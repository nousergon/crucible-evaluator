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


def test_metrics_binding_uses_s3_records_fanout():
    """T4 deliverable 1: one Signal per (tile, component) report-card row,
    via the `s3-records` driver's fan-out grammar (nousergon-console#98,
    shipped PR100). Pins the shape so a future edit cannot silently drop the
    fan-out back to a single-entity binding.
    """
    descriptor = _load_descriptor()
    metrics = descriptor["metrics"]
    assert metrics["driver"] == "s3-records"
    assert metrics["kind"] == "signal"
    assert metrics["records_path"] == "tiles.*.components"
    assert metrics["group_field"] == "tile"
    assert metrics["id_template"] == "{tile}:{name}"
    assert metrics["state_field"] == "status"
    assert "cadence_minutes" in metrics


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
    value_field = descriptor["metrics"]["fields"]["value"]
    assert "unit" not in value_field
    assert "baseline" not in value_field
    # The record's own unit/target are still surfaced, just not wired into
    # `value`'s own descriptor metadata yet.
    assert descriptor["metrics"]["fields"]["unit"]["path"] == "unit"
    assert descriptor["metrics"]["fields"]["target"]["path"] == "target"


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
