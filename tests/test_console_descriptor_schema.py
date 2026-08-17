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
commit e0aa0e6c83740a9cab219914f5cdefbf6740e97f (2026-08-10). It is a copy,
not a symlink, because this is a public AGPL repo and nousergon-console's
working tree is not guaranteed present at test time; re-sync it by hand if
the upstream schema changes (there is no cross-repo CI wiring for this yet —
see the nousergon-console issue filed alongside this PR for the driver-layer
gap this descriptor is blocked on).
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DESCRIPTOR_PATH = REPO_ROOT / "console.descriptor.yaml"
SCHEMA_PATH = REPO_ROOT / "tests" / "fixtures" / "component_descriptor.schema.json"


def _load_descriptor() -> dict:
    with open(DESCRIPTOR_PATH) as fh:
        return yaml.safe_load(fh)


def _load_schema() -> dict:
    with open(SCHEMA_PATH) as fh:
        return json.load(fh)


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


def test_descriptor_does_not_name_an_unregistered_driver():
    """A descriptor naming a driver that does not exist FAILS THE BUILD
    (console-policy.md §2.7) rather than rendering the component absent — a
    typo and a genuinely-gone component must not look the same. This test
    pins the corollary this PR relies on: until the missing driver capability
    (s3-records-shaped fan-out for `metrics`, a `state-machine` shape for
    `runs`) ships in nousergon-console, this descriptor must not claim either
    binding, because doing so would pass this schema test and then break the
    console build.
    """
    descriptor = _load_descriptor()
    known_drivers = {"object-store", "log-source", "sql-source",
                      "document-fields", "emitted-envelope"}

    def _bindings(value):
        if value is None:
            return []
        return value if isinstance(value, list) else [value]

    for key in ("runs", "artifacts", "metrics"):
        for binding in _bindings(descriptor.get(key)):
            assert binding["driver"] in known_drivers, (
                f"{key} binding names driver {binding.get('driver')!r}, "
                "which is not a registered nousergon-console driver as of "
                "this fixture's schema vintage."
            )

    # The two bindings I7477 asks for are explicitly NOT present yet — see
    # console.descriptor.yaml's header comment and the nousergon-console
    # issue filed alongside this PR.
    assert "metrics" not in descriptor
    assert "runs" not in descriptor
