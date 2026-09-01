"""tests/conftest.py — the unit-contract sweep (config#7485).

Wraps ``grading.metric_record.build_metric`` for the whole test session so
every value-bearing ``MetricRecord`` constructed while the ENTIRE suite runs
— i.e. every tile builder invoked against its own existing fixtures across
``tests/test_*_tile.py`` and friends — is inspected once, at session end,
rather than trusting one hand-picked happy-path fixture per tile. This is in
addition to (not instead of) the chokepoint enforcement in
``build_metric`` itself, which already raises ``MetricContractError``
synchronously on any violation; the sweep is a second, independent guard
that fails loud and names every offending call site if that enforcement is
ever weakened or bypassed, and asserts the sweep actually observed
production call sites (a wrapper that silently patches nothing would be a
detector with no coverage, not a passing test).
"""

from __future__ import annotations

import pytest

import grading.artifact_registry as artifact_registry_module
import grading.metric_record as metric_record_module

_seen: list[tuple[str, str, float | None, str | None]] = []
_original_build_metric = metric_record_module.build_metric


def _wrapped_build_metric(*args, **kwargs):
    record = _original_build_metric(*args, **kwargs)
    _seen.append((record.module, record.name, record.value, getattr(record, "unit", None)))
    return record


# Patched at CONFTEST IMPORT TIME (module level, not inside a fixture): pytest
# imports every conftest.py before it collects/imports the individual
# tests/test_*_tile.py modules, and each tile file does
# ``from grading.metric_record import build_metric`` at import time — binding
# its own local name to whatever ``metric_record_module.build_metric`` is AT
# THAT MOMENT. A fixture-scoped patch (even session-scoped autouse) runs too
# late: collection has already bound the original function in every caller.
metric_record_module.build_metric = _wrapped_build_metric


def unit_contract_sweep_results() -> list[tuple[str, str, float | None, str | None]]:
    """Read by ``tests/test_zz_unit_contract_sweep.py``, which runs last (by
    filename, under pytest's default alphabetical collection) so it sees every
    ``build_metric`` call made by the rest of the suite."""
    return list(_seen)


# ── The declared artifact registry (alpha-engine-config-I9731) ──────────────
#
# `grading/freshness_preflight.py` reads its predicates from the published
# ARTIFACT_REGISTRY.yaml mirror at run time instead of a hardcoded table, so
# every test that reaches `build_report_card` needs that document to exist.
# Installing the double here rather than seeding it in each module's own
# `_seed_freshness_inputs` keeps the change to one file and means a test module
# added later inherits it.
#
# `tests/test_freshness_preflight.py` marks itself `real_artifact_registry` and
# seeds the mirror into its own moto bucket instead: the S3 read path, the
# unreadable-registry failures and the dropped-row failures are that module's
# subject, and a suite-wide double would hide all three.


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_artifact_registry: read the ARTIFACT_REGISTRY mirror from S3 for "
        "real instead of the suite-wide test double",
    )


@pytest.fixture(autouse=True)
def _declared_artifact_registry(request, monkeypatch):
    if request.node.get_closest_marker("real_artifact_registry"):
        return
    from tests.artifact_registry_fixture import registry_document

    monkeypatch.setattr(
        artifact_registry_module,
        "load_registry",
        lambda *_args, **_kwargs: registry_document(),
    )
