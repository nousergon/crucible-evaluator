"""Smoke tests — Phase A scaffolding.

Verifies the package skeleton imports and the shared-lib dependency is wired.
Real grading + director tests arrive with Phases C and E.
"""

import importlib


def test_packages_import():
    for pkg in ("grading", "director"):
        mod = importlib.import_module(pkg)
        assert mod.__doc__ and len(mod.__doc__.strip()) > 0


def test_nousergon_lib_available():
    """The shared-contract dependency installs and imports."""
    lib = importlib.import_module("nousergon_lib")
    assert lib.__version__
