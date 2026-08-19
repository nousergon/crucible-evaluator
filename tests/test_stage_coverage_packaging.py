"""Consumer-contract test: `krepis.stage_coverage` must actually be
importable in this repo's own resolved environment, and the `krepis` floor
in requirements.txt must never regress below the version that first ships
it (alpha-engine-config-I7334).

Root cause this guards against: `nousergon_lib.stage_coverage` (the module
`grading/handler.py::_record_stage_coverage` used to import) was briefly
added to nousergon-lib and immediately removed as a duplicate of the krepis
landing (nousergon-lib#314/#315) — it never existed on any released
nousergon-lib version. Every Director/ReportCard invocation took the
`except ImportError` branch and logged `assertion unavailable`, so I7214's
coverage assertion never once executed in production (measured 2026-08-14,
CloudWatch `/aws/lambda/alpha-engine-evaluator-director`).

Mirrors `nousergon-data/tests/test_sf_preflight.py::
test_lambda_profile_imports_are_packaged` — derive the requirement from the
handler's real import, don't hand-maintain a parallel assertion.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_REQUIREMENTS = _REPO_ROOT / "requirements.txt"

# The krepis release that first ships `krepis.stage_coverage`
# (krepis#148, config-I7214/I7334).
#
# Corrected 0.59.1 -> 0.59.2 (config-I7357). Measured by walking the krepis
# tags rather than read off a changelog line:
#
#   $ git ls-tree -r v0.59.0 --name-only | grep -c stage_coverage   -> 0
#   $ git ls-tree -r v0.59.1 --name-only | grep -c stage_coverage   -> 0
#   $ git ls-tree -r v0.59.2 --name-only | grep -c stage_coverage   -> 2
#
# This constant is what the floor assertion below compares against, so while
# it said 0.59.1 the test PASSED on a floor that could resolve to a release
# lacking the module — the guard agreed with the bug. A contract test keyed on
# a wrong constant is worse than no test: it reports the property as held.
_STAGE_COVERAGE_INTRODUCED_IN = (0, 59, 2)


def _krepis_floor() -> tuple[int, int, int]:
    """Parse the resolved krepis version out of requirements.txt.

    Accepts either `krepis[...]>=X.Y.Z` (a floor) or `krepis[...]==X.Y.Z`
    (an exact pin, per §139 — first-party/fast-moving deps are pinned, never
    floored). The property under test is "the krepis this image resolves is
    at or above X" — an exact pin establishes that property MORE strongly
    than a floor (it IS the resolved version, not a lower bound on it), so
    the comparator itself is not what matters here (alpha-engine-config-I7635).

    Deliberately regex-based rather than a `packaging` dependency this repo
    does not otherwise declare — the fleet-standard derive-from-real-source
    pattern, not a hand-maintained duplicate of the pinned number.
    """
    text = _REQUIREMENTS.read_text()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        m = re.match(r"^krepis(\[[^\]]*\])?(?:>=|==)(\d+)\.(\d+)\.(\d+)", line)
        if m:
            return (int(m.group(2)), int(m.group(3)), int(m.group(4)))
    raise AssertionError(
        "requirements.txt has no `krepis>=X.Y.Z` or `krepis==X.Y.Z` line"
    )


def test_krepis_floor_parses_both_floor_and_exact_pin_forms():
    """Regression guard for the exact defect I7635 fixed: converting the
    floor to an exact pin must not make this guard report the requirement
    as ABSENT. Exercises the regex directly against both comparator forms
    rather than only the live requirements.txt line, so a future re-floor
    or re-pin can't silently drop coverage of the form not currently used."""
    pattern = re.compile(r"^krepis(\[[^\]]*\])?(?:>=|==)(\d+)\.(\d+)\.(\d+)")
    floor_match = pattern.match("krepis[flow-doctor]>=0.59.16")
    exact_match = pattern.match("krepis[flow-doctor]==0.59.16")
    assert floor_match is not None
    assert exact_match is not None
    assert (
        floor_match.group(2, 3, 4)
        == exact_match.group(2, 3, 4)
        == ("0", "59", "16")
    )


def test_krepis_floor_covers_stage_coverage_introduction():
    """The pin cannot regress silently: this fails the moment the floor
    drops below the release that actually ships the module the handler
    imports — the exact class of bug this issue fixes."""
    floor = _krepis_floor()
    assert floor >= _STAGE_COVERAGE_INTRODUCED_IN, (
        f"requirements.txt krepis floor {'.'.join(map(str, floor))} predates "
        f"{'.'.join(map(str, _STAGE_COVERAGE_INTRODUCED_IN))}, which first "
        f"ships krepis.stage_coverage — every _record_stage_coverage call "
        f"will take the ImportError branch again (config-I7334)"
    )


def test_krepis_stage_coverage_is_importable_in_this_environment():
    """The consumer-contract test I7214/I7334 required: actually import the
    module at test time, in the SAME environment CI resolves requirements.txt
    into. If this fails, the pin is wrong (or krepis regressed) — not the
    handler's degrade path, which is exercised separately and is expected
    to log loud and mark UNMEASURED, never to silently pass."""
    krepis_stage_coverage = pytest.importorskip(
        "krepis.stage_coverage",
        reason="krepis.stage_coverage unimportable — see requirements.txt krepis floor",
    )
    assert hasattr(krepis_stage_coverage, "assert_stage_coverage")
    assert callable(krepis_stage_coverage.assert_stage_coverage)


def _record_stage_coverage_source() -> str:
    tree = ast.parse((_REPO_ROOT / "grading" / "handler.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_record_stage_coverage":
            return ast.get_source_segment(
                (_REPO_ROOT / "grading" / "handler.py").read_text(), node
            ) or ""
    raise AssertionError("grading/handler.py has no _record_stage_coverage function")


def test_record_stage_coverage_imports_from_krepis_not_nousergon_lib():
    """Regression guard for the exact defect I7334 fixed: the call site must
    target `krepis.stage_coverage`, never the wrong (and never-shipped)
    `nousergon_lib.stage_coverage` namespace."""
    src = _record_stage_coverage_source()
    assert "from krepis.stage_coverage import assert_stage_coverage" in src
    assert "nousergon_lib.stage_coverage" not in src


def test_record_stage_coverage_never_leaves_the_key_absent_on_import_failure():
    """Static guard alongside the behavioral tests in test_handler.py /
    test_director.py: the source must assign `result["stage_coverage"]`
    inside the except ImportError branch, not merely log and fall through
    with the key absent — an absent key is exactly what a healthy stage
    also produces, so absence must never be the unavailable signal."""
    src = _record_stage_coverage_source()
    except_idx = src.index("except ImportError")
    tail = src[except_idx:]
    assert '"UNMEASURED"' in tail or "_STAGE_COVERAGE_STATUS_UNMEASURED" in tail
    assert 'result["stage_coverage"] = {' in tail or 'result["stage_coverage"] = ' in tail
