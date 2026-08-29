"""Declared deploy-path classification for evaluator image drift checks.

GitHub Actions evaluates ``on.push.paths`` before a checkout exists, so the
workflow keeps a static projection of ``DEPLOY_PATH_PATTERNS``. The companion
contract test makes that projection fail when it diverges from this declaration.

WHY THE PREDICATE IS THE EXACT COMPLEMENT OF THE DEPLOY FILTER
--------------------------------------------------------------
alpha-engine-config-I9168. Until 2026-08-28 this module carried a SECOND list
(``SAFE_NON_DEPLOY_PATH_PATTERNS``) and a fail-closed default: a path in
neither list was called deploy-relevant, so a SHA mismatch caused by it stayed
blocking. Two independently maintained lists over one partition leaves a hole
by construction, and on 2026-08-28 a commit fell in it: ``#280`` changed only
``.openrouter-allowlist.yaml`` — a root file matching no deploy pattern (so
``deploy.yml`` correctly did not run) and no safe pattern (so this predicate
called it drift). Both ``alpha-engine-evaluator:live`` and
``alpha-engine-evaluator-director:live`` were armed to hard-fail the
2026-08-29 09:00 UTC weekly run at ``EvaluatorDeployDriftGate``, ~30s in,
before any spend.

The fail-closed branch asserted something provably false. ``deploy.yml``'s
``paths:`` filter did not match that commit, so no deploy ran, so no deploy
COULD have run: the image at ``baked_sha`` is byte-identical to what a deploy
of ``upstream_sha`` would produce. Calling that drift is not conservatism, it
is a wrong answer that kills a four-hour run.

So the predicate is now exactly ``matches DEPLOY_PATH_PATTERNS`` — the
complement of the deploy trigger, one list, no default. This is the shape
``crucible-research/infrastructure/lambda_deploy_drift.py`` already reached
("a second copy of that list is a fork that would go stale in exactly the
direction that produces false pages"); that module reads ``deploy.yml`` at
runtime because it runs in a checkout, and this one materializes the same
single source because a Lambda image has no checkout — the contract test
pins them equal.

THE CONCERN THE FAIL-CLOSED DEFAULT WAS ENCODING — AND WHERE IT MOVED
----------------------------------------------------------------------
"A new source or infrastructure surface must not silently make a stale image
benign" is a real defect, but it is a DEPLOY-FILTER COMPLETENESS defect, not a
drift-verdict one, and the weekly critical path at 09:00 UTC is the wrong
place and the wrong time to discover it. It now fails at PR time instead:
``tests/test_deploy_path_contract.py`` asserts every source the ``Dockerfile``
COPYs into the image is covered by ``DEPLOY_PATH_PATTERNS``. That guard found
a live instance on its first run — ``flow-doctor.yaml`` is COPYed into the
image and was absent from the filter, so a change to it landed in no deploy at
all. It is in the list below now.
"""

from __future__ import annotations

from fnmatch import fnmatchcase


# The authoritative deploy-trigger paths. Keep this in GitHub Actions glob
# syntax; tests/test_deploy_path_contract.py asserts deploy.yml is exact AND
# that every Dockerfile COPY source is covered here.
DEPLOY_PATH_PATTERNS: tuple[str, ...] = (
    "grading/**",
    "director/**",
    "flow-doctor.yaml",
    "requirements*.txt",
    "Dockerfile*",
    "infrastructure/deploy.sh",
    ".github/workflows/deploy.yml",
)


def _matches(path: str, pattern: str) -> bool:
    """Match the small, root-relative GitHub Actions glob subset used here."""
    normalized = path.lstrip("/")
    if pattern.endswith("/**"):
        return normalized.startswith(pattern[:-2])
    if "/" not in pattern and "/" in normalized:
        return False
    return fnmatchcase(normalized, pattern)


def is_deploy_relevant_path(path: str) -> bool:
    """Whether *path* must keep a Lambda SHA mismatch blocking.

    Exactly the deploy trigger's own predicate: a path the deploy workflow
    would not rebuild for cannot have changed the image, so it cannot be
    drift. See the module docstring for why there is no second list and no
    fail-closed default.
    """
    return any(_matches(path, pattern) for pattern in DEPLOY_PATH_PATTERNS)
