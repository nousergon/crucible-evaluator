"""Declared deploy-path classification for evaluator image drift checks.

GitHub Actions evaluates ``on.push.paths`` before a checkout exists, so the
workflow keeps a static projection of ``DEPLOY_PATH_PATTERNS``. The companion
contract test makes that projection fail when it diverges from this declaration.
"""

from __future__ import annotations

from fnmatch import fnmatchcase


# The authoritative deploy-trigger paths. Keep this in GitHub Actions glob
# syntax; tests/test_deploy_path_contract.py asserts deploy.yml is exact.
DEPLOY_PATH_PATTERNS: tuple[str, ...] = (
    "grading/**",
    "director/**",
    "requirements*.txt",
    "Dockerfile*",
    "infrastructure/deploy.sh",
    ".github/workflows/deploy.yml",
)

# Only deliberately classified documentation, test, and CI changes can clear a
# SHA mismatch. Every other unmatched path is deploy-relevant by default: a new
# source or infrastructure surface must not silently make a stale image benign.
SAFE_NON_DEPLOY_PATH_PATTERNS: tuple[str, ...] = (
    ".github/**",
    "tests/**",
    "docs/**",
    "README*",
    "CONTRIBUTING.md",
    "LICENSE",
    "NOTICE",
    "SECURITY.md",
    "AGENTS.md",
    "CLAUDE.md",
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

    Declared deploy paths win over a broad safe class (notably
    ``.github/**``), and unknown paths fail closed.
    """
    if any(_matches(path, pattern) for pattern in DEPLOY_PATH_PATTERNS):
        return True
    if any(_matches(path, pattern) for pattern in SAFE_NON_DEPLOY_PATH_PATTERNS):
        return False
    return True
