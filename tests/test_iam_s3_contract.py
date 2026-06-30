"""IAM S3-prefix coverage CI guard (config#1404).

**The bug class this closes.** A new cross-module S3 artifact *prefix* can enter
producer/consumer code with no check that the evaluator role actually grants it.
It then surfaces only at the grading canary as a confusing ``AccessDenied`` (S3
returns 403, not 404, for a missing key when ``s3:ListBucket`` lacks the prefix)
-- not at PR review. Concrete instance: PR #75 introduced
``_substrate/deploy_success.json`` (Director producer write + substrate-tile read)
with no ``_substrate/`` grant on ``alpha-engine-evaluator-role`` -> canary crash ->
Deploy red on main. The pre-existing live-vs-codified IAM drift check (config#1154)
could not catch it: the codified policy *also* lacked the prefix, so they agreed.
The missing dimension is **code-referenced-prefix -> codified-grant** coverage.

**The chokepoint.** ``grading/iam_s3_contract.json`` is the code-side source of
truth for every top-level S3 prefix this code reads/writes and the access it needs.
This test makes growing the code's S3-access surface without updating that contract
a **PR-review failure** instead of a canary-deploy failure. The ops repo
(nous-ergon-ops) separately verifies the deployed policy grants exactly the
contract's prefixes -- closing the loop across the public-code / private-policy
split.

**Two complementary mechanisms** (and why both):

1. *Per-file access-site count pin* -- the primary, refactor-stable guard. Mirrors
   ``nousergon-data/tests/test_artifact_registry_coverage.py``, which deliberately
   pins per-file site counts rather than extracting key templates because static
   extraction from arbitrary f-strings / helper-routed keys is fragile. A new or
   changed ``put_object`` / ``get_object`` / ``list_objects_v2`` / ``paginate`` site
   in any grading or director file trips this guard and forces the operator to
   confirm the prefix is in the contract (and granted in nous-ergon-ops) before
   bumping the pin. This is what would have caught PR #75 (a new producer file with
   a new PUT site).

2. *Resolvable-prefix -> grant assertion* -- a tightening pass on the cleanly
   AST-resolvable call sites (boto3 ``Key=`` / ``Prefix=`` arguments whose leading
   path segment is a string literal, a module-level constant, a leading-literal
   f-string, or a parameter default). Every resolvable prefix must be declared in
   the contract, and any prefix the code *writes* must be declared ``readwrite``.
   Helper-routed dynamic keys that don't resolve here are still covered by (1).
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "grading" / "iam_s3_contract.json"

# Production packages whose S3 access the evaluator role must cover.
_SCAN_ROOTS: tuple[str, ...] = ("grading", "director")

# S3 access methods. ``put_object`` is the only write site in this code today;
# the others are reads/lists. (``copy_object`` / ``upload_*`` would also be writes
# -- listed here so a future write site is classified, not silently dropped.)
_WRITE_METHODS: frozenset[str] = frozenset({"put_object", "copy_object", "upload_file", "upload_fileobj"})
_READ_METHODS: frozenset[str] = frozenset({"get_object", "list_objects_v2", "head_object", "delete_object", "paginate"})
_ACCESS_METHODS: frozenset[str] = _WRITE_METHODS | _READ_METHODS

_ACCESS_SITE_RE = re.compile(
    r"\.(?:put_object|get_object|list_objects_v2|copy_object|head_object|delete_object|"
    r"upload_file|upload_fileobj|paginate)\("
)

# ── Per-file access-site count pin ──────────────────────────────────────────
# Captured 2026-06-30. When a file gains/loses an S3 access site:
#   1. Confirm the prefix it touches is declared in grading/iam_s3_contract.json
#      (and granted in nous-ergon-ops/.../alpha-engine-evaluator-policy.json).
#   2. Bump the count here. When a file is added/removed wholesale, add/remove
#      its entry AND mirror the contract change.
EXPECTED_PER_FILE_ACCESS_COUNTS: dict[str, int] = {
    "director/carryover.py": 2,
    "director/handler.py": 7,
    "grading/aggregate.py": 2,
    "grading/artifacts.py": 1,
    "grading/producers/deploy_success.py": 1,
    "grading/tiles/agent.py": 1,
    "grading/tiles/backtester.py": 3,
    "grading/tiles/behavioral.py": 1,
    "grading/tiles/executor.py": 1,
    "grading/tiles/portfolio_outcome.py": 1,
    "grading/tiles/predictor.py": 2,
    "grading/tiles/research.py": 1,
    "grading/tiles/substrate.py": 2,
}


# ── Contract loading ────────────────────────────────────────────────────────


def _load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text())


def _granted_prefixes() -> dict[str, str]:
    return _load_contract()["prefixes"]


# ── Access-site enumeration (mechanism 1) ───────────────────────────────────


def _tracked_py_files() -> list[str]:
    """Tracked ``.py`` files under the production roots (``git ls-files`` discipline
    so untracked scratch files don't pollute the scan, matching CI behaviour)."""
    out = subprocess.run(
        ["git", "ls-files", "--", *[f"{r}/*.py" for r in _SCAN_ROOTS]],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    return [line for line in out if line]


def _enumerate_access_sites() -> dict[str, int]:
    """``{relative_path: count}`` for every tracked production file containing an
    S3 access site."""
    counts: dict[str, int] = {}
    for rel in _tracked_py_files():
        text = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        n = len(_ACCESS_SITE_RE.findall(text))
        if n:
            counts[rel] = n
    return counts


# ── Resolvable-prefix extraction (mechanism 2) ──────────────────────────────


def _first_segment(value: str) -> str | None:
    """Leading path segment of an S3 key/prefix string ('a/b/c.json' -> 'a').
    Returns None for a bare token with no '/' (not a prefixed key)."""
    value = value.lstrip("/")
    return value.split("/", 1)[0] if "/" in value else None


class _PrefixResolver:
    """Resolves the leading S3 prefix segment of a ``Key=`` / ``Prefix=`` argument
    expression, within one module's scope. Handles the patterns this codebase
    actually uses: string literals, leading-literal f-strings, module-level
    constants, intra-function literal assignments, parameter defaults, and same-
    module helper-return / ``str.format`` chains. Unresolvable expressions return
    None and are left to the count-pin guard."""

    def __init__(self, tree: ast.Module):
        self.consts: dict[str, ast.expr] = {}
        self.funcs: dict[str, ast.FunctionDef] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        self.consts[tgt.id] = node.value
            elif isinstance(node, ast.FunctionDef):
                self.funcs[node.name] = node

    def resolve(self, node: ast.expr, local: dict[str, ast.expr]) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return _first_segment(node.value)
        if isinstance(node, ast.JoinedStr) and node.values:
            head = node.values[0]
            if isinstance(head, ast.Constant) and isinstance(head.value, str):
                seg = _first_segment(head.value)
                if seg:
                    return seg
                return None  # f-string starts with a literal lacking '/', e.g. "s3://"
            if isinstance(head, ast.FormattedValue):
                return self.resolve(head.value, local)
            return None
        if isinstance(node, ast.Name):
            if node.id in local:
                return self.resolve(local[node.id], local)
            if node.id in self.consts:
                return self.resolve(self.consts[node.id], {})
            return None
        if isinstance(node, ast.Call):
            fn = getattr(node.func, "id", None)
            if fn in self.funcs:
                for sub in ast.walk(self.funcs[fn]):
                    if isinstance(sub, ast.Return) and sub.value is not None:
                        seg = self.resolve(sub.value, {})
                        if seg:
                            return seg
            if isinstance(node.func, ast.Attribute) and node.func.attr == "format":
                return self.resolve(node.func.value, local)
        return None


def extract_prefix_accesses(source: str) -> list[tuple[str, str]]:
    """Return ``[(prefix, 'read'|'write'), ...]`` for every cleanly resolvable S3
    access site in a Python module source. Standalone (operates on a string) so the
    regression test can feed it a synthetic PR #75-style snippet."""
    tree = ast.parse(source)
    resolver = _PrefixResolver(tree)
    found: list[tuple[str, str]] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        local: dict[str, ast.expr] = {}
        args = fn.args
        if args.defaults:
            for arg, default in zip(args.args[-len(args.defaults):], args.defaults):
                local[arg.arg] = default
        for sub in ast.walk(fn):
            if isinstance(sub, ast.Assign) and len(sub.targets) == 1 and isinstance(sub.targets[0], ast.Name):
                local[sub.targets[0].id] = sub.value
        for call in ast.walk(fn):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)):
                continue
            method = call.func.attr
            if method not in _ACCESS_METHODS:
                continue
            kw = {k.arg: k.value for k in call.keywords if k.arg}
            arg = kw.get("Key") or kw.get("Prefix")
            if arg is None:
                continue
            prefix = resolver.resolve(arg, local)
            if prefix is None:
                continue
            found.append((prefix, "write" if method in _WRITE_METHODS else "read"))
    return found


def _resolvable_repo_accesses() -> list[tuple[str, str, str]]:
    """``(prefix, mode, 'file')`` across the production roots."""
    out: list[tuple[str, str, str]] = []
    for root in _SCAN_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            for prefix, mode in extract_prefix_accesses(path.read_text()):
                out.append((prefix, mode, str(path.relative_to(REPO_ROOT))))
    return out


# ── Tests: contract well-formedness ─────────────────────────────────────────


def test_contract_is_wellformed():
    contract = _load_contract()
    assert contract["bucket"] == "alpha-engine-research"
    assert contract["prefixes"], "contract must declare at least one prefix"
    for prefix, mode in contract["prefixes"].items():
        assert mode in ("read", "readwrite"), f"{prefix}: bad access mode {mode!r}"
        assert "/" not in prefix, f"{prefix}: declare top-level prefixes only (no '/')"


# ── Tests: mechanism 1 (surface pin) ────────────────────────────────────────


def test_every_s3_access_file_is_pinned():
    actual = _enumerate_access_sites()
    unpinned = sorted(set(actual) - set(EXPECTED_PER_FILE_ACCESS_COUNTS))
    assert not unpinned, (
        "New file(s) with S3 access sites detected but not pinned:\n"
        + "\n".join(f"  - {f} ({actual[f]} site(s))" for f in unpinned)
        + "\n\nResolution:\n"
        "  1. Confirm the S3 prefix(es) it touches are declared in "
        "grading/iam_s3_contract.json (and granted in nous-ergon-ops/"
        "alpha-engine-evaluator/.../alpha-engine-evaluator-policy.json).\n"
        "  2. Add the file to EXPECTED_PER_FILE_ACCESS_COUNTS with its site count."
    )


def test_every_pinned_file_still_exists():
    actual = _enumerate_access_sites()
    stale = sorted(set(EXPECTED_PER_FILE_ACCESS_COUNTS) - set(actual))
    assert not stale, (
        "Pinned file(s) no longer have S3 access sites (or were removed):\n"
        + "\n".join(f"  - {f}" for f in stale)
        + "\n\nResolution: remove the file from EXPECTED_PER_FILE_ACCESS_COUNTS."
    )


def test_pinned_counts_match_actual():
    actual = _enumerate_access_sites()
    deltas = [
        f"  - {p}: expected={c}, actual={actual.get(p, 0)}"
        for p, c in sorted(EXPECTED_PER_FILE_ACCESS_COUNTS.items())
        if actual.get(p, 0) != c
    ]
    assert not deltas, (
        "S3 access-site count drift:\n" + "\n".join(deltas)
        + "\n\nResolution: for a new/removed site, update the contract if the "
        "prefix set changed, then bump the pinned count."
    )


# ── Tests: mechanism 2 (resolvable prefix -> grant) ─────────────────────────


def test_resolvable_prefixes_are_declared():
    granted = _granted_prefixes()
    ungranted = sorted({
        f"{prefix}  ({mode}, {loc})"
        for prefix, mode, loc in _resolvable_repo_accesses()
        if prefix not in granted
    })
    assert not ungranted, (
        "Code references S3 prefix(es) with no entry in grading/iam_s3_contract.json:\n"
        + "\n".join(f"  - {u}" for u in ungranted)
        + "\n\nThis is the PR #75 bug class. Add the prefix to the contract and "
        "ensure the evaluator role grants it in nous-ergon-ops."
    )


def test_resolvable_write_prefixes_are_readwrite():
    granted = _granted_prefixes()
    bad = sorted({
        f"{prefix}  ({loc})"
        for prefix, mode, loc in _resolvable_repo_accesses()
        if mode == "write" and granted.get(prefix) != "readwrite"
    })
    assert not bad, (
        "Code WRITES to prefix(es) not declared 'readwrite' in the contract:\n"
        + "\n".join(f"  - {b}" for b in bad)
        + "\n\nA write site needs s3:PutObject -> declare the prefix 'readwrite'."
    )


# ── Test: regression -- would have failed on PR #75's pre-fix state ─────────


_PR75_PREFIX_SNIPPET = '''
DEPLOY_SUCCESS_KEY = "_substrate/deploy_success.json"

def write_deploy_success_doc(s3, bucket, doc, key=DEPLOY_SUCCESS_KEY):
    s3.put_object(Bucket=bucket, Key=key, Body=b"{}")
'''


def test_guard_would_have_failed_on_pr75_prefix():
    """The exact PR #75 producer pattern resolves to a ``_substrate`` *write*; with
    a pre-fix contract lacking ``_substrate`` the grant assertion would have failed
    at PR review instead of crashing the canary."""
    accesses = extract_prefix_accesses(_PR75_PREFIX_SNIPPET)
    assert ("_substrate", "write") in accesses, accesses

    pre_fix_contract_prefixes = {
        "backtest": "read", "predictor": "read", "trades": "read",
        "config": "read", "signals": "read", "evaluator": "readwrite",
        "director": "readwrite",
        # NOTE: no "_substrate" -- the pre-PR#75-fix state.
    }
    referenced = {p for p, _ in accesses}
    ungranted = referenced - set(pre_fix_contract_prefixes)
    assert ungranted == {"_substrate"}, (
        "regression guard should flag _substrate as ungranted pre-fix; "
        f"got {ungranted!r}"
    )

    # And the post-fix contract (what we ship) must grant it readwrite.
    assert _granted_prefixes().get("_substrate") == "readwrite"
