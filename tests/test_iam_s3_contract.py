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
    # config-I7157 (2026-08-17): _run gained a second put_object (the standing
    # director/latest/action_plan.json pointer) alongside the existing dated
    # director/{date}/action_plan.json write -- same "director" prefix, already
    # declared readwrite in grading/iam_s3_contract.json and granted on the
    # role; no contract/IAM change, just a pin bump. Exactly the
    # evaluator/latest/report_card.json precedent recorded below.
    # alpha-engine-config-I8332 (2026-08-25): _compute_registry_fingerprint
    # gained a head_object call on director/LLM_MODEL_REGISTRY.yaml (the S3
    # LastModified/VersionId half of the registry provenance stamp) --
    # same "director" prefix, already declared readwrite; no contract/IAM
    # change, just a pin bump. 8 -> 9.
    "director/handler.py": 9,
    # config-I2556: write_report_card gained a second put_object (the standing
    # evaluator/latest/report_card.json pointer) alongside the existing dated
    # snapshot write -- same "evaluator" prefix, already granted readwrite;
    # no contract/IAM change, just a pin bump.
    # RC v3 T1 (config-I7474, 2026-08-16): compare_to_backtester's
    # get_object(backtest/{date}/grading.json) site removed along with the
    # backtester-parity soak-compare mechanism (--compare CLI flag too) —
    # 3 -> 2. No contract/IAM change: the remaining 2 sites are the same
    # already-granted "evaluator" prefix writes.
    "grading/aggregate.py": 2,
    # §2.3a runtime attestation: one get_object call site, shared by
    # `_read_verdict_artifact` across TWO keys — backtest/{run_date}/attestation.json
    # (the simulation engine's verdict) and backtest/{run_date}/evaluator_attestation.json
    # (the Evaluator stage's). The count stays 1 because the second key reuses the
    # same reader, not because nothing was added; both live under the "backtest"
    # prefix, already declared `read` in the contract, so no contract/IAM change.
    "grading/attestation.py": 1,
    # alpha-engine-config-I8188 deliverables 6-7 (2026-08-28): the
    # Brinson-Fachler attribution producer. THREE sites -- `_get_json`'s
    # get_object (market_data/sectors/latest.json and the eleven
    # market_data/close_history/{ETF}.json), `read_eod_pnl_rows`' get_object
    # (trades/eod_pnl.csv, already granted) and `write_attribution`'s
    # put_object (evaluator/, already granted readwrite). "market_data" is a
    # NEW prefix and is granted by nous-ergon-ops-PR909 -- that PR merges
    # FIRST, or the evaluator Lambda gets the 403-not-404 AccessDenied this
    # contract exists to prevent.
    "grading/attribution.py": 3,
    # config#3104: grading/artifacts.py S3 access moved to nousergon_lib.artifact_resolution SSoT;
    # the contract pin for the library-side access sites lives in nousergon-lib, not here.
    # config#3077: experiment_record writes dated + latest JSON pointers
    # (put_object ×2) under the "experiments" prefix — newly declared
    # readwrite in the contract for this PR.
    "grading/experiment_record.py": 2,
    # config#7238: the published known-answer self-test writes one artifact,
    # evaluator/{run_date}/self_test.json, via a single put_object in
    # `write_self_test` — under the "evaluator" prefix, already declared
    # readwrite above for the report card, so no contract or IAM change. The
    # battery itself touches S3 not at all: it drives the production tile over
    # an in-memory `_FrozenS3` that serves one frozen fixture and raises
    # NoSuchKey for every other key.
    "grading/self_test.py": 1,
    # config#3058: freshness_preflight reads metrics.json/e2e_lift.json (via
    # get_object) and probes signals.json instance dates (via head_object) --
    # all under the already-granted "backtest"/"predictor"/"signals"/"trades"
    # read prefixes, no contract/IAM change.
    "grading/freshness_preflight.py": 3,
    "grading/history.py": 2,
    "grading/producers/deploy_success.py": 1,
    "grading/tiles/agent.py": 1,
    "grading/tiles/backtester.py": 3,
    "grading/tiles/behavioral.py": 1,
    "grading/tiles/director_quality.py": 1,
    "grading/tiles/executor.py": 1,
    # alpha-engine-config-I8189: _load_paused_lanes reads the pause-reconcile
    # paused-lane declaration (ops/checks/automation-pause-reconcile/
    # paused_lanes.json, nousergon-data infrastructure/pause_reconcile.py) so
    # a declared groom pause never renders as an undeclared broken-producer
    # gap — new "ops" read prefix, declared in grading/iam_s3_contract.json.
    "grading/tiles/groom.py": 3,
    # alpha-engine-config-I9702 (2026-09-01): the per-date signals.json read
    # moved out of this file into grading/regime_index.py, which serves the
    # date->market_regime join from one persisted index instead of one ~637KB
    # GET per session since inception. 3 -> 2 here; the moved site (plus the
    # index's own read and write) is pinned on regime_index.py below. No
    # contract/IAM change: "signals" was already `read` and the index lives
    # under "evaluator", already `readwrite`.
    "grading/tiles/portfolio_outcome.py": 2,  # I9684 pbo leaderboard + eod_pnl.csv
    # alpha-engine-config-I9702: three sites — `read_index`'s get_object and
    # `write_index`'s put_object on evaluator/indexes/market_regime.json (the
    # "evaluator" prefix, already declared readwrite for the report card), and
    # `_fetch_one`'s get_object on signals/{date}/signals.json (the "signals"
    # prefix, already declared read). New FILE, no new prefix.
    "grading/regime_index.py": 3,
    "grading/tiles/predictor.py": 2,
    "grading/tiles/research.py": 1,
    "grading/tiles/substrate.py": 1,
    # alpha-engine-config-I7476 (RC v3 T3): threshold champion/challenger.
    # scoring.py writes evaluator/{date}/threshold_leaderboard.json (one
    # put_object); promote.py reads a leaderboard (one get_object). Both on
    # the "evaluator" prefix, already granted readwrite -- pin only.
    "grading/thresholds/promote.py": 1,
    "grading/thresholds/scoring.py": 1,
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


# ── Mechanism 3: library-mediated prefixes -> grant (config-I8156) ───────────
#
# **Why mechanisms 1 and 2 could not have caught this.** Both read THIS repo's
# own AST: (1) counts boto3 access sites per file, (2) resolves the leading
# segment of a `Key=` / `Prefix=` argument at those same sites. A prefix
# written by an IMPORTED LIBRARY has no call site in either scan, so the
# contract's coverage claim was silently scoped to first-party writes while
# its docstring claimed "every top-level namespace the evaluator's grading +
# director code reads or writes".
#
# Measured 2026-08-22 (alpha-engine-config-I8152): `krepis.stage_coverage`
# writes `_stage_coverage/` and `krepis.cost_sink` writes `decision_artifacts/`.
# Neither was declared, `alpha-engine-evaluator-role` never granted either, and
# it stood undetected for as long as the prefixes have existed — four weekly
# stages recorded zero coverage verdicts since 2026-08-14 and the Director's
# LLM spend was never attributed. The failure mode produces no PR signal, no
# CI signal and no page: only a fail-soft ERROR log nobody reads.
#
# The repo had already WRITTEN THIS DOWN rather than closing it, in
# EXPECTED_PER_FILE_ACCESS_COUNTS above: "config#3104: grading/artifacts.py S3
# access moved to nousergon_lib.artifact_resolution SSoT; the contract pin for
# the library-side access sites lives in nousergon-lib, not here." The PIN
# moved. The GRANT OBLIGATION did not, and nothing checked it.
#
# The class fix is `krepis.s3_surface`: the WRITER declares its own top-level
# prefixes, and every consumer's contract test reads the same declaration. For
# an env-configured target (`cost_sink`'s KREPIS_COST_SINK_PREFIX) the
# declaration names the VARIABLE and this test resolves it against THIS repo's
# own deploy configuration — the value that will actually run, not whatever
# happens to be set on the machine running pytest.

_KREPIS_FLOOR_HINT = (
    "This needs a krepis that ships `krepis.s3_surface` (krepis >= 0.59.27). "
    "Bump the floor in requirements.txt; the krepis PR adding it merges "
    "before this one (alpha-engine-config-I8156)."
)


def _deploy_environment() -> dict[str, str]:
    """Lambda environment THIS repo's own deploy script sets.

    Parsed from `infrastructure/deploy.sh`'s `krepis.aws merge-lambda-env
    --set K=V` invocations. Deliberately NOT `os.environ`: the question is
    what the deployed function will be configured with, and reading the test
    runner's environment would answer about a different world -- the same
    class of error as a coverage claim scoped to a smaller set than its name
    implies.
    """
    script = REPO_ROOT / "infrastructure" / "deploy.sh"
    if not script.exists():  # packaged//sdist checkouts carry no infrastructure/
        return {}
    found: dict[str, str] = {}
    for match in re.finditer(
        r"--set\s+([A-Z_][A-Z0-9_]*)=([^\s\\]+)", script.read_text(encoding="utf-8")
    ):
        found[match.group(1)] = match.group(2)
    return found


def _imported_krepis_modules() -> dict[str, set[str]]:
    """``{relative_path: {dotted krepis submodule, ...}}`` over the scan roots.

    Walks the WHOLE tree rather than just module-level statements: every
    stage-coverage and cost-sink import in this repo is a LAZY import inside a
    function body, guarded by `try: ... except ImportError`, which is exactly
    where a module-level-only scan would find nothing and pass.
    """
    out: dict[str, set[str]] = {}
    for rel in _tracked_py_files():
        try:
            tree = ast.parse((REPO_ROOT / rel).read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a broken file fails elsewhere
            continue
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "krepis" or alias.name.startswith("krepis."):
                        modules.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "krepis":
                    # `from krepis import cost_sink` -- each alias is a module.
                    modules.update(f"krepis.{a.name}" for a in node.names)
                elif node.module.startswith("krepis."):
                    # `from krepis.stage_coverage import assert_stage_coverage`
                    modules.add(node.module)
        modules.discard("krepis")  # the package itself declares no surface
        if modules:
            out[rel] = modules
    return out


def _s3_surface():
    try:
        from krepis import s3_surface as module
    except ImportError as exc:  # pragma: no cover - exercised by the floor bump
        pytest.fail(f"cannot import krepis.s3_surface: {exc}. {_KREPIS_FLOOR_HINT}")
    return module


def test_the_krepis_import_scan_actually_finds_the_known_importers():
    """Guard the guard. A scan that finds nothing passes vacuously, which is
    precisely how mechanisms 1 and 2 reported full coverage while blind."""
    imports = _imported_krepis_modules()
    all_modules = {m for mods in imports.values() for m in mods}
    assert "krepis.stage_coverage" in all_modules, (
        "the krepis import scan found no krepis.stage_coverage importer under "
        f"{_SCAN_ROOTS}; found {sorted(all_modules)}"
    )
    assert "krepis.cost_sink" in all_modules, (
        "the krepis import scan found no krepis.cost_sink importer under "
        f"{_SCAN_ROOTS}; found {sorted(all_modules)}"
    )


def test_library_mediated_prefixes_are_declared():
    """Every prefix an IMPORTED krepis module declares it writes must be in the
    contract, with at least the declared mode."""
    s3_surface = _s3_surface()
    granted = _granted_prefixes()
    environment = _deploy_environment()
    failures: list[str] = []

    for rel, modules in sorted(_imported_krepis_modules().items()):
        for module in sorted(modules):
            try:
                declared = s3_surface.prefixes_for([module], environment=environment)
            except ImportError as exc:
                failures.append(f"{rel} imports {module}, which will not import: {exc}")
                continue
            for prefix, mode in sorted(declared.items()):
                have = granted.get(prefix)
                if have is None:
                    failures.append(
                        f"{prefix}  ({mode}) -- written by {module}, imported by {rel}"
                    )
                elif mode == "readwrite" and have != "readwrite":
                    failures.append(
                        f"{prefix}  (needs readwrite, contract says {have!r}) -- "
                        f"written by {module}, imported by {rel}"
                    )

    assert not failures, (
        "Imported krepis module(s) write S3 prefix(es) the contract does not "
        "grant:\n"
        + "\n".join(f"  - {f}" for f in failures)
        + "\n\nThis is the alpha-engine-config-I8156 bug class: the prefix has "
        "no call site in THIS repo, so mechanisms 1 and 2 are structurally "
        "blind to it, and the runtime failure is a fail-soft ERROR log nobody "
        "reads. Add the prefix to grading/iam_s3_contract.json with at least "
        "the declared mode AND grant it on alpha-engine-evaluator-role in "
        "nous-ergon-ops."
    )


def test_the_two_prefixes_i8152_found_are_covered_by_this_mechanism():
    """Regression: the exact pair that got through. Not a restatement of the
    contract file -- it asserts the MECHANISM derives them from the imports."""
    s3_surface = _s3_surface()
    imports = _imported_krepis_modules()
    modules = {m for mods in imports.values() for m in mods}
    derived = s3_surface.prefixes_for(modules, environment=_deploy_environment())
    assert derived.get("_stage_coverage") == "readwrite", derived
    assert derived.get("decision_artifacts") == "readwrite", derived


def test_the_cost_sink_prefix_is_resolved_from_this_repos_deploy_config():
    """The env-configured half. `decision_artifacts` is only derivable because
    infrastructure/deploy.sh sets KREPIS_COST_SINK_PREFIX; reading os.environ
    instead would answer about the test runner, not the Lambda."""
    s3_surface = _s3_surface()
    environment = _deploy_environment()
    assert environment.get("KREPIS_COST_SINK_PREFIX", "").startswith(
        "decision_artifacts"
    ), environment
    with_config = s3_surface.prefixes_for(["krepis.cost_sink"], environment=environment)
    without_config = s3_surface.prefixes_for(["krepis.cost_sink"], environment={})
    assert with_config == {"decision_artifacts": "readwrite"}
    assert without_config == {}
