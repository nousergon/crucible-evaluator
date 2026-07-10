"""Every top-level first-party package a Lambda handler imports at runtime
must have a matching ``COPY <pkg>/ <pkg>/`` line in the Dockerfile.

Ported from crucible-predictor#352, root-caused there by config#1282 (PR
#305): ``check_drift`` added ``from monitoring.drift_detector import
check_drift`` but the Dockerfile was never updated to copy ``monitoring/``
into the Lambda image. The module existed in the repo and passed CI (which
imports it directly off disk) but was absent from the deployed container —
every ``action=check_drift`` invocation 500'd with ``ModuleNotFoundError``
in prod, undetected because the canary only exercises a narrow happy path.
See also config#1853 for the same bug class recurring.

This repo's layout differs from crucible-predictor/crucible-research: there
is no separate ``lambda/`` directory — handler files live inside their own
package directories (``grading/handler.py``, ``director/handler.py``). Both
are live Lambda entrypoints: ``infrastructure/deploy.sh`` deploys the
grading handler as the primary ``alpha-engine-evaluator`` function (CMD
``grading.handler.handler``) AND a second function,
``alpha-engine-evaluator-director``, sharing the same image with a CMD
override of ``director.handler.handler``. The Director function is
flag-gated at runtime (``DIRECTOR_ENABLED``) but its code — and therefore
its imports — ships in the image and is exercised by deploy.sh's canary
regardless of the flag, so it is covered here too.

This test derives the required package set directly from each handler's
imports, so a future handler import of a NEW top-level package fails CI
immediately instead of shipping a silent 500 to prod.

Note on the COPY-line match: unlike crucible-predictor's Dockerfile (which
COPYs to a bare relative destination, e.g. ``COPY monitoring/
monitoring/``), this repo's Dockerfile COPYs into
``${LAMBDA_TASK_ROOT}/<pkg>/`` (e.g. ``COPY grading/
${LAMBDA_TASK_ROOT}/grading/``). The match below only pins the SOURCE side
(``COPY <pkg>/``) plus a following reference to that same package name, so
it holds regardless of the destination-path convention.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Every handler file that is a live Lambda entrypoint per infrastructure/
# deploy.sh (see module docstring for how each is wired).
HANDLERS = [
    REPO_ROOT / "grading" / "handler.py",
    REPO_ROOT / "director" / "handler.py",
]

# First-party top-level packages: repo-root directories that are real
# importable packages (have __init__.py). Anything a handler imports from a
# base name in this set must have a matching Dockerfile COPY line, or this
# test's own coverage would blind-spot it too.
_FIRST_PARTY_DIRS = {
    p.name for p in REPO_ROOT.iterdir() if p.is_dir() and (p / "__init__.py").exists()
}

_IMPORT_RE = re.compile(r"^\s*from (\w+)(?:\.\w+)* import ", re.MULTILINE)

# Matches ``COPY <pkg>/ <anything ending in <pkg>/>`` — pins the source side
# of the COPY instruction while tolerating either a bare relative
# destination (crucible-predictor's convention) or a
# ``${LAMBDA_TASK_ROOT}/<pkg>/`` destination (this repo's convention).
def _copy_re(pkg: str) -> re.Pattern[str]:
    return re.compile(rf"^COPY {re.escape(pkg)}/ \S*{re.escape(pkg)}/\s*$", re.MULTILINE)


def _first_party_packages_imported_by(handler: Path) -> set[str]:
    text = handler.read_text()
    bases = {m.group(1) for m in _IMPORT_RE.finditer(text)}
    return bases & _FIRST_PARTY_DIRS


def test_dockerfile_copies_every_package_handlers_import() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()

    imported: set[str] = set()
    for handler in HANDLERS:
        imported |= _first_party_packages_imported_by(handler)

    assert imported, (
        "Expected the evaluator handlers (grading/handler.py, "
        "director/handler.py) to import at least one first-party package "
        "(e.g. grading, director) — regex or package detection may have "
        "broken."
    )
    missing = {
        pkg for pkg in imported
        if not _copy_re(pkg).search(dockerfile)
    }
    assert not missing, (
        f"Dockerfile is missing COPY line(s) for package(s) {sorted(missing)}, "
        f"which a handler ({[str(h.relative_to(REPO_ROOT)) for h in HANDLERS]}) "
        f"imports at runtime. Without a 'COPY <pkg>/ ...<pkg>/' line, the "
        f"Lambda image ships without the module and the handler 500s with "
        f"ModuleNotFoundError in prod (config#1282 / config#1853 class of "
        f"bug) — the deploy canary may not exercise every handler code path."
    )
