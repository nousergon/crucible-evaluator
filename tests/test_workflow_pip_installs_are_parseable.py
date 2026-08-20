"""A CI step must never hand pip a requirements LINE it built with grep.

On 2026-08-19 the `krepis[...]` pin in `requirements.txt` gained a trailing
inline comment (`# pinned per §139 ...`). `deploy.yml` installed the runner-side
canary CLI with::

    pip install "$(grep '^krepis\\[' requirements.txt)"

The quoting hands pip the WHOLE matched line, comment included, as one
requirement string, and pip rejects it::

    ERROR: Invalid requirement: 'krepis[flow-doctor,openai]==0.59.17
    # pinned per §139 - first-party/fast-moving deps are pinned, never floored':
    Expected comma (within version specifier), semicolon (after version
    specifier) or end

The workflow went red at that commit and stayed red for four consecutive merges
(runs 32302653629, 32398441504, 32398939408, 32412323671), so crucible-evaluator
was **merged-but-not-live** from 2026-08-19 21:12 UTC until 2026-08-20
(alpha-engine-config-I7855). Nothing about the failure named its cause: it looks
like a dependency problem, not an editorial comment three lines away in a file
nobody thought CI was parsing.

The repair is to feed pip a requirements FILE — `pip install -r <(grep ...)` —
which moves comment handling into pip's own parser, where the format is defined.
This test exists so the class cannot come back: a comment added to any pinned
line must never be able to break a deploy again.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

WORKFLOWS = sorted(WORKFLOW_DIR.glob("*.yml")) + sorted(WORKFLOW_DIR.glob("*.yaml"))
IDS = [p.name for p in WORKFLOWS]

# `pip install "$(...)"` / `pip install '$(...)'` / `pip install $(...)` —
# any command substitution reaching pip as a bare requirement argument.
# `-r <(...)` is the sanctioned form and does not match: the `-r` makes the
# substituted text a FILE for pip to parse, not a requirement string.
BARE_SUBSTITUTION = re.compile(
    r"""pip\s+install\s+(?!-r\b)(?:[^\n]*?\s)?["']?\$\(""",
)


def test_the_workflow_dir_actually_has_workflows():
    """A discovery regression must fail rather than report a clean tree."""
    assert WORKFLOWS, f"no workflows found under {WORKFLOW_DIR}"


@pytest.mark.parametrize("path", WORKFLOWS, ids=IDS)
def test_no_pip_install_of_a_command_substitution(path: Path):
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue  # a comment explaining the anti-pattern is not the pattern
        assert not BARE_SUBSTITUTION.search(line), (
            f"{path.relative_to(REPO_ROOT)}:{lineno} passes a command "
            f"substitution to pip as a requirement argument:\n\n    {stripped}\n\n"
            "A grep of requirements.txt returns the whole line, trailing inline "
            "comment included, and pip rejects it — the deploy then goes red for "
            "an editorial change (alpha-engine-config-I7855). Use "
            "`pip install -r <(grep ... requirements.txt)` so pip parses the "
            "file with its own parser."
        )


def test_the_deploy_workflow_uses_the_sanctioned_form():
    """Positive assertion, so deleting the step does not quietly pass the guard
    above by removing the thing it guards."""
    deploy = (WORKFLOW_DIR / "deploy.yml").read_text()
    assert "pip install -r <(grep '^krepis\\[' requirements.txt)" in deploy, (
        "deploy.yml no longer installs krepis via `pip install -r <(grep ...)`. "
        "If the step moved or changed shape, update this test deliberately — "
        "do not delete it."
    )
