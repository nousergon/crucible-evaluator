"""Deploy-path contract for the Lambda deploy trigger and drift gate.

One list (``grading.deploy_paths.DEPLOY_PATH_PATTERNS``) is simultaneously the
deploy TRIGGER and the drift PREDICATE, so this file pins three things that
must never disagree (alpha-engine-config-I9168):

1. ``deploy.yml``'s ``on.push.paths`` is an exact projection of the list.
2. ``is_deploy_relevant_path`` is exactly membership in the list — no second
   list, no fail-closed default. A path the deploy workflow skips cannot have
   changed the image, so it cannot be drift.
3. Every source the ``Dockerfile`` COPYs into the image IS in the list. This
   is where the old fail-closed default's real concern moved: a new surface
   that ships in the image but triggers no deploy now fails on its own PR,
   not by hard-failing a four-hour weekly run days later.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from grading import deploy_paths


REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy.yml"
DOCKERFILE = REPO_ROOT / "Dockerfile"


def _workflow_push_paths() -> list[str]:
    workflow = yaml.load(DEPLOY_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    return workflow["on"]["push"]["paths"]


def test_workflow_path_filter_is_exact_projection_of_deploy_path_contract():
    if not DEPLOY_WORKFLOW.is_file():
        pytest.skip("GitHub workflow definitions are intentionally absent from the Lambda image")

    assert _workflow_push_paths() == list(deploy_paths.DEPLOY_PATH_PATTERNS)


def test_deploy_paths_are_classified_as_deploy_relevant():
    for path in (
        "grading/handler.py",
        "director/handler.py",
        "flow-doctor.yaml",
        "requirements.txt",
        "Dockerfile",
        "infrastructure/deploy.sh",
        ".github/workflows/deploy.yml",
    ):
        assert deploy_paths.is_deploy_relevant_path(path), path


def test_paths_the_deploy_filter_skips_are_never_drift():
    """The complement is total: anything deploy.yml would not rebuild for."""
    for path in (
        ".github/workflows/gate-label-guard.yml",
        "README.md",
        "docs/deployment.md",
        "tests/test_deploy_drift.py",
        # alpha-engine-config-I9168: the live instance. A root-level file in
        # NEITHER of the two former lists — it matched no deploy pattern, so
        # deploy.yml correctly skipped it, and it matched no safe pattern, so
        # the fail-closed default called it drift. Both evaluator Lambdas were
        # armed to hard-fail the 2026-08-29 09:00 UTC weekly run off it.
        ".openrouter-allowlist.yaml",
        ".pre-commit-config.yaml",
        ".gitleaks-baseline.json",
        "console.descriptor.yaml",
        "scripts/new_deploy_helper.py",
        "infrastructure/rollback.sh",
    ):
        assert not deploy_paths.is_deploy_relevant_path(path), path


def test_predicate_is_exactly_membership_in_the_deploy_filter():
    """No second list and no default may creep back in.

    Two independently maintained lists over one partition leave a hole by
    construction, which is what I9168 was.
    """
    assert not hasattr(deploy_paths, "SAFE_NON_DEPLOY_PATH_PATTERNS")
    for path in (
        "grading/x.py",
        "some/unknown/path.py",
        "brand_new_root_file.yaml",
        ".github/workflows/deploy.yml",
    ):
        expected = any(
            deploy_paths._matches(path, pattern)
            for pattern in deploy_paths.DEPLOY_PATH_PATTERNS
        )
        assert deploy_paths.is_deploy_relevant_path(path) is expected, path


def _dockerfile_copy_sources() -> list[str]:
    """Every non-``--from`` COPY/ADD source in the Dockerfile, as a FILE path.

    ``COPY <src>... <dest>`` — the last token is the destination. Multi-stage
    ``COPY --from=`` lines copy out of a previous BUILD STAGE, not the repo, so
    they carry no repo path and are excluded.

    A directory source (``COPY grading/ …``) is returned as a representative
    file INSIDE it, because that is the shape a commit's changed-file list
    actually carries and the shape ``is_deploy_relevant_path`` is defined over.
    Testing the bare directory name instead silently fails every ``dir/**``
    pattern and would report the whole image as un-deployable.
    """
    sources: list[str] = []
    for raw in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not re.match(r"^(COPY|ADD)\s", line):
            continue
        tokens = line.split()[1:]
        if any(t.startswith("--from=") for t in tokens):
            continue
        tokens = [t for t in tokens if not t.startswith("--")]
        if len(tokens) < 2:
            continue
        for src in tokens[:-1]:
            if src in (".", "./"):
                continue
            sources.append(f"{src}__file__" if src.endswith("/") else src)
    return sources


def test_every_dockerfile_copy_source_is_a_declared_deploy_path():
    """A source that ships in the image but triggers no deploy is un-deployable.

    This is the guard that replaced the drift predicate's fail-closed default
    (I9168): the same concern, moved from the weekly critical path at 09:00 UTC
    to the PR that introduces the surface. It found ``flow-doctor.yaml`` on its
    first run — COPYed into the image, absent from the deploy filter, so every
    change to it had shipped nowhere.
    """
    if not DOCKERFILE.is_file():
        pytest.skip("Dockerfile is intentionally absent from the Lambda image")

    uncovered = [
        src for src in _dockerfile_copy_sources()
        if not deploy_paths.is_deploy_relevant_path(src)
    ]
    assert not uncovered, (
        "Dockerfile COPYs these into the image but deploy.yml's paths filter "
        f"does not fire on them, so a change to any of them would never deploy: {uncovered}. "
        "Add each to grading/deploy_paths.py::DEPLOY_PATH_PATTERNS and to "
        ".github/workflows/deploy.yml's on.push.paths (both, they are pinned equal)."
    )


def test_dockerfile_copy_source_parser_ignores_build_stage_copies():
    """A ``COPY --from=builder`` source is a stage path, not a repo path."""
    if not DOCKERFILE.is_file():
        # docker-image-tests runs this suite INSIDE the built image, where
        # /var/task holds only what the Dockerfile COPYed — not the Dockerfile.
        # The sibling test above already carries this guard; omitting it here
        # made the image job red while every local run passed.
        pytest.skip("Dockerfile is intentionally absent from the Lambda image")
    sources = _dockerfile_copy_sources()
    assert sources, "parser found no COPY sources at all"
    assert not any(s.startswith("--") for s in sources)
    # Directory sources must arrive as a path INSIDE the directory, or the
    # `dir/**` patterns can never match and the guard inverts into noise.
    assert "grading/__file__" in sources
    assert "director/__file__" in sources
    assert "flow-doctor.yaml" in sources
