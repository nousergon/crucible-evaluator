"""Deploy-path contract for the Lambda deploy trigger and drift gate."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from grading import deploy_paths


REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy.yml"


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
        "requirements.txt",
        "Dockerfile",
        "infrastructure/deploy.sh",
        ".github/workflows/deploy.yml",
    ):
        assert deploy_paths.is_deploy_relevant_path(path), path


def test_known_ci_and_docs_paths_are_explicitly_safe_non_deploy_changes():
    for path in (
        ".github/workflows/gate-label-guard.yml",
        "README.md",
        "docs/deployment.md",
        "tests/test_deploy_drift.py",
    ):
        assert not deploy_paths.is_deploy_relevant_path(path), path


def test_unclassified_path_is_deploy_relevant_by_default():
    assert deploy_paths.is_deploy_relevant_path("scripts/new_deploy_helper.py")
