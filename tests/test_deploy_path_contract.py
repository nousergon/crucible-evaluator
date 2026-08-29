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


class TestCiOnlySurfacesCannotArmTheDriftGate:
    """A CI-only file must not read as deploy-relevant.

    alpha-engine-config#9168. `EvaluatorDeployDriftGate` /
    `EvaluatorDirectorDeployDriftGate` HARD-FAIL the weekly pipeline on
    `has_drift: true`, ~30 s in, before `WeeklyPreflight` and before any spot
    spend — so a misclassified file does not degrade the run, it ends it having
    done nothing.

    On 2026-08-28 PR #280 changed only `.openrouter-allowlist.yaml`. Everything
    behaved as designed: the deploy workflow's `paths:` filter correctly skipped
    a rebuild, and `_has_deploy_relevant_changes` correctly refused to call an
    unclassified root-level file benign (default-deny, by explicit design in
    `deploy_paths`). Both gates reported drift and the 2026-08-29 09:00 UTC
    scheduled run would have terminated FAILED. The gap was an unclassified
    CI-only surface, not a wrong rule — so this pins the classification rather
    than weakening the default.
    """

    def test_the_openrouter_allowlist_is_not_deploy_relevant(self):
        assert not deploy_paths.is_deploy_relevant_path(".openrouter-allowlist.yaml")

    def test_it_is_never_copied_into_the_image(self):
        """The classification is only true while the file stays out of the image."""
        dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
        copied = [
            line.strip()
            for line in dockerfile.splitlines()
            if line.strip().startswith(("COPY", "ADD"))
            and "openrouter-allowlist" in line
        ]
        assert not copied, (
            ".openrouter-allowlist.yaml is now baked into the image, so a change "
            "to it CAN make the deployed image stale — remove it from "
            "SAFE_NON_DEPLOY_PATH_PATTERNS: " + "; ".join(copied)
        )

    def test_an_unclassified_root_file_still_defaults_to_deploy_relevant(self):
        """The default-deny posture this entry sits inside must not have moved."""
        assert deploy_paths.is_deploy_relevant_path("some_new_root_surface.yaml")
        assert deploy_paths.is_deploy_relevant_path("scripts/new_helper.py")
