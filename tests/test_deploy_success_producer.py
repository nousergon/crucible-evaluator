"""Tests for grading/producers/deploy_success.py (config#1153 Batch E)."""

from datetime import UTC, datetime

import boto3
import pytest
from moto import mock_aws

from grading.producers.deploy_success import (
    DEPLOY_SUCCESS_KEY,
    build_deploy_success_doc,
    run,
    write_deploy_success_doc,
)

BUCKET = "alpha-engine-research"
NOW = datetime(2026, 6, 29, 12, 0, tzinfo=UTC)


def _run(name, path, branch, conclusion):
    return {"name": name, "path": path, "head_branch": branch,
            "status": "completed", "conclusion": conclusion}


def _fake_gh(runs_by_repo, *, missing=()):
    """Return a gh_request stub serving canned workflow_runs per repo."""
    def gh_request(method, url, token, body=None):
        # url = https://api.github.com/repos/<owner>/<repo>/actions/runs?...
        repo = url.split("/repos/", 1)[1].split("/actions/", 1)[0]
        if repo in missing:
            return 404, {"message": "Not Found"}
        page = int(url.split("&page=", 1)[1])
        runs = runs_by_repo.get(repo, [])
        return 200, {"workflow_runs": runs if page == 1 else []}
    return gh_request


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


class TestBuildDoc:
    def test_counts_only_deploy_runs_on_default_branch(self):
        runs = {
            "nousergon/crucible-evaluator": [
                _run("Deploy", ".github/workflows/deploy.yml", "main", "success"),
                _run("Deploy", ".github/workflows/deploy.yml", "main", "failure"),
                # excluded: CI workflow
                _run("CI", ".github/workflows/ci.yml", "main", "failure"),
                # excluded: PR/dependabot branch
                _run("Deploy", ".github/workflows/deploy.yml", "dependabot/x", "success"),
                # excluded: cancelled is not a pass/fail attempt
                _run("Deploy", ".github/workflows/deploy.yml", "main", "cancelled"),
            ],
        }
        doc = build_deploy_success_doc(
            "tok", repos=["nousergon/crucible-evaluator"], window_days=28,
            now=NOW, gh_request=_fake_gh(runs),
        )
        assert doc["total_runs"] == 2  # 1 success + 1 failure on main
        assert doc["success_runs"] == 1
        assert doc["success_rate"] == 0.5
        assert doc["per_repo"]["nousergon/crucible-evaluator"]["rate"] == 0.5
        assert doc["repos_measured"] == ["nousergon/crucible-evaluator"]
        assert doc["schema"] == "deploy_success/v1"

    def test_name_based_deploy_match_and_multi_repo_aggregate(self):
        runs = {
            "nousergon/crucible-dashboard": [
                _run("Deploy Marketing", ".github/workflows/deploy-marketing.yml", "main", "success"),
                _run("Deploy", ".github/workflows/deploy.yml", "main", "success"),
            ],
            "nousergon/nousergon-data": [
                _run("Deploy Infrastructure", ".github/workflows/deploy-infrastructure.yml", "master", "failure"),
            ],
        }
        doc = build_deploy_success_doc(
            "tok", repos=list(runs), window_days=28, now=NOW, gh_request=_fake_gh(runs),
        )
        assert doc["total_runs"] == 3
        assert doc["success_runs"] == 2
        assert doc["success_rate"] == pytest.approx(2 / 3)
        assert set(doc["repos_measured"]) == set(runs)

    def test_missing_repo_skipped_not_fatal(self):
        runs = {"nousergon/crucible-evaluator": [
            _run("Deploy", ".github/workflows/deploy.yml", "main", "success")]}
        doc = build_deploy_success_doc(
            "tok", repos=["nousergon/crucible-evaluator", "nousergon/renamed-gone"],
            window_days=28, now=NOW,
            gh_request=_fake_gh(runs, missing=["nousergon/renamed-gone"]),
        )
        assert doc["per_repo"]["nousergon/renamed-gone"] == {"status": "not_found"}
        assert doc["total_runs"] == 1
        assert doc["success_rate"] == 1.0

    def test_no_runs_yields_null_rate(self):
        doc = build_deploy_success_doc(
            "tok", repos=["nousergon/crucible-evaluator"], window_days=28,
            now=NOW, gh_request=_fake_gh({}),
        )
        assert doc["total_runs"] == 0
        assert doc["success_rate"] is None
        assert doc["repos_measured"] == []

    def test_generated_utc_is_iso_utc(self):
        doc = build_deploy_success_doc(
            "tok", repos=[], now=NOW, gh_request=_fake_gh({}),
        )
        assert doc["generated_utc"] == "2026-06-29T12:00:00+00:00"
        assert doc["since"] == "2026-06-01"


class TestRunAndWrite:
    def test_run_writes_rollup_to_s3(self, s3):
        runs = {"nousergon/crucible-evaluator": [
            _run("Deploy", ".github/workflows/deploy.yml", "main", "success")]}
        res = run(s3, BUCKET, "tok", repos=["nousergon/crucible-evaluator"],
                  now=NOW, gh_request=_fake_gh(runs))
        assert res["status"] == "ok"
        assert res["success_rate"] == 1.0
        obj = s3.get_object(Bucket=BUCKET, Key=DEPLOY_SUCCESS_KEY)
        import json
        doc = json.loads(obj["Body"].read())
        assert doc["success_rate"] == 1.0
        assert doc["total_runs"] == 1

    def test_write_returns_uri(self, s3):
        uri = write_deploy_success_doc(s3, BUCKET, {"x": 1})
        assert uri == f"s3://{BUCKET}/{DEPLOY_SUCCESS_KEY}"
