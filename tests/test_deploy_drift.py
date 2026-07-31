"""Tests for grading/deploy_drift.py — the evaluator's Lambda-SHA drift probe
(config#2348). Exercises the pure compare logic + the baked-stamp-file read
with a tmp_path fixture; the GitHub-fetch helper is owned + tested by
nousergon-lib, so it's mocked here via patch.object on the re-exported
module-level symbol (same pattern crucible-predictor's test_deploy_drift.py
uses for the identical re-import)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import grading.deploy_drift as dd

SHA_A = "a" * 40
SHA_B = "b" * 40


# ── Pure helpers ─────────────────────────────────────────────────────────────

def test_shas_match_exact():
    assert dd._shas_match(SHA_A, SHA_A) is True


def test_shas_match_short_prefix():
    assert dd._shas_match("aaaaaaaaaaaa", SHA_A) is True


def test_shas_match_mismatch():
    assert dd._shas_match(SHA_A, SHA_B) is False


def test_shas_match_none_deployed_passes():
    assert dd._shas_match(None, SHA_A) is True


def test_shas_match_none_upstream_passes():
    assert dd._shas_match(SHA_A, None) is True


def test_shas_match_malformed_deployed_passes():
    assert dd._shas_match("abc", SHA_A) is True  # <7 chars = malformed


# ── _read_baked_git_sha ───────────────────────────────────────────────────────

def test_read_baked_git_sha_happy_path(tmp_path: Path):
    stamp = tmp_path / "GIT_SHA.txt"
    stamp.write_text(SHA_A + "\n")
    assert dd._read_baked_git_sha(stamp) == SHA_A


def test_read_baked_git_sha_missing_file_returns_none(tmp_path: Path):
    stamp = tmp_path / "does_not_exist.txt"
    assert dd._read_baked_git_sha(stamp) is None


def test_read_baked_git_sha_unknown_sentinel_returns_none(tmp_path: Path):
    stamp = tmp_path / "GIT_SHA.txt"
    stamp.write_text("unknown")
    assert dd._read_baked_git_sha(stamp) is None


def test_read_baked_git_sha_empty_file_returns_none(tmp_path: Path):
    stamp = tmp_path / "GIT_SHA.txt"
    stamp.write_text("")
    assert dd._read_baked_git_sha(stamp) is None


def test_read_baked_git_sha_strips_whitespace(tmp_path: Path):
    stamp = tmp_path / "GIT_SHA.txt"
    stamp.write_text(f"  {SHA_A}  \n")
    assert dd._read_baked_git_sha(stamp) == SHA_A


# ── check_deploy_drift composition ───────────────────────────────────────────

def test_no_drift_when_shas_match(tmp_path: Path):
    stamp = tmp_path / "GIT_SHA.txt"
    stamp.write_text(SHA_A)
    with patch.object(dd, "_fetch_origin_main_sha", return_value=SHA_A):
        result = dd.check_deploy_drift(
            function_name="alpha-engine-evaluator", sha_file=stamp,
        )
    assert result["has_drift"] is False
    assert result["reason"] == "in_sync"
    assert result["stamp_present"] is True
    assert result["function_name"] == "alpha-engine-evaluator"


def test_drift_detected_on_sha_mismatch(tmp_path: Path):
    stamp = tmp_path / "GIT_SHA.txt"
    stamp.write_text(SHA_A)
    with patch.object(dd, "_fetch_origin_main_sha", return_value=SHA_B):
        result = dd.check_deploy_drift(
            function_name="alpha-engine-evaluator-director", sha_file=stamp,
        )
    assert result["has_drift"] is True
    assert result["reason"] == "sha_mismatch"
    assert result["baked_sha"] == SHA_A
    assert result["upstream_sha"] == SHA_B


def test_short_prefix_stamp_still_matches(tmp_path: Path):
    # deploy.sh could plausibly stamp a short SHA; the prefix-match rule
    # (mirrors predictor's _shas_match) must not false-positive drift.
    stamp = tmp_path / "GIT_SHA.txt"
    stamp.write_text(SHA_A[:12])
    with patch.object(dd, "_fetch_origin_main_sha", return_value=SHA_A):
        result = dd.check_deploy_drift(function_name="alpha-engine-evaluator", sha_file=stamp)
    assert result["has_drift"] is False


def test_missing_stamp_file_is_no_drift_not_hard_fail(tmp_path: Path):
    # Legacy/local image without the GIT_SHA stamp — fail-open, not fail-loud.
    stamp = tmp_path / "does_not_exist.txt"
    with patch.object(dd, "_fetch_origin_main_sha", return_value=SHA_A):
        result = dd.check_deploy_drift(function_name="alpha-engine-evaluator", sha_file=stamp)
    assert result["has_drift"] is False
    assert result["stamp_present"] is False
    assert result["reason"] == "no_git_sha_stamp"


def test_github_outage_is_no_drift(tmp_path: Path):
    stamp = tmp_path / "GIT_SHA.txt"
    stamp.write_text(SHA_A)
    with patch.object(dd, "_fetch_origin_main_sha", return_value=None):
        result = dd.check_deploy_drift(function_name="alpha-engine-evaluator", sha_file=stamp)
    assert result["has_drift"] is False
    assert result["reason"] == "github_unreachable"
    assert result["upstream_sha"] is None


def test_repo_and_branch_are_threaded_through(tmp_path: Path):
    stamp = tmp_path / "GIT_SHA.txt"
    stamp.write_text(SHA_A)
    with patch.object(dd, "_fetch_origin_main_sha", return_value=SHA_A) as mock_fetch:
        result = dd.check_deploy_drift(
            function_name="alpha-engine-evaluator",
            repo="nousergon/crucible-evaluator",
            branch="main",
            sha_file=stamp,
        )
    mock_fetch.assert_called_once_with("nousergon/crucible-evaluator", branch="main", timeout=5.0)
    assert result["repo"] == "nousergon/crucible-evaluator"
    assert result["branch"] == "main"


# ── _has_deploy_relevant_changes ─────────────────────────────────────────────


def _mock_urlopen_with_files(files):
    """Return a fake urlopen context-manager whose response body carries the
    given ``files`` list inside a compare-API-shaped JSON payload."""

    class _FakeResponse:
        @staticmethod
        def read():
            return json.dumps({"files": files}).encode()

    class _FakeCtx:
        def __enter__(self):
            return _FakeResponse()

        def __exit__(self, *args):
            pass

    return _FakeCtx()


def test_has_deploy_relevant_changes_true_on_grading_file():
    with patch.object(dd, "_safe_urlopen", return_value=_mock_urlopen_with_files(
        [{"filename": "grading/handler.py"}],
    )):
        assert dd._has_deploy_relevant_changes(
            "nousergon/crucible-evaluator", SHA_A, SHA_B,
        ) is True


def test_has_deploy_relevant_changes_true_on_director_file():
    with patch.object(dd, "_safe_urlopen", return_value=_mock_urlopen_with_files(
        [{"filename": "director/handler.py"}],
    )):
        assert dd._has_deploy_relevant_changes(
            "nousergon/crucible-evaluator", SHA_A, SHA_B,
        ) is True


def test_has_deploy_relevant_changes_true_on_requirements():
    with patch.object(dd, "_safe_urlopen", return_value=_mock_urlopen_with_files(
        [{"filename": "requirements.txt"}],
    )):
        assert dd._has_deploy_relevant_changes(
            "nousergon/crucible-evaluator", SHA_A, SHA_B,
        ) is True


def test_has_deploy_relevant_changes_true_on_dockerfile():
    with patch.object(dd, "_safe_urlopen", return_value=_mock_urlopen_with_files(
        [{"filename": "Dockerfile"}],
    )):
        assert dd._has_deploy_relevant_changes(
            "nousergon/crucible-evaluator", SHA_A, SHA_B,
        ) is True


def test_has_deploy_relevant_changes_true_on_deploy_sh():
    with patch.object(dd, "_safe_urlopen", return_value=_mock_urlopen_with_files(
        [{"filename": "infrastructure/deploy.sh"}],
    )):
        assert dd._has_deploy_relevant_changes(
            "nousergon/crucible-evaluator", SHA_A, SHA_B,
        ) is True


def test_has_deploy_relevant_changes_true_on_deploy_yml():
    with patch.object(dd, "_safe_urlopen", return_value=_mock_urlopen_with_files(
        [{"filename": ".github/workflows/deploy.yml"}],
    )):
        assert dd._has_deploy_relevant_changes(
            "nousergon/crucible-evaluator", SHA_A, SHA_B,
        ) is True


def test_has_deploy_relevant_changes_false_on_ci_only_change():
    with patch.object(dd, "_safe_urlopen", return_value=_mock_urlopen_with_files(
        [{"filename": ".github/workflows/gate-label-guard.yml"}],
    )):
        assert dd._has_deploy_relevant_changes(
            "nousergon/crucible-evaluator", SHA_A, SHA_B,
        ) is False


def test_has_deploy_relevant_changes_false_on_docs_change():
    with patch.object(dd, "_safe_urlopen", return_value=_mock_urlopen_with_files(
        [{"filename": "README.md"}],
    )):
        assert dd._has_deploy_relevant_changes(
            "nousergon/crucible-evaluator", SHA_A, SHA_B,
        ) is False


def test_has_deploy_relevant_changes_true_on_mixed_changes():
    # One deploy-relevant file among several non-relevant ones → True
    with patch.object(dd, "_safe_urlopen", return_value=_mock_urlopen_with_files(
        [
            {"filename": ".github/workflows/gate-label-guard.yml"},
            {"filename": "grading/handler.py"},
            {"filename": "README.md"},
        ],
    )):
        assert dd._has_deploy_relevant_changes(
            "nousergon/crucible-evaluator", SHA_A, SHA_B,
        ) is True


def test_has_deploy_relevant_changes_fail_closed_on_api_error():
    # API unreachable → fail-closed (treat as drift)
    with patch.object(dd, "_safe_urlopen", side_effect=OSError("timeout")):
        assert dd._has_deploy_relevant_changes(
            "nousergon/crucible-evaluator", SHA_A, SHA_B,
        ) is True


def test_has_deploy_relevant_changes_fail_closed_on_empty_files():
    # Compare API returned no files (force-push / empty merge) → conservative
    with patch.object(dd, "_safe_urlopen", return_value=_mock_urlopen_with_files([])):
        assert dd._has_deploy_relevant_changes(
            "nousergon/crucible-evaluator", SHA_A, SHA_B,
        ) is True


def test_has_deploy_relevant_changes_fail_closed_on_json_error():
    with (
        patch.object(dd, "_safe_urlopen", return_value=_mock_urlopen_with_files([])),
        patch.object(json, "loads", side_effect=json.JSONDecodeError("bad json", "", 0)),
    ):
        assert dd._has_deploy_relevant_changes(
            "nousergon/crucible-evaluator", SHA_A, SHA_B,
        ) is True


# ── check_deploy_drift + path-aware gate ───────────────────────────────────

def test_drift_with_non_deploy_paths_is_not_drift(tmp_path: Path):
    """When main is ahead but only non-deploy paths changed, has_drift must be False."""
    stamp = tmp_path / "GIT_SHA.txt"
    stamp.write_text(SHA_A)
    with (
        patch.object(dd, "_fetch_origin_main_sha", return_value=SHA_B),
        patch.object(dd, "_has_deploy_relevant_changes", return_value=False),
    ):
        result = dd.check_deploy_drift(
            function_name="alpha-engine-evaluator", sha_file=stamp,
        )
    assert result["has_drift"] is False
    assert result["reason"] == "sha_mismatch_non_deploy_paths"
    assert result["baked_sha"] == SHA_A
    assert result["upstream_sha"] == SHA_B


def test_drift_with_deploy_paths_remains_drift(tmp_path: Path):
    """When main is ahead AND deploy-relevant paths changed, has_drift must stay True."""
    stamp = tmp_path / "GIT_SHA.txt"
    stamp.write_text(SHA_A)
    with (
        patch.object(dd, "_fetch_origin_main_sha", return_value=SHA_B),
        patch.object(dd, "_has_deploy_relevant_changes", return_value=True),
    ):
        result = dd.check_deploy_drift(
            function_name="alpha-engine-evaluator", sha_file=stamp,
        )
    assert result["has_drift"] is True
    assert result["reason"] == "sha_mismatch"
    assert result["baked_sha"] == SHA_A
    assert result["upstream_sha"] == SHA_B


# ── _resolve_function_name ───────────────────────────────────────────────────

class _FakeContext:
    def __init__(self, function_name):
        self.function_name = function_name


def test_resolve_function_name_from_context():
    ctx = _FakeContext("alpha-engine-evaluator-director")
    assert dd._resolve_function_name(ctx) == "alpha-engine-evaluator-director"


def test_resolve_function_name_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "alpha-engine-evaluator")
    assert dd._resolve_function_name(None) == "alpha-engine-evaluator"


def test_resolve_function_name_falls_back_to_unknown(monkeypatch):
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
    assert dd._resolve_function_name(None) == "unknown"
