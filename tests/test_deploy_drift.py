"""Tests for grading/deploy_drift.py — the evaluator's Lambda-SHA drift probe
(config#2348). Exercises the pure compare logic + the baked-stamp-file read
with a tmp_path fixture; the GitHub-fetch helper is owned + tested by
nousergon-lib, so it's mocked here via patch.object on the re-exported
module-level symbol (same pattern crucible-predictor's test_deploy_drift.py
uses for the identical re-import)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

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
