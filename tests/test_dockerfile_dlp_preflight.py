"""The Lambda image asserts krepis's DLP prerequisites at build time.

nous-ergon-ops-I738: krepis-PR95 made ``LLMClient`` shell out to ``gitleaks``
and fail closed when it is absent. This image had no gitleaks, and the break
was found by an unrelated PR going red, not by anything watching. The binary
was added (crucible-evaluator#232), but nothing asserted it stays sufficient:
the next prerequisite krepis adds would be discovered the same way.

``python -m krepis.session_dlp preflight`` exits non-zero unless a real scan
could run on this substrate. Running it as a ``RUN`` step AFTER the dependency
install makes the image build itself the check, so a missing prerequisite fails
the build in CI and in deploy rather than the Director's first judge call.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

# The Dockerfile is not mounted into the docker-image-tests container.
pytestmark = pytest.mark.repo_tree

REPO_ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = re.compile(r"^RUN\s+python3?\s+-m\s+krepis\.session_dlp\s+preflight\b", re.M)


def _dockerfile() -> str:
    return (REPO_ROOT / "Dockerfile").read_text()


def test_image_build_runs_the_krepis_dlp_preflight() -> None:
    assert PREFLIGHT.search(_dockerfile()), (
        "Dockerfile no longer runs `python -m krepis.session_dlp preflight` as a "
        "build step. Without it, an image missing a krepis runtime prerequisite "
        "(gitleaks today) builds and deploys, and fails closed on the first LLM "
        "call instead (nous-ergon-ops-I738)."
    )


def test_preflight_runs_after_krepis_is_installed() -> None:
    docker = _dockerfile()
    preflight_at = PREFLIGHT.search(docker).start()
    install_at = docker.index("pip install --no-cache-dir -r /tmp/req-lambda.txt")
    gitleaks_at = docker.index("tar -xzf /tmp/gitleaks.tar.gz")
    assert install_at < preflight_at, (
        "the preflight must run against the INSTALLED krepis; before the "
        "requirements install it cannot import krepis at all"
    )
    assert gitleaks_at < preflight_at


def test_preflight_is_not_softened() -> None:
    line = PREFLIGHT.search(_dockerfile()).group(0)
    rest = _dockerfile()[PREFLIGHT.search(_dockerfile()).end():].split("\n", 1)[0]
    assert "||" not in rest and "; true" not in rest, (
        f"the preflight's exit status must fail the build: {line}{rest!r}"
    )
    assert not re.search(r"^ENV\s+KREPIS_DLP_DISABLED", _dockerfile(), re.M), (
        "KREPIS_DLP_DISABLED in the image turns the DLP control off, and "
        "preflight reports that as not ready"
    )
