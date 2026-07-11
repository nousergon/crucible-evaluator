"""The Dockerfile must not hardcode a second, independent nousergon-lib pin.

Root-caused 2026-07-11: the Dockerfile's dependency-install RUN line used to
hardcode `nousergon-lib[quant-stats] @ git+https://.../nousergon-lib@v0.83.0`
as a literal, separate from requirements.txt's own
`nousergon-lib[quant-stats,contracts] @ ...@v0.93.0` line (bumped + the
`contracts` extra added in PR#99, 2026-07-08). Because the RUN line grepped
the `nousergon-lib` line OUT of requirements.txt before installing the rest,
every image build since PR#99 silently installed the STALE v0.83.0 pin
without the `contracts` extra — `grading/tiles/backtester.py`'s
`contracts.conformance_errors()` call then raised `ImportError` in every
live ReportCard invocation for 9+ days, which cascaded into the Director
Lambda never receiving a valid card to build a weekly plan from
(config#1310 investigation, alpha-engine-config repo).

This test asserts the Dockerfile reads the pin dynamically out of
requirements.txt (`grep '^nousergon-lib' requirements.txt`) rather than
duplicating it as a literal `@vX.Y.Z` string — the only way to make this
class of drift structurally impossible instead of re-syncing it once more.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_does_not_hardcode_a_second_nousergon_lib_pin() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()

    hardcoded_pin = re.search(r"nousergon-lib\[[^\]]*\]\s*@\s*git\+https://\S+@v[\d.]+", dockerfile)
    assert hardcoded_pin is None, (
        f"Dockerfile hardcodes a literal nousergon-lib pin ({hardcoded_pin.group(0)!r}) "
        "instead of reading it out of requirements.txt. This is exactly the "
        "dual-source-of-truth pattern that let the Dockerfile silently drift "
        "to a stale pin (missing the `contracts` extra) for 9+ days after "
        "requirements.txt was bumped in PR#99 — see this test's module "
        "docstring. The Dockerfile's pip-install RUN line must derive the "
        "pin from `grep '^nousergon-lib' requirements.txt` instead."
    )

    assert "grep '^nousergon-lib' requirements.txt" in dockerfile or \
        'grep "^nousergon-lib" requirements.txt' in dockerfile, (
        "Dockerfile no longer appears to read the nousergon-lib pin "
        "dynamically from requirements.txt — if the extraction mechanism "
        "changed, update this assertion to match the new pattern, but keep "
        "requirements.txt as the single source of truth for the pin."
    )


def test_requirements_txt_nousergon_lib_pin_has_contracts_extra() -> None:
    requirements = (REPO_ROOT / "requirements.txt").read_text()
    lib_lines = [
        line for line in requirements.splitlines()
        if line.strip().startswith("nousergon-lib")
    ]
    assert len(lib_lines) == 1, (
        f"Expected exactly one nousergon-lib line in requirements.txt, found {lib_lines!r}"
    )
    assert "contracts" in lib_lines[0], (
        "requirements.txt's nousergon-lib pin no longer requests the "
        "[contracts] extra, which grading/tiles/backtester.py needs for "
        "contracts.conformance_errors() — see PR#99 (config#1861) and this "
        "test module's docstring for the 2026-07-11 incident this guards "
        "against."
    )
