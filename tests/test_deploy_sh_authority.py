"""The deploy script must not need read access to configuration to deploy.

`infrastructure/deploy.sh` runs as `github-actions-lambda-deploy`, a role scoped
to building images and moving Lambda aliases. On 2026-08-03 the Director half of
that script read `/alpha-engine/EMAIL_SENDER` and `/alpha-engine/EMAIL_RECIPIENTS`
from SSM to bake them into the function's environment. The role holds no
`ssm:GetParameter`, so the step died with AccessDenied *after* the grading
Lambda's alias had already been promoted (run 30829310846), stranding the
Director's `:live` alias three commits behind `main` on an image that still
carried the `exclude_route="litellm_proxy"` shortcut — a route that egresses to
openrouter.ai DLP-unscanned (`alpha-engine-config-I6183`).

The fix was to delete the bake: the Lambda is no longer VPC-attached, so
flow-doctor resolves both values from SSM at runtime, which is where the source
of truth already lives.

These tests exist so the *class* cannot come back. Reintroducing a config read in
the deploy path fails CI rather than half-deploying on a Saturday morning. The
alternative fix — granting the deploy role `ssm:GetParameter` — is the one this
asserts against: a deploy identity that can read configuration is authority the
deploy does not need.
"""

import re
from pathlib import Path

DEPLOY_SH = Path(__file__).resolve().parents[1] / "infrastructure" / "deploy.sh"


def _script() -> str:
    return DEPLOY_SH.read_text(encoding="utf-8")


def _executable_lines(text: str) -> list[str]:
    """Lines with comments stripped — a rule about behaviour, not about prose.

    The removal's rationale is written in the script as a comment block that
    names the deleted commands. Matching raw text would make that comment
    unwritable.
    """
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        out.append(line.split("#", 1)[0])
    return out


def test_deploy_does_not_read_ssm_parameters():
    hits = [ln for ln in _executable_lines(_script())
            if re.search(r"\bssm\s+get-parameters?\b", ln)]
    assert not hits, (
        "deploy.sh reads SSM. github-actions-lambda-deploy holds no "
        "ssm:GetParameter, so this fails mid-deploy with AccessDenied after "
        "the grading alias has already moved. Resolve configuration at "
        "runtime in the Lambda, where SSM is already the source of truth."
    )


def test_director_update_does_not_overwrite_the_live_environment():
    """`--environment` on the Director update path wipes operator-set vars.

    `DIRECTOR_ENABLED` and `KREPIS_LITELLM_PROXY_URL` are set out-of-band. A
    blind `--environment` write on the update branch silently switches the
    Director off; the create branch legitimately seeds the initial env.
    """
    lines = _executable_lines(_script())
    try:
        start = next(i for i, ln in enumerate(lines)
                     if "DIRECTOR_FUNCTION=" in ln and "${FUNCTION}" in ln)
    except StopIteration:  # pragma: no cover - structural guard
        raise AssertionError("deploy.sh no longer defines DIRECTOR_FUNCTION")
    create_at = next((i for i, ln in enumerate(lines[start:], start)
                      if "lambda create-function" in ln), len(lines))

    update_block = "\n".join(lines[start:create_at])
    assert "update-function-configuration" in update_block, (
        "the Director update path no longer configures the function — this "
        "test is anchored to the wrong block"
    )
    assert "--environment" not in update_block, (
        "deploy.sh writes --environment on the Director UPDATE path. That "
        "overwrites operator-set DIRECTOR_ENABLED / KREPIS_LITELLM_PROXY_URL. "
        "Omit the flag: an update that does not pass --environment leaves the "
        "live environment untouched."
    )
