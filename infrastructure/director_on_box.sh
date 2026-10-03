#!/usr/bin/env bash
# infrastructure/director_on_box.sh — run the weekly Director on the launcher spot.
#
# Invoked by the weekly SF's `Director` state over SSM (nousergon-data
# infrastructure/step_function.json), as root, on the box
# `DispatchWeeklyFreshnessSpot` launched, from a checkout the SF has already
# pinned for the execution (exec_code_pin.sh):
#
#   bash /home/ec2-user/crucible-evaluator/infrastructure/director_on_box.sh \
#     --date 2026-10-02 --dry-run false \
#     --gate-state-file /tmp/director-gate-state-<execution>.json \
#     --deadline-epoch <unix seconds> --execution-name <execution>
#
# Every argument is passed through to `python -m director.box_run`.
#
# WHY (alpha-engine-config-I11936). The Director's plan call no longer fits
# AWS Lambda's 900s maximum; Brian ruled on 2026-10-03 to move it off Lambda
# rather than change the model. This box has no such cap, so the plan call is
# bounded by director/hosting.py's WEEKLY_SPOT profile instead (1,800s plan
# ceiling inside a 2,700s wall) — a different hard bound, never no bound.
#
# What this script provisions, idempotently, before handing over:
#   1. gitleaks — krepis.session_dlp shells out to it on every LLM call and
#      fails CLOSED without it. The weekly box's bootstrap never installed it
#      because nothing on that box called a model until now. Same pin as this
#      repo's Dockerfile; tests/test_director_on_box.py asserts they match.
#   2. a venv for THIS repo's requirements, keyed on requirements.txt's sha256,
#      so a box that already built it (a second Director run on the same box)
#      reuses it, and a requirements change builds a new one rather than
#      mutating the old. It cannot share the box's other venvs: the dashboard
#      venv pins numpy<2 and this repo needs numpy~=2.5; the data venv carries
#      different krepis / nousergon-lib pins.
#   3. the DLP boot gate (`krepis.session_dlp preflight`), so a broken DLP
#      surface fails HERE, named, rather than at the first paid call.
#
# The time all of that takes is director/hosting.py's
# WEEKLY_SPOT_PROVISION_ALLOWANCE_S; the SF's --deadline-epoch is what keeps it
# honest — box_run clamps its wall to whatever is left.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_ROOT="${DIRECTOR_VENV_ROOT:-/opt/crucible-evaluator-venvs}"
PYTHON_BIN="${DIRECTOR_PYTHON_BIN:-python3.12}"

# gitleaks pin — LOCKSTEP with this repo's Dockerfile (and the fleet standard
# _spot_common.sh::install_gitleaks_dlp in nousergon-data). Bump together.
GITLEAKS_VERSION="8.30.1"
GITLEAKS_SHA256="551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb"

log() { echo "[director-on-box] $*" >&2; }

install_gitleaks() {
  if command -v gitleaks >/dev/null 2>&1; then
    return 0
  fi
  log "installing gitleaks ${GITLEAKS_VERSION}"
  local tmp
  tmp="$(mktemp -d)"
  curl -fsSL -o "${tmp}/gitleaks.tar.gz" \
    "https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz"
  echo "${GITLEAKS_SHA256}  ${tmp}/gitleaks.tar.gz" | sha256sum -c - >&2
  tar -xzf "${tmp}/gitleaks.tar.gz" -C /usr/local/bin gitleaks
  chmod +x /usr/local/bin/gitleaks
  rm -rf "${tmp}"
  command -v gitleaks >/dev/null 2>&1 || { log "FATAL: gitleaks unavailable after install"; exit 1; }
}

ensure_venv() {
  local req="${REPO}/requirements.txt" key venv
  key="$(sha256sum "${req}" | cut -c1-16)"
  venv="${VENV_ROOT}/${key}"
  if [ -f "${venv}/.ready" ]; then
    log "reusing venv ${venv}"
    echo "${venv}"
    return 0
  fi
  log "building venv ${venv} from requirements.txt (${key})"
  rm -rf "${venv}"
  mkdir -p "${VENV_ROOT}"
  "${PYTHON_BIN}" -m venv "${venv}"
  "${venv}/bin/pip" install --quiet --upgrade pip >&2
  # Same split as the Dockerfile: nousergon-lib (a git+https install) first,
  # then everything else minus the test-only dependencies.
  local lib_line
  lib_line="$(grep '^nousergon-lib' "${req}")"
  [ -n "${lib_line}" ] || { log "FATAL: no nousergon-lib line in requirements.txt"; exit 1; }
  "${venv}/bin/pip" install --quiet "${lib_line}" >&2
  grep -vE '^#|^$|^pytest|^pytest-cov|^moto|^python-dotenv|^nousergon-lib' "${req}" > "${venv}/req-runtime.txt"
  "${venv}/bin/pip" install --quiet -r "${venv}/req-runtime.txt" >&2
  touch "${venv}/.ready"
  echo "${venv}"
}

install_gitleaks
VENV="$(ensure_venv)"
PY="${VENV}/bin/python"

cd "${REPO}"
export PYTHONPATH="${REPO}"
export HOME="${HOME:-/home/ec2-user}"
export AWS_REGION="${AWS_REGION:-us-east-1}" AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

# The Director Lambda's environment, restated for this home. Values match the
# live function's configuration and infrastructure/deploy.sh; the test asserts
# the cost-sink pair against deploy.sh so the two cannot drift.
#
# DIRECTOR_ENABLED: on, because this state only runs when the SF did not
# bypass it — the box path's kill-switch is the SF input `skip_director`.
export DIRECTOR_ENABLED="${DIRECTOR_ENABLED:-true}"
export EVALUATOR_BUCKET="${EVALUATOR_BUCKET:-alpha-engine-research}"
export KREPIS_COST_SINK_BUCKET=alpha-engine-research
export KREPIS_COST_SINK_PREFIX=decision_artifacts/_cost_raw
export KREPIS_LITELLM_PROXY_URL="${KREPIS_LITELLM_PROXY_URL:-https://router.nousergon.ai:8443}"
export KREPIS_LITELLM_MASTER_KEY_SSM_PARAM="${KREPIS_LITELLM_MASTER_KEY_SSM_PARAM:-/alpha-engine/LITELLM_MASTER_KEY}"
# Router-only, exactly as from Lambda. krepis' context vocabulary is
# laptop / ec2 / lambda / ci, and `ec2` means the DASHBOARD box, whose loopback
# egress proxies make direct registry routes reachable. This spot carries no
# loopback proxy, so declaring `ec2` would offer routes that do not exist here
# and disarm director/agent.py's `_assert_routed_through_the_proxy` guard. The
# router edge is the only path this box has; `lambda` is the context that says
# so. (A vocabulary gap, not a routing choice — see the PR body.)
export KREPIS_EXEC_CONTEXT=lambda

log "DLP boot gate"
"${PY}" -m krepis.session_dlp preflight >&2

log "handing over to director.box_run"
exec "${PY}" -m director.box_run "$@"
