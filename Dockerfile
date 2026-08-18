FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.12

# git — required for the pip ``git+https://...`` install of nousergon-lib
# (the base Lambda image ships no git). tar/gzip are required below to
# unpack the pinned gitleaks release tarball (the base image ships curl +
# sha256sum but not tar/gzip). microdnf is the AL2023 minimal pkg manager.
# Mirrors the research/predictor Lambda images.
RUN microdnf install -y git tar gzip && microdnf clean all

# nous-ergon-ops-I738: krepis v0.59.15/16 (krepis-PR95) added an in-process
# DLP scan hook to LLMClient (session_dlp.py) that shells out to a
# ``gitleaks`` binary on PATH and fails closed (DLP_SCAN_ERROR) when it is
# missing. This Lambda's grading/director handlers make real LLM/Director
# judge calls through LLMClient at runtime, so the binary + its config must
# ship in the image or every judge call fails closed. Same pin (version +
# sha256) as krepis's own CI step (.github/workflows/test.yml) — do not
# re-derive independently; bump both in lockstep.
#
# krepis's session_dlp.py resolves its config directory in this order (env
# override, then /opt/llm-routing, /opt/groom-llm-routing,
# /opt/drain-llm-routing, then a bundled ``_gitleaks_config`` dir next to
# the installed krepis package, else defaulting to /opt/llm-routing) at
# MODULE IMPORT TIME — the directory must therefore exist in the image
# before the Lambda process starts, not just the binary. This image has no
# shared-dashboard-box provisioning of /opt/llm-routing, so we create it
# here with a minimal ``[extend] useDefault = true`` config, matching the
# standard/default resolution path rather than the Lambda-bundled fallback.
RUN set -euo pipefail; \
    GITLEAKS_VERSION="8.30.1"; \
    GITLEAKS_SHA256="551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb"; \
    curl -sSL -o /tmp/gitleaks.tar.gz "https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz"; \
    echo "${GITLEAKS_SHA256}  /tmp/gitleaks.tar.gz" | sha256sum -c -; \
    tar -xzf /tmp/gitleaks.tar.gz -C /usr/local/bin gitleaks; \
    chmod +x /usr/local/bin/gitleaks; \
    rm -f /tmp/gitleaks.tar.gz; \
    gitleaks version; \
    mkdir -p /opt/llm-routing; \
    printf '[extend]\nuseDefault = true\n' > /opt/llm-routing/gitleaks-egress.toml

# config#2348: stamp the image with the commit it was built from, so the
# weekly SF's new Lambda-SHA drift probe (grading/deploy_drift.py,
# nousergon_lib.preflight._fetch_origin_main_sha) can compare this against
# origin/main HEAD and catch a failed/skipped post-merge deploy leaving
# :live behind main. `--build-arg GIT_SHA=<sha>` (CI uses $GITHUB_SHA; local
# dev via infrastructure/deploy.sh defaults to `git rev-parse HEAD`).
# Mirrors crucible-predictor's Dockerfile stamping convention exactly.
ARG GIT_SHA=unknown
RUN echo "${GIT_SHA}" > /var/task/GIT_SHA.txt

# Dependencies. nousergon-lib is installed in its own layer (ahead of the
# rest of requirements.txt) purely so the slow git+https clone/build gets its
# own Docker cache layer, invalidated only when requirements.txt's
# nousergon-lib line itself changes — not on every unrelated dependency bump.
# The pin + extras are read DIRECTLY out of requirements.txt (single source
# of truth) rather than duplicated as a second hardcoded literal: a prior
# version hardcoded `nousergon-lib[quant-stats] @ ...@v0.83.0` here, which
# silently drifted out of lockstep with requirements.txt's v0.93.0 + the
# [contracts] extra added in PR#99 (2026-07-08) — every build since then
# installed the stale pin because this RUN line won and requirements.txt's
# nousergon-lib line was grepped OUT below. Root-caused + fixed
# 2026-07-11: no more second declaration to drift.
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN NOUSERGON_LIB_LINE="$(grep '^nousergon-lib' requirements.txt)" && \
    test -n "${NOUSERGON_LIB_LINE}" && \
    pip install --no-cache-dir "${NOUSERGON_LIB_LINE}" && \
    grep -vE "^#|^$|^pytest|^pytest-cov|^moto|^python-dotenv|^nousergon-lib" requirements.txt > /tmp/req-lambda.txt && \
    pip install --no-cache-dir -r /tmp/req-lambda.txt && \
    rm -rf /root/.cache/pip /tmp/req-lambda.txt

# config-I4799: the Lambda Python 3.12 base image ships boto3 1.34.x which may
# predate the appconfigdata service (boto3>=1.34.99). krepis 0.26.0's AppConfig
# registry resolution needs it for the Director's LLM_MODEL_REGISTRY lookup.
# Force-upgrade so appconfigdata client is available.
RUN pip install --no-cache-dir -U "boto3>=1.36" "botocore>=1.36"

# Application code (Layer B grading + Layer C director skeleton).
COPY grading/ ${LAMBDA_TASK_ROOT}/grading/
COPY director/ ${LAMBDA_TASK_ROOT}/director/

# flow-doctor config — resolved at runtime by setup_logging() in each handler.
# Ships in the Lambda task root (the handlers locate it via LAMBDA_TASK_ROOT).
COPY flow-doctor.yaml ${LAMBDA_TASK_ROOT}/

# Lambda entrypoint: the grading-layer producer. Builds the Report Card v2 and
# writes evaluator/{date}/report_card.json. (The Director, Part II, will add its
# own handler to the same image.)
#
# LLM_MODEL_REGISTRY.yaml is downloaded from S3 at Lambda startup by
# director/handler.py's _ensure_registry() rather than baked into the image.
# config-I4799: AppConfig env var is also set as a future enabler — once
# the AppConfig application is provisioned, the S3 download becomes redundant.
ENV KREPIS_APPCONFIG_APPLICATION=alpha-engine
CMD ["grading.handler.handler"]
