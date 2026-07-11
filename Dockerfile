FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.12

# git — required for the pip ``git+https://...`` install of nousergon-lib
# (the base Lambda image ships no git). microdnf is the AL2023 minimal pkg
# manager. Mirrors the research/predictor Lambda images.
RUN microdnf install -y git && microdnf clean all

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
    grep -vE "^#|^$|^pytest|^pytest-cov|^moto|^python-dotenv|^boto3|^botocore|^s3transfer|^nousergon-lib" requirements.txt > /tmp/req-lambda.txt && \
    pip install --no-cache-dir -r /tmp/req-lambda.txt && \
    rm -rf /root/.cache/pip /tmp/req-lambda.txt

# Application code (Layer B grading + Layer C director skeleton).
COPY grading/ ${LAMBDA_TASK_ROOT}/grading/
COPY director/ ${LAMBDA_TASK_ROOT}/director/

# flow-doctor config — resolved at runtime by setup_logging() in each handler.
# Ships in the Lambda task root (the handlers locate it via LAMBDA_TASK_ROOT).
COPY flow-doctor.yaml ${LAMBDA_TASK_ROOT}/

# Lambda entrypoint: the grading-layer producer. Builds the Report Card v2 and
# writes evaluator/{date}/report_card.json. (The Director, Part II, will add its
# own handler to the same image.)
CMD ["grading.handler.handler"]
