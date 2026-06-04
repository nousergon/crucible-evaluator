FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.12

# git — required for the pip ``git+https://...`` install of alpha-engine-lib
# (the base Lambda image ships no git). microdnf is the AL2023 minimal pkg
# manager. Mirrors the research/predictor Lambda images.
RUN microdnf install -y git && microdnf clean all

# Dependencies. alpha-engine-lib is the AUTHORITATIVE pin here — keep this
# `@vX.Y.Z` in lockstep with requirements.txt (the grep below strips the lib
# pin + Lambda-runtime-provided deps from requirements before `pip install -r`,
# so a requirements-only bump won't propagate to the image). The container has
# no 250MB unzip limit, so we install the full [quant-stats] extra
# (numpy+pandas+scipy) verbatim.
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir "alpha-engine-lib[quant-stats] @ git+https://github.com/cipher813/alpha-engine-lib@v0.52.0" && \
    grep -vE "^#|^$|^pytest|^pytest-cov|^moto|^python-dotenv|^boto3|^botocore|^s3transfer|^alpha-engine-lib" requirements.txt > /tmp/req-lambda.txt && \
    pip install --no-cache-dir -r /tmp/req-lambda.txt && \
    rm -rf /root/.cache/pip /tmp/req-lambda.txt

# Application code (Layer B grading + Layer C director skeleton).
COPY grading/ ${LAMBDA_TASK_ROOT}/grading/
COPY director/ ${LAMBDA_TASK_ROOT}/director/

# Lambda entrypoint: the grading-layer producer. Builds the Report Card v2 and
# writes evaluator/{date}/report_card.json. (The Director, Part II, will add its
# own handler to the same image.)
CMD ["grading.handler.handler"]
