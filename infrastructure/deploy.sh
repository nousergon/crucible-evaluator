#!/usr/bin/env bash
# deploy.sh — build + push the alpha-engine-evaluator container image and
# deploy the grading Lambda, then publish a version + point the `live` alias at
# it. Mirrors the research/predictor container-image deploy pattern.
#
# Prereqs (one-time, operator): ./infrastructure/iam/apply.sh (creates the role).
#
#   ./infrastructure/deploy.sh            # build, push, deploy, canary, alias
#   ./infrastructure/deploy.sh --no-canary
set -euo pipefail

FUNCTION="alpha-engine-evaluator"
HANDLER_CMD='["grading.handler.handler"]'
REGION="${AWS_REGION:-us-east-1}"
TIMEOUT=300
MEMORY=1024
NO_CANARY=false
[[ "${1:-}" == "--no-canary" ]] && NO_CANARY=true

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$REGION")
ROLE_ARN="${LAMBDA_ROLE_ARN:-arn:aws:iam::${ACCOUNT_ID}:role/alpha-engine-evaluator-role}"
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${FUNCTION}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== Building $FUNCTION image (linux/amd64) ==="
docker build --platform linux/amd64 --provenance=false -t "$FUNCTION:latest" .

echo "=== ECR login + ensure repo ==="
aws ecr get-login-password --region "$REGION" | \
  docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
aws ecr describe-repositories --repository-names "$FUNCTION" --region "$REGION" &>/dev/null || \
  aws ecr create-repository --repository-name "$FUNCTION" --region "$REGION" >/dev/null

echo "=== Push image ==="
docker tag "$FUNCTION:latest" "$ECR_REPO:latest"
docker push "$ECR_REPO:latest"
IMAGE_URI="$ECR_REPO:latest"

echo "=== Deploy Lambda ($FUNCTION) ==="
if aws lambda get-function --function-name "$FUNCTION" --region "$REGION" &>/dev/null; then
  aws lambda update-function-code --function-name "$FUNCTION" \
    --image-uri "$IMAGE_URI" --region "$REGION" \
    --query 'LastUpdateStatus' --output text
  aws lambda wait function-updated --function-name "$FUNCTION" --region "$REGION"
  aws lambda update-function-configuration --function-name "$FUNCTION" \
    --timeout "$TIMEOUT" --memory-size "$MEMORY" \
    --environment "Variables={EVALUATOR_BUCKET=alpha-engine-research}" \
    --region "$REGION" --query 'LastUpdateStatus' --output text
  aws lambda wait function-updated --function-name "$FUNCTION" --region "$REGION"
else
  aws lambda create-function --function-name "$FUNCTION" \
    --package-type Image --code "ImageUri=$IMAGE_URI" \
    --role "$ROLE_ARN" --timeout "$TIMEOUT" --memory-size "$MEMORY" \
    --environment "Variables={EVALUATOR_BUCKET=alpha-engine-research}" \
    --region "$REGION" --query 'FunctionArn' --output text
  aws lambda wait function-active --function-name "$FUNCTION" --region "$REGION"
fi

if ! $NO_CANARY; then
  echo "=== Canary invoke (write=false — builds the card, no S3 write) ==="
  aws lambda invoke --function-name "$FUNCTION" \
    --payload "$(echo '{"write": false}' | base64)" \
    --region "$REGION" /tmp/evaluator-canary.json --query 'StatusCode' --output text
  STATUS=$(python3 -c "import json; print(json.load(open('/tmp/evaluator-canary.json')).get('status'))")
  echo "  canary status: $STATUS"
  [[ "$STATUS" == "ok" ]] || { echo "CANARY FAILED — not promoting alias"; cat /tmp/evaluator-canary.json; exit 1; }
fi

echo "=== Publish version + point live alias ==="
VERSION=$(aws lambda publish-version --function-name "$FUNCTION" --region "$REGION" --query 'Version' --output text)
# Promote :live — try update, fall back to create. Mirrors the predictor/research
# deploy.sh idiom; needs only lambda:UpdateAlias + CreateAlias (NOT GetAlias, which
# the shared github-actions-lambda-deploy role does not grant — a get-alias gate
# here fails on AccessDenied once the alias exists and wrongly retries create).
aws lambda update-alias --function-name "$FUNCTION" --name live --function-version "$VERSION" --region "$REGION" --query 'AliasArn' --output text 2>/dev/null || \
  aws lambda create-alias --function-name "$FUNCTION" --name live --function-version "$VERSION" --region "$REGION" --query 'AliasArn' --output text
echo "=== Deployed $FUNCTION:live (version $VERSION) ==="

# ── Director (Layer C) — shares THIS image with a CMD override ────────────────
# Same ECR image, a SECOND Lambda function whose CMD points at the Director
# handler. Mirrors the alpha-engine-research eval-judge / rationale-clustering
# pattern (one runner image, per-Lambda ``--image-config`` CMD overrides) — the
# institutional way to run a 2nd handler from one image. The function is created
# DORMANT: ``DIRECTOR_ENABLED`` is unset (off), so the handler is a no-op
# (returns ``status: disabled``) until an operator flips the flag after a clean
# Saturday cycle:
#   aws lambda update-function-configuration --function-name alpha-engine-evaluator-director \
#     --environment 'Variables={EVALUATOR_BUCKET=alpha-engine-research,DIRECTOR_ENABLED=true}'
# (no redeploy needed — the flag is read at request time).
DIRECTOR_FUNCTION="${FUNCTION}-director"
DIRECTOR_CMD='["director.handler.handler"]'
echo "=== Deploy Director Lambda ($DIRECTOR_FUNCTION — image-share, CMD override) ==="
if aws lambda get-function --function-name "$DIRECTOR_FUNCTION" --region "$REGION" &>/dev/null; then
  aws lambda update-function-code --function-name "$DIRECTOR_FUNCTION" \
    --image-uri "$IMAGE_URI" --region "$REGION" --query 'LastUpdateStatus' --output text
  aws lambda wait function-updated --function-name "$DIRECTOR_FUNCTION" --region "$REGION"
  # NOTE: preserve any operator-set DIRECTOR_ENABLED — do NOT reset the env here.
  aws lambda update-function-configuration --function-name "$DIRECTOR_FUNCTION" \
    --image-config "Command=$DIRECTOR_CMD" \
    --timeout "$TIMEOUT" --memory-size "$MEMORY" \
    --region "$REGION" --query 'LastUpdateStatus' --output text
  aws lambda wait function-updated --function-name "$DIRECTOR_FUNCTION" --region "$REGION"
else
  aws lambda create-function --function-name "$DIRECTOR_FUNCTION" \
    --package-type Image --code "ImageUri=$IMAGE_URI" \
    --image-config "Command=$DIRECTOR_CMD" \
    --role "$ROLE_ARN" --timeout "$TIMEOUT" --memory-size "$MEMORY" \
    --environment "Variables={EVALUATOR_BUCKET=alpha-engine-research}" \
    --region "$REGION" --query 'FunctionArn' --output text
  aws lambda wait function-active --function-name "$DIRECTOR_FUNCTION" --region "$REGION"
fi

if ! $NO_CANARY; then
  echo "=== Director dormant canary (flag off → expect status: disabled) ==="
  aws lambda invoke --function-name "$DIRECTOR_FUNCTION" \
    --payload "$(echo '{"date": "2026-05-30"}' | base64)" \
    --region "$REGION" /tmp/director-canary.json --query 'StatusCode' --output text
  DSTATUS=$(python3 -c "import json; print(json.load(open('/tmp/director-canary.json')).get('status'))")
  echo "  director canary status: $DSTATUS"
  [[ "$DSTATUS" == "disabled" ]] || { echo "DIRECTOR CANARY UNEXPECTED (want 'disabled' while flag off) — not promoting"; cat /tmp/director-canary.json; exit 1; }
fi

echo "=== Publish Director version + point live alias ==="
DVERSION=$(aws lambda publish-version --function-name "$DIRECTOR_FUNCTION" --region "$REGION" --query 'Version' --output text)
# Promote :live — try update, fall back to create (see the grading alias note above:
# avoids needing lambda:GetAlias on the shared deploy role).
aws lambda update-alias --function-name "$DIRECTOR_FUNCTION" --name live --function-version "$DVERSION" --region "$REGION" --query 'AliasArn' --output text 2>/dev/null || \
  aws lambda create-alias --function-name "$DIRECTOR_FUNCTION" --name live --function-version "$DVERSION" --region "$REGION" --query 'AliasArn' --output text
echo "=== Deployed $DIRECTOR_FUNCTION:live (version $DVERSION, DORMANT — DIRECTOR_ENABLED off) ==="
