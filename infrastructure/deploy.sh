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
if aws lambda get-alias --function-name "$FUNCTION" --name live --region "$REGION" &>/dev/null; then
  aws lambda update-alias --function-name "$FUNCTION" --name live --function-version "$VERSION" --region "$REGION" --query 'AliasArn' --output text
else
  aws lambda create-alias --function-name "$FUNCTION" --name live --function-version "$VERSION" --region "$REGION" --query 'AliasArn' --output text
fi
echo "=== Deployed $FUNCTION:live (version $VERSION) ==="
