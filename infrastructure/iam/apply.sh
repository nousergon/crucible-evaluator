#!/usr/bin/env bash
# apply.sh — create/refresh the alpha-engine-evaluator Lambda execution role and
# its inline policy from the codified JSON in this directory. Idempotent.
#
#   ./infrastructure/iam/apply.sh            # apply
#   ./infrastructure/iam/apply.sh --dry-run  # print intended actions only
#
# Codified IAM is the single writer (mirrors the alpha-engine fleet convention):
# the deploy script never touches IAM; role/policy changes land here.
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ROLE="alpha-engine-evaluator-role"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/${ROLE}"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

run() { if $DRY_RUN; then echo "DRY-RUN: $*"; else eval "$*"; fi; }

echo "=== Applying IAM for ${ROLE} (region ${REGION}) ==="

# 1. Role (trust policy is a one-time bootstrap; create-or-skip).
if aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  echo "  role exists"
else
  run "aws iam create-role --role-name '$ROLE' \
    --assume-role-policy-document 'file://${DIR}/trust-policy.json' \
    --description 'alpha-engine-evaluator grading/director Lambda execution role' \
    --region '$REGION' >/dev/null"
  echo "  role created"
fi

# 2. CloudWatch Logs via the AWS-managed basic-execution policy.
run "aws iam attach-role-policy --role-name '$ROLE' \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole \
  --region '$REGION' >/dev/null"

# 3. Inline least-privilege policy (S3 module-artifact read + report-card write
#    + Step Functions execution-history read for sf_success_rate_4w).
run "aws iam put-role-policy --role-name '$ROLE' \
  --policy-name 'alpha-engine-evaluator-policy' \
  --policy-document 'file://${DIR}/alpha-engine-evaluator-policy.json' \
  --region '$REGION' >/dev/null"

echo "=== Done. Role ARN: arn:aws:iam::\$(aws sts get-caller-identity --query Account --output text):role/${ROLE} ==="
