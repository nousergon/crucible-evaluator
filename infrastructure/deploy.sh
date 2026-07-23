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

# config#2348: stamp the image with the commit it's built from (CI uses
# $GITHUB_SHA; local dev falls back to the checked-out HEAD) so the weekly
# SF's Lambda-SHA drift probe can compare it against origin/main. Mirrors
# crucible-predictor/infrastructure/deploy.sh's GIT_SHA stamping exactly.
GIT_SHA="${GITHUB_SHA:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
echo "  Stamping image with GIT_SHA=${GIT_SHA}"

echo "=== Building $FUNCTION image (linux/amd64) ==="
docker build --platform linux/amd64 --provenance=false \
  --build-arg "GIT_SHA=${GIT_SHA}" \
  -t "$FUNCTION:latest" .

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
  # Canary via the shared krepis.aws invoke-canary CLI (config#1494, krepis
  # 0.7.0) instead of a bare `aws lambda invoke`. The CLI retries ONLY on the
  # throttle/concurrency signal and writes the response payload to --out; it
  # takes raw JSON (boto3 path — no base64/`--cli-binary-format`). A non-zero
  # CLI exit (non-throttle error or throttle exhaustion) refuses to promote —
  # PRE-promotion, so the live alias is untouched.
  # Boot probe (config#3058 follow-up), NOT a report-card build. The old
  # `{"write": false}` canary invoked build_report_card, which since config#3058
  # runs an UNCONDITIONAL hard input-freshness preflight (assert_input_freshness)
  # — correct for the weekly assessment, but it hard-fails whenever the current
  # trading day's weekly backtest artifacts (backtest/{date}/metrics.json, …)
  # aren't present yet, which is TRUE on every off-cycle deploy (e.g. a routine
  # dependency bump). A deploy canary must validate that the freshly-pushed image
  # BOOTS and its handler wiring resolves — not assert the state of weekly
  # production data (that gate belongs to the weekly run, where the preflight
  # stays fully in force). `{"action": "canary"}` is the freshness-agnostic
  # boot-probe path (grading/handler.py::_canary_probe); it returns status "ok".
  echo "=== Canary invoke (action=canary — boot probe, no card build, no S3 write) ==="
  python3 -m krepis.aws invoke-canary --function-name "$FUNCTION" \
    --payload '{"action": "canary"}' \
    --region "$REGION" --out /tmp/evaluator-canary.json \
    --max-attempts 6 --label "$FUNCTION-canary" \
    || { echo "CANARY UNINVOKABLE — not promoting alias"; exit 1; }
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
  # Flag-AGNOSTIC canary: invoke with dry_run=true so it's side-effect-free in
  # BOTH flag states (the bug that broke the 2026-06-07 deploys: the old canary
  # asserted `disabled`, but once an operator flips DIRECTOR_ENABLED=true the
  # handler runs a REAL plan — status `ok` — and the assert failed AND the canary
  # wrote a bogus action_plan + polluted the shared carry-over ledger).
  #   - flag off  → handler short-circuits before dry_run → status: disabled
  #   - flag on   → _dry_run_probe → status: dry_run (langchain import + SSM key
  #                 fetch + ledger read validated; NO Opus call, NO S3 write)
  echo "=== Director canary (dry_run — expect 'disabled' when flag off, 'dry_run' when on) ==="
  # Canary via the shared krepis.aws invoke-canary CLI (config#1494, krepis 0.7.0);
  # raw JSON payload (boto3 path). Non-zero CLI exit refuses to promote (the
  # Director live alias is untouched — PRE-promotion).
  python3 -m krepis.aws invoke-canary --function-name "$DIRECTOR_FUNCTION" \
    --payload '{"date": "2026-05-30", "dry_run": true}' \
    --region "$REGION" --out /tmp/director-canary.json \
    --max-attempts 6 --label "$DIRECTOR_FUNCTION-canary" \
    || { echo "DIRECTOR CANARY UNINVOKABLE — not promoting"; exit 1; }
  DSTATUS=$(python3 -c "import json; print(json.load(open('/tmp/director-canary.json')).get('status'))")
  echo "  director canary status: $DSTATUS"
  [[ "$DSTATUS" == "disabled" || "$DSTATUS" == "dry_run" ]] || { echo "DIRECTOR CANARY UNEXPECTED (want 'disabled' or 'dry_run') — not promoting"; cat /tmp/director-canary.json; exit 1; }
fi

echo "=== Publish Director version + point live alias ==="
DVERSION=$(aws lambda publish-version --function-name "$DIRECTOR_FUNCTION" --region "$REGION" --query 'Version' --output text)
# Promote :live — try update, fall back to create (see the grading alias note above:
# avoids needing lambda:GetAlias on the shared deploy role).
aws lambda update-alias --function-name "$DIRECTOR_FUNCTION" --name live --function-version "$DVERSION" --region "$REGION" --query 'AliasArn' --output text 2>/dev/null || \
  aws lambda create-alias --function-name "$DIRECTOR_FUNCTION" --name live --function-version "$DVERSION" --region "$REGION" --query 'AliasArn' --output text
echo "=== Deployed $DIRECTOR_FUNCTION:live (version $DVERSION; DIRECTOR_ENABLED preserved as set) ==="
