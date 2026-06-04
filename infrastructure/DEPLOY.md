# alpha-engine-evaluator — deployment

The evaluator's grading layer (Layer B) runs as a Lambda invoked by the Saturday
Step Function, after the terminal evaluation states, to produce
`s3://alpha-engine-research/evaluator/{date}/report_card.json` (the Report Card
v2 substrate the Director will later consume). The Director (Layer C, Part II)
ships in the **same container image** as a second Lambda
(`alpha-engine-evaluator-director`, CMD override `director.handler.handler`) —
the fleet's image-share pattern (cf. the research eval-judge / rationale-
clustering Lambdas). It runs as the **final Saturday-SF task, after `ReportCard`**,
and is **flag-gated dormant** (`DIRECTOR_ENABLED` off) until an operator flips it.

This doc is the **runbook**; provisioning the Lambda + IAM role + the SF state is
a live-infra step (operator-run — not done from CI). Nothing here mutates live
infra on its own.

## Lambda

- **Function name:** `alpha-engine-evaluator` (proposed)
- **Runtime:** `python3.12`
- **Handler:** `grading.handler.handler`
- **Entry contract:** `handler(event, context)` where `event` may carry:
  - `date` — run date (ISO; the SF passes the normalized RUN_DATE, mirroring the
    backtester). Falls back to `EVALUATOR_RUN_DATE` env, then
    `alpha_engine_lib.dates.now_dual().trading_day`.
  - `bucket` — S3 bucket (default `alpha-engine-research`; or `EVALUATOR_BUCKET`).
  - `write` — default `true`; the SF run writes the card, a dry-run sets `false`.
- **Returns:** a compact summary — `tiles_overall_status`, per-tile `tile_status`,
  `real_graded` counts, and the written `report_card_key`.
- **Env:** `EVALUATOR_BUCKET` (optional), `EVALUATOR_RUN_DATE` (optional pin).

### Packaging + publish — container image (fleet pattern)

The runtime deps (numpy / pandas via `alpha-engine-lib[quant-stats]`) need
Amazon-Linux manylinux wheels, so — like the research/predictor Lambdas — this
ships as a **container image** (`Dockerfile` → ECR), not a zip. `infrastructure/
deploy.sh` does the whole cycle:

```
./infrastructure/iam/apply.sh          # one-time: create the execution role
./infrastructure/deploy.sh             # build → ECR push → create/update Lambda
                                       # → canary (write=false) → publish + live alias
```

- **Image:** `Dockerfile` (`public.ecr.aws/lambda/python:3.12`, `git` for the
  lib pip-install, `CMD ["grading.handler.handler"]`). The lib `@vX.Y.Z` pin in
  the Dockerfile is authoritative — keep in lockstep with `requirements.txt`.
- **Canary:** `deploy.sh` invokes with `{"write": false}` and only promotes the
  `live` alias when the returned `status == "ok"`.
- **Config:** timeout 300s, memory 1024 MB, env `EVALUATOR_BUCKET`.

## IAM (least-privilege)

The execution role needs:
- **S3 read:** `backtest/*`, `predictor/*`, `trades/*`, `config/*`,
  `signals/*` on `alpha-engine-research` (the tiles' sources).
- **S3 write:** `evaluator/*` on `alpha-engine-research` (the report card).
- **S3 read/write:** `director/*` on `alpha-engine-research` (the Director's
  action plan + carry-over ledger).
- **SSM read:** `ssm:GetParameter` / `ssm:GetParametersByPath` on
  `parameter/alpha-engine/*` — the Director fetches `ANTHROPIC_API_KEY` from
  `/alpha-engine/ANTHROPIC_API_KEY` via `alpha_engine_lib.secrets.get_secret`
  (mirrors the research `alpha-engine-ssm-read` grant; no `kms:Decrypt` needed
  for this parameter). No `ANTHROPIC_API_KEY` env wiring — the key flows from
  SSM at request time.
- **Step Functions read** (for the substrate `sf_success_rate_4w` follow-up):
  `states:ListExecutions`, `states:DescribeExecution` on the 3 pipeline ARNs.
- Standard `AWSLambdaBasicExecutionRole` (CloudWatch Logs).

## Step Function wiring (alpha-engine infra repo)

Add a `ReportCard` state to `alpha-engine-saturday-pipeline`, **after** the
terminal evaluation states (so it reads fresh grades), invoking this Lambda with
`{"date": "<RUN_DATE>"}`.

- **Failure isolation (load-bearing):** the state has its **own `Catch`** routing
  to a non-fatal continue — a grading failure must NEVER fail the run that
  produced the real trading artifacts (director-plan §5). The failure surfaces
  via the SF Catch + the freshness monitor on
  `evaluator/{date}/report_card.json`, not by breaking the pipeline.
- **Resource:** Lambda (pure S3 reads + computation + one S3 write; weekly
  cadence; no large on-disk DBs) — fits the Lambda-vs-spot heuristic.

## Director (Layer C, Part II) — image-share + flag flip

The Director is the **second function on the same image**, deployed by the same
`deploy.sh` run (after the grading function's `live` alias is set):

- **Function:** `alpha-engine-evaluator-director`, CMD override
  `director.handler.handler`, same role/image, timeout 300s, memory 1024 MB.
- **Created dormant:** `DIRECTOR_ENABLED` is unset → the handler returns
  `{"status": "disabled"}` (no LLM call, no key use, no cost). `deploy.sh`'s
  dormant canary asserts exactly that before promoting the alias.
- **SF wiring:** a non-fatal `Director` state runs **after `ReportCard`**
  (`ReportCard → Director → CheckShellRunNotify`), invoking
  `alpha-engine-evaluator-director:live` with `{"date": "<RUN_DATE>"}`, with its
  **own `Catch`** (advisory failure must never break the run).
- **Flip ON (operator, after a clean Saturday cycle):**

  ```
  aws lambda update-function-configuration --function-name alpha-engine-evaluator-director \
    --environment 'Variables={EVALUATOR_BUCKET=alpha-engine-research,DIRECTOR_ENABLED=true}'
  ```

  Read at request time → no redeploy. The flip must set **both** vars (the
  `--environment` map is a full replace). Flip OFF by dropping `DIRECTOR_ENABLED`.

## Local invoke

```
python -m grading.handler --date 2026-06-07            # builds + writes
python -m grading.handler --date 2026-05-30 --no-write # summary only, no S3 write
python -m grading.aggregate --date 2026-06-07 --compare # parity vs backtester grading.json

# Director (needs the SSM key or ANTHROPIC_API_KEY in env; --no-write prints the plan)
DIRECTOR_ENABLED=1 python -m director.handler --date 2026-05-30 --no-write
```

## Cutover (later Phase C step)

Once the evaluator's report card holds parity with the backtester's in-process
`grading.json` for ≥1 Saturday cycle: the dashboard RC-v2 surface reads
`evaluator/{date}/report_card.json`, and the backtester's `evaluate.py` drops its
in-process grading call. Until then both run in parallel (S3-contract safety).
