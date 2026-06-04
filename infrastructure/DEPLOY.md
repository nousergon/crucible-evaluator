# alpha-engine-evaluator — deployment

The evaluator's grading layer (Layer B) runs as a Lambda invoked by the Saturday
Step Function, after the terminal evaluation states, to produce
`s3://alpha-engine-research/evaluator/{date}/report_card.json` (the Report Card
v2 substrate the Director will later consume). The Director (Layer C) is added
to the same Lambda in Part II.

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

## Local invoke

```
python -m grading.handler --date 2026-06-07            # builds + writes
python -m grading.handler --date 2026-05-30 --no-write # summary only, no S3 write
python -m grading.aggregate --date 2026-06-07 --compare # parity vs backtester grading.json
```

## Cutover (later Phase C step)

Once the evaluator's report card holds parity with the backtester's in-process
`grading.json` for ≥1 Saturday cycle: the dashboard RC-v2 surface reads
`evaluator/{date}/report_card.json`, and the backtester's `evaluate.py` drops its
in-process grading call. Until then both run in parallel (S3-contract safety).
