# alpha-engine-evaluator — Evaluator Module

> System architecture, S3 layout, module overview, and cross-repo conventions: see [`~/Development/CLAUDE.md`](../CLAUDE.md). This file covers evaluator-specific operational details only.

## What this repo is

Measurement & evaluation layer. Layer B (`grading/`): Lambda reads S3 artifacts → 9 metric tiles (portfolio_outcome, research, predictor, executor, backtester, substrate, agent, behavioral, director_quality) → System Report Card v2 of MetricRecords (value, CI, N, target/red-line, trend, GREEN/WATCH/RED). Layer C (`director/`): weekly advisory agent reads the graded card → `DirectorWeeklyActionPlan` → S3 + approval-gated ROADMAP PR. **Harness contribution:** serves both experiment families (agentic + finance). Deploys via GHA → ECR → canary → `live` alias. Director proposes only — never writes live trading config.

## Stack

- Python 3.12, venv at `.venv/`
- Lambda container image (`public.ecr.aws/lambda/python:3.12`, `linux/amd64`), built + pushed to ECR, deployed via `aws lambda update-function-code` + version publish + `live` alias move
- `krepis[flow_doctor]>=0.11.2` — structured logging/flow-doctor error capture, dates (`now_dual`/`resolve_trading_day`), the runner-side `invoke-canary` CLI, and (director/retro.py) the provider-agnostic `krepis.llm` adapter for the Phase-G retro judge call
- `nousergon-lib[quant-stats,contracts]` (pinned via `git+https://...@v0.93.0`) — `MetricRecord` contract, statistical intervals, quant stats battery (Sharpe/Sortino/max_drawdown/PSR/DSR/CVaR/BH-FDR), agent schemas, secrets
- `langchain-anthropic` — Director's single structured Opus call for plan generation (`director/agent.py`, `DIRECTOR_MODEL`, locked to Opus); imported lazily so grading + the test suite don't require it when `DIRECTOR_ENABLED` is off
- boto3, pydantic, pyyaml, numpy, pandas
- pytest + pytest-cov + moto (mocked S3 for artifact-read integration tests)
- Secrets load from SSM via `alpha_engine_lib.secrets.get_secret()` — never committed. The Director's prompt template is proprietary and gitignored; falls back to `director/prompt_example.py` in this (public) repo when no tuned prompt is staged.

## Key files

```
grading/handler.py                  # Layer B Lambda entrypoint (Saturday SF state) — builds + writes evaluator/{date}/report_card.json; fail-loud, SF Catch makes it non-fatal
grading/aggregate.py                # build_report_card / write_report_card — assembles the 9 tiles into Report Card v2
grading/scorecard.py                # Report Card v2 core: MetricRecord aggregation, GREEN/WATCH/RED derivation
grading/metric_record.py            # MetricRecord-adjacent helpers (value, CI, N vs floor, target/red-line, trend)
grading/module_agg.py                # per-module aggregation helpers
grading/history.py                   # rolling history / trend computation
grading/artifacts.py                 # S3 artifact read helpers (raw producer outputs → tiles)
grading/iam_s3_contract.json         # declarative IAM S3 read/write contract for the grading Lambda role
grading/tiles/                       # 9 metric tiles: portfolio_outcome, research, predictor, executor, backtester, substrate, agent, behavioral, director_quality, groom
grading/producers/deploy_success.py  # deploy-success producer artifact
director/handler.py                  # Layer C Lambda entrypoint (final Saturday SF state, runs after ReportCard) — gated behind DIRECTOR_ENABLED (env, default OFF); OFF returns status:disabled
director/agent.py                    # build_action_plan — single structured Opus call (DIRECTOR_MODEL) producing DirectorWeeklyActionPlan
director/retro.py                    # Phase-G retro judge — Sonnet call via krepis.llm (judge != generator; separate model tier from agent.py's Opus)
director/carryover.py                # carry-over ledger load/merge/write across weekly runs
director/roadmap_pr.py               # approval-gated ROADMAP PR opener (Director proposes, never self-merges)
director/issue_filer.py              # GitHub issue filing + open/recently-closed proposal digests
director/emailer.py                  # weekly Director plan email delivery
director/schema.py                   # DirectorWeeklyActionPlan + related schemas
director/report_card_digest.py       # condenses the graded report card for the Director prompt
director/prompt_example.py           # public fallback prompt (real tuned prompt is gitignored, loaded from alpha-engine-config when staged)
director/retro_prompt_example.py     # public fallback retro-judge prompt
infrastructure/deploy.sh             # build image, push to ECR, deploy BOTH Lambdas (grading + director, same image, CMD override), canary each, move live alias — preserves operator-set DIRECTOR_ENABLED
infrastructure/README.md             # infra notes (IAM role setup, one-time apply)
Dockerfile                           # single image for both Lambdas; CMD grading.handler.handler; nousergon-lib installed in its own cache layer ahead of the rest of requirements.txt
flow-doctor.yaml                     # flow-doctor config (email + S3 alert channels), resolved at runtime; ${VAR} secrets come from SSM
tests/                                # pytest suite incl. consumer-contract tests (test_attractiveness_consumer_contract.py, test_apply_audit_consumer_contract.py), Dockerfile/lib-pin lockstep guards (test_dockerfile_handler_import_completeness.py, test_dockerfile_lib_pin_lockstep.py), IAM/S3 contract test, smoke test
```

## Deploy

`deploy.yml` triggers on push to `main` when `grading/**`, `director/**`, `requirements*.txt`, `Dockerfile*`, or `infrastructure/deploy.sh` change (path-filtered — docs/tests/CI-only changes don't redeploy). Builds one image, deploys both `alpha-engine-evaluator` (grading) and `alpha-engine-evaluator-director` Lambdas from it, canaries each (grading expects `status:ok`, director expects `status:disabled` while `DIRECTOR_ENABLED` is off), then moves the `live` alias. Manual redeploy: `workflow_dispatch` from the Actions UI, or locally `bash infrastructure/deploy.sh` (`--no-canary` to skip canaries).

Flipping `DIRECTOR_ENABLED` requires more than an env var edit: the Saturday SF invokes the `:live` alias, whose env is frozen at the published version — update `$LATEST` env, publish a new version, and move `live` to it.

## Tests

```bash
source .venv/bin/activate
PYTHONPATH=. pytest tests/ -v
```

CI (`ci.yml`) runs the full suite plus `pip-audit` (ignoring vulns listed in `.github/pip-audit-ignore.txt`) on every push/PR to `main`. All tests must pass before committing — including pre-existing failures.

## Repo-specific gotchas

- **Dockerfile / requirements.txt pin lockstep:** `nousergon-lib`'s pin is read directly out of `requirements.txt` inside the Dockerfile `RUN` line (`grep '^nousergon-lib' requirements.txt`) rather than hardcoded a second time — a prior version hardcoded a stale duplicate pin that silently drifted (root-caused + fixed 2026-07-11). `test_dockerfile_lib_pin_lockstep.py` guards this.
- **Director prompt is proprietary and gitignored.** The image builds purely from this public repo — no proprietary config/prompt is staged from `alpha-engine-config` at deploy time. If a tuned Director prompt is ever wanted in production, mirror the predictor/research `CONFIG_REPO_DIR` staging pattern.
- **`N/A` on a tile is not generic "insufficient data."** Four distinct engineering states are distinguished: not-implemented / not-run / low-N / missing-input. Preserve this distinction when adding or editing tiles.
- **No write path to live trading behavior.** The backtester's auto-apply loop owns continuous parameter tuning; the Director only proposes a reviewable PR — never moves a trading parameter directly.
