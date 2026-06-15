# alpha-engine-evaluator

> Part of [**Nous Ergon**](https://nousergon.ai) — a harness for rigorous AI/ML experiments in finance: an equity research-and-trading system instrumented end-to-end. Repo and S3 names use the underlying project name `alpha-engine`.

[![Part of Nous Ergon](https://img.shields.io/badge/Part_of-Nous_Ergon-1a73e8?style=flat-square)](https://nousergon.ai)
[![Python](https://img.shields.io/badge/python-3.12+-blue?style=flat-square)](https://www.python.org/)
[![Anthropic Claude](https://img.shields.io/badge/Anthropic_Claude-1a73e8?style=flat-square)](https://www.anthropic.com/)
[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-yellow?style=flat-square)](LICENSE)

The system's **measurement & evaluation layer**: it grades every module's output into a comprehensive report card, and (the **Director**) reviews those grades each week to propose an advisory action plan for refining the system.

> System overview, Step Function orchestration, and module relationships live in [`alpha-engine-docs`](https://github.com/nousergon/nousergon-docs).

## What this does

The "evaluator" is three conceptual layers the system previously conflated:

- **A — Measure & Tune** (stays in [`alpha-engine-backtester`](https://github.com/nousergon/crucible-backtester)): simulation, optimizers, and the per-module raw analyses. Produces raw metric artifacts + auto-tuned configs.
- **B — Judge** (this repo, `grading/`): aggregates the raw analysis artifacts into the **System Report Card v2** — every component a `MetricRecord` carrying value, confidence interval, sample size vs floor, target/red-line, trend, and a derived status (`GREEN`/`WATCH`/`RED` or a specific `N/A-*` reason). Built natively here, reading producer artifacts from S3.
- **C — Direct** (this repo, `director/`): a weekly **Director** agent reviews the graded report card and proposes a structured, **advisory** action plan with carry-over tracking. It *proposes* — it never writes to live trading config and never self-merges. Its plan is archived to the console and (when enabled) opened as an approval-gated PR against the planning docs for human review.

## Design invariants

- **No write path to live trading behavior.** The backtester's auto-apply loop owns continuous parameter tuning; the Director only proposes (a reviewable PR), never moves a trading parameter.
- **The letter is derived, not load-bearing.** The source of truth is the metric + its CI + threshold; letters are a summary surface.
- **`N/A` means something specific.** Four distinct engineering states (not-implemented / not-run / low-N / missing-input) instead of a generic "insufficient data."

## Module layout

| Path | Layer | Role |
|---|---|---|
| `grading/` | B | report-card aggregation: raw artifacts → `MetricRecord` → tiles |
| `director/` | C | weekly advisory action-plan agent + carry-over ledger |
| `lambda/` | — | Saturday Step Function entrypoint (added with the Director) |
| `infrastructure/` | — | deploy + IAM |

Shared contracts (`MetricRecord`, statistical intervals, agent schemas) live in [`alpha-engine-lib`](https://github.com/nousergon/nousergon-lib).

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest tests/ -v
```

Secrets load from SSM via `alpha_engine_lib.secrets.get_secret()` — never committed. The Director's prompt template is proprietary and gitignored; it loads at runtime.
