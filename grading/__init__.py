"""grading — Layer B: the System Report Card v2 aggregation.

Reads the raw per-module analysis artifacts that the backtester and predictor
persist to S3, and aggregates them into graded ``MetricRecord`` tiles (research /
predictor / executor / backtester / substrate / agent / portfolio outcome).

Built natively here (not lifted from the backtester) per the measurement-first
plan: the backtester produces + persists raw analyses; this layer judges them.

The per-component contract (``MetricRecord``) and the shared status/CI semantics
live in ``nousergon_lib.metrics`` + ``nousergon_lib.quant.stats.intervals``
so producer and consumers agree. This package adds the evaluator-specific
aggregation: critical-gate module roll-up and BH-FDR over each tile's family.

See ``alpha-engine-docs/private/system-report-card-revamp-260522.md`` and
``director-implementation-plan-260604.md`` §6 Phase C.
"""
