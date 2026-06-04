"""tiles — per-module Report Card v2 builders.

Each tile reads its source artifact(s) over S3, computes its components as
``MetricRecord`` objects (value + CI + N + target/red-line + status), and
returns a tile summary via ``grading.module_agg.build_tile``. Portfolio Outcome
(Tile 0) is the first; Predictor / Executor / Backtester / Substrate / Agent
tiles follow in subsequent Phase C increments.
"""
