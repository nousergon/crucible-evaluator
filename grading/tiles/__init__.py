"""tiles — per-module Report Card v2 builders.

Each tile reads its source artifact(s) over S3, computes its components as
``MetricRecord`` objects (value + CI + N + target/red-line + status), and
returns a tile summary via ``grading.module_agg.build_tile``. All NINE tiles
are live (registry: ``grading/aggregate.py``): portfolio_outcome, predictor,
research, executor, backtester, substrate, agent, behavioral, director_quality.
Historical numbering skips 8 (indices 0–7 then 9) — there is no Tile 8.
"""
