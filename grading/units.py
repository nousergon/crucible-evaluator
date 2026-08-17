"""units.py — the report card's fixed unit vocabulary (config#7485).

``grading/metric_record.py::build_metric`` requires every value-bearing
``MetricRecord`` to declare its measurement unit — the RETURN/measurement
unit, never the statistical *kind* already carried by ``metric_type``
(e.g. an IC's ``metric_type`` is ``"ic"``; its ``unit`` is
``SPEARMAN_RHO`` or ``RANK_IC`` depending on which estimator produced it).

One vocabulary, defined once, imported everywhere a tile builds a metric —
so two components measuring the same thing never drift onto two spellings
of the same unit. Add a new constant here rather than writing a fresh
string literal at a call site.
"""

from __future__ import annotations

# --- Correlation / rank statistics -----------------------------------------
SPEARMAN_RHO = "spearman_rho"
RANK_IC = "rank_ic"
PEARSON_R = "pearson_r"

# --- Proportions -------------------------------------------------------------
# PCT: expressed 0-100. FRACTION: expressed 0-1. Read the estimator, not the
# name — `hit_rate` computed as `hits / n` is a FRACTION even though it reads
# like a percentage in prose.
PCT = "pct"
FRACTION = "fraction"

# --- Counts (bare integer/float counting a noun) -----------------------------
COUNT_SIGNALS = "signals"
COUNT_TICKERS = "tickers"
COUNT_TRADES = "trades"
COUNT_POSITIONS = "positions"
COUNT_COMPONENTS = "components"
COUNT_ALERTS = "alerts"
COUNT_RUNS = "runs"
COUNT_CYCLES = "cycles"
COUNT_EVENTS = "events"
COUNT_ISSUES = "issues"
COUNT_ROWS = "rows"
COUNT_ITEMS = "items"
COUNT_AGENTS = "agents"
COUNT_CHUNKS = "chunks"

# --- Domain-specific measurement units (named, not a generic count/ratio) ----
WET = "wet"  # groom pipeline's Weighted Effort Tokens spend unit
WET_PER_COMPLETION = "wet_per_completion"

# --- Money / size -------------------------------------------------------------
USD = "usd"
SHARES = "shares"

# --- Duration ------------------------------------------------------------------
MILLISECONDS = "milliseconds"
SECONDS = "seconds"
MINUTES = "minutes"
HOURS = "hours"
DAYS = "days"

# --- Returns / risk --------------------------------------------------------
LOG_RETURN = "log_return"
LOG_RETURN_21D = "log_return_21d"
ANNUALIZED_RATIO = "annualized_ratio"  # Sharpe/Sortino-style annualized ratio
BPS = "bps"
ZSCORE = "zscore"

# --- Statistical tests -------------------------------------------------------
PROBABILITY = "probability"  # p-values and other [0,1] probabilities
BRIER_SCORE = "brier_score"
ECE = "ece"  # expected calibration error

# --- Scores / indices (bounded model-quality composites, not a natural unit) -
SCORE_0_1 = "score_0_1"
SCORE_0_100 = "score_0_100"
SCORE_POINTS = "score_points"  # a std-dev/spread measured on the 0-100 composite scale

# --- Rates (a count normalized by another count / time, not a plain fraction)
RATE_PER_DAY = "rate_per_day"
RATIO = "ratio"  # generic dimensionless ratio of two like-quantities
