"""Effective sample size for metrics graded on overlapping forward-return windows.

alpha-engine-config-I11094 / -I10065 (decision-queue triage 2026-10-25: "grade
this family on effective N ... Below the power floor it reads insufficient-N
rather than RED or GREEN"; "one evaluator change should cover both").

WHY ROW COUNTS ARE THE WRONG N. Two components on the card graded against the
number of (ticker, prediction_date) rows:

  * ``predictor`` :: ``direction_accuracy_vs_majority_baseline`` and
    ``{up,down}_precision_vs_base_rate`` — N=3,533 on the 2026-10-23 card;
  * ``backtester`` :: ``backtest_vs_live_parity`` — N=689 rows on the same card.

Every one of those rows is labelled with a 21-trading-day forward return, and
~30 names are predicted on each date. Rows on the same date share one market
realisation; rows on consecutive dates share 20 of their 21 return days. They
are nowhere near independent, so a significance or floor check against the row
count claims precision the data does not have. The producer of the live leg
already says so in its own artifact: ``predictor/metrics/latest.json ::
realized_ic`` on 2026-10-24 reads ``n_rows: 689``, ``n_prediction_dates: 22``,
``n_effective: 1``, ``se_effective: 1.0`` — and its ``note`` field states that
"n_effective, not n_rows, is the count a significance claim may use".

THE FORMULA — ONE DEFINITION, THE PRODUCER'S. Effective N is the number of
NON-OVERLAPPING label windows the pool spans:

    n_effective = max(1, n_prediction_dates // horizon_days)   (0 when no dates)

This is crucible-backtester ``pipeline_common.push_predictor_rolling_metrics``
(alpha-engine-config-I8701) verbatim. It is the standard non-overlapping-
observations correction for overlapping h-period returns: with daily sampling
of h-day forward returns, only every h-th observation is independent (Hansen &
Hodrick 1980, "Forward Exchange Rates as Optimal Predictors of Future Spot
Rates", JPE 88(5); the overlap-induced autocorrelation of order h-1 that
Newey-West/Hansen-Hodrick standard errors correct for). Clustering by date
first (rows on one date are one draw of the market) and then counting disjoint
windows is the conservative form of the same correction. It is deliberately not
re-derived here: an artifact that DECLARES ``n_effective`` is read as declared,
and the formula is applied only to an artifact that declares
``n_prediction_dates`` and its horizon but not the result, so the evaluator and
the producer can never disagree about what the same pool's N is.

WHAT IS NOT DONE. An artifact that declares neither yields ``None`` —
UNDECLARED, never a fallback to the row count. Falling back to rows is the
defect; a caller grades an undeclared effective N as insufficient-N and says
which producer field is missing.
"""

from __future__ import annotations

#: The parity ruling's floor (alpha-engine-config-I10065, triage 2026-10-25):
#: "Grade on ``realized_ic.n_effective``. Below 2, the tile reads
#: insufficient-N." A single independent window cannot estimate a correlation's
#: dispersion at all (the producer's own ``se_effective`` is 1.0 at N=1).
PARITY_MIN_EFFECTIVE_N = 2


def non_overlapping_windows(n_prediction_dates, horizon_days) -> int | None:
    """``max(1, n_dates // horizon)`` — the producer's formula; None if not computable."""
    n_dates = _as_count(n_prediction_dates)
    horizon = _as_count(horizon_days)
    if n_dates is None or horizon is None or horizon < 1:
        return None
    if n_dates == 0:
        return 0
    return max(1, n_dates // horizon)


def declared_effective_n(block, horizon_days=None) -> int | None:
    """Effective N as an artifact block declares it, or None when it does not.

    Order: the block's own ``n_effective``; else ``n_prediction_dates`` over the
    block's ``horizon_days`` (or the ``horizon_days`` passed in, which the caller
    takes from the same artifact's declared horizon). Never the row count.
    """
    if not isinstance(block, dict):
        return None
    declared = _as_count(block.get("n_effective"))
    if declared is not None:
        return declared
    return non_overlapping_windows(
        block.get("n_prediction_dates"),
        block.get("horizon_days") if block.get("horizon_days") is not None else horizon_days,
    )


def _as_count(v) -> int | None:
    """A non-negative integer count, or None. Booleans and non-integral floats are refused."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, int):
        return v if v >= 0 else None
    if isinstance(v, float) and v.is_integer() and v >= 0:
        return int(v)
    return None
