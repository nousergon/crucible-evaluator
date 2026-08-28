"""Canonical sector labels for benchmark-relative attribution.

alpha-engine-config-I8188 deliverables 6-7. The live book's own sector labels
are NOT internally consistent — measured across the 119 sessions in
``trades/eod_pnl.csv``, ``positions_snapshot`` carries all of:

    'Financial' AND 'Financials'
    'Health Care' AND 'Healthcare'
    'Information Technology' AND 'Technology'
    'Consumer Discretionary' (GICS) vs 'Consumer Cyclical' (yfinance)
    'Consumer Staples' (GICS) vs 'Consumer Defensive' (yfinance)
    plus 'Broad Market / Index', 'Unknown' and ''

Two labels for one sector do not merely look untidy: in a Brinson attribution
they SPLIT one sector into two groups, one of which then has a benchmark weight
of zero, and the whole of its return lands in the allocation effect as if the
book had deliberately over-weighted a sector the benchmark does not hold. The
number would be wrong in the most confident-looking way available.

The canonical set is the ELEVEN labels the benchmark weights are published
under — ``market_data/sectors/latest.json``'s ``spy_sector_weights``, which
yfinance emits from SPY's own fund data — because that is the only side of the
comparison whose vocabulary we do not control. The sector ETF for each is the
same map ``nousergon-data/collectors/universe_returns.py`` already uses; it is
mirrored here rather than reinvented, and belongs in ``nousergon-lib`` on the
next adoption (policy-shared-code second-adoption trigger; filed).

An unrecognised label RAISES. A silent fallthrough to 'Unknown' is how a
mis-spelled sector becomes a phantom benchmark under-weight.
"""

from __future__ import annotations

# The eleven canonical benchmark sectors, and the SPDR sector ETF that prices
# each one. Closes for all eleven are live at
# ``s3://alpha-engine-research/market_data/close_history/{ETF}.json`` with
# ``adjustment_basis: dividend_adjusted`` — i.e. TOTAL return, which is what the
# benchmark leg must be (I8188 defect 3).
SECTOR_TO_ETF: dict[str, str] = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

CANONICAL_SECTORS = tuple(SECTOR_TO_ETF)

# The book's own sleeve for an index/ETF holding. It is NOT a sector: SPY is
# held as a core position (portfolio-optimizer cutover, 2026-05-13) and is
# benchmarked against SPY, so it is carved out rather than classified.
INDEX_SLEEVE = "Index"

# Names the producer could not classify. Kept as its OWN group so its weight is
# visible and its benchmark weight of zero is an admitted gap rather than a
# silent allocation bet.
UNCLASSIFIED = "Unclassified"

_ALIASES: dict[str, str] = {
    # GICS ↔ yfinance vocabulary
    "information technology": "Technology",
    "technology": "Technology",
    "financials": "Financial Services",
    "financial": "Financial Services",
    "financial services": "Financial Services",
    "health care": "Healthcare",
    "healthcare": "Healthcare",
    "consumer discretionary": "Consumer Cyclical",
    "consumer cyclical": "Consumer Cyclical",
    "consumer staples": "Consumer Defensive",
    "consumer defensive": "Consumer Defensive",
    "energy": "Energy",
    "industrials": "Industrials",
    "materials": "Basic Materials",
    "basic materials": "Basic Materials",
    "utilities": "Utilities",
    "real estate": "Real Estate",
    "communication services": "Communication Services",
    "communications": "Communication Services",
    "telecommunication services": "Communication Services",
    # explicit non-sectors
    "broad market / index": INDEX_SLEEVE,
    "broad market": INDEX_SLEEVE,
    "index": INDEX_SLEEVE,
    "etf": INDEX_SLEEVE,
    "unknown": UNCLASSIFIED,
    "": UNCLASSIFIED,
}


class UnknownSectorError(ValueError):
    """A sector label no canonical mapping covers. Never silently bucketed."""


def canonical_sector(label: str | None) -> str:
    """Map any label the book emits onto the benchmark's vocabulary.

    Returns one of :data:`CANONICAL_SECTORS`, :data:`INDEX_SLEEVE` or
    :data:`UNCLASSIFIED`. Raises :class:`UnknownSectorError` on anything else —
    a new label must be mapped deliberately, because the default outcome of
    guessing is a phantom benchmark under-weight the size of the position.
    """
    key = (label or "").strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    # A label that is already canonical (exact, case-insensitive).
    for canon in CANONICAL_SECTORS:
        if key == canon.lower():
            return canon
    raise UnknownSectorError(
        f"sector label {label!r} is not mapped to any of the eleven canonical "
        f"benchmark sectors {CANONICAL_SECTORS}, to {INDEX_SLEEVE!r}, or to "
        f"{UNCLASSIFIED!r}. Add it to grading/sectors.py::_ALIASES — an "
        "unmapped label becomes a group with zero benchmark weight, and its "
        "whole return is then reported as an allocation bet the book never made."
    )
