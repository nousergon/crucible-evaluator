"""Benchmark-relative attribution for the LIVE book.

alpha-engine-config-I8188 deliverables 6 and 7.

**6 — Brinson-Fachler that reconciles to active return with a zero residual.**
The book's only attribution today is one ``alpha_contribution_usd`` per name
(``weight × (r_i − R_spy)``), which is neither allocation nor selection, plus a
sector rollup that never subtracts a benchmark weight — a *contribution*
breakdown, not an *active* one. The engine for the real thing already exists,
unused: ``nousergon_lib.quant.attribution.brinson_fachler`` and its Cariño
``link_periods``. Grepped fleet-wide, they had **zero consumers**. This module
is the consumer, so the fleet gets one implementation rather than a second.

**7 — SPY holdings excluded, and a beta-adjusted benchmark.** The book holds
SPY as a core position (portfolio-optimizer cutover, 2026-05-13) and is
benchmarked against SPY, so that sleeve's active return is mechanically ~0 and
dilutes the measurement of everything else. It is carved out and reported
separately. Beta-adjusted alpha is reported beside the raw active return because
the strategy's pillars ARE factors: a book running β≈0.63 is not comparable to
SPY at face value in either direction.

INPUTS — all already in S3, no new producer and no new vendor feed:

  ``trades/eod_pnl.csv``                        the book: NAV, cash, the named
                                                P&L lines PR490 added, and the
                                                per-name ``positions_snapshot``
  ``market_data/sectors/latest.json``           ``spy_sector_weights`` (yfinance
                                                SPY fund data, 11 sectors)
  ``market_data/close_history/{ETF}.json``      the eleven SPDR sector ETFs,
                                                ``dividend_adjusted`` — i.e.
                                                TOTAL return, matching the
                                                benchmark basis I8188 defect 3
                                                established for SPY itself

THE ONE APPROXIMATION, STATED. The benchmark leg is a **sector-ETF proxy** for
SPY: SPY's constituent weights are not free data and exist nowhere in the fleet.
So the decomposition ties exactly to ``R_p − R_b_proxy``, and the gap between
the proxy and SPY's own total return is published as
``benchmark_proxy_tracking_pct`` rather than absorbed. The three quantities
close exactly:

    active_return = bf_active_return + benchmark_proxy_tracking

and the BF residual against ``bf_active_return`` is zero by construction. The
benchmark WEIGHTS are a current snapshot applied across the window (there is no
dated history of them); ``benchmark_weights_as_of`` and
``benchmark_weights_are_current_proxy`` say so in the artifact.

NO PLUG. Every dollar entering the portfolio leg is a NAMED line from the
schema. What does not reconcile is published as ``input_closure_usd`` and
bounded — it is not folded into cash. That is the whole point of I8188: a
residual with no bound is an accounting hole, and one that is *defined* as the
remainder is worse, because its check can never fail.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
from typing import Any

import boto3
from botocore.exceptions import ClientError
from nousergon_lib.quant.attribution import BrinsonResult, brinson_fachler, link_periods

from grading.sectors import (
    INDEX_SLEEVE,
    SECTOR_TO_ETF,
    UNCLASSIFIED,
    canonical_sector,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
EOD_PNL_KEY = "trades/eod_pnl.csv"
SECTORS_KEY = "market_data/sectors/latest.json"
CLOSE_HISTORY_PREFIX = "market_data/close_history/"
ATTRIBUTION_LATEST_KEY = "evaluator/latest/attribution.json"

# The ticker held as the benchmark sleeve. Carved out of active return.
BENCHMARK_TICKER = "SPY"

# Closure bound on the portfolio leg. The named lines must account for the NAV
# move to within this, or the attribution is reported as NOT CLOSED rather than
# published with a silent gap. 25bp of NAV sits above the measured true-residual
# band (I8188: +$522 over 74 sessions after the sleeves are lifted) and below
# the per-session residual hard gate (50bp), so this fires only on a day the
# executor's own gate would also have something to say about.
INPUT_CLOSURE_NAV_BPS = 25.0

# Below this the linked attribution is not published — a handful of sessions
# cannot support a sector decomposition and would render as confident noise.
MIN_SESSIONS = 20

# Groups the benchmark does not hold. See the off_benchmark note below.
_OFF_BENCHMARK_GROUPS = ("Cash", INDEX_SLEEVE)


# ─────────────────────────────────────────────────────────────────────────────
# S3 inputs
# ─────────────────────────────────────────────────────────────────────────────

def _get_json(s3, bucket: str, key: str) -> dict | None:
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            logger.warning("attribution input absent: s3://%s/%s", bucket, key)
            return None
        raise
    return json.loads(resp["Body"].read().decode("utf-8"))


def read_eod_pnl_rows(bucket: str, s3_client=None) -> list[dict]:
    """Full ``eod_pnl.csv`` rows, date-ascending. Raises on any S3 error but
    NoSuchKey, which yields an empty list (the tile then renders N/A)."""
    s3 = s3_client or boto3.client("s3")
    try:
        resp = s3.get_object(Bucket=bucket, Key=EOD_PNL_KEY)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            logger.warning("eod_pnl.csv absent at s3://%s/%s", bucket, EOD_PNL_KEY)
            return []
        raise
    text = resp["Body"].read().decode("utf-8")
    csv.field_size_limit(1 << 27)  # positions_snapshot is a JSON blob per row
    rows = [r for r in csv.DictReader(io.StringIO(text)) if r.get("date")]
    rows.sort(key=lambda r: r["date"])
    return rows


def read_benchmark_sector_weights(bucket: str, s3_client=None) -> dict[str, Any]:
    """SPY's sector weights, normalised to sum to 1, in canonical labels."""
    s3 = s3_client or boto3.client("s3")
    art = _get_json(s3, bucket, SECTORS_KEY)
    if not art:
        return {"available": False, "weights": {}, "as_of": None,
                "reason": f"s3://{bucket}/{SECTORS_KEY} absent"}
    raw = art.get("spy_sector_weights") or {}
    if not raw:
        return {"available": False, "weights": {}, "as_of": art.get("as_of"),
                "reason": "artifact present but spy_sector_weights is empty"}
    weights: dict[str, float] = {}
    for label, w in raw.items():
        weights[canonical_sector(label)] = weights.get(canonical_sector(label), 0.0) + float(w)
    total = sum(weights.values())
    if total <= 0:
        return {"available": False, "weights": {}, "as_of": art.get("as_of"),
                "reason": "spy_sector_weights sum to zero"}
    return {
        "available": True,
        "weights": {k: v / total for k, v in weights.items()},
        "as_of": art.get("as_of"),
        "raw_sum": total,
        "reason": None,
    }


def read_sector_etf_returns(bucket: str, s3_client=None) -> dict[str, dict[str, float]]:
    """``{canonical sector: {date: daily total return fraction}}``.

    The close histories are ``dividend_adjusted``, so these are TOTAL returns —
    the same basis the portfolio leg is on (NAV) and the same basis I8188
    defect 3 put the SPY leg on.
    """
    s3 = s3_client or boto3.client("s3")
    out: dict[str, dict[str, float]] = {}
    for sector, etf in SECTOR_TO_ETF.items():
        art = _get_json(s3, bucket, f"{CLOSE_HISTORY_PREFIX}{etf}.json")
        if not art:
            continue
        if art.get("adjustment_basis") != "dividend_adjusted":
            # Fail loud: a price-return benchmark leg against a total-return
            # portfolio leg is exactly the defect this work exists to remove.
            raise ValueError(
                f"{etf} close history declares adjustment_basis="
                f"{art.get('adjustment_basis')!r}; the benchmark leg must be "
                "dividend_adjusted (total return) — see I8188 defect 3."
            )
        closes = art.get("closes") or []
        series: dict[str, float] = {}
        prev_px = None
        for row in closes:
            try:
                d, px = row[0], float(row[1])
            except (IndexError, TypeError, ValueError):
                continue
            if prev_px and prev_px > 0:
                series[d] = px / prev_px - 1.0
            prev_px = px
        out[sector] = series
    return out


# ─────────────────────────────────────────────────────────────────────────────
# One session
# ─────────────────────────────────────────────────────────────────────────────

def _f(row: dict, key: str) -> float | None:
    v = row.get(key)
    if v in (None, ""):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _snapshot(row: dict) -> dict[str, dict]:
    raw = row.get("positions_snapshot")
    if not raw:
        return {}
    try:
        snap = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return snap if isinstance(snap, dict) else {}


# The named non-position P&L lines. Every one is a real, separately-measured
# schema column (PR490/PR491) — none of them is a remainder.
_CASH_SLEEVE_COLUMNS = (
    "interest_usd",
    "rotation_realized_usd",
    "pricing_timing_usd",
    "unattributed_true_usd",
)


def session_portfolio_groups(prior_row: dict, row: dict) -> dict[str, Any]:
    """Beginning-of-day sector weights and the day's per-sector returns.

    Weights are BOD (prior close) — the institutional daily convention — so
    ``Σ w_s · r_s`` collapses to ``Σ P&L_s / prior_nav`` and the group weights
    cancel; the portfolio leg therefore ties to the NAV-based daily return
    exactly whenever the named lines account for the NAV move. That gap is
    measured and returned as ``input_closure_usd``; it is never plugged.
    """
    prior_nav = _f(prior_row, "portfolio_nav")
    nav_change = _f(row, "nav_change_usd")
    if not prior_nav or prior_nav <= 0 or nav_change is None:
        return {"available": False, "reason": "prior NAV or nav_change_usd absent"}

    prior_snap = _snapshot(prior_row)
    today_snap = _snapshot(row)
    if not prior_snap:
        return {"available": False, "reason": "prior positions_snapshot absent"}

    prior_mv: dict[str, float] = {}
    index_prior_mv = 0.0
    for tkr, pos in prior_snap.items():
        mv = pos.get("market_value")
        if mv in (None, ""):
            continue
        sector = canonical_sector(pos.get("sector"))
        if tkr == BENCHMARK_TICKER or sector == INDEX_SLEEVE:
            index_prior_mv += float(mv)
            continue
        prior_mv[sector] = prior_mv.get(sector, 0.0) + float(mv)

    pnl: dict[str, float] = {}
    index_pnl = 0.0
    for tkr, pos in today_snap.items():
        usd = pos.get("daily_return_usd")
        if usd in (None, ""):
            continue
        sector = canonical_sector(pos.get("sector"))
        if tkr == BENCHMARK_TICKER or sector == INDEX_SLEEVE:
            index_pnl += float(usd)
            continue
        pnl[sector] = pnl.get(sector, 0.0) + float(usd)

    named_cash = 0.0
    named_cash_lines = {}
    for col in _CASH_SLEEVE_COLUMNS:
        v = _f(row, col) or 0.0
        named_cash_lines[col] = v
        named_cash += v

    # A sector holding P&L today but nothing at yesterday's close was funded
    # out of cash intraday; it has no BOD weight to carry it. Its dollars go to
    # the cash sleeve, which is where the capital was, and are NAMED.
    entry_pnl = 0.0
    for sector in list(pnl):
        if prior_mv.get(sector, 0.0) <= 0:
            entry_pnl += pnl.pop(sector)

    weights: dict[str, float] = {s: mv / prior_nav for s, mv in prior_mv.items()}
    returns: dict[str, float] = {
        s: pnl.get(s, 0.0) / prior_mv[s] for s in prior_mv if prior_mv[s] > 0
    }

    index_weight = index_prior_mv / prior_nav
    cash_weight = 1.0 - sum(weights.values()) - index_weight
    cash_dollars = cash_weight * prior_nav
    cash_pnl = named_cash + entry_pnl
    cash_return = (cash_pnl / cash_dollars) if cash_dollars else 0.0

    accounted = sum(pnl.values()) + index_pnl + cash_pnl
    closure = nav_change - accounted

    return {
        "available": True,
        "date": row.get("date"),
        "prior_nav": prior_nav,
        "nav_change_usd": nav_change,
        "weights": weights,
        "returns": returns,
        "index_weight": index_weight,
        "index_prior_mv_usd": index_prior_mv,
        "index_pnl_usd": index_pnl,
        "cash_weight": cash_weight,
        "cash_return": cash_return,
        "cash_pnl_usd": cash_pnl,
        "named_cash_lines": named_cash_lines,
        "entry_pnl_usd": entry_pnl,
        "unclassified_weight": weights.get(UNCLASSIFIED, 0.0),
        "input_closure_usd": closure,
        "reason": None,
    }


def session_brinson(
    groups: dict[str, Any],
    benchmark_weights: dict[str, float],
    benchmark_returns_today: dict[str, float],
) -> BrinsonResult | None:
    """One session's Brinson-Fachler, portfolio sectors + Cash + Index sleeve.

    The Index sleeve (held SPY) and Cash are carried as portfolio groups with a
    benchmark weight of zero so the weights sum to 1 and the totals tie; the
    active return attributable to each is then read directly off its own row
    rather than smeared across the sectors — which is what deliverable 7 asks
    for when it says SPY holdings must be excluded from active return.
    """
    if not groups.get("available") or not benchmark_weights:
        return None
    weights_p = dict(groups["weights"])
    returns_p = dict(groups["returns"])
    weights_p["Cash"] = groups["cash_weight"]
    returns_p["Cash"] = groups["cash_return"]
    if groups["index_weight"]:
        weights_p[INDEX_SLEEVE] = groups["index_weight"]
        returns_p[INDEX_SLEEVE] = (
            groups["index_pnl_usd"] / groups["index_prior_mv_usd"]
            if groups["index_prior_mv_usd"] else 0.0
        )
    weights_b = dict(benchmark_weights)
    returns_b = {
        s: benchmark_returns_today[s]
        for s in weights_b
        if s in benchmark_returns_today
    }
    # A benchmark sector with no return today (an ETF close missing for the
    # date) is DROPPED from the benchmark and its weight renormalised, so the
    # benchmark total is not silently dragged toward zero by a data gap.
    weights_b = {s: w for s, w in weights_b.items() if s in returns_b}
    total_b = sum(weights_b.values())
    if total_b <= 0:
        return None
    weights_b = {s: w / total_b for s, w in weights_b.items()}
    return brinson_fachler(weights_p, returns_p, weights_b, returns_b)


# ─────────────────────────────────────────────────────────────────────────────
# Window: Cariño-linked BF, the SPY carve-out, beta-adjusted alpha
# ─────────────────────────────────────────────────────────────────────────────

def beta_vs_benchmark(port: list[float], bench: list[float]) -> float | None:
    """OLS beta of the daily portfolio return on the daily benchmark return."""
    n = min(len(port), len(bench))
    if n < 2:
        return None
    mp = sum(port[:n]) / n
    mb = sum(bench[:n]) / n
    cov = sum((port[i] - mp) * (bench[i] - mb) for i in range(n))
    var = sum((bench[i] - mb) ** 2 for i in range(n))
    if var <= 0:
        return None
    return cov / var


def _cum(returns: list[float]) -> float:
    out = 1.0
    for r in returns:
        out *= (1.0 + r)
    return out - 1.0


def build_attribution(
    bucket: str = "alpha-engine-research",
    *,
    s3_client=None,
    run_date: str | None = None,
) -> dict[str, Any]:
    """The full attribution artifact for the live book.

    Returns a payload with ``status`` in ``{"ok", "insufficient_data",
    "missing_input"}``. Never raises for a data gap; raises only on a genuine
    contract violation (a price-return benchmark leg, an unmapped sector).
    """
    s3 = s3_client or boto3.client("s3")
    rows = read_eod_pnl_rows(bucket, s3_client=s3)
    bench_w = read_benchmark_sector_weights(bucket, s3_client=s3)
    if not rows:
        return _empty("missing_input", f"s3://{bucket}/{EOD_PNL_KEY} absent", run_date)
    if not bench_w["available"]:
        return _empty("missing_input", bench_w["reason"], run_date)
    bench_r = read_sector_etf_returns(bucket, s3_client=s3)
    if not bench_r:
        return _empty("missing_input", "no sector-ETF close history available", run_date)

    periods: list[BrinsonResult] = []
    per_session: list[dict[str, Any]] = []
    port_daily: list[float] = []
    spy_daily: list[float] = []
    port_ex_index_daily: list[float] = []
    closure_breaches: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}

    for prior_row, row in zip(rows, rows[1:]):
        date = row["date"]
        groups = session_portfolio_groups(prior_row, row)
        if not groups.get("available"):
            skipped[groups.get("reason", "unknown")] = skipped.get(groups.get("reason", "unknown"), 0) + 1
            continue

        nav = _f(row, "portfolio_nav") or groups["prior_nav"]
        closure_bound = INPUT_CLOSURE_NAV_BPS / 10_000.0 * nav
        if abs(groups["input_closure_usd"]) > closure_bound:
            closure_breaches.append({
                "date": date,
                "input_closure_usd": groups["input_closure_usd"],
                "bound_usd": closure_bound,
                "pct_of_nav": groups["input_closure_usd"] / nav * 100.0,
            })
            skipped["input did not close"] = skipped.get("input did not close", 0) + 1
            continue

        today_bench = {s: series.get(date) for s, series in bench_r.items()}
        today_bench = {s: v for s, v in today_bench.items() if v is not None}
        bf = session_brinson(groups, bench_w["weights"], today_bench)
        if bf is None:
            skipped["no benchmark returns for the date"] = (
                skipped.get("no benchmark returns for the date", 0) + 1
            )
            continue

        p = _f(row, "daily_return_pct")
        s = _f(row, "spy_return_pct")
        if p is None or s is None:
            # A session without both published legs cannot enter the window: a
            # benchmark series over a DIFFERENT set of days than the portfolio
            # series is the two-clocks defect, and it would break the exact
            # closure asserted below.
            skipped["daily_return_pct or spy_return_pct absent"] = (
                skipped.get("daily_return_pct or spy_return_pct absent", 0) + 1
            )
            continue
        port_daily.append(p / 100.0)
        spy_daily.append(s / 100.0)
        # The active sleeve: the book with its held-SPY carved out.
        idx_w = groups["index_weight"]
        idx_r = (
            groups["index_pnl_usd"] / groups["index_prior_mv_usd"]
            if groups["index_prior_mv_usd"] else 0.0
        )
        denom = 1.0 - idx_w
        port_ex_index_daily.append(
            ((p / 100.0) - idx_w * idx_r) / denom if denom > 1e-9 else 0.0
        )

        periods.append(bf)
        per_session.append({
            "date": date,
            "allocation": bf.allocation,
            "selection": bf.selection,
            "interaction": bf.interaction,
            "portfolio_return": bf.portfolio_return,
            "benchmark_proxy_return": bf.benchmark_return,
            "active_return": bf.active_return,
            "residual": bf.active_return - bf.total_effect,
            "index_sleeve_weight": groups["index_weight"],
            "unclassified_weight": groups["unclassified_weight"],
            "input_closure_usd": groups["input_closure_usd"],
        })

    if len(periods) < MIN_SESSIONS:
        return _empty(
            "insufficient_data",
            f"{len(periods)} usable session(s), below the {MIN_SESSIONS} floor; "
            f"skipped: {skipped}",
            run_date,
        )

    linked = link_periods(periods)
    residual = linked.active_return - linked.total_effect
    _off_total = sum(
        g.total for g in linked.groups if g.group in _OFF_BENCHMARK_GROUPS
    )

    cum_port = _cum(port_daily)
    cum_spy = _cum(spy_daily)
    cum_ex = _cum(port_ex_index_daily)
    beta = beta_vs_benchmark(port_daily, spy_daily)
    beta_alpha = (
        (linked.portfolio_return - beta * cum_spy) if beta is not None else None
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "run_date": run_date,
        "sessions_used": len(periods),
        "sessions_skipped": skipped,
        "source_paths": [
            f"s3://{bucket}/{EOD_PNL_KEY}",
            f"s3://{bucket}/{SECTORS_KEY}",
            f"s3://{bucket}/{CLOSE_HISTORY_PREFIX}{{sector ETF}}.json",
        ],
        # ── Deliverable 6 ───────────────────────────────────────────────────
        "brinson_fachler": {
            "linking": "carino",
            "allocation": linked.allocation,
            "selection": linked.selection,
            "interaction": linked.interaction,
            "total_effect": linked.total_effect,
            "portfolio_return": linked.portfolio_return,
            "benchmark_proxy_return": linked.benchmark_return,
            "bf_active_return": linked.active_return,
            "residual": residual,
            "residual_pct_of_active": (
                abs(residual) / abs(linked.active_return) * 100.0
                if linked.active_return else None
            ),
            "groups": [
                {"group": g.group, "allocation": g.allocation,
                 "selection": g.selection, "interaction": g.interaction,
                 "total": g.total}
                for g in linked.groups
            ],
            # Brinson-Fachler places the WHOLE effect of a group the benchmark
            # does not hold into INTERACTION: with w_b = 0 the allocation term
            # (w_p - w_b)(r_b,i - R_b) vanishes because a group the benchmark
            # doesn't hold takes r_b,i = R_b, and selection w_b(r_p - r_b) is
            # zero at zero weight. Cash and the held-SPY Index sleeve are both
            # off-benchmark, so reading "interaction" as a cross-term here would
            # be wrong — it is the off-benchmark drag, and it is split out
            # rather than left to be misread.
            "off_benchmark": {
                "groups": _OFF_BENCHMARK_GROUPS,
                "note": (
                    "Cash and the held-SPY Index sleeve carry zero benchmark "
                    "weight, so BF assigns their entire effect to interaction. "
                    "This is the deployment-ramp drag and the benchmark sleeve, "
                    "not a sector cross-term."
                ),
                "total": _off_total,
            },
            "on_benchmark": {
                "allocation": linked.allocation,
                "selection": linked.selection,
                "interaction": linked.interaction - _off_total,
                "total": linked.total_effect - _off_total,
            },
        },
        # ── The benchmark leg, and what the proxy costs ─────────────────────
        "benchmark": {
            "basis": "total_return",
            "construction": "spdr_sector_etf_proxy",
            "weights_as_of": bench_w["as_of"],
            "weights_are_current_proxy": True,
            "weights": bench_w["weights"],
            "spy_total_return": cum_spy,
            "proxy_total_return": linked.benchmark_return,
            "proxy_tracking": linked.benchmark_return - cum_spy,
        },
        # ── Deliverable 7 ───────────────────────────────────────────────────
        "active_return": {
            # BF-consistent legs: the portfolio return is the one the
            # decomposition compounds (Cariño-linked per-session BF totals,
            # each session's total = Σ w_p,i · r_p,i over the SAME named
            # groups — sectors, Cash, Index sleeve — that feed allocation /
            # selection / interaction), so `active = bf_active +
            # proxy_tracking` holds EXACTLY.
            #
            # This is NOT the same quantity as chain-linked
            # nav_change_usd / prior_nav (corrected alpha-engine-config-I9025;
            # was miswritten as "nav_change / prior_nav, Cariño-linked" and
            # attributed below to I8188 defect 4, which is a different pair
            # entirely — daily_return_pct vs the NAV RATIO, gated by
            # executor/pnl_integrity.py::verify_twr_closes). The two diverge
            # by exactly `input_closure_usd`: the per-session dollars
            # `session_portfolio_groups` cannot attribute to a named group.
            # That residual is independently bounded at
            # INPUT_CLOSURE_NAV_BPS and a breaching session is excluded from
            # this window entirely (see `skipped["input did not close"]`), so
            # this leg and `stored_return_chain` below are computed over an
            # IDENTICAL session set by construction — the gap the field below
            # publishes is what the WITHIN-BOUND closure residual costs when
            # compounded across the window, not a day-set mismatch (that
            # mismatch is the separate, executor-side defect I9025 measured
            # and gated at the SOURCE eod_pnl.csv: verify_nav_change_basis_
            # closes, alpha-engine-config-I9025).
            "portfolio_total_return": linked.portfolio_return,
            "active_return_vs_spy": linked.portfolio_return - cum_spy,
            # The SAME window and SAME session set (see above), measured from
            # the STORED daily_return_pct chain instead of the BF-attributed
            # groups sum.
            "stored_return_chain": cum_port,
            "return_chain_basis_gap": cum_port - linked.portfolio_return,
            "portfolio_return_ex_index_sleeve": cum_ex,
            "active_return_ex_index_sleeve": cum_ex - cum_spy,
            "index_sleeve_mean_weight": (
                sum(s["index_sleeve_weight"] for s in per_session) / len(per_session)
            ),
            "beta_vs_spy": beta,
            "beta_adjusted_alpha": beta_alpha,
            "factor_attribution": None,
            "factor_attribution_reason": (
                "NOT IMPLEMENTED — a size/value/momentum/quality/low-vol "
                "decomposition needs per-name factor loadings joined to the "
                "book on every session; the loadings live in the feature store, "
                "not in eod_pnl.csv, and no producer joins them today. Tracked "
                "separately; this field is an admitted gap, not a zero."
            ),
        },
        "input_closure": {
            "bound_nav_bps": INPUT_CLOSURE_NAV_BPS,
            "breaches": closure_breaches,
            "n_breaches": len(closure_breaches),
        },
        "per_session": per_session,
    }


def _empty(status: str, reason: str, run_date: str | None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "reason": reason,
        "run_date": run_date,
        "sessions_used": 0,
        "brinson_fachler": None,
        "benchmark": None,
        "active_return": None,
    }


def write_attribution(
    payload: dict[str, Any],
    *,
    bucket: str = "alpha-engine-research",
    run_date: str | None = None,
    s3_client=None,
) -> list[str]:
    """Write the standing pointer, plus a dated copy when ``run_date`` is given."""
    s3 = s3_client or boto3.client("s3")
    body = json.dumps(payload, indent=2, default=str).encode("utf-8")
    keys = [ATTRIBUTION_LATEST_KEY]
    if run_date:
        keys.append(f"evaluator/{run_date}/attribution.json")
    for key in keys:
        s3.put_object(Bucket=bucket, Key=key, Body=body,
                      ContentType="application/json")
        logger.info("attribution written to s3://%s/%s", bucket, key)
    return keys
