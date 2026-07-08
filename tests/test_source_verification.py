"""Source-verification harness — the grader graded (config#1958 trust battery).

Every other tile test seeds an artifact and asserts against HARDCODED expected
constants — which verifies the grader agrees with the test author, not with
the source artifact. This harness closes that gap: it re-derives graded
MetricRecord values from the same synthetic source artifact via INDEPENDENT
implementations (plain numpy/math written here, no ``grading.tiles`` helpers,
no ``nousergon_lib.quant`` calls) and asserts agreement. A transcription or
aggregation drift between the tile builder and its source now fails loudly.

v1 scope: the Portfolio Outcome tile's deterministically re-derivable metrics
(``alpha_vs_spy``, ``max_drawdown``, ``sharpe_ratio``, ``sortino_ratio``'s
sign/regime is CI-fuzzy so excluded; PSR/DSR/CIs are estimator-defined, not
independently re-derivable, and are out of scope by design — their trust legs
are the lib quant battery + null calibration on the backtester side).
"""

import math

import boto3
import numpy as np
import pytest
from moto import mock_aws

from grading.tiles.portfolio_outcome import EOD_PNL_KEY, build_portfolio_outcome_tile

BUCKET = "alpha-engine-research"
_HEADER = "date,portfolio_nav,daily_return_pct,spy_return_pct,daily_alpha_pct,positions_snapshot,created_at"


def _synth_book(n: int = 90, seed: int = 42):
    """Deterministic pseudo-random book: returns (csv_text, port, spy, nav).

    Percent columns in the CSV (matching the live ledger); fraction series
    returned for the independent recomputation.
    """
    rng = np.random.default_rng(seed)
    # Quantize to the CSV's written precision FIRST — the truth series must be
    # exactly what the artifact contains (re-derive from the artifact, not
    # from pre-rounding memory), so agreement bounds are machine-epsilon.
    port = np.round(rng.normal(0.0006, 0.011, n) * 100, 6) / 100
    spy = np.round(rng.normal(0.0004, 0.009, n) * 100, 6) / 100
    nav_series, nav = [], 1_000_000.0
    rows = [_HEADER]
    for i in range(n):
        nav *= 1 + port[i]
        nav = round(nav, 4)
        nav_series.append(nav)
        d = f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}"
        rows.append(
            f"{d},{nav:.4f},{port[i] * 100:.6f},{spy[i] * 100:.6f},"
            f"{(port[i] - spy[i]) * 100:.6f},{{}},2026-01-01T00:00:00+00:00"
        )
    return "\n".join(rows) + "\n", port, spy, np.asarray(nav_series)


@pytest.fixture
def tile_and_truth():
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        csv_text, port, spy, nav = _synth_book()
        s3.put_object(Bucket=BUCKET, Key=EOD_PNL_KEY, Body=csv_text.encode())
        tile = build_portfolio_outcome_tile(BUCKET, s3_client=s3)
        yield tile, port, spy, nav


def _component(tile: dict, name: str) -> dict:
    matches = [c for c in tile["components"] if c["name"] == name]
    assert matches, f"{name} missing from portfolio_outcome components"
    return matches[0]


class TestSourceVerification:
    """Independent re-derivation vs the tile builder, same source artifact."""

    def test_alpha_vs_spy_matches_independent_log_sum(self, tile_and_truth):
        tile, port, spy, _ = tile_and_truth
        # Independent: cumulative log-alpha, plain math on the source series.
        expected = float(np.sum(np.log1p(port) - np.log1p(spy)))
        got = _component(tile, "alpha_vs_spy")["value"]
        assert got == pytest.approx(expected, abs=1e-9)

    def test_max_drawdown_matches_independent_running_peak(self, tile_and_truth):
        tile, _, _, nav = tile_and_truth
        # Independent: most negative value/running-peak - 1.
        expected = float(np.min(nav / np.maximum.accumulate(nav) - 1.0))
        got = _component(tile, "max_drawdown")["value"]
        assert got == pytest.approx(expected, abs=1e-9)

    def test_sharpe_matches_independent_annualized(self, tile_and_truth):
        tile, port, _, _ = tile_and_truth
        # Independent: mean/sample-std (ddof=1) × √252, zero risk-free.
        expected = float(np.mean(port) / np.std(port, ddof=1) * math.sqrt(252))
        got = _component(tile, "sharpe_ratio")["value"]
        assert got == pytest.approx(expected, rel=1e-9)

    def test_percent_to_fraction_round_trip_is_exact(self, tile_and_truth):
        # The %→fraction conversion is where the daily_alpha_pct-vs-alpha_pct
        # class of bug lives (2026-07-08: the producer's manifest key differs
        # from its own CSV column). n_samples must equal the row count — a
        # silent row drop would shrink N and quietly weaken every CI.
        tile, port, _, _ = tile_and_truth
        assert _component(tile, "sharpe_ratio")["n_samples"] == len(port)

    def test_source_path_names_the_actual_artifact(self, tile_and_truth):
        # Every value-bearing component must be traceable to the ledger it
        # was computed from — the re-derivability contract the /dash trust
        # page cites.
        tile, _, _, _ = tile_and_truth
        for comp in tile["components"]:
            if comp.get("value") is not None:
                assert EOD_PNL_KEY in (comp.get("source_path") or ""), comp["name"]
