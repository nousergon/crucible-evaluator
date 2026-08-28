"""Canonical sector labels (alpha-engine-config-I8188 deliverables 6-7)."""

from __future__ import annotations

import pytest

from grading.sectors import (
    CANONICAL_SECTORS,
    INDEX_SLEEVE,
    SECTOR_TO_ETF,
    UNCLASSIFIED,
    UnknownSectorError,
    canonical_sector,
)


class TestTheLiveLabelsCollapse:
    """THE DEFECT: `positions_snapshot` carries both spellings of three
    sectors across the 119 live sessions. Two labels for one sector split it
    into two Brinson groups, one with zero benchmark weight, and the whole of
    its return is then reported as an allocation bet the book never made."""

    @pytest.mark.parametrize("a,b", [
        ("Financial", "Financials"),
        ("Health Care", "Healthcare"),
        ("Information Technology", "Technology"),
        ("Consumer Discretionary", "Consumer Cyclical"),
        ("Consumer Staples", "Consumer Defensive"),
    ])
    def test_both_spellings_map_to_one_group(self, a, b):
        assert canonical_sector(a) == canonical_sector(b)
        assert canonical_sector(a) in CANONICAL_SECTORS

    def test_every_label_observed_live_is_mapped(self):
        observed = [
            "", "Broad Market / Index", "Communication Services",
            "Consumer Discretionary", "Consumer Staples", "Energy",
            "Financial", "Financials", "Health Care", "Healthcare",
            "Industrials", "Information Technology", "Real Estate",
            "Technology", "Unknown", "Utilities",
        ]
        for label in observed:
            assert canonical_sector(label) in (
                set(CANONICAL_SECTORS) | {INDEX_SLEEVE, UNCLASSIFIED}
            )

    def test_the_index_holding_is_carved_out_not_classified(self):
        assert canonical_sector("Broad Market / Index") == INDEX_SLEEVE

    def test_missing_and_unknown_go_to_their_own_visible_group(self):
        assert canonical_sector("") == UNCLASSIFIED
        assert canonical_sector(None) == UNCLASSIFIED
        assert canonical_sector("Unknown") == UNCLASSIFIED


def test_an_unmapped_label_raises_rather_than_being_bucketed():
    with pytest.raises(UnknownSectorError) as e:
        canonical_sector("Semiconductors & Semiconductor Equipment")
    assert "allocation bet the book never made" in str(e.value)


def test_every_canonical_sector_has_an_etf():
    assert set(SECTOR_TO_ETF) == set(CANONICAL_SECTORS)
    assert len(CANONICAL_SECTORS) == 11
