"""cohort.py — the paired (card, realized-objective) cohort both arms are scored on.

An arm in this slot decides, for one metric on one card, whether the reading is
GREEN, WATCH or RED. The only honest way to grade that decision is against what
happened NEXT: the alpha the portfolio actually realized over the following
cycles. This module builds that pairing.

Two binding rules, both inherited rather than invented:

  * **Prior CARDS are the SSOT for graded values** (``grading/history.py``).
    Nothing here re-derives a past week's value from raw upstream artifacts —
    the graded card is the producer-owned fact, and a re-derivation path is the
    rebuild-writer bug class.
  * **The objective is realized, not predicted** (champion-challenger §8). The
    yardstick is ``portfolio_outcome.alpha_vs_spy``, which is CUMULATIVE
    log-alpha vs SPY since inception, so the alpha realized between two cards is
    the DIFFERENCE of their values. A card at the end of the horizon is required;
    an unpaired card contributes no observation and is counted as such.

An absent or N/A reading contributes nothing and is never zero-filled — the same
rule the trend history states, for the same reason.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import boto3

from grading.history import list_card_keys, read_cards
from grading.thresholds.registry import ThresholdRegistry, load_registry

logger = logging.getLogger(__name__)

#: The tile + component carrying the objective. Named once; the registry's
#: ``slot.objective.source`` documents the same pair for a human reader.
OBJECTIVE_TILE = "portfolio_outcome"
OBJECTIVE_METRIC = "alpha_vs_spy"

_NA_PREFIX = "N/A"


@dataclass(frozen=True)
class Cell:
    """One component's reading on one card — what an arm has to judge."""

    value: float
    n_samples: int | None
    n_floor: int


@dataclass(frozen=True)
class CardRow:
    date: str
    cells: dict[tuple[str, str], Cell]
    objective_level: float | None


@dataclass(frozen=True)
class Cohort:
    """Cards oldest → newest, plus the horizon they are paired over."""

    rows: list[CardRow]
    horizon_cycles: int
    n_cards_loaded: int
    warnings: list[str] = field(default_factory=list)

    @property
    def dates(self) -> list[str]:
        return [r.date for r in self.rows]

    def objective(self, i: int) -> float | None:
        """Realized log-alpha between card ``i`` and card ``i + horizon``.

        ``None`` when either end is missing — an unpaired card, not a zero.
        """
        j = i + self.horizon_cycles
        if j >= len(self.rows):
            return None
        start, end = self.rows[i].objective_level, self.rows[j].objective_level
        if start is None or end is None:
            return None
        return end - start

    def paired_indices(self) -> list[int]:
        return [i for i in range(len(self.rows)) if self.objective(i) is not None]

    @property
    def n_paired(self) -> int:
        return len(self.paired_indices())


def _extract_cells(card: dict) -> tuple[dict[tuple[str, str], Cell], float | None]:
    cells: dict[tuple[str, str], Cell] = {}
    objective_level: float | None = None
    tiles = card.get("tiles")
    if not isinstance(tiles, dict):
        return cells, None
    for tile_name, tile in tiles.items():
        for comp in (tile or {}).get("components") or []:
            if not isinstance(comp, dict):
                continue
            name, value = comp.get("name"), comp.get("value")
            status = str(comp.get("status", ""))
            if not name or value is None or status.startswith(_NA_PREFIX):
                continue
            try:
                fval = float(value)
            except (TypeError, ValueError):
                logger.warning("Non-numeric cohort value for (%s, %s): %r — skipped",
                               tile_name, name, value)
                continue
            n_floor = comp.get("n_floor")
            cells[(tile_name, name)] = Cell(
                value=fval,
                n_samples=comp.get("n_samples"),
                n_floor=int(n_floor) if n_floor is not None else 1,
            )
            if tile_name == OBJECTIVE_TILE and name == OBJECTIVE_METRIC:
                objective_level = fval
    return cells, objective_level


def load_cohort(
    bucket: str,
    run_date: str,
    s3_client=None,
    *,
    registry: ThresholdRegistry | None = None,
) -> Cohort:
    """Load the scoring cohort from prior report cards.

    The card count is bounded by the registry's ``scoring.cohort_max_cards``,
    which the registry loader has already checked against the objective horizon
    and the retention of ``evaluator/`` (champion-challenger §7.1).
    """
    reg = registry or load_registry()
    s3 = s3_client or boto3.client("s3")

    dated_keys = list_card_keys(s3, bucket, run_date, reg.slot.cohort_max_cards)
    rows: list[CardRow] = []
    warnings: list[str] = []
    for date_s, card in read_cards(s3, bucket, dated_keys):
        cells, objective_level = _extract_cells(card)
        if objective_level is None:
            warnings.append(
                f"{date_s}: card carries no value-bearing "
                f"{OBJECTIVE_TILE}.{OBJECTIVE_METRIC} — cannot anchor the objective"
            )
        rows.append(CardRow(date=date_s, cells=cells, objective_level=objective_level))

    cohort = Cohort(
        rows=rows,
        horizon_cycles=reg.slot.horizon_cycles,
        n_cards_loaded=len(rows),
        warnings=warnings,
    )
    logger.info(
        "Threshold cohort for %s: %d card(s) loaded, %d paired at horizon %d cycle(s) "
        "(floor %d cards).",
        run_date, cohort.n_cards_loaded, cohort.n_paired, cohort.horizon_cycles,
        reg.slot.n_floor_cards,
    )
    return cohort
