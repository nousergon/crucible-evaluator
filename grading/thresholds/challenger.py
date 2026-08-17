"""challenger.py — the ``history_bands_v1`` arm: bands derived from what happened.

The champion's bands were chosen by hand. This arm asks the data instead: over
the cohort, what did this metric READ in the cycles that preceded a positive
realized objective, and what did it read before a negative one? The median of
each group becomes the band.

  target    = median of the readings that preceded objective > 0
  red_line  = median of the readings that preceded objective <= 0

Deliberately simple and deliberately reproducible: a quantile of an observed
conditional distribution, with no fitted parameters to overfit and nothing a
reader has to take on trust. It is a CHALLENGER — its job is to be scored beside
the champion on predictive validity (``scoring.py``), not to be believed.

Three abstention states, all explicit, none of which may render as a band:

  ``insufficient``  fewer paired cards than the slot's floor, or fewer readings
                    in either outcome group than ``n_floor_per_status``.
  ``degenerate``    the two medians do not order in the metric's declared
                    direction — the data says this metric does not separate the
                    outcome, and inventing a band anyway would be a fabrication.
  ``ungraded``      the champion imposes no bar on this metric, so there is no
                    decision to challenge.

An abstention removes the cell from EVERY arm's cohort, never just this one
(champion-challenger §4, count-matching).
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass

from grading.thresholds.cohort import Cohort
from grading.thresholds.registry import ThresholdRegistry, load_registry

logger = logging.getLogger(__name__)

ARM_ID = "history_bands_v1"


@dataclass(frozen=True)
class ProposedBand:
    module: str
    name: str
    status: str  # "proposed" | "insufficient" | "degenerate" | "ungraded"
    target: float | None
    red_line: float | None
    higher_is_better: bool
    n_pairs: int
    n_positive: int
    n_negative: int
    reason: str

    @property
    def usable(self) -> bool:
        return self.status == "proposed"

    def to_dict(self) -> dict:
        return {
            "module": self.module,
            "metric": self.name,
            "status": self.status,
            "target": self.target,
            "red_line": self.red_line,
            "higher_is_better": self.higher_is_better,
            "n_pairs": self.n_pairs,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "reason": self.reason,
        }


def _direction(registry: ThresholdRegistry, module: str, name: str) -> bool:
    """The metric's declared direction, falling back to the champion ordering.

    ``derive_status`` infers direction from ``target >= red_line``; the
    challenger must orient itself the same way or it would propose a band that
    grades backwards.
    """
    row = registry.rows.get((module, name), {})
    declared = row.get("higher_is_better")
    if declared is not None:
        return bool(declared)
    target, red_line = row.get("target"), row.get("red_line")
    return target is None or red_line is None or target >= red_line


def propose_bands(
    cohort: Cohort,
    registry: ThresholdRegistry | None = None,
) -> dict[tuple[str, str], ProposedBand]:
    """Propose one band per graded metric. Shadow output — never consumed for status."""
    reg = registry or load_registry()
    floor_cards = reg.slot.n_floor_cards
    floor_status = int(reg.slot.scoring["n_floor_per_status"])
    paired = cohort.paired_indices()

    out: dict[tuple[str, str], ProposedBand] = {}
    for module, name in reg.graded_keys():
        hib = _direction(reg, module, name)
        positives: list[float] = []
        negatives: list[float] = []
        for i in paired:
            cell = cohort.rows[i].cells.get((module, name))
            if cell is None:
                continue
            objective = cohort.objective(i)
            (positives if objective > 0 else negatives).append(cell.value)

        n_pairs = len(positives) + len(negatives)
        if len(paired) < floor_cards or min(len(positives), len(negatives)) < floor_status:
            out[(module, name)] = ProposedBand(
                module=module, name=name, status="insufficient", target=None, red_line=None,
                higher_is_better=hib, n_pairs=n_pairs, n_positive=len(positives),
                n_negative=len(negatives),
                reason=(
                    f"insufficient: {len(paired)} paired card(s) vs floor {floor_cards}; "
                    f"{len(positives)} reading(s) before a positive objective and "
                    f"{len(negatives)} before a non-positive one vs floor {floor_status} each"
                ),
            )
            continue

        target = statistics.median(positives)
        red_line = statistics.median(negatives)
        ordered = target > red_line if hib else target < red_line
        if not ordered:
            out[(module, name)] = ProposedBand(
                module=module, name=name, status="degenerate", target=None, red_line=None,
                higher_is_better=hib, n_pairs=n_pairs, n_positive=len(positives),
                n_negative=len(negatives),
                reason=(
                    f"degenerate: median before a positive objective ({target:.6g}) does not "
                    f"sit on the {'high' if hib else 'low'} side of the median before a "
                    f"non-positive one ({red_line:.6g}) — this metric did not separate the "
                    f"outcome over the cohort, so no band is proposed"
                ),
            )
            continue

        out[(module, name)] = ProposedBand(
            module=module, name=name, status="proposed", target=target, red_line=red_line,
            higher_is_better=hib, n_pairs=n_pairs, n_positive=len(positives),
            n_negative=len(negatives),
            reason=(
                f"median reading before a positive objective {target:.6g} (N={len(positives)}) "
                f"vs before a non-positive one {red_line:.6g} (N={len(negatives)}) over "
                f"{len(paired)} paired card(s)"
            ),
        )

    n_proposed = sum(1 for b in out.values() if b.usable)
    logger.info(
        "%s proposed %d band(s) of %d graded metric(s); %d abstained.",
        ARM_ID, n_proposed, len(out), len(out) - n_proposed,
    )
    return out
