"""scoring.py — both arms graded on predictive validity, every cycle.

champion-challenger §3 is the load-bearing rule: **every arm is scored every
cycle regardless of promotion.** So this runs whether or not anything could be
promoted, and the champion is scored on exactly the axis the challenger is.

The question each arm answers is: *given the status you assigned this metric on
card t, how likely was the portfolio to realize positive alpha over the next
horizon?* The score is the Brier score of that probability against what actually
happened.

  * The probability for a status is that status's empirical hit rate over the
    cohort **with the observation itself held out**. Without the hold-out an arm
    would be scored against a rate it defined, which is the closed loop §8
    forbids.
  * Both arms are scored over the SAME cells (§4). A cell either arm abstains on
    leaves every arm's cohort.
  * The judge is never scored on the card it grades: the label is realized
    alpha, never agreement with the champion.

**`insufficient` is a result, not a pass and not an absence** (§5.1, §7.2). With
17 cards in existence on 2026-08-16 and a floor of 26 paired cards, `insufficient`
is the CORRECT output of this module today, and it is written to the artifact and
rendered on the card with its counts rather than omitted.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import boto3
from krepis.metrics import derive_status

from grading.thresholds.challenger import ARM_ID as CHALLENGER_ARM
from grading.thresholds.challenger import ProposedBand, propose_bands
from grading.thresholds.cohort import Cohort, load_cohort
from grading.thresholds.registry import ThresholdRegistry, load_registry

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "threshold_leaderboard.v1"
LEADERBOARD_PREFIX = "evaluator"


def leaderboard_key(run_date: str) -> str:
    return f"{LEADERBOARD_PREFIX}/{run_date}/threshold_leaderboard.json"


@dataclass(frozen=True)
class ArmScore:
    arm: str
    role: str
    status: str  # "scored" | "insufficient" | "vacuous" | "unmeasurable"
    brier: float | None
    n_cards_paired: int
    n_observations: int
    by_status: dict[str, dict[str, float | int]]
    reason: str

    def to_dict(self) -> dict:
        return {
            "arm": self.arm,
            "role": self.role,
            "status": self.status,
            "brier": self.brier,
            "n_cards_paired": self.n_cards_paired,
            "n_observations": self.n_observations,
            "by_status": self.by_status,
            "reason": self.reason,
        }


def _bands_for_arm(
    arm: str,
    registry: ThresholdRegistry,
    proposals: dict[tuple[str, str], ProposedBand],
    key: tuple[str, str],
) -> tuple[float | None, float | None] | None:
    """This arm's (target, red_line) for one metric, or None when it abstains."""
    if arm == registry.slot.champion:
        row = registry.rows.get(key, {})
        return (row.get("target"), row.get("red_line"))
    proposal = proposals.get(key)
    if proposal is None or not proposal.usable:
        return None
    return (proposal.target, proposal.red_line)


def _score_one_arm(
    arm: str,
    role: str,
    observations: list[tuple[str, int]],
    n_cards_paired: int,
    registry: ThresholdRegistry,
) -> ArmScore:
    floor_cards = registry.slot.n_floor_cards
    floor_status = int(registry.slot.scoring["n_floor_per_status"])

    buckets: dict[str, list[int]] = {}
    for status, label in observations:
        buckets.setdefault(status, []).append(label)
    by_status = {
        s: {"n": len(ys), "hit_rate": (sum(ys) / len(ys)) if ys else None}
        for s, ys in sorted(buckets.items())
    }

    shortfalls = []
    if n_cards_paired < floor_cards:
        shortfalls.append(f"{n_cards_paired} paired card(s) vs floor {floor_cards}")
    thin = [f"{s} N={len(ys)}" for s, ys in buckets.items() if len(ys) < floor_status]
    if thin:
        shortfalls.append(f"status buckets below floor {floor_status}: {', '.join(sorted(thin))}")
    if not observations:
        shortfalls.append("no scored observations")
    if shortfalls:
        return ArmScore(
            arm=arm, role=role, status="insufficient", brier=None,
            n_cards_paired=n_cards_paired, n_observations=len(observations),
            by_status=by_status,
            reason="insufficient: " + "; ".join(shortfalls),
        )

    squared_errors: list[float] = []
    for status, label in observations:
        ys = buckets[status]
        # Leave-one-out empirical rate: the arm may not be graded against a
        # probability this very observation set.
        p_hat = (sum(ys) - label) / (len(ys) - 1)
        squared_errors.append((p_hat - label) ** 2)
    brier = sum(squared_errors) / len(squared_errors)
    return ArmScore(
        arm=arm, role=role, status="scored", brier=brier,
        n_cards_paired=n_cards_paired, n_observations=len(observations),
        by_status=by_status,
        reason=(
            f"Brier {brier:.4f} over {len(observations)} observation(s) across "
            f"{n_cards_paired} paired card(s); leave-one-out empirical rate per status"
        ),
    )


def score_slot(
    cohort: Cohort,
    registry: ThresholdRegistry | None = None,
) -> dict:
    """Score every arm over the shared cohort and return the leaderboard document."""
    reg = registry or load_registry()
    proposals = propose_bands(cohort, reg)
    paired = cohort.paired_indices()
    statuses_scored = set(reg.slot.scoring["statuses_scored"])

    arms = [reg.slot.champion] + [a for a in reg.slot.arms if a != reg.slot.champion]
    observations: dict[str, list[tuple[str, int]]] = {a: [] for a in arms}
    n_cells = 0
    identical_cells = 0

    for i in paired:
        objective = cohort.objective(i)
        label = 1 if objective > 0 else 0
        for key in reg.graded_keys():
            cell = cohort.rows[i].cells.get(key)
            if cell is None:
                continue
            bands = {a: _bands_for_arm(a, reg, proposals, key) for a in arms}
            # §4 count-matching: a cell any arm abstains on leaves EVERY arm's
            # cohort, so the comparison stays like-for-like.
            if any(b is None for b in bands.values()):
                continue
            per_arm_status = {}
            for arm in arms:
                target, red_line = bands[arm]
                per_arm_status[arm] = derive_status(
                    value=cell.value, n_samples=cell.n_samples, n_floor=cell.n_floor,
                    target=target, red_line=red_line,
                )
            if not any(s in statuses_scored for s in per_arm_status.values()):
                continue
            n_cells += 1
            if len({bands[a] for a in arms}) == 1:
                identical_cells += 1
            for arm in arms:
                if per_arm_status[arm] in statuses_scored:
                    observations[arm].append((per_arm_status[arm], label))

    scores: list[ArmScore] = []
    for arm in arms:
        role = "champion" if arm == reg.slot.champion else "challenger"
        score = _score_one_arm(arm, role, observations[arm], len(paired), reg)
        # §4 vacuity guard — a rule compared with itself is not a comparison.
        if role == "challenger" and n_cells > 0 and identical_cells == n_cells:
            score = ArmScore(
                arm=arm, role=role, status="vacuous", brier=None,
                n_cards_paired=len(paired), n_observations=score.n_observations,
                by_status=score.by_status,
                reason=(
                    f"vacuous: this arm's bands resolved identically to the champion's on "
                    f"all {n_cells} scored cell(s) — nothing was compared"
                ),
            )
        scores.append(score)

    doc = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(),
        "slot": reg.slot.id,
        "champion": reg.slot.champion,
        "objective": dict(reg.slot.objective),
        "scoring": dict(reg.slot.scoring),
        "hysteresis": dict(reg.slot.hysteresis),
        "cohort": {
            "n_cards_loaded": cohort.n_cards_loaded,
            "n_cards_paired": len(paired),
            "horizon_cycles": cohort.horizon_cycles,
            "dates": cohort.dates,
            "n_scored_cells": n_cells,
            "warnings": cohort.warnings,
        },
        "arms": [s.to_dict() for s in scores],
        "challenger_proposed_bands": [
            proposals[k].to_dict() for k in sorted(proposals)
        ],
        "promotion": _promotion_verdict(scores, reg),
    }
    logger.info(
        "Threshold leaderboard %s: %s",
        reg.slot.id,
        "; ".join(f"{s.arm}={s.status}" + (f" brier={s.brier:.4f}" if s.brier is not None else "")
                  for s in scores),
    )
    return doc


def _promotion_verdict(scores: list[ArmScore], registry: ThresholdRegistry) -> dict:
    """Whether any challenger has earned the champion's seat this cycle.

    Advisory only — promotion is an edit to ``registry.yaml`` made by
    ``grading/thresholds/promote.py``, never an automatic swap from a scoring
    run. §5.2: lead by a margin, not merely lead.
    """
    margin = float(registry.slot.hysteresis["promotion_margin_brier"])
    champion = next((s for s in scores if s.role == "champion"), None)
    if champion is None or champion.status != "scored":
        return {
            "eligible": [],
            "reason": (
                "no promotion is assessable: the champion is "
                f"{champion.status if champion else 'absent'} "
                f"({champion.reason if champion else 'no champion arm'}). A challenger is "
                "never promoted against an unscored incumbent."
            ),
        }
    eligible = [
        {"arm": s.arm, "brier": s.brier, "lead": champion.brier - s.brier}
        for s in scores
        if s.role == "challenger" and s.status == "scored"
        and s.brier is not None and champion.brier - s.brier >= margin
    ]
    return {
        "eligible": eligible,
        "margin": margin,
        "reason": (
            f"{len(eligible)} challenger(s) lead the champion's Brier "
            f"{champion.brier:.4f} by at least {margin}"
        ),
    }


def build_leaderboard(
    bucket: str,
    run_date: str,
    s3_client=None,
    registry: ThresholdRegistry | None = None,
) -> dict:
    """Load the cohort and score every arm. Raises on a real S3 failure."""
    reg = registry or load_registry()
    cohort = load_cohort(bucket, run_date, s3_client=s3_client, registry=reg)
    return score_slot(cohort, reg)


def write_leaderboard(bucket: str, run_date: str, doc: dict, s3_client=None) -> str:
    """Persist the leaderboard beside the card. Returns the key written."""
    s3 = s3_client or boto3.client("s3")
    key = leaderboard_key(run_date)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(doc, indent=2, sort_keys=False).encode("utf-8"),
        ContentType="application/json",
    )
    logger.info("Wrote threshold leaderboard to s3://%s/%s", bucket, key)
    return key


ARM_METRIC_PREFIX = "threshold_arm_brier_"


def arm_metric_name(arm: str) -> str:
    """The MetricRecord name for one arm's score.

    Keyed by ARM, not by role: a promoted arm keeps its history (§3), so the
    series must not rename itself the moment the champion changes.
    """
    return f"{ARM_METRIC_PREFIX}{arm}"


def build_arm_components(
    leaderboard: dict | None,
    *,
    module: str,
    source_path: str,
    registry: ThresholdRegistry | None = None,
    error: str | None = None,
) -> list:
    """One ``diagnostic`` MetricRecord per arm, in MACHINE-HEALTH vocabulary.

    champion-challenger §8: machine health and experiment performance never
    share a grade vocabulary. So the STATUS here answers "was this arm scored
    this cycle?" — never "did this arm win?". Which arm is ahead lives in
    ``threshold_leaderboard.json`` and in the reason text, where it cannot be
    mistaken for a component grade on the card.

    The records carry no target/red_line by design (their registry rows declare
    none): a Brier bar would be a new hand-set literal, which is the thing this
    whole slot exists to remove.

    Every arm gets a record every cycle. An arm that produced no score is a
    recorded MISS (``N/A-LOW-N`` with its counts, or ``N/A-NOT-RUN`` when the
    scoring itself failed), never an omission — silent absence and a genuine
    zero must never render identically (§3).
    """
    from grading.metric_record import build_metric  # local: avoids an import cycle
    from grading.units import BRIER_SCORE

    reg = registry or load_registry()
    by_arm = {a["arm"]: a for a in (leaderboard or {}).get("arms", [])}
    cohort = (leaderboard or {}).get("cohort", {})
    floor_cards = reg.slot.n_floor_cards

    components = []
    for arm in reg.slot.arms:
        name = arm_metric_name(arm)
        role = "champion" if arm == reg.slot.champion else "challenger"
        score = by_arm.get(arm)
        if score is None:
            components.append(build_metric(
                name=name, module=module, metric_type="ratio", criticality="diagnostic",
                estimator="brier_leave_one_out_status_rate",
                measurement_horizon=f"{reg.slot.horizon_cycles}_cycles",
                n_floor=floor_cards, source_path=source_path, ran=False,
                na_detail=(
                    error or
                    f"{name}: the threshold slot produced no leaderboard this cycle — "
                    f"arm '{arm}' was not scored (champion-challenger §3: an unscored "
                    f"cycle is unrecoverable)."
                ),
            ))
            continue

        n_paired = int(score.get("n_cards_paired") or 0)
        brier = score.get("brier")
        detail = (
            f"{name}: arm '{arm}' ({role}) — {score.get('reason')}. Objective: "
            f"{reg.slot.objective['name']} over {reg.slot.horizon_cycles} cycle(s) vs "
            f"{reg.slot.objective['benchmark']}. Cohort {cohort.get('n_cards_loaded')} card(s) "
            f"loaded, {n_paired} paired, floor {floor_cards}."
        )
        if brier is None:
            # `insufficient` / `vacuous` are RESULTS. They render, with counts,
            # and they never render as a pass (§5.1, §7.2).
            components.append(build_metric(
                name=name, module=module, metric_type="ratio", criticality="diagnostic",
                estimator="brier_leave_one_out_status_rate",
                measurement_horizon=f"{reg.slot.horizon_cycles}_cycles",
                n_samples=n_paired, n_floor=floor_cards, source_path=source_path,
                na_detail=detail,
            ))
            continue
        components.append(build_metric(
            name=name, module=module, metric_type="ratio", criticality="diagnostic",
            estimator="brier_leave_one_out_status_rate",
            measurement_horizon=f"{reg.slot.horizon_cycles}_cycles",
            value=brier, unit=BRIER_SCORE, n_samples=n_paired, n_floor=floor_cards, source_path=source_path,
            higher_is_better=False, reason=detail,
        ))
    return components
