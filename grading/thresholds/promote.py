"""promote.py — swap the champion of the threshold slot, reversibly.

champion-challenger §1: **measurement is unconditional; promotion is cheap and
reversible; retirement is slow.** A promotion here is a two-line edit to
``registry.yaml`` plus a dated entry in ``EXPERIMENTS.md``. Reverting is the same
command with the old arm named. Nothing about the measurement changes: every arm
is still scored every cycle (§3), and the promoted arm keeps its history.

    python -m grading.thresholds.promote --to history_bands_v1 \\
        --leaderboard s3://alpha-engine-research/evaluator/2026-11-07/threshold_leaderboard.json \\
        --experiments ../alpha-engine-config/private-docs/EXPERIMENTS.md

What it refuses to do, and why:

  * **Promote without evidence.** The leaderboard for the cycle must show the
    arm ``scored`` and leading the champion by the registry's hysteresis margin
    (§5.2). ``--force`` overrides with a mandatory ``--rationale`` that is
    written into the EXPERIMENTS.md entry — an override is a decision on the
    record, never a quiet one.
  * **Promote an unregistered arm.** An arm that is not in ``slot.arms`` has no
    scoring path, which is the ``thinktank_coverage`` defect (§10).
  * **Write anything if either write would fail.** The registry edit and the
    EXPERIMENTS.md entry land together or not at all, so the file can never
    claim a champion no record explains.

This is an operator CLI. It is deliberately NOT wired into any automated path:
promotion is cheap, but it is still a decision, and the loop's job is to make it
obvious rather than to make it silently.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from grading.thresholds.registry import REGISTRY_PATH, load_registry


class PromotionRefused(RuntimeError):
    """The promotion was not performed, and the reason is on the message."""


def _read_leaderboard(location: str) -> dict:
    if location.startswith("s3://"):
        import boto3

        parsed = urlparse(location)
        body = boto3.client("s3").get_object(
            Bucket=parsed.netloc, Key=parsed.path.lstrip("/")
        )["Body"].read()
        return json.loads(body)
    return json.loads(Path(location).read_text())


def check_evidence(leaderboard: dict, to_arm: str, margin: float) -> str:
    """Return the evidence line for a legitimate promotion, or raise."""
    arms = {a["arm"]: a for a in leaderboard.get("arms", [])}
    champion_id = leaderboard.get("champion")
    champion, challenger = arms.get(champion_id), arms.get(to_arm)
    if challenger is None:
        raise PromotionRefused(
            f"arm {to_arm!r} does not appear in the leaderboard — an arm with no scoring "
            f"path is a rumour, not a challenger (champion-challenger §3)"
        )
    if champion is None or champion.get("status") != "scored":
        raise PromotionRefused(
            f"the incumbent {champion_id!r} is "
            f"{champion.get('status') if champion else 'absent'} — a challenger is never "
            f"promoted against an unscored incumbent"
        )
    if challenger.get("status") != "scored":
        raise PromotionRefused(
            f"arm {to_arm!r} is {challenger.get('status')!r}: {challenger.get('reason')}. "
            f"`insufficient` is a result, not a near-pass"
        )
    lead = champion["brier"] - challenger["brier"]
    if lead < margin:
        raise PromotionRefused(
            f"arm {to_arm!r} leads the incumbent's Brier {champion['brier']:.4f} by only "
            f"{lead:.4f}, under the {margin} hysteresis margin (§5.2: lead by a margin, "
            f"not merely lead)"
        )
    return (
        f"Brier {challenger['brier']:.4f} vs incumbent {champion['brier']:.4f} "
        f"(lead {lead:.4f} >= margin {margin}) over {challenger['n_cards_paired']} paired "
        f"card(s), {challenger['n_observations']} observation(s)"
    )


def swap_champion(registry_text: str, to_arm: str) -> str:
    """Rewrite the ``champion:`` line. Raises if it is not exactly one line."""
    pattern = re.compile(r"^(  champion: )(\S+)\s*$", re.MULTILINE)
    matches = pattern.findall(registry_text)
    if len(matches) != 1:
        raise PromotionRefused(
            f"expected exactly one `champion:` line in {REGISTRY_PATH.name}, found "
            f"{len(matches)} — refusing to guess"
        )
    return pattern.sub(lambda m: f"{m.group(1)}{to_arm}", registry_text)


def experiments_entry(*, from_arm: str, to_arm: str, evidence: str,
                      leaderboard_ref: str, rationale: str | None) -> str:
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    body = [
        f"## {stamp} — report-card threshold slot: champion {from_arm} → {to_arm}",
        "",
        f"- **Slot:** `report_card_thresholds` (crucible-evaluator, "
        f"`grading/thresholds/registry.yaml`)",
        f"- **Evidence:** {evidence}",
        f"- **Leaderboard:** {leaderboard_ref}",
        f"- **Reversal:** `python -m grading.thresholds.promote --to {from_arm}` — "
        f"both arms keep being scored either way (champion-challenger §3).",
    ]
    if rationale:
        body.append(f"- **Forced against the gate.** Rationale: {rationale}")
    return "\n".join(body) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--to", required=True, help="arm id to promote (must be in slot.arms)")
    parser.add_argument("--leaderboard", help="path or s3:// URL of the cycle's leaderboard")
    parser.add_argument("--experiments", required=True,
                        help="path to EXPERIMENTS.md — the promotion is recorded there")
    parser.add_argument("--force", action="store_true",
                        help="promote against the evidence gate (requires --rationale)")
    parser.add_argument("--rationale", help="why the gate was overridden; written to the record")
    parser.add_argument("--dry-run", action="store_true", help="print the changes, write nothing")
    args = parser.parse_args(argv)

    registry = load_registry()
    if args.to not in registry.slot.arms:
        print(f"refused: {args.to!r} is not a registered arm "
              f"({', '.join(registry.slot.arms)}) — register it, with its scoring path, "
              f"before promoting it (§10)", file=sys.stderr)
        return 2
    if args.to == registry.slot.champion:
        print(f"refused: {args.to!r} is already the champion", file=sys.stderr)
        return 2

    rationale = None
    if args.force:
        if not args.rationale:
            print("refused: --force requires --rationale — an override is a decision on "
                  "the record", file=sys.stderr)
            return 2
        rationale = args.rationale
        evidence = "PROMOTED AGAINST THE GATE (--force)"
    else:
        if not args.leaderboard:
            print("refused: --leaderboard is required (or --force with --rationale)",
                  file=sys.stderr)
            return 2
        try:
            evidence = check_evidence(
                _read_leaderboard(args.leaderboard), args.to,
                float(registry.slot.hysteresis["promotion_margin_brier"]),
            )
        except PromotionRefused as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 1

    from_arm = registry.slot.champion
    new_registry = swap_champion(REGISTRY_PATH.read_text(), args.to)
    entry = experiments_entry(
        from_arm=from_arm, to_arm=args.to, evidence=evidence,
        leaderboard_ref=args.leaderboard or "n/a (forced)", rationale=rationale,
    )

    if args.dry_run:
        print(f"--- {REGISTRY_PATH}\n+  champion: {args.to}\n\n--- {args.experiments}\n{entry}")
        return 0

    experiments = Path(args.experiments)
    if not experiments.exists():
        print(f"refused: {experiments} does not exist — the record lands with the swap, "
              f"never after it", file=sys.stderr)
        return 2
    experiments.write_text(experiments.read_text().rstrip("\n") + "\n\n" + entry)
    REGISTRY_PATH.write_text(new_registry)
    print(f"champion {from_arm} → {args.to}; recorded in {experiments}. "
          f"Commit both files in one PR (champion-challenger §10).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
