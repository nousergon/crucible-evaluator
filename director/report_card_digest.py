"""
report_card_digest.py — condense a Report Card v2 into a compact, Director-ready
digest.

Feeding all ~65 raw MetricRecords (the 63 KB report_card.json) into the prompt
is token-heavy and buries the signal. The Director's job is to weigh the
*current issues / weaknesses*, so the digest leads with the overall status,
each tile's status/grade, and then the components that actually warrant
attention — every RED/WATCH (with its value, target/red-line, status_reason and
trend) plus a roll-up of what's N/A and why (which producers aren't wired). A
GREEN component with no adverse trend is summarized in one line, not expanded.

Output is plain text (markdown-ish) so it drops straight into the prompt.
"""

from __future__ import annotations

from grading.pipeline_gates import MEASURED, PIPELINE_GATES_KEY
from grading.run_scope import MEASURED as SCOPE_MEASURED
from grading.run_scope import RUN_SCOPE_KEY

TILE_ORDER = [
    "portfolio_outcome", "research", "predictor", "executor",
    "backtester", "substrate", "agent",
]


def _is_na(status: str) -> bool:
    return str(status).startswith("N/A")


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _component_line(c: dict) -> str:
    parts = [f"  - {c.get('name')} [{c.get('criticality', '?')}] = {_chip(c.get('status'))}"]
    val = c.get("value")
    if val is not None:
        seg = f"value {_fmt(val)}"
        if c.get("ci_low") is not None and c.get("ci_high") is not None:
            seg += f" (CI [{_fmt(c['ci_low'])}, {_fmt(c['ci_high'])}])"
        if c.get("target") is not None:
            seg += f" vs target {_fmt(c['target'])}"
        if c.get("red_line") is not None:
            seg += f" / red-line {_fmt(c['red_line'])}"
        parts.append(seg)
    if c.get("trend_decoration") and c["trend_decoration"] != "→":
        parts.append(f"trend {c['trend_decoration']}")
    # L4562 / ARCHITECTURE §18 — surface metric reliability so the Director can
    # hedge: a low-reliability metric (or one measured at a non-canonical
    # horizon) must NOT drive a confident root-cause/de-risk prescription.
    if c.get("measurement_horizon"):
        parts.append(f"horizon {c['measurement_horizon']}")
    if c.get("reliability") == "low":
        parts.append("⚠ reliability LOW — verify metric validity before acting")
    reason = c.get("status_reason")
    line = " · ".join(parts)
    if reason:
        line += f"\n      reason: {reason}"
    return line


def _chip(status) -> str:
    return str(status or "N/A")


def summarize_report_card(card: dict) -> str:
    """Return a compact text digest of the report card for the prompt."""
    if not card:
        return "No Report Card available for this cycle."

    prov = card.get("_provenance", {}) or {}
    run_date = prov.get("run_date", "?")
    overall = card.get("tiles_overall_status", "N/A")
    tiles = card.get("tiles", {}) or {}

    out = [f"# Report Card v2 — run_date {run_date}", f"OVERALL: {overall}"]

    # sf-pipeline-policy §2.3a rule 3 — every surface presenting the run's numbers
    # carries the correctness-verdict state. The Director acts on these numbers
    # (files issues, grades its own prior plan), so it depends on them being
    # uncontaminated and must see when that was not established. Rendered BEFORE
    # the tiles so it cannot be read past.
    #
    # `degraded_staleness` is surfaced here too: aggregate.py has documented since
    # config#2885 that "the Director agent's prompt MUST check this before treating
    # the card as ground truth", but the flag never reached the digest — the same
    # verdict-does-not-propagate defect one axis over.
    att = card.get("attestation") or {}
    verdict = att.get("verdict")
    # `as_of` rides with the state so a verdict reads STALE rather than green:
    # "PASS" with a timestamp from a previous cycle is a different fact from
    # "PASS" established minutes ago, and a state rendered without its timestamp
    # cannot express the difference.
    as_of = att.get("as_of") or {}
    stamps = ", ".join(
        f"{k} as-of {v or 'never'}" for k, v in sorted(as_of.items())
    )
    stamp_suffix = f" [{stamps}]" if stamps else ""
    if verdict == "PASS":
        out.append(
            "CORRECTNESS ATTESTATION: PASS — the deployed backtest engine, the "
            "Evaluator stage's ranking metrics, and the evaluator's own quant "
            "primitives all agreed with their known answers." + stamp_suffix
        )
    elif verdict:
        out.append(
            f"⚠ CORRECTNESS ATTESTATION: {verdict} — {att.get('reason', '')} "
            "The numbers below are NOT established as correct: do not assert a "
            "metric moved, and do not prescribe an action premised on its level."
            + stamp_suffix
        )
    else:
        out.append(
            "⚠ CORRECTNESS ATTESTATION: UNKNOWN — this card carries no attestation "
            "block. Treat every number below as unverified."
        )
    if att.get("promotion_withheld"):
        # A withheld promotion is a fact about the LIVE system, not about the
        # card: the executor is still running last cycle's parameters. The
        # Director prescribes actions premised on the current config, so it must
        # not be able to read past this.
        out.append(
            "⚠ PROMOTION WITHHELD: the Evaluator stage ran under a forced freeze "
            "this cycle — config/executor_params.json and config/producer_champion.json "
            "were NOT updated. The live executor is on the PREVIOUS cycle's "
            "parameters; do not describe any config change as having taken effect."
        )
    # alpha-engine-config-I7282 — §2.3a rule 3. The attestation above says
    # whether the arithmetic behind these numbers is right; this says whether the
    # pipeline's own pre-spend correctness gates ran at all before it spent. Both
    # polarities render: a line that appears only on the bad week is
    # indistinguishable from a producer that stopped emitting.
    gates = card.get(PIPELINE_GATES_KEY) or {}
    statement = gates.get("statement")
    if statement:
        out.append(("PIPELINE GATES: " if gates.get("verdict") == MEASURED
                    else "⚠ PIPELINE GATES: ") + statement)
    else:
        out.append(
            "⚠ PIPELINE GATES: UNKNOWN — this card carries no pipeline_gates "
            "block, so nothing says whether the weekly run's pre-spend "
            "correctness gates ran. Treat the numbers below as unattested."
        )
    # alpha-engine-config-I7620 — the DENOMINATOR, rendered before the tiles for
    # the same reason as the two verdicts above: the Director ACTS on these
    # numbers, and every one of them was computed over whatever stages this run
    # dispatched. That set moves. On 2026-08-14 the Director called the
    # deliberate absence of pit_parity "the producer never ran this cycle" and
    # withheld its acting authority, because nothing on the card distinguished a
    # stage switched off by `skip_parity` from a stage that died.
    #
    # Both polarities render. A scope line that appears only on a narrow week is
    # indistinguishable from a producer that stopped emitting one.
    scope = card.get(RUN_SCOPE_KEY) or {}
    scope_statement = scope.get("statement")
    if scope_statement and scope.get("verdict") == SCOPE_MEASURED:
        out.append("RUN SCOPE: " + scope_statement)
    elif scope_statement:
        out.append(
            "⚠ RUN SCOPE: " + scope_statement
            + " Grade nothing against a stage list this card cannot confirm was "
            "dispatched — a narrow run and an unmeasured one are different "
            "findings."
        )
    else:
        out.append(
            "⚠ RUN SCOPE: UNKNOWN — this card carries no run_scope block, so "
            "which stages the week actually dispatched is not established. A "
            "stage that was switched off by an operator flag is indistinguishable "
            "here from one that ran and failed; do not report either as the other."
        )

    if card.get("degraded_staleness"):
        out.append("⚠ DEGRADED (staleness): stale tiles — "
                   + ", ".join(card.get("stale_tiles") or []))
    out.append("")

    for key in TILE_ORDER:
        tile = tiles.get(key)
        if not tile:
            continue
        comps = tile.get("components", []) or []
        adverse = [c for c in comps if str(c.get("status")) in ("RED", "WATCH")]
        na = [c for c in comps if _is_na(c.get("status"))]
        green = [c for c in comps if str(c.get("status")) == "GREEN"]
        grade = tile.get("numeric_grade")
        head = (f"## {key} — {tile.get('status')} (letter {tile.get('letter', 'N/A')}"
                + (f", {grade:.0f}/100" if grade is not None else "")
                + f"); {len(green)} GREEN, {len(adverse)} adverse, {len(na)} N/A")
        out.append(head)
        # Expand the adverse (RED/WATCH) components — these are the issues.
        for c in adverse:
            out.append(_component_line(c))
        # Roll up N/A by reason-kind (don't expand each).
        if na:
            kinds: dict[str, int] = {}
            for c in na:
                kinds[str(c.get("status"))] = kinds.get(str(c.get("status")), 0) + 1
            out.append("  - N/A: " + ", ".join(f"{k}×{v}" for k, v in sorted(kinds.items())))
        # GREEN with a downward drift is still worth a flag (drift-watch).
        for c in green:
            if c.get("trend_decoration") in ("↓", "↓↓"):
                out.append(f"  - {c.get('name')} GREEN but trending {c['trend_decoration']} (drift-watch)")
        out.append("")

    return "\n".join(out).strip()
