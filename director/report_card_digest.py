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

    out = [f"# Report Card v2 — run_date {run_date}", f"OVERALL: {overall}", ""]

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
