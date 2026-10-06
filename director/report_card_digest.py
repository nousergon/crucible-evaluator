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

**Pinned components (alpha-engine-config-I8380).** That collapse is a one-way
ratchet for carry-over: a ledger item can only be closed on evidence the
Director's prompt actually carries, and a metric that recovered to GREEN was
counted but never named, so the Director could never cite the recovery. The
caller (``director.agent.build_messages`` / ``director.retro.build_messages``)
may pass ``pinned_components`` — the lower-cased component names cited by open
carry-over items / the prior plan's action items, resolved against this same
card — and every pinned name is rendered in full (name, grade, value, status)
regardless of colour, alongside the adverse expansion. This is Design 1 of the
two the issue offered (pin only cited names, not "always expand every GREEN
component") because it keeps the digest bounded by the size of the *carried
backlog*, not the size of the *card*: `DirectorPlanPromptChars` feeds the
plan-call latency/cost telemetry a sibling issue (I8164/I8200) is separately
re-anchoring, so an unconditional expansion of every GREEN/N-A component would
add real cost for rows nothing is asking to close.

**Every tile, every component (alpha-engine-config-I11989).** ``TILE_ORDER``
is a preferred RENDERING ORDER, never the census. Until 2026-10-05 it was also
the only iteration source, so the three tiles it did not list — behavioral,
contribution_lift and director_quality — never reached the Director or the
retro judge, and with them every adverse measurement they carried
(exit_rules_contribution_lift RED with a wholly negative CI, behavioral
cost_adjusted_quality RED) plus the Director's own GREEN quality grade. The
digest now renders the known tiles in that order and then EVERY remaining tile
on the input card, sorted, so a tile a future producer adds is carried the day
it appears rather than the day someone remembers this list. Each component is
classified exactly once — healthy (GREEN), adverse (RED/WATCH), unmeasured
(N/A), permanent absence (N/A with a declared ``permanent_na``) or UNCLASSIFIED
(any status none of those match, named rather than dropped) — and a
card-level ``COMPONENT CENSUS`` line states the totals, so an omission is a
visible arithmetic mismatch instead of a silence. Critical unmeasured and
unreported components are NAMED; non-critical ones are counted by reason-kind,
which keeps the digest bounded by the card's problems, not its size.

Output is plain text (markdown-ish) so it drops straight into the prompt.
"""

from __future__ import annotations

from grading.pipeline_gates import MEASURED, PIPELINE_GATES_KEY
from grading.run_scope import MEASURED as SCOPE_MEASURED
from grading.run_scope import RUN_SCOPE_KEY

#: Preferred rendering ORDER for the tiles this module knows by name. It is
#: not a whitelist: :func:`ordered_tile_keys` appends every other tile on the
#: card after these (alpha-engine-config-I11989), so a missing entry here
#: costs position, never visibility.
TILE_ORDER = [
    "portfolio_outcome", "research", "predictor", "executor",
    "backtester", "substrate", "agent",
    "behavioral", "contribution_lift", "director_quality",
]

#: The five dispositions every component is accounted under — exactly one each.
HEALTHY = "healthy"
ADVERSE = "adverse"
UNMEASURED = "unmeasured"
PERMANENT = "permanent_absence"
UNCLASSIFIED = "unclassified"
DISPOSITIONS = (HEALTHY, ADVERSE, UNMEASURED, PERMANENT, UNCLASSIFIED)


def ordered_tile_keys(tiles: dict) -> list[str]:
    """Every tile key on the card: the known ones in ``TILE_ORDER`` order,
    then every remaining key, sorted, so the order is deterministic and a
    tile this module has never heard of is still rendered."""
    known = [k for k in TILE_ORDER if k in tiles]
    rest = sorted((k for k in tiles if k not in TILE_ORDER), key=str)
    return known + rest


def classify_component(c) -> str:
    """The one disposition a component is accounted under.

    A status none of GREEN / RED / WATCH / ``N/A*`` matches (a new status
    vocabulary, a missing status, an entry that is not a mapping) is
    UNCLASSIFIED — rendered by name further down, never folded into a count
    it does not belong to and never dropped.
    """
    if not isinstance(c, dict):
        return UNCLASSIFIED
    status = str(c.get("status") or "")
    if status in ("RED", "WATCH"):
        return ADVERSE
    if status == "GREEN":
        return HEALTHY
    if status and _is_na(status):
        return PERMANENT if c.get("permanent_na") else UNMEASURED
    return UNCLASSIFIED


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
    # alpha-engine-config-I11089 — the window and source the value was computed
    # over, as the tile declared it ("undeclared" when it could not be read).
    # Two components on one tile can carry different N over different dates;
    # without this the Director reads them as one population.
    if (wl := _window_label(c)):
        parts.append(f"window {wl}")
    if c.get("reliability") == "low":
        parts.append("⚠ reliability LOW — verify metric validity before acting")
    reason = c.get("status_reason")
    line = " · ".join(parts)
    if reason:
        line += f"\n      reason: {reason}"
    return line


def _window_label(obj) -> str | None:
    w = obj.get("window") if isinstance(obj, dict) else None
    if not isinstance(w, dict):
        return None
    return str(w.get("label") or "") or None


def _chip(status) -> str:
    return str(status or "N/A")


def summarize_report_card(card: dict, *, pinned_components: set[str] | None = None) -> str:
    """Return a compact text digest of the report card for the prompt.

    ``pinned_components``: lower-cased component names to render in full
    (name, grade, value, status) regardless of colour — components named by
    an open carry-over item / prior-plan action item. See the module
    docstring, "Pinned components", for why this is bounded by the carried
    backlog rather than by the card.
    """
    pinned_components = pinned_components or set()
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
    # config-I7620 follow-up. PASS above now means "the arithmetic we MEASURED is
    # right". When the contamination producer was not dispatched, that is a
    # second, weaker claim than the four-halves one, and the Director must be
    # told which it is holding — otherwise it re-proposes "fix the pit_parity
    # timeout" every week against a stage an operator switched off on purpose,
    # which is exactly what the 2026-08-14 plan did as its P0.
    if att.get("contamination_in_scope") is False:
        scope_reason = (att.get("contamination") or {}).get("scope", {}).get("reason", "")
        out.append(
            "⚠ CONTAMINATION NOT MEASURED — the look-ahead check was NOT DISPATCHED "
            f"this run. {scope_reason} This is an operator decision already tracked, "
            "NOT a failure: do not propose fixing, re-running or diagnosing that "
            "producer, and do not describe the run as contamination-free."
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
    # alpha-engine-config-I8380 — distinguish "absent from the card" from "not
    # carried into this digest". A GREEN/N-A component that is not itemized
    # below is COUNTED, not absent: it appears in its tile's "N GREEN, N
    # adverse, N N/A" head, and if it is named by an open carry-over item it is
    # additionally pinned and expanded in full further down. Only a name that
    # appears in NEITHER a tile head count NOR a pinned/adverse line is
    # genuinely absent from report_card.json.
    out.append(
        "NOTE: every component on this card is counted in its tile's "
        "GREEN/adverse/N-A totals below, even when not itemized by name. "
        "'Not visible on this card' is true ONLY for a metric absent from "
        "every tile's components list AND every count — never say it about a "
        "metric this digest merely counted instead of naming."
    )
    out.extend(_census_lines(card, tiles))
    out.append("")

    for key in ordered_tile_keys(tiles):
        out.extend(_tile_lines(key, tiles.get(key), pinned_components))
        out.append("")

    return "\n".join(out).strip()


def _comps(tile) -> list:
    if not isinstance(tile, dict):
        return []
    comps = tile.get("components")
    return list(comps) if isinstance(comps, list) else []


def _census_lines(card: dict, tiles: dict) -> list[str]:
    """alpha-engine-config-I11989 — the whole-card census, before any tile.

    States how many tiles and components the card carries and how every one
    of them is dispositioned, so the per-tile sections below can be checked
    against a total rather than trusted to be complete. Also carries the
    producer's own roster-vs-card census (``degraded_component_census``,
    alpha-engine-config-I8193) in BOTH polarities: a line that appears only on
    the bad week is indistinguishable from a producer that stopped emitting.
    """
    keys = ordered_tile_keys(tiles)
    counts = {d: 0 for d in DISPOSITIONS}
    for key in keys:
        for c in _comps(tiles.get(key)):
            counts[classify_component(c)] += 1
    total = sum(counts.values())
    out = [
        f"COMPONENT CENSUS: {len(keys)} tiles ({', '.join(str(k) for k in keys) or 'none'}), "
        f"{total} components — {counts[HEALTHY]} healthy (GREEN), "
        f"{counts[ADVERSE]} adverse (RED/WATCH), {counts[UNMEASURED]} unmeasured (N/A), "
        f"{counts[PERMANENT]} permanent absence (declared N/A), "
        f"{counts[UNCLASSIFIED]} unclassified. Every tile and component is accounted "
        "for in its tile section below."
    ]
    census_flag = card.get("degraded_component_census")
    if census_flag is False:
        out.append("PRODUCER CENSUS: the registered component roster and this card agreed "
                   "(nothing unreported, nothing unregistered).")
    elif census_flag:
        unreported = card.get("component_census_unreported") or []
        unregistered = card.get("component_census_unregistered") or []
        detail = []
        if unreported:
            detail.append("registered but not rendered: " + ", ".join(map(str, unreported)))
        if unregistered:
            detail.append("rendered but not registered: " + ", ".join(map(str, unregistered)))
        if card.get("component_census_error"):
            detail.append(f"census error: {card['component_census_error']}")
        out.append("⚠ PRODUCER CENSUS DEGRADED — " + ("; ".join(detail) or "no detail on the card")
                   + ". A component named here is MISSING from the card, not healthy.")
    else:
        out.append("⚠ PRODUCER CENSUS: UNKNOWN — this card carries no degraded_component_census "
                   "flag, so nothing establishes that every registered component was rendered.")
    return out


def _tile_lines(key: str, tile, pinned_components: set[str]) -> list[str]:
    """One tile's section. Every component on the tile lands in exactly one
    disposition, and every disposition is either expanded by name or counted
    in the head — nothing on the tile is skipped."""
    if not isinstance(tile, dict):
        return [f"## {key} — ⚠ UNREADABLE tile entry ({type(tile).__name__}); "
                "no components could be read — treat this tile as unmeasured, not healthy."]
    comps = _comps(tile)
    by: dict[str, list] = {d: [] for d in DISPOSITIONS}
    for c in comps:
        by[classify_component(c)].append(c)
    adverse, green = by[ADVERSE], by[HEALTHY]
    na, permanent, other = by[UNMEASURED], by[PERMANENT], by[UNCLASSIFIED]
    grade = tile.get("numeric_grade")
    head = (f"## {key} — {tile.get('status')} (letter {tile.get('letter', 'N/A')}"
            + (f", {grade:.0f}/100" if isinstance(grade, (int, float)) else "")
            + f"); {len(green)} GREEN, {len(adverse)} adverse, {len(na) + len(permanent)} N/A"
            + (f" ({len(permanent)} permanent)" if permanent else "")
            + (f", {len(other)} UNCLASSIFIED" if other else ""))
    out = [head]
    if (wl := _window_label(tile)):
        out.append(f"  - window: {wl}")
    if not comps:
        out.append("  - ⚠ this tile carries NO components — nothing on it is measured.")
    declared_n = tile.get("n_components")
    if isinstance(declared_n, int) and declared_n != len(comps):
        out.append(f"  - ⚠ tile declares n_components={declared_n} but carries {len(comps)} "
                   "component rows — the difference is not visible to this digest.")
    if tile.get("unreported"):
        out.append("  - ⚠ unreported (registered, no record): "
                   + ", ".join(map(str, tile["unreported"])))
    if tile.get("any_stale"):
        out.append(f"  - ⚠ stale source artifacts: {tile.get('stale_artifact_count', '?')} "
                   f"(max age {tile.get('max_artifact_age_days', '?')}d)")

    def _pinned(c) -> bool:
        return str(c.get("name") or "").strip().lower() in pinned_components

    # Expand the adverse (RED/WATCH) components — these are the issues.
    for c in adverse:
        out.append(_component_line(c))
    # alpha-engine-config-I8380 — a GREEN/N-A component named by an open
    # carry-over item is pinned into full detail regardless of colour, so
    # the Director can cite its RECOVERY and close the item. Rendered once
    # here (not also in the rollups below) — a pinned name is excluded
    # from the N/A kind-count line and the drift-watch loop.
    pinned_green = [c for c in green if _pinned(c)]
    pinned_na = [c for c in na + permanent if _pinned(c)]
    for c in pinned_green + pinned_na:
        out.append(_component_line(c) + " (carryover-pinned — cite this to close the item)")
    # A status outside the known vocabulary is named, never counted into a
    # bucket it does not belong to.
    for c in other:
        if isinstance(c, dict):
            out.append(_component_line(c) + f" (UNCLASSIFIED status {c.get('status')!r} — "
                       "not GREEN/RED/WATCH/N/A; treat as unverified)")
        else:
            out.append(f"  - ⚠ unparseable component entry ({type(c).__name__}) — "
                       "treat as unmeasured")
    # Roll up N/A by reason-kind (don't expand each), excluding pinned ones
    # already expanded above.
    na_unpinned = [c for c in na if c not in pinned_na]
    if na_unpinned:
        out.append("  - N/A: " + _kind_counts(na_unpinned))
        # I11989 — a CRITICAL component that is unmeasured, or a registered
        # one that rendered no record at all, is a blocker the Director must
        # be able to name; count-only would hide which one.
        missing = [c for c in na_unpinned
                   if c.get("criticality") == "critical" or c.get("unreported")]
        if missing:
            out.append("  - critical/unreported unmeasured: " + ", ".join(
                f"{c.get('name')} ({c.get('status')})" for c in missing))
    perm_unpinned = [c for c in permanent if c not in pinned_na]
    if perm_unpinned:
        line = "  - permanent absence (declared, excluded from coverage): " + _kind_counts(perm_unpinned)
        crit = [str(c.get("name")) for c in perm_unpinned if c.get("criticality") == "critical"]
        if crit:
            line += "; critical: " + ", ".join(crit)
        out.append(line)
    # GREEN with a downward drift is still worth a flag (drift-watch),
    # unless already expanded above as pinned.
    for c in green:
        if c in pinned_green:
            continue
        if c.get("trend_decoration") in ("↓", "↓↓"):
            out.append(f"  - {c.get('name')} GREEN but trending {c['trend_decoration']} (drift-watch)")
    return out


def _kind_counts(comps: list) -> str:
    kinds: dict[str, int] = {}
    for c in comps:
        kinds[str(c.get("status"))] = kinds.get(str(c.get("status")), 0) + 1
    return ", ".join(f"{k}×{v}" for k, v in sorted(kinds.items()))
