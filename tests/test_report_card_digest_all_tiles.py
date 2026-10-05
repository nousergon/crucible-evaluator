"""Regression fixture for alpha-engine-config-I11989: the Director/retro digest
carries EVERY tile on the card and accounts for EVERY component.

Before the fix, ``summarize_report_card`` iterated a seven-key ``TILE_ORDER``
and nothing else, so a real ten-tile card lost behavioral, contribution_lift
and director_quality whole — including critical RED contribution measurements
with wholly negative CIs and a RED cost-adjusted-quality reading — without any
line saying so.

The fixture below is shaped like the 2026-10-02 ten-tile card (same tile keys,
the same real component names on the three previously-omitted tiles, the same
status / criticality / permanent_na vocabulary) plus one tile no producer
emits today, ``liquidity_risk``, standing in for whatever a future producer
adds. Values are illustrative, not the production numbers.
"""

from __future__ import annotations

import re

import pytest

from director import agent, retro
from director.report_card_digest import TILE_ORDER, summarize_report_card

_PASS = {
    "verdict": "PASS",
    "as_of": {"backtester": "2026-10-03T15:19:19Z", "evaluator_stage": "2026-10-03T15:18:18Z"},
}


def _c(name, status, crit="supporting", **kw):
    return {"name": name, "criticality": crit, "status": status, "trend_decoration": "→", **kw}


def _tile(status, letter, grade, comps, **kw):
    return {"status": status, "letter": letter, "numeric_grade": grade,
            "n_components": len(comps), "components": comps, "unreported": [], **kw}


FUTURE_TILE = "liquidity_risk"


def _card() -> dict:
    return {
        "_provenance": {"run_date": "2026-10-02"},
        "tiles_overall_status": "RED",
        "attestation": _PASS,
        "degraded_component_census": False,
        "tiles": {
            # Deliberately NOT in TILE_ORDER order: rendering order must come
            # from the module, not from the card's key order.
            FUTURE_TILE: _tile("RED", "F", 30.0, [
                _c("spread_capture", "RED", "critical", value=-0.004, red_line=0.0,
                   status_reason="spread capture negative."),
                _c("adv_participation", "GREEN", value=0.02),
                _c("borrow_cost_drag", "PENDING-PRODUCER", value=None,
                   status_reason="producer not yet emitting."),
                _c("halt_exposure", "N/A-MISSING-INPUT"),
            ]),
            "director_quality": _tile("GREEN", "A", 97.5, [
                _c("director_grounding", "GREEN", value=98.0, target=75.0, red_line=40.0),
                _c("director_calibration", "GREEN", value=87.0, target=75.0, red_line=40.0),
                _c("director_actionability", "GREEN", value=95.0, target=75.0, red_line=40.0),
                _c("director_route_degraded", "GREEN", value=0.0, target=0.0, red_line=1.0),
            ]),
            "contribution_lift": _tile("RED", "F", 40.0, [
                _c("exit_rules_contribution_lift", "RED", "critical", value=-0.025,
                   ci_low=-0.031, ci_high=-0.020, red_line=0.0, measurement_horizon="21d",
                   reliability="high", status_reason="exit rules lift negative, CI wholly below 0, N=83."),
                _c("momentum_l1_ic_contribution_lift", "RED", "critical", value=-0.007,
                   ci_low=-0.013, ci_high=-0.0012, red_line=0.0, measurement_horizon="21d",
                   status_reason="momentum lift negative, N=83."),
                _c("meta_l2_ic_contribution_lift", "GREEN", "critical", value=0.051,
                   ci_low=0.040, ci_high=0.063, red_line=0.0),
                _c("sector_teams_avg_contribution_lift", "N/A-NOT-IMPL", "critical",
                   permanent_na=True, permanent_na_reason="no ablation arm declared."),
                _c("thinktank_coverage_ic_contribution_lift", "N/A-NOT-IMPL", "diagnostic",
                   permanent_na=True, permanent_na_reason="no ablation arm declared."),
                _c("entry_triggers_contribution_lift", "N/A-MISSING-INPUT", "critical"),
            ], any_stale=False, stale_artifact_count=0, max_artifact_age_days=0),
            "behavioral": _tile("WATCH", "C", 31.9, [
                _c("turnover", "GREEN", "diagnostic", value=0.28, target=0.3, red_line=0.6),
                _c("decision_reversal", "WATCH", value=0.265, target=0.1, red_line=0.3,
                   measurement_horizon="10d_window", reliability="low",
                   status_reason="31/117 exits re-entered within 10td."),
                _c("conviction_stability", "GREEN", "diagnostic", value=2.9, target=5.0, red_line=15.0),
                _c("cost_adjusted_quality", "RED", value=-0.097, target=0.5, red_line=0.0,
                   measurement_horizon="per_roundtrip",
                   status_reason="median net alpha after entry slippage -0.10%."),
                _c("portfolio_state_drift", "WATCH", "diagnostic", value=0.11, target=0.05, red_line=0.2),
            ]),
            "agent": _tile("WATCH", "C", 100.0, [
                _c("judge_rubric_distribution", "WATCH", "diagnostic", value=0.43),
                _c("agent_validation_failure_rate", "N/A-MISSING-INPUT", "critical"),
                _c("judge_calibration_cohen_kappa", "N/A-NOT-IMPL", "critical"),
                _c("retry_storm", "N/A-NOT-IMPL", permanent_na=True),
            ]),
            "substrate": _tile("RED", "F", 80.8, [
                _c("sf_success_rate", "RED", "critical", value=0.604, target=0.95, red_line=0.8),
                _c("artifact_freshness", "GREEN", value=1.0, trend_decoration="↓"),
            ]),
            "backtester": _tile("RED", "F", 72.2, [
                _c("optimizer_churn", "N/A-LOW-N", "critical"),
                _c("pbo", "WATCH", value=0.4),
            ]),
            "executor": _tile("WATCH", "C", 97.5, [
                _c("entry_triggers", "N/A-MISSING-INPUT", "critical"),
                _c("fill_rate", "GREEN", value=0.99),
            ]),
            "predictor": _tile("RED", "F", 65.3, [
                _c("momentum_l1_ic", "RED", "critical", value=-0.0015, target=0.03, red_line=0.0),
                _c("inference_coverage", "GREEN", "critical", value=1.0),
            ]),
            "research": _tile("WATCH", "C", 63.0, [
                _c("composite_ic", "WATCH", "critical", value=0.01),
                _c("cio", "N/A-NOT-IMPL", "diagnostic", permanent_na=True),
                _c("research_basket", "N/A-MISSING-INPUT"),
                _c("macro", "GREEN", value=0.1),
            ]),
            "portfolio_outcome": _tile("RED", "F", 48.5, [
                _c("information_ratio", "RED", "critical", value=-1.79, ci_low=-4.4, ci_high=0.88,
                   target=0.5, red_line=0.0, measurement_horizon="since_inception"),
                _c("sharpe_ratio", "WATCH", "critical", value=0.49),
                _c("alpha_vs_spy", "RED", "critical", value=-0.121),
                _c("dsr", "N/A-LOW-N"),
                _c("max_drawdown", "GREEN", value=-0.08),
            ]),
        },
    }


_HEAD = re.compile(
    r"^## (\S+) — .*?; (\d+) GREEN, (\d+) adverse, (\d+) N/A"
    r"(?: \((\d+) permanent\))?(?:, (\d+) UNCLASSIFIED)?$",
    re.M,
)


def _heads(text: str) -> dict[str, tuple[int, ...]]:
    return {m.group(1): tuple(int(g or 0) for g in m.groups()[1:]) for m in _HEAD.finditer(text)}


def test_every_tile_rendered_once_known_order_then_unknown():
    card = _card()
    text = summarize_report_card(card)
    order = re.findall(r"^## (\S+) — ", text, re.M)
    assert order == TILE_ORDER + [FUTURE_TILE]
    assert set(order) == set(card["tiles"])  # ten real tiles + the future one


def test_every_tile_head_accounts_for_every_component():
    card = _card()
    heads = _heads(summarize_report_card(card))
    assert set(heads) == set(card["tiles"])
    for key, tile in card["tiles"].items():
        green, adverse, na, _permanent, unclassified = heads[key]
        assert green + adverse + na + unclassified == len(tile["components"]), key


def test_card_census_line_totals_every_component():
    card = _card()
    text = summarize_report_card(card)
    m = re.search(
        r"COMPONENT CENSUS: (\d+) tiles \(.*?\), (\d+) components — (\d+) healthy \(GREEN\), "
        r"(\d+) adverse \(RED/WATCH\), (\d+) unmeasured \(N/A\), (\d+) permanent absence "
        r"\(declared N/A\), (\d+) unclassified\.",
        text,
    )
    assert m, text[:2000]
    n_tiles, total, healthy, adverse, unmeasured, permanent, unclassified = map(int, m.groups())
    assert n_tiles == 11
    assert total == sum(len(t["components"]) for t in card["tiles"].values()) == 40
    assert (healthy, adverse, unmeasured, permanent, unclassified) == (13, 14, 8, 4, 1)
    assert healthy + adverse + unmeasured + permanent + unclassified == total
    assert "PRODUCER CENSUS: the registered component roster and this card agreed" in text


def test_previously_omitted_measurements_reach_the_digest():
    text = summarize_report_card(_card())
    # contribution_lift — critical RED with a wholly negative CI.
    assert "exit_rules_contribution_lift [critical] = RED · value -0.025 (CI [-0.031, -0.02])" in text
    assert "momentum_l1_ic_contribution_lift [critical] = RED" in text
    # behavioral — net alpha after slippage.
    assert "cost_adjusted_quality [supporting] = RED · value -0.097" in text
    assert "median net alpha after entry slippage" in text
    # director_quality — the Director's own GREEN grade is visible.
    assert "## director_quality — GREEN (letter A, 98/100); 4 GREEN, 0 adverse, 0 N/A" in text
    # The future tile's adverse component is expanded like any other.
    assert "spread_capture [critical] = RED" in text


def test_critical_missing_and_permanent_absence_are_named():
    text = summarize_report_card(_card())
    assert ("critical/unreported unmeasured: entry_triggers_contribution_lift (N/A-MISSING-INPUT)"
            in text)
    assert ("critical/unreported unmeasured: agent_validation_failure_rate (N/A-MISSING-INPUT), "
            "judge_calibration_cohen_kappa (N/A-NOT-IMPL)") in text
    assert "critical/unreported unmeasured: entry_triggers (N/A-MISSING-INPUT)" in text
    assert ("permanent absence (declared, excluded from coverage): N/A-NOT-IMPL×2; "
            "critical: sector_teams_avg_contribution_lift") in text
    # Non-critical N/A stays counted, not named.
    assert "halt_exposure" not in text
    assert "research_basket" not in text


def test_unknown_status_is_named_not_dropped():
    text = summarize_report_card(_card())
    assert "borrow_cost_drag" in text
    assert "UNCLASSIFIED status 'PENDING-PRODUCER'" in text
    assert _heads(text)[FUTURE_TILE] == (1, 1, 1, 0, 1)


def test_pinned_recovery_visible_once():
    text = summarize_report_card(
        _card(), pinned_components={"meta_l2_ic_contribution_lift", "director_grounding",
                                    "sector_teams_avg_contribution_lift"})
    for name in ("meta_l2_ic_contribution_lift", "director_grounding",
                 "sector_teams_avg_contribution_lift"):
        assert text.count(name) == 1, name
        line = next(ln for ln in text.splitlines() if name in ln)
        assert "carryover-pinned" in line
    assert "director_grounding [supporting] = GREEN · value 98 vs target 75 / red-line 40" in text
    # A pinned permanent absence leaves the permanent roll-up, which then has
    # no critical name left to list.
    assert "critical: sector_teams_avg_contribution_lift" not in text


@pytest.mark.parametrize("pins", [set(), {"meta_l2_ic_contribution_lift", "turnover", "dsr",
                                          "exit_rules_contribution_lift"}])
def test_no_component_rendered_twice(pins):
    """No component is named on more than one line, pinned or not. (Roll-up
    lines such as ``N/A: N/A-NOT-IMPL×1`` legitimately repeat across tiles;
    they name no component.)"""
    card = _card()
    lines = summarize_report_card(card, pinned_components=pins).splitlines()
    for tile in card["tiles"].values():
        for c in tile["components"]:
            token = re.compile(rf"(?<![a-z0-9_]){re.escape(c['name'])}(?![a-z0-9_])")
            hits = [ln for ln in lines if token.search(ln)]
            assert len(hits) <= 1, (c["name"], hits)
    tile_heads = [ln for ln in lines if ln.startswith("## ")]
    assert len(tile_heads) == len(set(tile_heads)) == len(card["tiles"])


def test_reliability_horizon_attestation_and_scope_caveats_preserved():
    text = summarize_report_card(_card())
    assert "decision_reversal" in text and "reliability LOW" in text
    assert "horizon 10d_window" in text and "horizon 21d" in text
    assert "CORRECTNESS ATTESTATION: PASS" in text
    assert "RUN SCOPE: UNKNOWN" in text
    assert "artifact_freshness GREEN but trending ↓ (drift-watch)" in text


def test_unreadable_empty_and_miscounted_tiles_are_named():
    card = _card()
    card["tiles"]["ghost_tile"] = None
    card["tiles"]["empty_tile"] = {"status": "N/A-MISSING-INPUT", "components": []}
    card["tiles"]["executor"]["n_components"] = 3
    card["tiles"]["contribution_lift"].update(any_stale=True, stale_artifact_count=2,
                                              max_artifact_age_days=9)
    text = summarize_report_card(card)
    assert "## ghost_tile — ⚠ UNREADABLE tile entry (NoneType)" in text
    assert "## empty_tile — N/A-MISSING-INPUT" in text
    assert "this tile carries NO components" in text
    assert "tile declares n_components=3 but carries 2 component rows" in text
    assert "stale source artifacts: 2 (max age 9d)" in text
    assert "COMPONENT CENSUS: 13 tiles" in text


def test_degraded_producer_census_names_the_missing_components():
    card = _card()
    card["degraded_component_census"] = True
    card["component_census_unreported"] = ["executor.slippage_bps"]
    text = summarize_report_card(card)
    assert "⚠ PRODUCER CENSUS DEGRADED — registered but not rendered: executor.slippage_bps" in text
    del card["degraded_component_census"]
    assert "⚠ PRODUCER CENSUS: UNKNOWN" in summarize_report_card(card)


def test_plan_and_retro_share_the_same_complete_digest():
    card = _card()
    digest = summarize_report_card(card)
    plan_human = agent.build_messages(card)[1][1]
    retro_human = retro.build_messages({"run_date": "2026-09-25"}, card)[1][1]
    assert digest in plan_human
    assert digest in retro_human
    for key in card["tiles"]:
        assert f"## {key} — " in plan_human and f"## {key} — " in retro_human
