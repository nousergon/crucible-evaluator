"""The carry-over ledger is bounded WHERE THE PROMPT READS IT
(alpha-engine-config-I8163).

`carryover.ACTIVE_LEDGER_MAX_ITEMS` (40) is applied by
``merge_plan_into_ledger``, which runs AFTER the plan call — so it is a bound
on what the ledger keeps, not on what the prompt carries, and the read path had
no bound at all. `DIRECTOR_PLAN_CEILING_S` is now derived from AWS Lambda's
900s function maximum, so there is no third raise available and the input has
to be bounded instead.

These tests pin the three properties that make the bound real rather than
nominal:

  1. It holds for a ledger of ANY size, in characters, not just in item count.
  2. Nothing is elided silently — the counted summary line and the
     ``DirectorPlanCarryoverOmitted`` metric both report the omission.
  3. A P0 is never elided, and a P0 set that exceeds the cap says so.
"""

from __future__ import annotations

import json

import pytest

from director.agent import (
    CARRYOVER_PROMPT_CHAR_BUDGET,
    CARRYOVER_PROMPT_MAX_ITEMS,
    _CARRYOVER_OMITTED_MARKER,
    _carryover_context,
    _carryover_item_count,
    _carryover_omitted_count,
    _emit_plan_latency,
    build_messages,
    select_carryover_rows,
)
from director.carryover import ACTIVE_LEDGER_MAX_ITEMS, order_for_prompt

_PRIORITIES = ["P1", "P2", "P3"]


def _row(n: int, *, priority: str | None = None, carry_count: int | None = None) -> dict:
    """A ledger row shaped like the live artifact, with generous free text.

    ``title`` and ``rationale`` are model-authored on the real ledger, which is
    why the char budget cannot be inferred from the item cap alone — these are
    sized well above the 214-char maximum measured on the live ledger
    (2026-08-21) so the test exercises the budget, not just the count.
    """
    return {
        "id": f"item-{n:04d}",
        "title": f"Investigate {n} " + ("long-title-filler " * 12),
        "status": "carried_over",
        "proposed_owner": "research",
        "priority": priority or _PRIORITIES[n % len(_PRIORITIES)],
        "carry_count": n % 6 if carry_count is None else carry_count,
        "first_seen": f"2026-06-{1 + (n % 28):02d}",
        "last_seen": "2026-08-21",
        "evidence": ["sharpe_ratio reads RED", "alpha_vs_spy reads RED"],
        "rationale": f"Rationale {n} " + ("rationale-filler " * 20),
    }


@pytest.fixture
def card() -> dict:
    """A report card carrying enough metrics that rows get annotated.

    The annotation is what makes a rendered row ~300 chars rather than ~180, so
    a bound measured without it would be measured against the wrong prompt.
    """
    return {
        "components": [
            {
                "component": "research",
                "metrics": [
                    {"name": "sharpe_ratio", "status": "RED"},
                    {"name": "alpha_vs_spy", "status": "RED"},
                ],
            }
        ],
        "_provenance": {"run_date": "2026-08-21"},
    }


# ── 1. The bound ─────────────────────────────────────────────────────────────


def test_two_hundred_item_ledger_stays_inside_the_declared_char_budget(card):
    """The headline assertion of alpha-engine-config-I8163."""
    ledger = {"items": [_row(n) for n in range(200)]}
    section = _carryover_context(ledger, card)
    assert len(section) <= CARRYOVER_PROMPT_CHAR_BUDGET, (
        f"carry-over section is {len(section)} chars against a declared budget "
        f"of {CARRYOVER_PROMPT_CHAR_BUDGET}"
    )


def test_the_bound_is_monotone_not_merely_satisfied_at_200(card):
    """A ledger 10x larger must not produce a larger section.

    The failure this catches is a cap that bounds the ITEM count while some
    per-item cost still scales with the ledger — the section would keep growing
    while the count-based test kept passing.
    """
    small = _carryover_context({"items": [_row(n) for n in range(200)]}, card)
    huge = _carryover_context({"items": [_row(n) for n in range(2000)]}, card)
    # The elision line carries a count, so the huge ledger's section may be a
    # handful of characters longer; it may not be materially larger.
    assert len(huge) <= len(small) + 40
    assert len(huge) <= CARRYOVER_PROMPT_CHAR_BUDGET


def test_the_whole_prompt_is_bounded_not_only_the_section(card):
    """``build_messages`` is what the LLM is handed; the bound has to hold there."""
    lean = build_messages(card, carryover={"items": [_row(n) for n in range(5)]})
    fat = build_messages(card, carryover={"items": [_row(n) for n in range(200)]})
    lean_chars = sum(len(c) for _, c in lean)
    fat_chars = sum(len(c) for _, c in fat)
    assert fat_chars - lean_chars <= CARRYOVER_PROMPT_CHAR_BUDGET


def test_item_cap_is_at_most_the_ledgers_own_active_bound():
    """A prompt cap above the ledger's own active bound would never bite."""
    assert CARRYOVER_PROMPT_MAX_ITEMS <= ACTIVE_LEDGER_MAX_ITEMS


# ── 2. The elision is reported, never silent ─────────────────────────────────


def test_elision_is_counted_and_characterised(card):
    ledger = {"items": [_row(n) for n in range(60)]}
    section = _carryover_context(ledger, card)
    omitted = 60 - CARRYOVER_PROMPT_MAX_ITEMS
    assert f"{_CARRYOVER_OMITTED_MARKER}{omitted}" in section
    # A bare count is weaker than a count plus what was in the tail.
    assert "oldest first seen 2026-06-" in section
    assert "longest carried" in section
    assert "PROMPT BOUND" in section
    assert "P3×" in section  # the priority distribution of the omitted set


def test_an_uncapped_ledger_says_nothing_about_omission(card):
    section = _carryover_context({"items": [_row(n) for n in range(5)]}, card)
    assert _CARRYOVER_OMITTED_MARKER not in section
    assert "carry-over ledger, active items=5" in section


def test_markers_are_parseable_from_the_messages(card):
    messages = build_messages(card, carryover={"items": [_row(n) for n in range(60)]})
    assert _carryover_item_count(messages) == CARRYOVER_PROMPT_MAX_ITEMS
    assert _carryover_omitted_count(messages) == 60 - CARRYOVER_PROMPT_MAX_ITEMS


def test_omitted_count_is_published_on_the_metric_including_the_zero(capsys):
    """`principles.md` §2.7: a metric that only appears when something is wrong
    is indistinguishable from a dead emitter."""
    for omitted in (0, 17):
        record = _emit_plan_latency(
            elapsed_s=1.0,
            outcome="ok",
            prompt_chars=100,
            carryover_items=20,
            carryover_omitted=omitted,
        )
        assert record["DirectorPlanCarryoverOmitted"] == omitted
    emitted = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("{")
    ]
    names = {
        m["Name"]
        for rec in emitted
        for m in rec["_aws"]["CloudWatchMetrics"][0]["Metrics"]
    }
    assert "DirectorPlanCarryoverOmitted" in names
    assert [rec["DirectorPlanCarryoverOmitted"] for rec in emitted] == [0, 17]


def test_carried_plus_omitted_equals_the_ledger(card):
    """The two counts must decompose the ledger, or the omission is unreadable."""
    for size in (1, 19, 20, 21, 137):
        messages = build_messages(card, carryover={"items": [_row(n) for n in range(size)]})
        assert _carryover_item_count(messages) + _carryover_omitted_count(messages) == size


# ── 3. Selection: defensible, deterministic, and P0-safe ─────────────────────


def test_selection_is_priority_then_weeks_carried(card):
    rows = [
        _row(1, priority="P3", carry_count=9),
        _row(2, priority="P1", carry_count=1),
        _row(3, priority="P1", carry_count=7),
        _row(4, priority="P0", carry_count=0),
    ]
    assert [r["priority"] for r in order_for_prompt(rows)] == ["P0", "P1", "P1", "P3"]
    p1s = [r for r in order_for_prompt(rows) if r["priority"] == "P1"]
    assert [r["carry_count"] for r in p1s] == [7, 1]


def test_selection_is_deterministic():
    rows = [_row(n) for n in range(80)]
    shown_a, omitted_a, _ = select_carryover_rows(rows)
    shown_b, omitted_b, _ = select_carryover_rows(list(reversed(rows)))
    assert [r["id"] for r in shown_a] == [r["id"] for r in shown_b]
    assert sorted(r["id"] for r in omitted_a) == sorted(r["id"] for r in omitted_b)


def test_a_p0_is_never_elided(card):
    """A cap that can drop the Director's own highest-priority carried
    commitment is not worth the tokens it saves."""
    rows = [_row(n) for n in range(200)]
    rows.append(_row(999, priority="P0", carry_count=0))
    shown, omitted, over_cap = select_carryover_rows(rows)
    assert not over_cap
    assert "item-0999" in {r["id"] for r in shown}
    assert "item-0999" not in {r["id"] for r in omitted}
    assert "item-0999" in _carryover_context({"items": rows}, card)


def test_a_p0_set_larger_than_the_cap_carries_all_of_them_and_says_so(card):
    """That is a finding about the backlog, not a prompt-sizing problem."""
    n_p0 = CARRYOVER_PROMPT_MAX_ITEMS + 6
    rows = [_row(n, priority="P0") for n in range(n_p0)] + [_row(500 + n) for n in range(30)]
    shown, omitted, over_cap = select_carryover_rows(rows)
    assert over_cap
    assert len(shown) == n_p0
    assert all(r["priority"] == "P0" for r in shown)
    assert not any(r["priority"] == "P0" for r in omitted)
    section = _carryover_context({"items": rows}, card)
    assert "P0 SET EXCEEDS THE PROMPT CAP" in section
    # The carve-out is allowed to breach the budget; it is not allowed to do so
    # quietly, and the ELIDED tail is still reported.
    assert _CARRYOVER_OMITTED_MARKER in section


def test_selection_never_reads_the_live_card_annotation(card):
    """The cap and the annotation are separate mechanisms.

    Eliding a row because this week's card contradicts it would be the
    close-and-look-away suppression ``loop_verification`` refuses, and the
    ledger is known to carry stale rows (alpha-engine-config-I8178, open). So
    the selection must be identical whether or not a card is supplied.
    """
    rows = [_row(n) for n in range(60)]
    with_card = _carryover_context({"items": rows}, card)
    without_card = _carryover_context({"items": rows}, None)
    ids_with = [line.split("]")[0] for line in with_card.splitlines() if line.startswith("  - [")]
    ids_without = [line.split("]")[0] for line in without_card.splitlines() if line.startswith("  - [")]
    assert ids_with == ids_without


def test_malformed_rows_do_not_raise(card):
    """Pure and total: the ledger is a model-written artifact."""
    rows = [
        {},
        {"id": "no-priority"},
        {"id": "bad-priority", "priority": "URGENT", "carry_count": "n/a"},
        {"id": "bad-date", "priority": "P1", "first_seen": "not-a-date"},
    ]
    section = _carryover_context({"items": rows}, card)
    assert "no-priority" in section
    assert len(section) <= CARRYOVER_PROMPT_CHAR_BUDGET


def test_empty_ledger_is_unchanged(card):
    assert "No prior action plan on record" in _carryover_context({"items": []}, card)
    assert "No prior action plan on record" in _carryover_context(None, card)


# ── 4. The char budget is a real second bound, not decoration ────────────────


def test_pathologically_long_rows_spill_before_the_item_cap_is_reached(card):
    """The item cap alone cannot bound the section.

    `title` is model-authored free text on the live ledger, so a handful of very
    long rows must be able to exhaust the budget before 20 rows have rendered —
    otherwise `CARRYOVER_PROMPT_CHAR_BUDGET` would be a number that never binds
    and the bound would rest entirely on an assumption about row size.
    """
    rows = []
    for n in range(40):
        row = _row(n)
        row["title"] = f"Investigate {n} " + ("x" * 1500)
        rows.append(row)
    section = _carryover_context({"items": rows}, card)
    assert len(section) <= CARRYOVER_PROMPT_CHAR_BUDGET
    rendered = [line for line in section.splitlines() if line.startswith("  - [")]
    assert len(rendered) < CARRYOVER_PROMPT_MAX_ITEMS, (
        "the char budget never bound — it is not a second bound, it is decoration"
    )
    assert "character section budget" in section
    assert _CARRYOVER_OMITTED_MARKER in section


def test_a_spilled_row_is_whole_never_truncated(card):
    """A row cut mid-sentence is a row the model can misread; a half-rendered
    claim is worse than an elided one that is counted and named."""
    rows = []
    for n in range(40):
        row = _row(n)
        row["title"] = f"Investigate {n} " + ("x" * 1500)
        rows.append(row)
    section = _carryover_context({"items": rows}, card)
    for line in section.splitlines():
        if line.startswith("  - ["):
            assert line.endswith(")"), f"row rendered truncated: {line[-60:]!r}"


def test_spilling_preserves_rank_order(card):
    """Once one row spills every lower-ranked row spills too, or the published
    ordering would not describe what was actually carried."""
    rows = []
    for n in range(40):
        row = _row(n, priority="P1", carry_count=40 - n)
        row["title"] = f"Investigate {n} " + ("x" * 1200)
        rows.append(row)
    section = _carryover_context({"items": rows}, card)
    ids = [line.split("[")[1].split("]")[0] for line in section.splitlines() if line.startswith("  - [")]
    ranked = [r["id"] for r in order_for_prompt(rows)]
    assert ids == ranked[: len(ids)]
