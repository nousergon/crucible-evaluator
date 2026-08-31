"""The raw run_scope artifact and the normalized block collide on a key —
alpha-engine-config-I9001.

`grading.attestation.contamination_scope` decided whether its input was
already normalized with ``"graded_stages" in block``. Both shapes carry that
key, with different meanings:

* the RAW artifact (written by ``nousergon-data``'s ``weekly-run-scope``
  Lambda) carries the stage FAMILIES it graded — measured on the live object
  ``s3://alpha-engine-research/backtest/2026-08-28/run_scope.json``:
  ``['EvalJudge', 'LibPinDriftCheck', 'PostEval', 'SaturdayHealthCheck']``;
* the NORMALIZED block carries every stage whose disposition is in
  ``GRADED_DISPOSITIONS``.

So in production the discriminator matched the raw body, normalization never
ran, the derived ``disabled_stages`` key was never produced, and
``contamination_scope`` read ``None`` for it — resolving ``in_scope=True`` on
every run. Contamination went ``UNKNOWN``, the combined verdict went with it,
and the phase-2 exit clause that reads
``director/latest/action_plan.json::attestation.verdict`` could not be
satisfied. The mechanism was dead for its whole deployed life.

**Why the existing suite did not catch it.** ``test_contamination_scope.py``'s
``_scope()`` helper builds a raw body of ``{schema, run_date, stages}`` — no
``graded_stages``. Every test therefore took the normalize branch that
production never took. These tests use the REAL producer's top-level key set
instead, so a fixture that drifts from the artifact fails here.
"""
from __future__ import annotations

import grading.attestation as att
from grading.attestation import (
    CONTAMINATION_PRODUCER_STAGES,
    NOT_IN_SCOPE,
    contamination_scope,
)
from grading.run_scope import is_normalized, read_run_scope

#: Top-level keys of the live artifact, measured 2026-08-31 from
#: s3://alpha-engine-research/backtest/2026-08-28/run_scope.json.
LIVE_TOP_LEVEL_KEYS = frozenset({
    "calendar_run_date", "counts", "execution_arn", "graded_stages",
    "run_date", "schema", "schema_version", "stages", "state_machine_arn",
    "statement",
})


def _row(disposition: str, disabled_by: str | None = None) -> dict:
    row = {"disposition": disposition}
    if disabled_by:
        row["disabled_by"] = disabled_by
    return row


def _production_shape(stages: dict) -> dict:
    """A raw artifact carrying EVERY top-level key the live producer emits."""
    return {
        "schema": "run_scope-1.0.0",
        "schema_version": 1,
        "run_date": "2026-08-28",
        "calendar_run_date": "2026-08-29",
        "execution_arn": "arn:aws:states:us-east-1:711398986525:execution:x:y",
        "state_machine_arn": "arn:aws:states:us-east-1:711398986525:stateMachine:x",
        # The colliding key. Raw meaning: graded stage FAMILIES.
        "graded_stages": ["EvalJudge", "LibPinDriftCheck", "PostEval",
                          "SaturdayHealthCheck"],
        "counts": {"DISABLED": 1},
        "statement": "derived from the execution history",
        "stages": stages,
    }


def test_the_production_shape_carries_the_colliding_key():
    """Pins the premise. If the producer ever stops emitting `graded_stages`
    this test fails and the tests below stop being a regression guard."""
    assert "graded_stages" in LIVE_TOP_LEVEL_KEYS
    assert LIVE_TOP_LEVEL_KEYS <= set(_production_shape({}))


def test_raw_production_shape_is_not_mistaken_for_normalized():
    """The regression. Before the fix `is_normalized`'s job was done inline by
    `"graded_stages" in block`, which is True here — the raw body was used as
    a normalized one."""
    raw = _production_shape({"Parity": _row("DISABLED", "skip_parity")})
    assert "graded_stages" in raw          # the discriminator that misfired
    assert is_normalized(raw) is False     # ...and no longer decides anything


def test_parity_disabled_is_out_of_scope_from_the_RAW_production_artifact():
    """The defect, end to end. This is the exact input the card receives from
    S3 and from the ReportCard Task payload; before the fix it resolved
    `in_scope=True` and contamination went UNKNOWN on every single run."""
    raw = _production_shape({
        **{s: _row("ENABLED_COMPLETED") for s in CONTAMINATION_PRODUCER_STAGES},
        "Parity": _row("DISABLED", "skip_parity"),
    })
    scope = contamination_scope(raw)
    assert scope["in_scope"] is False
    assert scope["disabled_stages"] == ["Parity"]
    assert scope["disabled_by"] == ["skip_parity"]


def test_dispatched_parity_from_the_raw_shape_stays_in_scope():
    """The opposite polarity: normalization must not manufacture an exclusion.
    A run that DID dispatch parity keeps the half in scope, so a genuinely
    absent pit_parity.json still withholds the guarantee."""
    raw = _production_shape(
        {s: _row("ENABLED_COMPLETED") for s in CONTAMINATION_PRODUCER_STAGES}
    )
    assert contamination_scope(raw)["in_scope"] is True


def test_not_reached_parity_is_never_read_as_disabled():
    """`NOT_REACHED` is an absence of evidence, not a decision — it must keep
    the half IN scope. Normalizing the raw shape must not blur the two."""
    raw = _production_shape({
        **{s: _row("ENABLED_COMPLETED") for s in CONTAMINATION_PRODUCER_STAGES},
        "Parity": _row("NOT_REACHED"),
    })
    assert contamination_scope(raw)["in_scope"] is True


def test_read_run_scope_is_idempotent():
    """Normalizing twice is normalizing once. This is what lets every caller
    drop its own shape guess."""
    raw = _production_shape({"Parity": _row("DISABLED", "skip_parity")})
    once = read_run_scope(raw)
    assert is_normalized(once) is True
    assert read_run_scope(once) is once
    assert read_run_scope(read_run_scope(raw)) == once


def test_out_of_scope_contamination_is_still_never_a_pass():
    """Guards the lazy version of this fix: excluding the half must not grade
    it. `NOT_IN_SCOPE` stays outside the pass vocabulary."""
    assert att.verdict_is_pass(NOT_IN_SCOPE) is False
    assert NOT_IN_SCOPE not in att._VALID_VERDICTS


def test_the_reason_names_only_the_flags_that_disabled_the_parity_family():
    """The run-wide `disabled_by` union is not an attribution. Measured on the
    first run this mechanism actually worked: the reason read "Parity ... took
    the skip branch, disabled by skip_backtester, skip_challenger_shadow,
    skip_counterfactual" — every flag on the run, printed under a sentence
    about the look-ahead check."""
    raw = _production_shape({
        **{s: _row("ENABLED_COMPLETED") for s in CONTAMINATION_PRODUCER_STAGES},
        "Parity": _row("DISABLED", "skip_parity"),
        # Unrelated stages, disabled by unrelated flags, on the same run.
        "Backtester": _row("DISABLED", "skip_backtester"),
        "ChallengerShadow": _row("DISABLED", "skip_challenger_shadow"),
    })
    scope = contamination_scope(raw)
    assert scope["in_scope"] is False
    assert scope["disabled_by"] == ["skip_parity"]
    assert "skip_backtester" not in scope["reason"]
    assert "skip_challenger_shadow" not in scope["reason"]
