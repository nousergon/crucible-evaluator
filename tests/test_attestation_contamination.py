"""The contamination half of the Report Card attestation (config#7199).

The three halves that shipped on 2026-08-12/13 all answer *"did we compute this
correctly?"*. None of them answers *"was the input allowed to see the future?"*.

On 2026-08-07 the check that answers the second question timed out after 2700s,
wrote `{"status": "failed", ...}`, was referenced by nothing in this repo, and
that cycle's report card was written `status: "ok"`, `degraded_staleness: false`,
grade 55.7. Nothing distinguished a non-answer from a pass.

These tests pin the properties that make that irreproducible:
  - a missing / stale / unrecognised contamination verdict is UNKNOWN,
  - a PARTIAL is never a pass,
  - the two claims stay two fields, and
  - an unverified card cannot present itself as `status: "ok"`.
"""

from __future__ import annotations

import io
import json

import pytest

from grading import attestation as att


class _S3:
    """Minimal S3 stand-in. ``objects`` maps key -> (body_bytes | Exception)."""

    def __init__(self, objects: dict):
        self.objects = objects

    def get_object(self, Bucket=None, Key=None):  # noqa: N803 — boto3 casing
        val = self.objects.get(Key)
        if val is None:
            from botocore.exceptions import ClientError

            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject",
            )
        if isinstance(val, Exception):
            raise val
        return {"Body": io.BytesIO(val), "LastModified": _FakeDT()}


class _FakeDT:
    def strftime(self, fmt):  # noqa: D102
        return "2026-08-15T09:00:00Z"


def _report(**kw) -> bytes:
    doc = {
        "schema": "pit_parity-1.0.0",
        "run_date": "2026-08-15",
        "status": "ok",
        "verdict": att.PASS,
        "verdict_reason": "PASS: not distinguishable from zero.",
        "coverage": {"coverage_fraction": 1.0, "budget_stopped": False,
                     "complete": True, "measured": True,
                     "covered_through": "2026-08-14"},
        "materiality": {"material": False},
    }
    doc.update(kw)
    return json.dumps(doc).encode()


_KEY = "backtest/2026-08-15/pit_parity.json"


# ════════════════════════════════════════════════════════════════════════════
# The vocabulary
# ════════════════════════════════════════════════════════════════════════════

def test_partial_is_in_the_vocabulary_and_is_not_a_pass():
    assert att.PARTIAL in att._VALID_VERDICTS
    assert not att.verdict_is_pass(att.PARTIAL)


@pytest.mark.parametrize(
    "verdicts,expected",
    [
        ((att.PASS, att.PASS), att.PASS),
        ((att.PASS, att.PARTIAL), att.PARTIAL),
        ((att.PARTIAL, att.UNKNOWN), att.UNKNOWN),
        ((att.PARTIAL, att.FAIL), att.FAIL),
        ((att.UNKNOWN, att.FAIL), att.FAIL),
        ((att.PASS, "ok"), att.UNKNOWN),
    ],
)
def test_worst_of_ordering(verdicts, expected):
    """FAIL > UNKNOWN > PARTIAL > PASS. PARTIAL above UNKNOWN because it carries
    incomplete evidence rather than none — and an unrecognised string is never
    allowed to be the best of a pair."""
    assert att._worst(*verdicts) == expected


# ════════════════════════════════════════════════════════════════════════════
# The reader
# ════════════════════════════════════════════════════════════════════════════

def test_present_pass_report_reads_pass_with_coverage():
    res = att.read_contamination_verdict(
        "b", "2026-08-15", s3_client=_S3({_KEY: _report()}),
    )
    assert res["verdict"] == att.PASS
    assert res["coverage_fraction"] == 1.0
    assert res["as_of"] == "2026-08-15T09:00:00Z"


def test_absent_report_is_unknown_never_a_pass():
    res = att.read_contamination_verdict("b", "2026-08-15", s3_client=_S3({}))
    assert res["verdict"] == att.UNKNOWN
    assert not att.verdict_is_pass(res["verdict"])


def test_report_with_no_verdict_key_is_unknown():
    """Every pit_parity.json written before config#7199 is this case: a complete,
    plausible-looking contamination report with no verdict field at all."""
    body = json.loads(_report())
    del body["verdict"]
    del body["verdict_reason"]
    res = att.read_contamination_verdict(
        "b", "2026-08-15", s3_client=_S3({_KEY: json.dumps(body).encode()}),
    )
    assert res["verdict"] == att.UNKNOWN
    assert res["reason"]


def test_the_2026_08_07_artifact_shape_reads_unknown():
    """The literal failed artifact from the cycle that motivated this change."""
    body = json.dumps({
        "schema": "pit_parity-1.0.0", "run_date": "2026-08-15",
        "status": "failed", "error_class": "RuntimeError",
        "error_msg": "pit_parity walkforward pass timed out after 2700s: ...",
        "observational": True,
    }).encode()
    res = att.read_contamination_verdict(
        "b", "2026-08-15", s3_client=_S3({_KEY: body}),
    )
    assert res["verdict"] == att.UNKNOWN


def test_a_verdict_from_another_cycle_is_not_inherited():
    res = att.read_contamination_verdict(
        "b", "2026-08-15",
        s3_client=_S3({_KEY: _report(run_date="2026-08-08")}),
    )
    assert res["verdict"] == att.UNKNOWN
    assert "not" in res["reason"].lower()


def test_unparseable_body_is_unknown_and_does_not_raise():
    res = att.read_contamination_verdict(
        "b", "2026-08-15", s3_client=_S3({_KEY: b"{not json"}),
    )
    assert res["verdict"] == att.UNKNOWN


def test_partial_report_surfaces_its_coverage_fraction():
    res = att.read_contamination_verdict(
        "b", "2026-08-15",
        s3_client=_S3({_KEY: _report(
            verdict=att.PARTIAL,
            verdict_reason="PARTIAL: clean over 62.0% of the window.",
            coverage={"coverage_fraction": 0.62, "budget_stopped": True,
                      "complete": False, "measured": True},
        )}),
    )
    assert res["verdict"] == att.PARTIAL
    assert res["coverage_fraction"] == pytest.approx(0.62)
    assert res["budget_stopped"] is True
    assert "62" in res["reason"]


# ════════════════════════════════════════════════════════════════════════════
# The combined block
# ════════════════════════════════════════════════════════════════════════════

def _build(monkeypatch, contamination_body, arithmetic_verdict=att.PASS):
    """Drive build_run_attestation with the three arithmetic halves pinned, so
    the assertion is about the contamination half alone."""
    monkeypatch.setattr(att, "run_evaluator_attestation",
                        lambda: {"verdict": arithmetic_verdict})
    monkeypatch.setattr(att, "read_backtester_attestation",
                        lambda *a, **k: {"verdict": arithmetic_verdict, "as_of": None})
    monkeypatch.setattr(att, "read_evaluator_stage_attestation",
                        lambda *a, **k: {"verdict": arithmetic_verdict, "as_of": None})
    objects = {} if contamination_body is None else {_KEY: contamination_body}
    return att.build_run_attestation("b", "2026-08-15", s3_client=_S3(objects))


def test_arithmetic_and_contamination_are_two_fields(monkeypatch):
    """The load-bearing property of this change. An external reader asks 'are
    the numbers right?' and 'could they have seen the future?' separately; a
    single boolean answers neither."""
    block = _build(monkeypatch, _report())
    assert block["arithmetic_verdict"] == att.PASS
    assert block["contamination_verdict"] == att.PASS
    assert block["verdict"] == att.PASS


def test_clean_arithmetic_with_absent_contamination_is_not_a_pass(monkeypatch):
    """Exactly the 2026-08-07 configuration. Arithmetic fine, contamination
    never answered — the combined verdict must withhold."""
    block = _build(monkeypatch, None)
    assert block["arithmetic_verdict"] == att.PASS
    assert block["contamination_verdict"] == att.UNKNOWN
    assert block["verdict"] == att.UNKNOWN
    assert not att.verdict_is_pass(block["verdict"])
    assert "contamination=UNKNOWN" in block["reason"]


def test_partial_contamination_does_not_hide_a_clean_arithmetic_half(monkeypatch):
    block = _build(monkeypatch, _report(
        verdict=att.PARTIAL, verdict_reason="PARTIAL: 62.0% covered.",
        coverage={"coverage_fraction": 0.62, "budget_stopped": True,
                  "complete": False, "measured": True},
    ))
    assert block["arithmetic_verdict"] == att.PASS
    assert block["contamination_verdict"] == att.PARTIAL
    assert block["verdict"] == att.PARTIAL
    assert block["contamination_coverage_fraction"] == pytest.approx(0.62)


def test_contamination_fail_dominates(monkeypatch):
    block = _build(monkeypatch, _report(
        verdict=att.FAIL, verdict_reason="MATERIAL contamination detected.",
        materiality={"material": True},
    ))
    assert block["verdict"] == att.FAIL


def test_arithmetic_fail_survives_a_clean_contamination_half(monkeypatch):
    block = _build(monkeypatch, _report(), arithmetic_verdict=att.FAIL)
    assert block["arithmetic_verdict"] == att.FAIL
    assert block["contamination_verdict"] == att.PASS
    assert block["verdict"] == att.FAIL


def test_nothing_here_raises_on_a_broken_s3(monkeypatch):
    """CONTRACT: a verdict stage that dies must not kill the card."""
    class _Boom:
        def get_object(self, **kw):
            raise RuntimeError("s3 is down")

    monkeypatch.setattr(att, "run_evaluator_attestation", lambda: {"verdict": att.PASS})
    block = att.build_run_attestation("b", "2026-08-15", s3_client=_Boom())
    assert block["verdict"] == att.UNKNOWN


# ════════════════════════════════════════════════════════════════════════════
# The reason SENTENCE, not only the verdict field (config#7199)
# ════════════════════════════════════════════════════════════════════════════
#
# `verdict_reason` is an OPTIONAL producer field. The verdict field being right
# is not sufficient: the reason line is what the Report Card, the Director
# digest and the console render as prose, and a non-PASS described in prose as
# a pass is the "renders measured-X as Y" failure this fleet has shipped before.

@pytest.mark.parametrize("verdict", [att.FAIL, att.PARTIAL, att.UNKNOWN])
def test_a_non_pass_never_reads_as_a_pass_when_the_producer_sent_no_reason(verdict):
    """The failing branch, proved. Drop `verdict_reason` from the artifact and
    the rendered sentence must still say the guarantee is withheld."""
    body = _report(verdict=verdict)
    doc = json.loads(body)
    doc.pop("verdict_reason")
    result = att.read_contamination_verdict(
        "b", "2026-08-15", s3_client=_S3({_KEY: json.dumps(doc).encode()}),
    )
    assert result["verdict"] == verdict
    reason = result["reason"]
    # The sentence names THIS verdict and says the guarantee is withheld. The
    # pre-fix behaviour rendered the literal "contamination verdict PASS." here.
    assert verdict in reason, reason
    assert "verdict PASS" not in reason, reason
    assert "not a pass" in reason.lower() or "NOT established" in reason, reason
    assert not att.verdict_is_pass(result["verdict"])


def test_a_pass_with_no_producer_reason_still_reads_as_a_pass():
    """The other polarity — the guard above must not turn every verdict into a
    withholding sentence, or it would be a check that cannot pass."""
    doc = json.loads(_report())
    doc.pop("verdict_reason")
    result = att.read_contamination_verdict(
        "b", "2026-08-15", s3_client=_S3({_KEY: json.dumps(doc).encode()}),
    )
    assert result["verdict"] == att.PASS
    assert "PASS" in result["reason"]


def test_a_reasonless_fail_coverage_is_stated_not_implied():
    doc = json.loads(_report(
        verdict=att.FAIL,
        coverage={"coverage_fraction": 0.62, "budget_stopped": True},
        materiality={"material": True},
    ))
    doc.pop("verdict_reason")
    result = att.read_contamination_verdict(
        "b", "2026-08-15", s3_client=_S3({_KEY: json.dumps(doc).encode()}),
    )
    assert "62%" in result["reason"]


def test_a_producer_supplied_reason_is_never_overwritten(monkeypatch):
    block = _build(monkeypatch, _report(
        verdict=att.FAIL, verdict_reason="MATERIAL contamination detected.",
        materiality={"material": True},
    ))
    assert "MATERIAL contamination detected." in block["contamination"]["reason"]


def test_an_explicit_unknown_is_not_described_as_a_malformed_verdict():
    """An artifact saying `verdict: "UNKNOWN"` is HONEST, not malformed. Calling
    it 'not one of [FAIL, PARTIAL, PASS, UNKNOWN]' is a false sentence that
    sends a reader hunting a producer bug that does not exist."""
    doc = json.loads(_report(verdict=att.UNKNOWN))
    doc.pop("verdict_reason")
    result = att.read_contamination_verdict(
        "b", "2026-08-15", s3_client=_S3({_KEY: json.dumps(doc).encode()}),
    )
    assert result["verdict"] == att.UNKNOWN
    assert "not one of" not in result["reason"], result["reason"]
    assert "did not answer" in result["reason"]


def test_a_genuinely_unrecognised_verdict_still_names_the_vocabulary():
    """The other polarity — a producer that starts writing "ok" must still be
    called out, or the guard above would suppress a real contract break."""
    doc = json.loads(_report(verdict="ok"))
    doc.pop("verdict_reason")
    result = att.read_contamination_verdict(
        "b", "2026-08-15", s3_client=_S3({_KEY: json.dumps(doc).encode()}),
    )
    assert result["verdict"] == att.UNKNOWN
    assert "not one of" in result["reason"]
    assert "'ok'" in result["reason"]
