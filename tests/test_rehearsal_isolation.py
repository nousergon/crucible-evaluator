"""alpha-engine-config-I12351: the ReportCard stage under a weekly-SF rehearsal.

``rehearsal-2026-10-16-1``'s ReportCard read live ``signals/``, found the
previous week's instance, and raised ``StaleInputArtifactError``, because its
Task payload named nothing that said "rehearsal". The SF now threads
``execution_name`` into that payload, and the handler isolates on it.

Both directions are pinned here:

- NO MARKER (no ``execution_name``, or a non-rehearsal one): the old
  behaviour, unchanged. A stale input raises; the card lands at the live keys.
- REHEARSAL MARKER: inputs resolve from ``rehearsal/<name>/`` first and live
  second; a stale or missing input is a recorded degradation, not a raise; and
  every write lands under the namespace, never at a live key.

Real boto3 clients against moto, so the botocore hook is exercised for real.
"""

from __future__ import annotations

import json

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from grading import handler as H
from grading.freshness_preflight import StaleInputArtifactError, assert_input_freshness
from grading.rehearsal import (
    RehearsalLiveWriteRefused,
    isolate,
    namespace_prefix,
    rehearsal_name,
    tolerate_stale,
)

BUCKET = "alpha-engine-research"
#: A trading day (Fri), so the handler's trading-day normalization is a no-op.
RUN_DATE = "2026-10-16"
#: The previous week's instance: outside [2026-10-12, 2026-10-16].
STALE_DATE = "2026-10-09"
REHEARSAL = "rehearsal-2026-10-16-1"
NS = namespace_prefix(REHEARSAL)
PROD_ARN = "arn:aws:states:us-east-1:711398986525:execution:ne-weekly-freshness-pipeline:a1b2c3d4"

LIVE_WRITE_KEYS = (
    "evaluator/latest/report_card.json",
    f"evaluator/{RUN_DATE}/report_card.json",
    f"evaluator/{RUN_DATE}/self_test.json",
)


@pytest.fixture
def s3(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.delenv("RUN_TOKEN", raising=False)
    monkeypatch.delenv("AE_PIPELINE_ROLE", raising=False)
    with mock_aws():
        # A fresh default session per test so a hook can never leak between
        # tests through boto3's module-level session.
        boto3.setup_default_session()
        client = boto3.client("s3")
        client.create_bucket(Bucket=BUCKET)
        yield client
    boto3.DEFAULT_SESSION = None


def _put(s3, key: str, body) -> None:
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    s3.put_object(Bucket=BUCKET, Key=key, Body=data)


def _seed_live_inputs(s3, *, signals_date: str) -> None:
    """Every freshness-gated input at its LIVE key, fresh except signals."""
    _put(s3, f"backtest/{RUN_DATE}/metrics.json", {"run_date": RUN_DATE, "status": "ok"})
    _put(s3, f"backtest/{RUN_DATE}/e2e_lift.json", {"status": "ok"})
    _put(s3, "predictor/weights/meta/manifest.json", {
        "training_date": RUN_DATE,
        "meta_model_oos_ic_cpcv": {"status": "ok", "n_combos": 4, "mean_ic": 0.1,
                                   "frac_positive": 0.75, "ics": [0.1, 0.1, 0.1, 0.1]},
    })
    _put(s3, f"signals/{signals_date}/signals.json", {"market_regime": "neutral"})
    _put(s3, "trades/eod_pnl.csv", (
        "date,portfolio_nav,daily_return_pct,spy_return_pct,daily_alpha_pct,positions_snapshot,created_at\n"
        f"{RUN_DATE},1000000,0.1,0.05,0.05,{{}},x\n"
    ).encode())


def _seed_rehearsal_signals(s3) -> None:
    """What the rehearsal's own SignalsEnvelope wrote (crucible-research
    data/rehearsal.py re-keys it under the same namespace)."""
    _put(s3, f"{NS}signals/{RUN_DATE}/signals.json", {"market_regime": "neutral"})


def _keys(s3) -> set[str]:
    resp = s3.list_objects_v2(Bucket=BUCKET)
    return {o["Key"] for o in resp.get("Contents", [])}


def _signals_check(provenance: dict) -> dict:
    return next(c for c in provenance["checks"] if c["artifact_id"] == "research_signals")


# ── the marker ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "event,expected",
    [
        ({"execution_name": REHEARSAL}, REHEARSAL),
        ({"execution_arn": f"{PROD_ARN.rsplit(':', 1)[0]}:{REHEARSAL}"}, REHEARSAL),
        ({"execution_name": "a1b2c3d4-weekly"}, None),
        ({"execution_name": "watch-rerun-2026-10-17-1"}, None),
        ({"execution_arn": PROD_ARN}, None),
        ({"date": RUN_DATE, "dry_run": False, "snapshot": True}, None),
        ({}, None),
    ],
)
def test_rehearsal_name(event, expected):
    assert rehearsal_name(event, {}) == expected


def test_the_namespace_matches_the_sf_and_the_iam_grant():
    # nousergon-data ApplyRehearsalNamespace: States.Format('rehearsal/{}/', $$.Execution.Name);
    # nous-ergon-ops grants rehearsal/rehearsal-*/* to this role.
    assert NS == f"rehearsal/{REHEARSAL}/"


def test_tolerate_stale_is_false_outside_a_rehearsal():
    assert tolerate_stale("stale", where="x") is False


# ── the freshness preflight ────────────────────────────────────────────────


class TestPreflightWithoutTheMarker:
    """The old behaviour, unchanged: no context, so every refusal raises."""

    def test_stale_live_signals_still_raise(self, s3):
        _seed_live_inputs(s3, signals_date=STALE_DATE)
        with pytest.raises(StaleInputArtifactError, match="research_signals is stale"):
            assert_input_freshness(BUCKET, RUN_DATE, s3_client=s3)

    @pytest.mark.parametrize("event", [{}, {"execution_name": "a1b2c3d4"}, {"execution_arn": PROD_ARN}])
    def test_a_non_rehearsal_execution_still_raises(self, s3, event):
        _seed_live_inputs(s3, signals_date=STALE_DATE)
        _seed_rehearsal_signals(s3)  # present, and must be IGNORED by production
        with isolate(event, stage="ReportCard") as ctx:
            assert ctx is None
            with pytest.raises(StaleInputArtifactError, match="research_signals is stale"):
                assert_input_freshness(BUCKET, RUN_DATE, s3_client=boto3.client("s3"))

    def test_fresh_live_inputs_carry_no_rehearsal_provenance(self, s3):
        _seed_live_inputs(s3, signals_date=RUN_DATE)
        prov = assert_input_freshness(BUCKET, RUN_DATE, s3_client=s3)
        assert "rehearsal" not in prov and "degraded" not in prov
        assert {c["status"] for c in prov["checks"]} == {"ok"}


class TestPreflightUnderARehearsal:
    def test_reads_the_rehearsals_own_signals_first(self, s3):
        """The rehearsal-2026-10-16-1 case: live signals are last week's, the
        rehearsal's SignalsEnvelope wrote this week's under its namespace."""
        _seed_live_inputs(s3, signals_date=STALE_DATE)
        _seed_rehearsal_signals(s3)
        with isolate({"execution_name": REHEARSAL}, stage="ReportCard") as ctx:
            prov = assert_input_freshness(BUCKET, RUN_DATE, s3_client=boto3.client("s3"))
        sig = _signals_check(prov)
        assert sig["status"] == "ok" and sig["content_date"] == RUN_DATE
        assert prov["rehearsal"] == REHEARSAL and prov["degraded"] == 0
        assert ctx.degradations == []

    def test_falls_back_to_live_for_inputs_the_rehearsal_did_not_write(self, s3):
        _seed_live_inputs(s3, signals_date=RUN_DATE)
        with isolate({"execution_name": REHEARSAL}, stage="ReportCard"):
            prov = assert_input_freshness(BUCKET, RUN_DATE, s3_client=boto3.client("s3"))
        assert {c["status"] for c in prov["checks"]} == {"ok"}
        assert prov["degraded"] == 0

    def test_staleness_is_a_recorded_degradation_not_a_raise(self, s3):
        _seed_live_inputs(s3, signals_date=STALE_DATE)
        with isolate({"execution_name": REHEARSAL}, stage="ReportCard") as ctx:
            prov = assert_input_freshness(BUCKET, RUN_DATE, s3_client=boto3.client("s3"))
        sig = _signals_check(prov)
        assert sig["status"] == "DEGRADED_REHEARSAL"
        assert "research_signals is stale" in sig["reason"]
        assert prov["degraded"] == 1
        assert [d["where"] for d in ctx.degradations] == ["freshness_preflight:research_signals"]
        assert ctx.summary()["degraded"] is True

    def test_a_read_that_cannot_be_performed_still_raises(self, s3, monkeypatch):
        """Absent and unreadable must not collapse, on a rehearsal as anywhere."""
        _seed_live_inputs(s3, signals_date=RUN_DATE)
        with isolate({"execution_name": REHEARSAL}, stage="ReportCard"):
            client = boto3.client("s3")

            def _denied(**_kw):
                raise ClientError({"Error": {"Code": "AccessDenied", "Message": "denied"}}, "HeadObject")

            monkeypatch.setattr(client, "head_object", _denied)
            with pytest.raises(ClientError):
                assert_input_freshness(BUCKET, RUN_DATE, s3_client=client)


def test_a_rehearsal_refuses_an_s3_write_it_cannot_rekey(s3):
    with isolate({"execution_name": REHEARSAL}, stage="ReportCard"):
        with pytest.raises(RehearsalLiveWriteRefused):
            boto3.client("s3").put_bucket_tagging(
                Bucket=BUCKET, Tagging={"TagSet": [{"Key": "k", "Value": "v"}]},
            )


# ── the handler, end to end ────────────────────────────────────────────────


class TestHandlerWithoutTheMarker:
    def test_stale_input_raises_and_writes_nothing(self, s3):
        _seed_live_inputs(s3, signals_date=STALE_DATE)
        before = _keys(s3)
        with pytest.raises(StaleInputArtifactError):
            H.handler({"date": RUN_DATE, "bucket": BUCKET, "snapshot": True})
        assert not {k for k in _keys(s3) - before if k.startswith("evaluator/")}

    def test_fresh_inputs_write_the_live_card_and_no_rehearsal_block(self, s3):
        _seed_live_inputs(s3, signals_date=RUN_DATE)
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET, "snapshot": True})
        assert "rehearsal" not in out
        assert out["latest_key"] == "evaluator/latest/report_card.json"
        keys = _keys(s3)
        assert set(LIVE_WRITE_KEYS) <= keys
        assert not any(k.startswith("rehearsal/") for k in keys)

    def test_a_production_execution_name_is_the_old_path(self, s3):
        _seed_live_inputs(s3, signals_date=RUN_DATE)
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET, "snapshot": True,
                         "execution_name": "a1b2c3d4-weekly"})
        assert "rehearsal" not in out
        assert set(LIVE_WRITE_KEYS) <= _keys(s3)
        assert not any(k.startswith("rehearsal/") for k in _keys(s3))


class TestHandlerUnderARehearsal:
    def test_reads_its_own_signals_and_writes_only_under_the_namespace(self, s3):
        _seed_live_inputs(s3, signals_date=STALE_DATE)
        _seed_rehearsal_signals(s3)
        _put(s3, "evaluator/latest/report_card.json", {"production": True})
        before = _keys(s3)

        out = H.handler({"date": RUN_DATE, "bucket": BUCKET, "snapshot": True,
                         "execution_name": REHEARSAL})

        assert out["status"] == "ok"
        assert out["rehearsal"]["namespace"] == NS
        assert out["rehearsal"]["degraded"] is False
        new = _keys(s3) - before
        assert new, "the rehearsal wrote nothing"
        assert all(k.startswith(NS) for k in new), sorted(new)
        for key in LIVE_WRITE_KEYS:
            assert f"{NS}{key}" in new
        # The production card is untouched.
        body = s3.get_object(Bucket=BUCKET, Key="evaluator/latest/report_card.json")["Body"].read()
        assert json.loads(body) == {"production": True}
        card = json.loads(s3.get_object(Bucket=BUCKET, Key=f"{NS}evaluator/latest/report_card.json")["Body"].read())
        sig = _signals_check(card["_provenance"]["freshness_preflight"])
        assert sig["status"] == "ok" and sig["content_date"] == RUN_DATE

    def test_a_stale_input_degrades_the_stage_instead_of_failing_it(self, s3):
        """rehearsal-2026-10-16-1 exactly: only last week's live signals exist."""
        _seed_live_inputs(s3, signals_date=STALE_DATE)
        before = _keys(s3)

        out = H.handler({"date": RUN_DATE, "bucket": BUCKET, "snapshot": True,
                         "execution_name": REHEARSAL})

        assert out["status"] == "ok"
        assert out["rehearsal"]["degraded"] is True
        refusals = [d["refusal"] for d in out["rehearsal"]["degradations"]]
        assert refusals and all("research_signals is stale" in r for r in refusals)
        new = _keys(s3) - before
        assert all(k.startswith(NS) for k in new), sorted(new)
        assert "evaluator/latest/report_card.json" not in _keys(s3)

    def test_the_hook_is_inert_after_the_rehearsal_invocation(self, s3):
        """A warm container must not carry one rehearsal's re-keying into the
        next (production) invocation."""
        _seed_live_inputs(s3, signals_date=RUN_DATE)
        H.handler({"date": RUN_DATE, "bucket": BUCKET, "snapshot": True,
                   "execution_name": REHEARSAL})
        out = H.handler({"date": RUN_DATE, "bucket": BUCKET, "snapshot": True})
        assert "rehearsal" not in out
        assert set(LIVE_WRITE_KEYS) <= _keys(s3)
