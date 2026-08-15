"""Tests for grading/freshness_preflight.py — the hard input-freshness gate
(alpha-engine-config#3058, Brian ruling 2026-07-20).

Acceptance criteria (closes-when): a weekly Evaluator run against a
research-free-derived artifact whose content is older than the run's week
HARD-FAILS at preflight with a named-artifact error, AND the normal
fresh-input path passes unchanged. Covers: stale → raise; fresh → pass;
missing → raise (not skip).

`alpha-engine-config-I7392` adds the dry path (the Friday-PM shell run) at the
bottom of this module. Its tests are written to hold the line the easy version
of that change would have crossed: the real run must still hard-fail
identically, and a dry run must still fail loud on a read it could not PERFORM,
as opposed to an artifact that is simply absent.
"""

from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws

from grading.freshness_preflight import (
    MissingInputArtifactError,
    StaleInputArtifactError,
    assert_input_freshness,
)

BUCKET = "alpha-engine-research"
RUN_DATE = "2026-07-18"  # a Saturday, mirrors the incident's own run_date


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _seed_metrics(s3, run_date=RUN_DATE):
    s3.put_object(
        Bucket=BUCKET, Key=f"backtest/{run_date}/metrics.json",
        Body=json.dumps({"run_date": run_date, "status": "ok"}).encode(),
    )


def _seed_e2e_lift(s3, date):
    s3.put_object(
        Bucket=BUCKET, Key=f"backtest/{date}/e2e_lift.json",
        Body=json.dumps({"status": "ok"}).encode(),
    )


def _seed_manifest(s3, training_date):
    s3.put_object(
        Bucket=BUCKET, Key="predictor/weights/meta/manifest.json",
        Body=json.dumps({
            "training_date": training_date,
            "meta_model_oos_ic_cpcv": {"status": "ok", "mean_ic": 0.1},
        }).encode(),
    )


def _seed_signals(s3, date):
    s3.put_object(
        Bucket=BUCKET, Key=f"signals/{date}/signals.json",
        Body=json.dumps({"market_regime": "neutral"}).encode(),
    )


def _seed_eod_pnl(s3, dates):
    rows = ["date,portfolio_nav,daily_return_pct,spy_return_pct,daily_alpha_pct"]
    nav = 1_000_000.0
    for d in dates:
        rows.append(f"{d},{nav:.2f},0.1,0.05,0.05")
    s3.put_object(Bucket=BUCKET, Key="trades/eod_pnl.csv", Body="\n".join(rows).encode())


def _seed_all_fresh(s3, run_date=RUN_DATE):
    """Every declared input, dated exactly run_date — the trivially-fresh
    baseline. Individual tests below override ONE input to exercise a
    specific stale/missing failure mode."""
    _seed_metrics(s3, run_date)
    _seed_e2e_lift(s3, run_date)
    _seed_manifest(s3, run_date)
    _seed_signals(s3, run_date)
    _seed_eod_pnl(s3, [run_date])


class TestFreshInputsPass:
    def test_all_fresh_passes_and_returns_provenance(self, s3):
        _seed_all_fresh(s3)
        result = assert_input_freshness(BUCKET, RUN_DATE, s3_client=s3)
        assert result["run_date"] == RUN_DATE
        ids = {c["artifact_id"] for c in result["checks"]}
        assert ids == {
            "metrics_json", "e2e_lift_json", "predictor_meta_weights_manifest",
            "research_signals", "eod_reconcile_pnl",
        }

    def test_e2e_lift_earlier_in_same_week_passes(self, s3):
        # The Saturday run's own week started Monday 2026-07-13; an
        # e2e_lift.json dated mid-week (a Wednesday off-cycle write) is still
        # IN this week — must not false-alarm on an exact-day mismatch.
        _seed_metrics(s3)
        _seed_e2e_lift(s3, "2026-07-15")
        _seed_manifest(s3, RUN_DATE)
        _seed_signals(s3, RUN_DATE)
        _seed_eod_pnl(s3, [RUN_DATE])
        result = assert_input_freshness(BUCKET, RUN_DATE, s3_client=s3)
        e2e = next(c for c in result["checks"] if c["artifact_id"] == "e2e_lift_json")
        assert e2e["content_date"] == "2026-07-15"

    def test_eod_pnl_one_trading_day_behind_passes(self, s3):
        # eod_sf/daily cadence tolerates up to 1 NYSE trading-day lag
        # (T+1 publish latency) — RUN_DATE is a Saturday, so Friday 07-17
        # is the last trading day and is exactly fresh (0 sessions behind).
        _seed_metrics(s3)
        _seed_e2e_lift(s3, RUN_DATE)
        _seed_manifest(s3, RUN_DATE)
        _seed_signals(s3, RUN_DATE)
        _seed_eod_pnl(s3, ["2026-07-17"])
        result = assert_input_freshness(BUCKET, RUN_DATE, s3_client=s3)
        eod = next(c for c in result["checks"] if c["artifact_id"] == "eod_reconcile_pnl")
        assert eod["content_date"] == "2026-07-17"


class TestStaleInputsRaise:
    """The 2026-07-18 incident class: a silently no-op'd producer leaves an
    artifact carrying last week's (or older) cohort."""

    def test_stale_e2e_lift_raises_named_error(self, s3):
        # Mirrors the actual incident: e2e_lift.json's freshest resolvable
        # instance is over a week stale (prior Saturday, outside this week).
        _seed_metrics(s3)
        _seed_e2e_lift(s3, "2026-07-10")  # 8 days before run_date, prior week
        _seed_manifest(s3, RUN_DATE)
        _seed_signals(s3, RUN_DATE)
        _seed_eod_pnl(s3, [RUN_DATE])
        with pytest.raises(StaleInputArtifactError, match="e2e_lift.json is stale"):
            assert_input_freshness(BUCKET, RUN_DATE, s3_client=s3)

    def test_stale_metrics_json_raises_named_error(self, s3):
        # metrics.json read directly at backtest/{run_date}/ (no windowing) —
        # a stale run_date FIELD inside an otherwise-present file must still
        # raise (the content date, not just presence, is what's asserted).
        s3.put_object(
            Bucket=BUCKET, Key=f"backtest/{RUN_DATE}/metrics.json",
            Body=json.dumps({"run_date": "2026-07-10", "status": "ok"}).encode(),
        )
        _seed_e2e_lift(s3, RUN_DATE)
        _seed_manifest(s3, RUN_DATE)
        _seed_signals(s3, RUN_DATE)
        _seed_eod_pnl(s3, [RUN_DATE])
        with pytest.raises(StaleInputArtifactError, match="metrics.json is stale"):
            assert_input_freshness(BUCKET, RUN_DATE, s3_client=s3)

    def test_stale_predictor_manifest_raises_named_error(self, s3):
        _seed_metrics(s3)
        _seed_e2e_lift(s3, RUN_DATE)
        _seed_manifest(s3, "2026-06-20")  # weeks stale
        _seed_signals(s3, RUN_DATE)
        _seed_eod_pnl(s3, [RUN_DATE])
        with pytest.raises(StaleInputArtifactError, match="predictor manifest is stale"):
            assert_input_freshness(BUCKET, RUN_DATE, s3_client=s3)

    def test_stale_signals_raises_named_error(self, s3):
        _seed_metrics(s3)
        _seed_e2e_lift(s3, RUN_DATE)
        _seed_manifest(s3, RUN_DATE)
        _seed_signals(s3, "2026-07-08")  # prior week, within the 10-day walk-back window
        _seed_eod_pnl(s3, [RUN_DATE])
        with pytest.raises(StaleInputArtifactError, match="signals.json is stale"):
            assert_input_freshness(BUCKET, RUN_DATE, s3_client=s3)

    def test_stale_eod_pnl_raises_named_error(self, s3):
        _seed_metrics(s3)
        _seed_e2e_lift(s3, RUN_DATE)
        _seed_manifest(s3, RUN_DATE)
        _seed_signals(s3, RUN_DATE)
        _seed_eod_pnl(s3, ["2026-07-01"])  # over a week of trading days stale
        with pytest.raises(StaleInputArtifactError, match="eod_pnl.csv is stale"):
            assert_input_freshness(BUCKET, RUN_DATE, s3_client=s3)


class TestMissingInputsRaiseNotSkip:
    """A declared input's absence must raise — never silently skip/degrade.
    Distinct from the tiles' own graceful-N/A posture for OPTIONAL artifacts
    (veto_value.json etc.) that this preflight deliberately does not gate."""

    def test_missing_metrics_json_raises(self, s3):
        _seed_e2e_lift(s3, RUN_DATE)
        _seed_manifest(s3, RUN_DATE)
        _seed_signals(s3, RUN_DATE)
        _seed_eod_pnl(s3, [RUN_DATE])
        with pytest.raises(MissingInputArtifactError, match="metrics.json"):
            assert_input_freshness(BUCKET, RUN_DATE, s3_client=s3)

    def test_missing_e2e_lift_raises(self, s3):
        _seed_metrics(s3)
        _seed_manifest(s3, RUN_DATE)
        _seed_signals(s3, RUN_DATE)
        _seed_eod_pnl(s3, [RUN_DATE])
        with pytest.raises(MissingInputArtifactError, match="e2e_lift.json"):
            assert_input_freshness(BUCKET, RUN_DATE, s3_client=s3)

    def test_missing_predictor_manifest_raises(self, s3):
        _seed_metrics(s3)
        _seed_e2e_lift(s3, RUN_DATE)
        _seed_signals(s3, RUN_DATE)
        _seed_eod_pnl(s3, [RUN_DATE])
        with pytest.raises(MissingInputArtifactError, match="predictor manifest"):
            assert_input_freshness(BUCKET, RUN_DATE, s3_client=s3)

    def test_missing_signals_raises(self, s3):
        _seed_metrics(s3)
        _seed_e2e_lift(s3, RUN_DATE)
        _seed_manifest(s3, RUN_DATE)
        _seed_eod_pnl(s3, [RUN_DATE])
        with pytest.raises(MissingInputArtifactError, match="signals.json"):
            assert_input_freshness(BUCKET, RUN_DATE, s3_client=s3)

    def test_missing_eod_pnl_raises(self, s3):
        _seed_metrics(s3)
        _seed_e2e_lift(s3, RUN_DATE)
        _seed_manifest(s3, RUN_DATE)
        _seed_signals(s3, RUN_DATE)
        with pytest.raises(MissingInputArtifactError, match="eod_pnl.csv"):
            assert_input_freshness(BUCKET, RUN_DATE, s3_client=s3)

    def test_empty_bucket_raises_on_first_check_not_silent(self, s3):
        with pytest.raises(MissingInputArtifactError):
            assert_input_freshness(BUCKET, RUN_DATE, s3_client=s3)

    def test_manifest_present_but_no_date_field_raises(self, s3):
        # A manifest that exists but carries none of training_date/run_date/
        # date is indistinguishable from "can't verify freshness" — must
        # raise, not silently pass an unverifiable artifact.
        _seed_metrics(s3)
        _seed_e2e_lift(s3, RUN_DATE)
        s3.put_object(
            Bucket=BUCKET, Key="predictor/weights/meta/manifest.json",
            Body=json.dumps({"meta_model_oos_ic_cpcv": {"status": "ok"}}).encode(),
        )
        _seed_signals(s3, RUN_DATE)
        _seed_eod_pnl(s3, [RUN_DATE])
        with pytest.raises(MissingInputArtifactError, match="predictor manifest"):
            assert_input_freshness(BUCKET, RUN_DATE, s3_client=s3)


class TestOtherS3ErrorsPropagate:
    def test_non_404_client_error_raises_unchanged(self, s3, monkeypatch):
        from unittest.mock import MagicMock

        from botocore.exceptions import ClientError

        broken = MagicMock()
        broken.get_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "denied"}}, "GetObject"
        )
        with pytest.raises(ClientError):
            assert_input_freshness(BUCKET, RUN_DATE, s3_client=broken)


class TestBadRunDate:
    def test_non_iso_run_date_raises(self, s3):
        with pytest.raises(MissingInputArtifactError):
            assert_input_freshness(BUCKET, "not-a-date", s3_client=s3)


# ── The dry path (alpha-engine-config-I7392) ───────────────────────────────
#
# Context, because these tests only make sense with it. The weekly SF's
# `ApplyShellRunDefaults` runs every producer with `--preflight-only`, so the
# Friday shell run writes NO artifacts — and this gate then hard-failed on the
# absence the same pipeline had just guaranteed. Measured on execution
# `friday-shell-2026-08-14-validate-final`: the run reached the END of the
# pipeline with every workload stage green, then died here on
# `metrics.json: no artifact at backtest/2026-08-14/`.
#
# The 2026-07-20 ruling this gate exists for is about a REAL run evaluating
# stale data. A dry run evaluates nothing and persists nothing, so the premise
# does not reach it. Everything below exists to keep that carve-out honest.


class TestDryPathIsMeasuredNotSkipped:
    """The dry path must remain evidence. A rehearsal that cannot fail is worse
    than one that always fails — it is green by construction."""

    def test_dry_run_records_unmeasured_instead_of_raising(self, s3):
        """The whole point: nothing seeded, dry_run=True, no exception."""
        prov = assert_input_freshness(BUCKET, RUN_DATE, s3_client=s3, dry_run=True)

        assert prov["dry_run"] is True
        assert prov["measured"] == 0
        assert prov["unmeasured"] == len(prov["checks"])
        assert prov["checks"], "the gate reported no checks at all"
        for check in prov["checks"]:
            assert check["status"] == "UNMEASURED"
            assert check["reason"], (
                f"{check['artifact_id']}: UNMEASURED with no reason — "
                "indistinguishable from a check that did not run"
            )

    def test_the_real_run_still_hard_fails_on_the_same_input(self, s3):
        """The ruling, unchanged. Same bucket, same date, same absence — only
        `dry_run` differs, and the real path must still raise a NAMED-artifact
        error. If this ever passes, the carve-out has eaten the guard."""
        with pytest.raises(MissingInputArtifactError) as exc:
            assert_input_freshness(BUCKET, RUN_DATE, s3_client=s3, dry_run=False)
        assert "metrics.json" in str(exc.value)

    def test_dry_run_defaults_to_false(self, s3):
        """A caller that does not opt in gets the strict gate. The dry path must
        never be reachable by omission."""
        with pytest.raises(MissingInputArtifactError):
            assert_input_freshness(BUCKET, RUN_DATE, s3_client=s3)

    def test_a_fully_seeded_dry_run_measures_everything(self, s3):
        """Both polarities (sf-pipeline-policy 2.3a): a dry run that HAD its
        inputs must not render like one that had none. Without this, `measured`
        could be hardcoded to 0 and every assertion above would still pass."""
        _seed_all_fresh(s3)
        prov = assert_input_freshness(BUCKET, RUN_DATE, s3_client=s3, dry_run=True)

        assert prov["unmeasured"] == 0
        assert prov["measured"] == len(prov["checks"])
        assert all(c["status"] == "ok" for c in prov["checks"])

    def test_dry_run_still_raises_when_the_read_cannot_be_PERFORMED(self, s3, monkeypatch):
        """The load-bearing test.

        An artifact that is ABSENT and a read that is DENIED must not collapse
        to the same verdict. `_get_json_body` returns None on 404 and re-raises
        every other ClientError; only the two domain exceptions are absorbed on
        the dry path. So an AccessDenied still propagates — a dry run against a
        broken IAM grant fails, exactly as it must, because proving S3
        IAM/transport works is most of why the rehearsal exists.
        """
        from botocore.exceptions import ClientError

        def _denied(*_args, **_kwargs):
            raise ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "denied"}}, "GetObject",
            )

        monkeypatch.setattr(s3, "get_object", _denied)

        with pytest.raises(ClientError) as exc:
            assert_input_freshness(BUCKET, RUN_DATE, s3_client=s3, dry_run=True)
        assert exc.value.response["Error"]["Code"] == "AccessDenied"

    def test_dry_run_absorbs_staleness_too_but_reports_it(self, s3):
        """Stale is the other domain failure. On the dry path it is recorded
        with its reason rather than raised — and the reason must actually name
        the staleness, not be an empty string standing in for one."""
        _seed_all_fresh(s3)
        # Overwrite metrics.json with content from outside the run week.
        _seed_metrics(s3, run_date=RUN_DATE)
        s3.put_object(
            Bucket=BUCKET, Key=f"backtest/{RUN_DATE}/metrics.json",
            Body=json.dumps({"run_date": "2026-06-01", "status": "ok"}).encode(),
        )

        prov = assert_input_freshness(BUCKET, RUN_DATE, s3_client=s3, dry_run=True)
        stale = [c for c in prov["checks"] if c["status"] == "UNMEASURED"]
        assert stale, "a stale artifact was not reported on the dry path"
        assert any("stale" in c["reason"].lower() for c in stale)

        # ...and the real run still refuses it.
        with pytest.raises(StaleInputArtifactError):
            assert_input_freshness(BUCKET, RUN_DATE, s3_client=s3, dry_run=False)
