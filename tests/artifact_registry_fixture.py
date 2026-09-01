"""tests/artifact_registry_fixture.py — the declared-registry test double.

`grading/freshness_preflight.py` reads its predicates from the published
ARTIFACT_REGISTRY.yaml mirror at run time (`alpha-engine-config-I9731`), so
every test that builds a Report Card needs that document to exist. One copy of
it lives here rather than in each test module, and `tests/conftest.py` installs
it for the whole suite.

`tests/test_freshness_preflight.py` opts OUT (`pytest.mark.real_artifact_registry`)
and seeds the mirror into its own moto bucket instead — the S3 read path, the
unreadable-registry failures and the dropped-row failures are that module's
actual subject, and a suite-wide double would hide all three.
"""

from __future__ import annotations

# Mirror of the ARTIFACT_REGISTRY.yaml rows this gate depends on, as the
# published S3 mirror serves them (`alpha-engine-config-I9731`). Verbatim from
# alpha-engine-config/private-docs/ARTIFACT_REGISTRY.yaml on 2026-09-01, for
# the fields this module reads. A TEST FIXTURE, not a second source of truth:
# production reads the live mirror, `load_specs` raises when a row is absent,
# and alpha-engine-config's own validator fails in the OWNING repo if any id
# below is renamed or dropped.
REGISTRY_FIXTURE = """
artifacts:
  - artifact_id: backtest_metrics
    s3_bucket: alpha-engine-research
    s3_key_template: "backtest/{trading_day}/metrics.json"
    cadence: saturday_sf
    sla_minutes_after_cron: 360
    severity: warning
    owner_repo: alpha-engine-backtester
    created_at: 2026-05-27
  - artifact_id: backtest_e2e_lift
    s3_bucket: alpha-engine-research
    s3_key_template: "backtest/{trading_day}/e2e_lift.json"
    cadence: saturday_sf
    sla_minutes_after_cron: 360
    severity: warning
    owner_repo: alpha-engine-backtester
    created_at: 2026-08-21
  - artifact_id: predictor_meta_weights_manifest
    s3_bucket: alpha-engine-research
    s3_key_template: "predictor/weights/meta/manifest.json"
    cadence: event_driven
    liveness_via: model_zoo_leaderboard_latest
    sla_minutes_after_cron: 300
    severity: critical
    owner_repo: alpha-engine-predictor
    created_at: 2026-06-01
  - artifact_id: research_signals
    s3_bucket: alpha-engine-research
    s3_key_template: "signals/{trading_day}/signals.json"
    cadence: saturday_sf
    sla_minutes_after_cron: 180
    severity: critical
    owner_repo: alpha-engine-research
    created_at: 2026-05-27
  - artifact_id: eod_reconcile_pnl
    s3_bucket: alpha-engine-research
    s3_key_template: "trades/eod_pnl.csv"
    cadence: eod_sf
    sla_minutes_after_cron: 60
    severity: critical
    owner_repo: alpha-engine
    created_at: 2026-05-27
  - artifact_id: model_zoo_leaderboard_latest
    s3_bucket: alpha-engine-research
    s3_key_template: "predictor/model_zoo/leaderboard/latest.json"
    cadence: saturday_sf
    sla_minutes_after_cron: 300
    severity: critical
    owner_repo: alpha-engine-predictor
    created_at: 2026-06-01
"""




def registry_document() -> dict:
    """The fixture parsed, as ``krepis.stage_coverage.load_registry`` returns it."""
    import yaml

    return yaml.safe_load(REGISTRY_FIXTURE)
