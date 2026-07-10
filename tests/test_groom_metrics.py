"""Tests for grading/tiles/groom.py — groom-pipeline Agent-tile components (config#2151)."""

import json

import boto3
import pytest
from moto import mock_aws

from grading.tiles.agent import build_agent_tile
from grading.tiles.groom import GroomArtifactError, build_groom_components

BUCKET = "alpha-engine-research"
RUN_DATE = "2026-07-10"  # window = 2026-07-04 → 2026-07-10 inclusive

GROOM_NAMES = (
    "groom_completion_rate",
    "groom_wet_per_completion",
    "groom_comment_churn",
    "groom_lost_chunks",
)


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _comp(components, name):
    recs = [c.model_dump(mode="json") for c in components]
    return next(c for c in recs if c["name"] == name)


def _issue(number, disposition, repo="alpha-engine-config"):
    return {"repo": repo, "number": number, "priority": "P2",
            "disposition": disposition, "detail": ""}


def _put_run(s3, date, run_id, *, issues=None, chunk_log=None, run_wet=0.0, body=None):
    """Write one schema_version-6 run artifact (or raw ``body`` bytes)."""
    if body is None:
        body = json.dumps({
            "schema_version": 6,
            "run_kind": "coverage",
            "run_start": f"{date}T19:00:00+00:00",
            "model": "haiku",
            "issue_filter": "P2",
            "stop_reason": "queue drained",
            "total_issues": len(issues or []),
            "engaged": 0,
            "processed": len(issues or []),
            "fresh_skipped": 0,
            "run_wet": run_wet,
            "issues": issues if issues is not None else [],
            "chunk_log": chunk_log if chunk_log is not None else [],
        }).encode()
    s3.put_object(Bucket=BUCKET, Key=f"groom/{date}/{run_id}.json", Body=body)


_OK_CHUNK = "- chunk 1 (4 issues, P2+): agent=ok, touched≈4. all dispositioned"
_MAX_TURNS_CHUNK_A = ("- chunk 1 (9 issues, P1+): agent=ERR, touched≈2. "
                      "Reached maximum number of turns")
_MAX_TURNS_CHUNK_B = ('- chunk 2 (5 issues, P2+): agent=ERR, touched≈0. '
                      '{"terminal_reason": "max_turns"}')


def _put_window_fixture(s3):
    """Five in-window runs + one out-of-window run (2026-07-03, must be excluded).

    In-window totals: 35 dispositions, 4 completions (rate 4/35 ≈ 11.4% → RED),
    4.0M WET over the 4 non-null-WET runs (1.0M per completion), 1 churning
    issue (#3: 3× commented, never completed), 2 max_turns chunk failures.
    """
    _put_run(s3, "2026-07-04", "190001", run_wet=500_000.0, chunk_log=[_OK_CHUNK],
             issues=[_issue(40, "closed")])
    _put_run(s3, "2026-07-05", "190002", run_wet=1_000_000.0, chunk_log=[_OK_CHUNK],
             issues=[_issue(1, "closed"), _issue(2, "pr_opened"),
                     _issue(3, "commented"), _issue(4, "untouched")])
    _put_run(s3, "2026-07-08", "190003", run_wet=500_000.0,
             chunk_log=[_MAX_TURNS_CHUNK_A, _MAX_TURNS_CHUNK_B, _OK_CHUNK],
             issues=[_issue(3, "commented"), _issue(3, "commented"),
                     _issue(7, "commented"), _issue(7, "commented"),
                     _issue(7, "commented"), _issue(7, "closed"),
                     _issue(8, "commented"), _issue(8, "commented"),
                     _issue(5, "labeled")]
                    + [_issue(100 + i, "untouched") for i in range(20)])
    # Sweep run: no queued-issue dispositions, but its WET spend still counts.
    _put_run(s3, "2026-07-09", "070001", run_wet=2_000_000.0, issues=[], chunk_log=[])
    # Producer's fail-safe WET compute errored → run_wet null BY CONTRACT.
    _put_run(s3, "2026-07-10", "190004", run_wet=None, chunk_log=[_OK_CHUNK],
             issues=[_issue(50, "untouched")])
    # OUT of the trailing-7d window — huge WET + a completion + a max_turns
    # chunk; the exact assertions below prove it never enters the computation.
    _put_run(s3, "2026-07-03", "190000", run_wet=9_999_999.0,
             chunk_log=[_MAX_TURNS_CHUNK_A], issues=[_issue(60, "closed")])


class TestGroomComponents:
    def test_completion_rate_red_at_baseline_like_rate(self, s3):
        _put_window_fixture(s3)
        cr = _comp(build_groom_components(BUCKET, RUN_DATE, s3_client=s3),
                   "groom_completion_rate")
        assert cr["value"] == pytest.approx(4 / 35)
        assert cr["n_samples"] == 35
        assert cr["status"] == "RED"  # 11.4% < 15% red-line
        assert cr["criticality"] == "supporting"
        assert "4/35" in cr["status_reason"]

    def test_completion_rate_green_when_above_target(self, s3):
        _put_run(s3, "2026-07-09", "190001", chunk_log=[_OK_CHUNK],
                 issues=[_issue(i, "closed") for i in range(12)]
                        + [_issue(100 + i, "untouched") for i in range(20)])
        cr = _comp(build_groom_components(BUCKET, RUN_DATE, s3_client=s3),
                   "groom_completion_rate")
        assert cr["value"] == pytest.approx(12 / 32)
        assert cr["status"] == "GREEN"  # 37.5% ≥ 30% target

    def test_wet_per_completion_excludes_null_wet_runs(self, s3):
        _put_window_fixture(s3)
        wet = _comp(build_groom_components(BUCKET, RUN_DATE, s3_client=s3),
                    "groom_wet_per_completion")
        # 4.0M WET (null-WET run excluded, out-of-window 9,999,999 excluded)
        # over 4 completions = 1.0M per completion.
        assert wet["value"] == pytest.approx(1_000_000.0)
        assert wet["n_samples"] == 4
        assert wet["criticality"] == "diagnostic"
        assert wet["status"] == "GREEN"  # band-less / directional-down during soak
        assert "1 run(s) with null run_wet excluded" in wet["status_reason"]

    def test_wet_undefined_when_zero_completions(self, s3):
        _put_run(s3, "2026-07-09", "190001", run_wet=750_000.0, chunk_log=[_OK_CHUNK],
                 issues=[_issue(1, "commented"), _issue(2, "untouched")])
        wet = _comp(build_groom_components(BUCKET, RUN_DATE, s3_client=s3),
                    "groom_wet_per_completion")
        assert wet["value"] is None
        assert wet["status"] == "N/A-LOW-N"  # ratio undefined, never a fabricated 0
        assert "0 completions" in wet["status_reason"]

    def test_comment_churn_requires_three_comments_and_no_completion(self, s3):
        _put_window_fixture(s3)
        churn = _comp(build_groom_components(BUCKET, RUN_DATE, s3_client=s3),
                      "groom_comment_churn")
        # #3: 3× commented, never completed → churns. #7: 3× commented but
        # closed → no. #8: only 2× commented → no.
        assert churn["value"] == 1.0
        assert churn["n_samples"] == 29  # distinct (repo, number) in-window
        assert churn["status"] == "GREEN"  # 1 ≤ provisional target 5
        assert churn["criticality"] == "supporting"

    def test_comment_churn_red_when_over_red_line(self, s3):
        _put_run(s3, "2026-07-09", "190001", chunk_log=[_OK_CHUNK],
                 issues=[_issue(n, "commented") for n in range(25) for _ in range(3)])
        churn = _comp(build_groom_components(BUCKET, RUN_DATE, s3_client=s3),
                      "groom_comment_churn")
        assert churn["value"] == 25.0
        assert churn["status"] == "RED"  # 25 > 20 red-line

    def test_lost_chunks_counts_max_turns_entries(self, s3):
        _put_window_fixture(s3)
        lost = _comp(build_groom_components(BUCKET, RUN_DATE, s3_client=s3),
                     "groom_lost_chunks")
        # Both signature spellings count; ok chunks and the out-of-window
        # max_turns chunk don't.
        assert lost["value"] == 2.0
        assert lost["n_samples"] == 5  # runs in-window
        assert lost["status"] == "WATCH"  # 0 < 2 < red-line 5
        assert "config#2148" in lost["status_reason"]

    def test_lost_chunks_red_over_red_line(self, s3):
        for i in range(5):
            _put_run(s3, "2026-07-09", f"19000{i}", chunk_log=[_MAX_TURNS_CHUNK_A, _MAX_TURNS_CHUNK_B],
                     issues=[_issue(i, "untouched")])
        lost = _comp(build_groom_components(BUCKET, RUN_DATE, s3_client=s3),
                     "groom_lost_chunks")
        assert lost["value"] == 10.0
        assert lost["status"] == "RED"

    def test_all_records_carry_module_and_window_horizon(self, s3):
        _put_window_fixture(s3)
        comps = build_groom_components(BUCKET, RUN_DATE, s3_client=s3)
        assert [c.name for c in comps] == list(GROOM_NAMES)
        for c in comps:
            assert c.module == "agent"
            assert c.source_path.startswith(f"s3://{BUCKET}/groom/")


class TestGroomDegradedWindows:
    def test_zero_artifact_window_grades_missing_input_naming_producer(self, s3):
        """An empty week (groomer dark / writer broken) must surface loudly on
        all four records — never a silent GREEN."""
        comps = build_groom_components(BUCKET, RUN_DATE, s3_client=s3)
        assert len(comps) == 4
        for c in (c.model_dump(mode="json") for c in comps):
            assert c["status"] == "N/A-MISSING-INPUT", c["name"]
            assert "write_run_artifact" in c["status_reason"], c["name"]
            assert "config#2151" in c["status_reason"], c["name"]

    def test_sweep_only_window_grades_low_n_not_green(self, s3):
        # Sweep runs carry zero queued-issue dispositions → the rate is
        # undefined and grades N/A, never a fabricated value.
        _put_run(s3, "2026-07-09", "070001", run_wet=1_000_000.0, issues=[], chunk_log=[])
        cr = _comp(build_groom_components(BUCKET, RUN_DATE, s3_client=s3),
                   "groom_completion_rate")
        assert cr["value"] is None
        assert cr["status"] == "N/A-LOW-N"
        assert "sweep-only" in cr["status_reason"]


class TestGroomMalformedArtifacts:
    def test_unparseable_json_raises(self, s3):
        _put_run(s3, "2026-07-09", "190001", body=b"{not json")
        with pytest.raises(GroomArtifactError, match="unparseable"):
            build_groom_components(BUCKET, RUN_DATE, s3_client=s3)

    def test_non_list_issues_raises(self, s3):
        _put_run(s3, "2026-07-09", "190001",
                 body=json.dumps({"schema_version": 6, "issues": "oops",
                                  "chunk_log": []}).encode())
        with pytest.raises(GroomArtifactError, match="issues/chunk_log"):
            build_groom_components(BUCKET, RUN_DATE, s3_client=s3)

    def test_malformed_issue_record_raises(self, s3):
        _put_run(s3, "2026-07-09", "190001",
                 issues=[{"repo": "alpha-engine-config", "number": 1}])  # no disposition
        with pytest.raises(GroomArtifactError, match="malformed issue record"):
            build_groom_components(BUCKET, RUN_DATE, s3_client=s3)

    def test_non_numeric_run_wet_raises(self, s3):
        _put_run(s3, "2026-07-09", "190001", run_wet="a lot")
        with pytest.raises(GroomArtifactError, match="non-numeric run_wet"):
            build_groom_components(BUCKET, RUN_DATE, s3_client=s3)


class TestAgentTileIntegration:
    def test_agent_tile_carries_groom_components(self, s3):
        _put_window_fixture(s3)
        tile = build_agent_tile(BUCKET, RUN_DATE, s3_client=s3)
        names = {c["name"] for c in tile["components"]}
        assert set(GROOM_NAMES) <= names
        assert tile["n_components"] == 11
        cr = next(c for c in tile["components"] if c["name"] == "groom_completion_rate")
        assert cr["status"] == "RED"
        # supporting RED + the pre-existing critical N/As → tile WATCH.
        assert tile["status"] == "WATCH"

    def test_agent_tile_groom_absent_stays_transparent(self, s3):
        tile = build_agent_tile(BUCKET, RUN_DATE, s3_client=s3)
        statuses = {c["status"] for c in tile["components"] if c["name"] in GROOM_NAMES}
        assert statuses == {"N/A-MISSING-INPUT"}
