"""A fallback-served Director plan is marked DEGRADED — alpha-engine-config-I8165.

The `ultra` group gained a second arm on 2026-08-22 (`deepseek-v4-pro`, behind a
different provider from the Zhipu primary). That arm is deliberately WEAKER for
the Director's plan call: it is admitted for AVAILABILITY only
(model-router-policy R33), because before it the group was a chain of ONE and a
slow or unavailable Zhipu was a full outage — four plan attempts censored with
zero completion tokens took the 2026-08-22 weekly run and its rerun down
(alpha-engine-config-I8151).

Brian's ruling admitting that arm was CONDITIONED on the degradation being
visible: a weaker model's plan must never enter the record as if the champion
wrote it. These tests are that condition. They assert the property on both
surfaces named in the ruling — the plan ARTIFACT and the REPORT CARD — and they
assert the negative and the unknown cases too, because a marker that is always
on, or one that renders "unmeasured" as "champion served", would satisfy the
letter of the ruling and none of its purpose.

The contract test at the bottom is the one that survives a rename: producer and
consumer name these keys independently (the Report Card Lambda does not package
`director/`), so nothing but this test stops one side from drifting.
"""

import json

import boto3
import pytest
from moto import mock_aws

from director.agent import (
    PLAN_KEY_DEGRADED_REASON,
    PLAN_KEY_ROUTE_DEGRADED,
    PLAN_KEY_ROUTE_PRIMARY_MODEL,
    PLAN_KEY_SERVED_MODEL,
    _KrepisStructuredDirector,
    _stamp_route_degradation,
)
from director.schema import DirectorWeeklyActionPlan
from director.verdict import stamp_plan_artifact
from grading.tiles import director_quality as dq
from grading.tiles.director_quality import (
    LATEST_ACTION_PLAN_KEY,
    build_director_quality_tile,
)

BUCKET = "alpha-engine-research"
RUN_DATE = "2026-08-29"

PRIMARY = "glm-5.2"          # ultra's declared primary (zhipu)
FALLBACK = "deepseek-v4-pro"  # ultra's second arm (deepseek), weaker by design


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _plan() -> DirectorWeeklyActionPlan:
    """A minimal valid plan.

    ``action_items`` is deliberately empty: these tests are about the route
    stamp, and pinning them to ``ActionItem``'s required-field list — four of
    which are ``Literal`` enums that have changed more than once — would make
    them fail for reasons that have nothing to do with what they assert.
    """
    return DirectorWeeklyActionPlan(
        run_date=RUN_DATE,
        system_summary="A summary.",
        top_risks=["a risk"],
        action_items=[],
    )


class _Result:
    """Stand-in for ``krepis.llm.StructuredResult`` — only what invoke() reads."""

    def __init__(self, model: str, parsed):
        self.model = model
        self.parsed = parsed
        self.usage = None


class _Client:
    def __init__(self, served_model: str):
        self._served = served_model

    def structured(self, **kwargs):
        return _Result(self._served, _plan())


# ── The plan artifact ────────────────────────────────────────────────────────

class TestPlanArtifact:
    def test_fallback_served_plan_is_marked_degraded_and_names_the_model(self):
        llm = _KrepisStructuredDirector(
            _Client(FALLBACK), director_model="ultra-glm-5.2-direct",
            primary_model=PRIMARY,
        )
        plan = llm.invoke([{"role": "system", "content": "s"},
                           {"role": "user", "content": "u"}])

        assert getattr(plan, PLAN_KEY_ROUTE_DEGRADED) is True
        assert getattr(plan, PLAN_KEY_SERVED_MODEL) == FALLBACK
        assert getattr(plan, PLAN_KEY_ROUTE_PRIMARY_MODEL) == PRIMARY
        # Naming the model that served is the deliverable, not a nicety: a bare
        # "degraded" flag cannot tell a reader WHICH weaker model wrote it.
        reason = getattr(plan, PLAN_KEY_DEGRADED_REASON)
        assert FALLBACK in reason and PRIMARY in reason

    def test_primary_served_plan_is_not_marked_degraded(self):
        llm = _KrepisStructuredDirector(
            _Client(PRIMARY), director_model="ultra-glm-5.2-direct",
            primary_model=PRIMARY,
        )
        plan = llm.invoke([{"role": "system", "content": "s"},
                           {"role": "user", "content": "u"}])

        # A marker that is always on is the I6185 failure mode arrived at from
        # the other direction — it trains the reader to ignore the week it means
        # something.
        assert getattr(plan, PLAN_KEY_ROUTE_DEGRADED) is False
        assert getattr(plan, PLAN_KEY_SERVED_MODEL) == PRIMARY
        assert getattr(plan, PLAN_KEY_DEGRADED_REASON) is None

    def test_unknown_primary_stamps_none_not_false(self):
        """`None` and `False` are different answers and must stay different.

        A route that declared no primary cannot tell "the champion served" from
        "nobody looked". Collapsing that into `False` renders an unmeasured week
        green — `principles.md` §2.7.
        """
        llm = _KrepisStructuredDirector(
            _Client(FALLBACK), director_model="ultra-glm-5.2-direct",
            primary_model=None,
        )
        plan = llm.invoke([{"role": "system", "content": "s"},
                           {"role": "user", "content": "u"}])
        assert getattr(plan, PLAN_KEY_ROUTE_DEGRADED) is None
        assert "unknown" in getattr(plan, PLAN_KEY_DEGRADED_REASON)

    def test_stamp_survives_serialization_into_the_written_artifact(self):
        """The stamp must reach S3, not just the in-memory object.

        `stamp_plan_artifact` is what `director/handler.py` actually writes. The
        stamp rides as a pydantic EXTRA (the plan model is the LLM's own output
        schema — a declared field would be a field the model is asked to
        produce, and a plan cannot be trusted to report its own degradation), so
        this asserts `extra="allow"` really does carry it through
        `model_dump_json`.
        """
        plan = _plan()
        _stamp_route_degradation(plan, served_model=FALLBACK, primary_model=PRIMARY)
        body = json.loads(stamp_plan_artifact(plan, {}))

        assert body[PLAN_KEY_ROUTE_DEGRADED] is True
        assert body[PLAN_KEY_SERVED_MODEL] == FALLBACK
        assert body[PLAN_KEY_ROUTE_PRIMARY_MODEL] == PRIMARY

    def test_stamping_never_raises_and_never_loses_the_plan(self):
        """Telemetry failure must not take down the weekly plan."""
        class _Unstampable:
            __slots__ = ()

        assert _stamp_route_degradation(
            _Unstampable(), served_model=FALLBACK, primary_model=PRIMARY,
        ) is True


# ── The Report Card ──────────────────────────────────────────────────────────

def _put_plan(s3, **stamp):
    s3.put_object(
        Bucket=BUCKET, Key=LATEST_ACTION_PLAN_KEY,
        Body=json.dumps({"run_date": RUN_DATE, **stamp}).encode("utf-8"),
    )


def _comp(tile, name):
    return next(c for c in tile["components"] if c["name"] == name)


class TestReportCard:
    def test_fallback_plan_renders_red_and_names_the_served_model(self, s3):
        _put_plan(s3, **{
            PLAN_KEY_ROUTE_DEGRADED: True,
            PLAN_KEY_SERVED_MODEL: FALLBACK,
            PLAN_KEY_ROUTE_PRIMARY_MODEL: PRIMARY,
            PLAN_KEY_DEGRADED_REASON: "plan produced by FALLBACK model",
        })
        tile = build_director_quality_tile(BUCKET, RUN_DATE, s3_client=s3)
        c = _comp(tile, "director_route_degraded")

        assert c["value"] == 1.0
        assert c["status"] == "RED"
        # The ruling's words: "naming the model that actually served".
        assert FALLBACK in c["status_reason"]
        assert PRIMARY in c["status_reason"]

    def test_primary_plan_renders_green(self, s3):
        _put_plan(s3, **{
            PLAN_KEY_ROUTE_DEGRADED: False,
            PLAN_KEY_SERVED_MODEL: PRIMARY,
            PLAN_KEY_ROUTE_PRIMARY_MODEL: PRIMARY,
            PLAN_KEY_DEGRADED_REASON: None,
        })
        tile = build_director_quality_tile(BUCKET, RUN_DATE, s3_client=s3)
        c = _comp(tile, "director_route_degraded")
        assert c["value"] == 0.0
        assert c["status"] == "GREEN"
        assert PRIMARY in c["status_reason"]

    def test_absent_artifact_renders_na_not_green(self, s3):
        tile = build_director_quality_tile(BUCKET, RUN_DATE, s3_client=s3)
        c = _comp(tile, "director_route_degraded")
        assert c["status"] == "N/A-MISSING-INPUT"
        assert c["value"] is None

    def test_pre_i8165_plan_without_the_stamp_renders_na_not_green(self, s3):
        """An artifact written before this landed asserts nothing, not health."""
        _put_plan(s3)
        c = _comp(build_director_quality_tile(BUCKET, RUN_DATE, s3_client=s3),
                  "director_route_degraded")
        assert c["status"] == "N/A-MISSING-INPUT"

    def test_unknown_stamp_renders_na_not_green(self, s3):
        _put_plan(s3, **{
            PLAN_KEY_ROUTE_DEGRADED: None,
            PLAN_KEY_SERVED_MODEL: None,
            PLAN_KEY_ROUTE_PRIMARY_MODEL: None,
            PLAN_KEY_DEGRADED_REASON: "served-model unknown",
        })
        c = _comp(build_director_quality_tile(BUCKET, RUN_DATE, s3_client=s3),
                  "director_route_degraded")
        assert c["status"] == "N/A-MISSING-INPUT"
        assert c["value"] is None

    def test_route_component_is_supporting_and_cannot_force_overall_red(self, s3):
        """RED here is a loud signal, not a pipeline halt.

        A degraded plan is a quality event, not a correctness failure: the plan
        is still real and still advisory. `supporting` keeps it off the critical
        gate while leaving it red on the surface a human reads.
        """
        _put_plan(s3, **{
            PLAN_KEY_ROUTE_DEGRADED: True,
            PLAN_KEY_SERVED_MODEL: FALLBACK,
            PLAN_KEY_ROUTE_PRIMARY_MODEL: PRIMARY,
            PLAN_KEY_DEGRADED_REASON: "plan produced by FALLBACK model",
        })
        tile = build_director_quality_tile(BUCKET, RUN_DATE, s3_client=s3)
        assert _comp(tile, "director_route_degraded")["criticality"] == "supporting"


# ── The producer/consumer key contract ───────────────────────────────────────

class TestArtifactContract:
    def test_producer_and_consumer_name_the_same_keys(self):
        """`director/agent.py` and the tile declare these literals separately.

        The Report Card Lambda does not package `director/`, so the tile cannot
        import them. Nothing else in CI compares the two lists — a rename on one
        side would leave the Report Card silently reading a key nobody writes,
        which renders N/A forever and looks exactly like "the Director was off".
        """
        from director import agent

        for producer_const, consumer_const in (
            (agent.PLAN_KEY_ROUTE_DEGRADED, dq.PLAN_KEY_ROUTE_DEGRADED),
            (agent.PLAN_KEY_SERVED_MODEL, dq.PLAN_KEY_SERVED_MODEL),
            (agent.PLAN_KEY_ROUTE_PRIMARY_MODEL, dq.PLAN_KEY_ROUTE_PRIMARY_MODEL),
            (agent.PLAN_KEY_DEGRADED_REASON, dq.PLAN_KEY_DEGRADED_REASON),
        ):
            assert producer_const == consumer_const

    def test_tile_reads_the_standing_pointer_the_handler_writes(self):
        """Both sides must name `director/latest/action_plan.json`.

        Binding the tile to the DATED key would render absent every week the
        template did not match (config-I7157's reasoning), so the pointer is the
        contract — asserted against the handler's own constant.
        """
        from director.handler import LATEST_ACTION_PLAN_KEY as produced

        assert produced == LATEST_ACTION_PLAN_KEY
