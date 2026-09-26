"""The Director's status is no better than the worst of its own sub-results.

``sf-pipeline-policy.md`` §2.3b, clause
``SFP-2.3b-stage-status-is-the-worst-substatus``; ``alpha-engine-config-I11299``.

MEASURED on three of the four 2026-09-19 executions that entered ``Director``
(read from their execution histories 2026-09-21) — the scheduled run and
``watch-rerun-2026-09-18-1`` and ``-3``, the last two terminating
``ExecutionSucceeded`` with ``degraded_summary.degraded: false``::

    director_result.Payload.status  = "ok"
    director_result.Payload.retro   = "error"
    director_result.Payload.retro_error =
        "Retro judge served 'deepseek-v4-pro', which is the model that
         produced the plan it was grading … Refusing to publish the grade"

The refusal is CORRECT — that is a judge doing its job, and its own root cause
is ``alpha-engine-config-I8202`` (registry invariant 15 compares entry ids, not
served models). What is wrong is that nothing above the sub-result could see
it: no RetroGrade has been published for at least two consecutive weekly cycles
and nothing anywhere went red.

The policy's own test is used verbatim below: *name the status this stage would
return if exactly one of those sub-results errored; if it is the same string it
returns today, the stage is in violation.*
"""

from __future__ import annotations

from director.substatus import (
    ERROR_SUB_STATUSES,
    PASS_SUB_STATUSES,
    REFUSED_SUB_STATUSES,
    STAGE_DEGRADED,
    STAGE_OK,
    STAGE_REFUSED,
    SUB_RESULT_KEYS,
    UNCLASSIFIED,
    apply_substatus_honesty,
    build_sub_statuses,
    classify,
    stage_status,
)

#: The Director summary of the 2026-09-19 SCHEDULED run, reduced to the fields
#: this clause reads. Every value is copied from that execution's own
#: ``director_result.Payload``.
SCHEDULED_2026_09_19 = {
    "status": "ok",
    "run_date": "2026-09-18",
    "n_action_items": 11,
    "digest_email": "sent",
    "retro": "refused",
    "retro_error": (
        "Retro judge served 'deepseek-v4-pro', which is the model that produced "
        "the plan it was grading (run_date '2026-09-11', served 'deepseek-v4-pro') "
        "— self-grading bias (config#1673)."
    ),
    "retro_outcome": "refused",
    "director_loop": "ok",
    "director_issues": "ok",
    "director_issues_n_filed": 3,
    "deploy_success": "ok",
    "deploy_success_rate": 0.988479262672811,
}


def test_the_measured_scheduled_run_reports_refused_not_ok_and_not_degraded():
    """The regression case, end to end.

    The 2026-09-19 retro was a GUARD REFUSAL, so the honest outcome is
    ``refused`` — never ``ok`` (the defect I11299 names) and never
    ``degraded`` (which would terminate a complete 5.4-hour cycle ``Fail``
    and page Brian for a guard doing its job).
    """
    summary = apply_substatus_honesty(dict(SCHEDULED_2026_09_19))

    assert summary["status"] == STAGE_REFUSED
    assert summary["status"] != STAGE_OK
    assert summary["degraded_sub_results"] == []
    assert summary["refused_sub_results"] == ["retro"]
    assert summary["retro_refused"] is True
    assert summary["sub_statuses"]["retro"]["status"] == "refused"
    assert summary["sub_statuses"]["retro"]["verdict"] == "refused"
    assert "self-grading bias" in summary["sub_statuses"]["retro"]["detail"]


def test_a_genuinely_errored_retro_still_degrades():
    """The other side of the split: an exception is not a refusal."""
    summary = apply_substatus_honesty(
        {**SCHEDULED_2026_09_19, "retro": "error", "retro_outcome": "failed",
         "retro_error": "judge overloaded"}
    )
    assert summary["status"] == STAGE_DEGRADED
    assert summary["degraded_sub_results"] == ["retro"]
    assert summary["refused_sub_results"] == []
    assert summary["retro_refused"] is False


def test_an_error_outranks_a_refusal_and_both_stay_named():
    """Worst-of decides the STATUS; the two lists keep both facts."""
    summary = apply_substatus_honesty(
        {**SCHEDULED_2026_09_19, "deploy_success": "error"}
    )
    assert summary["status"] == STAGE_DEGRADED
    assert summary["degraded_sub_results"] == ["deploy_success"]
    assert summary["refused_sub_results"] == ["retro"]
    assert summary["retro_refused"] is True


def test_retro_refused_is_emitted_on_every_fan_out_including_false():
    """principles.md §2.7: a component emitting nothing is unobserved, not
    healthy. The SF reads this boolean directly (a Choice cannot measure an
    array's length), so its ABSENCE must mean "no fan-out", never "no
    refusal"."""
    clean = apply_substatus_honesty({**SCHEDULED_2026_09_19, "retro": "ok"})
    assert clean["retro_refused"] is False
    assert clean["status"] == STAGE_OK
    no_fan_out = apply_substatus_honesty({"status": "disabled"})
    assert "retro_refused" not in no_fan_out


def test_the_healthy_legs_are_still_declared():
    """A stage that names only its failures cannot be read as "the rest was
    checked"."""
    summary = apply_substatus_honesty(dict(SCHEDULED_2026_09_19))
    for key in ("director_loop", "director_issues", "deploy_success"):
        assert summary["sub_statuses"][key]["verdict"] == "pass"


def test_the_flat_fields_every_existing_consumer_reads_are_unchanged():
    """Additive, never a rename: the digest emailer and the console row read
    the bare strings, and this clause must not move them."""
    summary = apply_substatus_honesty(dict(SCHEDULED_2026_09_19))
    for key in SUB_RESULT_KEYS:
        assert summary[key] == SCHEDULED_2026_09_19[key]
    assert summary["retro_error"] == SCHEDULED_2026_09_19["retro_error"]


def test_the_substatus_block_is_a_dict_of_dicts_carrying_status():
    """What makes the EXISTING conformance probe see this stage at all.

    ``nous-ergon-ops/scripts/sf_substatus_honesty.py::_scan_node`` descends
    only into sub-results that are dicts carrying their own ``status`` key, so
    the Director's bare-string legs were invisible to it: run live against the
    2026-09-19 scheduled execution on 2026-09-21 it reported *0 stage(s)
    reported a pass over an errored sub-result, across 88 stage result(s)* —
    blind to the very instance the clause was amended for.
    """
    block = build_sub_statuses(SCHEDULED_2026_09_19)
    for key, entry in block.items():
        assert isinstance(entry, dict), key
        assert isinstance(entry["status"], str), key


def test_a_clean_run_stays_ok():
    summary = apply_substatus_honesty(
        {**SCHEDULED_2026_09_19, "retro": "ok", "retro_error": None, "retro_outcome": None}
    )
    assert summary["status"] == STAGE_OK
    assert summary["degraded_sub_results"] == []


def test_a_skipped_retro_does_not_degrade_the_run():
    """``error`` and ``skipped`` are different sub-statuses.

    The first cycle has no prior plan to grade, and a late invocation declines
    a call it cannot finish. Neither is work that failed, and
    ``alpha-engine-config-I11299`` names this carve-out explicitly.
    """
    summary = apply_substatus_honesty(
        {
            **SCHEDULED_2026_09_19,
            "retro": "skipped",
            "retro_reason": "no prior plan (first cycle)",
            "retro_error": None,
        }
    )
    assert summary["status"] == STAGE_OK
    assert summary["degraded_sub_results"] == []


def test_a_withheld_leg_does_not_double_count_the_2_3a_verdict():
    """§2.3a withholding is already its own reported family."""
    summary = apply_substatus_honesty(
        {**SCHEDULED_2026_09_19, "retro": "ok", "director_issues": "withheld"}
    )
    assert summary["status"] == STAGE_OK


def test_director_issues_nochange_is_a_pass_not_unclassified():
    """Measured 2026-09-26 cadence run: sole DegradedRun reason was
    weekly_director_substatus_degraded with degraded_sub_results=["director_issues"]
    and director_issues="nochange" (all items already filed). That is correctly
    did less — same family as skipped — not an unclassified failure.
    """
    summary = apply_substatus_honesty(
        {**SCHEDULED_2026_09_19, "retro": "ok", "director_issues": "nochange"}
    )
    assert summary["status"] == STAGE_OK
    assert summary["degraded_sub_results"] == []
    entry = summary["sub_statuses"]["director_issues"]
    assert entry["status"] == "nochange"
    assert entry["verdict"] == "pass"


def test_an_unrecognised_sub_status_is_unclassified_and_degrades():
    """The vocabulary is closed at BOTH ends (§2.3b).

    A reader that treats an unrecognised status as healthy goes quiet the first
    time a producer invents a word for failure.
    """
    summary = apply_substatus_honesty(
        {**SCHEDULED_2026_09_19, "retro": "ok", "deploy_success": "quarantined"}
    )
    assert summary["status"] == STAGE_DEGRADED
    assert summary["degraded_sub_results"] == ["deploy_success"]
    entry = summary["sub_statuses"]["deploy_success"]
    assert entry["status"] == UNCLASSIFIED
    assert entry["reported_status"] == "quarantined"


def test_a_non_string_sub_status_is_unclassified():
    summary = apply_substatus_honesty({**SCHEDULED_2026_09_19, "retro": {"status": "ok"}})
    assert summary["status"] == STAGE_DEGRADED
    assert summary["sub_statuses"]["retro"]["status"] == UNCLASSIFIED


def test_a_leg_that_did_not_run_is_not_invented():
    """A summary from a path with no fan-out declares no legs it does not have."""
    block = build_sub_statuses({"status": "ok", "retro": "ok"})
    assert set(block) == {"retro"}


def test_the_policy_s_own_test_for_every_leg():
    """*Name the status this stage would return if exactly one sub-result
    errored. If it is the same string it returns today, the stage is in
    violation.* — sf-pipeline-policy.md §2.3b."""
    for key in SUB_RESULT_KEYS:
        clean = {**SCHEDULED_2026_09_19, "retro": "ok"}
        assert apply_substatus_honesty(dict(clean))["status"] == STAGE_OK
        errored = apply_substatus_honesty({**clean, key: "error"})
        assert errored["status"] == STAGE_DEGRADED, key
        assert key in errored["degraded_sub_results"], key


def test_the_three_vocabularies_are_disjoint():
    assert not (PASS_SUB_STATUSES & ERROR_SUB_STATUSES)
    assert not (PASS_SUB_STATUSES & REFUSED_SUB_STATUSES)
    assert not (ERROR_SUB_STATUSES & REFUSED_SUB_STATUSES)


def test_classify_is_total():
    assert classify(None) == "pass"
    assert classify("ok") == "pass"
    assert classify("nochange") == "pass"
    assert classify("error") == "error"
    assert classify("refused") == "refused"
    assert classify("nonsense") == UNCLASSIFIED
    assert classify(7) == UNCLASSIFIED


def test_stage_status_of_an_empty_fan_out_is_ok():
    assert stage_status({}) == STAGE_OK
