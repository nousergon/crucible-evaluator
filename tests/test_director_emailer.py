"""Director weekly digest email — thin summary + console deep-link.

The Director gains its own digest email (mirroring the EOD / model-zoo /
backtester patterns): a short summary that deep-links to the console Director
page for the full proposed action plan. The send is best-effort (the lib's
transport never raises; the build is wrapped) so it can never break the run.
"""
import director.emailer as de
from director.schema import ActionItem, DirectorWeeklyActionPlan


def _plan() -> DirectorWeeklyActionPlan:
    return DirectorWeeklyActionPlan(
        run_date="2026-06-26",
        system_summary="System is RED on predictor calibration; research edge flat.",
        top_risks=["Predictor calibration breakdown", "Thin liquidity coverage"],
        action_items=[
            ActionItem(
                id="fix-cal", title="Investigate calibration breakdown",
                rationale="ECE > 0.10 on the predictor tile.", evidence=["predictor.ece"],
                proposed_owner="predictor", priority="P0", horizon="this_week",
                suggested_change_type="investigation", confidence=80,
            ),
            ActionItem(
                id="watch-liq", title="Watch liquidity coverage",
                rationale="Scanner coverage dipped.", evidence=["research.coverage"],
                proposed_owner="research", priority="P2", horizon="watch",
                suggested_change_type="no_action_monitor", confidence=55,
            ),
        ],
    )


def test_director_plan_url_and_slug():
    assert de.DIRECTOR_SLUG == "director"
    assert (
        de.director_plan_url("2026-06-26")
        == "https://dashboard.nousergon.ai/director?date=2026-06-26"
        # NOT console.nousergon.ai: that hostname was handed to nousergon-console
        # v2 and the Streamlit dashboard this slug resolves against now lives at
        # dashboard.nousergon.ai (krepis#137 / alpha-engine-config#6140, merged
        # 2026-08-12). krepis.console.DEFAULT_CONSOLE_BASE_URL is the single
        # source; this literal exists to catch it moving again, so it is pinned
        # rather than imported.
    )
    assert de.director_plan_url(
        "2026-06-26", "https://stage.example.com/"
    ) == "https://stage.example.com/director?date=2026-06-26"


def test_build_director_digest_summary_risks_items_and_link():
    subject, plain, html = de.build_director_digest(_plan(), "2026-06-26")
    url = "https://dashboard.nousergon.ai/director?date=2026-06-26"
    # Subject summarizes count + priority mix.
    assert "Director | 2026-06-26 | 2 action items" in subject
    assert "P0:1" in subject and "P2:1" in subject
    # Console deep-link in both bodies.
    assert url in plain
    assert f'href="{url}"' in html
    # Content present.
    assert "calibration" in plain
    assert "Predictor calibration breakdown" in html      # top risk
    assert "Investigate calibration breakdown" in html    # action item
    # P0 sorts before P2 in both renderings.
    assert plain.index("[P0]") < plain.index("[P2]")


def test_build_director_digest_accepts_plain_dict():
    subject, plain, _ = de.build_director_digest(
        {"system_summary": "ok", "top_risks": [], "action_items": []}, "2026-06-26"
    )
    assert "0 action items" in subject
    assert "(none proposed this week)" in plain


def test_send_director_digest_returns_transport_result(monkeypatch):
    sent = {}

    def _fake_send(subject, body, *, html=None):
        sent.update(subject=subject, body=body, html=html)
        return True

    # Patch where it is looked up (imported inside the function from krepis).
    import krepis.email_sender as ks
    monkeypatch.setattr(ks, "send_email", _fake_send)
    assert de.send_director_digest(_plan(), "2026-06-26") is True
    assert "Director | 2026-06-26" in sent["subject"]
    assert "dashboard.nousergon.ai/director?date=2026-06-26" in sent["body"]


def test_send_director_digest_never_raises_on_bad_plan(monkeypatch):
    # A malformed plan must not break the Director — the send returns False.
    import krepis.email_sender as ks
    monkeypatch.setattr(ks, "send_email", lambda *a, **k: True)

    class _Boom:
        def model_dump(self):
            raise ValueError("corrupt plan")

    assert de.send_director_digest(_Boom(), "2026-06-26") is False


def test_director_digest_makes_unexamined_ledger_rows_visible():
    """A zero-action result is not a clean Director loop when rows had no issue."""
    _, plain, _ = de.build_director_digest(
        _plan(),
        "2026-06-26",
        loop_summary={
            "director_loop": "ok",
            "director_loop_examined": 0,
            "director_loop_skipped_no_issue": 28,
            "director_loop_backfilled": 0,
            "director_loop_corrections": 0,
        },
    )
    assert "0 examined" in plain
    assert "28 skipped: no issue number" in plain
    assert "0 backfilled" in plain
    assert "0 corrections" in plain


def test_director_digest_marks_skipped_loop_not_run_in_plain_and_html():
    _, plain, html = de.build_director_digest(
        _plan(),
        "2026-06-26",
        loop_summary={"director_loop": "skipped", "director_loop_reason": "budget exhausted"},
    )
    assert "Director loop: NOT RUN — budget exhausted" in plain
    assert "Director loop: NOT RUN — budget exhausted" in html


def test_director_digest_marks_backfill_failure_partial_in_plain_and_html():
    _, plain, html = de.build_director_digest(
        _plan(),
        "2026-06-26",
        loop_summary={
            "director_loop": "partial",
            "director_loop_examined": 0,
            "director_loop_skipped_no_issue": 28,
            "director_loop_backfilled": 0,
            "director_loop_corrections": 0,
            "director_loop_backfill_error": "GitHub unavailable",
        },
    )
    assert "Director loop: PARTIAL" in plain
    assert "backfill FAILED" in plain
    assert "Director loop: PARTIAL" in html
    assert "backfill FAILED" in html


def test_director_digest_marks_withheld_backfill_failure():
    _, plain, html = de.build_director_digest(
        _plan(),
        "2026-06-26",
        loop_summary={
            "director_loop": "mutations_withheld",
            "director_loop_backfilled": 0,
            "director_loop_backfill_error": "GitHub unavailable",
        },
    )
    assert "backfill FAILED" in plain
    assert "backfill FAILED" in html
