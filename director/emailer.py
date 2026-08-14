"""Director weekly digest email — a thin summary that deep-links to the console
Director page for the full proposed action plan.

Mirrors the EOD / model-zoo / backtester digest patterns: the email is a short
executive summary (system read, top risks, the action-items table) with a
prominent link to the console Director page, where the full plan (rationale,
evidence, carry-over, self-grade) lives. The slug ``director`` is pinned in
crucible-dashboard ``app.py`` (``url_path="director"``) and guarded by
``tests/test_director_page.py``; the page honors ``?date=YYYY-MM-DD`` keyed by
the Director's ``run_date`` (the last completed trading day — Friday for a
Saturday run), so the link opens the exact week.

Transport is ``krepis.email_sender.send_email`` (Gmail SMTP primary, SES
fallback; resolves ``EMAIL_SENDER`` / ``EMAIL_RECIPIENTS`` /
``GMAIL_APP_PASSWORD`` from SSM via ``get_secret``; **never raises**). The send
is best-effort: missing config or a transport failure logs + returns ``False``
and never breaks the Director run.
"""
from __future__ import annotations

import logging
from typing import Any

from krepis.console import console_url

log = logging.getLogger(__name__)

# Cross-repo contract: equals the dashboard's pinned ``url_path`` for the
# Director page (tests/test_director_page.py guards both sides). Stays local;
# only the base-URL builder is lifted.
DIRECTOR_SLUG = "director"

# P0 first when ordering the action-items table.
_PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


def director_plan_url(run_date: str, console_base_url: str | None = None) -> str:
    """Deep-link to the console Director page for ``run_date``.

    Thin wrapper over the lifted :func:`krepis.console.console_url` chokepoint
    (config#1300) — the base-URL literal now lives once in krepis.
    """
    return console_url(DIRECTOR_SLUG, date=run_date, base=console_base_url)


def _as_dict(plan: Any) -> dict:
    """Normalize a ``DirectorWeeklyActionPlan`` (pydantic) or a plain dict."""
    if hasattr(plan, "model_dump"):
        return plan.model_dump()
    if isinstance(plan, dict):
        return plan
    return {}


_LOOP_SUMMARY_LABELS = (
    ("open", "open"),
    ("closed_verified", "closed & verified"),
    ("closed_unrecovered", "closed but UNRECOVERED (reopened)"),
    ("closed_unverifiable", "closed, unverifiable"),
    ("escalated", "escalated to Decision Queue"),
)


def _loop_summary_line(loop_summary: dict | None) -> str | None:
    """One line reporting Director-loop status (config#3145 point 4): items
    open / closed-and-verified / closed-but-unrecovered / escalated. ``None``
    if the pass didn't run this cycle (no GH token, or an error — those cases
    still show up in the Lambda's own summary/logs, not worth cluttering the
    digest email over)."""
    if not loop_summary or loop_summary.get("director_loop") != "ok":
        return None
    parts = [
        f"{loop_summary.get(f'director_loop_{key}', 0)} {label}"
        for key, label in _LOOP_SUMMARY_LABELS
    ]
    return "Director loop: " + ", ".join(parts)


def _verdict_banner(verdict_block: dict | None) -> tuple[str, str, str] | None:
    """``(subject_prefix, plain_banner, html_banner)`` for a non-PASS verdict.

    ``sf-pipeline-policy.md`` §2.3a rule 3: every surface presenting the run's
    results carries the verdict state. This email IS such a surface — it is the
    one Brian reads first — and a digest of action items derived from numbers
    nothing checked, sent with no qualifier, is the guarantee being granted by
    default in the most consequential place it could be.

    ``None`` on PASS: the banner exists to withhold, and an all-clear banner on
    every one of ~52 emails a year trains the eye past the ones that matter.
    The verdict is still stated in the footer on a PASS.
    """
    from director.verdict import PIPELINE_GATES_KEY, actions_withheld
    from grading.pipeline_gates import gates_unmeasured

    vb = verdict_block or {}
    gates = vb.get(PIPELINE_GATES_KEY) or {}
    if not actions_withheld(vb):
        # alpha-engine-config-I7282 — the attestation passed, but the pipeline's
        # own pre-spend correctness gates may not have run. That is a genuinely
        # milder finding and gets a genuinely milder banner: AMBER, and it says
        # in its own first sentence that the numbers are unaffected and nothing
        # was withheld. The calibration is the point — a reader who cannot tell
        # "a check did not run" from "the numbers are wrong" starts discarding
        # both, and the week the arithmetic really moves is the week the habit
        # costs something.
        if gates and gates_unmeasured(gates):
            return _gate_only_banner(gates)
        return None

    verdict = vb.get("verdict", "UNKNOWN")
    reason = (vb.get("reason") or "").strip()
    as_of = vb.get("as_of") or {}
    as_of_line = ", ".join(
        f"{k}: {v or 'never'}" for k, v in sorted(as_of.items())
    ) or "no verdict timestamps recorded"

    if verdict == "FAIL":
        headline = (
            "CORRECTNESS ATTESTATION: FAIL — the numbers behind this plan are "
            "WRONG, not merely unverified."
        )
        prefix = "[NUMBERS WRONG] "
    else:
        headline = (
            "CORRECTNESS ATTESTATION: UNKNOWN — the numbers behind this plan are "
            "NOT established as correct."
        )
        prefix = "[UNVERIFIED] "

    withheld_line = (
        "Because of this the Director did NOT file issues and did NOT run its "
        "reopen/escalate loop this cycle. The plan below is advisory diagnosis "
        "only; nothing was tracked, reopened or escalated from it."
    )
    gate_line = (gates.get("statement") or "").strip()
    plain = "\n".join([
        f"!! {headline}",
        f"   {reason}" if reason else "",
        f"   Verdict as-of — {as_of_line}",
        f"   {withheld_line}",
        f"   Pipeline gates — {gate_line}" if gate_line else "",
        "",
    ])
    html = (
        "<div style=\"border:2px solid #b00;background:#fff3f3;padding:10px 12px;"
        "margin:0 0 16px;\">"
        f"<p style='margin:0 0 6px;font-size:14px;'><b>&#9888; {headline}</b></p>"
        + (f"<p style='margin:0 0 6px;font-size:12px;'>{reason}</p>" if reason else "")
        + f"<p style='margin:0 0 6px;font-size:11px;color:#555;'>Verdict as-of — {as_of_line}</p>"
        f"<p style='margin:0;font-size:12px;'>{withheld_line}</p>"
        + (f"<p style='margin:6px 0 0;font-size:11px;color:#555;'>Pipeline gates — "
           f"{gate_line}</p>" if gate_line else "")
        + "</div>"
    )
    return prefix, plain, html


def _gate_only_banner(gates: dict) -> tuple[str, str, str]:
    """The amber banner for an unmeasured pre-spend gate on an otherwise-clean run.

    ``alpha-engine-config-I7282``. Deliberately NOT the red attestation banner:

    * the subject prefix is ``[GATES UNVERIFIED]``, not ``[UNVERIFIED]`` — the
      two findings are different and must not read as the same one;
    * the first sentence of the body states what is NOT wrong, because the
      failure mode being designed against is Brian discarding a usable card;
    * nothing is withheld, and the banner says so, so the absence of withheld
      actions is not read as the banner being decorative.
    """
    statement = (gates.get("statement") or "").strip()
    unmeasured = ", ".join(gates.get("unmeasured") or []) or "unnamed"
    headline = (
        "PIPELINE GATES: NOT VERIFIED — the weekly run's pre-spend correctness "
        f"gates did not all run this cycle ({unmeasured})."
    )
    context = (
        "The numbers in this plan are unaffected and nothing was withheld: the "
        "correctness attestation PASSED, the tiles are real, and the Director "
        "filed and escalated as usual. What is missing is the earlier check that "
        "the pipeline's own contract and library pins were sound before it spent. "
        "Read the plan; do not read it as gate-verified."
    )
    plain = "\n".join([
        f"!  {headline}",
        f"   {statement}" if statement else "",
        f"   {context}",
        "",
    ])
    html = (
        "<div style=\"border:2px solid #b58900;background:#fffbe6;padding:10px 12px;"
        "margin:0 0 16px;\">"
        f"<p style='margin:0 0 6px;font-size:14px;'><b>&#9888; {headline}</b></p>"
        + (f"<p style='margin:0 0 6px;font-size:12px;'>{statement}</p>" if statement else "")
        + f"<p style='margin:0;font-size:12px;color:#555;'>{context}</p>"
        "</div>"
    )
    return "[GATES UNVERIFIED] ", plain, html


def _verdict_footer(verdict_block: dict | None) -> str:
    """The one-line verdict statement every digest carries, PASS included.

    §2.3a rule 3 admits no "only when bad" reading: a surface that states the
    verdict only when it is not PASS is a surface where silence means pass, and
    silence is exactly what an absent verdict produces.
    """
    from director.verdict import PIPELINE_GATES_KEY

    vb = verdict_block or {}
    if not vb:
        return ("Correctness attestation: NOT READ by this digest. "
                "Pipeline gates: NOT READ by this digest.")
    as_of = vb.get("as_of") or {}
    stamps = ", ".join(f"{k} {v or 'never'}" for k, v in sorted(as_of.items()))
    # alpha-engine-config-I7282: the footer is the BOTH-POLARITY surface, so the
    # gate verdict is stated here on a clean run too. A line that appears only on
    # the bad week cannot be told apart from a producer that stopped emitting.
    gates = vb.get(PIPELINE_GATES_KEY) or {}
    gate_verdict = gates.get("verdict", "UNKNOWN") if gates else "NOT READ"
    gate_detail = ", ".join(gates.get("unmeasured") or []) if gates else ""
    return (
        f"Correctness attestation: {vb.get('verdict', 'UNKNOWN')}"
        + (f" (as-of {stamps})" if stamps else " (no as-of recorded)")
        + f". Pipeline gates: {gate_verdict}"
        + (f" (unmeasured: {gate_detail})" if gate_detail else "")
        + "."
    )


def build_director_digest(
    plan: Any, run_date: str, *, console_base_url: str | None = None,
    loop_summary: dict | None = None, verdict_block: dict | None = None,
) -> tuple[str, str, str]:
    """Build ``(subject, plain_body, html_body)`` for the weekly action plan.

    Thin by design — the full rationale/evidence/carry-over/self-grade stays on
    the console Director page this links to. ``loop_summary`` (the return of
    ``director.loop_verification.verify_and_correct``, folded into the
    handler's summary dict) adds a one-line status report on last week's
    carried-over items (config#3145 point 4) — open / closed-and-verified /
    closed-but-unrecovered / escalated.
    """
    p = _as_dict(plan)
    url = director_plan_url(run_date, console_base_url)
    summary = (p.get("system_summary") or "").strip()
    risks = [r for r in (p.get("top_risks") or []) if r]
    items = list(p.get("action_items") or [])

    counts: dict[str, int] = {}
    for it in items:
        counts[str(it.get("priority", "?"))] = counts.get(str(it.get("priority", "?")), 0) + 1
    pri_summary = " ".join(f"{k}:{counts[k]}" for k in sorted(counts, key=lambda k: _PRIORITY_ORDER.get(k, 9)))
    banner = _verdict_banner(verdict_block)
    footer = _verdict_footer(verdict_block)
    subject = (
        (banner[0] if banner else "")
        + f"Alpha Engine Director | {run_date} | {len(items)} action items"
        + (f" ({pri_summary})" if counts else "")
    )

    ordered = sorted(items, key=lambda it: _PRIORITY_ORDER.get(str(it.get("priority")), 9))

    # ── plain body ──
    plain_lines = []
    # §2.3a rule 3 — the verdict precedes the numbers. Above the console link,
    # because a reader who clicks straight through must have seen it first.
    if banner:
        plain_lines += [banner[1]]
    plain_lines += [
        f"View the full proposed action plan on the console:\n{url}",
        "",
        f"Alpha Engine — Director Weekly Action Plan ({run_date})",
        "",
    ]
    if summary:
        plain_lines += ["System read:", f"  {summary}", ""]
    if risks:
        plain_lines += ["Top risks:"] + [f"  - {r}" for r in risks] + [""]
    plain_lines += [f"Action items ({len(items)}):"]
    if ordered:
        for it in ordered:
            plain_lines.append(
                f"  [{it.get('priority', '?')}] {it.get('title', '(untitled)')} "
                f"— owner={it.get('proposed_owner', '?')} "
                f"horizon={it.get('horizon', '?')} conf={it.get('confidence', '?')}"
            )
    else:
        plain_lines.append("  (none proposed this week)")
    loop_line = _loop_summary_line(loop_summary)
    if loop_line:
        plain_lines += ["", loop_line]
    plain_lines += ["", footer,
                    f"Full detail (rationale, evidence, carry-over, self-grade): {url}"]
    plain_body = "\n".join(plain_lines)

    # ── html body ──
    rows = "".join(
        f"<tr><td style='padding:3px 8px;'><b>{it.get('priority', '?')}</b></td>"
        f"<td style='padding:3px 8px;'>{it.get('title', '(untitled)')}</td>"
        f"<td style='padding:3px 8px;'>{it.get('proposed_owner', '?')}</td>"
        f"<td style='padding:3px 8px;'>{it.get('horizon', '?')}</td>"
        f"<td style='padding:3px 8px;'>{it.get('confidence', '?')}</td></tr>"
        for it in ordered
    ) or "<tr><td colspan='5' style='padding:4px 8px;color:#888;'>(none proposed this week)</td></tr>"
    risks_html = (
        "<ul style='margin:4px 0;'>" + "".join(f"<li>{r}</li>" for r in risks) + "</ul>"
        if risks else ""
    )
    html_body = (
        "<html><body style=\"font-family:sans-serif;font-size:13px;color:#222;max-width:680px;\">"
        f"<h2 style='margin-bottom:4px;'>Director — Weekly Action Plan</h2>"
        f"<p style='color:#555;font-size:12px;margin-top:0;'>{run_date}</p>"
        + (banner[2] if banner else "")
        + f"<p style='font-size:14px;margin:0 0 16px;'>&#9654; "
        f"<a href=\"{url}\"><b>View the full proposed action plan on the console</b></a></p>"
        + (f"<p><b>System read.</b> {summary}</p>" if summary else "")
        + (f"<h3 style='margin-bottom:2px;'>Top risks</h3>{risks_html}" if risks else "")
        + f"<h3 style='margin-bottom:4px;'>Action items ({len(items)})</h3>"
        "<table style='border-collapse:collapse;font-size:12px;'>"
        "<tr style='background:#e0e0e0;'>"
        "<th style='padding:3px 8px;'>Priority</th><th style='padding:3px 8px;'>Title</th>"
        "<th style='padding:3px 8px;'>Owner</th><th style='padding:3px 8px;'>Horizon</th>"
        "<th style='padding:3px 8px;'>Conf</th></tr>"
        f"{rows}</table>"
        + (f"<p style='font-size:12px;'>{loop_line}</p>" if loop_line else "")
        + f"<p style='font-size:11px;color:#555;margin-top:14px;'>{footer}</p>"
        + "<p style='font-size:10px;color:#aaa;margin-top:20px;'>"
        "Advisory only — the Director proposes; rationale, evidence, carry-over, "
        f"and self-grade are on the console Director page (<a href=\"{url}\">link</a>).</p>"
        "</body></html>"
    )
    return subject, plain_body, html_body


def send_director_digest(
    plan: Any, run_date: str, *, console_base_url: str | None = None,
    loop_summary: dict | None = None, verdict_block: dict | None = None,
) -> bool:
    """Build + send the Director digest. Best-effort: returns the send result and
    NEVER raises (transport is the lib's fire-and-forget ``send_email``; the
    build is wrapped so a malformed plan can't break the Director run)."""
    try:
        subject, plain_body, html_body = build_director_digest(
            plan, run_date, console_base_url=console_base_url, loop_summary=loop_summary,
            verdict_block=verdict_block,
        )
    except Exception:  # noqa: BLE001 — the email must never break the Director
        log.warning("Director digest: build failed — skipping email", exc_info=True)
        return False
    from krepis.email_sender import send_email

    ok = send_email(subject, plain_body, html=html_body)
    log.info("Director digest email: %s", "sent" if ok else "not sent (see prior warning)")
    return ok
