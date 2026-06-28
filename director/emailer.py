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


def build_director_digest(
    plan: Any, run_date: str, *, console_base_url: str | None = None
) -> tuple[str, str, str]:
    """Build ``(subject, plain_body, html_body)`` for the weekly action plan.

    Thin by design — the full rationale/evidence/carry-over/self-grade stays on
    the console Director page this links to.
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
    subject = (
        f"Alpha Engine Director | {run_date} | {len(items)} action items"
        + (f" ({pri_summary})" if counts else "")
    )

    ordered = sorted(items, key=lambda it: _PRIORITY_ORDER.get(str(it.get("priority")), 9))

    # ── plain body ──
    plain_lines = [
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
    plain_lines += ["", f"Full detail (rationale, evidence, carry-over, self-grade): {url}"]
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
        f"<p style='font-size:14px;margin:0 0 16px;'>&#9654; "
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
        "<p style='font-size:10px;color:#aaa;margin-top:20px;'>"
        "Advisory only — the Director proposes; rationale, evidence, carry-over, "
        f"and self-grade are on the console Director page (<a href=\"{url}\">link</a>).</p>"
        "</body></html>"
    )
    return subject, plain_body, html_body


def send_director_digest(
    plan: Any, run_date: str, *, console_base_url: str | None = None
) -> bool:
    """Build + send the Director digest. Best-effort: returns the send result and
    NEVER raises (transport is the lib's fire-and-forget ``send_email``; the
    build is wrapped so a malformed plan can't break the Director run)."""
    try:
        subject, plain_body, html_body = build_director_digest(
            plan, run_date, console_base_url=console_base_url
        )
    except Exception:  # noqa: BLE001 — the email must never break the Director
        log.warning("Director digest: build failed — skipping email", exc_info=True)
        return False
    from krepis.email_sender import send_email

    ok = send_email(subject, plain_body, html=html_body)
    log.info("Director digest email: %s", "sent" if ok else "not sent (see prior warning)")
    return ok
