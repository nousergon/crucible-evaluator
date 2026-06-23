"""
handler.py — the Director's Lambda entrypoint (Layer C).

Designed to run as the FINAL task of the Saturday SF, immediately after the
``ReportCard`` state (which writes ``evaluator/{date}/report_card.json``). The
Director reads that fresh card, weighs the week's issues, and emits an advisory
``DirectorWeeklyActionPlan`` → ``director/{date}/action_plan.json`` + merges the
carry-over ledger.

**The switch:** gated behind ``DIRECTOR_ENABLED`` (env, default OFF), checked at
request time. OFF → no-op (returns ``status: disabled``), so the Director SF
state can be wired non-fatally now and activated after the foundation validates
on a clean Saturday cycle. NOTE: the Saturday SF invokes this Lambda's ``:live``
alias, whose env is FROZEN at the published version — so flipping a gate env var
needs the operator to update ``$LATEST`` env, publish a new version, AND move the
``live`` alias (no CODE change, but it is a config redeploy — not a bare
``update-function-configuration`` on ``$LATEST``).

Fail-loud on a genuine error (the SF state's own Catch makes it non-fatal — an
advisory failure must never break the run that produced the real trading
artifacts). The Anthropic key + langchain are only needed when the flag is on.

Lambda handler reference: ``director.handler.handler``.
"""

from __future__ import annotations

import json
import logging
import os

import boto3

from director.agent import build_action_plan
from director.carryover import load_ledger, merge_plan_into_ledger, write_ledger
from director.issue_filer import (
    DEFAULT_REPO,
    file_director_issues,
    open_issues_digest,
    recently_closed_proposals_digest,
)
from director.roadmap_pr import TOKEN_SECRET_NAME
from grading.handler import _resolve_run_date

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "alpha-engine-research"

# Operator-facing SSM path for the ROADMAP-PR PAT — used ONLY in log/skip
# messages. Deliberately named without "token"/"secret" so CodeQL's
# clear-text-logging name heuristic doesn't false-positive: this is a parameter
# PATH, never the secret value (the value is fetched via get_secret and never
# logged). `TOKEN_SECRET_NAME` remains the real get_secret lookup key.
_GH_PARAM_HINT = "/alpha-engine/DIRECTOR_GITHUB_TOKEN"


def _enabled() -> bool:
    return os.environ.get("DIRECTOR_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


def _issue_filing_enabled() -> bool:
    """Phase H issue-filing channel (repointed from the ROADMAP-PR channel,
    config#978). **Default ON** — Director proposals land as
    ``area:director-proposals`` issues carrying the Director's suggested ``P#``
    label, which Brian re-triages; triage IS the gate, so there is no soak flag.
    Kill-switch: ``DIRECTOR_ROADMAP_PR_ENABLED`` is
    PRESERVED (its semantics carry over to the new channel — #978); the clearer
    ``DIRECTOR_ISSUE_FILING_ENABLED`` is accepted as an alias. Either set to a
    falsey string disables filing."""
    for name in ("DIRECTOR_ISSUE_FILING_ENABLED", "DIRECTOR_ROADMAP_PR_ENABLED"):
        val = os.environ.get(name)
        if val is not None:
            return val.strip().lower() in ("1", "true", "yes", "on")
    return True


def _director_github_token() -> str | None:
    """Fine-grained PAT for ``alpha-engine-config`` from SSM
    ``/alpha-engine/DIRECTOR_GITHUB_TOKEN``. The repoint needs this token scoped
    ``issues:write`` (the prior ``contents``/``pull_requests`` write scopes filed
    ROADMAP PRs and are no longer used — config#978). ``None`` if unconfigured —
    Phase H then records a skip rather than failing (the plan + ledger are the
    primary deliverables; issue filing is secondary). Mirrors the cyphering
    release-queue token pattern + the fleet's institutional ``get_secret`` path."""
    try:
        from alpha_engine_lib.secrets import get_secret
        tok = (get_secret(TOKEN_SECRET_NAME) or "").strip()
        return tok or None
    except Exception as e:  # noqa: BLE001 — absence is a recorded skip, not fatal
        logger.warning(
            "Director: SSM %s not readable (%s) — issue-filing channel will skip.",
            _GH_PARAM_HINT, e,
        )
        return None


def _fetch_backlog_digest_best_effort(token: str) -> str | None:
    """Phase H input half: read the live OPEN issue backlog and condense to a
    digest so the director won't re-propose tracked work. Repointed from the
    ROADMAP digest now that the backlog lives in GitHub Issues (config#978).
    Best-effort — a fetch failure just means the LLM runs without the dedup
    context (the slug-skip at file time is the second line of defense)."""
    try:
        digest = open_issues_digest(DEFAULT_REPO, token)
        return digest or None
    except Exception as e:  # noqa: BLE001
        logger.warning("Director: backlog digest fetch failed (%s); proceeding without digest.", e)
        return None


def _fetch_resolved_digest_best_effort(token: str, run_date: str) -> str | None:
    """Phase H input half (companion to the open-backlog digest): the
    director-proposals CLOSED in the last ~8 weeks, so the LLM won't re-litigate
    a just-resolved concern under a fresh slug (config#1164 follow-up). The open
    digest only carries OPEN issues and the file-time skip only catches the EXACT
    slug — this closes the semantically-similar-new-slug gap. Best-effort: a
    fetch failure or unparseable run_date just drops the extra dedup context."""
    try:
        digest = recently_closed_proposals_digest(DEFAULT_REPO, token, run_date=run_date)
        return digest or None
    except Exception as e:  # noqa: BLE001
        logger.warning("Director: resolved-proposals digest fetch failed (%s); proceeding without it.", e)
        return None


def _file_issues_best_effort(plan, run_date: str, token: str | None) -> dict:
    """Phase H output half: file the weekly proposals as ``area:director-proposals``
    GitHub issues (Brian triages). Best-effort + fail-loud — the plan is already
    persisted, so a missing token or a GitHub error is WARN-logged AND recorded
    in the returned summary (no silent swallow — [[feedback_no_silent_fails]]: a
    secondary write hung off a primary path that records the failure). NEVER
    fatal: the advisory channel must not break the run that produced the real
    trading artifacts."""
    if not _issue_filing_enabled():
        return {"director_issues": "disabled"}
    if not token:
        logger.warning(
            "Director: issue filing enabled but no token — skipped (set SSM %s, "
            "scoped issues:write).",
            _GH_PARAM_HINT,
        )
        return {"director_issues": "skipped", "director_issues_reason": "no token configured"}
    try:
        res = file_director_issues(plan, run_date, token=token)
        return {
            "director_issues": res.get("status", "ok"),
            "director_issues_n_filed": res.get("n_filed"),
            "director_issues_urls": [i.get("url") for i in res.get("issues", [])],
        }
    except Exception as e:  # noqa: BLE001 — advisory channel; plan already shipped
        logger.warning("Director issue filing failed (plan already written, non-fatal): %s", e)
        return {"director_issues": "error", "director_issues_error": str(e)}


def _load_report_card(s3, bucket: str, run_date: str) -> dict | None:
    from botocore.exceptions import ClientError
    key = f"evaluator/{run_date}/report_card.json"
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        raise
    return json.loads(resp["Body"].read())


RETRO_TREND_KEY = "director/retro_trend.json"


def _load_prior_plan(s3, bucket: str, run_date: str) -> dict | None:
    """The most recent ``director/{date}/action_plan.json`` with date < run_date
    — the plan the Phase-G retro grades against the current card. None on the
    first cycle (no prior plan yet)."""
    paginator = s3.get_paginator("list_objects_v2")
    dates: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix="director/", Delimiter="/"):
        for cp in page.get("CommonPrefixes", []) or []:
            seg = cp["Prefix"].split("/")[1]  # director/{seg}/
            if len(seg) == 10 and seg[4] == "-" and seg < run_date:
                dates.append(seg)
    if not dates:
        return None
    prior = max(dates)
    from botocore.exceptions import ClientError
    try:
        resp = s3.get_object(Bucket=bucket, Key=f"director/{prior}/action_plan.json")
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        raise
    return json.loads(resp["Body"].read())


def _persist_retro(s3, bucket: str, run_date: str, grade) -> str:
    """Write the per-run retro + upsert the trend ledger (dashboard self-grade
    trend). Returns the retro key."""
    from botocore.exceptions import ClientError

    retro_key = f"director/{run_date}/retro.json"
    s3.put_object(
        Bucket=bucket, Key=retro_key,
        Body=grade.model_dump_json(indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    # Append/upsert the trend ledger by prior_run_date (idempotent on re-run).
    try:
        resp = s3.get_object(Bucket=bucket, Key=RETRO_TREND_KEY)
        trend = json.loads(resp["Body"].read())
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            trend = {"grades": []}
        else:
            raise
    row = {"retro_run_date": run_date, **grade.model_dump()}
    grades = [g for g in trend.get("grades", []) if g.get("prior_run_date") != grade.prior_run_date]
    grades.append(row)
    grades.sort(key=lambda g: g.get("prior_run_date", ""))
    trend = {"updated": run_date, "grades": grades}
    s3.put_object(
        Bucket=bucket, Key=RETRO_TREND_KEY,
        Body=json.dumps(trend, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    return retro_key


def _run_retro_best_effort(s3, bucket: str, run_date: str, card: dict) -> dict:
    """Phase-G retro — judge LAST week's plan against THIS week's card. Best-effort:
    the plan (primary deliverable) is already persisted, so a retro failure must
    not fail the Director run. Records the failure via WARN + a summary field
    (no silent swallow — [[feedback_no_silent_fails]]: secondary observability
    hung off a primary path that records the failure)."""
    from director.retro import grade_prior_plan

    prior = _load_prior_plan(s3, bucket, run_date)
    if prior is None:
        return {"retro": "skipped", "retro_reason": "no prior plan (first cycle)"}
    try:
        grade = grade_prior_plan(prior, card)
        retro_key = _persist_retro(s3, bucket, run_date, grade)
        return {
            "retro": "ok",
            "retro_key": retro_key,
            "retro_prior_run_date": grade.prior_run_date,
            "retro_grounding": grade.grounding,
            "retro_calibration": grade.calibration,
            "retro_actionability": grade.actionability,
        }
    except Exception as e:  # noqa: BLE001 — secondary path; the plan already shipped
        logger.warning("Director retro failed (plan already written, non-fatal): %s", e)
        return {"retro": "error", "retro_error": str(e)}


def _dry_run_probe(bucket: str, run_date: str, card: dict | None, s3) -> dict:
    """Preflight probe: exercise the Director's bootstrap/import/IAM surface with
    no Opus call and no S3 write.

    What it validates (the Saturday-fatal-break classes):
      - the ``langchain-anthropic`` lazy import + ``ChatAnthropic`` construction,
      - the SSM ``ANTHROPIC_API_KEY`` fetch (the ``ReadAnthropicSecret`` IAM grant),
      - the carry-over-ledger S3 read + (when a card is present) the digest build.

    The card is normally ABSENT on a preflight (the dry ``ReportCard`` upstream did
    not write one), so a missing card is expected here — we still construct the
    client and read the ledger, then stop short of the digest. Anything that raises
    (broken import, revoked key grant, unreadable ledger) propagates to the SF
    state's non-fatal Catch as a caught preflight failure.
    """
    from director.agent import _default_llm, build_messages

    _default_llm()  # raises on a missing langchain dep or a broken SSM key-fetch grant
    ledger = load_ledger(bucket, s3_client=s3)  # validates ledger read IAM
    digest_built = False
    if card is not None:
        build_messages(card, carryover=ledger)  # exercises the digest path
        digest_built = True

    summary = {
        "status": "dry_run",
        "run_date": run_date,
        "card_present": card is not None,
        "llm_constructed": True,
        "digest_built": digest_built,
        "ledger_size": len(ledger.get("items", [])),
    }
    logger.info("Director preflight probe ok: %s", summary)
    return summary


def handler(event: dict | None = None, context=None) -> dict:
    """Build + persist the weekly Director action plan (flag-gated)."""
    event = event or {}
    if not _enabled():
        logger.info("Director disabled (DIRECTOR_ENABLED off) — no-op.")
        return {"status": "disabled", "reason": "DIRECTOR_ENABLED is off"}

    bucket = event.get("bucket") or os.environ.get("EVALUATOR_BUCKET") or DEFAULT_BUCKET
    run_date = _resolve_run_date(event)
    dry_run = bool(event.get("dry_run", False))
    s3 = boto3.client("s3")

    card = _load_report_card(s3, bucket, run_date)

    if dry_run:
        # Friday-PM Preflight Pipeline (SF passes dry_run=$.research_dry). Exercise
        # the Saturday-fatal-break surface — container boot, the langchain-anthropic
        # lazy import, the SSM ANTHROPIC_API_KEY fetch (validates the
        # ReadAnthropicSecret IAM grant), the S3 reads — but make NO paid Opus call
        # and NO write (no action_plan, no carry-over-ledger mutation: that ledger
        # is shared + non-date-scoped, so a preflight write would pollute the real
        # Saturday run, ROADMAP L4504). Fail-loud (the SF state's Catch is non-fatal):
        # a broken import / revoked key grant surfaces as a caught preflight failure
        # ~18h before the real Saturday Director would hit it.
        return _dry_run_probe(bucket, run_date, card, s3)

    if card is None:
        # The ReportCard state should have produced it; absence is a real gap
        # the SF Catch + freshness monitor surface. Fail loud.
        raise RuntimeError(
            f"Director: no report_card.json at evaluator/{run_date}/ — the ReportCard "
            "state must run before the Director."
        )

    ledger = load_ledger(bucket, s3_client=s3)

    # Phase H token — shared by the backlog digest read (in) and issue filing (out).
    gh_token = _director_github_token()

    # Phase H input half: feed the live open-issue backlog digest to the director
    # so it doesn't re-propose tracked work. An explicit event-supplied digest
    # wins. (kwarg name `roadmap_digest` retained on build_action_plan — the
    # source is now issues, not ROADMAP; config#978.)
    backlog_digest = event.get("roadmap_digest")
    resolved_digest = None
    if backlog_digest is None and gh_token and _issue_filing_enabled():
        backlog_digest = _fetch_backlog_digest_best_effort(gh_token)
        resolved_digest = _fetch_resolved_digest_best_effort(gh_token, run_date)
    plan = build_action_plan(card, run_date=run_date, carryover=ledger,
                             roadmap_digest=backlog_digest, resolved_digest=resolved_digest)

    plan_key = f"director/{run_date}/action_plan.json"
    s3.put_object(
        Bucket=bucket, Key=plan_key,
        Body=plan.model_dump_json(indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    merged = merge_plan_into_ledger(ledger, plan, run_date)
    ledger_key = write_ledger(bucket, merged, s3_client=s3)

    # Phase G — self-grading retro loop. Judge LAST week's plan against THIS
    # week's card (the realized-outcome feedback the in-call SelfGrade can't give).
    # Best-effort: the plan above is the primary deliverable and is already
    # persisted; a retro failure is recorded, never fatal.
    retro_summary = _run_retro_best_effort(s3, bucket, run_date, card)

    # Phase H output half: file the weekly proposals as area:director-proposals
    # issues (Brian triages). Best-effort — the plan above is the primary deliverable.
    issues_summary = _file_issues_best_effort(plan, run_date, gh_token)

    summary = {
        "status": "ok",
        "run_date": run_date,
        "n_action_items": len(plan.action_items),
        "n_top_risks": len(plan.top_risks),
        "action_plan_key": plan_key,
        "ledger_key": ledger_key,
        "ledger_size": len(merged.get("items", [])),
        **retro_summary,
        **issues_summary,
    }
    logger.info("Director plan written: %s", summary)
    return summary


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    """Local manual validation: ``DIRECTOR_ENABLED=1 ANTHROPIC_API_KEY=… \\
    python -m director.handler --date 2026-05-30 [--no-write]``."""
    import argparse

    parser = argparse.ArgumentParser(description="Run the Director against a report card (manual validation).")
    parser.add_argument("--date", default=None)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--no-write", action="store_true",
                        help="Build the plan + print it, but do not write to S3.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.no_write:
        # Build-only path (no S3 writes) for eyeballing plan quality.
        os.environ.setdefault("DIRECTOR_ENABLED", "1")
        s3 = boto3.client("s3")
        run_date = args.date or _resolve_run_date({})
        card = _load_report_card(s3, args.bucket, run_date)
        if card is None:
            print(f"No report_card at evaluator/{run_date}/"); return 1
        plan = build_action_plan(card, run_date=run_date, carryover=load_ledger(args.bucket, s3_client=s3))
        print(plan.model_dump_json(indent=2))
        return 0

    out = handler({"date": args.date, "bucket": args.bucket})
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
