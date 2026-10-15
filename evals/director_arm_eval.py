#!/usr/bin/env python3
"""director_arm_eval.py — grade the Director's capability tier as a slot.

`alpha-engine-config-I9486` asked whether `ultra` earns its place on the
Director's plan call, and recorded that **it had never been measured**:
`ultra` is a tier the fleet asserts, not one it has graded. That is precisely
the gap `champion-challenger-policy.md` exists to close — §3, *"an arm that is
not scored is not a challenger, it is a rumour"* — and the absence of an
arm-vs-arm comparison here is the same absence that let a collapsed predictor
champion be promoted on 2026-09-08.

**The slot.** One swappable decision: *which registry capability class serves
`build_action_plan`*. Champion = `ultra`. Challenger = `high`. Both are
addressed as GROUP HANDLES through ``krepis.router`` — this module never names
a model id, a base url, a provider, or constructs an SDK client
(`principles.md` §2.8). If a new arm is added to the registry it is graded here
by adding its group name to ``--arms``; nothing in this file changes.

**Fair comparison** (§4). Every arm sees the byte-identical prompt, built by
``director.agent.build_messages`` from the same archived Report Card and the
same carry-over ledger, through the same schema, the same retry loop and the
same client construction. The only thing that varies is the group. Arms are
paired **per report card** and compared on the intersection of the cards on
which both produced a plan; an arm that failed on a card is a
``schema_valid=0`` observation on that card, never a dropped row — silent
absence and a genuine zero must not render identically (§3).

**What this is NOT.** It is not a promotion. It emits a verdict object with an
explicit ``promote`` field that this module never acts on, because re-pointing
`DIRECTOR_GROUP` is a routing decision reserved to Brian (`I9486` is his). A
thin result is reported as ``inconclusive``, which is a real answer and is
never collapsed into "no difference found".

── THE ACCEPTANCE CRITERION ─────────────────────────────────────────────────

There was none. Nothing in this repo, in `LLM_MODEL_REGISTRY.yaml`, or in
`I9486`/`I9314`/`I9379` states what "the Director is served acceptably"
means in terms a number can express — the tier was chosen in 2026-08-02 by
migrating whatever the previous pinned model was into the nearest group. So it
is DEFINED HERE, and the definition is stated rather than implied
(`principles.md` §2.2: an evaluation with no rejection criterion is not one).

It is derived from what the Director's own contract already says its output
must be, not invented for this measurement:

  * ``director/agent.py`` — *"the model emits the structured plan directly
    (krepis structured-output + Pydantic — no freeform parsing)"*.
  * ``director/schema.py`` — *"every action item must cite the MetricRecord
    names / artifacts it leaned on (``evidence``), so the plan is grounded
    rather than plausible-but-ungrounded"*.
  * ``director/loop_verification.py`` — the plan's citations are the thing the
    loop later reconciles against the card, so a citation that resolves to no
    component is an item the loop can never close.

Four **gates**, each a hard pass/fail on the arm, and one **primary metric**
on which the arms are ranked:

  G1 STRUCTURED     ``schema_valid`` == 1.00 on every card. A plan that comes
                    back as prose is not a degraded plan, it is no plan — the
                    weekly run fails. Registry rows for both `glm-5.2` arms
                    already carry a note about exactly this happening.
  G2 NATIVE RUNG    ``structured_output_rung == "native"`` on every card.
                    ``prompt_only`` is the model-portability §7 ladder's
                    bottom rung; an arm that can only reach it is one provider
                    change away from G1.
  G3 GROUNDING      ``grounding_rate`` >= ``MIN_GROUNDING_RATE`` (0.80) —
                    the fraction of action items citing at least one evidence
                    token that resolves to a real component on the card the
                    plan was built from, using the Director's OWN resolver
                    (``loop_verification.item_cited_metrics``: the item's
                    code-checked typed ``metric:`` citations, or its legacy
                    ``evidence`` prose when it has none), not a second
                    implementation of it.
  G4 RED COVERAGE   ``red_coverage`` >= ``MIN_RED_COVERAGE`` (0.50) — the
                    fraction of the card's RED components addressed by at
                    least one action item. An arm that writes a fluent plan
                    about the green tiles has not read the card.

  PRIMARY METRIC    ``grounding_rate``, paired per card, compared with the
                    fleet's anytime-valid confidence sequence
                    (``nousergon_lib.arena.confseq``, champion-challenger
                    §5.0). A lead is supported only when the whole interval
                    sits off zero.

**The rejection criterion, stated as a rejection:** the challenger is rejected
— `ultra` keeps the slot — if the challenger fails ANY gate on ANY card, or if
the confidence sequence supports the CHAMPION's lead on grounding. It is
accepted only if it passes every gate on every card AND the sequence does not
support a champion lead; cost then decides, because at equal measured quality
a 3.2x input / 5.1x output / 72x cache-read price difference is not a close
call. Anything else is ``inconclusive`` and the champion holds by default —
never by silence.

Cost, latency and measured cache-hit rate are reported for BOTH arms on every
run and are not gates: they are what the decision trades, and `I9486` needs
them side by side.

── WHY THE COMPARISON IS BOUNDED, AND WHAT IT COSTS ─────────────────────────

Real completions. The Director's prompt is ~11-25k prompt tokens and the plan
draws 18-23k completion tokens, so a call is not cheap and a sloppy sweep here
is a real bill. The default is 3 cards x 2 reps x 2 arms = 12 calls, priced
from ``krepis/model_pricing.yaml`` at roughly $0.12/call on `ultra` and
$0.026/call on `high` — under $1 for a full run. ``--dry-run`` prices the plan
and makes no calls; ``--max-calls`` is a hard stop that raises rather than
truncating silently.

Usage:

    python3 -m evals.director_arm_eval --dry-run
    python3 -m evals.director_arm_eval --cards 2026-08-21,2026-08-28,2026-09-04 \
        --arms ultra,high --reps 2 --out /tmp/arm_eval.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)

RESEARCH_BUCKET = os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")

#: The slot's champion — the group `director.agent.DIRECTOR_GROUP` names
#: today. Read from the module rather than restated, so this harness cannot
#: drift into grading a champion that is no longer serving.
def champion_group() -> str:
    from director.agent import DIRECTOR_GROUP

    return DIRECTOR_GROUP


#: Gate thresholds. Stated as named constants, with the reasoning, because a
#: gate whose number lives inline in a comparison is a number nobody can find
#: when it is time to argue about it.
#:
#: 0.80, not 1.00: an action item may legitimately cite an artifact path or a
#: cross-repo issue rather than a card component ("s3://.../signals.json",
#: "alpha-engine-config-I9486"), and `resolve_cited_metrics` resolves card
#: components only. Measured against the live 2026-08-22 ledger,
#: `loop_verification`'s own docstring records 4 of 28 rows unresolvable after
#: the token-resolution fix — 86%. 0.80 sits below that with room for one
#: legitimately off-card item in a five-item plan, and far above the failure
#: worth rejecting an arm for: a model that writes plausible prose citing
#: metric names that do not exist lands near zero.
MIN_GROUNDING_RATE = 0.80

#: 0.50, not 1.00: the card carries components a weekly plan can correctly
#: decline to action (a RED tile already carried as a P0 in the ledger, a
#: component whose owner is `operator`). Requiring full coverage would reward
#: an arm that emits one shallow item per red tile over one that concentrates.
#: Below half, the arm is not reading the card it was handed.
MIN_RED_COVERAGE = 0.50

#: `nousergon_lib.arena.confseq` needs a declared per-observation clip to make
#: the interval's validity checkable from configuration alone (§5.0). The
#: primary metric is a RATE, so the paired difference is bounded on [-1, 1] by
#: construction and the clip is that bound — declared, not estimated.
GROUNDING_DIFF_CLIP = 1.0

#: Statuses that make a component a thing the plan ought to be about. Imported
#: from the Director rather than restated — a second copy would silently stop
#: agreeing the first time the card's status vocabulary moves.
def _adverse_statuses() -> set:
    from director.loop_verification import ADVERSE_STATUSES

    return set(ADVERSE_STATUSES)


# ── Pricing ──────────────────────────────────────────────────────────────────
#
# Priced from `krepis/src/krepis/model_pricing.yaml`, the fleet's single price
# table, looked up by the model the provider REPORTED serving
# (`result.model`), never by the group we asked for. Those differ exactly when
# a fallback served, which is the case where a cost figure keyed on the
# request would be wrong and would look right.


def _pricing_cards() -> list:
    import yaml
    from importlib import resources

    with resources.files("krepis").joinpath("model_pricing.yaml").open() as fh:
        return (yaml.safe_load(fh) or {}).get("cards", []) or []


def price_call(model: str, usage, cards: list | None = None) -> dict:
    """USD for one call, split into the three token classes.

    Returns ``{"input", "cache_read", "output", "total", "priced"}``.
    ``priced`` is False when no card matches the served model — reported, not
    swallowed into a $0.00 that would read as a free call
    (`principles.md` §2.7).
    """
    cards = _pricing_cards() if cards is None else cards

    def _bare(name: str) -> str:
        # The price table carries both `deepseek/deepseek-v4-pro` (the
        # OpenRouter id) and `deepseek-v4-pro` (the direct id) for the same
        # model. Match on the segment after the last "/" so the two forms
        # resolve to the same price, and match EXACTLY on it — a substring
        # match would price `deepseek-v4-flash` off the `deepseek-v4-pro`
        # card, which is a 3x error that would look like a plausible number.
        return str(name).rsplit("/", 1)[-1].strip()

    want = _bare(model)
    card = next((c for c in cards if want and _bare(c.get("model_name", "")) == want),
                None)
    out = {"input": 0.0, "cache_read": 0.0, "output": 0.0, "total": 0.0,
           "priced": card is not None}
    if card is None:
        return out
    cache_read = int(getattr(usage, "cache_read_tokens", 0) or 0)
    inp = max(int(getattr(usage, "input_tokens", 0) or 0) - cache_read, 0)
    outp = int(getattr(usage, "output_tokens", 0) or 0)
    out["input"] = inp / 1e6 * float(card.get("input_per_1m") or 0.0)
    out["cache_read"] = cache_read / 1e6 * float(card.get("cache_read_per_1m") or 0.0)
    out["output"] = outp / 1e6 * float(card.get("output_per_1m") or 0.0)
    out["total"] = out["input"] + out["cache_read"] + out["output"]
    return out


def cache_hit_rate(usage) -> "float | None":
    """Share of this call's PROMPT tokens served from cache, or ``None``.

    Always ``cache_read / input_tokens``. Deliberately NOT
    ``read / (read + miss)``, which was this function's first form and which
    was wrong in the one case that matters: krepis reports
    ``prompt_cache_miss_tokens`` as 0 for a provider that does not publish the
    field, so the ratio collapsed to ``read/read`` and printed **1.00 for a
    call whose measured cache coverage was 49.8%**. A denominator that can
    silently become the numerator is not a rate — and reporting 100% cache
    coverage on the arm under evaluation is exactly the kind of flattering
    error a comparison must not be able to make.

    ``input_tokens`` is the whole prompt and is always reported, so the
    denominator is the real quantity in every case rather than one that is
    only sometimes present.

    ``None`` means the provider reported no prompt tokens at all — the call
    did not get far enough to have a cache rate. That is NOT zero, and
    rendering it as zero is how "no measurement" becomes indistinguishable
    from "the cache is dead".
    """
    total = int(getattr(usage, "input_tokens", 0) or 0)
    if total <= 0:
        return None
    read = int(getattr(usage, "cache_read_tokens", 0) or 0)
    return min(read / total, 1.0)


# ── Scoring: pure functions over (plan, card) ────────────────────────────────


@dataclass
class PlanScore:
    """Deterministic quality reading of one plan against the card it saw."""

    schema_valid: int = 0
    structured_rung: str = ""
    native_rung: int = 0
    action_items: int = 0
    grounded_items: int = 0
    grounding_rate: float = 0.0
    adverse_targeting_rate: float = 0.0
    red_components: int = 0
    red_covered: int = 0
    red_coverage: float = 0.0
    carryover_reviewed: int = 0
    failure: str = ""


def _plan_texts(plan) -> list:
    """Every string an action item formally staked its claim on.

    ``evidence`` ONLY, deliberately, and for the same reason
    ``loop_verification.evidence_still_adverse`` reads only ``evidence``: the
    rationale is prose that may mention a metric in passing, and counting it
    would grade an arm on how many component names it sprinkled through a
    paragraph rather than on what it actually cited.
    """
    return [e for item in getattr(plan, "action_items", []) or []
            for e in (getattr(item, "evidence", None) or [])]


def score_plan(plan, card: dict, *, rung: str = "") -> PlanScore:
    """Grade one plan against the report card it was built from. Pure."""
    from director.loop_verification import component_status_map, item_cited_metrics

    status_map = component_status_map(card or {})
    adverse = _adverse_statuses()
    s = PlanScore(schema_valid=1, structured_rung=rung,
                  native_rung=1 if rung == "native" else 0)

    items = list(getattr(plan, "action_items", []) or [])
    s.action_items = len(items)
    covered: set = set()
    grounded = 0
    adverse_hits = 0
    for item in items:
        row = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        hits = item_cited_metrics(row, status_map)
        if hits:
            grounded += 1
            covered |= set(hits)
        if any(st in adverse for st in hits.values()):
            adverse_hits += 1
    s.grounded_items = grounded
    # An EMPTY plan is not a perfect one. 0/0 is scored 0.0, not 1.0: an arm
    # that emitted no action items grounded nothing, and a rate that read 100%
    # there would rank the emptiest arm first.
    s.grounding_rate = (grounded / len(items)) if items else 0.0
    s.adverse_targeting_rate = (adverse_hits / len(items)) if items else 0.0

    reds = {c for c, st in status_map.items() if st == "RED"}
    s.red_components = len(reds)
    s.red_covered = len(reds & covered)
    # No RED components on the card is `red_coverage = 1.0` — the arm covered
    # everything there was to cover. Distinguished from the empty-plan case
    # above because the vacuity is in the CARD, not in the arm's output, and
    # penalising an arm for a healthy week would make the gate a function of
    # the fleet's mood.
    s.red_coverage = (s.red_covered / len(reds)) if reds else 1.0
    s.carryover_reviewed = len(getattr(plan, "carryover_review", None) or [])
    return s


def failed_score(reason: str) -> PlanScore:
    """The score of a call that produced no plan. A recorded zero, not a gap."""
    return PlanScore(schema_valid=0, failure=reason)


# ── Gates ────────────────────────────────────────────────────────────────────


def gate_verdicts(scores: list) -> dict:
    """Apply G1-G4 to one arm's scores across all cards.

    Every gate reports its verdict, including the ones that passed — a gate
    list containing only failures cannot be audited (champion-challenger §6).
    """
    if not scores:
        return {"G1_structured": "unmeasurable", "G2_native_rung": "unmeasurable",
                "G3_grounding": "unmeasurable", "G4_red_coverage": "unmeasurable",
                "passed": False, "reason": "no observations"}
    valid = [s for s in scores if s.schema_valid]
    g1 = all(s.schema_valid for s in scores)
    g2 = bool(valid) and all(s.native_rung for s in valid)
    g3 = bool(valid) and all(s.grounding_rate >= MIN_GROUNDING_RATE for s in valid)
    g4 = bool(valid) and all(s.red_coverage >= MIN_RED_COVERAGE for s in valid)
    out = {
        "G1_structured": "pass" if g1 else "FAIL",
        "G2_native_rung": "pass" if g2 else "FAIL",
        "G3_grounding": "pass" if g3 else "FAIL",
        "G4_red_coverage": "pass" if g4 else "FAIL",
    }
    out["passed"] = g1 and g2 and g3 and g4
    out["reason"] = "" if out["passed"] else "; ".join(
        f"{k}={v}" for k, v in out.items() if v == "FAIL")
    return out


# ── The comparison ───────────────────────────────────────────────────────────


def pair_on_cards(champ: dict, chal: dict) -> list:
    """Per-card ``(card, champion_rate, challenger_rate)`` on the common window.

    The intersection of the cards on which BOTH arms were run, paired card by
    card (champion-challenger §4). Two arms compared on windows that barely
    overlap is the specific defect that rule exists to stop.
    """
    common = sorted(set(champ) & set(chal))
    return [(c, champ[c], chal[c]) for c in common]


def compare_arms(champion: str, challenger: str, per_card: dict,
                 gates: dict, alpha: float = 0.05) -> dict:
    """Verdict on one challenger against the champion. Pure.

    ``per_card`` maps arm -> {card -> mean grounding_rate over reps}.
    """
    from nousergon_lib.arena.confseq import confidence_sequence

    pairs = pair_on_cards(per_card.get(champion, {}), per_card.get(challenger, {}))
    out = {
        "champion": champion,
        "challenger": challenger,
        "paired_cards": [c for c, _, _ in pairs],
        "n_paired": len(pairs),
        "champion_mean": None,
        "challenger_mean": None,
        "mean_diff": None,
        "interval": None,
        "supported": None,
        "gates": gates,
        "verdict": "inconclusive",
        "promote": False,
        "reason": "",
    }
    if not pairs:
        out["verdict"] = "unmeasurable"
        out["reason"] = (
            f"no report card on which both {champion!r} and {challenger!r} "
            f"produced a scored plan — an empty intersection is unmeasurable, "
            f"never a tie")
        return out

    champ_vals = [a for _, a, _ in pairs]
    chal_vals = [b for _, _, b in pairs]
    diffs = [b - a for _, a, b in pairs]  # challenger minus champion
    out["champion_mean"] = sum(champ_vals) / len(champ_vals)
    out["challenger_mean"] = sum(chal_vals) / len(chal_vals)
    out["mean_diff"] = sum(diffs) / len(diffs)

    bound = confidence_sequence(diffs, alpha=alpha, clip=GROUNDING_DIFF_CLIP)
    lo = getattr(bound, "lower", None)
    hi = getattr(bound, "upper", None)
    out["interval"] = [lo, hi]
    if lo is not None and hi is not None:
        if lo > 0:
            out["supported"] = "challenger"
        elif hi < 0:
            out["supported"] = "champion"
        else:
            out["supported"] = "neither"

    chal_gates = gates.get(challenger, {})
    champ_gates = gates.get(champion, {})
    if not chal_gates.get("passed"):
        out["verdict"] = "rejected"
        out["reason"] = (
            f"{challenger!r} failed an acceptance gate: {chal_gates.get('reason')}. "
            f"Gates are hard — an arm that cannot produce a structured, grounded "
            f"plan on every card is not a candidate however it scores.")
    elif out["supported"] == "champion":
        out["verdict"] = "rejected"
        out["reason"] = (
            f"the anytime-valid interval on the paired grounding difference "
            f"[{lo:.4f}, {hi:.4f}] lies entirely below zero — the champion's "
            f"lead is supported.")
    elif out["supported"] == "neither":
        out["verdict"] = "no-quality-difference-supported"
        out["reason"] = (
            f"{challenger!r} clears every acceptance gate and the interval "
            f"[{lo:.4f}, {hi:.4f}] straddles zero, so neither arm's lead is "
            f"supported on {len(pairs)} paired cards. The quality question is "
            f"answered 'not distinguishable at this evidence'; cost is then the "
            f"deciding axis, and that is a RULING, not this harness's call.")
    else:
        out["verdict"] = "challenger-leads"
        out["reason"] = (
            f"{challenger!r} clears every gate and the interval "
            f"[{lo:.4f}, {hi:.4f}] lies entirely above zero.")
    if not champ_gates.get("passed"):
        out["reason"] += (
            f" NOTE: the CHAMPION also failed a gate ({champ_gates.get('reason')}) "
            f"— a finding about the incumbent, independent of this comparison.")
    # `promote` is emitted and never acted on. Re-pointing DIRECTOR_GROUP is a
    # routing decision reserved to Brian (alpha-engine-config-I9486 is his
    # issue); this field is the harness's recommendation, in a machine-readable
    # place, so a later ruling can be checked against what was measured.
    out["promote"] = out["verdict"] == "challenger-leads"
    return out


# ── Running it ───────────────────────────────────────────────────────────────


@dataclass
class CallRecord:
    arm: str
    card: str
    rep: int
    ok: bool = False
    served_model: str = ""
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    cache_hit_rate: "float | None" = None
    cost_usd: float = 0.0
    cost_priced: bool = True
    structured_rung: str = ""
    error: str = ""
    score: dict = field(default_factory=dict)


def load_card(date: str, bucket: str = RESEARCH_BUCKET, s3_client=None) -> dict:
    """Read one archived Report Card v2 from the research bucket.

    Read-only, against an ARTIFACT bucket, so it is legal from the laptop —
    unlike `alpha-engine-data` (ArcticDB), which denies even `ne-admin` from
    here (`alpha-engine-config-I9771`).
    """
    import boto3

    s3 = s3_client or boto3.client("s3")
    key = f"evaluator/{date}/report_card.json"
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(body)


def load_carryover(bucket: str = RESEARCH_BUCKET, s3_client=None) -> dict:
    from director.carryover import load_ledger

    return load_ledger(bucket, s3_client=s3_client)


def run_one(arm: str, card_date: str, card: dict, carryover: dict, rep: int) -> CallRecord:
    """One real plan call on one arm. Never raises — a failure is a RECORD."""
    from director.agent import _default_llm, build_action_plan

    rec = CallRecord(arm=arm, card=card_date, rep=rep)
    t0 = time.time()
    try:
        # The client is built HERE, by the Director's own factory, so the arm
        # is addressed the one legal way (a group handle through
        # `krepis.router`) and the harness can read the usage and the
        # structured-output rung the factory's adapter records. `build_action_plan`
        # is then handed the same client every production call gets.
        #
        # Its own `callsite_id`, so an evaluation's spend never lands on the
        # production plan call's cost history.
        llm = _default_llm(group=arm, callsite_id="director-plan-arm-eval")
        plan = build_action_plan(
            card, run_date=card_date, carryover=carryover, llm=llm,
        )
    except Exception as exc:  # noqa: BLE001 — a failed arm is an OBSERVATION
        rec.latency_s = time.time() - t0
        rec.error = f"{type(exc).__name__}: {exc}"
        rec.score = asdict(failed_score(rec.error))
        logger.warning("arm=%s card=%s rep=%s FAILED: %s", arm, card_date, rep, rec.error)
        return rec
    rec.latency_s = time.time() - t0
    rec.ok = True
    rec.served_model = getattr(plan, "resolved_model", "") or ""
    rung = getattr(llm, "last_structured_rung", None) or ""
    rec.structured_rung = rung
    usage = getattr(llm, "last_usage", None)
    if usage is not None:
        rec.input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        rec.output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        rec.cache_read_tokens = int(getattr(usage, "cache_read_tokens", 0) or 0)
        rec.prompt_cache_miss_tokens = int(
            getattr(usage, "prompt_cache_miss_tokens", 0) or 0)
        rec.cache_hit_rate = cache_hit_rate(usage)
        priced = price_call(rec.served_model, usage)
        rec.cost_usd = priced["total"]
        rec.cost_priced = priced["priced"]
    rec.score = asdict(score_plan(plan, card, rung=rung))
    logger.info("arm=%s card=%s rep=%s ok served=%s %.1fs $%.4f grounding=%.2f",
                arm, card_date, rep, rec.served_model, rec.latency_s,
                rec.cost_usd, rec.score.get("grounding_rate", 0.0))
    return rec


def estimate_cost(arms: list, cards: list, reps: int) -> dict:
    """What the run will cost, BEFORE it runs. Priced at the measured shape of
    a Director call: ~20k prompt tokens in, ~20k completion tokens out, zero
    cache credit assumed (the pessimistic end, and on `ultra` the realistic one
    — I9486 measures its cache at 40-70%)."""
    cards_tbl = _pricing_cards()
    per_arm = {}
    from krepis.router import resolve_group_spec

    for arm in arms:
        try:
            _spec, route = resolve_group_spec(
                arm, exec_context=os.environ.get("KREPIS_EXEC_CONTEXT", "laptop"),
                wire="openai", requires=("streaming",))
            # `route["primary_model"]` is the BILLABLE UPSTREAM id
            # (`glm-5.2`), which is what the price table is keyed on.
            # `spec.model` is the deployment id (`ultra-glm-5.2-direct`) and
            # matches no card — pricing off it silently returns $0.00, which
            # is a free-call claim, not a missing-price one.
            model = route.get("primary_model") or _spec.model
        except Exception as exc:  # noqa: BLE001
            per_arm[arm] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        usage = type("U", (), {"input_tokens": 20_000, "output_tokens": 20_000,
                               "cache_read_tokens": 0,
                               "prompt_cache_miss_tokens": 0})()
        p = price_call(model, usage, cards_tbl)
        n = len(cards) * reps
        per_arm[arm] = {"primary_model": model, "calls": n,
                        "usd_per_call": round(p["total"], 4),
                        "usd_total": round(p["total"] * n, 4),
                        "priced": p["priced"]}
    total = sum(v.get("usd_total", 0.0) for v in per_arm.values())
    unpriced = sorted(a for a, v in per_arm.items() if not v.get("priced", False))
    if unpriced:
        # A $0.00 estimate for an arm the price table does not know is a lie
        # that reads as good news. Refuse to run rather than proceed on it.
        raise SystemExit(
            f"no price card in krepis/model_pricing.yaml for the primary of "
            f"{unpriced} — refusing to spend against an estimate of $0.00. "
            f"Add the card, or name an arm that has one.")
    return {"per_arm": per_arm, "calls": len(arms) * len(cards) * reps,
            "usd_total_estimate": round(total, 4)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cards", default="2026-08-21,2026-08-28,2026-09-04",
                    help="Comma-separated report-card run_dates (evaluator/{date}/report_card.json).")
    ap.add_argument("--arms", default="",
                    help="Comma-separated registry GROUP handles. Default: the "
                         "champion (director.agent.DIRECTOR_GROUP) plus 'high'.")
    ap.add_argument("--reps", type=int, default=2,
                    help="Independent repeats per (arm, card). >1 measures the "
                         "arm's own run-to-run spread, which a single draw cannot.")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-calls", type=int, default=24,
                    help="Hard stop. Exceeding it RAISES rather than truncating.")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="")
    ap.add_argument("--dry-run", action="store_true",
                    help="Price the run and resolve every arm; make no calls.")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cards = [c.strip() for c in args.cards.split(",") if c.strip()]
    champ = champion_group()
    arms = ([a.strip() for a in args.arms.split(",") if a.strip()]
            or [champ, "high"])
    if champ not in arms:
        raise SystemExit(
            f"the champion group {champ!r} is not in --arms {arms} — "
            f"champion-challenger §3: a leaderboard whose champion field is "
            f"null is a broken leaderboard, not one with a vacancy.")

    planned = len(arms) * len(cards) * args.reps
    if planned > args.max_calls:
        raise SystemExit(
            f"{planned} calls exceeds --max-calls {args.max_calls}. Raise the "
            f"limit deliberately; this is real money.")
    est = estimate_cost(arms, cards, args.reps)
    print(json.dumps({"plan": {"arms": arms, "cards": cards, "reps": args.reps,
                               "estimate": est}}, indent=2))
    if args.dry_run:
        return 0

    carryover = load_carryover()
    loaded = {d: load_card(d) for d in cards}

    jobs = [(arm, d, loaded[d], carryover, r)
            for arm in arms for d in cards for r in range(args.reps)]
    records: list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for rec in ex.map(lambda j: run_one(*j), jobs):
            records.append(rec)

    # Fold to per-arm, per-card mean grounding, and per-arm score lists.
    per_card: dict = {}
    scores: dict = {}
    for rec in records:
        s = PlanScore(**rec.score) if rec.score else failed_score("no score")
        scores.setdefault(rec.arm, []).append(s)
        per_card.setdefault(rec.arm, {}).setdefault(rec.card, []).append(s.grounding_rate)
    per_card = {a: {c: sum(v) / len(v) for c, v in cd.items()} for a, cd in per_card.items()}
    gates = {a: gate_verdicts(v) for a, v in scores.items()}

    comparisons = [compare_arms(champ, a, per_card, gates, alpha=args.alpha)
                   for a in arms if a != champ]

    def _agg(arm: str, fn, attr: str):
        vals = [getattr(r, attr) for r in records if r.arm == arm and r.ok
                and getattr(r, attr) is not None]
        return fn(vals) if vals else None

    summary = {}
    for arm in arms:
        arm_recs = [r for r in records if r.arm == arm]
        ok = [r for r in arm_recs if r.ok]
        summary[arm] = {
            "calls": len(arm_recs),
            "ok": len(ok),
            "served_models": sorted({r.served_model for r in ok if r.served_model}),
            "mean_latency_s": _agg(arm, lambda v: sum(v) / len(v), "latency_s"),
            "mean_cost_usd": _agg(arm, lambda v: sum(v) / len(v), "cost_usd"),
            "total_cost_usd": round(sum(r.cost_usd for r in arm_recs), 4),
            "mean_cache_hit_rate": _agg(arm, lambda v: sum(v) / len(v), "cache_hit_rate"),
            "mean_input_tokens": _agg(arm, lambda v: sum(v) / len(v), "input_tokens"),
            "mean_output_tokens": _agg(arm, lambda v: sum(v) / len(v), "output_tokens"),
            "mean_grounding_rate": (
                sum(s.grounding_rate for s in scores[arm]) / len(scores[arm])
                if scores.get(arm) else None),
            "mean_red_coverage": (
                sum(s.red_coverage for s in scores[arm]) / len(scores[arm])
                if scores.get(arm) else None),
            "mean_action_items": (
                sum(s.action_items for s in scores[arm]) / len(scores[arm])
                if scores.get(arm) else None),
            "gates": gates.get(arm),
        }

    result = {
        "slot": "director-plan-capability-tier",
        "champion": champ,
        "cards": cards,
        "reps": args.reps,
        "acceptance_criterion": {
            "defined_by": "evals/director_arm_eval.py (alpha-engine-config-I9486)",
            "gates": {
                "G1_structured": "schema_valid == 1.0 on every card",
                "G2_native_rung": "structured_output_rung == 'native' on every card",
                "G3_grounding": f"grounding_rate >= {MIN_GROUNDING_RATE}",
                "G4_red_coverage": f"red_coverage >= {MIN_RED_COVERAGE}",
            },
            "primary_metric": "grounding_rate, paired per card, anytime-valid "
                              "confidence sequence (champion-challenger §5.0)",
        },
        "summary": summary,
        "comparisons": comparisons,
        "actual_spend_usd": round(sum(r.cost_usd for r in records), 4),
        "estimate": est,
        "records": [asdict(r) for r in records],
    }
    text = json.dumps(result, indent=2, default=str)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
