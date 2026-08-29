"""attestation.py — the Report Card's runtime correctness verdict.

WHY THIS EXISTS
---------------
The Report Card is the surface Brian and the Director read the week's numbers off.
Nothing on it says whether those numbers are *right*.

Two independent gaps produce that silence, and `sf-pipeline-policy.md` §2.3a names
the shape of both: a missing **data artifact** makes a consumer fail visibly, while
a missing **correctness verdict** makes every consumer succeed *as though the check
had passed*.

1. **The evaluator's own arithmetic.** Sharpe, Sortino, max drawdown, CVaR and PSR
   are computed from raw daily rows via ``nousergon_lib.quant.*`` — a shared library
   whose pin moves independently of this repo and is resolved into the Lambda image
   at build time. No test in this repo pins any of those to a value:
   ``test_portfolio_outcome.py``'s Sharpe assertion is ``sharpe > 0``. A changed
   ``ddof``, annualization factor or downside-deviation denominator in the lib would
   move every risk-adjusted tile on the card at once — a coherent, plausible, and
   entirely invisible shift.

2. **The backtester's arithmetic.** The Backtester tile grades coverage and parity
   of ``backtest/{date}/grading.json``, and would grade a run whose simulation
   engine had silently changed exactly as it grades a clean one. The backtester now
   emits its own runtime verdict at ``backtest/{run_date}/attestation.json``
   (``crucible-backtester analysis/attestation.py``); this module is the consumer
   half of that contract.

WHAT IT PRODUCES
----------------
``build_run_attestation()`` returns the block the card carries at
``report_card["attestation"]``, combining the halves. The combined verdict is the
worst: any ``FAIL`` → ``FAIL``; otherwise any ``UNKNOWN`` → ``UNKNOWN``; otherwise
any ``PARTIAL`` → ``PARTIAL``; only all ``PASS`` → ``PASS``. §2.3a rule 2 — a
missing verdict propagates as ``UNKNOWN``, never as a pass, and the report card,
the Director digest and the console all render the state rather than assuming it.

3. **Look-ahead contamination (config#7199).** The three halves above all answer
   *"did we compute this correctly?"* — none of them answers *"was the input
   allowed to see the future?"*. That is a different claim and, for a number
   shown outside the firm, the load-bearing one: an arithmetic error produces a
   wrong number, look-ahead contamination produces a **spectacular** number that
   is entirely fake, and it is the failure mode a backtest cannot self-diagnose
   because the results look better rather than worse. The backtester's
   ``PitParityCompare`` stage emits that verdict at
   ``backtest/{run_date}/pit_parity.json``; this module is its consumer, and the
   block renders ``arithmetic_verdict`` and ``contamination_verdict`` as two
   fields because a reader asks about them as two questions.

   Measured motivation: on 2026-08-07 that check timed out after 2700s, its
   artifact read ``status: failed``, nothing consumed it, and the card was
   written ``status: "ok"`` with grade 55.7.

CONTRACT
--------
Nothing here raises. A verdict-producing stage that dies must not kill stages that
do not depend on it (§2.3a) — so every failure path resolves to ``UNKNOWN`` with the
cause recorded in the returned body and an ERROR log line. That is the one
legitimate broad catch in this module: the swallowed failure mode is "the
attestation itself could not run", the primary deliverable (the report card) is
untouched, and the recording surface is the emitted block plus the log.
"""

from __future__ import annotations

import io
import json
import logging
import math
import platform
import time
from typing import Any, Callable, NamedTuple

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

SCHEMA = "report_card_attestation-1.0.0"
BACKTESTER_ATTESTATION_SCHEMA_PREFIX = "backtest_attestation-"

PASS = "PASS"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"
#: config#7199 — a check that answered honestly over a strict SUBSET of what it
#: was asked to cover. Distinct from UNKNOWN (no answer at all) because the
#: diagnosis and the remedy differ: PARTIAL means the budget bound, UNKNOWN
#: means the stage died. Both withhold the guarantee; neither is a pass.
PARTIAL = "PARTIAL"

_VALID_VERDICTS = frozenset({PASS, FAIL, PARTIAL, UNKNOWN})

#: config-I7620 follow-up — a half whose PRODUCER WAS NOT DISPATCHED THIS RUN.
#:
#: Deliberately OUTSIDE ``_VALID_VERDICTS``: no producer may ever write this
#: value, and :func:`_normalize_verdict` can never yield it. It is assigned by
#: this module alone, on exactly TWO grounds, and each is a fact about DISPATCH
#: rather than about the evidence:
#:
#: 1. the run-scope block — derived from the state machine definition and the
#:    execution history, which cannot disagree with reality — says the stage
#:    that writes the evidence took its skip branch; or
#: 2. this is a declared dry run (``dry_run=True``, the Friday-evening shell
#:    rehearsal), on which every producer is dispatched ``--preflight-only`` and
#:    writes no attestation by design, and the object was genuinely ABSENT
#:    (``alpha-engine-config-I7392`` — see :func:`_mark_rehearsal_out_of_scope`).
#:
#: It is NOT a pass. :func:`verdict_is_pass` still returns False for it, so
#: ``degraded_contamination`` stays True and no surface claims the run was
#: established contamination-free. What it changes is the COMBINED verdict: an
#: out-of-scope half is excluded from :func:`_worst` rather than dragging the
#: whole block to UNKNOWN.
#:
#: Why that distinction is the whole point: ``skip_parity: true`` has been set
#: on the live ``alpha-engine-saturday`` EventBridge target since 2026-08-13 by
#: a recorded ruling (config-I7309, re-enable gated to phase 3). With the
#: producer switched off on purpose, the pre-scope reader resolved contamination
#: to UNKNOWN every single week, the combined verdict to UNKNOWN with it, and
#: the Director — which reads that verdict under sf-pipeline-policy §2.3a —
#: withheld ``issue_filing`` and ``loop_verification`` indefinitely. An operator
#: decision to stop measuring one thing had silently become a decision to stop
#: the weekly advisory from acting on anything, and nothing on any surface said
#: so. "The producer never ran this cycle" is a true sentence about a stage that
#: crashed and a false one about a stage nobody started.
NOT_IN_SCOPE = "NOT_IN_SCOPE"

#: The stages that must ALL run for ``backtest/{date}/pit_parity.json`` to
#: exist: the umbrella gate plus the two passes and the compare that writes the
#: artifact. Named after the ``CheckSkip*`` gates in the weekly SF definition
#: with the prefix stripped, which is exactly the key the run-scope producer
#: emits (``nousergon-data infrastructure/lambdas/weekly-run-scope/run_scope.py``).
#:
#: Any ONE of them disabled means there is no comparison to report, so the
#: contamination half is out of scope — not merely thinner.
CONTAMINATION_PRODUCER_STAGES: tuple[str, ...] = (
    "Parity",
    "PitParityWalkforward",
    "PitParityLookahead",
    "PitParityCompare",
)

_TRADING_DAYS = 252
_RTOL = 1e-12
_ATOL = 1e-15

#: The frozen daily-return series every quant check is derived on. Five
#: observations, chosen so the sample mean, the sample standard deviation, the
#: downside deviation and the 95% loss quantile are all short exact decimals.
FROZEN_RETURNS: tuple[float, ...] = (0.01, -0.02, 0.03, 0.0, 0.01)

#: Frozen NAV level path: peak 120 → trough 90 is the worst peak-to-trough leg,
#: and the later 130 → 104 leg (−20%) is shallower, so the running-peak walk must
#: report −0.25 and not the most recent decline.
FROZEN_NAV: tuple[float, ...] = (100.0, 120.0, 90.0, 130.0, 104.0)

#: A frozen two-row eod_pnl.csv. The `_pct` columns are in PERCENT; the reader
#: divides by 100. Getting that units contract wrong scales every downstream
#: portfolio metric by 100 — the units-suffix class that produced the
#: `avg_volume_20d` scanner incident, one layer up.
FROZEN_EOD_PNL_CSV = (
    "date,portfolio_nav,daily_return_pct,spy_return_pct,daily_alpha_pct\n"
    "2026-01-02,1000000.0,1.0,0.5,0.5\n"
    "2026-01-05,1010000.0,-2.0,-1.0,-1.0\n"
)


def verdict_is_pass(verdict: str | None) -> bool:
    """True only for an explicit ``PASS``.

    §2.3a rule 2. Consumers call this rather than testing truthiness, so the
    "missing reads as pass" bug cannot be written: ``None``, ``""``, ``"ok"`` and
    ``UNKNOWN`` all withhold the guarantee.
    """
    return verdict == PASS


def _normalize_verdict(raw) -> str:
    """Map a producer's verdict field onto the closed vocabulary.

    Anything unrecognised — including a producer that starts writing ``"ok"`` —
    becomes ``UNKNOWN``. A verdict vocabulary that silently accepts new truthy
    strings is not a verdict.
    """
    return raw if raw in _VALID_VERDICTS else UNKNOWN


def _worst(*verdicts: str) -> str:
    """Worst-of over the closed vocabulary: FAIL > UNKNOWN > PARTIAL > PASS.

    PARTIAL sits ABOVE UNKNOWN because it carries real, if incomplete,
    evidence, and UNKNOWN carries none — so a combined verdict of PARTIAL tells
    a reader something a combined UNKNOWN does not. Neither is a pass:
    :func:`verdict_is_pass` tests for ``PASS`` alone, so no ordering choice here
    can grant the guarantee by accident.
    """
    if any(v == FAIL for v in verdicts):
        return FAIL
    if any(v == UNKNOWN or v not in _VALID_VERDICTS for v in verdicts):
        return UNKNOWN
    if any(v == PARTIAL for v in verdicts):
        return PARTIAL
    return PASS


def _environment() -> dict:
    env = {"python": platform.python_version()}
    for mod in ("nousergon_lib", "numpy"):
        try:
            env[mod] = getattr(__import__(mod), "__version__", "<no __version__>")
        except Exception as exc:  # noqa: BLE001 — a version probe never blocks
            env[mod] = f"<unavailable: {type(exc).__name__}>"
    return env


# ════════════════════════════════════════════════════════════════════════════
# PRODUCER HALF — the evaluator's own known-answer battery
# ════════════════════════════════════════════════════════════════════════════

class Check(NamedTuple):
    """One known-answer check.

    ``expected`` is derived from the metric's *definition* using ``math`` alone;
    ``compute`` drives the production path. They are separate — rather than
    ``compute`` returning a bool — so the emitted block carries both numbers and a
    divergence is diagnosable from the artifact alone.
    """

    name: str
    description: str
    expected: float
    compute: Callable[[], float]
    rtol: float = _RTOL
    atol: float = _ATOL


def _expected_sharpe() -> float:
    """(mean / sample-sd) * sqrt(252), sample sd with ddof=1.

    Written out from the definition — no lib call. A lib that switched to a
    population sd (ddof=0) would inflate every Sharpe on the card by
    sqrt(n/(n-1)) with no other symptom.
    """
    r = FROZEN_RETURNS
    n = len(r)
    mean = sum(r) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in r) / (n - 1))
    return (mean / sd) * math.sqrt(_TRADING_DAYS)


def _expected_sortino() -> float:
    """(mean / downside-deviation) * sqrt(252), downside deviation over ALL n
    observations (not only the negative ones) against a zero target."""
    r = FROZEN_RETURNS
    n = len(r)
    mean = sum(r) / n
    dd = math.sqrt(sum(min(0.0, x) ** 2 for x in r) / n)
    return (mean / dd) * math.sqrt(_TRADING_DAYS)


def _expected_cvar_95() -> float:
    """Mean of the losses at or beyond the interpolated 95% loss quantile.

    losses sorted ascending = [-0.03, -0.01, -0.01, 0.0, 0.02];
    rank = 0.95 * (5-1) = 3.8 → VaR = 0.0 + 0.8 * (0.02 - 0.0) = 0.016;
    the tail at-or-beyond 0.016 is [0.02] → CVaR = 0.02.
    """
    return 0.02


def _expected_cumulative_log_alpha() -> float:
    """Σ (log(1+r_p) − log(1+r_spy)) over the frozen pair."""
    port = FROZEN_RETURNS
    spy = tuple(x / 2.0 for x in FROZEN_RETURNS)
    return sum(math.log(1.0 + p) - math.log(1.0 + s) for p, s in zip(port, spy))


class _FrozenS3:
    """Minimal S3 stand-in serving the frozen CSV — lets the attestation drive the
    REAL ``read_eod_pnl`` (units conversion, row filtering, sort) rather than a
    re-implementation of it."""

    def get_object(self, Bucket=None, Key=None):  # noqa: N803 — boto3 kwarg casing
        return {"Body": io.BytesIO(FROZEN_EOD_PNL_CSV.encode())}


def _observed_sharpe() -> float:
    from nousergon_lib.quant.riskstats import sharpe_ratio

    return float(sharpe_ratio(list(FROZEN_RETURNS)))


def _observed_sortino() -> float:
    from nousergon_lib.quant.riskstats import sortino_ratio

    return float(sortino_ratio(list(FROZEN_RETURNS)))


def _observed_max_drawdown() -> float:
    from nousergon_lib.quant.riskstats import max_drawdown

    return float(max_drawdown(list(FROZEN_NAV)))


def _observed_cvar_95() -> float:
    from nousergon_lib.quant.risk_measures import historical_cvar

    return float(historical_cvar(list(FROZEN_RETURNS), confidence=0.95))


def _observed_cumulative_log_alpha() -> float:
    from grading.tiles.portfolio_outcome import cumulative_log_alpha

    port = list(FROZEN_RETURNS)
    spy = [x / 2.0 for x in FROZEN_RETURNS]
    return float(cumulative_log_alpha(port, spy))


def _observed_eod_pnl_first_return() -> float:
    """The first parsed daily portfolio return from the frozen CSV.

    The CSV says ``1.0`` PERCENT; the parsed fraction must be ``0.01``. A reader
    that stopped dividing by 100 would multiply every portfolio metric on Tile 0
    by 100 and still render a complete, plausible-looking tile.
    """
    from grading.tiles.portfolio_outcome import read_eod_pnl

    series = read_eod_pnl("attestation-frozen", s3_client=_FrozenS3())
    return float(series.port[0])


def _EVALUATOR_CHECKS() -> list[Check]:
    """The battery. A callable (not a constant) so a test can substitute it, and so
    no lib import happens at import time of this module."""
    return [
        Check(
            name="sharpe_annualization",
            description="sample-sd (ddof=1) Sharpe on the frozen series, ×√252",
            expected=_expected_sharpe(),
            compute=_observed_sharpe,
        ),
        Check(
            name="sortino_downside_denominator",
            description="downside deviation over all n obs vs a zero target, ×√252",
            expected=_expected_sortino(),
            compute=_observed_sortino,
        ),
        Check(
            name="max_drawdown_running_peak",
            description="worst peak-to-trough on NAV 100→120→90→130→104 = -0.25",
            expected=-0.25,
            compute=_observed_max_drawdown,
        ),
        Check(
            name="cvar_95_tail_mean",
            description="mean loss at/beyond the interpolated 95% loss quantile = 0.02",
            expected=_expected_cvar_95(),
            compute=_observed_cvar_95,
        ),
        Check(
            name="cumulative_log_alpha",
            description="Σ log(1+r_p) − log(1+r_spy) over the frozen pair",
            expected=_expected_cumulative_log_alpha(),
            compute=_observed_cumulative_log_alpha,
        ),
        Check(
            name="eod_pnl_percent_to_fraction",
            description="eod_pnl.csv `_pct` columns are PERCENT: 1.0 parses to 0.01",
            expected=0.01,
            compute=_observed_eod_pnl_first_return,
        ),
    ]


def run_evaluator_attestation() -> dict:
    """Run the evaluator's known-answer battery against the deployed lib. Never raises."""
    started = time.monotonic()
    try:
        checks = _EVALUATOR_CHECKS()
    except Exception as exc:  # noqa: BLE001 — see CONTRACT: this becomes UNKNOWN
        logger.error(
            "attestation: evaluator battery could not be constructed (%s: %s) — "
            "verdict UNKNOWN; the card must withhold the correctness guarantee.",
            type(exc).__name__, exc, exc_info=True,
        )
        return {
            "status": "error", "verdict": UNKNOWN, "checks": [],
            "n_checks": 0, "n_failed": 0, "n_errored": 0,
            "engine": _environment(),
            "error_class": type(exc).__name__, "error_msg": str(exc)[:500],
            "wall_clock_seconds": round(time.monotonic() - started, 4),
        }

    records: list[dict] = []
    for ck in checks:
        record = {
            "name": ck.name, "description": ck.description,
            "expected": ck.expected, "observed": None, "abs_error": None,
            "rtol": ck.rtol, "atol": ck.atol, "passed": False,
        }
        try:
            observed = float(ck.compute())
            band = max(ck.atol, ck.rtol * abs(ck.expected))
            err = abs(observed - ck.expected)
            record.update(observed=observed, abs_error=err, passed=bool(err <= band))
            if not record["passed"]:
                logger.error(
                    "attestation check FAILED: %s expected=%r observed=%r abs_error=%r band=%r",
                    ck.name, ck.expected, observed, err, band,
                )
        except Exception as exc:  # noqa: BLE001 — a check that cannot run is UNKNOWN
            record["errored"] = True
            record["error_class"] = type(exc).__name__
            record["error_msg"] = str(exc)[:500]
            logger.error("attestation check ERRORED: %s (%s: %s)", ck.name,
                         type(exc).__name__, exc, exc_info=True)
        records.append(record)

    # A check that DISAGREED is evidence the numbers moved (FAIL); one that could
    # not RUN is absence of evidence (UNKNOWN). Both withhold the guarantee, but
    # collapsing them would make an environment problem read as a regression.
    n_failed = sum(1 for r in records if not r["passed"] and not r.get("errored"))
    n_errored = sum(1 for r in records if r.get("errored"))
    verdict = FAIL if n_failed else (UNKNOWN if (n_errored or not records) else PASS)
    if verdict != PASS:
        logger.error(
            "evaluator attestation %s — %d failed / %d errored of %d checks.",
            verdict, n_failed, n_errored, len(records),
        )
    return {
        "status": "ok", "verdict": verdict, "checks": records,
        "n_checks": len(records), "n_failed": n_failed, "n_errored": n_errored,
        "engine": _environment(),
        "wall_clock_seconds": round(time.monotonic() - started, 4),
    }


# ════════════════════════════════════════════════════════════════════════════
# CONSUMER HALF — the backtester's verdict
# ════════════════════════════════════════════════════════════════════════════

def backtester_attestation_key(run_date: str) -> str:
    return f"backtest/{run_date}/attestation.json"


def evaluator_stage_attestation_key(run_date: str) -> str:
    """The Evaluator STAGE's verdict — distinct from this Lambda's own battery.

    ``run_evaluator_attestation`` above attests the quant primitives *this image*
    computes the card's risk-adjusted tiles from. This key is the verdict of the
    ``EvaluatorDiagnostics``/``EvaluatorOptimize`` SF stages, which run
    ``crucible-backtester evaluate.py`` on a spot box: the ranking metrics (IC,
    hit rate, calibration) it grades on, plus whether it was permitted to promote
    ``config/executor_params.json`` and ``config/producer_champion.json`` this
    cycle. Two different pieces of arithmetic on two different substrates; the
    card needs both.
    """
    return f"backtest/{run_date}/evaluator_attestation.json"


def _read_verdict_artifact(
    bucket: str,
    key: str,
    run_date: str,
    label: str,
    s3_client=None,
    enrich: "Callable[[dict, dict], dict] | None" = None,
) -> dict:
    """Read a §2.3a verdict artifact and normalize it. Never raises.

    Every failure path — absent object, unreadable body, unparseable JSON, an
    unrecognised verdict string, a body stamped with a DIFFERENT ``run_date`` —
    resolves to ``UNKNOWN`` with a reason (rule 2).

    The run_date check is rule 1 made concrete: a verdict from an earlier cycle
    says nothing about this cycle's numbers, and inheriting it is exactly how a
    stale pass gets granted by default.

    ``as_of`` carries the object's S3 ``LastModified`` so every surface can
    render *when* the verdict was established alongside *what* it said — a
    verdict with no timestamp cannot read as stale, which is the failure mode
    one layer up from reading absence as green.
    """
    source_path = f"s3://{bucket}/{key}"
    base = {"source_path": source_path, "run_date": run_date, "as_of": None}
    s3 = s3_client or boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read()
        last_modified = obj.get("LastModified")
        if last_modified is not None:
            base["as_of"] = last_modified.strftime("%Y-%m-%dT%H:%M:%SZ")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        absent = code in ("NoSuchKey", "404")
        reason = (f"{label} attestation absent at {source_path} — the producer never ran "
                  "this cycle." if absent
                  else f"{label} attestation unreadable ({code}).")
        logger.warning("attestation: %s", reason)
        # `absent` is carried, not merely stated in prose: a half that is
        # UNKNOWN because the OBJECT IS NOT THERE is the only one a declared
        # rehearsal may re-classify as NOT_IN_SCOPE, and a reader deciding that
        # by substring-matching the reason string is a reader that will get it
        # wrong (alpha-engine-config-I7392).
        return {**base, "verdict": UNKNOWN, "reason": reason, "absent": absent}
    except Exception as exc:  # noqa: BLE001 — see CONTRACT: never raises
        reason = f"{label} attestation read failed ({type(exc).__name__}: {exc})."
        logger.error("attestation: %s", reason)
        return {**base, "verdict": UNKNOWN, "reason": reason}

    try:
        doc = json.loads(body)
    except Exception as exc:  # noqa: BLE001
        reason = f"{label} attestation body is not JSON ({type(exc).__name__})."
        logger.error("attestation: %s", reason)
        return {**base, "verdict": UNKNOWN, "reason": reason}

    if not isinstance(doc, dict):
        reason = f"{label} attestation body is not a JSON object."
        logger.error("attestation: %s", reason)
        return {**base, "verdict": UNKNOWN, "reason": reason}

    stamped = doc.get("run_date")
    if stamped != run_date:
        reason = (f"{label} attestation carries run_date {stamped!r}, not {run_date!r} — "
                  "a verdict from another cycle is not inherited.")
        logger.warning("attestation: %s", reason)
        return {**base, "verdict": UNKNOWN, "reason": reason,
                "stamped_run_date": stamped}

    verdict = _normalize_verdict(doc.get("verdict"))
    if enrich is not None:
        # A verdict artifact whose body is not a known-answer battery (the
        # contamination report) supplies its own summary fields and its own
        # reason. The read, the run_date-stamp rule and the UNKNOWN-on-anything
        # posture above are identical, so they are shared rather than copied —
        # a second copy of "absence is never a pass" is a second place it can
        # be got wrong.
        return enrich({**base, "verdict": verdict}, doc)
    result = {
        **base,
        "verdict": verdict,
        "schema": doc.get("schema"),
        "n_checks": doc.get("n_checks"),
        "n_failed": doc.get("n_failed"),
        "n_errored": doc.get("n_errored"),
        "engine": doc.get("engine"),
    }
    # The Evaluator stage's artifact nests its own battery under `own`; surface
    # the counts and the promotion decision so the card can say what was
    # withheld, not merely that something was.
    own = doc.get("own")
    if isinstance(own, dict):
        result["n_checks"] = own.get("n_checks")
        result["n_failed"] = own.get("n_failed")
        result["n_errored"] = own.get("n_errored")
        result["engine"] = own.get("engine")
    if "promotion_withheld" in doc:
        result["promotion_withheld"] = bool(doc.get("promotion_withheld"))

    if verdict == UNKNOWN:
        result["reason"] = (
            f"{label} attestation verdict {doc.get('verdict')!r} is not one of "
            f"{sorted(_VALID_VERDICTS)} — treated as UNKNOWN, never as a pass."
        )
    elif verdict == FAIL:
        checks = (own or doc).get("checks") or []
        failed = [c.get("name") for c in checks if c.get("passed") is False]
        result["reason"] = (
            f"{label} attestation FAILED on {result.get('n_failed')} known-answer "
            f"check(s){': ' + ', '.join(str(f) for f in failed) if failed else ''}. "
            f"This cycle's {label} numbers are NOT trustworthy."
        )
    else:
        result["reason"] = (
            f"{label} attestation PASS ({result.get('n_checks')} known-answer checks)."
        )
    if result.get("promotion_withheld"):
        result["reason"] += (
            " Config and champion promotion were WITHHELD this cycle — the live "
            "executor is still on last cycle's parameters."
        )
    return result


def read_backtester_attestation(bucket: str, run_date: str, s3_client=None) -> dict:
    """Read the simulation engine's verdict at ``backtest/{run_date}/attestation.json``."""
    return _read_verdict_artifact(
        bucket, backtester_attestation_key(run_date), run_date, "backtester",
        s3_client=s3_client,
    )


def read_evaluator_stage_attestation(bucket: str, run_date: str, s3_client=None) -> dict:
    """Read the Evaluator stage's verdict at ``backtest/{run_date}/evaluator_attestation.json``."""
    return _read_verdict_artifact(
        bucket, evaluator_stage_attestation_key(run_date), run_date, "evaluator stage",
        s3_client=s3_client,
    )


def contamination_key(run_date: str) -> str:
    """The look-ahead-contamination verdict at ``backtest/{run_date}/pit_parity.json``.

    Written by ``crucible-backtester analysis/pit_stats_artifact.py::
    run_compare_and_publish`` (the ``PitParityCompare`` SF stage), which reads
    both pit_parity passes and emits ``verdict`` over the same closed
    vocabulary this module normalizes onto.
    """
    return f"backtest/{run_date}/pit_parity.json"


def _enrich_contamination(result: dict, doc: dict) -> dict:
    """Turn a pit_parity report into a contamination half of the attestation.

    The producer already computed the verdict and its reason — this does not
    re-derive them, it surfaces them plus the coverage fraction, which is the
    number that makes a ``PARTIAL`` actionable ("clean over 62% of the window"
    rather than a bare "incomplete").
    """
    coverage = doc.get("coverage") if isinstance(doc.get("coverage"), dict) else {}
    fraction = coverage.get("coverage_fraction")
    try:
        fraction = None if fraction is None else float(fraction)
    except (TypeError, ValueError):
        fraction = None
    result.update({
        "schema": doc.get("schema"),
        "status": doc.get("status"),
        "coverage_fraction": fraction,
        "budget_stopped": bool(coverage.get("budget_stopped", False)),
        "covered_through": coverage.get("covered_through"),
        "material": (doc.get("materiality") or {}).get("material"),
    })
    producer_reason = doc.get("verdict_reason")
    # Only when the raw field was genuinely OUTSIDE the vocabulary. An artifact
    # that says `verdict: "UNKNOWN"` explicitly is honest, not malformed, and
    # describing it as "not one of [FAIL, PARTIAL, PASS, UNKNOWN]" is a false
    # sentence that sends a reader hunting a producer bug that is not there.
    if (
        result["verdict"] == UNKNOWN
        and doc.get("verdict") not in _VALID_VERDICTS
        and not producer_reason
    ):
        # Includes every pit_parity.json written BEFORE config#7199 — those
        # carry no `verdict` key at all, and the absence of the field is
        # exactly the condition §2.3a rule 2 governs.
        producer_reason = (
            f"contamination report at s3://{result['source_path'].split('s3://')[-1]} "
            f"carries verdict {doc.get('verdict')!r}, which is not one of "
            f"{sorted(_VALID_VERDICTS)} — treated as UNKNOWN, never as a pass."
        ) if "source_path" in result else (
            "contamination verdict absent or unrecognised — treated as UNKNOWN."
        )
    result["reason"] = producer_reason or _default_contamination_reason(result)
    return result


def _default_contamination_reason(result: dict) -> str:
    """The reason line when the producer supplied no ``verdict_reason``.

    Derived from the VERDICT, never a constant. A single default string here is
    the "renders measured-X as Y" class this fleet has shipped before: the
    producer's ``verdict_reason`` is an optional field, so a ``FAIL`` written
    without one would otherwise carry the literal sentence "contamination
    verdict PASS." onto the Report Card, the Director digest and the console —
    a FAIL described, in prose, as a pass. The verdict field would still be
    correct; the sentence a human reads would not, and the sentence is the
    surface. Defence in depth: the producer is expected to supply a reason on
    every non-PASS, and this must be safe when it does not.
    """
    verdict = result.get("verdict")
    fraction = result.get("coverage_fraction")
    covered = "" if fraction is None else f" Coverage {fraction:.0%} of the window."
    if verdict == PASS:
        return "contamination verdict PASS — no material look-ahead delta." + covered
    if verdict == FAIL:
        return (
            "contamination verdict FAIL, with no reason supplied by the producer — "
            "this cycle's numbers are NOT established free of look-ahead "
            "contamination." + covered
        )
    if verdict == PARTIAL:
        return (
            "contamination verdict PARTIAL, with no reason supplied by the producer "
            "— the check answered over a strict subset of the window and the "
            "remainder is unmeasured. Not a pass." + covered
        )
    return (
        "contamination verdict UNKNOWN — the check did not answer. Not a pass."
        + covered
    )


def read_contamination_verdict(bucket: str, run_date: str, s3_client=None) -> dict:
    """Read the look-ahead-contamination verdict. Never raises.

    Absent, unparseable, stamped with another cycle's ``run_date``, or carrying
    a verdict outside the closed vocabulary all resolve to ``UNKNOWN`` — the
    same posture as the arithmetic halves, and for the same reason: a
    contamination check that did not answer must not read as one that passed.
    """
    return _read_verdict_artifact(
        bucket, contamination_key(run_date), run_date, "contamination",
        s3_client=s3_client, enrich=_enrich_contamination,
    )


def contamination_scope(run_scope: Any) -> dict:
    """Whether the contamination producer was DISPATCHED this run.

    ``run_scope`` is the raw ``run_scope.json`` body (or the already-normalized
    block — both are accepted, because the card reads it in one place and the
    Director in another and threading two shapes through would be one more way
    to get it wrong).

    Returns ``{"in_scope": bool, "disabled_stages": [...], "disabled_by": [...],
    "reason": str}``. **Absence is never out-of-scope.** A missing or
    unmeasured run-scope artifact leaves the half IN scope, so it resolves to
    UNKNOWN exactly as it did before this function existed — the failure mode
    this guards against is a scope reader that goes quiet and thereby excuses
    every half it can no longer see.
    """
    from grading.run_scope import read_run_scope, scope_unknown

    block = run_scope if isinstance(run_scope, dict) and "graded_stages" in run_scope \
        else read_run_scope(run_scope)
    if scope_unknown(block):
        return {
            "in_scope": True,
            "disabled_stages": [],
            "disabled_by": [],
            "reason": (
                "run scope not established, so the contamination half is read as "
                "in scope — an unmeasured denominator never excuses a missing half."
            ),
        }
    disabled = [s for s in CONTAMINATION_PRODUCER_STAGES
                if s in set(block.get("disabled_stages") or ())]
    if not disabled:
        return {
            "in_scope": True,
            "disabled_stages": [],
            "disabled_by": [],
            "reason": "the contamination producer was dispatched this run.",
        }
    flags = sorted(block.get("disabled_by") or ()) or ["an operator skip flag"]
    return {
        "in_scope": False,
        "disabled_stages": disabled,
        "disabled_by": flags,
        "reason": (
            "contamination NOT IN SCOPE this run — "
            f"{', '.join(disabled)} took the skip branch, disabled by "
            f"{', '.join(flags)}. The look-ahead check was not asked to run, so "
            "its silence is a decision and not an absence of evidence. This is "
            "NOT a pass: the run is not established free of look-ahead "
            "contamination, it is unmeasured for it on purpose "
            "(config-I7309 re-enables it in phase 3)."
        ),
    }


#: The halves whose PRODUCER is dispatched with ``--preflight-only`` on the
#: Friday-evening shell (dry) run and therefore writes no verdict artifact BY
#: DESIGN. ``evaluator`` is deliberately absent from this tuple: it is the
#: in-process known-answer battery over THIS image's quant primitives, it runs
#: identically on a rehearsal, and a rehearsal that could not fail on it would
#: be a rehearsal of nothing.
REHEARSAL_OUT_OF_SCOPE_HALVES: tuple[str, ...] = (
    "backtester", "evaluator_stage", "contamination",
)


#: The one sentence explaining the dry-run ground, held once rather than
#: interpolated per half: three halves go out of scope together on a rehearsal,
#: and three copies of the same paragraph in one ``reason`` is a surface nobody
#: finishes reading.
REHEARSAL_GROUND = (
    "This is a declared DRY RUN (the Friday-evening shell rehearsal), on which "
    "every producer is dispatched --preflight-only and writes no attestation BY "
    "DESIGN, so its silence is a decision and not an absence of evidence. This "
    "is NOT a pass: those numbers are not established correct, they are "
    "unmeasured on purpose (alpha-engine-config-I7392)."
)


def _mark_rehearsal_out_of_scope(half: dict, label: str) -> bool:
    """On a declared dry run, re-classify a half whose artifact is ABSENT.

    ``alpha-engine-config-I7392``. The Friday shell run dispatches every
    producer with ``--preflight-only``; none of them writes its attestation, and
    the pre-fix card read that guaranteed absence as "the producer never ran
    this cycle" and logged ERROR — a structural false alarm, every week, from a
    run in which nothing failed. Measured instance: the ERROR page at
    2026-08-29T01:44Z from execution ``offcycle-shell-20260829-004717``, naming
    ``backtester=UNKNOWN, evaluator_stage=UNKNOWN, contamination=UNKNOWN``.

    **Narrow on purpose, in three ways, because the whole risk here is a
    rehearsal flag that quietly buys a guarantee:**

    1. Only on an explicit ``dry_run``. A real Saturday is untouched.
    2. Only when the half is ``UNKNOWN`` **and** the object was genuinely
       ABSENT. A half that read a real artifact keeps its real verdict — so a
       dry run over a date a real run already wrote still surfaces that run's
       ``FAIL``, and an unreadable/corrupt/mis-stamped body stays UNKNOWN and
       still pages.
    3. ``NOT_IN_SCOPE`` is not a pass. :func:`verdict_is_pass` returns False for
       it, so no surface gains a guarantee it did not earn; what it changes is
       that the half is excluded from the combined worst-of rather than dragging
       it to UNKNOWN.

    Returns True if the half was re-classified.
    """
    if half.get("verdict") != UNKNOWN or not half.get("absent"):
        return False
    half["verdict"] = NOT_IN_SCOPE
    half["reason"] = f"{label} NOT IN SCOPE this run. {REHEARSAL_GROUND}"
    half["out_of_scope_because"] = "dry_run"
    return True


def _worst_in_scope(*verdicts: str) -> str:
    """:func:`_worst` over the halves that were IN SCOPE.

    ``NOT_IN_SCOPE`` is excluded rather than folded in — passing it through
    :func:`_worst` would map it to UNKNOWN, since it is deliberately outside
    ``_VALID_VERDICTS``, and dragging the whole block down is exactly the
    behaviour this vocabulary exists to stop. With nothing left in scope the
    answer is UNKNOWN, never PASS: an empty numerator is an absence of
    evidence. (Unreachable today — the ``evaluator`` half is always in scope —
    and written this way so it stays correct if that ever changes.)
    """
    in_scope = [v for v in verdicts if v != NOT_IN_SCOPE]
    return _worst(*in_scope) if in_scope else UNKNOWN


def build_run_attestation(
    bucket: str, run_date: str, s3_client=None, *, run_scope: Any = None,
    dry_run: bool = False,
) -> dict:
    """The combined block the Report Card carries at ``report_card["attestation"]``.

    Three halves, because three distinct pieces of arithmetic stand between the
    raw data and a number on this card, on three substrates that version
    independently:

    ``evaluator``
        this image's quant primitives (Sharpe, Sortino, drawdown, CVaR, the
        cumulative log-alpha headline) — attested in-process on every build.
    ``backtester``
        the simulation engine's fills, fees, NAV marking and classification
        counts — attested on the spot box that produced the week's numbers.
    ``evaluator_stage``
        the ranking metrics the Evaluator stage grades and promotes on — IC,
        hit rate, calibration — attested on the spot box that ran it, together
        with whether promotion was permitted this cycle.

    ``contamination``
        config#7199 — the look-ahead-contamination verdict from
        ``backtest/{run_date}/pit_parity.json``: whether the point-in-time
        replay of the same window differs materially from the look-ahead one.

    **The last of those is a DIFFERENT CLAIM from the first three, and the block
    renders it as one.** Arithmetic-correct answers "did we compute this
    right"; contamination-free answers "was the input allowed to see the
    future". An arithmetic error produces a wrong number and is embarrassing;
    look-ahead contamination produces a spectacular number that is entirely
    fake, and it is the first thing a competent external reader tests for. So
    the block carries ``arithmetic_verdict`` and ``contamination_verdict``
    separately as well as the combined ``verdict``, because a reader asks about
    them separately and a single boolean cannot answer both.

    Never raises; the worst IN-SCOPE half wins. Any in-scope half UNKNOWN
    withholds the guarantee, because a card whose numbers are only
    three-quarters attested is not a verified card — §2.3a rule 2 admits no
    partial pass.

    ``run_scope`` and ``dry_run`` are the two things that can put a half OUT of
    scope, and neither is a pass — see :data:`NOT_IN_SCOPE`. ``dry_run`` is the
    ReportCard Task's ``$.research_dry`` (the canonical shell-run signal),
    threaded from ``grading/handler.py``; it reaches here and NOT only the
    freshness preflight, which is the half ``alpha-engine-config-I7392`` fixed
    first and the half that left this one paging ERROR every Friday.
    """
    evaluator = run_evaluator_attestation()
    backtester = read_backtester_attestation(bucket, run_date, s3_client=s3_client)
    evaluator_stage = read_evaluator_stage_attestation(bucket, run_date, s3_client=s3_client)
    contamination = read_contamination_verdict(bucket, run_date, s3_client=s3_client)

    # config-I7620 follow-up. The contamination half is the only one whose
    # producer an operator routinely switches off (`skip_parity`), and until the
    # run-scope artifact existed there was no machine-readable way to tell
    # "switched off" from "died". Now there is, so ask it — and only it: this
    # never inspects the SF input or an env var, because the definition and the
    # execution history are the two sources that cannot disagree with reality.
    scope = contamination_scope(run_scope)
    contamination["scope"] = scope
    if not scope["in_scope"]:
        contamination["verdict"] = NOT_IN_SCOPE
        contamination["reason"] = scope["reason"]
        contamination["out_of_scope_because"] = "run_scope"

    # alpha-engine-config-I7392 — the SECOND way a producer can be legitimately
    # silent: not switched off, but dispatched `--preflight-only` by a declared
    # rehearsal. The scope artifact above cannot answer this one; it reports the
    # stage as ENABLED_COMPLETED, which is TRUE (the Backtester stage did run —
    # it ran the preflight) and does not mean an attestation exists. So the
    # rehearsal flag is asked, and only it, and only about halves whose object
    # was genuinely absent. See _mark_rehearsal_out_of_scope for the three ways
    # this is deliberately narrow.
    if dry_run:
        for label, half in (
            ("backtester", backtester),
            ("evaluator stage", evaluator_stage),
            ("contamination", contamination),
        ):
            _mark_rehearsal_out_of_scope(half, label)

    arithmetic_verdict = _worst_in_scope(
        evaluator["verdict"], backtester["verdict"], evaluator_stage["verdict"],
    )
    contamination_verdict = contamination["verdict"]
    # An out-of-scope half is EXCLUDED from the worst-of rather than folded in.
    # Folding it in is what made every week UNKNOWN; passing NOT_IN_SCOPE
    # through `_worst` would do exactly that again, since it is deliberately
    # outside `_VALID_VERDICTS` and `_worst` maps anything it does not
    # recognise to UNKNOWN. The combined verdict answers "is what we MEASURED
    # correct"; `contamination_verdict` continues to answer, separately and
    # honestly, "did we measure contamination at all" — and NOT_IN_SCOPE is
    # never a pass there.
    verdict = _worst_in_scope(arithmetic_verdict, contamination_verdict)

    _halves = (
        ("evaluator", evaluator),
        ("backtester", backtester),
        ("evaluator_stage", evaluator_stage),
        ("contamination", contamination),
    )
    # BOTH SIDES, always (sf-pipeline-policy.md §2.3a rule 4, second obligation):
    # what was withheld AND what was never asked. A list that appears only when
    # something stopped cannot be distinguished from a producer that stopped
    # emitting — principles.md §2.7.
    withheld = [
        f"{name}={half['verdict']}" for name, half in _halves
        if half["verdict"] not in (PASS, NOT_IN_SCOPE)
    ]
    not_in_scope = [name for name, half in _halves
                    if half["verdict"] == NOT_IN_SCOPE]
    # One sentence per distinct GROUND, not per half. Three halves leave scope
    # together on a rehearsal and they leave it for the same reason; repeating
    # it three times is how a surface stops being read.
    _grounds: list[str] = []
    for _name, _half in _halves:
        if _half["verdict"] != NOT_IN_SCOPE:
            continue
        _sentence = (REHEARSAL_GROUND
                     if _half.get("out_of_scope_because") == "dry_run"
                     else _half.get("reason", ""))
        if _sentence and _sentence not in _grounds:
            _grounds.append(_sentence)
    not_in_scope_reasons = " ".join(_grounds).strip()
    reasons = " ".join(
        half.get("reason", "")
        for half in (backtester, evaluator_stage, contamination)
        if half["verdict"] not in (PASS, NOT_IN_SCOPE)
    ).strip()
    block = {
        "schema": SCHEMA,
        "run_date": run_date,
        "verdict": verdict,
        # Two claims, surfaced as two. Do not collapse these into `verdict`
        # alone — an external reader asks "are the numbers right?" and "could
        # they have seen the future?" as separate questions, and only the
        # second is answerable by the contamination half.
        "arithmetic_verdict": arithmetic_verdict,
        "contamination_verdict": contamination_verdict,
        "contamination_coverage_fraction": contamination.get("coverage_fraction"),
        "as_of": {
            "backtester": backtester.get("as_of"),
            "evaluator_stage": evaluator_stage.get("as_of"),
            "contamination": contamination.get("as_of"),
        },
        "evaluator": evaluator,
        "backtester": backtester,
        "evaluator_stage": evaluator_stage,
        "contamination": contamination,
        "contamination_in_scope": contamination_verdict != NOT_IN_SCOPE,
        # The named halves this run never asked for, beside the withheld set —
        # never one without the other (§2.3a rule 4).
        "not_in_scope_halves": not_in_scope,
        "withheld_halves": withheld,
        "dry_run": bool(dry_run),
        "promotion_withheld": bool(evaluator_stage.get("promotion_withheld")),
        # config-I7620 follow-up: an out-of-scope contamination half gets its own
        # sentence. Rendering it under the four-halves-attested line would claim
        # a check that was never asked to run, and rendering it as WITHHELD would
        # repeat the sentence that stopped the Director for three weeks.
        "reason": (
            "All four halves attested — the deployed quant primitives, the backtest "
            "engine, and the Evaluator stage's ranking metrics each agreed with their "
            "hand-derived known answers, and the point-in-time replay found no "
            "material look-ahead contamination over the full window."
            if verdict == PASS and not not_in_scope else
            "Attested over the halves this run dispatched. "
            f"{', '.join(n.upper() + ' NOT MEASURED' for n in not_in_scope)}: "
            f"{not_in_scope_reasons}".strip()
            if verdict == PASS else
            f"Correctness guarantee WITHHELD: {', '.join(withheld)}. {reasons}".strip()
        ),
    }
    # Both polarities, and the SEVERITY is the polarity a pager reads.
    #
    # A non-PASS verdict now means an IN-SCOPE half withheld the guarantee —
    # something the run was asked to measure did not answer — so it stays
    # ERROR, on a real Saturday and on a rehearsal alike: a genuinely dead
    # producer, an unreadable body, or a failing known-answer battery pages
    # either way. What no longer pages is a half nobody asked for
    # (alpha-engine-config-I7392): that is a WARNING, which still renders, still
    # names every out-of-scope half and why, and still is not a pass.
    if verdict != PASS:
        logger.error("report card attestation %s for %s: %s", verdict, run_date, block["reason"])
    elif not_in_scope:
        logger.warning(
            "report card attestation PASS for %s: %s", run_date, block["reason"],
        )
    else:
        logger.info("report card attestation PASS for %s — all four halves attested.",
                    run_date)
    return block
