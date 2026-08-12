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
``report_card["attestation"]``, combining both halves. The combined verdict is the
worst of the two: any ``FAIL`` → ``FAIL``; otherwise any ``UNKNOWN`` → ``UNKNOWN``;
only both ``PASS`` → ``PASS``. §2.3a rule 2 — a missing verdict propagates as
``UNKNOWN``, never as a pass, and the report card, the Director digest and the
console all render the state rather than assuming it.

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
from typing import Callable, NamedTuple

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

SCHEMA = "report_card_attestation-1.0.0"
BACKTESTER_ATTESTATION_SCHEMA_PREFIX = "backtest_attestation-"

PASS = "PASS"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"

_VALID_VERDICTS = frozenset({PASS, FAIL, UNKNOWN})

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
    if any(v == FAIL for v in verdicts):
        return FAIL
    if any(v != PASS for v in verdicts):
        return UNKNOWN
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


def read_backtester_attestation(bucket: str, run_date: str, s3_client=None) -> dict:
    """Read ``backtest/{run_date}/attestation.json`` and normalize it.

    Every failure path — absent object, unreadable body, unparseable JSON, an
    unrecognised verdict string, a body stamped with a DIFFERENT ``run_date`` —
    resolves to ``UNKNOWN`` with a reason. §2.3a rule 2.

    The run_date check is rule 1 made concrete: a verdict from an earlier cycle
    says nothing about this cycle's numbers, and inheriting it is exactly how a
    stale pass gets granted by default.
    """
    key = backtester_attestation_key(run_date)
    source_path = f"s3://{bucket}/{key}"
    base = {"source_path": source_path, "run_date": run_date}
    s3 = s3_client or boto3.client("s3")
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        reason = (f"backtester attestation absent at {source_path} — the producer never ran "
                  "this cycle." if code in ("NoSuchKey", "404")
                  else f"backtester attestation unreadable ({code}).")
        logger.warning("attestation: %s", reason)
        return {**base, "verdict": UNKNOWN, "reason": reason}
    except Exception as exc:  # noqa: BLE001 — see CONTRACT: never raises
        reason = f"backtester attestation read failed ({type(exc).__name__}: {exc})."
        logger.error("attestation: %s", reason)
        return {**base, "verdict": UNKNOWN, "reason": reason}

    try:
        doc = json.loads(body)
    except Exception as exc:  # noqa: BLE001
        reason = f"backtester attestation body is not JSON ({type(exc).__name__})."
        logger.error("attestation: %s", reason)
        return {**base, "verdict": UNKNOWN, "reason": reason}

    stamped = doc.get("run_date")
    if stamped != run_date:
        reason = (f"backtester attestation carries run_date {stamped!r}, not {run_date!r} — "
                  "a verdict from another cycle is not inherited.")
        logger.warning("attestation: %s", reason)
        return {**base, "verdict": UNKNOWN, "reason": reason,
                "stamped_run_date": stamped}

    verdict = _normalize_verdict(doc.get("verdict"))
    result = {
        **base,
        "verdict": verdict,
        "schema": doc.get("schema"),
        "n_checks": doc.get("n_checks"),
        "n_failed": doc.get("n_failed"),
        "n_errored": doc.get("n_errored"),
        "engine": doc.get("engine"),
    }
    if verdict == UNKNOWN:
        result["reason"] = (
            f"backtester attestation verdict {doc.get('verdict')!r} is not one of "
            f"{sorted(_VALID_VERDICTS)} — treated as UNKNOWN, never as a pass."
        )
    elif verdict == FAIL:
        failed = [c.get("name") for c in (doc.get("checks") or []) if c.get("passed") is False]
        result["reason"] = (
            f"backtester attestation FAILED on {doc.get('n_failed')} known-answer "
            f"check(s){': ' + ', '.join(str(f) for f in failed) if failed else ''}. "
            "This cycle's backtest numbers are NOT trustworthy."
        )
    else:
        result["reason"] = (
            f"backtester attestation PASS ({doc.get('n_checks')} known-answer checks)."
        )
    return result


def build_run_attestation(bucket: str, run_date: str, s3_client=None) -> dict:
    """The combined block the Report Card carries at ``report_card["attestation"]``.

    Never raises; the worst half wins.
    """
    evaluator = run_evaluator_attestation()
    backtester = read_backtester_attestation(bucket, run_date, s3_client=s3_client)
    verdict = _worst(evaluator["verdict"], backtester["verdict"])
    block = {
        "schema": SCHEMA,
        "run_date": run_date,
        "verdict": verdict,
        "evaluator": evaluator,
        "backtester": backtester,
        "reason": (
            "Both halves attested — the deployed quant primitives and the backtest "
            "engine each agreed with their hand-derived known answers."
            if verdict == PASS else
            f"Correctness guarantee WITHHELD: evaluator={evaluator['verdict']}, "
            f"backtester={backtester['verdict']}. {backtester.get('reason', '')}".strip()
        ),
    }
    if verdict != PASS:
        logger.error("report card attestation %s for %s: %s", verdict, run_date, block["reason"])
    return block
