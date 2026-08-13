"""self_test.py — the evaluator's published known-answer SELF-TEST.

WHY THIS EXISTS
---------------
Brian, 2026-08-13: *"is there a way we can input certain params into backtester
and evaluator to confirm they are even computing correctly and include outputs of
this test?"*

CI proves the code is correct **on a runner**. It does not prove the **deployed
instrument** is. Every risk number on the Report Card — Sharpe, Sortino, Calmar,
CVaR(95), max drawdown — comes from ``nousergon_lib.quant.*``, a shared library
whose pin moves independently of this repo and is resolved into the Lambda image
at **build** time. A changed ``ddof``, annualization factor or downside-deviation
denominator would move every risk-adjusted tile at once — coherently, plausibly,
and entirely invisibly, with CI green throughout.

So this battery runs **inside the grading Lambda, on the image's own
site-packages**, drives the **production** tile builder
(``grading.tiles.portfolio_outcome.build_portfolio_outcome_tile``) over a frozen
in-memory ``eod_pnl.csv``, and publishes its inputs, expectations, observations
and per-case verdicts to ``evaluator/{run_date}/self_test.json``. **The library
versions in the header are the point** — they are what makes this an instrument
check rather than a code check.

WHAT IT ASSERTS, AND WHY IT DRIVES THE TILE RATHER THAN THE LIB
----------------------------------------------------------------
Each case runs the same code path the weekly card runs — the CSV parse, the
percent-to-fraction conversion, the NAV-vs-returns column split, the lib call,
and (for Calmar) the annualization arithmetic that lives in the tile itself, not
in any library. Calling the lib functions directly would have left the tile's own
arithmetic and its unit conversions untested, which is where the
``avg_volume_20d`` class of bug lives.

The frozen series is 100 daily rows: 60 at **+1.00%** followed by 40 at
**-1.00%**, SPY flat, NAV compounded from those returns off $1,000,000. Every
expectation below is derived **on paper from the metric's definition** and
recomputed here in plain arithmetic — never by calling the code under test, which
would agree with whatever that code ever does.

``sharpe_closed_form``
    mean = (60(0.01) + 40(-0.01))/100 = 0.002; sample variance (ddof=1) =
    [60(0.008)^2 + 40(0.012)^2]/99 = 0.0096/99; Sharpe = mean/sd x sqrt(252).

``sortino_closed_form``
    Downside deviation is the RMS of ``min(0, r)`` over **all N** observations
    (not over the count of negatives): dd = sqrt(40(0.01)^2/100) = 0.00632455…;
    Sortino = mean/dd x sqrt(252). The denominator convention is the load-bearing
    half — the two plausible alternatives differ by sqrt(100/40) = 1.58x.

``max_drawdown_closed_form``
    NAV rises monotonically for 60 days, then falls monotonically for 40, so the
    peak is at day 60 and the trough at the end: mdd = 0.99^40 - 1.

``calmar_closed_form``
    The tile's own arithmetic: total_ret = nav[-1]/nav[0] - 1 (nav[0] already
    carries day 1's return, so the ratio is 1.01^59 x 0.99^40), annualized by
    ^(252/100), divided by |mdd|.

``cvar_95_closed_form``
    The empirical 95% CVaR of a two-valued series: the interpolated loss quantile
    at rank 0.95(99) = 94.05 lands on 0.01, and the mean of the 40 losses at or
    beyond it is 0.01 — reported by the tile on the RETURN scale, so **-0.01**
    exactly. Pins both the tail definition and the sign convention.

``flat_returns_sharpe_is_undefined``
    A constant return series has zero volatility, so its Sharpe is **undefined**.
    1.0 iff the tile reports ``None`` rather than a finite number: a measured zero
    and an undefined value must be distinguishable (`principles.md` §2.7 at the
    metric layer). Sortino and Calmar are checked the same way in the same case
    family.

``sharpe_scale_invariance`` / ``max_drawdown_currency_invariance``
    METAMORPHIC. Multiplying every daily return by 3 leaves the Sharpe ratio
    unchanged (mean and volatility scale together at a zero risk-free rate);
    multiplying the whole NAV series by 7 leaves max drawdown unchanged (it is a
    ratio). These catch bugs where the correct ANSWER is unknown but the correct
    RELATIONSHIP is known — a class structurally invisible to a closed-form test,
    which can only check inputs whose answer someone already wrote down.

RELATIONSHIP TO ``grading/attestation.py``
-------------------------------------------
``attestation.py`` is the machine-facing §2.3a *verdict* the Report Card, the
Director and the console consume to decide whether to act on the week's numbers.
This module is the human-facing *evidence*: full inputs per case, published
tolerances, and the resolved version of every quant library actually loaded at
runtime plus the code SHA — so a divergence six weeks from now is diagnosable
from the artifact alone, with nobody's memory involved. Both share the fleet
verdict vocabulary (``PASS``/``FAIL``/``UNKNOWN``) so a reader never translates.

CONTRACT
--------
``run_self_test()`` **never raises**, and the caller writes its output
unconditionally. A case that DISAGREED is ``FAIL`` (evidence the numbers are
wrong); a case that could not RUN is ``UNKNOWN`` (absence of evidence).
Collapsing the two would make a broken image read as a correctness regression.
Per Brian's ruling 2026-08-13, **a case that exceeds its time budget is FAIL,
never UNKNOWN**.

This module introduces no hard-fail path. The handler writes the artifact and
carries the verdict in the stage's terminal output; a non-PASS verdict withholds
a guarantee rather than failing the run (`sf-pipeline-policy.md` §2.3a).
"""

from __future__ import annotations

import datetime as _dt
import io
import json
import logging
import math
import os
import platform
import signal
import threading
import time
from pathlib import Path
from typing import Any, Callable, NamedTuple

from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

SCHEMA = "evaluator_self_test-1.0.0"
COMPONENT = "evaluator"

PASS = "PASS"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"

#: ``evaluator/{run_date}/self_test.json`` — beside the run's report card.
_KEY_TEMPLATE = "evaluator/{run_date}/self_test.json"

#: Quant distributions whose resolved version decides what every number on the
#: Report Card means. Recorded via ``importlib.metadata`` (the DISTRIBUTION
#: version pip actually resolved into the image) rather than
#: ``module.__version__`` — an attribute a package may lack, lie about, or
#: inherit from a vendored copy.
_TRACKED_DISTRIBUTIONS = (
    "nousergon-lib",
    "numpy",
    "pandas",
    "scipy",
    "krepis",
    "pydantic",
    "boto3",
)

#: Per-case wall-clock budget. Each case is one tile build over 100 in-memory
#: rows; anything approaching this is a hang, not a slow machine.
CASE_TIMEOUT_SECONDS = 30.0

#: 1e-9 absolute, per the specification. Every expectation is an exact float64
#: identity, and the observed agreement is ~1e-15 — the band is far tighter than
#: any accounting or convention change could hide under, and is not tuned to
#: make the battery pass.
TOLERANCE = 1e-9

# ── the frozen series ───────────────────────────────────────────────────────
_UP = 0.01
_DOWN = -0.01
_N_UP = 60
_N_DOWN = 40
_N = _N_UP + _N_DOWN
_NAV0 = 1_000_000.0
_FLAT_RETURN = 0.005
_RETURN_SCALE = 3.0
_NAV_SCALE = 7.0
_TRADING_DAYS = 252


class _CaseTimeout(Exception):
    """Raised when a case exceeds :data:`CASE_TIMEOUT_SECONDS`."""


class Case(NamedTuple):
    """One known-answer or metamorphic check.

    ``expected`` is derived by hand from the metric's definition; ``compute``
    drives the production tile and returns the comparable observed number. They
    are kept apart — rather than ``compute`` returning a bool — so the artifact
    carries both numbers and a later divergence is diagnosable from the artifact
    alone. ``inputs`` is published verbatim: a reader must be able to re-derive
    ``expected`` on paper without opening this file.
    """

    name: str
    description: str
    inputs: dict
    expected: float
    compute: Callable[[], float]
    tolerance: float = TOLERANCE


# ════════════════════════════════════════════════════════════════════════════
# The frozen eod_pnl.csv and its in-memory S3
# ════════════════════════════════════════════════════════════════════════════

def _returns() -> list[float]:
    return [_UP] * _N_UP + [_DOWN] * _N_DOWN


def _eod_pnl_csv(returns: list[float], *, nav_scale: float = 1.0) -> str:
    """Render the frozen series as the exact CSV shape the tile parses.

    ``daily_return_pct``/``spy_return_pct`` are in PERCENT (the executor's export
    convention the tile divides by 100) — so this fixture also exercises the unit
    conversion, which is where the ``avg_volume_20d`` class of bug lives.
    ``repr`` is used rather than a format string so no expectation is lost to
    rounding in the fixture itself.
    """
    day = _dt.date(2024, 1, 1)
    nav = _NAV0
    lines = ["date,portfolio_nav,daily_return_pct,spy_return_pct"]
    for r in returns:
        while day.weekday() >= 5:      # trading days only
            day += _dt.timedelta(days=1)
        nav *= 1.0 + r
        lines.append(f"{day.isoformat()},{nav * nav_scale!r},{r * 100.0!r},{0.0!r}")
        day += _dt.timedelta(days=1)
    return "\n".join(lines) + "\n"


class _FrozenS3:
    """Serves the frozen CSV and NOTHING else.

    Every other key raises ``NoSuchKey``, which is the tile's documented
    absent-artifact path — so the battery touches neither the network nor the
    real bucket, and a case can never accidentally grade production data.
    """

    def __init__(self, text: str, key: str):
        self._text = text
        self._key = key

    def get_object(self, Bucket=None, Key=None):   # noqa: N803 — boto3 signature
        if Key == self._key:
            return {"Body": io.BytesIO(self._text.encode())}
        raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")


_TILE_CACHE: dict[str, dict] = {}


def _tile_values(variant: str) -> dict[str, float | None]:
    """Build the production tile over a frozen variant and return {name: value}.

    Cached for the life of the process so the nine cases pay for four tile
    builds, not nine. The input is frozen and the path is pure, so the cache
    cannot mask a change — a redeploy is a new process.
    """
    if variant in _TILE_CACHE:
        return _TILE_CACHE[variant]

    from grading.tiles.portfolio_outcome import EOD_PNL_KEY, build_portfolio_outcome_tile

    if variant == "base":
        text = _eod_pnl_csv(_returns())
    elif variant == "flat":
        text = _eod_pnl_csv([_FLAT_RETURN] * _N)
    elif variant == "returns_scaled":
        text = _eod_pnl_csv([r * _RETURN_SCALE for r in _returns()])
    elif variant == "nav_scaled":
        text = _eod_pnl_csv(_returns(), nav_scale=_NAV_SCALE)
    else:  # pragma: no cover — defensive
        raise ValueError(f"unknown fixture variant {variant!r}")

    tile = build_portfolio_outcome_tile(
        "self-test-frozen-fixture", s3_client=_FrozenS3(text, EOD_PNL_KEY),
    )
    values = {c["name"]: c.get("value") for c in tile.get("components", [])}
    if "sharpe_ratio" not in values:
        raise AssertionError(
            "the production tile emitted no sharpe_ratio component for the frozen "
            f"fixture (components: {sorted(values)}) — the self-test graded nothing."
        )
    _TILE_CACHE[variant] = values
    return values


def _value(variant: str, metric: str) -> float:
    value = _tile_values(variant)[metric]
    if value is None:
        raise AssertionError(
            f"{metric} is None on the '{variant}' fixture — the frozen series is "
            "designed to produce a defined value, so this is a contract break, not "
            "a measurement."
        )
    return float(value)


def _is_undefined(variant: str, metric: str) -> float:
    """1.0 iff ``metric`` is reported as undefined (``None``/NaN) on ``variant``."""
    value = _tile_values(variant)[metric]
    if value is None:
        return 1.0
    return 0.0 if math.isfinite(float(value)) else 1.0


# ════════════════════════════════════════════════════════════════════════════
# The expectations — plain arithmetic from each metric's definition
# ════════════════════════════════════════════════════════════════════════════

def _expected_mean() -> float:
    return (_N_UP * _UP + _N_DOWN * _DOWN) / _N


def _expected_sharpe() -> float:
    mean = _expected_mean()
    variance = (_N_UP * (_UP - mean) ** 2 + _N_DOWN * (_DOWN - mean) ** 2) / (_N - 1)
    return mean / math.sqrt(variance) * math.sqrt(_TRADING_DAYS)


def _expected_sortino() -> float:
    mean = _expected_mean()
    downside_deviation = math.sqrt(_N_DOWN * _DOWN**2 / _N)
    return mean / downside_deviation * math.sqrt(_TRADING_DAYS)


def _expected_max_drawdown() -> float:
    return (1.0 + _DOWN) ** _N_DOWN - 1.0


def _expected_calmar() -> float:
    # nav[0] already carries day 1's return, so nav[-1]/nav[0] is 1.01^59 * 0.99^40.
    growth = (1.0 + _UP) ** (_N_UP - 1) * (1.0 + _DOWN) ** _N_DOWN
    annualized = growth ** (_TRADING_DAYS / _N) - 1.0
    return annualized / abs(_expected_max_drawdown())


def _expected_cvar_95() -> float:
    # Two-valued series: the interpolated 95% loss quantile lands on 0.01 and the
    # mean of the 40 losses at or beyond it is 0.01. Reported on the RETURN scale.
    return _DOWN


# ════════════════════════════════════════════════════════════════════════════
# The battery
# ════════════════════════════════════════════════════════════════════════════

def _base_inputs(**extra) -> dict:
    return {"n_days": _N, "daily_return_up": _UP, "n_up": _N_UP,
            "daily_return_down": _DOWN, "n_down": _N_DOWN,
            "spy_daily_return": 0.0, "initial_nav": _NAV0,
            "annualization_days": _TRADING_DAYS, **extra}


def build_cases() -> list[Case]:
    """The battery. A callable (not a module constant) so no tile import happens
    at import time of this module, and so a test can substitute it."""
    return [
        Case(
            name="sharpe_closed_form",
            description=(
                "mean = (60(+1%) + 40(-1%))/100 = 0.002; sample variance (ddof=1) "
                "= [60(0.008)^2 + 40(0.012)^2]/99 = 0.0096/99; "
                "Sharpe = mean/sd * sqrt(252)"
            ),
            inputs=_base_inputs(ddof=1, risk_free_rate=0.0, units="ratio"),
            expected=_expected_sharpe(),
            compute=lambda: _value("base", "sharpe_ratio"),
        ),
        Case(
            name="sortino_closed_form",
            description=(
                "Downside deviation is the RMS of min(0, r) over ALL 100 "
                "observations, not over the 40 negatives: dd = sqrt(40(0.01)^2/100); "
                "Sortino = mean/dd * sqrt(252). The denominator convention is the "
                "load-bearing half — the plausible alternative differs by 1.58x"
            ),
            inputs=_base_inputs(downside_denominator="N (all observations)",
                                target=0.0, units="ratio"),
            expected=_expected_sortino(),
            compute=lambda: _value("base", "sortino_ratio"),
        ),
        Case(
            name="max_drawdown_closed_form",
            description=(
                "NAV rises monotonically for 60 days then falls monotonically for "
                "40, so peak is day 60 and trough is the last day: "
                "max_drawdown = 0.99^40 - 1"
            ),
            inputs=_base_inputs(units="fraction (negative)"),
            expected=_expected_max_drawdown(),
            compute=lambda: _value("base", "max_drawdown"),
        ),
        Case(
            name="calmar_closed_form",
            description=(
                "annualized return over |max drawdown|: nav[-1]/nav[0] = "
                "1.01^59 * 0.99^40 (nav[0] already carries day 1's return), "
                "annualized by ^(252/100), divided by |0.99^40 - 1|"
            ),
            inputs=_base_inputs(annualization_exponent=_TRADING_DAYS / _N,
                                units="ratio"),
            expected=_expected_calmar(),
            compute=lambda: _value("base", "calmar_ratio"),
        ),
        Case(
            name="cvar_95_closed_form",
            description=(
                "Empirical 95% CVaR of the two-valued series: the interpolated loss "
                "quantile at rank 0.95*99 = 94.05 lands on 0.01, and the mean of the "
                "40 losses at or beyond it is 0.01 — reported on the RETURN scale, "
                "so -0.01 exactly. Pins the tail definition AND the sign convention"
            ),
            inputs=_base_inputs(confidence=0.95, sign_convention="return scale",
                                units="fraction (negative)"),
            expected=_expected_cvar_95(),
            compute=lambda: _value("base", "cvar_95_daily"),
        ),
        Case(
            name="flat_returns_sharpe_is_undefined",
            description=(
                f"A constant {_FLAT_RETURN:.3%}/day series has zero volatility, so "
                "its Sharpe is UNDEFINED. 1.0 iff the tile reports None rather than "
                "a finite number — a measured zero and an undefined value must be "
                "distinguishable"
            ),
            inputs={"n_days": _N, "constant_daily_return": _FLAT_RETURN,
                    "units": "boolean-encoded contract"},
            expected=1.0,
            compute=lambda: _is_undefined("flat", "sharpe_ratio"),
            tolerance=0.0,
        ),
        Case(
            name="flat_returns_sortino_is_undefined",
            description=(
                "Same series has no below-target day, so downside deviation is zero "
                "and Sortino is UNDEFINED. 1.0 iff the tile reports None"
            ),
            inputs={"n_days": _N, "constant_daily_return": _FLAT_RETURN,
                    "units": "boolean-encoded contract"},
            expected=1.0,
            compute=lambda: _is_undefined("flat", "sortino_ratio"),
            tolerance=0.0,
        ),
        Case(
            name="sharpe_scale_invariance",
            description=(
                f"METAMORPHIC. Multiplying every daily return by {_RETURN_SCALE:.0f} "
                "leaves the Sharpe ratio unchanged — mean and volatility scale "
                "together at a zero risk-free rate. Difference expected 0.0"
            ),
            inputs=_base_inputs(return_scale=_RETURN_SCALE,
                                units="ratio (difference)"),
            expected=0.0,
            compute=lambda: (_value("returns_scaled", "sharpe_ratio")
                             - _value("base", "sharpe_ratio")),
        ),
        Case(
            name="max_drawdown_currency_invariance",
            description=(
                f"METAMORPHIC. Multiplying the whole NAV series by {_NAV_SCALE:.0f} "
                "leaves max drawdown unchanged — it is a ratio, so it carries no "
                "currency units. Difference expected 0.0"
            ),
            inputs=_base_inputs(nav_scale=_NAV_SCALE,
                                units="fraction (difference)"),
            expected=0.0,
            compute=lambda: (_value("nav_scaled", "max_drawdown")
                             - _value("base", "max_drawdown")),
        ),
    ]


# ════════════════════════════════════════════════════════════════════════════
# Provenance header — the reason this is an INSTRUMENT check, not a code check
# ════════════════════════════════════════════════════════════════════════════

def resolved_library_versions(
    distributions: tuple[str, ...] = _TRACKED_DISTRIBUTIONS,
) -> dict[str, str]:
    """The installed version of every quant distribution loaded at runtime.

    ``importlib.metadata.version`` reads the DISTRIBUTION metadata pip resolved
    into this image — the thing that moves between the CI runner and the deployed
    Lambda. A missing distribution is recorded explicitly, never omitted: an
    absent key and a missing library must not look the same.
    """
    from importlib.metadata import PackageNotFoundError, version

    resolved: dict[str, str] = {}
    for dist in distributions:
        try:
            resolved[dist] = version(dist)
        except PackageNotFoundError:
            resolved[dist] = "<not installed>"
        except Exception as exc:  # noqa: BLE001 — a version probe never blocks
            resolved[dist] = f"<unavailable: {type(exc).__name__}>"
    return resolved


def code_sha() -> str:
    """The SHA of the code that ran, without shelling out.

    Deploy-time stamps first (``GIT_SHA`` env, then ``/var/task/GIT_SHA.txt`` —
    the fleet's Lambda-image convention, stamped by ``infrastructure/deploy.sh``
    via ``--build-arg GIT_SHA``), then the checkout's own ``.git`` refs for a
    local run. ``unknown`` is a legitimate answer and is recorded as one — a
    fabricated SHA is worse than an absent one.
    """
    for env_key in ("GIT_SHA", "CODE_SHA", "GITHUB_SHA"):
        stamped = os.environ.get(env_key)
        if stamped:
            return stamped.strip()
    try:
        lambda_stamp = Path("/var/task/GIT_SHA.txt")
        if lambda_stamp.is_file():
            stamped = lambda_stamp.read_text().strip()
            if stamped:
                return stamped
    except Exception:  # noqa: BLE001 — provenance never blocks the battery
        pass
    try:
        git_dir = Path(__file__).resolve().parents[1] / ".git"
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:].strip()
        ref_path = git_dir / ref
        if ref_path.is_file():
            return ref_path.read_text().strip()
        for line in (git_dir / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split(" ", 1)[0].strip()
        return "unknown"
    except Exception:  # noqa: BLE001 — provenance never blocks the battery
        return "unknown"


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

def _call_with_timeout(fn: Callable[[], float], seconds: float) -> float:
    """Run ``fn`` under a wall-clock budget.

    A SIGALRM budget is installed when this is the main thread of a platform
    that has one (a Lambda handler thread qualifies). Where it is not available
    the elapsed time is checked after the call instead — that cannot interrupt a
    hang, but it does catch an overrun, and the caller distinguishes neither:
    both are FAIL.
    """
    can_interrupt = (
        hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )
    if not can_interrupt:
        started = time.monotonic()
        value = fn()
        elapsed = time.monotonic() - started
        if elapsed > seconds:
            raise _CaseTimeout(
                f"case exceeded its {seconds:g}s budget "
                f"({elapsed:.1f}s, detected after the fact)"
            )
        return value

    def _fire(_signum, _frame):
        raise _CaseTimeout(f"case exceeded its {seconds:g}s budget")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def run_self_test(
    run_date: str | None = None,
    *,
    case_provider: "Callable[[], list[Case]] | None" = None,
    component: str = COMPONENT,
    schema: str = SCHEMA,
    case_timeout_seconds: float = CASE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run the known-answer battery on the DEPLOYED instrument and return the artifact body.

    Never raises — see the module docstring's CONTRACT section.
    """
    started = time.monotonic()
    header = {
        "schema": schema,
        "component": component,
        "run_date": run_date,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "code_sha": code_sha(),
        "libraries": resolved_library_versions(),
        "case_timeout_seconds": case_timeout_seconds,
    }

    provider = case_provider or build_cases
    try:
        # Materialised inside the try: a provider returning a lazy or broken
        # iterable would otherwise raise at the FOR loop below, outside every
        # handler, and take down the stage this must never be able to fail.
        cases = list(provider())
    except Exception as exc:  # noqa: BLE001 — see CONTRACT: this becomes UNKNOWN
        logger.error(
            "self-test: the battery could not be constructed (%s: %s) — verdict "
            "UNKNOWN. No correctness guarantee is granted this cycle.",
            type(exc).__name__, exc, exc_info=True,
        )
        return {**header, "status": "error", "verdict": UNKNOWN, "cases": [],
                "n_cases": 0, "n_failed": 0, "n_errored": 0,
                "error_class": type(exc).__name__, "error_msg": str(exc)[:500],
                "wall_clock_seconds": round(time.monotonic() - started, 3)}

    records: list[dict] = []
    for case in cases:
        record: dict[str, Any] = {
            "case": case.name,
            "description": case.description,
            "inputs": case.inputs,
            "expected": case.expected,
            "actual": None,
            "abs_error": None,
            "tolerance": case.tolerance,
            "verdict": UNKNOWN,
        }
        case_started = time.monotonic()
        try:
            actual = float(_call_with_timeout(case.compute, case_timeout_seconds))
            error = abs(actual - case.expected)
            record["actual"] = actual
            record["abs_error"] = error
            record["verdict"] = PASS if error <= case.tolerance else FAIL
            if record["verdict"] == FAIL:
                logger.error(
                    "self-test case FAILED: %s expected=%r actual=%r abs_error=%r "
                    "tolerance=%r", case.name, case.expected, actual, error,
                    case.tolerance,
                )
        except _CaseTimeout as exc:
            # Brian ruling 2026-08-13: a timeout is FAIL, never UNKNOWN. A
            # known-answer case over 100 in-memory rows that cannot finish in its
            # budget is itself evidence something is wrong with the instrument —
            # it must not buy the benefit of the doubt that "the battery could
            # not be constructed" gets.
            record["verdict"] = FAIL
            record["timed_out"] = True
            record["error_class"] = type(exc).__name__
            record["error_msg"] = str(exc)[:500]
            logger.error("self-test case TIMED OUT (=> FAIL): %s (%s)", case.name, exc)
        except Exception as exc:  # noqa: BLE001 — a case that could not RUN is UNKNOWN
            record["verdict"] = UNKNOWN
            record["errored"] = True
            record["error_class"] = type(exc).__name__
            record["error_msg"] = str(exc)[:500]
            logger.error(
                "self-test case ERRORED (=> UNKNOWN): %s (%s: %s)",
                case.name, type(exc).__name__, exc, exc_info=True,
            )
        record["wall_clock_seconds"] = round(time.monotonic() - case_started, 3)
        records.append(record)

    n_failed = sum(1 for r in records if r["verdict"] == FAIL)
    n_errored = sum(1 for r in records if r["verdict"] == UNKNOWN)
    if n_failed:
        verdict = FAIL
    elif n_errored or not records:
        verdict = UNKNOWN
    else:
        verdict = PASS

    body = {**header, "status": "ok", "verdict": verdict, "cases": records,
            "n_cases": len(records), "n_failed": n_failed, "n_errored": n_errored,
            "wall_clock_seconds": round(time.monotonic() - started, 3)}

    if verdict == PASS:
        logger.info(
            "self-test PASS — %d/%d known-answer cases agreed on %s (%s)",
            len(records), len(records), header["python"],
            ", ".join(f"{k} {v}" for k, v in header["libraries"].items()),
        )
    elif verdict == UNKNOWN:
        logger.error(
            "self-test UNKNOWN — %d/%d cases could not run. The correctness "
            "guarantee is WITHHELD this cycle (never granted by default).",
            n_errored, len(records),
        )
    else:
        logger.error(
            "self-test FAIL — %d/%d known-answer cases DISAGREE with their "
            "hand-derived expectation on the DEPLOYED libraries (%s). THIS "
            "CYCLE'S REPORT CARD NUMBERS ARE NOT TRUSTWORTHY.",
            n_failed, len(records),
            ", ".join(f"{k} {v}" for k, v in header["libraries"].items()),
        )
    return body


def verdict_is_pass(verdict: str | None) -> bool:
    """True only for an explicit PASS — ``None`` and ``"ok"`` withhold the guarantee."""
    return verdict == PASS


def self_test_key(run_date: str) -> str:
    """S3 key of the evaluator's published self-test for ``run_date``."""
    return _KEY_TEMPLATE.format(run_date=run_date)


def write_self_test(bucket: str, run_date: str, body: dict, s3_client=None) -> str:
    """Persist the artifact. Returns the key written.

    Raises on failure: this artifact IS the evidence the stage graded its own
    arithmetic, so a silent write failure would reproduce exactly the absence the
    self-test exists to remove. The caller isolates it (see ``grading/handler.py``)
    so the report card — the primary deliverable — survives regardless.
    """
    import boto3

    client = s3_client or boto3.client("s3")
    key = self_test_key(run_date)
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(body, indent=2, default=str).encode(),
        ContentType="application/json",
    )
    return key
