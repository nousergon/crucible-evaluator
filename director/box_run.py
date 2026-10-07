"""box_run.py — the Director's off-Lambda entrypoint (alpha-engine-config-I11936).

Run by ``infrastructure/director_on_box.sh`` on the weekly SF's launcher spot,
over SSM, from the SF's ``Director`` state::

    python -m director.box_run --date 2026-10-02 --dry-run false \\
        --gate-state-file /tmp/director-gate-state-<execution>.json \\
        --deadline-epoch 1791100000

It runs exactly the code the Lambda runs — ``director.handler.run_director`` —
with one difference: the ``HostProfile`` it is quoted against
(``director/hosting.py::WEEKLY_SPOT``). Every artifact the Lambda writes, this
writes, at the same keys.

THE BUDGET. There is no Lambda context here, so this module supplies the same
countdown one would: a wall clock that starts when the process does. The wall
is ``WEEKLY_SPOT.wall_s`` (2,700s), CLAMPED to the deadline the SSM command
actually exported (``--deadline-epoch``, the command's start plus its
``executionTimeout``, less a margin). So if the Step Function ever grants less
time than ``hosting.py`` asks for, the Director quotes its calls shorter and
declines what it cannot afford — it is never SIGKILLed mid-call by SSM, which
writes nothing and logs no cause.

THE RESULT. The process exit code carries the two facts the SF routes on
(``hosting.exit_code_for``); the full summary is logged (and shipped to S3 by
``krepis.ssm_log_capture``) and printed as the last stdout line for a human
reading the log. The SF never parses stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time

logger = logging.getLogger("director.box_run")

#: Seconds held back between the in-process wall and the SSM hard deadline:
#: interpreter shutdown, the cost-sink flush and the log ship all happen after
#: the wall and before SSM's kill.
DEADLINE_MARGIN_S = 60.0

#: Exit code for a run that could not start or raised. Distinct from every
#: completed-run code in ``hosting.COMPLETED_EXIT_CODES``.
EXIT_FAILED = 1


class WallContext:
    """The slice of a Lambda context ``director.budget.InvocationBudget`` reads.

    ``get_remaining_time_in_millis`` counts down from ``wall_s`` on a monotonic
    clock, exactly as Lambda's does from the function timeout.
    """

    function_name = None

    def __init__(self, wall_s: float, *, clock=time.monotonic) -> None:
        if wall_s <= 0:
            raise ValueError(f"wall_s must be > 0, got {wall_s}")
        self._clock = clock
        self._deadline = clock() + wall_s

    def get_remaining_time_in_millis(self) -> int:
        return int(max(0.0, self._deadline - self._clock()) * 1000)


def effective_wall_s(configured_wall_s: float, deadline_epoch: float | None,
                     *, now: float | None = None) -> float:
    """The wall this run may use: the configured one, clamped to the deadline.

    ``deadline_epoch`` is the SSM command's hard kill time; ``None`` means the
    caller exported none (a manual run), and the configured wall stands.
    Raises when the deadline leaves nothing — starting would only guarantee a
    kill mid-call.
    """
    if deadline_epoch is None:
        return float(configured_wall_s)
    now = time.time() if now is None else now
    available = float(deadline_epoch) - now - DEADLINE_MARGIN_S
    if available <= 0:
        raise RuntimeError(
            f"Director: the SSM deadline leaves {available:.0f}s after the "
            f"{DEADLINE_MARGIN_S:.0f}s margin — provisioning consumed the whole "
            "command budget. Refusing to start a run SSM would kill mid-call."
        )
    if available < configured_wall_s:
        logger.warning(
            "Director: wall clamped %.0fs -> %.0fs by the SSM deadline (the Step "
            "Function grants less than director/hosting.py asks for).",
            configured_wall_s, available,
        )
    return min(float(configured_wall_s), available)


def _parse_bool(text: str) -> bool:
    value = str(text).strip().lower()
    if value in ("true", "1", "yes"):
        return True
    if value in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {text!r}")


def load_gate_state(path: str | None) -> dict | None:
    """The SF's gate_state, written by the SSM command. Fail loud if malformed.

    Absent (``None``) is legitimate — a manual run has no SF gate state, and
    the Director already treats a missing one as UNKNOWN. A file that EXISTS
    but does not parse is a wiring defect and must not read as "no gates".
    """
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        state = json.load(f)
    if not isinstance(state, dict):
        raise ValueError(f"gate state at {path} is {type(state).__name__}, not an object")
    return state


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the weekly Director on the launcher spot.")
    p.add_argument("--date", required=True, help="the SF's $.run_date")
    p.add_argument("--dry-run", type=_parse_bool, default=False,
                   help="the SF's $.research_dry (true on the Friday preflight)")
    p.add_argument("--gate-state-file", default=None,
                   help="JSON written from the SF's gate_state object")
    p.add_argument("--deadline-epoch", type=float, default=None,
                   help="the SSM command's hard kill time (unix seconds)")
    p.add_argument("--execution-name", default=None, help="the SF execution name, for logs")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Nothing on this host extracts the EMF lines the Director prints, so its
    # metrics are also published directly (director/emf.py).
    from director import emf

    emf.enable_direct_put()
    try:
        from director.hosting import WEEKLY_SPOT, HostProfile, exit_code_for

        wall = effective_wall_s(WEEKLY_SPOT.wall_s, args.deadline_epoch)
        host = HostProfile(WEEKLY_SPOT.name, wall, WEEKLY_SPOT.plan_ceiling_s)
        event = {
            "date": args.date,
            "dry_run": args.dry_run,
            "gate_state": load_gate_state(args.gate_state_file),
        }
        # A fresh registry copy per run — see handler._ensure_registry.
        os.environ["DIRECTOR_REGISTRY_DEST"] = os.path.join(
            tempfile.mkdtemp(prefix="director-registry-"), "LLM_MODEL_REGISTRY.yaml"
        )
        logger.info(
            "Director on %s: run_date=%s dry_run=%s wall=%.0fs plan_ceiling=%.0fs execution=%s",
            host.name, args.date, args.dry_run, host.wall_s, host.plan_ceiling_s,
            args.execution_name,
        )

        from director.handler import run_director

        summary = run_director(event, WallContext(host.wall_s), host=host)
    except Exception:  # noqa: BLE001 — every failure becomes one loud, non-zero exit
        logger.exception("Director FAILED on the weekly spot")
        return EXIT_FAILED

    code = exit_code_for(summary)
    logger.info("Director finished: status=%s retro_refused=%s exit=%d",
                summary.get("status"), summary.get("retro_refused"), code)
    print(json.dumps(summary, default=str, separators=(",", ":")), flush=True)
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
