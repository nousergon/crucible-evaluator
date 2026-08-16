"""Both Lambda handlers flush the cost sink before returning (config-I7423).

An AWS Lambda container is FROZEN between invocations, not exited, so
``krepis.cost_sink.S3JsonlCostSink``'s ``atexit`` hook never runs. The sink
buffers to 200 records per ``(date, callsite_id)`` group, so a handler that
finishes below the threshold writes NOTHING at all.

Measured 2026-08-15 on weekly-SF execution ``watch-rerun-2026-08-15-2``:
``AggregateCosts`` failed the whole run with ``2 stage(s) ran and emitted no
cost record ... Observed producers: (none)``. PR #197 had merged the cost-sink
environment onto the Director days earlier and was correct; the records were
priced, accepted, and lost in memory.

Structural, not behavioural: a NEW handler in this repo cannot ship without
the flush, and this does not depend on predicting which handlers call a model.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HANDLERS = sorted(_REPO_ROOT.glob("*/handler.py"))


def test_handler_set_is_not_empty():
    """Guards the parametrisation: an empty glob would pass vacuously."""
    assert len(_HANDLERS) >= 2, f"only found {[str(p) for p in _HANDLERS]}"


@pytest.mark.parametrize("path", _HANDLERS, ids=lambda p: p.parent.name)
def test_handler_flushes_the_cost_sink_in_a_finally(path: Path):
    tree = ast.parse(path.read_text())
    handler = next(
        (n for n in tree.body
         if isinstance(n, ast.FunctionDef) and n.name == "handler"),
        None,
    )
    assert handler is not None, f"{path} defines no module-level `handler`"

    tries = [n for n in ast.walk(handler) if isinstance(n, ast.Try) and n.finalbody]
    assert tries, (
        f"{path}: handler has no try/finally. The flush must sit in a "
        f"`finally` — `_run` has several return paths and raises on others, "
        f"and a flush on one of them is the defect config-I7423 fixed."
    )
    flushed = any(
        (isinstance(node, ast.Name) and node.id == "flush_default_sink")
        or (isinstance(node, ast.Attribute) and node.attr == "flush_default_sink")
        or (isinstance(node, ast.alias) and node.name == "flush_default_sink")
        for t in tries for stmt in t.finalbody for node in ast.walk(stmt)
    )
    assert flushed, (
        f"{path}: the handler's `finally` does not call "
        f"krepis.cost_sink.flush_default_sink() — this Lambda's cost records "
        f"die in the in-process buffer (config-I7423)"
    )


@pytest.mark.parametrize("path", _HANDLERS, ids=lambda p: p.parent.name)
def test_flush_failure_cannot_break_the_handler(path: Path):
    tree = ast.parse(path.read_text())
    handler = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "handler"
    )
    guarded = any(
        isinstance(n, ast.Try)
        and any(isinstance(node, ast.alias) and node.name == "flush_default_sink"
                for node in ast.walk(n))
        and n.handlers
        for n in ast.walk(handler)
    )
    assert guarded, (
        f"{path}: the flush import is not exception-guarded — an image "
        f"resolving krepis <0.59.8 would fail the handler on a telemetry concern"
    )
