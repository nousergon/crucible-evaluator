"""director — Layer C: the weekly advisory action-plan agent.

A single Opus director agent runs as the final task of the Saturday pipeline,
reviews the graded report card (Layer B) plus the latest ROADMAP digest, and
emits a structured **advisory** ``DirectorWeeklyActionPlan`` with a carry-over
ledger.

Hard invariants:
  - **Proposes only.** No write path to any live trading config — it cannot move
    a single trading parameter (the backtester's auto-apply loop owns that).
  - **Never self-merges.** Its plan is archived to the console AND (Phase H,
    ``director/roadmap_pr.py``) opened as an approval-gated PR against the
    planning docs (``ROADMAP.md``) for Brian to review + merge — his review IS
    the gate (no soak flag; default on, ``DIRECTOR_ROADMAP_PR_ENABLED`` is a
    kill-switch).

Added after the measurement foundation (Layer B) clears its acceptance gate. See
``director-implementation-plan-260604.md`` Part II (Phases E–H).
"""
