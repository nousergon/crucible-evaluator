"""Guard: every ``_lift_to_grade`` / ``_fmt_lift`` call declares its units.

alpha-engine-config-I2318. ``_lift_to_grade``'s anchors are calibrated in
PERCENTAGE POINTS, but most backtester producers emit raw return fractions.
7 of 19 live call sites passed a fraction against pp anchors, which pinned the
term at ~40.0 (C-) for every realistic input — a constant wearing the costume
of a measurement, on the surface Brian reads to decide what to fix next.

The runtime signature now makes a silent mismatch impossible (``units`` is
required and keyword-only), so this test guards the thing the signature cannot:
that a future call site does not sidestep the declaration by passing it
positionally through ``*args``, and that ``_fmt_lift`` — which has a default
neither the signature nor the type checker will complain about — is always told
explicitly. Detection blindness outranks the defects it hides: without this,
the next producer added re-opens the class in silence.
"""

import ast
from pathlib import Path

import pytest

SCORECARD = Path(__file__).resolve().parents[1] / "grading" / "scorecard.py"

# Both helpers take a units contract; both had the same defect.
GUARDED = ("_lift_to_grade", "_fmt_lift")

VALID_UNITS = {"pp", "fraction", "native"}


def _call_sites():
    tree = ast.parse(SCORECARD.read_text(), filename=str(SCORECARD))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
        if name in GUARDED:
            yield name, node


def test_call_sites_exist():
    """Guard the guard: an empty scan must fail, not vacuously pass."""
    sites = list(_call_sites())
    assert len(sites) >= 19, (
        f"expected >=19 guarded call sites in {SCORECARD.name}, found {len(sites)} — "
        "the AST scan is matching nothing and this whole module would pass vacuously"
    )


@pytest.mark.parametrize("name,node", [
    pytest.param(n, c, id=f"{n}:L{c.lineno}") for n, c in _call_sites()
])
def test_units_declared_and_valid(name, node):
    if name == "_lift_to_grade":
        kw = {k.arg for k in node.keywords}
        assert "units" in kw, (
            f"{SCORECARD.name}:{node.lineno} calls _lift_to_grade without units=. "
            "Read the PRODUCER's arithmetic and declare 'pp', 'fraction' or 'native' "
            "— do not infer the scale from the field's name."
        )
        units_node = next(k.value for k in node.keywords if k.arg == "units")
    else:  # _fmt_lift takes units as the 2nd positional arg
        if len(node.args) >= 2:
            units_node = node.args[1]
        else:
            kw = {k.arg: k.value for k in node.keywords}
            assert "units" in kw, (
                f"{SCORECARD.name}:{node.lineno} calls _fmt_lift without units."
            )
            units_node = kw["units"]

    assert isinstance(units_node, ast.Constant), (
        f"{SCORECARD.name}:{node.lineno} passes a non-literal units value. It must be "
        "a literal so this guard can verify it statically."
    )
    assert units_node.value in VALID_UNITS, (
        f"{SCORECARD.name}:{node.lineno} declares units={units_node.value!r}; "
        f"valid: {sorted(VALID_UNITS)}"
    )


def test_lift_to_grade_rejects_missing_units():
    from grading.scorecard import _lift_to_grade

    with pytest.raises(TypeError):
        _lift_to_grade(0.01, floor=-1.0, ceiling=2.0)  # type: ignore[call-arg]


def test_lift_to_grade_rejects_unknown_units():
    from grading.scorecard import _lift_to_grade

    with pytest.raises(ValueError, match="units must be one of"):
        _lift_to_grade(0.01, floor=-1.0, ceiling=2.0, units="percent")


def test_fraction_and_pp_agree_on_the_same_underlying_value():
    """The whole defect in one assertion: 0.0031 and 0.31 are the same lift."""
    from grading.scorecard import _lift_to_grade

    as_fraction = _lift_to_grade(0.0031, floor=-1.5, ceiling=2.5, units="fraction")
    as_pp = _lift_to_grade(0.31, floor=-1.5, ceiling=2.5, units="pp")
    assert as_fraction == pytest.approx(as_pp)


def test_native_units_are_not_scaled():
    from grading.scorecard import _lift_to_grade

    assert _lift_to_grade(0.3, floor=-0.3, ceiling=0.5, units="native") == pytest.approx(76.0)


def test_fmt_lift_renders_a_fraction_in_percentage_points():
    """The measured live value that rendered as an exact '-0.00%'."""
    from grading.scorecard import _fmt_lift

    assert _fmt_lift(-0.0031, "fraction") == "-0.31%"
    assert _fmt_lift(0.31, "pp") == "+0.31%"
    assert _fmt_lift(None, "fraction") is None
