"""The referee's value spec — gates, blends, honest misses."""

from __future__ import annotations

import math

import pytest

from gamekit.scoring import (
    AXES,
    DIRECTIONS,
    ScoreGate,
    ScoringSpec,
    score,
)


def test_axes_and_directions_vocabulary():
    assert AXES == ("speed", "memory", "launch", "params")
    assert DIRECTIONS == ("at_most", "at_least")


def test_gate_validation_rejects_bad_rows():
    with pytest.raises(ValueError):
        ScoreGate("", 1.0)
    with pytest.raises(ValueError):
        ScoreGate("speed", float("nan"))
    with pytest.raises(ValueError):
        ScoreGate("speed", 1.0, direction="sideways")


def test_gate_pass_margin_ratio_at_most():
    g = ScoreGate("speed", 0.95)
    assert g.passed(0.9) and not g.passed(1.0)
    assert g.margin(0.9) == pytest.approx(0.05)
    assert g.ratio(0.9) == pytest.approx(0.9 / 0.95)
    # zero threshold: only non-positive measurements clear
    z = ScoreGate("params", 0.0)
    assert z.ratio(0.0) == 0.0 and z.ratio(1.0) == math.inf


def test_gate_pass_margin_ratio_at_least():
    g = ScoreGate("win_rate", 0.5, direction="at_least")
    assert g.passed(0.5) and not g.passed(0.49)
    assert g.margin(0.6) == pytest.approx(0.1)
    # at_least inverts the ratio: clearing the floor drops it below 1
    assert g.ratio(1.0) == pytest.approx(0.5)
    assert g.ratio(0.0) == math.inf


def test_spec_normalises_and_validates():
    with pytest.raises(ValueError):
        ScoringSpec(gates=(ScoreGate("speed", 1.0),), weights={"speed": -1.0})
    with pytest.raises(ValueError):
        ScoringSpec(
            gates=(ScoreGate("speed", 1.0),),
            weights={"memory": 1.0},  # neither gated nor referenced
        )
    with pytest.raises(ValueError):
        ScoringSpec(gates=(), references={"memory": 0.0})
    spec = ScoringSpec(gates=[ScoreGate("speed", 1.0)])
    assert isinstance(spec.gates, tuple)


def test_spec_blend_weights_and_reference():
    spec = ScoringSpec(
        gates=(ScoreGate("speed", 0.95),),
        weights={"speed": 1.0, "memory": 0.25},
        references={"memory": 4.0e8},
    )
    assert spec.blend_weights() == {"speed": 1.0, "memory": 0.25}
    assert spec.reference("speed") == 0.95
    assert spec.reference("memory") == 4.0e8
    assert spec.reference("launch") is None
    # empty weights → every gated axis at weight 1.0
    g = ScoringSpec(gates=(ScoreGate("speed", 1.0), ScoreGate("params", 1.0)))
    assert g.blend_weights() == {"speed": 1.0, "params": 1.0}


def test_score_all_gates_clear():
    spec = ScoringSpec(
        gates=(
            ScoreGate("speed", 0.95),
            ScoreGate("params", 1.0),
            ScoreGate("win_rate", 0.5, direction="at_least"),
        ),
        weights={"speed": 1.0, "memory": 0.25},
        references={"memory": 4.0e8},
    )
    v = score(spec, {"speed": 0.9, "params": 0.8, "win_rate": 0.6, "memory": 2.0e8})
    assert v.passed
    assert v.value == pytest.approx(0.9 / 0.95 + 0.25 * 0.5)
    assert v.missing == ()
    assert len(v.gates) == 3 and all(g.passed for g in v.gates)


def test_score_missing_axis_fails_honestly():
    spec = ScoringSpec(
        gates=(ScoreGate("speed", 0.95), ScoreGate("params", 1.0)),
    )
    v = score(spec, {"speed": 0.9})
    assert not v.passed
    assert v.missing == ("params",)
    missed = next(g for g in v.gates if g.gate.axis == "params")
    assert not missed.passed and math.isnan(missed.value)


def test_score_weighted_axis_missing_is_nan_value():
    spec = ScoringSpec(
        gates=(ScoreGate("speed", 0.95),),
        weights={"speed": 1.0, "memory": 0.5},
        references={"memory": 4.0e8},
    )
    v = score(spec, {"speed": 0.9})
    assert v.passed  # gates still resolve
    assert math.isnan(v.value)  # but the row cannot be priced
    assert v.missing == ("memory",)


def test_score_at_least_blend_inverts():
    spec = ScoringSpec(
        gates=(ScoreGate("win_rate", 0.5, direction="at_least"),),
        weights={"win_rate": 1.0},
    )
    v = score(spec, {"win_rate": 1.0})
    assert v.passed and v.value == pytest.approx(0.5)
    z = score(spec, {"win_rate": 0.0})
    assert math.isnan(z.value) is False  # v<=0 → inf blend, not nan
    assert z.value == math.inf
