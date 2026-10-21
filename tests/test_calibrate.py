"""Score calibration — Platt scaling, isotonic regression, metrics."""

from __future__ import annotations

import math

import pytest

from gamekit.calibrate import (
    IsotonicCal,
    PlattCal,
    brier,
    calibration_metrics,
    calibration_slope,
    ece,
    log_loss,
)


def test_platt_requires_aligned_nonempty():
    with pytest.raises(ValueError):
        PlattCal().fit([], [])
    with pytest.raises(ValueError):
        PlattCal().fit([0.5], [])


def test_platt_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        PlattCal().predict([0.5])


def test_platt_fit_maps_scores_monotone():
    scores = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
    labels = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    cal = PlattCal().fit(scores, labels)
    assert cal.a_ is not None and cal.b_ is not None
    assert cal.loss_ is not None and cal.loss_ < 0.4
    assert cal.n_iter_ > 0
    preds = cal.predict(scores)
    assert preds == sorted(preds)
    assert all(0.0 <= p <= 1.0 for p in preds)
    assert preds[0] < 0.5 < preds[-1]


def test_isotonic_requires_aligned_nonempty():
    with pytest.raises(ValueError):
        IsotonicCal().fit([], [])
    with pytest.raises(ValueError):
        IsotonicCal().fit([0.1], [])


def test_isotonic_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        IsotonicCal().predict([0.5])


def test_isotonic_fit_is_monotone_and_clamps():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    labels = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    cal = IsotonicCal().fit(scores, labels)
    assert cal.blocks_ and cal.knots_
    preds = cal.predict(scores)
    assert preds == sorted(preds)  # non-decreasing
    # clamps to the end block rates outside the fitted range
    assert cal.predict([0.0])[0] == pytest.approx(preds[0])
    assert cal.predict([1.0])[0] == pytest.approx(preds[-1])
    # interpolates between knots
    mid = cal.predict([0.45])[0]
    assert preds[3] <= mid <= preds[4]


def test_log_loss_and_brier():
    assert log_loss([], []) == 0.0
    assert log_loss([1.0], [1.0]) == pytest.approx(0.0, abs=1e-6)
    assert log_loss([0.0], [1.0]) > 2.0  # clamped but large
    assert brier([], []) == 0.0
    assert brier([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
    assert brier([0.5, 0.5], [1.0, 0.0]) == pytest.approx(0.25)


def test_ece_perfect_and_hedged():
    assert ece([], []) == 0.0
    perfect = ece([1.0, 0.0, 1.0], [1.0, 0.0, 1.0])
    assert perfect == pytest.approx(0.0, abs=1e-9)
    hedged = ece([0.5] * 4, [1.0, 0.0, 1.0, 0.0])
    assert hedged == pytest.approx(0.0, abs=1e-9)
    over = ece([0.9] * 4, [0.0] * 4)
    assert over == pytest.approx(0.9, abs=1e-9)


def test_calibration_slope():
    # degenerate labels → no slope exists
    assert calibration_slope([0.5, 0.5], [1.0, 1.0]) == (None, None)
    scores = [0.1, 0.3, 0.5, 0.7, 0.9]
    labels = [0.0, 0.0, 0.0, 1.0, 1.0]
    slope, intercept = calibration_slope(scores, labels)
    assert slope is not None and intercept is not None
    assert math.isfinite(slope) and math.isfinite(intercept)


def test_calibration_metrics_bundle():
    m = calibration_metrics([0.9, 0.1, 0.8], [1.0, 0.0, 1.0])
    assert m["n"] == 3 and m["pos_rate"] == pytest.approx(2 / 3, abs=0.001)
    assert m["brier"] <= 0.1 and m["slope"] is not None
    empty = calibration_metrics([], [])
    assert empty["n"] == 0 and empty["pos_rate"] is None
    assert empty["slope"] is None
