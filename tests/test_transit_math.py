"""Tests for exoplanet_finder.data_preparation.transit_math."""

import numpy as np
import pytest

from exoplanet_finder.data_preparation.folding import fold
from exoplanet_finder.data_preparation.transit_math import (
    orbit_window_bounds,
    transit_window_bounds,
)


def test_transit_window_bounds_are_symmetric_around_zero():
    # Act
    low, high = transit_window_bounds(duration_days=0.2, n_durations=3)

    # Assert
    assert low == pytest.approx(-0.3)
    assert high == pytest.approx(0.3)


def test_transit_window_total_width_is_n_durations():
    # Act
    low, high = transit_window_bounds(duration_days=0.125, n_durations=5)

    # Assert
    assert high - low == pytest.approx(5 * 0.125)


def test_transit_window_of_one_duration_spans_exactly_the_transit():
    # Act
    bounds = transit_window_bounds(duration_days=0.1, n_durations=1)

    # Assert
    assert bounds == pytest.approx((-0.05, 0.05))


def test_orbit_window_bounds_span_one_period_centered_on_transit():
    # Act
    low, high = orbit_window_bounds(period=9.488)

    # Assert
    assert low == pytest.approx(-4.744)
    assert high == pytest.approx(4.744)
    assert high - low == pytest.approx(9.488)


def test_orbit_window_contains_every_folded_point():
    # Arrange: four years of Kepler-like time stamps, folded on a 1.737-day orbit.
    period, t0 = 1.737, 131.5
    times = np.arange(130.0, 1590.0, 0.0204)
    curve = np.column_stack([times, np.ones_like(times)])

    # Act
    phase = fold(curve, period, t0)[:, 0]
    low, high = orbit_window_bounds(period)

    # Assert: bin_curve() uses [low, high), so no point is dropped.
    assert np.all((phase >= low) & (phase < high))
