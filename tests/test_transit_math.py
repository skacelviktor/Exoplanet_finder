"""Tests for exoplanet_finder.data_preparation.transit_math."""

import pytest

from exoplanet_finder.data_preparation.transit_math import transit_window_bounds


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
