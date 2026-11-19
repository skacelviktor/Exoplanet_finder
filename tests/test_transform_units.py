"""Tests for exoplanet_finder.data_preparation.transform_units."""

import pandas as pd

from exoplanet_finder.data_preparation.transform_units import hours_to_days


def test_hours_to_days_divides_by_24():
    # Arrange
    hours = pd.Series([24.0, 1.5, 0.0])

    # Act
    days = hours_to_days(hours)

    # Assert
    pd.testing.assert_series_equal(days, pd.Series([1.0, 0.0625, 0.0]))


def test_hours_to_days_keeps_index_and_name_and_leaves_input_unchanged():
    # Arrange
    hours = pd.Series([12.0, 36.0], index=[7, 9], name="koi_duration")
    original = hours.copy()

    # Act
    days = hours_to_days(hours)

    # Assert
    pd.testing.assert_series_equal(
        days, pd.Series([0.5, 1.5], index=[7, 9], name="koi_duration")
    )
    pd.testing.assert_series_equal(hours, original)
