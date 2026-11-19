"""Tests for exoplanet_finder.visualizations.graphs."""

import matplotlib

matplotlib.use("Agg")  # draw without opening windows

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from exoplanet_finder.visualizations.graphs import plot_binned_curve  # noqa: E402

# Four bins over [-0.1, 0.1) days: centers at -0.075, -0.025, 0.025, 0.075 days.
VALUES = np.array([1.0, 0.999, 1.0, 1.0005])
LOW, HIGH = -0.1, 0.1


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_plot_binned_curve_puts_bin_centers_in_hours_and_flux_change_in_ppm():
    # Act
    ax = plot_binned_curve(VALUES, LOW, HIGH)

    # Assert
    curve = ax.lines[0]
    np.testing.assert_allclose(curve.get_xdata(), [-1.8, -0.6, 0.6, 1.8])
    np.testing.assert_allclose(curve.get_ydata(), [0.0, -1000.0, 0.0, 500.0], atol=1e-6)


def test_plot_binned_curve_marks_transit_center():
    # Act
    ax = plot_binned_curve(VALUES, LOW, HIGH)

    # Assert
    center_line = ax.lines[1]
    np.testing.assert_array_equal(center_line.get_xdata(), [0, 0])


def test_plot_binned_curve_draws_on_given_axes_with_title():
    # Arrange
    _, (left, right) = plt.subplots(1, 2)

    # Act
    result = plot_binned_curve(VALUES, LOW, HIGH, title="Kepler-10 b", ax=right)

    # Assert
    assert result is right
    assert right.get_title() == "Kepler-10 b"
    assert len(left.lines) == 0


def test_plot_binned_curve_creates_labelled_figure_when_no_axes_given():
    # Act
    ax = plot_binned_curve(VALUES, LOW, HIGH)

    # Assert
    assert ax.figure is not None
    assert ax.get_xlabel() == "Time from transit center [hours]"
    assert ax.get_ylabel() == "Flux change [ppm]"
