"""Tests for exoplanet_finder.folding: fold() and bin_curve().

Every light curve here is a small, hand-made array of shape (N, 2):
column 0 is time (or phase, after folding) and column 1 is flux.
"""

import numpy as np
import pytest

from exoplanet_finder.folding import bin_curve, fold


def make_curve(times, fluxes):
    """Build an (N, 2) light-curve array from separate time and flux lists."""
    return np.column_stack(
        [np.asarray(times, dtype=float), np.asarray(fluxes, dtype=float)]
    )


# ---------------------------------------------------------------- fold()


def test_fold_puts_every_transit_center_at_phase_zero():
    # Arrange
    period, t0 = 3.7, 12.4
    transit_times = t0 + period * np.arange(-2, 4)
    curve = make_curve(transit_times, np.ones_like(transit_times))

    # Act
    folded = fold(curve, period, t0)

    # Assert
    np.testing.assert_allclose(folded[:, 0], 0.0, atol=1e-9)


def test_fold_gives_signed_time_from_nearest_transit():
    # Arrange: points just after (+) and just before (-) transit centers.
    period, t0 = 4.0, 10.0
    times = [t0 + 2 * period + 0.3, t0 + 2 * period - 0.3, t0 + 1.9, t0 + 2.1]
    curve = make_curve(times, [1.0] * 4)

    # Act
    folded = fold(curve, period, t0)

    # Assert: t0 + 2.1 is closer to the next transit (t0 + 4), so it is -1.9.
    np.testing.assert_allclose(folded[:, 0], [0.3, -0.3, 1.9, -1.9], atol=1e-9)


def test_fold_handles_times_before_t0():
    # Arrange: real light curves have many points before the first transit,
    # so time - t0 is negative for them.
    period, t0 = 4.0, 100.0
    times = [t0 - 1.0, t0 - period + 0.5, t0 - 10 * period - 1.5]
    curve = make_curve(times, [1.0] * 3)

    # Act
    folded = fold(curve, period, t0)

    # Assert
    np.testing.assert_allclose(folded[:, 0], [-1.0, 0.5, -1.5], atol=1e-9)


def test_fold_keeps_every_phase_within_half_open_range():
    # Arrange
    period, t0 = 3.7, 1.3
    times = np.linspace(-50.0, 150.0, 2001)
    curve = make_curve(times, np.ones_like(times))

    # Act
    phase = fold(curve, period, t0)[:, 0]

    # Assert
    assert np.all(phase >= -period / 2)
    assert np.all(phase < period / 2)


def test_fold_keeps_flux_column_and_row_order():
    # Arrange
    fluxes = [1.0, 0.98, 1.02, 0.99]
    curve = make_curve([5.0, 0.1, 3.3, 9.9], fluxes)

    # Act
    folded = fold(curve, period=2.0, t0=0.5)

    # Assert
    assert folded.shape == curve.shape
    np.testing.assert_array_equal(folded[:, 1], fluxes)


def test_fold_returns_new_array_and_leaves_input_unchanged():
    # Arrange
    curve = make_curve([0.0, 1.0, 2.5, 7.0], [1.0, 0.99, 1.01, 1.0])
    original = curve.copy()

    # Act
    folded = fold(curve, period=2.0, t0=0.5)

    # Assert
    assert folded is not curve
    np.testing.assert_array_equal(curve, original)


# ----------------------------------------------------------- bin_curve()


def test_bin_curve_returns_one_value_per_bin():
    # Arrange
    rng = np.random.default_rng(seed=0)
    phases = rng.uniform(-1.0, 1.0, size=500)
    folded = make_curve(phases, rng.normal(1.0, 0.001, size=500))

    # Act
    binned = bin_curve(folded, n_bins=7, low=-1.0, high=1.0)

    # Assert
    assert binned.shape == (7,)


def test_bin_curve_averages_points_in_each_bin():
    # Arrange: 4 bins of width 1 over [0, 4).
    folded = make_curve(
        [0.2, 0.8, 1.5, 2.1, 2.4, 2.9, 3.5],
        [1.0, 3.0, 5.0, 1.0, 2.0, 6.0, 7.0],
    )

    # Act
    binned = bin_curve(folded, n_bins=4, low=0.0, high=4.0)

    # Assert
    np.testing.assert_allclose(binned, [2.0, 5.0, 3.0, 7.0])


def test_bin_curve_bin_edges_depend_on_window_not_on_data():
    # Arrange: one point in the middle of each of 10 bins, flux = bin number.
    full = make_curve(np.arange(10) + 0.5, np.arange(10.0))
    # A data gap removes the point in the first bin; the data now starts at 1.5.
    with_gap = full[1:]

    # Act
    binned_full = bin_curve(full, n_bins=10, low=0.0, high=10.0)
    binned_gap = bin_curve(with_gap, n_bins=10, low=0.0, high=10.0)

    # Assert: bins 1-9 do not move; only the emptied first bin is filled.
    np.testing.assert_allclose(binned_full, np.arange(10.0))
    np.testing.assert_allclose(binned_gap[1:], np.arange(1.0, 10.0))
    assert binned_gap[0] == pytest.approx(1.0)


def test_bin_curve_ignores_points_outside_window():
    # Arrange: the window is [0, 2). Phase 0.0 is inside, 2.0 is outside.
    folded = make_curve(
        [-0.1, 0.0, 1.5, 2.0, 2.5],
        [100.0, 5.0, 7.0, 100.0, 100.0],
    )

    # Act
    binned = bin_curve(folded, n_bins=2, low=0.0, high=2.0)

    # Assert
    np.testing.assert_allclose(binned, [5.0, 7.0])


def test_bin_curve_drops_nan_flux():
    # Arrange
    folded = make_curve([0.2, 0.7, 1.5], [2.0, np.nan, 4.0])

    # Act
    binned = bin_curve(folded, n_bins=2, low=0.0, high=2.0)

    # Assert
    assert not np.isnan(binned).any()
    np.testing.assert_allclose(binned, [2.0, 4.0])


def test_bin_curve_fills_single_empty_bin_with_average_of_neighbors():
    # Arrange: bin 1 has no points.
    folded = make_curve([0.5, 2.5], [2.0, 6.0])

    # Act
    binned = bin_curve(folded, n_bins=3, low=0.0, high=3.0)

    # Assert
    np.testing.assert_allclose(binned, [2.0, 4.0, 6.0])


def test_bin_curve_fills_run_of_empty_bins_with_average_of_nearest_filled_bins():
    # Arrange: bins 1, 2 and 3 have no points.
    folded = make_curve([0.5, 4.5], [2.0, 10.0])

    # Act
    binned = bin_curve(folded, n_bins=5, low=0.0, high=5.0)

    # Assert
    np.testing.assert_allclose(binned, [2.0, 6.0, 6.0, 6.0, 10.0])


def test_bin_curve_fills_end_bins_from_one_side_without_wraparound():
    # Arrange: bins 0, 1 and 4 have no points.
    folded = make_curve([2.5, 3.5], [4.0, 6.0])

    # Act
    binned = bin_curve(folded, n_bins=5, low=0.0, high=5.0)

    # Assert: the left end copies bin 2, the right end copies bin 3.
    np.testing.assert_allclose(binned, [4.0, 4.0, 4.0, 6.0, 6.0])


def test_bin_curve_fills_bin_that_only_had_nan_points():
    # Arrange: bin 1's only point is NaN, so bin 1 counts as empty.
    folded = make_curve([0.5, 1.5, 2.5], [2.0, np.nan, 6.0])

    # Act
    binned = bin_curve(folded, n_bins=3, low=0.0, high=3.0)

    # Assert
    np.testing.assert_allclose(binned, [2.0, 4.0, 6.0])


@pytest.mark.parametrize(
    ("phases", "fluxes"),
    [
        pytest.param([-3.0, 5.0], [1.0, 1.0], id="all-points-outside-window"),
        pytest.param([0.5, 1.5], [np.nan, np.nan], id="all-flux-nan"),
        pytest.param([], [], id="no-points"),
    ],
)
def test_bin_curve_raises_when_window_has_no_usable_points(phases, fluxes):
    # Arrange
    folded = make_curve(phases, fluxes)

    # Act / Assert
    with pytest.raises(ValueError):
        bin_curve(folded, n_bins=2, low=0.0, high=2.0)


# -------------------------------------------------- fold() + bin_curve()


def test_fold_then_bin_recovers_box_shaped_transit():
    # Arrange: a fake star, flux 1.0, with a 1% box-shaped dip lasting
    # 0.2 days (center +/- 0.1) every 3 days, sampled every 0.02 days.
    period, t0, depth, half_duration = 3.0, 1.005, 0.01, 0.1
    times = np.arange(0.0, 60.0, 0.02)
    flux = np.ones_like(times)
    for k in range(-1, 21):
        transit_center = t0 + k * period
        flux[np.abs(times - transit_center) < half_duration] = 1.0 - depth
    curve = make_curve(times, flux)

    # Act: 5 bins of 0.2 days over [-0.5, 0.5); only bin 2, [-0.1, 0.1), is the dip.
    binned = bin_curve(fold(curve, period, t0), n_bins=5, low=-0.5, high=0.5)

    # Assert
    np.testing.assert_allclose(binned, [1.0, 1.0, 1.0 - depth, 1.0, 1.0], atol=1e-9)
