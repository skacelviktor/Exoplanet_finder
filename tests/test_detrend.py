"""Tests for exoplanet_finder.data_preparation.detrend.

The fake star has a slow sine-shaped drift (1% over 15 days) multiplied by
box-shaped dips, sampled at Kepler's long cadence.
"""

import lightkurve as lk
import numpy as np
import pandas as pd

from exoplanet_finder.data_preparation.detrend import flatten_lightcurve

KEPLER_CADENCE_DAYS = 0.0204  # ~29.4 minutes


def make_star(signals, n_days=60.0, drift_amplitude=0.01, drift_period=15.0):
    """Build a fake light curve with drift and dips.

    Args:
        signals: List of (period, t0, duration, depth) tuples, in days and
            fractional depth.

    Returns:
        (light curve, list of in-transit boolean masks, one per signal).
    """
    time = np.arange(0.0, n_days, KEPLER_CADENCE_DAYS)
    flux = 1.0 + drift_amplitude * np.sin(2 * np.pi * time / drift_period)
    in_transit = []
    for period, t0, duration, depth in signals:
        distance = np.abs((time - t0 + period / 2) % period - period / 2)
        dip = distance < duration / 2
        flux = np.where(dip, flux * (1.0 - depth), flux)
        in_transit.append(dip)
    return lk.LightCurve(time=time, flux=flux), in_transit


def signal_series(signals):
    """Split (period, t0, duration, depth) tuples into the three Series."""
    periods, t0s, durations, _ = zip(*signals)
    return pd.Series(periods), pd.Series(t0s), pd.Series(durations)


def test_flatten_lightcurve_removes_slow_drift_outside_transits():
    # Arrange
    signals = [(3.1, 1.2, 0.2, 0.005)]
    lc, (dip,) = make_star(signals)
    raw_scatter = np.std(lc.flux.value[~dip])

    # Act
    flat = flatten_lightcurve(lc, *signal_series(signals))

    # Assert: out-of-transit flux is back at 1 and the drift is gone.
    out_of_transit = flat.flux.value[~dip]
    assert raw_scatter > 0.005
    assert abs(np.median(out_of_transit) - 1.0) < 1e-4
    assert np.std(out_of_transit) < 2e-4


def test_flatten_lightcurve_keeps_transit_depth():
    # Arrange: a 0.4-day dip. Without the transit mask, flatten() would keep
    # only about 57% of its depth; with the mask it keeps all of it.
    depth = 0.005
    signals = [(3.1, 1.2, 0.4, depth)]
    lc, (dip,) = make_star(signals)

    # Act
    flat = flatten_lightcurve(lc, *signal_series(signals))

    # Assert: the dip is still 0.5% deep (within 5% of the depth).
    measured_depth = 1.0 - np.median(flat.flux.value[dip])
    assert abs(measured_depth - depth) < 0.05 * depth


def test_flatten_lightcurve_protects_every_listed_signal():
    # Arrange: two signals on the same star, long enough that an unmasked
    # dip would lose a large part of its depth.
    signals = [(3.1, 1.2, 0.4, 0.005), (7.3, 2.5, 0.5, 0.002)]
    lc, dips = make_star(signals)

    # Act
    flat = flatten_lightcurve(lc, *signal_series(signals))

    # Assert
    for (_, _, _, depth), dip in zip(signals, dips):
        measured_depth = 1.0 - np.median(flat.flux.value[dip])
        assert abs(measured_depth - depth) < 0.05 * depth


def test_flatten_lightcurve_leaves_input_unchanged():
    # Arrange
    signals = [(3.1, 1.2, 0.2, 0.005)]
    lc, _ = make_star(signals)
    original_flux = lc.flux.value.copy()

    # Act
    flatten_lightcurve(lc, *signal_series(signals))

    # Assert
    np.testing.assert_array_equal(lc.flux.value, original_flux)
