"""Tests for exoplanet_finder.data_preparation.signal_input.

The fake stars have a slow sine-shaped drift (1% over 15 days) multiplied by
box-shaped dips, sampled at Kepler's long cadence. No network is used.
"""

import lightkurve as lk
import numpy as np
import pandas as pd
import pytest

from exoplanet_finder.data_preparation import detrend, signal_input
from exoplanet_finder.data_preparation.folding import bin_curve
from exoplanet_finder.data_preparation.signal_input import (
    bin_or_fill,
    build_signal_input,
)

KEPLER_CADENCE_DAYS = 0.0204  # ~29.4 minutes
N_BINS = 30
N_DURATIONS = 3  # window = 3 durations, so bins 10-19 lie inside the dip
CENTER = slice(12, 18)  # bins well inside the dip
EDGES = np.r_[0:8, 22:30]  # bins well outside the dip


def make_koi(period, t0, duration, depth_even, depth_odd=None, depth_secondary=0.0):
    """Describe one fake signal: its KOI columns plus the depths to inject."""
    return {
        "koi_period": period,
        "koi_time0bk": t0,
        "koi_duration_days": duration,
        "depth_even": depth_even,
        "depth_odd": depth_even if depth_odd is None else depth_odd,
        "depth_secondary": depth_secondary,
    }


def make_star(kois, n_days=120.0, drift_amplitude=0.01):
    """Build a fake light curve and the matching KOI table for one star.

    Returns:
        (light curve, star_kois DataFrame with the three KOI columns).
    """
    time = np.arange(0.0, n_days, KEPLER_CADENCE_DAYS)
    flux = 1.0 + drift_amplitude * np.sin(2 * np.pi * time / 15.0)
    for koi in kois:
        period, t0, half = koi["koi_period"], koi["koi_time0bk"], koi["koi_duration_days"] / 2
        number = np.round((time - t0) / period)
        in_transit = np.abs(time - (t0 + number * period)) < half
        depth = np.where(number % 2 == 0, koi["depth_even"], koi["depth_odd"])
        flux = np.where(in_transit, flux * (1.0 - depth), flux)
        sec_number = np.round((time - t0 - period / 2) / period)
        in_secondary = np.abs(time - (t0 + period / 2 + sec_number * period)) < half
        flux = np.where(in_secondary, flux * (1.0 - koi["depth_secondary"]), flux)
    columns = ["koi_period", "koi_time0bk", "koi_duration_days"]
    star_kois = pd.DataFrame([{c: koi[c] for c in columns} for koi in kois])
    return lk.LightCurve(time=time, flux=flux), star_kois


def build(star_kois, light_curve, index=0):
    return build_signal_input(
        star_kois.iloc[index], star_kois, light_curve, n_bins=N_BINS, n_durations=N_DURATIONS
    )


def dip_depth(channel):
    return 1.0 - np.median(channel[CENTER])


# ------------------------------------------------------------ bin_or_fill()


def test_bin_or_fill_returns_bin_curve_values_when_window_has_data():
    # Arrange
    folded = np.column_stack([[0.25, 0.75, 1.5], [2.0, 4.0, 7.0]])

    # Act
    values, filled = bin_or_fill(folded, n_bins=2, low=0.0, high=2.0)

    # Assert
    np.testing.assert_allclose(values, bin_curve(folded, 2, 0.0, 2.0))
    assert filled is False


def test_bin_or_fill_returns_flat_ones_when_window_is_empty():
    # Arrange: every point lies outside the window [0, 2).
    folded = np.column_stack([[-3.0, 5.0], [0.99, 0.98]])

    # Act
    values, filled = bin_or_fill(folded, n_bins=4, low=0.0, high=2.0)

    # Assert
    np.testing.assert_array_equal(values, np.ones(4))
    assert filled is True


def test_bin_or_fill_does_not_hide_other_errors():
    # Arrange: a wrong argument type is a bug, not an empty window.
    folded = np.column_stack([[0.5], [1.0]])

    # Act / Assert
    with pytest.raises(TypeError):
        bin_or_fill(folded, n_bins="4", low=0.0, high=2.0)


# ----------------------------------------------------- build_signal_input()


def test_build_signal_input_returns_three_channels_and_three_flags():
    # Arrange
    light_curve, star_kois = make_star([make_koi(3.1, 1.2, 0.4, depth_even=0.005)])

    # Act
    flags, channels = build(star_kois, light_curve)

    # Assert
    assert channels.shape == (3, N_BINS)
    assert flags.shape == (3,)


def test_build_signal_input_planet_has_same_dip_in_odd_and_even_and_flat_secondary():
    # Arrange: a planet-like signal, equal dips, no secondary eclipse.
    light_curve, star_kois = make_star([make_koi(3.1, 1.2, 0.4, depth_even=0.005)])

    # Act
    flags, channels = build(star_kois, light_curve)
    odd, even, secondary = channels

    # Assert
    np.testing.assert_array_equal(flags, [0.0, 0.0, 0.0])
    assert dip_depth(odd) == pytest.approx(0.005, rel=0.05)
    assert dip_depth(even) == pytest.approx(0.005, rel=0.05)
    assert abs(dip_depth(secondary)) < 1e-4
    assert np.all(np.abs(odd[EDGES] - 1.0) < 2e-4)


def test_build_signal_input_puts_odd_and_even_dips_in_their_own_channels():
    # Arrange: a binary reported at half its period: even dips 0.4%, odd 1%.
    koi = make_koi(3.1, 1.2, 0.4, depth_even=0.004, depth_odd=0.010)
    light_curve, star_kois = make_star([koi])

    # Act
    _, channels = build(star_kois, light_curve)

    # Assert: channel 1 holds the odd dips, channel 2 the even ones.
    assert dip_depth(channels[0]) == pytest.approx(0.010, rel=0.05)
    assert dip_depth(channels[1]) == pytest.approx(0.004, rel=0.05)


def test_build_signal_input_keeps_secondary_eclipse_depth():
    # Arrange: a 0.4-day secondary eclipse. Without the secondary mask in the
    # second flatten, flatten() would keep only about 57% of its depth.
    koi = make_koi(3.1, 1.2, 0.4, depth_even=0.010, depth_secondary=0.003)
    light_curve, star_kois = make_star([koi])

    # Act
    _, channels = build(star_kois, light_curve)

    # Assert
    assert dip_depth(channels[2]) == pytest.approx(0.003, rel=0.05)


def test_build_signal_input_fills_and_flags_channels_without_data():
    # Arrange: a 200-day orbit seen for 60 days. The only transit is #0
    # (even) at day 10; the secondary (day 110) is after the data ends.
    koi = make_koi(200.0, 10.0, 0.4, depth_even=0.005)
    light_curve, star_kois = make_star([koi], n_days=60.0)

    # Act
    flags, channels = build(star_kois, light_curve)

    # Assert: odd and secondary channels are filler, even holds the dip.
    np.testing.assert_array_equal(flags, [1.0, 0.0, 1.0])
    np.testing.assert_array_equal(channels[0], np.ones(N_BINS))
    np.testing.assert_array_equal(channels[2], np.ones(N_BINS))
    assert dip_depth(channels[1]) == pytest.approx(0.005, rel=0.05)


def test_build_signal_input_uses_the_target_kois_own_period():
    # Arrange: two signals on one star whose dips never overlap each other's
    # windows (A every 4 days from day 1, B every 8 days from day 3). No drift
    # here: this test is only about which period is used. (With drift, masks
    # this close together make flatten() distort the baseline.)
    koi_a = make_koi(4.0, 1.0, 0.2, depth_even=0.005)
    koi_b = make_koi(8.0, 3.0, 0.4, depth_even=0.002)
    light_curve, star_kois = make_star([koi_a, koi_b], drift_amplitude=0.0)

    # Act
    _, channels_a = build(star_kois, light_curve, index=0)
    _, channels_b = build(star_kois, light_curve, index=1)

    # Assert
    assert dip_depth(channels_a[0]) == pytest.approx(0.005, rel=0.05)
    assert dip_depth(channels_b[0]) == pytest.approx(0.002, rel=0.05)
    assert dip_depth(channels_b[1]) == pytest.approx(0.002, rel=0.05)


def test_build_signal_input_leaves_light_curve_unchanged():
    # Arrange
    light_curve, star_kois = make_star([make_koi(3.1, 1.2, 0.4, depth_even=0.005)])
    original_flux = light_curve.flux.value.copy()

    # Act
    build(star_kois, light_curve)

    # Assert
    np.testing.assert_array_equal(light_curve.flux.value, original_flux)


# ------------------------------------------- the two flattens and their masks


def spy_on_flatten(monkeypatch, transit_t0s):
    """Record every flatten_lightcurve() call made by build_signal_input().

    Curves from a flatten whose mask includes anything other than the given
    transit times (i.e. a secondary eclipse) come back with their flux
    doubled, so a channel built from that curve has a baseline of 2.0
    instead of 1.0. That shows which flatten each channel was built from.

    Returns:
        List of recorded calls; each is the sorted list of masked t0 values.
    """
    calls = []
    real_flatten = detrend.flatten_lightcurve

    def flatten(lc, periods, transit_times, durations_days):
        t0s = np.atleast_1d(np.asarray(transit_times, dtype=float))
        calls.append(sorted(t0s.tolist()))
        flat = real_flatten(lc, periods, transit_times, durations_days)
        masks_a_secondary = any(not np.isclose(t, transit_t0s).any() for t in t0s)
        if masks_a_secondary:
            return lk.LightCurve(time=flat.time.value, flux=flat.flux.value * 2.0)
        return flat

    monkeypatch.setattr(detrend, "flatten_lightcurve", flatten)
    monkeypatch.setattr(signal_input, "flatten_lightcurve", flatten, raising=False)
    return calls


def test_build_signal_input_masks_only_the_target_kois_secondary(monkeypatch):
    # Arrange: two KOIs on one star. A: every 4 days from day 1.5 (its
    # secondary would be at day 3.5). B, the target: every 8 days from day 3
    # (secondary at day 7).
    koi_a = make_koi(4.0, 1.5, 0.2, depth_even=0.005)
    koi_b = make_koi(8.0, 3.0, 0.4, depth_even=0.002)
    light_curve, star_kois = make_star([koi_a, koi_b], drift_amplitude=0.0)
    calls = spy_on_flatten(monkeypatch, transit_t0s=[1.5, 3.0])

    # Act
    build(star_kois, light_curve, index=1)

    # Assert: flatten (a) masks both transits; flatten (b) adds only B's
    # secondary. A's secondary (day 3.5) is never masked.
    assert sorted(calls) == [[1.5, 3.0], [1.5, 3.0, 7.0]]
    assert not any(np.isclose(3.5, call).any() for call in calls)


def test_build_signal_input_builds_odd_and_even_from_the_transit_only_flatten(monkeypatch):
    # Arrange
    light_curve, star_kois = make_star([make_koi(3.1, 1.2, 0.4, depth_even=0.005)])
    spy_on_flatten(monkeypatch, transit_t0s=[1.2])

    # Act
    _, channels = build(star_kois, light_curve)
    odd, even, secondary = channels

    # Assert: only channel 3 comes from the flatten with the secondary mask
    # (the spy doubles that curve, so its baseline is 2.0).
    np.testing.assert_allclose(odd[EDGES], 1.0, atol=1e-3)
    np.testing.assert_allclose(even[EDGES], 1.0, atol=1e-3)
    np.testing.assert_allclose(secondary[EDGES], 2.0, atol=2e-3)


def test_build_signal_input_passes_float_mask_columns_to_flatten(monkeypatch):
    # Arrange: real KOI rows also hold text (e.g. kepoi_name), which makes a
    # single row an object-dtype Series.
    light_curve, star_kois = make_star([make_koi(3.1, 1.2, 0.4, depth_even=0.005)])
    star_kois["kepoi_name"] = ["K00001.01"]
    dtype_kinds = []
    real_flatten = detrend.flatten_lightcurve

    def flatten(lc, periods, transit_times, durations_days):
        columns = (periods, transit_times, durations_days)
        dtype_kinds.append({np.asarray(column).dtype.kind for column in columns})
        return real_flatten(lc, periods, transit_times, durations_days)

    monkeypatch.setattr(detrend, "flatten_lightcurve", flatten)
    monkeypatch.setattr(signal_input, "flatten_lightcurve", flatten, raising=False)

    # Act
    build(star_kois, light_curve)

    # Assert: both flattens get float ("f") columns, never object ("O").
    assert dtype_kinds == [{"f"}, {"f"}]
