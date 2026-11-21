"""Build the CNN input for one transit signal (one KOI)."""

import lightkurve as lk
import numpy as np
import pandas as pd

from exoplanet_finder.data_preparation.detrend import flatten_lightcurve
from exoplanet_finder.data_preparation.folding import bin_curve, fold
from exoplanet_finder.data_preparation.transit_math import (
    transit_numbers,
    transit_window_bounds,
)


def bin_or_fill(
    folded: np.ndarray, n_bins: int, low: float, high: float
) -> tuple[np.ndarray, bool]:
    """Bin a folded light curve, or return flat filler if the window has no data.

    Calls ``bin_curve()``. If it raises ``ValueError`` (no usable point in
    ``[low, high)``), this returns ``n_bins`` values of 1.0, the flat
    baseline of a normalized light curve, and marks the result as filled.
    Only ``ValueError`` is caught, so genuine bugs (e.g. ``TypeError``) still
    surface.

    Args:
        folded: Array of shape (N, 2) with phase and flux, as for
            ``bin_curve()``.
        n_bins: Number of bins.
        low: Lower edge of the window (inclusive), in days.
        high: Upper edge of the window (exclusive), in days.

    Returns:
        ``(values, filled)``: a 1D array of length ``n_bins``, and True if the
        values are filler, False if they come from data.
    """
    try:
        return (bin_curve(folded, n_bins, low, high), False)
    except ValueError:
        return (np.ones((n_bins,)), True)


def _fold_and_bin(
    light_curve: np.ndarray, koi: pd.Series, n_durations: float = 2, n_bins: int = 200
) -> tuple[np.ndarray, bool]:
    """Fold a light curve at a KOI's transits and bin the window around them.

    Folds at the KOI's ``koi_time0bk`` with its ``koi_period``, then bins the
    window ``transit_window_bounds(koi_duration_days, n_durations)`` with
    ``bin_or_fill()``. Passing a copy of the KOI whose ``koi_time0bk`` is
    shifted by half a period centers the window on the secondary eclipse
    instead.

    Args:
        light_curve: Array of shape (N, 2) with time (BKJD days) in column 0
            and flux in column 1.
        koi: KOI row with ``koi_period``, ``koi_time0bk`` and
            ``koi_duration_days``.
        n_durations: Window width in transit durations.
        n_bins: Number of bins.

    Returns:
        ``(values, filled)`` from ``bin_or_fill()``.
    """
    fold_light_curve = fold(light_curve, koi["koi_period"], koi["koi_time0bk"])
    low, high = transit_window_bounds(koi["koi_duration_days"], n_durations)
    return bin_or_fill(fold_light_curve, n_bins, low, high)


def build_signal_input(
    koi: pd.Series,
    star_kois: pd.DataFrame,
    light_curve: lk.LightCurve,
    n_bins: int,
    n_durations: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn one KOI and its star's light curve into the CNN's input.

    Steps:
        1. Flatten ``light_curve`` twice with ``flatten_lightcurve()``:

           a. Masking the transits (``koi_time0bk``) of every KOI in
              ``star_kois``. Channels 1 and 2 use this curve, so both come
              from the same flatten with as few masked points as possible.
           b. Masking the same transits plus the secondary eclipse
              (``koi_time0bk + koi_period / 2``) of the target ``koi`` only.
              Channel 3 uses this curve, so a secondary eclipse keeps its
              depth. Other KOIs' secondaries are not masked: most KOIs have
              none, and every extra mask removes good baseline points.
        2. Number each point's nearest transit of ``koi`` with
           ``transit_numbers()``.
        3. Channel 1: points of flatten (a) from odd transits only, folded at
           the KOI's ``t0`` and binned in the window from
           ``transit_window_bounds(duration, n_durations)``.
        4. Channel 2: the same for points from even transits.
        5. Channel 3: all points of flatten (b), folded at
           ``t0 + period / 2`` and binned in the same window, which centers
           it on the secondary eclipse.

        Every channel is binned with ``bin_or_fill()``, so an empty window
        becomes 1.0s and sets that channel's flag.

    Args:
        koi: KOI table row of the target signal. Uses ``koi_period`` (days),
            ``koi_time0bk`` (BKJD days) and ``koi_duration_days`` (days; add
            this column with ``hours_to_days()``). ``koi_duration`` (hours)
            is not used.
        star_kois: All KOI rows of the same star, including ``koi``, with
            the same columns. Their transits are masked in both flattens.
        light_curve: The star's light curve from ``get_lightcurve()``, not
            yet flattened. It is not modified.
        n_bins: Number of bins in every channel.
        n_durations: Window width in transit durations (see
            ``transit_window_bounds()``).

    Returns:
        ``(flags, channels)``. ``flags`` has shape (3,): 1.0 where that
        channel was filled, 0.0 where it holds real data. ``channels`` has
        shape (3, n_bins), in the order odd, even, secondary.
    """
    # Flatten the light curve, one with second eclipse, one without it
    flat = flatten_lightcurve(
        light_curve,
        star_kois["koi_period"],
        star_kois["koi_time0bk"],
        star_kois["koi_duration_days"],
    )

    second_eclipse_koi = koi.copy()
    second_eclipse_koi["koi_time0bk"] += koi["koi_period"] / 2

    # Keep only the number columns, as floats: a whole KOI row mixes text and
    # numbers, so to_frame().T alone would turn every column into object.
    mask_columns = ["koi_period", "koi_time0bk", "koi_duration_days"]
    second_eclipse_star_kois = pd.concat(
        [star_kois[mask_columns], second_eclipse_koi[mask_columns].to_frame().T],
        ignore_index=True,
    ).astype(float)

    second_eclipse_flat = flatten_lightcurve(
        light_curve,
        second_eclipse_star_kois["koi_period"],
        second_eclipse_star_kois["koi_time0bk"],
        second_eclipse_star_kois["koi_duration_days"],
    )

    # Convert to numpy
    np_light_curve = np.column_stack([flat.time.value, flat.flux.value])

    second_eclipse_np_light_curve = np.column_stack(
        [second_eclipse_flat.time.value, second_eclipse_flat.flux.value]
    )

    # divide to the even/odd transit
    transit_number = transit_numbers(
        np_light_curve[:, 0], koi["koi_period"], koi["koi_time0bk"]
    )
    parity = transit_number % 2

    even_light_curve = np_light_curve[parity == 0]
    odd_light_curve = np_light_curve[parity == 1]

    # fold and bin every lightcurves
    channel1, flag1 = _fold_and_bin(
        odd_light_curve, koi, n_bins=n_bins, n_durations=n_durations
    )
    channel2, flag2 = _fold_and_bin(
        even_light_curve, koi, n_bins=n_bins, n_durations=n_durations
    )
    channel3, flag3 = _fold_and_bin(
        second_eclipse_np_light_curve,
        second_eclipse_koi,
        n_bins=n_bins,
        n_durations=n_durations,
    )

    flags = np.array([flag1, flag2, flag3], dtype=float)
    channels = np.array([channel1, channel2, channel3])

    return (flags, channels)
