"""Phase folding and binning of light curves.

A light curve is stored as a 2D numpy array of shape (N, 2):
column 0 holds time in days (or phase in days, after folding) and
column 1 holds flux.

Typical use::

    folded = fold(light_curve, period, t0)
    values = bin_curve(folded, n_bins=200, low=-0.5, high=0.5)
"""

import numpy as np


def fold(light_curve: np.ndarray, period: float, t0: float) -> np.ndarray:
    """Phase-fold a light curve so every transit lands at phase 0.

    Each time value is replaced by its phase: the signed time in days from
    the nearest transit center. The phase is
    ``((time - t0 + period / 2) % period) - period / 2``, which puts every
    transit at 0 and maps every point into ``[-period / 2, period / 2)``.
    Python's ``%`` never returns a negative result for a positive
    ``period``, so times before ``t0`` fold correctly too.

    Args:
        light_curve: Array of shape (N, 2). Column 0 is time in days, in
            the same time system as ``t0`` (for Kepler: BKJD). Column 1 is
            flux.
        period: Orbital period in days. Must be positive.
        t0: Mid-transit time of any one transit (the TCE epoch), in days.
            This is the center of the dip, not its start.

    Returns:
        A new array of shape (N, 2). Column 0 is the phase in days, within
        ``[-period / 2, period / 2)``. Column 1 is the unchanged flux. Rows
        keep their original order. The input array is not modified.
    """
    folded = light_curve.copy()
    folded[:, 0] = ((folded[:, 0] - t0 + (period / 2)) % period) - (period / 2)
    return folded


def _next_filled_value(array: np.ndarray, start_pos: int) -> float | None:
    """Return the first non-NaN value in ``array`` at or after ``start_pos``.

    ``bin_curve()`` uses this to find the nearest non-empty bin to the right
    of an empty one; empty bins hold NaN.

    Args:
        array: 1D array of bin values, with NaN marking empty bins.
        start_pos: Index to start searching from (inclusive).

    Returns:
        The first non-NaN value found, or None if every value from
        ``start_pos`` to the end is NaN.
    """
    for i in array[start_pos:]:
        if not np.isnan(i):
            return i

    return None


def _get_mean_from_nearest_points(left: float | None, right: float | None) -> float:
    """Compute the fill value for an empty bin from its nearest non-empty bins.

    Args:
        left: Value of the nearest non-empty bin to the left, or None if
            there is none (the empty bin is at the left end of the window).
        right: Value of the nearest non-empty bin to the right, or None if
            there is none (the empty bin is at the right end of the window).

    Returns:
        The average of ``left`` and ``right``. If only one side exists, that
        side's value. If both are None, returns None; ``bin_curve()`` never
        lets that happen, because it raises ValueError when no bin has data.
    """
    if left is None:
        return right  # ty: ignore[invalid-explicit-return]

    elif right is None:
        return left

    else:
        return 0.5 * (left + right)


def bin_curve(folded: np.ndarray, n_bins: int, low: float, high: float) -> np.ndarray:
    """Average a folded light curve into evenly spaced phase bins.

    The window ``[low, high)`` is split into ``n_bins`` bins of equal width
    ``(high - low) / n_bins``. Bin ``k`` covers
    ``[low + k * width, low + (k + 1) * width)``, so where a bin sits
    depends only on ``low``, ``high`` and ``n_bins``, never on where the
    data points happen to lie. Pass the whole folded curve: points outside
    the window are ignored, so no separate window-cutting step is needed.

    Steps:
        1. Drop points whose flux is NaN.
        2. Ignore points whose phase is outside ``[low, high)``.
        3. Set each bin to the mean flux of its points.
        4. Fill each empty bin with the average of the nearest non-empty
           bin on its left and the nearest non-empty bin on its right. A
           run of several empty bins gets that same value throughout. If
           only one side has a non-empty bin (empty bins at either end of
           the window), use that side's value. There is no wrap-around from
           the last bin to the first.

    Args:
        folded: Array of shape (N, 2) with phase in column 0 and flux in
            column 1, as returned by ``fold()``.
        n_bins: Number of bins. Must be at least 1.
        low: Lower edge of the window (inclusive), in the same units as the
            phase.
        high: Upper edge of the window (exclusive). Must be greater than
            ``low``.

    Returns:
        A 1D array of length ``n_bins`` with no NaN values. Element ``k`` is
        the value of bin ``k``, whose center is at
        ``low + (k + 0.5) * width``.

    Raises:
        ValueError: If no point with a non-NaN flux falls inside
            ``[low, high)``.
    """
    # get the splits from the bins
    width = (high - low) / n_bins

    mask = (folded[:, 0] >= low) & (folded[:, 0] < high) & ~np.isnan(folded[:, 1])

    idx = np.minimum(((folded[mask, 0] - low) / width).astype(int), n_bins - 1)

    count = np.bincount(idx, minlength=n_bins)
    sums = np.bincount(idx, weights=folded[mask, 1], minlength=n_bins)

    # Compute mean
    means = np.full(n_bins, np.nan)
    valid = count > 0

    if not np.any(valid):
        raise ValueError("No valid points inside [low, high).")

    means[valid] = sums[valid] / count[valid]

    # Compute the missing values
    filled = means.copy()
    left = None
    for k, mean in enumerate(means):
        if np.isnan(mean):
            right = _next_filled_value(means, k)

            filled[k] = _get_mean_from_nearest_points(left, right)

        else:
            left = mean

    return filled
