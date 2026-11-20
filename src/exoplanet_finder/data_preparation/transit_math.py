"""Geometry of the phase window around a transit."""

import numpy as np


def transit_window_bounds(
    duration_days: float, n_durations: float
) -> tuple[float, float]:
    """Return the ``(low, high)`` bounds of a phase window centered on the transit.

    The window is ``n_durations`` transit durations wide in total and
    symmetric around phase 0. For example, a 0.2-day transit with
    ``n_durations=3`` gives ``(-0.3, 0.3)``. Pass the result to
    ``bin_curve()`` as ``low`` and ``high``.

    Args:
        duration_days: Transit duration in days.
        n_durations: Total window width, in units of the transit duration.

    Returns:
        ``(low, high)`` in days, with ``low == -high``.
    """
    high = duration_days * (n_durations / 2)
    low = -high

    return (low, high)


def orbit_window_bounds(period: float) -> tuple[float, float]:
    """Return the ``(low, high)`` bounds of a window covering the whole orbit.

    The window is one full period wide and centered on the transit, which is
    exactly the range ``[-period / 2, period / 2)`` that ``fold()`` maps every
    point into, so ``bin_curve()`` keeps all the data. For a circular orbit,
    a secondary eclipse (if there is one) lands at both ends of this window:
    phases ``-period / 2`` and ``+period / 2`` are the same moment.

    Args:
        period: Orbital period in days.

    Returns:
        ``(-period / 2, period / 2)`` in days.
    """
    half_period = period / 2
    return (-half_period, half_period)


def transit_numbers(time: np.ndarray, period: float, t0: float) -> np.ndarray:
    """Return the number of the nearest transit for every time stamp.

    Transit 0 is the one at ``t0``, transit 1 the next one, transit -1 the
    one before, and so on. Each point belongs to the transit it is closest
    to (the same "nearest transit" idea that ``fold()`` uses), so a point
    just before a transit center gets that transit's number, not the
    previous one. Use ``transit_numbers(...) % 2`` to split the dips into
    even (0) and odd (1) ones.

    Args:
        time: Time stamps in days (BKJD), any shape.
        period: Orbital period in days.
        t0: Mid-transit time of transit 0, in days.

    Returns:
        An integer array with the same shape as ``time``.
    """
    return np.floor((time - t0) / period + 0.5).astype(int)
