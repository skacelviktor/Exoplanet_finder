"""Geometry of the phase window around a transit."""


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
