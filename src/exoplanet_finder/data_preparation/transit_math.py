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
