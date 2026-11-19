"""Remove slow brightness drift from light curves while protecting the transits."""

import lightkurve as lk
import pandas as pd


def flatten_lightcurve(
    lc: lk.LightCurve,
    periods: pd.Series,
    transit_times: pd.Series,
    durations_days: pd.Series,
) -> lk.LightCurve:
    """Remove slow drift from a light curve without flattening its transits.

    Builds a mask that is True during every transit of every listed signal
    (``lc.create_transit_mask``) and passes it to ``lc.flatten()``. The drift
    is then estimated from the out-of-transit points only, so the dips keep
    their depth. List every signal (KOI) known on this star; a dip that is
    not masked can be partly flattened away.

    Args:
        lc: Light curve to detrend, e.g. from ``get_lightcurve()``.
        periods: Orbital period of each signal on the star, in days.
        transit_times: Mid-transit time of each signal (KOI ``koi_time0bk``),
            in BKJD days.
        durations_days: Transit duration of each signal, in days. KOI
            ``koi_duration`` is in hours; convert it with ``hours_to_days()``.

    Returns:
        A new light curve with the drift divided out (flux close to 1 outside
        the transits). The input light curve is not modified.
    """
    mask = lc.create_transit_mask(
        period=periods, transit_time=transit_times, duration=durations_days
    )

    return lc.flatten(mask=mask)
