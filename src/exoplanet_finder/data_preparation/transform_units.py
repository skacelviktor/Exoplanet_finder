"""Unit conversions for KOI table columns."""

import pandas as pd


def hours_to_days(col: pd.Series) -> pd.Series:
    """Convert durations from hours to days.

    KOI ``koi_duration`` is in hours, while periods, transit times and the
    folded phase are all in days.

    Args:
        col: Durations in hours.

    Returns:
        A new Series with the same durations in days and the same index and
        name. The input is not modified.
    """
    return col / 24
