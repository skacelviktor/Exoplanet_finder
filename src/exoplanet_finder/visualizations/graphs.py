"""Plots for binned, folded light curves."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

HOURS_PER_DAY = 24
PPM = 1e6


def plot_binned_curve(
    values: np.ndarray,
    low: float,
    high: float,
    title: str = "",
    ax: Axes | None = None,
) -> Axes:
    """Plot a binned, folded light curve as flux change around the transit.

    The x-axis shows each bin's center in hours from the transit center. The
    y-axis shows ``value - 1`` in parts per million (ppm), so the flux must be
    normalized to 1 (as ``get_lightcurve()`` and ``flatten_lightcurve()``
    return it). A dashed vertical line marks the transit center.

    Args:
        values: Bin values from ``bin_curve()``.
        low: The ``low`` bound that was passed to ``bin_curve()``, in days.
        high: The ``high`` bound that was passed to ``bin_curve()``, in days.
        title: Plot title.
        ax: Axes to draw on, e.g. one panel of a subplot grid. If None, a new
            8 x 4 inch figure is created.

    Returns:
        The Axes the curve was drawn on.
    """
    n_bins = len(values)
    width = (high - low) / n_bins
    bin_centers_days = low + (np.arange(n_bins) + 0.5) * width

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        bin_centers_days * HOURS_PER_DAY, (values - 1) * PPM, marker="o", markersize=3
    )
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time from transit center [hours]")
    ax.set_ylabel("Flux change [ppm]")
    ax.set_title(title)
    return ax
