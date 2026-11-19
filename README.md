# Exoplanet Finder

A 1D convolutional neural network that reads NASA Kepler light curves and decides whether a transit-like dip comes from a real planet or from a look-alike, such as an eclipsing binary.

> **Status: work in progress.** Data preparation (phase folding and binning) is built and tested. Data exploration is next; the CNN is not built yet.

## Background

When a planet passes in front of its star, the star dims by a tiny fraction, and the dip repeats once per orbit. Other things make similar dips, most often two stars eclipsing each other (an eclipsing binary). Kepler watched about 150,000 stars for four years. NASA's pipeline flagged tens of thousands of repeating dips, and many of them were later labeled as planets or false positives. This project trains a model on those labels.

The approach follows Shallue & Vanderburg (2018), who used a CNN on Kepler light curves and found Kepler-90i, the eighth planet of the Kepler-90 system.

## How it works

```
Kepler light curve -> remove slow drift -> phase fold -> bin -> 1D CNN -> planet / not planet
```

- **Phase folding** (`fold`): every time stamp becomes the signed time in days from the nearest transit center, so all the dips land on top of each other at phase 0.
- **Binning** (`bin_curve`): the folded points inside a phase window are averaged into a fixed number of evenly spaced bins. Every signal becomes an array of the same length, and averaging many dips reduces the noise. Empty bins are filled from their nearest non-empty neighbours.

## Data

- **Labels:** the Kepler Objects of Interest (KOI) table (`cumulative`) from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/).
- **Light curves:** Kepler long-cadence light curves from MAST, downloaded with [lightkurve](https://lightkurve.github.io/lightkurve/).

Downloaded data goes in `data/` and is not committed.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
uv run pytest
```

## Project structure

```
src/exoplanet_finder/
  data_preparation/     detrending, phase folding and binning (folding.py: fold, bin_curve)
  download_data/        KOI table and Kepler light-curve downloads
  visualizations/       plotting helpers
  paths.py              project data paths
tests/                  pytest tests
notebooks/              data exploration
data/                   downloaded data (not committed)
```

## Reference

Shallue, C. J. & Vanderburg, A. (2018). Identifying Exoplanets with Deep Learning: A Five-planet Resonant Chain around Kepler-80 and an Eighth Planet around Kepler-90. *The Astronomical Journal*, 155(2), 94.

## License

MIT. See [LICENSE](LICENSE).
