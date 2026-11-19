"""Download and cache the KOI table and Kepler light curves from NASA archives."""

import shutil
import time
from pathlib import Path
from urllib.parse import urlencode

import lightkurve as lk
import pandas as pd
import requests
from lightkurve.utils import LightkurveError

MAX_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 5


def load_from_api(default_url: str, query: str, path: str | Path) -> pd.DataFrame:
    """Return a NASA Exoplanet Archive table, downloading it only once.

    If ``path`` exists, the table is read from that CSV file and no request is
    made, even if ``query`` has changed since the file was saved. Delete the
    file to download again. Otherwise the query is sent to the archive's TAP
    service and the result is saved to ``path`` before it is returned.

    Args:
        default_url: Base URL of the TAP service, ending in ``?``, e.g.
            ``"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"``. The
            URL-encoded query and ``format=csv`` are appended to it.
        query: ADQL query, e.g. ``"select kepid, koi_period from cumulative"``.
        path: CSV file used as the cache. Its folder must exist.

    Returns:
        The table as a DataFrame.

    Raises:
        urllib.error.HTTPError: If the archive rejects the query (e.g. an
            unknown column gives HTTP 400). Nothing is cached in that case.
    """
    if isinstance(path, str):
        path = Path(path)

    if path.exists():
        return pd.read_csv(path)

    url = default_url + urlencode({"query": query, "format": "csv"})
    data_frame = pd.read_csv(url)

    data_frame.to_csv(path, index=False)

    return data_frame


def _delete_downloaded_files(search: lk.SearchResult, directory: Path | str) -> None:
    """Delete the files a failed ``download_all()`` left behind.

    lightkurve stores each file under
    ``<directory>/mastDownload/<obs_collection>/<obs_id>/`` and reuses files it
    finds there, so a partly downloaded (corrupt) file would break every
    later attempt. This removes the folder of every row in ``search``;
    folders that don't exist are skipped.

    Args:
        search: The search whose downloaded files should be removed.
        directory: The ``download_dir`` that was passed to ``download_all()``.
    """
    for row in search.table:  # ty: ignore[not-iterable]  (astropy Table iterates via __getitem__)
        folder = (
            Path(directory) / "mastDownload" / row["obs_collection"] / row["obs_id"]
        )
        shutil.rmtree(folder, ignore_errors=True)


def _search_and_download(
    kepid: int, path: str | Path
) -> lk.LightCurveCollection | None:
    """Search MAST for a star's Kepler long-cadence light curves and download them.

    Args:
        kepid: Kepler Input Catalog ID of the star (KOI ``kepid``).
        path: Folder to download into. Files already there are reused.

    Returns:
        One light curve per quarter, or None if the search found nothing.

    Raises:
        LightkurveError: If a downloaded file can't be read. The partial
            files are deleted first, so the next attempt starts clean.
    """
    search: lk.SearchResult = lk.search_lightcurve(
        f"KIC {kepid}", mission="Kepler", author="Kepler", cadence="long"
    )
    try:
        return search.download_all(download_dir=str(path))
    except LightkurveError:
        _delete_downloaded_files(search, path)
        raise


def _download_with_retries(
    kepid: int, path: str | Path
) -> lk.LightCurveCollection | None:
    """Call ``_search_and_download()``, retrying on network errors.

    After a failed attempt it waits ``RETRY_WAIT_SECONDS * attempt`` seconds
    (5 s, then 10 s) and tries again, up to ``MAX_ATTEMPTS`` attempts in
    total. Only ``requests`` network errors are retried; any other error is
    raised immediately.

    Args:
        kepid: Kepler Input Catalog ID of the star.
        path: Folder to download into.

    Returns:
        One light curve per quarter, or None if the search found nothing.

    Raises:
        requests.exceptions.RequestException: If every attempt fails with a
            network error (the last error is raised).
    """
    attempt = 1
    while True:
        try:
            return _search_and_download(kepid, path)
        except requests.exceptions.RequestException:
            if attempt >= MAX_ATTEMPTS:
                raise
            time.sleep(RETRY_WAIT_SECONDS * attempt)
            attempt += 1


def get_lightcurve(kepid: int, path: str | Path) -> lk.LightCurve:
    """Download a star's Kepler light curve and join all quarters into one.

    Args:
        kepid: Kepler Input Catalog ID of the star (KOI ``kepid``).
        path: Folder for the downloaded files, e.g. ``LIGHT_CURVES_DIR``.
            Files already there are reused instead of downloaded again.

    Returns:
        A light curve with only ``time`` (BKJD days), ``flux`` and
        ``flux_err``. Each quarter is divided by its own median flux before
        the quarters are joined, so the flux is close to 1 everywhere.

    Raises:
        LookupError: If MAST has no Kepler long-cadence light curves for the
            star.
        requests.exceptions.RequestException: If the download keeps failing
            with network errors.
        LightkurveError: If a downloaded file can't be read.
    """
    lcs = _download_with_retries(kepid, path)

    if lcs is None:
        raise LookupError(f"No Kepler long-cadence light curves for KIC {kepid}")

    lc: lk.LightCurve = lcs.stitch()
    return lk.LightCurve(time=lc.time, flux=lc.flux, flux_err=lc.flux_err)
