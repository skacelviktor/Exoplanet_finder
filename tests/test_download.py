"""Tests for exoplanet_finder.download_data.download.

No test touches the network: NASA's archive, MAST and the waits between
retries are all replaced with small fakes.
"""

import lightkurve as lk
import numpy as np
import pandas as pd
import pytest
import requests
from lightkurve.utils import LightkurveError

from exoplanet_finder.download_data import download
from exoplanet_finder.download_data.download import get_lightcurve, load_from_api

FAKE_TAP_URL = "https://archive.invalid/TAP/sync?"


# --------------------------------------------------------- load_from_api()


@pytest.fixture
def fake_archive(monkeypatch):
    """Answer URL reads with a small table and record each requested URL."""
    requested_urls = []
    real_read_csv = pd.read_csv

    def read_csv(source, *args, **kwargs):
        if str(source).startswith("http"):
            requested_urls.append(source)
            return pd.DataFrame({"kepid": [11904151], "koi_period": [0.837]})
        return real_read_csv(source, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", read_csv)
    return requested_urls


def test_load_from_api_reads_existing_file_without_downloading(tmp_path, fake_archive):
    # Arrange
    cache = tmp_path / "koi.csv"
    pd.DataFrame({"kepid": [1, 2]}).to_csv(cache, index=False)

    # Act
    table = load_from_api(FAKE_TAP_URL, "select kepid from cumulative", cache)

    # Assert
    assert table["kepid"].tolist() == [1, 2]
    assert fake_archive == []


def test_load_from_api_downloads_saves_and_then_reuses_the_file(tmp_path, fake_archive):
    # Arrange
    cache = tmp_path / "koi.csv"
    query = "select kepid, koi_period from cumulative"

    # Act: the first call downloads; the second (with a str path) reads the file.
    first = load_from_api(FAKE_TAP_URL, query, cache)
    second = load_from_api(FAKE_TAP_URL, query, str(cache))

    # Assert
    assert fake_archive == [
        FAKE_TAP_URL + "query=select+kepid%2C+koi_period+from+cumulative&format=csv"
    ]
    assert cache.is_file()
    pd.testing.assert_frame_equal(first, second)


# ---------------------------------------------- _delete_downloaded_files()


class FakeSearch:
    """Stands in for lk.SearchResult: a table of rows plus download_all()."""

    def __init__(self, rows, download_result=None, download_error=None):
        self.table = rows
        self.download_result = download_result
        self.download_error = download_error
        self.download_dir = None

    def download_all(self, download_dir):
        self.download_dir = download_dir
        if self.download_error is not None:
            raise self.download_error
        return self.download_result


def make_download_folder(base, obs_id):
    folder = base / "mastDownload" / "Kepler" / obs_id
    folder.mkdir(parents=True)
    (folder / "lightcurve.fits").write_text("partial download")
    return folder


def test_delete_downloaded_files_removes_only_this_searchs_folders(tmp_path):
    # Arrange
    q1 = make_download_folder(tmp_path, "kplr001_q1")
    q2 = make_download_folder(tmp_path, "kplr001_q2")
    other_star = make_download_folder(tmp_path, "kplr002_q1")
    search = FakeSearch(
        [
            {"obs_collection": "Kepler", "obs_id": "kplr001_q1"},
            {"obs_collection": "Kepler", "obs_id": "kplr001_q2"},
            {"obs_collection": "Kepler", "obs_id": "never_downloaded"},
        ]
    )

    # Act
    download._delete_downloaded_files(search, tmp_path)

    # Assert
    assert not q1.exists()
    assert not q2.exists()
    assert (other_star / "lightcurve.fits").is_file()


# ------------------------------------------------- _search_and_download()


def test_search_and_download_asks_mast_for_kepler_long_cadence(monkeypatch, tmp_path):
    # Arrange
    search = FakeSearch([], download_result="quarters")
    calls = []

    def search_lightcurve(target, **kwargs):
        calls.append((target, kwargs))
        return search

    monkeypatch.setattr(lk, "search_lightcurve", search_lightcurve)

    # Act
    result = download._search_and_download(11904151, tmp_path)

    # Assert
    assert result == "quarters"
    assert calls == [
        ("KIC 11904151", {"mission": "Kepler", "author": "Kepler", "cadence": "long"})
    ]
    assert search.download_dir == str(tmp_path)


def test_search_and_download_deletes_partial_files_and_reraises(monkeypatch, tmp_path):
    # Arrange
    partial = make_download_folder(tmp_path, "kplr001_q1")
    search = FakeSearch(
        [{"obs_collection": "Kepler", "obs_id": "kplr001_q1"}],
        download_error=LightkurveError("file may be corrupt"),
    )
    monkeypatch.setattr(lk, "search_lightcurve", lambda *args, **kwargs: search)

    # Act / Assert
    with pytest.raises(LightkurveError):
        download._search_and_download(1, tmp_path)
    assert not partial.exists()


# ------------------------------------------------ _download_with_retries()


@pytest.fixture
def waits(monkeypatch):
    """Record the requested sleep times instead of actually sleeping."""
    recorded = []
    monkeypatch.setattr(download.time, "sleep", recorded.append)
    return recorded


def fake_attempts(monkeypatch, outcomes):
    """Make _search_and_download raise or return each outcome in turn."""
    attempts = []

    def search_and_download(kepid, path):
        attempts.append(kepid)
        outcome = outcomes[len(attempts) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(download, "_search_and_download", search_and_download)
    return attempts


def test_download_with_retries_retries_network_errors_then_succeeds(monkeypatch, waits):
    # Arrange
    outcomes = [
        requests.exceptions.ConnectionError(),
        requests.exceptions.Timeout(),
        "quarters",
    ]
    attempts = fake_attempts(monkeypatch, outcomes)

    # Act
    result = download._download_with_retries(1, "unused")

    # Assert: waits grow with each attempt (5 s, then 10 s).
    assert result == "quarters"
    assert len(attempts) == 3
    assert waits == [download.RETRY_WAIT_SECONDS * 1, download.RETRY_WAIT_SECONDS * 2]


def test_download_with_retries_gives_up_after_max_attempts(monkeypatch, waits):
    # Arrange
    outcomes = [requests.exceptions.ConnectionError()] * download.MAX_ATTEMPTS
    attempts = fake_attempts(monkeypatch, outcomes)

    # Act / Assert
    with pytest.raises(requests.exceptions.ConnectionError):
        download._download_with_retries(1, "unused")
    assert len(attempts) == download.MAX_ATTEMPTS
    assert len(waits) == download.MAX_ATTEMPTS - 1


def test_download_with_retries_does_not_retry_other_errors(monkeypatch, waits):
    # Arrange
    attempts = fake_attempts(monkeypatch, [LightkurveError("corrupt"), "never reached"])

    # Act / Assert
    with pytest.raises(LightkurveError):
        download._download_with_retries(1, "unused")
    assert len(attempts) == 1
    assert waits == []


# --------------------------------------------------------- get_lightcurve()


def test_get_lightcurve_raises_lookup_error_when_nothing_is_found(monkeypatch):
    # Arrange
    monkeypatch.setattr(download, "_download_with_retries", lambda kepid, path: None)

    # Act / Assert
    with pytest.raises(LookupError, match="KIC 42"):
        get_lightcurve(42, "unused")


def test_get_lightcurve_stitches_normalized_quarters_and_keeps_three_columns(monkeypatch):
    # Arrange: two quarters at very different brightness, one with an extra column.
    quarter_1 = lk.LightCurve(
        time=np.arange(0.0, 1.0, 0.1), flux=np.full(10, 100.0), flux_err=np.full(10, 1.0)
    )
    quarter_1["centroid_col"] = np.zeros(10)
    quarter_2 = lk.LightCurve(
        time=np.arange(1.0, 2.0, 0.1), flux=np.full(10, 200.0), flux_err=np.full(10, 2.0)
    )
    collection = lk.LightCurveCollection([quarter_1, quarter_2])
    monkeypatch.setattr(download, "_download_with_retries", lambda kepid, path: collection)

    # Act
    lc = get_lightcurve(1, "unused")

    # Assert
    assert len(lc) == 20
    np.testing.assert_allclose(lc.flux.value, 1.0)
    np.testing.assert_allclose(lc.flux_err.value, 0.01)
    assert set(lc.colnames) == {"time", "flux", "flux_err"}
