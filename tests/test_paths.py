"""Tests for exoplanet_finder.paths."""

from exoplanet_finder.paths import (
    DATA_DIR,
    KOI_TABLE_PATH,
    LIGHT_CURVES_DIR,
    PROJECT_ROOT,
)


def test_project_root_is_the_folder_with_pyproject():
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


def test_data_paths_live_in_the_data_folder():
    assert DATA_DIR == PROJECT_ROOT / "data"
    assert KOI_TABLE_PATH.parent == DATA_DIR
    assert LIGHT_CURVES_DIR.parent == DATA_DIR
