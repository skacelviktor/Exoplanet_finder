"""File locations for project data.

All paths are built from the project root, so they work from any working
directory (notebooks, scripts, tests). ``PROJECT_ROOT`` is found from this
file's location, which assumes the package is used from the source tree, as
``uv sync`` installs it (editable).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
KOI_TABLE_PATH = DATA_DIR / "koi_cumulative.csv"
LIGHT_CURVES_DIR = DATA_DIR / "light_curves"
