import numpy as np
import pytest
from pathlib import Path

from src.data_processing.dataset_io import load_lo_su_file

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SCANPATH_DIR = DATA_DIR / "scanpaths"

def _get_hscan_files():
    if not SCANPATH_DIR.exists():
        return []
    return sorted(SCANPATH_DIR.glob("Hscanpath_*.txt"))

HSCAN_FILES = _get_hscan_files()

@pytest.mark.skipif(
    len(HSCAN_FILES) == 0,
    reason="No Hscanpath_*.txt files found in data/scanpaths."
)
@pytest.mark.parametrize("filepath", HSCAN_FILES)
def test_load_lo_su_file_shapes_and_types(filepath):
    times, quats = load_lo_su_file(filepath)

    assert isinstance(times, np.ndarray)
    assert isinstance(quats, np.ndarray)

    assert times.ndim == 1
    assert quats.ndim == 2
    assert quats.shape[1] == 4
    assert quats.shape[0] == times.shape[0]

    assert times.dtype.kind == "f"
    assert quats.dtype.kind == "f"

    assert not np.isnan(times).any()
    assert not np.isnan(quats).any()
