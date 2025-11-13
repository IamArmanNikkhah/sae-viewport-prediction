import numpy as np
from pathlib import Path

from src.data_processing.quaternion_hygiene import load_and_clean_quaternions


def load_lo_su_file(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Adapter around load_and_clean_quaternions so tests can call load_lo_su_file.

    Args:
        path: path to a Lo & Su scanpath txt file.

    Returns:
        times: (T,) array of timestamps as float
        quats: (T, 4) array of quaternions [w, x, y, z]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    quats, times = load_and_clean_quaternions(str(path))
    # Tests expect (times, quats) in that order:
    return times.astype(float), quats.astype(float)
