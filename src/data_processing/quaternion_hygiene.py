import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def normalize_quaternions(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize quaternions to unit length.

    Args:
        q: array of shape (..., 4) representing quaternions [w, x, y, z].
        eps: small epsilon to avoid division by zero.

    Returns:
        Normalized quaternions with same shape as q.
    """
    q = np.asarray(q, dtype=float)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    norms = np.maximum(norms, eps)
    return q / norms


def enforce_antipodal_continuity(q: np.ndarray) -> np.ndarray:
    """
    Enforce antipodal continuity on a sequence of quaternions.

    Quaternions q and -q represent the same rotation. This function flips the
    sign of quaternions when necessary so consecutive quaternions are as close
    as possible in Euclidean distance.

    Args:
        q: (T, 4) array of quaternions [w, x, y, z].

    Returns:
        (T, 4) array with sign flips applied for continuity.
    """
    q = np.asarray(q, dtype=float).copy()
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] = -q[i]
    return q


def load_and_clean_quaternions(file_path: str):
    """

    Loads one Hscanpath_*.txt file, converts lon/lat to quaternions,
    normalizes, enforces antipodal continuity, and sorts by timestamp.

    Args:
        file_path: path to a single scanpath txt file.

    Returns:
        quats: (T, 4) array of quaternions [w, x, y, z]
        times: (T,) array of timestamps as float
    """
    
    df = pd.read_csv(file_path, skipinitialspace=True)

    # Strip any stray spaces from column names just in case
    df.columns = [c.strip() for c in df.columns]

    # Now columns should be: "Idx", "longitude", "latitude", "start timestamp"
    lon = df["longitude"].astype(float).to_numpy()
    lat = df["latitude"].astype(float).to_numpy()
    
    # Use the last column as timestamp (or explicitly "start timestamp")
    times = df[df.columns[-1]].astype(float).to_numpy()

    # Convert lon/lat → rotation using Euler angles
    # Assume: yaw = longitude, pitch = latitude, roll = 0
    angles = np.vstack([lon, lat, np.zeros_like(lon)]).T  # (T, 3)

    # SciPy Rotation.as_quat gives [x, y, z, w]
    rot = R.from_euler("YXZ", angles, degrees=False)
    quats_xyzw = rot.as_quat()  # (T, 4)

    # Convert to [w, x, y, z] convention
    quats_wxyz = np.concatenate(
        [quats_xyzw[:, 3:4], quats_xyzw[:, 0:3]],
        axis=1,
    )

    # Sort by timestamp, remove duplicate timestamps
    order = np.argsort(times)
    times = times[order]
    quats_wxyz = quats_wxyz[order]

    _, uniq_idx = np.unique(times, return_index=True)
    times = times[uniq_idx]
    quats_wxyz = quats_wxyz[uniq_idx]

    # Hygiene: normalize + antipodal continuity
    quats_wxyz = normalize_quaternions(quats_wxyz)
    quats_wxyz = enforce_antipodal_continuity(quats_wxyz)

    return quats_wxyz, times
