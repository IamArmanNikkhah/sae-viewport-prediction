import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """
    Low-level spherical linear interpolation between two quaternions.

    This function works on single quaternions:
    q1, q2: [w, x, y, z]
    t: in [0, 1]

    Returns:
        Interpolated quaternion [w, x, y, z], unit length.
    """
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)

    # Normalize input
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    dot = np.dot(q1, q2)

    # Ensure shortest path
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    # If very close, use lerp + normalize
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta

    out = w1 * q1 + w2 * q2
    return out / np.linalg.norm(out)


def slerp_quaternions(quats: np.ndarray, timestamps: np.ndarray, freq: float = 60.0):
    """
    Args:
        quats: (T, 4) array of quaternions [w, x, y, z]
        timestamps: (T,) array of times (seconds or ms, consistent units)
        freq: target frequency in Hz (default 60)

    Returns:
        rot_uniform: scipy Rotation object at uniform times
        t_uniform: (N,) array of resampled timestamps
    """
    quats = np.asarray(quats, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)

    # Convert [w,x,y,z] → [x,y,z,w] for SciPy
    quats_xyzw = np.concatenate(
        [quats[:, 1:4], quats[:, 0:1]],
        axis=1,
    )

    rot = R.from_quat(quats_xyzw)
    slerp_obj = Slerp(timestamps, rot)

    dt = 1.0 / freq
    t_uniform = np.arange(timestamps[0], timestamps[-1], dt)
    rot_uniform = slerp_obj(t_uniform)

    return rot_uniform, t_uniform
