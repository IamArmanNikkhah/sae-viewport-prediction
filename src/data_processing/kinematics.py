import numpy as np
from scipy.spatial.transform import Rotation as R


def quaternion_to_optical_axis(q: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion to an optical axis unit vector x_t.

    Assumes q is [w, x, y, z]. We treat the optical axis as the
    forward vector (0, 0, 1) rotated by q.

    Args:
        q: array-like [w, x, y, z]

    Returns:
        x: (3,) unit vector
    """
    q = np.asarray(q, dtype=float)
    # Convert [w,x,y,z] → [x,y,z,w] for SciPy
    quat_xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=float)

    rot = R.from_quat(quat_xyzw)
    forward = np.array([0.0, 0.0, 1.0], dtype=float)
    x = rot.apply(forward)
    # Ensure it's unit length (test checks this)
    return x / np.linalg.norm(x)


def compute_kinematics(rot_uniform: R, dt: float = 1.0 / 60.0):
    """
    Args:
        rot_uniform: scipy Rotation of shape (N,)
        dt: time step between frames (default 60 Hz)

    Returns:
        x_t: (N, 3) optical axis vectors
        omega_t: (N-1, 3) angular velocity vectors (rotvec / dt)
    """
    # Optical axis at each time
    forward = np.array([0.0, 0.0, 1.0], dtype=float)
    x_t = rot_uniform.apply(forward)

    # Angular velocity: delta rotation between consecutive frames
    R1 = rot_uniform[:-1]
    R2 = rot_uniform[1:]
    delta_rot = R1.inv() * R2
    rotvec = delta_rot.as_rotvec()  # (N-1, 3) axis * angle

    omega_t = rotvec / dt
    return x_t, omega_t
