import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # required for 3D plots


def quaternion_to_view_vector(q: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion to a 3D unit view vector (optical axis).

    Assumes the quaternion is in [x, y, z, w] order (SciPy convention).

    We treat the view / optical axis as the forward vector [0, 0, 1]
    rotated by the quaternion.

    Args:
        q: array-like of shape (4,), quaternion [x, y, z, w]

    Returns:
        view_vec: (3,) unit vector giving the viewing direction
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {q.shape}")

    rot = R.from_quat(q)
    forward = np.array([0.0, 0.0, 1.0], dtype=float)
    view_vec = rot.apply(forward)

    # Normalize for safety
    norm = np.linalg.norm(view_vec)
    if norm == 0.0:
        raise ValueError("Zero-length view vector produced from quaternion.")
    return view_vec / norm


def calculate_angular_velocity(q_t: np.ndarray,
                               q_next: np.ndarray,
                               delta_t: float) -> np.ndarray:
    """
    Compute the angular velocity vector ω_t between two quaternions.

    Args:
        q_t:    quaternion at time t, shape (4,) [x, y, z, w]
        q_next: quaternion at time t + delta_t, shape (4,) [x, y, z, w]
        delta_t: time step between the two quaternions (in seconds)

    Returns:
        omega_t: (3,) angular velocity vector, where:
            - direction is axis of rotation
            - magnitude is speed of rotation (radians per second)
    """
    if delta_t <= 0.0:
        raise ValueError("delta_t must be positive for angular velocity.")

    q_t = np.asarray(q_t, dtype=float)
    q_next = np.asarray(q_next, dtype=float)

    if q_t.shape != (4,) or q_next.shape != (4,):
        raise ValueError("q_t and q_next must both have shape (4,)")

    R1 = R.from_quat(q_t)
    R2 = R.from_quat(q_next)

    delta_rot = R1.inv() * R2
    rotvec = delta_rot.as_rotvec()  # (3,)

    omega_t = rotvec / delta_t
    return omega_t


def compute_kinematics(rot_uniform: R,
                       dt: float = 1.0 / 60.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a Rotation object over time (e.g., from 60 Hz quaternions),
    compute the sequence of optical-axis vectors x_t and angular velocities ω_t.

    Args:
        rot_uniform: scipy Rotation object of shape (N,)
        dt:          time step between frames (seconds, default 1/60)

    Returns:
        x_t:     (N, 3) array of unit view vectors
        omega_t: (N-1, 3) array of angular velocities between frames
    """
    if not isinstance(rot_uniform, R):
        raise TypeError("rot_uniform must be a scipy.spatial.transform.Rotation object.")

    # Optical axis at each time step
    forward = np.array([0.0, 0.0, 1.0], dtype=float)
    x_t = rot_uniform.apply(forward)  # (N, 3)

    # Relative rotations between consecutive frames
    R1 = rot_uniform[:-1]
    R2 = rot_uniform[1:]
    delta_rot = R1.inv() * R2
    rotvec = delta_rot.as_rotvec()  # (N-1, 3)

    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    omega_t = rotvec / dt           # (N-1, 3)

    return x_t, omega_t


def plot_view_vector_trajectory(x_t: np.ndarray,
                                max_points: int = 300) -> None:
    """
    Plot the 3D path traced by the view vectors x_t on the unit sphere.

    Args:
        x_t: (N, 3) array of unit view vectors.
        max_points: maximum number of points to plot (for clarity).
    """
    x_t = np.asarray(x_t, dtype=float)
    if x_t.ndim != 2 or x_t.shape[1] != 3:
        raise ValueError(f"x_t must have shape (N, 3), got {x_t.shape}")

    pts = x_t[:max_points]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("View Vector Trajectory (x_t)")

    # Make axes roughly equal so path doesn't look squished
    max_range = np.max(np.ptp(pts, axis=0))
    mid = np.mean(pts, axis=0)
    for setter, m in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid):
        setter(m - max_range / 2, m + max_range / 2)

    plt.tight_layout()
    plt.show()