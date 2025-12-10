

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.data_processing.kinematics import (
    compute_kinematics,
    quaternion_to_view_vector,
    calculate_angular_velocity,
)


def test_quaternion_to_view_vector_unit():
    """
    quaternion_to_view_vector(q) should return a 3D unit vector.
    """
    q_identity = np.array([0.0, 0.0, 0.0, 1.0])

    v = quaternion_to_view_vector(q_identity)

    # Shape (3,)
    assert v.shape == (3,)
    # Unit length
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-6)


def test_compute_kinematics_shapes():
    """
    compute_kinematics should return:
      - x_t with shape (N, 3)
      - omega_t with shape (N-1, 3)
    """
    # Make a small sequence of 3 quaternions (e.g., tiny rotations around z)
    angles = np.array([0.0, 0.1, 0.2])  # radians
    rot_seq = R.from_euler("z", angles, degrees=False)

    dt = 1.0 / 60.0
    x_t, omega_t = compute_kinematics(rot_seq, dt=dt)

    # N = 3
    assert x_t.shape == (3, 3)
    # N-1 = 2
    assert omega_t.shape == (2, 3)


def test_static_sequence_zero_angular_velocity():
    """
    For a static (non-moving) sequence, angular velocity must be zero.
    """
    # 5 identical quaternions (no motion)
    q_identity = np.array([0.0, 0.0, 0.0, 1.0])
    quats = np.tile(q_identity, (5, 1))  # shape (5, 4)

    rot_uniform = R.from_quat(quats)
    dt = 1.0 / 60.0

    x_t, omega_t = compute_kinematics(rot_uniform, dt=dt)

    # Angular velocity should be (N-1, 3) of all zeros
    assert omega_t.shape == (4, 3)
    assert np.allclose(omega_t, 0.0, atol=1e-8)


def test_calculate_angular_velocity_static_pair():
    """
    Directly test calculate_angular_velocity for a static pair.
    """
    q_t = np.array([0.0, 0.0, 0.0, 1.0])
    q_next = np.array([0.0, 0.0, 0.0, 1.0])
    dt = 1.0 / 60.0

    omega = calculate_angular_velocity(q_t, q_next, dt)

    assert omega.shape == (3,)
    assert np.allclose(omega, 0.0, atol=1e-8)