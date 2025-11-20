
import numpy as np
from scipy.spatial.transform import Rotation as R
from src.data_processing.resampling import slerp, resample_sequence_to_60hz


def test_slerp_endpoints():
    """
    slerp(q0, q1, t=0) should return q0
    slerp(q0, q1, t=1) should return q1
    """
    q0 = np.array([0.0, 0.0, 0.0, 1.0])   # identity
    q1 = np.array([0.0, 0.0, 1.0, 0.0])   # 180° rotation about z-axis

    q_t0 = slerp(q0, q1, t=0.0)
    q_t1 = slerp(q0, q1, t=1.0)

    assert np.allclose(q_t0, q0, atol=1e-6)
    assert np.allclose(q_t1, q1, atol=1e-6)


def test_slerp_halfway_unit_norm():
    """
    Basic halfway test — quaternion should remain unit-length.
    """
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = np.array([0.0, 0.0, 1.0, 0.0])

    q_half = slerp(q0, q1, t=0.5)

    assert np.isclose(np.linalg.norm(q_half), 1.0, atol=1e-6)


def test_slerp_halfway_exact_rotation():
    """
    Strong acceptance test:
    slerp(q0, q1, 0.5) must be the rotation exactly halfway
    along the shortest arc.
    """
    q0 = np.array([0.0, 0.0, 0.0, 1.0])   # identity
    q1 = np.array([0.0, 0.0, 1.0, 0.0])   # 180° rotation around z

    q_half = slerp(q0, q1, t=0.5)

    # Convert to Rotation objects
    R0 = R.from_quat(q0)
    R1 = R.from_quat(q1)
    R_half = R.from_quat(q_half)

    # total rotation between R0 & R1 → should be π radians
    total_angle = (R0.inv() * R1).magnitude()

    # halfway rotation should be π/2
    half_angle = (R0.inv() * R_half).magnitude()

    assert np.isclose(half_angle, total_angle / 2.0, atol=1e-6), \
        "SLERP halfway quaternion is not at half the rotational distance."


def test_resample_sequence_length_10s_60hz():
    """
    10 seconds of data resampled to 60 Hz should produce 600 samples.
    """
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = np.array([0.0, 0.0, 1.0, 0.0])

    quats = np.vstack([q0, q1])
    timestamps = np.array([0.0, 10.0])

    resampled_quats, t_uniform = resample_sequence_to_60hz(quats, timestamps, freq=60.0)

    assert len(t_uniform) == 600
    assert resampled_quats.shape == (600, 4)