import numpy as np
from src.data_processing.resampling import slerp, resample_sequence_to_60hz


def test_slerp_endpoints():
    q0 = np.array([0.0, 0.0, 0.0, 1.0])  # identity
    q1 = np.array([0.0, 0.0, 1.0, 0.0])  # 180° around z (example)

    q_t0 = slerp(q0, q1, t=0.0)
    q_t1 = slerp(q0, q1, t=1.0)

    assert np.allclose(q_t0, q0, atol=1e-6)
    assert np.allclose(q_t1, q1, atol=1e-6)


def test_slerp_halfway_unit_norm():
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = np.array([0.0, 0.0, 1.0, 0.0])

    q_half = slerp(q0, q1, t=0.5)

    # Should still be unit length
    assert np.isclose(np.linalg.norm(q_half), 1.0, atol=1e-6)


def test_resample_sequence_length_10s_60hz():
    # Fake data: two quats over a 10-second span
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = np.array([0.0, 0.0, 1.0, 0.0])

    quats = np.vstack([q0, q1])
    timestamps = np.array([0.0, 10.0])

    resampled_quats, t_uniform = resample_sequence_to_60hz(quats, timestamps, freq=60.0)

    # 10 seconds at 60 Hz -> 600 samples
    assert len(t_uniform) == 600
    assert resampled_quats.shape == (600, 4)