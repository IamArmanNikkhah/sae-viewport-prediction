import numpy as np
from src.data_processing.slerp import slerp

def test_slerp_endpoints():
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    q2 = np.array([0.0, 1.0, 0.0, 0.0])

    assert np.allclose(slerp(q1, q2, 0.0), q1)
    assert np.allclose(slerp(q1, q2, 1.0), q2)

def test_slerp_midpoint_normalized():
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    q2 = np.array([0.0, 1.0, 0.0, 0.0])

    qm = slerp(q1, q2, 0.5)
    assert np.isclose(np.linalg.norm(qm), 1.0, atol=1e-6)
