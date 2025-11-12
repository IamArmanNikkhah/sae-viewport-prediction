import numpy as np
from src.data_processing.quaternion_hygiene import (
    normalize_quaternions,
    enforce_antipodal_continuity,
)

def test_normalize_quaternions_unit_length():
    q = np.array([[2.0, 0.0, 0.0, 0.0]])
    qn = normalize_quaternions(q)
    norms = np.linalg.norm(qn, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-6)

def test_antipodal_continuity_flips_sign():
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    q2 = -q1
    qs = np.stack([q1, q2], axis=0)

    qc = enforce_antipodal_continuity(qs)

    
    assert np.allclose(qc[0], qc[1])
