import numpy as np
from src.data_processing.loader import normalize_quaternion, enforce_antipodal_continuity

def test_normalize_quaternion():
    q = np.array([1, 2, 3, 4])
    q_norm = normalize_quaternion(q)
    assert np.isclose(np.linalg.norm(q_norm), 1.0, atol=1e-6)

def test_antipodal_continuity():
    q1 = np.array([1, 0, 0, 0])
    q2 = np.array([-1, 0, 0, 0])
    quats = np.vstack([q1, q2])
    quats_fixed = enforce_antipodal_continuity(quats)
    dot_product = np.dot(quats_fixed[0], quats_fixed[1])
    assert dot_product >= 0

if __name__ == "__main__":
    test_normalize_quaternion()
    test_antipodal_continuity()
    print("All tests passed!")
