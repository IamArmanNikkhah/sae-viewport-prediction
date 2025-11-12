import numpy as np
from src.data_processing.kinematics import quaternion_to_optical_axis

def test_quaternion_to_optical_axis_unit_vector():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    x = quaternion_to_optical_axis(q)
    assert np.isclose(np.linalg.norm(x), 1.0, atol=1e-6)
