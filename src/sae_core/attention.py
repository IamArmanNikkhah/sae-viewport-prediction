from __future__ import annotations

import numpy as np


def attention_direction(
    x_t: np.ndarray,
    omega_t: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Predict where the user *should* be looking next using Rodrigues' rotation.

    Inputs:
        x_t      : current viewing direction (3D unit vector, shape (3,))
        omega_t  : angular velocity (rad/s, 3D vector)
        dt       : time step in seconds

    We treat omega_t as giving:
        - axis of rotation (its direction),
        - speed of rotation |omega_t| in rad/s.

    Total rotation angle:
        theta = ||omega_t|| * dt

    Axis:
        u = omega_t / ||omega_t||.

    Rodrigues' formula:
        v_t = x_t * cos(theta)
              + (u x x_t) * sin(theta)
              + u * (u · x_t) * (1 - cos(theta))
    """
    x = np.asarray(x_t, dtype=float)
    w = np.asarray(omega_t, dtype=float)

    if x.shape != (3,):
        raise ValueError(f"x_t must have shape (3,), got {x.shape}")
    if w.shape != (3,):
        raise ValueError(f"omega_t must have shape (3,), got {w.shape}")
    if dt < 0.0:
        raise ValueError("dt must be non-negative.")

    w_norm = float(np.linalg.norm(w))

    # No motion or zero dt => no rotation
    if w_norm == 0.0 or dt == 0.0:
        return x.copy()

    theta = w_norm * dt           # angle in radians
    u = w / w_norm                # unit axis

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    term1 = x * cos_theta
    term2 = np.cross(u, x) * sin_theta
    term3 = u * (np.dot(u, x)) * (1.0 - cos_theta)

    v_t = term1 + term2 + term3

    # Re-normalize to unit length for safety
    v_norm = np.linalg.norm(v_t)
    if v_norm > 0.0:
        v_t = v_t / v_norm

    return v_t


def motion_adaptive_concentration(
    omega_t: np.ndarray,
    lambda_base: float = 10.0,
    gamma: float = 0.1,
) -> float:
    """
    Motion-adaptive concentration:

        lambda_s = lambda_base * (1 + gamma * |omega_t|)

    Inputs:
        omega_t      : angular velocity vector (rad/s, shape (3,))
        lambda_base  : baseline concentration (e.g. 10.0)
        gamma        : motion sensitivity (e.g. 0.1)

    Output:
        lambda_s     : scalar motion-adaptive concentration.
    """
    w = np.asarray(omega_t, dtype=float)
    if w.shape != (3,):
        raise ValueError(f"omega_t must have shape (3,), got {w.shape}")
    if lambda_base < 0.0:
        raise ValueError("lambda_base must be non-negative.")
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative.")

    w_norm = float(np.linalg.norm(w))
    lambda_s = lambda_base * (1.0 + gamma * w_norm)
    return float(lambda_s)
