import numpy as np


def slerp(q0, q1, t, eps=1e-6):
    """
    Spherical Linear intERPolation between two quaternions.

    Parameters
    ----------
    q0 : array-like, shape (4,)
        Start quaternion (x, y, z, w)
    q1 : array-like, shape (4,)
        End quaternion (x, y, z, w)
    t : float
        Interpolation fraction in [0, 1]
    eps : float
        Small tolerance for numerical stability

    Returns
    -------
    q_t : ndarray, shape (4,)
        Interpolated unit quaternion
    """
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)

    # Normalize inputs just to be safe
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    # Dot product to check angle between quaternions
    dot = np.dot(q0, q1)

    # If dot < 0, flip one quat to take the shortest path on the sphere
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    # If very close, use linear interpolation (nlerp) to avoid division by zero
    if dot > 1.0 - eps:
        q_t = (1.0 - t) * q0 + t * q1
        q_t /= np.linalg.norm(q_t)
        return q_t

    # Standard SLERP formula
    theta_0 = np.arccos(dot)          # angle between q0 and q1
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * t               # angle for interpolation fraction
    sin_theta = np.sin(theta)

    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0

    q_t = s0 * q0 + s1 * q1
    q_t /= np.linalg.norm(q_t)        # re-normalize for safety
    return q_t


def resample_sequence_to_60hz(quaternions, timestamps, freq=60.0):
    """
    Resample a sequence of quaternions to a uniform frequency using SLERP.

    Parameters
    ----------
    quaternions : array-like, shape (N, 4)
        Input unit quaternions ordered in time.
    timestamps : array-like, shape (N,)
        Original timestamps (float, seconds or ms – must be consistent).
    freq : float
        Target sampling frequency in Hz (default 60.0).

    Returns
    -------
    resampled_quats : ndarray, shape (M, 4)
        Quaternions resampled at uniform frequency.
    t_uniform : ndarray, shape (M,)
        Uniform timestamps corresponding to resampled_quats.
    """
    quats = np.asarray(quaternions, dtype=np.float64)
    times = np.asarray(timestamps, dtype=np.float64)

    if quats.ndim != 2 or quats.shape[1] != 4:
        raise ValueError("quaternions must be of shape (N, 4)")

    if times.ndim != 1 or times.shape[0] != quats.shape[0]:
        raise ValueError("timestamps must be a 1D array of length N")

    if quats.shape[0] < 2:
        raise ValueError("At least two quaternions are required for resampling")

    # Ensure strictly increasing timestamps
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    quats = quats[sort_idx]

    # Build uniform time grid
    dt = 1.0 / freq
    t_start = times[0]
    t_end = times[-1]
    t_uniform = np.arange(t_start, t_end, dt)

    resampled_quats = np.empty((len(t_uniform), 4), dtype=np.float64)

    # For each target time, find its bracketing keyframes and SLERP
    for i, t in enumerate(t_uniform):
        # index of first timestamp strictly greater than t
        j = np.searchsorted(times, t, side='right')

        if j == 0:
            # Before the first sample – clamp to first
            resampled_quats[i] = quats[0]
        elif j >= len(times):
            # After the last sample – clamp to last
            resampled_quats[i] = quats[-1]
        else:
            t0, t1 = times[j - 1], times[j]
            q0, q1 = quats[j - 1], quats[j]

            # Normalize interpolation fraction in [0, 1]
            if t1 == t0:
                alpha = 0.0
            else:
                alpha = (t - t0) / (t1 - t0)

            resampled_quats[i] = slerp(q0, q1, alpha)

    return resampled_quats, t_uniform