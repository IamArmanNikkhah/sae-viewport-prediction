import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from src.sae_core.streaming_estimator import StreamingVMFState


def load_view_vectors_from_cleaned_csv(csv_path: str, dt: float):
    """
    Load unit quaternions from cleaned_quaternions.csv and convert them into:

      - x_t: view direction unit vectors on S^2 (shape: [T, 3])
      - omega_t: angular velocity vectors (shape: [T-1, 3])
      - t_uniform: time stamps assuming constant sample spacing dt

    The CSV has columns: x, y, z, w
    """

    # 1) Read the CSV (no timestamp column, just quaternions)
    df = pd.read_csv(csv_path)

    # Expect exactly these four columns from your file
    quats = df[["x", "y", "z", "w"]].to_numpy(dtype=float)

    # 2) Build a SciPy Rotation object.
    # SciPy's Rotation.from_quat expects [x, y, z, w] order,
    # which matches your CSV header.
    rot = R.from_quat(quats)

    # 3) Convert each quaternion into a 3D view direction.
    # Here we choose the forward direction as the +Z axis in the local frame.
    # You could also use [1, 0, 0] or [0, 0, -1], but this is consistent.
    forward = np.array([0.0, 0.0, 1.0])
    x_t = rot.apply(forward)  # shape: [T, 3]

    # 4) Construct a synthetic uniform time axis:
    #    t_uniform[t] = t * dt
    T = x_t.shape[0]
    t_uniform = np.arange(T, dtype=float) * dt

    # 5) Compute angular velocity vectors omega_t.
    # For each consecutive pair, we compute the relative rotation and
    # then divide the rotation vector by dt to get an angular velocity
    # in rad/s.
    #
    # rot[t] maps world -> head at time t
    # relative rotation from t to t+1:
    #   R_rel = R_{t+1} * R_t^{-1}
    # rotation vector (axis * angle) in R^3:
    #   rv = R_rel.as_rotvec()
    #
    # Then omega ≈ rv / dt
    if T >= 2:
        rot_next = rot[1:]
        rot_prev = rot[:-1]
        rel = rot_next * rot_prev.inv()
        rot_vec = rel.as_rotvec()  # shape: [T-1, 3]
        omega_t = rot_vec / dt     # approximate angular velocity
    else:
        omega_t = np.zeros((0, 3), dtype=float)

    return x_t, omega_t, t_uniform


def main():
    # ---------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------
    csv_path = "cleaned_quaternions.csv"

    # Sampling interval Δt (seconds).
    # If your Week 1 processing resampled to 60 Hz, dt = 1/60.
    dt = 1.0 / 60.0

    # Streaming vMF timescale τ (seconds)
    tau = 1.0

    print("Loading view vectors from cleaned_quaternions.csv...")
    x_t, omega_t, t_uniform = load_view_vectors_from_cleaned_csv(csv_path, dt)

    print("x_t shape:", x_t.shape)
    print("omega_t shape:", omega_t.shape)
    print("Duration (seconds):", t_uniform[-1] if len(t_uniform) > 0 else 0.0)

    # ---------------------------------------------------------------
    # Initialize streaming estimator
    # ---------------------------------------------------------------
    vmf_state = StreamingVMFState(tau=tau, dt=dt)

    kappa_list = []
    angle_error_list = []

    # For angular velocity magnitude, we will align omega_t with x_t[1:].
    omega_mag = np.linalg.norm(omega_t, axis=1) if omega_t.shape[0] > 0 else np.array([])

    # ---------------------------------------------------------------
    # Streaming loop
    # ---------------------------------------------------------------
    for i, x in enumerate(x_t):
        # Update streaming vMF state
        vmf_state.update(x)

        # Extract parameters
        mu_s, kappa_s, r_s = vmf_state.get_params()

        # Store kappa
        kappa_list.append(kappa_s)

        # Angle error between current mean direction mu_s and x_t
        # angle = arccos( clamp(mu·x, -1, 1) )  [radians] -> degrees
        dot_val = float(np.dot(mu_s, x))
        dot_val = max(-1.0, min(1.0, dot_val))
        angle_rad = np.arccos(dot_val)
        angle_deg = np.degrees(angle_rad)
        angle_error_list.append(angle_deg)

    kappa_arr = np.array(kappa_list, dtype=float)
    angle_err_arr = np.array(angle_error_list, dtype=float)

    # ---------------------------------------------------------------
    # Plot results
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # 1) κ_s over time
    axes[0].plot(t_uniform, kappa_arr)
    axes[0].set_ylabel("kappa_s")
    axes[0].set_title("Streaming vMF concentration (kappa_s) over time")

    # 2) Angle error μ_s vs x_t
    axes[1].plot(t_uniform, angle_err_arr)
    axes[1].set_ylabel("Angle error (deg)")
    axes[1].set_title("Angle between μ_s and x_t")

    # 3) Angular velocity magnitude |omega|
    # Align omega_mag (length T-1) with midpoints or t_uniform[1:]
    if omega_mag.size > 0:
        axes[2].plot(t_uniform[1:], omega_mag)
    axes[2].set_ylabel("|omega_t| (rad/s)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Angular velocity magnitude")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
