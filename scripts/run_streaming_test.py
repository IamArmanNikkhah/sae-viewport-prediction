# scripts/run_streaming_test.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from src.data_processing.resampling import resample_sequence_to_60hz
from src.data_processing.kinematics import compute_kinematics
from src.sae_core.streaming_estimator import StreamingVMFState


def load_view_vectors_from_cleaned_csv(csv_path: str):
    """
    Load cleaned quaternions from CSV and convert to:
      - 60 Hz uniform quaternions
      - view vectors x_t (N, 3)
      - angular velocities omega_t (N-1, 3)
      - uniform timestamps t_uniform (N,)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python3 src/data_processing/loader.py` first."
        )

    df = pd.read_csv(csv_path)

    # Expect columns: timestamp, x, y, z, w (from loader.py)
    timestamps = df["timestamp"].to_numpy(dtype=float)
    quats = df[["x", "y", "z", "w"]].to_numpy(dtype=float)

    # Resample to 60 Hz using your SLERP-based function
    quats_60, t_uniform = resample_sequence_to_60hz(quats, timestamps, freq=60.0)

    # Convert to Rotation
    rot_uniform = R.from_quat(quats_60)

    # Use Week 1 kinematics to get x_t and omega_t
    dt = 1.0 / 60.0
    x_t, omega_t = compute_kinematics(rot_uniform, dt=dt)

    return x_t, omega_t, t_uniform


def main():
    csv_path = "cleaned_quaternions.csv"

    print(f"Loading view vectors from {csv_path}...")
    x_t, omega_t, t_uniform = load_view_vectors_from_cleaned_csv(csv_path)
    dt = 1.0 / 60.0

    print(f"x_t shape: {x_t.shape}")
    print(f"omega_t shape: {omega_t.shape}")
    print(f"Duration (seconds): {t_uniform[-1] - t_uniform[0]:0.2f}")

    # ------------------------------------------------------------------
    # Set up streaming estimator
    # ------------------------------------------------------------------
    tau = 1.0  # seconds
    state = StreamingVMFState(tau=tau, dt=dt)

    kappas = []
    ang_errors_deg = []
    times = t_uniform

    for i, x in enumerate(x_t):
        # Update streaming state with current view vector
        state.update(x)

        mu_s, kappa_s, r_s = state.get_params()
        kappas.append(kappa_s)

        # Angle between current mean direction and x_t
        dot = float(np.clip(np.dot(mu_s, x), -1.0, 1.0))
        ang_err = np.arccos(dot)  # radians
        ang_errors_deg.append(np.degrees(ang_err))

    kappas = np.array(kappas)
    ang_errors_deg = np.array(ang_errors_deg)

    # For correlation, compute angular speed magnitude ||omega_t||
    omega_speed = np.linalg.norm(omega_t, axis=1)
    # Pad to length N by repeating last value
    if len(omega_speed) < len(times):
        omega_speed = np.concatenate([omega_speed, omega_speed[-1:]])

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(times, kappas)
    axs[0].set_ylabel(r"$\kappa_s$")
    axs[0].set_title("Streaming vMF Concentration over Time")

    axs[1].plot(times, ang_errors_deg)
    axs[1].set_ylabel("Angle error (deg)")
    axs[1].set_title(r"Angle between $\mu_s$ and current x_t")

    axs[2].plot(times, omega_speed)
    axs[2].set_ylabel("|omega_t| (rad/s)")
    axs[2].set_xlabel("time (s)")
    axs[2].set_title("Angular velocity magnitude")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


