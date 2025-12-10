import os
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def latlon_to_xyz(lat, lon):
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=1)

def resample_to_60hz(xyz, timestamps):
    ts = timestamps - timestamps[0]
    t_new = np.arange(0, ts[-1], 1000 / 60)  # 60 Hz → ~16.666 ms
    x_new = np.zeros((len(t_new), 3), dtype=np.float32)

    for i in range(3):
        x_new[:, i] = np.interp(t_new, ts, xyz[:, i])

    return x_new

def compute_omega(xyz_60):
    omega = np.zeros_like(xyz_60)
    omega[1:] = xyz_60[1:] - xyz_60[:-1]
    return omega

def process_one_file(path, seq_id):
    print(f"Processing {path.name} (seq {seq_id}) ...")

    df = pd.read_csv(
        path,
        sep=r"[, ]+",
        engine="python",
        names=["idx", "longitude", "latitude", "timestamp"],
        skiprows=1
    )

    lon = df["longitude"].astype(float).values
    lat = df["latitude"].astype(float).values
    ts  = df["timestamp"].astype(float).values

    xyz = latlon_to_xyz(lat, lon)
    xyz_60 = resample_to_60hz(xyz, ts)
    omega = compute_omega(xyz_60)

    features = np.concatenate([xyz_60, omega], axis=1).astype(np.float32)
    out_path = OUTPUT_DIR / f"sequence_{seq_id}.npy"
    np.save(out_path, features)

    print(f"Saved {out_path} with shape {features.shape}")

def main():
    raw_dir = Path("data/Scanpaths")
    files = sorted(raw_dir.glob("Hscanpath_*.txt"))

    if not files:
        raise FileNotFoundError("No Hscanpath_*.txt files in data/Scanpaths")

    for i, f in enumerate(files):
        process_one_file(f, i)

    print("\n✓ All files processed!")

if __name__ == "__main__":
    main()
