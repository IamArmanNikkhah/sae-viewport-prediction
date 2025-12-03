# scripts/make_npy_from_cleaned.py

import os
import numpy as np
import pandas as pd


INPUT_CSV = "cleaned_quaternions.csv"
OUTPUT_DIR = "data/processed"
OUTPUT_NPY = os.path.join(OUTPUT_DIR, "quaternions_sequence0.npy")


def main():
    print(f"Loading {INPUT_CSV} ...")
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Could not find {INPUT_CSV}. "
            "Make sure cleaned_quaternions.csv is in the project root."
        )

    df = pd.read_csv(INPUT_CSV)

    expected_cols = ["x", "y", "z", "w"]
    for c in expected_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {INPUT_CSV}")

    quats = df[expected_cols].to_numpy(dtype=np.float32)  # (T, 4)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(OUTPUT_NPY, quats)

    print(f"Saved quaternions to {OUTPUT_NPY} with shape {quats.shape}")


if __name__ == "__main__":
    main()
