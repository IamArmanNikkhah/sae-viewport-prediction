# src/data/scanpath_dataset.py

import os
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.sae_core.attention import attention_direction


class ScanpathSequenceDataset(Dataset):
    """
    Dataset for Week 5, Option A:
      - Use a SINGLE long quaternion sequence from cleaned_quaternions.csv
      - Converted to data/processed/quaternions_sequence0.npy

    We:
      * Load quaternions (x, y, z, w), shape (T, 4)
      * Convert to 3D directions (unit vectors) from the vector part (x, y, z)
      * Compute angular velocity via finite differences
      * Use attention_direction(x_t, omega_t, dt=1/60) to get a smoothed direction
      * Build future directions for 250 / 500 / 1000 ms
      * Create sliding windows of length seq_len
    """

    def __init__(
        self,
        root: str = "data/processed",
        npy_name: str = "quaternions_sequence0.npy",
        seq_len: int = 120,
        dt_hz: float = 60.0,
    ):
        """
        Args:
            root: directory containing the .npy file
            npy_name: filename of the .npy with quaternions, shape (T, 4)
            seq_len: sliding window length (timesteps)
            dt_hz: sampling rate in Hz (default 60)
        """
        super().__init__()
        self.root = root
        self.npy_path = os.path.join(root, npy_name)
        self.seq_len = seq_len
        self.dt = 1.0 / dt_hz

        if not os.path.exists(self.npy_path):
            raise FileNotFoundError(
                f"Could not find quaternion npy file at {self.npy_path}. "
                "Run scripts/make_npy_from_cleaned.py first."
            )

        # ------------------------------------------------------------
        # 1. Load quaternions
        # ------------------------------------------------------------
        quats = np.load(self.npy_path).astype(np.float32)  # (T, 4)
        if quats.ndim != 2 or quats.shape[1] != 4:
            raise ValueError(
                f"Expected shape (T, 4) for quaternions, got {quats.shape}"
            )

        # Vector part as direction proxy: v = (x, y, z)
        v = quats[:, :3]  # (T, 3)

        # Normalize to unit vectors on S^2
        norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
        x_t = (v / norms).astype(np.float32)  # (T, 3)

        # ------------------------------------------------------------
        # 2. Angular velocity approximation via finite differences
        # ------------------------------------------------------------
        omega_t = np.zeros_like(x_t, dtype=np.float32)
        omega_t[1:] = x_t[1:] - x_t[:-1]  # simple finite diff

        T = x_t.shape[0]

        # ------------------------------------------------------------
        # 3. Attention-based direction using Week 1 function
        # ------------------------------------------------------------
        dirs = np.zeros((T, 3), dtype=np.float32)
        for t in range(T):
            dirs[t] = attention_direction(x_t[t], omega_t[t], dt=self.dt)

        # ------------------------------------------------------------
        # 4. Future directions at 250 / 500 / 1000 ms
        # ------------------------------------------------------------
        def future_direction(horizon_ms: float) -> np.ndarray:
            horizon_frames = int((horizon_ms / 1000.0) * (1.0 / self.dt))
            d = np.zeros((T, 3), dtype=np.float32)
            for t in range(T):
                f = min(t + horizon_frames, T - 1)
                d[t] = dirs[f]
            return d

        target_250 = future_direction(250.0)   # (T, 3)
        target_500 = future_direction(500.0)   # (T, 3)
        target_1000 = future_direction(1000.0) # (T, 3)

        # ------------------------------------------------------------
        # 5. Features = [x_t, omega_t] -> shape (T, 6)
        # ------------------------------------------------------------
        features = np.concatenate([x_t, omega_t], axis=1).astype(np.float32)  # (T, 6)

        # ------------------------------------------------------------
        # 6. Build sliding windows
        #    idx i -> window [i : i + seq_len)
        # ------------------------------------------------------------
        if T <= seq_len:
            raise ValueError(
                f"Sequence length T={T} must be > seq_len={seq_len} to form windows."
            )

        self.features = features
        self.target_250 = target_250
        self.target_500 = target_500
        self.target_1000 = target_1000
        self.T = T

        self.num_windows = T - seq_len + 1

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self.num_windows:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_windows}")

        start = idx
        end = idx + self.seq_len  # not inclusive

        f = self.features[start:end]          # (seq_len, 6)
        t250 = self.target_250[start:end]     # (seq_len, 3)
        t500 = self.target_500[start:end]     # (seq_len, 3)
        t1000 = self.target_1000[start:end]   # (seq_len, 3)

        return {
            "features": torch.from_numpy(f),        # (T, F)
            "target_250": torch.from_numpy(t250),   # (T, 3)
            "target_500": torch.from_numpy(t500),   # (T, 3)
            "target_1000": torch.from_numpy(t1000)  # (T, 3)
        }
