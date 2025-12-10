# src/data/processed_npy_dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from src.sae_core.attention import attention_direction


class ProcessedNPYDataset(Dataset):
    """
    Loads processed quaternion sequences saved from Week 1 and produces:
        features: (T, 6)   → [xyz, omega]
        target_250, target_500, target_1000: each (T, 3)
    """

    def __init__(self, root="data/processed", split="train"):
        super().__init__()

        # For now we only have 1 sequence. Same file used for all splits.
        self.root = root
        self.files = sorted([
            f for f in os.listdir(root)
            if f.endswith(".npy")
        ])

        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npy files in {root}")

        # Simple 70/10/20 split on list of files
        n = len(self.files)
        n_train = int(0.7 * n)
        n_val = int(0.1 * n)

        if split == "train":
            self.files = self.files[:n_train]
        elif split == "val":
            self.files = self.files[n_train : n_train+n_val]
        else:   # test
            self.files = self.files[n_train+n_val :]

        # If asking for a split that is empty (e.g., only 1 file available)
        if len(self.files) == 0:
            self.files = [sorted(os.listdir(root))[0]]   # fallback to 1 file

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.files[idx])
        quat = np.load(path)          # (T, 4)
        T = quat.shape[0]

        # --------------------------------------------------------
        # Convert quaternion → direction vector
        # --------------------------------------------------------
        x = quat[:, 0]
        y = quat[:, 1]
        z = quat[:, 2]
        w = quat[:, 3]

        # Standard quaternion → forward direction (0,0,1)
        # v' = q * (0,0,1) * q^-1
        # resulting direction:
        dirs = np.zeros((T, 3), dtype=np.float32)
        dirs[:, 0] = 2 * (x*z + w*y)
        dirs[:, 1] = 2 * (y*z - w*x)
        dirs[:, 2] = 1 - 2 * (x*x + y*y)

        # --------------------------------------------------------
        # Angular velocity (finite difference on direction)
        # --------------------------------------------------------
        omega = np.zeros_like(dirs)
        omega[1:] = dirs[1:] - dirs[:-1]

        # --------------------------------------------------------
        # Attention direction (per timestep)
        # --------------------------------------------------------
        att = np.zeros_like(dirs)
        for t in range(T):
            att[t] = attention_direction(dirs[t], omega[t], dt=1/60)

        # --------------------------------------------------------
        # Future-target directions
        # --------------------------------------------------------
        def future_dir(horizon_ms):
            frames = int((horizon_ms / 1000) * 60)
            out = np.zeros((T, 3), dtype=np.float32)
            for t in range(T):
                f = min(t + frames, T - 1)
                out[t] = att[f]
            return out

        target_250 = future_dir(250)
        target_500 = future_dir(500)
        target_1000 = future_dir(1000)

        # --------------------------------------------------------
        # Features = [xyz, omega]
        # --------------------------------------------------------
        features = np.concatenate([dirs, omega], axis=1)

        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "target_250": torch.tensor(target_250, dtype=torch.float32),
            "target_500": torch.tensor(target_500, dtype=torch.float32),
            "target_1000": torch.tensor(target_1000, dtype=torch.float32),
        }
