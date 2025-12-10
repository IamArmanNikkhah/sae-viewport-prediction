from pathlib import Path
from typing import List, Tuple

def get_processed_files(root_dir: str) -> List[Path]:
    root = Path(root_dir)
    files = sorted(root.glob("seq_*.npy"))
    if not files:
        raise FileNotFoundError(f"No seq_*.npy files found in {root_dir}")
    return files

def split_files(files: List[Path],
                train_frac=0.7,
                val_frac=0.1,
                test_frac=0.2):

    assert abs(train_frac + val_frac + test_frac - 1) < 1e-6

    n = len(files)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = files[:n_train]
    val = files[n_train:n_train + n_val]
    test = files[n_train + n_val:]

    return train, val, test

def load_split(root: str, split: str):
    files = get_processed_files(root)
    train, val, test = split_files(files)

    if split == "train": return train
    if split == "val": return val
    if split == "test": return test

    raise ValueError(f"Invalid split: {split}")
