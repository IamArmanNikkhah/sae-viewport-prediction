# src/data/collate.py

import torch

def pad_seq(x, T):
    """Pad a (t, f) tensor to length T along time dimension."""
    t, f = x.shape
    if t == T:
        return x
    pad = torch.zeros((T - t, f), dtype=x.dtype)
    return torch.cat([x, pad], dim=0)

def collate_scanpaths(batch):
    # 1) find max sequence length in batch
    max_T = max(item["features"].shape[0] for item in batch)

    feats, t250, t500, t1000 = [], [], [], []

    # 2) pad all sequences to max_T
    for item in batch:
        feats.append(pad_seq(item["features"], max_T))
        t250.append(pad_seq(item["target_250"], max_T))
        t500.append(pad_seq(item["target_500"], max_T))
        t1000.append(pad_seq(item["target_1000"], max_T))

    # 3) stack
    return {
        "features": torch.stack(feats, dim=0),
        "target_250": torch.stack(t250, dim=0),
        "target_500": torch.stack(t500, dim=0),
        "target_1000": torch.stack(t1000, dim=0),
    }
