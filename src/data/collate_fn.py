import torch

def pad_collate(batch):
    """
    Pads variable-length sequences in the batch.
    Produces:
      features: (B, T_max, F)
      target_250, target_500, target_1000: (B, T_max, 3)
      lengths: (B,)
    """
    # Extract sequences
    feats = [item["features"] for item in batch]
    t250 = [item["target_250"] for item in batch]
    t500 = [item["target_500"] for item in batch]
    t1000 = [item["target_1000"] for item in batch]

    lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    T_max = lengths.max().item()

    # Prepare padded tensors
    F = feats[0].shape[-1]
    B = len(batch)

    feats_pad = torch.zeros((B, T_max, F), dtype=torch.float32)
    t250_pad = torch.zeros((B, T_max, 3), dtype=torch.float32)
    t500_pad = torch.zeros((B, T_max, 3), dtype=torch.float32)
    t1000_pad = torch.zeros((B, T_max, 3), dtype=torch.float32)

    # Copy variable-length sequences into padded tensors
    for i in range(B):
        L = lengths[i]
        feats_pad[i, :L] = feats[i]
        t250_pad[i, :L] = t250[i]
        t500_pad[i, :L] = t500[i]
        t1000_pad[i, :L] = t1000[i]

    return {
        "features": feats_pad,
        "target_250": t250_pad,
        "target_500": t500_pad,
        "target_1000": t1000_pad,
        "lengths": lengths,
    }
