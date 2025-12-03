# src/evaluation/metrics.py

import torch
import torch.nn.functional as F
import numpy as np

from src.sae_core.vmf_utils import compute_log_c3


def great_circle_mae(mu_pred: torch.Tensor, x_true: torch.Tensor) -> float:
    """
    Great-circle angular error in **degrees**.

        MAE = mean( arccos(mu_pred ⋅ x_true) ) * 180/pi
    """
    # Normalize for safety
    mu_pred = F.normalize(mu_pred, dim=-1)
    x_true = F.normalize(x_true, dim=-1)

    dot = torch.clamp((mu_pred * x_true).sum(dim=-1), -1.0, 1.0)
    angles = torch.arccos(dot)    # radians
    deg = angles.mean().item() * 180.0 / np.pi
    return deg


def nll_vmf(mu_pred: torch.Tensor, kappa_pred: torch.Tensor, x_true: torch.Tensor) -> float:
    """
    NLL in nats, averaged over dataset:

        NLL = -log(c3(kappa)) - kappa (mu_pred ⋅ x_true)
    """
    mu_pred = F.normalize(mu_pred, dim=-1)
    x_true = F.normalize(x_true, dim=-1)

    dot = (mu_pred * x_true).sum(dim=-1)  # (B,T)

    # Compute log c3 using stable Week 2 function
    kappa_np = kappa_pred.detach().cpu().numpy()
    log_c3_np = compute_log_c3(kappa_np)
    log_c3 = torch.tensor(log_c3_np, dtype=kappa_pred.dtype, device=kappa_pred.device)

    nll = -log_c3 - kappa_pred * dot
    return float(nll.mean().item())
