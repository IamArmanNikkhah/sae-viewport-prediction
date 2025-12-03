# src/training/loss.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sae_core.vmf_utils import compute_log_c3


def compute_log_c3_torch(kappa: torch.Tensor) -> torch.Tensor:
    """
    Torch-friendly wrapper around NumPy-based compute_log_c3.

    We do NOT backprop through compute_log_c3 itself.
    """
    kappa_np = kappa.detach().cpu().numpy()
    log_c3_np = compute_log_c3(kappa_np)
    log_c3_t = torch.from_numpy(np.asarray(log_c3_np)).to(kappa.device)
    return log_c3_t.type_as(kappa)


class NLLLoss(nn.Module):
    """
    Negative log-likelihood loss for vMF on S^2:

        NLL = -log(c3(kappa)) - kappa * (μ_pred^T x_true)

    Total training loss:
        Total = NLL + mae_weight * MAE(μ_pred, x_true)
    """

    def __init__(self, mae_weight: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.mae = nn.L1Loss()
        self.mae_weight = mae_weight
        self.eps = eps

    def forward(self, pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   : (B, T, 4)
                     [:, :, :3] = μ_pred (direction on S^2)
                     [:, :, 3]  = κ_pred (concentration, >= 0)
            x_true : (B, T, 3)
                     ground-truth direction (unit-ish)
        """
        mu_pred = pred[..., :3]  # (B, T, 3)
        kappa_pred = pred[..., 3]  # (B, T)

        # Normalize directions for safety
        mu_pred = F.normalize(mu_pred, dim=-1, eps=self.eps)
        x_true = F.normalize(x_true, dim=-1, eps=self.eps)

        # Dot product μ_pred^T x_true
        dot = (mu_pred * x_true).sum(dim=-1)  # (B, T)

        # Stable log c3(kappa)
        log_c3 = compute_log_c3_torch(kappa_pred)  # (B, T)

        # vMF negative log-likelihood
        nll = -log_c3 - kappa_pred * dot   # (B, T)
        nll = nll.mean()                   # scalar

        # Auxiliary MAE on directions
        mae = self.mae(mu_pred, x_true)

        return nll + self.mae_weight * mae
