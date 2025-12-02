from __future__ import annotations

from typing import Tuple

import numpy as np

from src.sae_core.vmf_utils import compute_A3


def fisher_proxy(N_s: np.ndarray, kappa_s: np.ndarray) -> np.ndarray:
    """
    Fisher proxy for each scale s:

        I_s = N_s * kappa_s * A3(kappa_s)

    where A3 is obtained from compute_A3 in vmf_utils.

    Inputs:
        N_s     : array of effective masses (shape (S,))
        kappa_s : array of concentrations (shape (S,))

    Returns:
        I_s     : array of Fisher proxy values, same shape as N_s
    """
    N = np.asarray(N_s, dtype=float)
    kappa = np.asarray(kappa_s, dtype=float)

    if N.shape != kappa.shape:
        raise ValueError(
            f"N_s and kappa_s must have the same shape, got {N.shape} and {kappa.shape}"
        )

    A3_vals = compute_A3(kappa)         # element-wise A3(kappa_s)
    I_s = N * kappa * A3_vals
    return I_s


def stabilized_softmax(logits: np.ndarray) -> np.ndarray:
    """
    Numerically-stable softmax over 'logits' (here, I_s):

        alpha_s = exp(I_s - max(I_s)) / sum_s exp(I_s - max(I_s))

    This prevents overflow when logits are large.
    """
    x = np.asarray(logits, dtype=float)

    if x.size == 0:
        return x

    max_x = np.max(x)
    shifted = x - max_x
    exp_shifted = np.exp(shifted)
    total = np.sum(exp_shifted)

    if total == 0.0:
        # Degenerate case: fall back to uniform weights
        return np.full_like(x, 1.0 / x.size)

    alpha = exp_shifted / total
    return alpha


def fuse_multiscale_sae(
    N_s: np.ndarray,
    kappa_s: np.ndarray,
    sae_s_t: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Full multiscale fusion for a single time step t.

    Steps:
        1) I_s       = N_s * kappa_s * A3(kappa_s)
        2) alpha_s   = softmax(I_s)   # stabilized softmax
        3) SAE(t)    = sum_s alpha_s * SAE_s(t)

    Inputs:
        N_s     : array of effective masses (shape (S,))
        kappa_s : array of concentrations (shape (S,))
        sae_s_t : array of per-scale SAE_s(t) scores (shape (S,))

    Returns:
        sae_t   : scalar aggregated SAE(t)
        alpha_s : fusion weights (sum to 1.0)
        I_s     : Fisher proxy values
    """
    N = np.asarray(N_s, dtype=float)
    kappa = np.asarray(kappa_s, dtype=float)
    sae_vals = np.asarray(sae_s_t, dtype=float)

    if not (N.shape == kappa.shape == sae_vals.shape):
        raise ValueError(
            f"N_s, kappa_s, sae_s_t shapes must match, got "
            f"{N.shape}, {kappa.shape}, {sae_vals.shape}"
        )

    # 1) Fisher proxy
    I_s = fisher_proxy(N, kappa)

    # 2) Softmax -> fusion weights
    alpha_s = stabilized_softmax(I_s)

    # 3) Weighted sum -> scalar SAE(t)
    sae_t = float(np.sum(alpha_s * sae_vals))

    return sae_t, alpha_s, I_s
