# scripts/generate_vmf_lut.py
#!/usr/bin/env python
"""
Offline script to generate a Look-Up Table (LUT) for A3(kappa) and log c3(kappa)
for the mid-kappa regime used in vmf_utils.py.

It writes: src/sae_core/vmf_lut_midk.npz
"""

import numpy as np
import mpmath as mp
from pathlib import Path

# These MUST match vmf_utils.kappa_sm / kappa_lg
KAPPA_SM = 1e-2
KAPPA_LG = 20.0
N_KNOTS = 4096  # number of grid points


def A3_mp(kappa: mp.mpf) -> mp.mpf:
    """High-precision A3(kappa) = coth(kappa) - 1/kappa."""
    return mp.coth(kappa) - 1 / kappa


def log_c3_mp(kappa: mp.mpf) -> mp.mpf:
    """
    High-precision log c3(kappa) for 3D vMF:
      c3(kappa) = kappa / (4*pi*sinh(kappa))
      log c3 = log(kappa) - log(4*pi) - log(sinh(kappa))
    """
    return mp.log(kappa) - mp.log(4 * mp.pi) - mp.log(mp.sinh(kappa))


def main():
    mp.mp.dps = 80  # 80 decimal digits of precision

    # Log-spaced kappa grid in [KAPPA_SM, KAPPA_LG]
    kappa_grid = np.logspace(
        np.log10(KAPPA_SM), np.log10(KAPPA_LG), N_KNOTS, dtype=np.float64
    )

    A3_vals = np.empty_like(kappa_grid)
    log_c3_vals = np.empty_like(kappa_grid)

    for i, k in enumerate(kappa_grid):
        k_mp = mp.mpf(k)
        A3_vals[i] = float(A3_mp(k_mp))
        log_c3_vals[i] = float(log_c3_mp(k_mp))

    # Save into src/sae_core next to vmf_utils.py
    project_root = Path(__file__).resolve().parents[1]  # .. (repo root)
    out_path = project_root / "src" / "sae_core" / "vmf_lut_midk.npz"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        kappa_grid=kappa_grid,
        A3_vals=A3_vals,
        log_c3_vals=log_c3_vals,
    )
    print(f"Saved LUT to {out_path}")
    print(f"kappa range: [{kappa_grid[0]}, {kappa_grid[-1]}], N={len(kappa_grid)}")


if __name__ == "__main__":
    main()
