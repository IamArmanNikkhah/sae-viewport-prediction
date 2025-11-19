"""
Benchmark the mid-kappa regime: direct vs LUT interpolation
for compute_A3 and compute_log_c3.

Designed for pure-NumPy LUT version (fast on all systems).
"""

import time
import numpy as np
from src.sae_core.vmf_utils import (
    compute_A3,
    compute_log_c3,
    _A3_mid_lut,
    _log_c3_mid_lut,
    KAPPA_SMALL,
    KAPPA_LARGE,
)

# ----------------------------
# Direct mid-kappa reference functions
# ----------------------------

def _A3_direct_mid(k):
    """Stable direct formula: coth(k) - 1/k."""
    k = np.asarray(k, dtype=np.float64)
    return np.cosh(k) / np.sinh(k) - 1.0 / k


def _log_c3_direct_mid(k):
    """
    log c3 = log(k) - log(4π) - log(sinh(k))
    using stable log(sinh) = k + log1p(-exp(-2k)) - log(2)
    """
    k = np.asarray(k, dtype=np.float64)
    s = np.exp(-2.0 * k)
    log_sinh = k + np.log1p(-s) - np.log(2.0)
    return np.log(k) - np.log(4.0 * np.pi) - log_sinh


# ----------------------------
# Warm-up LUT so first call isn't counted
# ----------------------------

def warmup():
    compute_A3(0.5)
    compute_log_c3(0.5)
    _ = _A3_mid_lut(0.5)
    _ = _log_c3_mid_lut(0.5)


# ----------------------------
# Benchmark utility
# ----------------------------

def timeit(fn, x, n_reps=5):
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn(x)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times)


# ----------------------------
# MAIN
# ----------------------------

def main():
    print("\n==== vMF Mid-Kappa Benchmark (Improved) ====\n")

    warmup()

    # 100k mid-kappa values
    N = 100_000
    ks = np.linspace(KAPPA_SMALL, KAPPA_LARGE, N).astype(np.float64)
    print(f"Benchmarking {N:,} mid-kappa values...\n")

    # ---- A3 ----
    t_direct_A3 = timeit(_A3_direct_mid, ks)
    t_lut_A3 = timeit(_A3_mid_lut, ks)

    print("---- A3(kappa) ----")
    print(f"Direct mid-kappa   : {t_direct_A3:0.6f} s")
    print(f"LUT (NumPy-fast)   : {t_lut_A3:0.6f} s")
    print(f"Speedup A3: {t_direct_A3 / t_lut_A3:0.2f}x\n")

    # ---- log c3 ----
    t_direct_logc3 = timeit(_log_c3_direct_mid, ks)
    t_lut_logc3 = timeit(_log_c3_mid_lut, ks)

    print("---- log c3(kappa) ----")
    print(f"Direct mid-kappa   : {t_direct_logc3:0.6f} s")
    print(f"LUT (NumPy-fast)   : {t_lut_logc3:0.6f} s")
    print(f"Speedup log c3: {t_direct_logc3 / t_lut_logc3:0.2f}x\n")

    print("==== Benchmark complete ====\n")


if __name__ == "__main__":
    main()
