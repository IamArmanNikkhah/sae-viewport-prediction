from __future__ import annotations

import math
from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]

# ---------------------------------------------------------------------------
# Constants and regime thresholds
# ---------------------------------------------------------------------------

LOG_4PI = math.log(4.0 * math.pi)
LOG_2PI = math.log(2.0 * math.pi)


# Regime breakpoints (can be tuned if the paper specifies different ones)
KAPPA_SMALL: float = 1e-3   # “small κ” threshold
KAPPA_LARGE: float = 50.0   # “large κ” threshold
KAPPA_MAX: float = 1e4  # upper clamp for invert_A3


# ---------------------------------------------------------------------------
# Small-κ regime: Taylor expansions
# ---------------------------------------------------------------------------

def _A3_small(kappa: np.ndarray) -> np.ndarray:
    """
    Taylor series for A3(kappa) around kappa = 0.

    For p=3, we have:
        A3(k) = coth(k) - 1/k
              = k/3 - k^3/45 + 2 k^5/945 + O(k^7)

    We use:
        A3(k) ≈ (k/3) * (1 - k^2/15 + 2 k^4/315)
    """
    k = np.asarray(kappa, dtype=float)
    k2 = k * k
    return (k / 3.0) * (1.0 - k2 / 15.0 + 2.0 * k2 * k2 / 315.0)


def _log_c3_small(kappa: np.ndarray) -> np.ndarray:
    """
    Taylor series for log c3(kappa) around kappa = 0.

    For p=3, c3(kappa) = kappa / (4π sinh(kappa)).
    For small kappa, one can show:

        log c3(k) ≈ -log(4π) - k^2/6 + k^4/180 + O(k^6)

    This avoids the log(0) issues of the direct formula.
    """
    k = np.asarray(kappa, dtype=float)
    k2 = k * k
    return -LOG_4PI - k2 / 6.0 + (k2 * k2) / 180.0


# ---------------------------------------------------------------------------
# Mid-κ regime: direct formulas (with stable log-sinh)
# ---------------------------------------------------------------------------

def _A3_mid(kappa: np.ndarray) -> np.ndarray:
    """
    Direct formula for A3(kappa) in the mid-κ regime:

        A3(kappa) = coth(kappa) - 1/kappa

    In this regime, cosh/sinh are numerically safe.
    """
    k = np.asarray(kappa, dtype=float)
    coth_k = np.cosh(k) / np.sinh(k)
    return coth_k - 1.0 / k


def _log_sinh_stable(kappa: np.ndarray) -> np.ndarray:
    """
    Numerically stable log(sinh(kappa)) using a log-sum-exp style rewrite:

        sinh(k)  = 0.5 * (e^k - e^{-k})
                 = 0.5 * e^k * (1 - e^{-2k})

        log sinh(k) = k + log(1 - e^{-2k}) - log 2

    This never forms e^{+k} directly in a way that overflows for large k.
    """
    k = np.asarray(kappa, dtype=float)
    return k + np.log1p(-np.exp(-2.0 * k)) - math.log(2.0)


def _log_c3_mid(kappa: np.ndarray) -> np.ndarray:
    """
    Direct formula for log c3(kappa) in the mid-κ regime, using the
    stable log-sinh helper:

        log c3(kappa) = log(kappa) - log(4π) - log(sinh(kappa))
    """
    k = np.asarray(kappa, dtype=float)
    log_sinh = _log_sinh_stable(k)
    return np.log(k) - LOG_4PI - log_sinh


# ---------------------------------------------------------------------------
# Large-κ regime: asymptotic (Padé-like) / log-sum-exp-style
# ---------------------------------------------------------------------------

def _A3_large(kappa: np.ndarray) -> np.ndarray:
    """
    Asymptotic / Padé-like approximation for A3(kappa) at large κ.

    For p=3, A3(kappa) = coth(kappa) - 1/kappa and for large kappa,

        coth(kappa) ≈ 1  (up to exponentially small terms),
        => A3(kappa) ≈ 1 - 1/kappa.

    This can be viewed as a very simple rational (Padé) form:
        A3(kappa) ≈ (kappa - 1) / kappa
    which is extremely accurate once kappa is large.
    """
    k = np.asarray(kappa, dtype=float)
    return 1.0 - 1.0 / k


def _log_c3_large(kappa: np.ndarray) -> np.ndarray:
    """
    Asymptotic formula for log c3(kappa) at large κ.

    For p=3:

        c3(k) = k / (4π sinh(k))

    and for large k:

        sinh(k) ≈ 0.5 e^k
        => c3(k) ≈ k / (4π * 0.5 e^k) = k e^{-k} / (2π)

        => log c3(k) ≈ log(k) - k - log(2π)

    This avoids overflow and underflow while being extremely accurate
    in the large-κ regime.
    """
    k = np.asarray(kappa, dtype=float)
    return np.log(k) - k - LOG_2PI

# ---------------------------------------------------------------------------
# Derivative A3'(kappa) with three regimes
# ---------------------------------------------------------------------------

def _A3_prime_small(kappa: np.ndarray) -> np.ndarray:
    """
    Taylor series for A3'(kappa) around kappa = 0.

    From A3(k) = k/3 - k^3/45 + 2k^5/945 + ... we get:

        A3'(k) = 1/3 - k^2/15 + 10 k^4/945 + O(k^6)
    """
    k = np.asarray(kappa, dtype=float)
    k2 = k * k
    return (1.0 / 3.0) - (k2 / 15.0) + (10.0 * k2 * k2 / 945.0)


def _A3_prime_mid(kappa: np.ndarray) -> np.ndarray:
    """
    Direct derivative in the mid-κ regime.

    A3(k)   = coth(k) - 1/k
    A3'(k)  = -csch^2(k) + 1/k^2
            = -1/sinh(k)^2 + 1/k^2
    """
    k = np.asarray(kappa, dtype=float)
    s = np.sinh(k)
    return -1.0 / (s * s) + 1.0 / (k * k)


def _A3_prime_large(kappa: np.ndarray) -> np.ndarray:
    """
    Large-κ asymptotic for A3'(kappa).

    For large k, A3(k) ≈ 1 - 1/k, so:

        A3'(k) ≈ 1/k^2
    """
    k = np.asarray(kappa, dtype=float)
    return 1.0 / (k * k)


def compute_A3_prime(kappa: ArrayLike) -> ArrayLike:
    """
    Numerically stable derivative A3'(kappa) for 3D vMF.

    Uses the same three-regime system as compute_A3:
        - small kappa: Taylor series
        - mid   kappa: exact derivative with sinh
        - large kappa: asymptotic 1/k^2
    """
    k = np.asarray(kappa, dtype=float)

    if np.any(k < 0.0):
        raise ValueError("kappa must be non-negative")

    scalar_input = (k.ndim == 0)
    if scalar_input:
        k = k[None]

    result = np.empty_like(k, dtype=float)

    mask_small = k < KAPPA_SMALL
    mask_large = k > KAPPA_LARGE
    mask_mid = ~(mask_small | mask_large)

    if np.any(mask_small):
        result[mask_small] = _A3_prime_small(k[mask_small])
    if np.any(mask_mid):
        result[mask_mid] = _A3_prime_mid(k[mask_mid])
    if np.any(mask_large):
        result[mask_large] = _A3_prime_large(k[mask_large])

    if scalar_input:
        return float(result[0])
    return result


# ---------------------------------------------------------------------------
# Public wrappers: compute_A3, compute_log_c3
# ---------------------------------------------------------------------------

def compute_A3(kappa: ArrayLike) -> ArrayLike:
    """
    Numerically stable computation of A3(kappa) for 3D vMF.

    Uses a three-regime system:
        - small kappa: 0 <= κ <  KAPPA_SMALL   -> Taylor series
        - mid   kappa: KAPPA_SMALL <= κ <= KAPPA_LARGE -> direct formula
        - large kappa: κ > KAPPA_LARGE         -> asymptotic / Padé-like

    Accepts scalar float or numpy array; returns same shape.
    """
    k = np.asarray(kappa, dtype=float)

    if np.any(k < 0.0):
        raise ValueError("kappa must be non-negative")

    scalar_input = (k.ndim == 0)
    if scalar_input:
        k = k[None]  # make it 1D for mask assignment

    result = np.empty_like(k, dtype=float)

    mask_small = k < KAPPA_SMALL
    mask_large = k > KAPPA_LARGE
    mask_mid = ~(mask_small | mask_large)

    if np.any(mask_small):
        result[mask_small] = _A3_small(k[mask_small])
    if np.any(mask_mid):
        result[mask_mid] = _A3_mid(k[mask_mid])
    if np.any(mask_large):
        result[mask_large] = _A3_large(k[mask_large])

    if scalar_input:
        return float(result[0])
    return result


def compute_log_c3(kappa: ArrayLike) -> ArrayLike:
    """
    Numerically stable computation of log c3(kappa) for 3D vMF.

    Uses a three-regime system:
        - small kappa: 0 <= κ <  KAPPA_SMALL   -> Taylor series
        - mid   kappa: KAPPA_SMALL <= κ <= KAPPA_LARGE -> direct with log-sinh
        - large kappa: κ > KAPPA_LARGE         -> asymptotic

    Accepts scalar float or numpy array; returns same shape.
    """
    k = np.asarray(kappa, dtype=float)

    if np.any(k < 0.0):
        raise ValueError("kappa must be non-negative")

    scalar_input = (k.ndim == 0)
    if scalar_input:
        k = k[None]

    result = np.empty_like(k, dtype=float)

    mask_small = k < KAPPA_SMALL
    mask_large = k > KAPPA_LARGE
    mask_mid = ~(mask_small | mask_large)

    if np.any(mask_small):
        result[mask_small] = _log_c3_small(k[mask_small])
    if np.any(mask_mid):
        result[mask_mid] = _log_c3_mid(k[mask_mid])
    if np.any(mask_large):
        result[mask_large] = _log_c3_large(k[mask_large])

    if scalar_input:
        return float(result[0])
    return result

# ---------------------------------------------------------------------------
# Inverse of A3: given r in [0, 1), solve A3(kappa) = r
# ---------------------------------------------------------------------------

def invert_A3(r: ArrayLike,
              r_min: float = 1e-6,
              kappa_max: float = KAPPA_MAX) -> ArrayLike:
    """
    Invert A3(kappa) = r to obtain kappa.

    Implements a single safeguarded Newton step using:

        kappa_0  = Sra's 3D analytic initializer
                 ≈ r * (3 - r^2) / (1 - r^2)
        kappa_1  = kappa_0 - (A3(kappa_0) - r) / A3'(kappa_0)

    Safeguards:
        - r <= r_min -> return 0
        - clamp kappa_1 into [0, kappa_max]
        - handle pathological derivative values safely

    Accepts scalar or numpy arrays, returns same shape.
    """
    r_arr = np.asarray(r, dtype=float)

    if np.any(r_arr < 0.0) or np.any(r_arr >= 1.0 + 1e-12):
        raise ValueError("r must be in the range [0, 1).")

    # Clamp extremely close to 1 to avoid division by zero in initializer
    r_clamped = np.minimum(r_arr, 1.0 - 1e-12)

    scalar_input = (r_clamped.ndim == 0)
    if scalar_input:
        r_clamped = r_clamped[None]

    kappa_out = np.zeros_like(r_clamped, dtype=float)

    # Very small r -> kappa ≈ 0
    mask_small_r = r_clamped <= r_min
    mask_pos_r = ~mask_small_r

    if np.any(mask_small_r):
        kappa_out[mask_small_r] = 0.0

    if np.any(mask_pos_r):
        r_pos = r_clamped[mask_pos_r]

        # ---- Analytic initializer (Sra's 3D approximation) ----
        # kappa0 ≈ r * (3 - r^2) / (1 - r^2)
        denom = np.maximum(1e-8, 1.0 - r_pos * r_pos)
        kappa0 = r_pos * (3.0 - r_pos * r_pos) / denom

        # ---- One safeguarded Newton step ----
        A0 = compute_A3(kappa0)
        dA0 = compute_A3_prime(kappa0)

        # Avoid division by zero / negative derivatives (should not happen in theory)
        dA0_safe = np.maximum(dA0, 1e-12)
        kappa1 = kappa0 - (A0 - r_pos) / dA0_safe

        # Clamp to valid range
        kappa1 = np.clip(kappa1, 0.0, kappa_max)

        kappa_out[mask_pos_r] = kappa1

    if scalar_input:
        return float(kappa_out[0])
    return kappa_out


