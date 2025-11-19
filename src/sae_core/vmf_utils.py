from __future__ import annotations

import math
from typing import Union
import pathlib
import numpy as np

ArrayLike = Union[float, np.ndarray]

# ---------------------------------------------------------------------------
# Constants and regime thresholds
# ---------------------------------------------------------------------------

LOG_4PI = math.log(4.0 * math.pi)
LOG_2PI = math.log(2.0 * math.pi)

KAPPA_SMALL: float = 1e-3
KAPPA_LARGE: float = 50.0
KAPPA_MAX: float = 1e4

# ---------------------------------------------------------------------------
# LUT Globals
# ---------------------------------------------------------------------------

_LUT_LOADED = False
_lut_kappa_grid = None
_lut_A3_vals = None
_lut_logc3_vals = None


def _load_vmf_lut():
    """Load the LUT only once into memory."""
    global _LUT_LOADED, _lut_kappa_grid, _lut_A3_vals, _lut_logc3_vals
    if _LUT_LOADED:
        return

    lut_path = pathlib.Path(__file__).resolve().parent / "vmf_lut_midk.npz"
    data = np.load(lut_path)

    # Use the actual keys stored inside your LUT file
    _lut_kappa_grid = data["kappa_grid"].astype(np.float64)
    _lut_A3_vals    = data["A3_vals"].astype(np.float64)
    _lut_logc3_vals = data["log_c3_vals"].astype(np.float64)

    _LUT_LOADED = True


# ---------------------------------------------------------------------------
# Pure NumPy interpolation (fastest & most portable)
# ---------------------------------------------------------------------------

def _interp_numpy(x: np.ndarray, grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Vectorized 1D linear interpolation (pure NumPy, very fast)."""
    x = np.asarray(x, dtype=np.float64)
    idx = np.searchsorted(grid, x)
    idx = np.clip(idx, 1, grid.size - 1)

    x0 = grid[idx - 1]
    x1 = grid[idx]
    y0 = values[idx - 1]
    y1 = values[idx]

    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


# ---------------------------------------------------------------------------
# Small-kappa Taylor series
# ---------------------------------------------------------------------------

def _A3_small(kappa: np.ndarray) -> np.ndarray:
    k = np.asarray(kappa, dtype=float)
    k2 = k * k
    return (k / 3.0) * (1.0 - k2 / 15.0 + 2.0 * k2 * k2 / 315.0)


def _log_c3_small(kappa: np.ndarray) -> np.ndarray:
    k = np.asarray(kappa, dtype=float)
    k2 = k * k
    return -LOG_4PI - k2 / 6.0 + (k2 * k2) / 180.0


# ---------------------------------------------------------------------------
# Mid-kappa using LUT interpolation
# ---------------------------------------------------------------------------

def _A3_mid_lut(kappa):
    _load_vmf_lut()
    k = np.asarray(kappa, dtype=np.float64)
    return _interp_numpy(k, _lut_kappa_grid, _lut_A3_vals)


def _log_c3_mid_lut(kappa):
    _load_vmf_lut()
    k = np.asarray(kappa, dtype=np.float64)
    return _interp_numpy(k, _lut_kappa_grid, _lut_logc3_vals)


# ---------------------------------------------------------------------------
# Large-kappa asymptotic formulas
# ---------------------------------------------------------------------------

def _A3_large(kappa: np.ndarray) -> np.ndarray:
    k = np.asarray(kappa, dtype=float)
    return 1.0 - 1.0 / k


def _log_c3_large(kappa: np.ndarray) -> np.ndarray:
    k = np.asarray(kappa, dtype=float)
    return np.log(k) - k - LOG_2PI


# ---------------------------------------------------------------------------
# Derivative A3'(kappa)
# ---------------------------------------------------------------------------

def _A3_prime_small(kappa: np.ndarray) -> np.ndarray:
    k = np.asarray(kappa, dtype=float)
    k2 = k * k
    return (1.0 / 3.0) - (k2 / 15.0) + (10.0 * k2 * k2 / 945.0)


def _A3_prime_mid(kappa: np.ndarray) -> np.ndarray:
    k = np.asarray(kappa, dtype=float)
    s = np.sinh(k)
    return -1.0 / (s * s) + 1.0 / (k * k)


def _A3_prime_large(kappa: np.ndarray) -> np.ndarray:
    k = np.asarray(kappa, dtype=float)
    return 1.0 / (k * k)


def compute_A3_prime(kappa: ArrayLike) -> ArrayLike:
    k = np.asarray(kappa, dtype=float)
    if np.any(k < 0.0):
        raise ValueError("kappa must be non-negative")

    scalar = (k.ndim == 0)
    if scalar:
        k = k[None]

    out = np.empty_like(k)

    mask_small = k < KAPPA_SMALL
    mask_large = k > KAPPA_LARGE
    mask_mid   = ~(mask_small | mask_large)

    if np.any(mask_small):
        out[mask_small] = _A3_prime_small(k[mask_small])
    if np.any(mask_mid):
        out[mask_mid] = _A3_prime_mid(k[mask_mid])
    if np.any(mask_large):
        out[mask_large] = _A3_prime_large(k[mask_large])

    return float(out[0]) if scalar else out


# ---------------------------------------------------------------------------
# Public compute_A3 and compute_log_c3 (with regimes + LUT)
# ---------------------------------------------------------------------------

def compute_A3(kappa: ArrayLike) -> ArrayLike:
    k = np.asarray(kappa, dtype=float)
    if np.any(k < 0.0):
        raise ValueError("kappa must be non-negative")

    scalar = (k.ndim == 0)
    if scalar:
        k = k[None]

    out = np.empty_like(k)

    mask_small = k < KAPPA_SMALL
    mask_large = k > KAPPA_LARGE
    mask_mid   = ~(mask_small | mask_large)

    if np.any(mask_small):
        out[mask_small] = _A3_small(k[mask_small])
    if np.any(mask_mid):
        out[mask_mid] = _A3_mid_lut(k[mask_mid])
    if np.any(mask_large):
        out[mask_large] = _A3_large(k[mask_large])

    return float(out[0]) if scalar else out


def compute_log_c3(kappa: ArrayLike) -> ArrayLike:
    k = np.asarray(kappa, dtype=float)
    if np.any(k < 0.0):
        raise ValueError("kappa must be non-negative")

    scalar = (k.ndim == 0)
    if scalar:
        k = k[None]

    out = np.empty_like(k)

    mask_small = k < KAPPA_SMALL
    mask_large = k > KAPPA_LARGE
    mask_mid   = ~(mask_small | mask_large)

    if np.any(mask_small):
        out[mask_small] = _log_c3_small(k[mask_small])
    if np.any(mask_mid):
        out[mask_mid] = _log_c3_mid_lut(k[mask_mid])
    if np.any(mask_large):
        out[mask_large] = _log_c3_large(k[mask_large])

    return float(out[0]) if scalar else out


# ---------------------------------------------------------------------------
# Invert A3
# ---------------------------------------------------------------------------

def invert_A3(r: ArrayLike,
              r_min: float = 1e-6,
              kappa_max: float = KAPPA_MAX) -> ArrayLike:

    r_arr = np.asarray(r, dtype=float)
    if np.any(r_arr < 0.0) or np.any(r_arr >= 1.0 + 1e-12):
        raise ValueError("r must be in [0,1).")

    r_clamped = np.minimum(r_arr, 1.0 - 1e-12)

    scalar = (r_clamped.ndim == 0)
    if scalar:
        r_clamped = r_clamped[None]

    out = np.zeros_like(r_clamped)

    mask_small = r_clamped <= r_min
    mask_pos   = ~mask_small

    if np.any(mask_small):
        out[mask_small] = 0.0

    if np.any(mask_pos):
        r_pos = r_clamped[mask_pos]

        denom = np.maximum(1e-8, 1.0 - r_pos * r_pos)
        k0 = r_pos * (3.0 - r_pos * r_pos) / denom

        A0  = compute_A3(k0)
        dA0 = compute_A3_prime(k0)
        dA0 = np.maximum(dA0, 1e-12)

        k1 = k0 - (A0 - r_pos) / dA0
        k1 = np.clip(k1, 0.0, kappa_max)

        out[mask_pos] = k1

    return float(out[0]) if scalar else out
