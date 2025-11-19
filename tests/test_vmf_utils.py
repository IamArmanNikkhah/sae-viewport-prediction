import numpy as np

from src.sae_core.vmf_utils import (
    compute_A3,
    compute_log_c3,
    KAPPA_SMALL,
    KAPPA_LARGE,
)


def _direct_A3(kappa: np.ndarray) -> np.ndarray:
    """Direct A3(kappa) using the textbook formula (for comparison only)."""
    k = np.asarray(kappa, dtype=float)
    return np.cosh(k) / np.sinh(k) - 1.0 / k


def _direct_log_c3(kappa: np.ndarray) -> np.ndarray:
    """Direct log c3(kappa) using the textbook formula (for comparison only)."""
    k = np.asarray(kappa, dtype=float)
    LOG_4PI = np.log(4.0 * np.pi)
    return np.log(k) - LOG_4PI - np.log(np.sinh(k))


def test_no_nan_or_inf_over_wide_kappa_range():
    # κ from 1e-8 to 1e4, log-spaced
    kappas = np.logspace(-8, 4, 100)
    A_vals = compute_A3(kappas)
    logc_vals = compute_log_c3(kappas)

    assert np.all(np.isfinite(A_vals))
    assert np.all(np.isfinite(logc_vals))
    # A3 should be in [0, 1) for κ >= 0
    assert np.all(A_vals >= 0.0)
    assert np.all(A_vals < 1.0)


def test_small_kappa_matches_direct_just_below_threshold():
    # Pick a value just below the small-κ threshold
    kappa = KAPPA_SMALL * 0.9  # inside small-κ regime
    approx = compute_A3(kappa)
    direct = _direct_A3(kappa)

    # They should match very closely
    assert np.isclose(approx, direct, rtol=1e-8, atol=1e-10)

    approx_log = compute_log_c3(kappa)
    direct_log = _direct_log_c3(kappa)
    assert np.isclose(approx_log, direct_log, rtol=1e-8, atol=1e-10)


def test_large_kappa_matches_direct_just_above_threshold():
    # Pick a value just above the large-κ threshold
    kappa = KAPPA_LARGE * 1.1  # inside large-κ regime
    approx = compute_A3(kappa)
    direct = _direct_A3(kappa)

    assert np.isclose(approx, direct, rtol=1e-8, atol=1e-10)

    approx_log = compute_log_c3(kappa)
    direct_log = _direct_log_c3(kappa)
    assert np.isclose(approx_log, direct_log, rtol=1e-8, atol=1e-10)


def test_boundary_continuity_at_small_and_large_thresholds():
    # At KAPPA_SMALL
    k_small = KAPPA_SMALL
    A_small = compute_A3(k_small)
    logc_small = compute_log_c3(k_small)
    direct_small_A = _direct_A3(k_small)
    direct_small_logc = _direct_log_c3(k_small)

    assert np.isclose(A_small, direct_small_A, rtol=1e-8, atol=1e-10)
    assert np.isclose(logc_small, direct_small_logc, rtol=1e-8, atol=1e-10)

    # At KAPPA_LARGE
    k_large = KAPPA_LARGE
    A_large = compute_A3(k_large)
    logc_large = compute_log_c3(k_large)
    direct_large_A = _direct_A3(k_large)
    direct_large_logc = _direct_log_c3(k_large)

    assert np.isclose(A_large, direct_large_A, rtol=1e-8, atol=1e-10)
    assert np.isclose(logc_large, direct_large_logc, rtol=1e-8, atol=1e-10)


def test_scalar_and_vector_inputs_behave_consistently():
    kappas = np.array([0.0, 1e-4, 0.1, 1.0, 10.0])

    A_vec = compute_A3(kappas)
    logc_vec = compute_log_c3(kappas)

    assert A_vec.shape == kappas.shape
    assert logc_vec.shape == kappas.shape

    # Compare scalar calls to the vectorized results
    for i, k in enumerate(kappas):
        assert np.isclose(compute_A3(float(k)), A_vec[i], rtol=0, atol=1e-15)
        assert np.isclose(compute_log_c3(float(k)), logc_vec[i], rtol=0, atol=1e-15)


        
