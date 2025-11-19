import numpy as np
import pytest

from src.sae_core.vmf_utils import (
    compute_A3,
    invert_A3,
    KAPPA_SMALL,
    KAPPA_LARGE,
    KAPPA_MAX,
)


def test_r_zero_returns_zero_kappa():
    kappa = invert_A3(0.0)
    assert kappa == pytest.approx(0.0, abs=1e-12)


def test_invert_A3_outputs_valid_range():
    # r in [0, 0.999] should yield kappa in [0, KAPPA_MAX]
    r_values = np.linspace(0.0, 0.999, 20)
    kappas = invert_A3(r_values)

    assert kappas.shape == r_values.shape
    assert np.all(kappas >= 0.0)
    assert np.all(kappas <= KAPPA_MAX)


def test_invert_A3_rejects_invalid_r():
    with pytest.raises(ValueError):
        invert_A3(-0.1)
    with pytest.raises(ValueError):
        invert_A3(1.1)


def test_forward_inverse_identity_across_regimes():
    # Choose kappa values spanning small, mid, and large regimes
    kappas = np.array([
        0.0,
        1e-4,
        KAPPA_SMALL * 0.5,     # inside small-κ regime
        KAPPA_SMALL * 2.0,     # mid-κ
        0.1,
        1.0,
        10.0,
        KAPPA_LARGE * 0.8,     # mid-large transition
        KAPPA_LARGE * 1.2,     # large-κ regime
        100.0,
        1000.0,
    ], dtype=float)

    r = compute_A3(kappas)
    kappa_hat = invert_A3(r)

    # For kappa > 0, the recovered value should be close to original
    # (allow a bit more tolerance for very large kappa)
    for k_true, k_est in zip(kappas, kappa_hat):
        if k_true == 0.0:
            assert k_est == pytest.approx(0.0, abs=1e-8)
        else:
            assert k_est == pytest.approx(k_true, rel=5e-3, abs=5e-2)


def test_vector_input_consistency():
    r_values = np.array([0.001, 0.1, 0.5, 0.9])
    kappas_vec = invert_A3(r_values)

    assert kappas_vec.shape == r_values.shape

    # Scalar vs vector consistency
    for i, r in enumerate(r_values):
        k_scalar = invert_A3(float(r))
        assert k_scalar == pytest.approx(kappas_vec[i], rel=0, abs=1e-12)

