# src/sae_core/streaming_estimator.py

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from src.sae_core.vmf_utils import invert_A3  # Week 2 inversion


@dataclass
class StreamingVMFState:
    """
    Streaming vMF estimator using an Exponential Moving Average (EMA)
    of the sufficient statistics (m_s, N_s).

    At each step we update:
        rho   = 1 - exp(-dt / tau)
        m_s  ← (1 - rho) * m_s + rho * x_t
        N_s  ← (1 - rho) * N_s + rho

    Then we derive:
        r_s     = ||m_s|| / N_s
        mu_s    = m_s / ||m_s||
        kappa_s = invert_A3(r_s)
    """

    tau: float          # timescale τ (seconds)
    dt: float           # sampling interval Δt (seconds)

    def __post_init__(self) -> None:
        if self.tau <= 0.0:
            raise ValueError("tau must be positive.")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")

        # Forgetting factor ρ = 1 - exp(-Δt / τ)
        self.rho: float = 1.0 - math.exp(-self.dt / self.tau)

        # Initialize EMA state: m_s is 3D vector, N_s is scalar
        self.m_s: np.ndarray = np.zeros(3, dtype=float)
        self.N_s: float = 0.0

    # ------------------------------------------------------------------
    # Update step
    # ------------------------------------------------------------------
    def update(self, x_t: np.ndarray) -> None:
        """
        Ingest a new 3D unit vector x_t and update the EMA state.
        """
        x = np.asarray(x_t, dtype=float)
        if x.shape != (3,):
            raise ValueError(f"x_t must have shape (3,), got {x.shape}")

        # Optional: you can normalize x here for extra safety
        # x_norm = np.linalg.norm(x)
        # if x_norm == 0.0:
        #     raise ValueError("Zero-length x_t is invalid.")
        # x = x / x_norm

        rho = self.rho
        one_minus_rho = 1.0 - rho

        # EMA updates (O(1) memory)
        self.m_s = one_minus_rho * self.m_s + rho * x
        self.N_s = one_minus_rho * self.N_s + rho

    # ------------------------------------------------------------------
    # Parameter extraction
    # ------------------------------------------------------------------
    def get_params(self) -> Tuple[np.ndarray, float, float]:
        """
        Return (mu_s, kappa_s, r_s) for the current state.

        mu_s: 3D unit mean direction
        kappa_s: concentration scalar
        r_s: resultant length in [0, 1)
        """
        if self.N_s <= 0.0:
            # No data yet → return defaults
            mu_s = np.array([1.0, 0.0, 0.0], dtype=float)
            kappa_s = 0.0
            r_s = 0.0
            return mu_s, kappa_s, r_s

        # Resultant length r_s = ||m_s|| / N_s
        m_norm = float(np.linalg.norm(self.m_s))
        if m_norm == 0.0:
            mu_s = np.array([1.0, 0.0, 0.0], dtype=float)
            kappa_s = 0.0
            r_s = 0.0
            return mu_s, kappa_s, r_s

        r_s = m_norm / self.N_s

        # Guard r_s to [0, 1)
        r_s = min(max(r_s, 0.0), 1.0 - 1e-12)

        # Mean direction μ_s = m_s / ||m_s||
        mu_s = self.m_s / m_norm

        # Concentration via Week 2 inversion
        kappa_s = float(invert_A3(r_s))

        return mu_s, kappa_s, r_s

