# src/models/lstm_predictor.py

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAEPredictor(nn.Module):
    """
    LSTM-based predictor for μ and κ at multiple future horizons.

    Input:  (B, T, feature_dim)
    Output: 3 heads, each (B, T, 4)
        [:3] = μ  (normalized direction vector)
        [ 3] = κ  (positive scalar)
    """

    def __init__(
        self,
        feature_dim: int = 14,   # <-- Track A confirmed feature size
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Three horizon heads
        self.head_250 = nn.Linear(hidden_size, 4)
        self.head_500 = nn.Linear(hidden_size, 4)
        self.head_1000 = nn.Linear(hidden_size, 4)

    def _decode_head(self, h, head):
        """
        Convert LSTM hidden states -> μ (unit vector) + κ (positive scalar)
        """
        raw = head(h)  # (B, T, 4)

        # Normalize first 3 dims
        m_hat = raw[..., :3]
        mu = F.normalize(m_hat, dim=-1, eps=1e-6)

        # Softplus for κ
        kappa_hat = raw[..., 3:4]
        kappa = F.softplus(kappa_hat)

        return torch.cat([mu, kappa], dim=-1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        out_250 = self._decode_head(lstm_out, self.head_250)
        out_500 = self._decode_head(lstm_out, self.head_500)
        out_1000 = self._decode_head(lstm_out, self.head_1000)

        return out_250, out_500, out_1000


if __name__ == "__main__":
    # quick shape sanity check
    B, T = 2, 5
    dummy = torch.randn(B, T, 14)
    model = SAEPredictor(feature_dim=14)

    o1, o2, o3 = model(dummy)
    print(o1.shape, o2.shape, o3.shape)  # (B,T,4)
