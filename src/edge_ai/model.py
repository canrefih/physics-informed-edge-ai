import torch
from torch import nn


class DriftRegressor(nn.Module):
    """Small baseline model for future drift estimation."""

    def __init__(self, window_size=32, channels=2, hidden_dim=64):
        super().__init__()

        self.window_size = window_size
        self.channels = channels

        self.network = nn.Sequential(
            nn.Linear(window_size * channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        # Input: [batch, time, channels]
        x = x.flatten(start_dim=1)
        return self.network(x).squeeze(-1)
