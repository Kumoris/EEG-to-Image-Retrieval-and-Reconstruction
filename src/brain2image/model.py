from __future__ import annotations

import torch
from torch import nn


class EEGEncoder(nn.Module):
    def __init__(self, channels: int, embedding_dim: int = 512, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.net(eeg.float())


class SqueezeExcite1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        inner = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inner, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(inner, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class DepthwiseTemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, groups=channels),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            SqueezeExcite1d(channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TSConvEEGEncoder(nn.Module):
    """A compact TSConv-style encoder inspired by CogCapPro's EEG expert branch."""

    def __init__(self, channels: int, embedding_dim: int = 512, hidden_dim: int = 192, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_norm = nn.BatchNorm1d(channels)
        self.stem = nn.Sequential(
            nn.Conv1d(channels, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(
            DepthwiseTemporalBlock(hidden_dim, kernel_size=9, dilation=1, dropout=dropout),
            DepthwiseTemporalBlock(hidden_dim, kernel_size=9, dilation=2, dropout=dropout),
            DepthwiseTemporalBlock(hidden_dim, kernel_size=7, dilation=4, dropout=dropout),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        x = eeg.float()
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / x.std(dim=-1, keepdim=True).clamp_min(1e-5)
        x = self.input_norm(x)
        x = self.stem(x)
        x = self.temporal(x)
        pooled = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1)
        return self.proj(pooled)


def build_eeg_encoder(
    model_kind: str,
    *,
    channels: int,
    embedding_dim: int,
    hidden_dim: int,
    dropout: float = 0.1,
) -> nn.Module:
    if model_kind == "conv":
        return EEGEncoder(channels=channels, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    if model_kind == "tsconv":
        return TSConvEEGEncoder(
            channels=channels,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported model kind '{model_kind}'.")
