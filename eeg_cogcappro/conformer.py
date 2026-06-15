from __future__ import annotations

import torch
from torch import nn

from .atm_s import ResidualMLPProjector


class EEGConformer(nn.Module):
    """
    Channel-wise EEG-Conformer: treats each EEG channel as a token.

    Architecture:
      1. Per-Channel Conv1d — projects each channel's time series (T=250)
         to a d_model-dimensional token via a shared 1-D convolution.
      2. Channel Transformer — deep pre-norm Transformer (default 6 layers)
         over the 63 channel tokens with learnable positional embedding and
         an optional CLS token.  Captures inter-channel dependencies.
      3. Pooling — CLS token or mean pooling → single vector.
      4. ResidualMLPProjector — same projection head as ATM-S.

    Input:  ``(B, num_channels, time_dim)``   e.g. (B, 63, 250)
    Output: ``(B, embed_dim)``
    """

    def __init__(
        self,
        num_channels: int,
        time_dim: int,
        embed_dim: int = 768,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        use_cls: bool = True,
        proj_hidden: int = 1024,
        proj_blocks: int = 2,
        proj_drop: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.time_dim = time_dim
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.use_cls = use_cls

        self.channel_proj = nn.Sequential(
            nn.Conv1d(1, d_model // 2, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(d_model // 2),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        num_tokens = num_channels + (1 if use_cls else 0)
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.token_drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.projector = ResidualMLPProjector(
            d_model, hidden_dim=proj_hidden, out_dim=embed_dim,
            n_blocks=proj_blocks, drop=proj_drop,
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        B, C, T = eeg.shape

        x = eeg.float().reshape(B * C, 1, T)            # (B*C, 1, T)
        x = self.channel_proj(x)                          # (B*C, d_model, 1)
        x = x.squeeze(-1).reshape(B, C, -1)              # (B, C, d_model)
        x = self.token_drop(x)

        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)                # (B, C+1, d_model)
        x = x + self.pos_embed

        x = self.norm(self.transformer(x))                # (B, C+1, d_model)

        if self.use_cls:
            x = x[:, 0]                                   # CLS token
        else:
            x = x.mean(dim=1)                             # mean pool

        return self.projector(x)                           # (B, embed_dim)
