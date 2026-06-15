from __future__ import annotations

import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, num_channels: int, time_dim: int, num_heads: int = 8, depth: int = 2, ff_mult: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        if time_dim % num_heads != 0:
            for cand in (10, 8, 5, 4, 2, 1):
                if time_dim % cand == 0:
                    num_heads = cand
                    break
        self.pos_embed = nn.Parameter(torch.zeros(1, num_channels, time_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=time_dim,
            nhead=num_heads,
            dim_feedforward=time_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(time_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.transformer(x + self.pos_embed))


class ShallowNetBackbone(nn.Module):
    def __init__(
        self,
        num_channels: int,
        time_dim: int,
        n_filters_time: int = 40,
        filter_time_length: int = 25,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        drop_prob: float = 0.25,
    ) -> None:
        super().__init__()
        self.temporal_conv = nn.Conv2d(1, n_filters_time, (1, filter_time_length), bias=False)
        self.spatial_conv = nn.Conv2d(n_filters_time, n_filters_time, (num_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters_time)
        self.pool = nn.AvgPool2d((1, pool_time_length), stride=(1, pool_time_stride))
        self.drop = nn.Dropout(drop_prob)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, time_dim)
            out = self.temporal_conv(dummy)
            out = self.spatial_conv(out)
            out = self.bn(out)
            out = out * out
            out = self.pool(out)
            out = torch.log(out.clamp(min=1e-6))
            self.out_dim = int(out.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = x * x
        x = self.pool(x)
        x = torch.log(x.clamp(min=1e-6))
        return self.drop(x).flatten(1)


class ResidualMLPProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 1024, out_dim: int = 768, n_blocks: int = 2, drop: float = 0.3) -> None:
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(drop),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(n_blocks)
            ]
        )
        self.proj_out = nn.Linear(hidden_dim, out_dim)
        self.norm_out = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        for block in self.blocks:
            x = x + block(x)
        return self.norm_out(self.proj_out(x))


class ATM_S(nn.Module):
    def __init__(self, num_channels: int, time_dim: int, embed_dim: int = 768, attn_heads: int = 8, attn_depth: int = 2) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.time_dim = time_dim
        self.embed_dim = embed_dim
        self.channel_attn = ChannelAttention(num_channels, time_dim, num_heads=attn_heads, depth=attn_depth)
        self.backbone = ShallowNetBackbone(num_channels, time_dim)
        self.projector = ResidualMLPProjector(self.backbone.out_dim, hidden_dim=1024, out_dim=embed_dim)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.projector(self.backbone(self.channel_attn(eeg.float())))
