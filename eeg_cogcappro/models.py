from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class LogitScale(nn.Module):
    def __init__(self, init_temperature: float = 0.07, max_log: float = math.log(100.0)) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature), dtype=torch.float32))
        self.max_log = max_log

    def forward(self) -> torch.Tensor:
        return self.logit_scale.clamp(0.0, self.max_log)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, dim: int, kernel: int = 9, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        pad = dilation * (kernel // 2)
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel, padding=pad, dilation=dilation, groups=dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, 1),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class AttentionPool1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        w = torch.softmax(self.score(x_t).squeeze(-1), dim=-1)
        return torch.sum(x_t * w[..., None], dim=1)


class EEGExpertEncoder(nn.Module):
    def __init__(self, channels: int, embed_dim: int = 512, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(channels)
        self.stem = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparableBlock(hidden, 9, 1, dropout),
            DepthwiseSeparableBlock(hidden, 9, 2, dropout),
            DepthwiseSeparableBlock(hidden, 7, 4, dropout),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=max(1, min(8, hidden // 32)),
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.attn = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.pool = AttentionPool1d(hidden)
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        x = eeg.float()
        x = (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True).clamp_min(1e-5)
        x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.attn(x.transpose(1, 2)).transpose(1, 2)
        return F.normalize(self.proj(self.pool(x)), dim=-1, eps=1e-8)


class MultiExpertEEGEncoder(nn.Module):
    def __init__(self, channels: int, embed_dim: int = 512, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.img_expert = EEGExpertEncoder(channels, embed_dim, hidden, dropout)
        self.text_expert = EEGExpertEncoder(channels, embed_dim, hidden, dropout)
        self.depth_expert = EEGExpertEncoder(channels, embed_dim, hidden, dropout)
        self.edge_expert = EEGExpertEncoder(channels, embed_dim, hidden, dropout)

    def forward(self, eeg: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "img": self.img_expert(eeg),
            "text": self.text_expert(eeg),
            "depth": self.depth_expert(eeg),
            "edge": self.edge_expert(eeg),
        }


class FusionEncoder(nn.Module):
    def __init__(self, embed_dim: int = 512, layers: int = 2, heads: int = 8, dropout: float = 0.1, modality_dropout: bool = True) -> None:
        super().__init__()
        self.modality_dropout = modality_dropout
        self.pre = nn.ModuleDict({k: nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU()) for k in ["img", "text", "depth", "edge"]})
        self.modality_embed = nn.Parameter(torch.randn(4, embed_dim) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.out = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim, embed_dim))

    def forward(self, experts: dict[str, torch.Tensor]) -> torch.Tensor:
        keys = ["img", "text", "depth", "edge"]
        toks = torch.stack([self.pre[k](experts[k]) for k in keys], dim=1)
        if self.training and self.modality_dropout:
            drop_idx = torch.randint(0, 4, (1,), device=toks.device).item()
            toks[:, drop_idx] = 0
        toks = toks + self.modality_embed[None, :, :]
        pooled = self.encoder(toks).mean(dim=1)
        return F.normalize(self.out(pooled) + pooled, dim=-1, eps=1e-8)


class _Head(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, in_dim), nn.LayerNorm(in_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(in_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1, eps=1e-8)


class STHAlign(nn.Module):
    def __init__(self, embed_dim: int = 512, dropout: float = 0.1, modality_dropout: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.modality_dropout = modality_dropout
        dim = embed_dim * 4
        layers = []
        for _ in range(4):
            layers += [nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(), nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*layers)
        self.img_head = _Head(dim, embed_dim, dropout)
        self.text_head = _Head(dim, embed_dim, dropout)
        self.depth_head = _Head(dim, embed_dim, dropout)
        self.edge_head = _Head(dim, embed_dim, dropout)
        self.fusion_head = _Head(dim, embed_dim, dropout)

    def forward(self, experts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        xs = [experts[k] for k in ["img", "text", "depth", "edge"]]
        if self.training and self.modality_dropout:
            i = torch.randint(0, 4, (1,), device=xs[0].device).item()
            xs[i] = torch.zeros_like(xs[i])
        h = self.trunk(torch.cat(xs, dim=-1))
        return {
            "img": self.img_head(h),
            "text": self.text_head(h),
            "depth": self.depth_head(h),
            "edge": self.edge_head(h),
            "fusion": self.fusion_head(h),
        }


class CogCapProModel(nn.Module):
    def __init__(self, channels: int, embed_dim: int = 512, hidden: int = 256, dropout: float = 0.1, fusion_layers: int = 2, fusion_heads: int = 8) -> None:
        super().__init__()
        self.channels = channels
        self.embed_dim = embed_dim
        self.hidden = hidden
        self.dropout = dropout
        self.experts = MultiExpertEEGEncoder(channels, embed_dim, hidden, dropout)
        self.fusion = FusionEncoder(embed_dim, fusion_layers, fusion_heads, dropout)
        self.align = STHAlign(embed_dim, dropout)
        self.logit_scale = LogitScale()

    def forward(self, eeg: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        experts = self.experts(eeg)
        fusion = self.fusion(experts)
        aligned = self.align(experts)
        return {"experts": experts, "fusion": fusion, "aligned": aligned, "logit_scale": self.logit_scale()}


def build_model_from_checkpoint(ckpt: dict, device: torch.device | str = "cpu") -> CogCapProModel:
    cfg = ckpt.get("model_config", {})
    model = CogCapProModel(
        channels=int(ckpt["channels"]),
        embed_dim=int(ckpt["embed_dim"]),
        hidden=int(cfg.get("expert_hidden", cfg.get("hidden", 256))),
        dropout=float(cfg.get("dropout", 0.1)),
        fusion_layers=int(cfg.get("fusion_layers", 2)),
        fusion_heads=int(cfg.get("fusion_heads", 8)),
    )
    model.load_state_dict(ckpt["model_state"])
    return model.to(device)
