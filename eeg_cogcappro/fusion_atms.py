from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from .atm_s import ATM_S

MODALITY_ORDER = ["image", "text", "depth", "edge"]


class ATMFusionEncoder(nn.Module):
    def __init__(
        self,
        num_channels: int,
        time_dim: int,
        embed_dim: int = 768,
        depth: int = 2,
        heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        modality_dropout_p: float = 0.2,
        freeze_experts: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.modality_dropout_p = modality_dropout_p
        self.experts = nn.ModuleDict(
            OrderedDict(
                (name, ATM_S(num_channels=num_channels, time_dim=time_dim, embed_dim=embed_dim))
                for name in MODALITY_ORDER
            )
        )
        if freeze_experts:
            for p in self.experts.parameters():
                p.requires_grad = False
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.modality_pos = nn.Parameter(torch.zeros(1, len(MODALITY_ORDER) + 1, embed_dim))
        nn.init.trunc_normal_(self.modality_pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def load_expert_checkpoints(self, ckpt_paths: dict[str, str]) -> None:
        for name, path in ckpt_paths.items():
            blob = torch.load(path, map_location="cpu", weights_only=False)
            sd = blob.get("model", blob.get("state_dict", blob))
            self.experts[name].load_state_dict(sd, strict=True)

    def encode_experts(self, eeg: torch.Tensor) -> torch.Tensor:
        outs = [self.experts[name](eeg) for name in MODALITY_ORDER]
        return torch.stack(outs, dim=1)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        feats = self.encode_experts(eeg)
        b, m, _ = feats.shape
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, feats], dim=1) + self.modality_pos
        mask = None
        if self.training and self.modality_dropout_p > 0:
            drop = torch.rand(b, m, device=eeg.device) < self.modality_dropout_p
            drop[drop.sum(dim=1) == m, 0] = False
            mask = torch.zeros(b, m + 1, dtype=torch.bool, device=eeg.device)
            mask[:, 1:] = drop
        x = self.fusion(tokens, src_key_padding_mask=mask)
        return self.norm(x)[:, 0]
