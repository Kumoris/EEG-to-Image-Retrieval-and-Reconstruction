from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import Dataset

from .atm_s import ATM_S
from .data import EEGRecords
from .features import features_for_ids
from .transforms_eeg import EEGTrainTransform


class MultiScaleBlurDataset(Dataset):
    def __init__(
        self,
        records: EEGRecords,
        cache: dict,
        feature_keys: list[str],
        augment: bool = False,
        noise_std: float = 0.01,
        channel_dropout_p: float = 0.1,
        time_mask_frac: float = 0.1,
    ) -> None:
        self.records = records
        self.feature_keys = feature_keys
        self.transform = (
            EEGTrainTransform(
                noise_std=noise_std,
                channel_dropout_p=channel_dropout_p,
                temporal_jitter=0,
                time_mask_frac=time_mask_frac,
            )
            if augment
            else None
        )
        scale_feats = [
            features_for_ids(cache, records.image_ids, k) for k in feature_keys
        ]
        self.scale_features = torch.cat(scale_feats, dim=1)

    def __len__(self) -> int:
        return len(self.records.eeg)

    def __getitem__(self, idx: int):
        eeg = self.records.eeg[idx]
        if self.transform is not None:
            eeg = self.transform(eeg)
        return eeg, self.scale_features[idx], self.records.labels[idx]


class ScaleLinearFusion(nn.Module):
    def __init__(self, n_scales: int, feature_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_scales = n_scales
        self.feature_dim = feature_dim
        in_dim = n_scales * feature_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, scale_features: torch.Tensor) -> torch.Tensor:
        return self.proj(scale_features)


class ScaleAttentionFusion(nn.Module):
    def __init__(self, n_scales: int, feature_dim: int, out_dim: int, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_scales = n_scales
        self.feature_dim = feature_dim
        self.scale_embed = nn.Parameter(torch.zeros(1, n_scales, feature_dim))
        nn.init.trunc_normal_(self.scale_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=num_heads,
            dim_feedforward=feature_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.attn = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Sequential(
            nn.Linear(n_scales * feature_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, scale_features: torch.Tensor) -> torch.Tensor:
        x = scale_features.view(-1, self.n_scales, self.feature_dim)
        x = x + self.scale_embed
        x = self.attn(x)
        x = x.reshape(x.shape[0], -1)
        return self.proj(x)


class MultiscaleBlurModel(nn.Module):
    def __init__(
        self,
        num_channels: int,
        time_dim: int,
        n_scales: int,
        feature_dim: int,
        embed_dim: int = 768,
        eeg_attn_heads: int = 8,
        eeg_attn_depth: int = 2,
        visual_encoder_type: str = "linear",
        visual_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.eeg_encoder = ATM_S(
            num_channels=num_channels,
            time_dim=time_dim,
            embed_dim=embed_dim,
            attn_heads=eeg_attn_heads,
            attn_depth=eeg_attn_depth,
        )
        if visual_encoder_type == "attention":
            self.visual_encoder = ScaleAttentionFusion(
                n_scales=n_scales,
                feature_dim=feature_dim,
                out_dim=embed_dim,
                num_heads=min(feature_dim // 64, 8),
                dropout=visual_dropout,
            )
        else:
            self.visual_encoder = ScaleLinearFusion(
                n_scales=n_scales,
                feature_dim=feature_dim,
                out_dim=embed_dim,
                dropout=visual_dropout,
            )

    def forward(self, eeg: torch.Tensor, scale_features: torch.Tensor):
        eeg_emb = self.eeg_encoder(eeg)
        vis_emb = self.visual_encoder(scale_features)
        return eeg_emb, vis_emb