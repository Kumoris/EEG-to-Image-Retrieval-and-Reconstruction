from __future__ import annotations

import random

import torch


class EEGTrainTransform:
    def __init__(
        self,
        noise_std: float = 0.01,
        channel_dropout_p: float = 0.1,
        temporal_jitter: int = 0,
        time_mask_frac: float = 0.1,
    ) -> None:
        self.noise_std = noise_std
        self.channel_dropout_p = channel_dropout_p
        self.temporal_jitter = temporal_jitter
        self.time_mask_frac = time_mask_frac

    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        x = eeg.clone()
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        if self.channel_dropout_p > 0:
            mask = torch.rand(x.shape[0], device=x.device) < self.channel_dropout_p
            x[mask] = 0
        if self.temporal_jitter > 0 and x.shape[-1] > 2 * self.temporal_jitter:
            shift = random.randint(-self.temporal_jitter, self.temporal_jitter)
            if shift:
                x = torch.roll(x, shifts=shift, dims=-1)
        if self.time_mask_frac > 0:
            width = int(x.shape[-1] * self.time_mask_frac)
            if width > 0:
                start = random.randint(0, max(0, x.shape[-1] - width))
                x[:, start : start + width] = 0
        return x
