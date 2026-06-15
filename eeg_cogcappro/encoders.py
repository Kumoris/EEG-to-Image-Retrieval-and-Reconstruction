from __future__ import annotations

from typing import Any

import torch.nn as nn

from .atm_s import ATM_S
from .conformer import EEGConformer

ENCODER_REGISTRY: dict[str, type[nn.Module]] = {
    "atm_s": ATM_S,
    "conformer": EEGConformer,
}


def build_eeg_encoder(
    model_type: str,
    channels: int,
    time_steps: int,
    embed_dim: int,
    **kwargs: Any,
) -> nn.Module:
    """Instantiate an EEG encoder by name.

    Extra ``**kwargs`` are forwarded to the model constructor when the model
    type accepts them (e.g. ``d_model``, ``num_layers`` for the Conformer).
    Unrecognised kwargs are silently ignored so that a single config dict can
    be passed without careful key filtering.
    """
    cls = ENCODER_REGISTRY[model_type]
    import inspect

    sig = inspect.signature(cls.__init__)
    valid: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            valid[k] = v
    return cls(channels, time_steps, embed_dim=embed_dim, **valid)
