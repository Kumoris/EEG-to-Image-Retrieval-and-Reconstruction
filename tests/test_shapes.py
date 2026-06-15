import torch

from eeg_cogcappro.models import CogCapProModel


def test_model_shapes_no_nan():
    model = CogCapProModel(channels=8, embed_dim=512, hidden=64, fusion_heads=8)
    out = model(torch.randn(4, 8, 64))
    for value in out["experts"].values():
        assert value.shape == (4, 512)
        assert not torch.isnan(value).any()
    assert out["fusion"].shape == (4, 512)
    assert not torch.isnan(out["fusion"]).any()
    for value in out["aligned"].values():
        assert value.shape == (4, 512)
        assert not torch.isnan(value).any()
