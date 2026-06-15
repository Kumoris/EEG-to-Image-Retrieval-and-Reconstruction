import torch

from eeg_cogcappro.losses import scm_loss, symmetric_clip_loss


def test_clip_loss_backward():
    x = torch.randn(6, 32, requires_grad=True)
    y = torch.randn(6, 32)
    loss = symmetric_clip_loss(x, y, torch.tensor(1.0))
    loss.backward()
    assert x.grad is not None


def test_scm_loss_duplicate_labels_backward():
    x = torch.randn(6, 32, requires_grad=True)
    y = torch.randn(6, 32)
    labels = torch.tensor([0, 0, 1, 1, 2, 3])
    loss = scm_loss(x, y, labels, top_k=2)
    loss.backward()
    assert x.grad is not None
