import torch

from eeg_cogcappro.utils import compute_retrieval_metrics


def test_identity_logits_metrics():
    logits = torch.eye(8)
    metrics = compute_retrieval_metrics(logits)
    assert metrics["top1_acc"] == 1.0
    assert metrics["top5_acc"] == 1.0
