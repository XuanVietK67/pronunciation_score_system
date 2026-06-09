"""Unit tests for the GOPT model (pure torch, no SpeechBrain / data needed)."""
import torch

from stage3.model import GOPT


def _model():
    return GOPT(input_dim=10, vocab_size=12, d_model=16, nhead=2, num_layers=2)


def test_output_shape():
    feats = torch.randn(2, 5, 10)
    pids = torch.randint(0, 12, (2, 5))
    out = _model()(feats, pids)
    assert out.shape == (2, 5)


def test_padding_mask_runs_and_is_finite():
    feats = torch.randn(2, 5, 10)
    pids = torch.randint(0, 12, (2, 5))
    mask = torch.tensor([[False] * 5, [False, False, False, True, True]])
    out = _model()(feats, pids, padding_mask=mask)
    assert out.shape == (2, 5)
    assert torch.isfinite(out).all()
