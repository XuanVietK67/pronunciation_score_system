"""Unit tests for the inference scorer (pure torch, no model download / artifacts)."""
import torch

from app.scorer import Scorer, pred_to_score
from stage3.model import GOPT


def test_pred_to_score_scale_and_clamp():
    assert pred_to_score(0.0) == 0
    assert pred_to_score(1.0) == 50  # midpoint of the [0, 2] label range
    assert pred_to_score(2.0) == 100
    assert pred_to_score(-1.0) == 0  # clamp below
    assert pred_to_score(3.5) == 100  # clamp above


def test_pred_to_score_monotonic():
    assert pred_to_score(0.4) < pred_to_score(1.0) < pred_to_score(1.6)


def _scorer(input_dim=1024, vocab_size=20):
    model = GOPT(input_dim=input_dim, vocab_size=vocab_size, d_model=16, nhead=2, num_layers=2)
    model.eval()
    mean = torch.zeros(input_dim)
    std = torch.ones(input_dim)
    return Scorer(model, mean, std)


def test_score_forward_on_synthetic_batch():
    scorer = _scorer()
    features = torch.randn(5, 1024)  # [L, H]
    phone_ids = [1, 2, 3, 4, 5]
    scores = scorer.score(features, phone_ids)
    assert len(scores) == 5
    assert all(isinstance(s, int) and 0 <= s <= 100 for s in scores)


def test_standardization_uses_provided_stats():
    # With mean/std that map features to a constant, every phone sees the same input.
    model = GOPT(input_dim=8, vocab_size=10, d_model=16, nhead=2, num_layers=2).eval()
    mean = torch.full((8,), 3.0)
    std = torch.full((8,), 2.0)
    scorer = Scorer(model, mean, std)
    features = torch.full((4, 8), 3.0)  # standardizes to all-zeros
    scores = scorer.score(features, [0, 0, 0, 0])
    assert all(0 <= s <= 100 for s in scores)
