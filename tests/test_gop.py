"""Unit tests for Stage 2 GOP, using a synthetic emission (no model download)."""
import pytest
import torch

from app.gop import align_and_score, gop_to_score, score_to_label


def _emission() -> torch.Tensor:
    """T=6, V=4 (blank=0). Frames clearly favor phones 1,1,2,2,3,3."""
    plan = [1, 1, 2, 2, 3, 3]
    logits = torch.full((len(plan), 4), -5.0)
    for t, p in enumerate(plan):
        logits[t, p] = 5.0
    return logits.log_softmax(-1)


ID2PHONE = {0: "<blank>", 1: "a", 2: "b", 3: "c"}


def test_one_span_per_phone_and_high_gop():
    res = align_and_score(_emission(), [1, 2, 3], ID2PHONE, blank_id=0)
    assert [r["phone"] for r in res] == ["a", "b", "c"]
    # each canonical phone dominates its span -> gop near 0
    assert all(r["gop"] > -0.5 for r in res)
    # spans monotonic and non-overlapping
    assert res[0]["end_sec"] <= res[1]["start_sec"] + 1e-6
    assert res[1]["end_sec"] <= res[2]["start_sec"] + 1e-6


def test_lpp_feature_vector_present():
    res = align_and_score(_emission(), [1, 2, 3], ID2PHONE, blank_id=0)
    assert all(len(r["lpp"]) == 4 for r in res)  # one entry per vocab phone


def test_audio_too_short_raises():
    with pytest.raises(ValueError):
        align_and_score(_emission()[:2], [1, 2, 3], ID2PHONE, blank_id=0)


def test_score_label_thresholds():
    assert score_to_label(90) == "good"
    assert score_to_label(60) == "practice"
    assert score_to_label(20) == "wrong"


def test_gop_to_score_monotonic():
    assert gop_to_score(0.0) > gop_to_score(-2.0) > gop_to_score(-5.0)
