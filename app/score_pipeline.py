"""Shared audio -> per-phone score pipeline.

One implementation used by both POST /score and scripts/score_file.py so the HTTP path
and the CLI never drift. Feature construction mirrors stage3/extract_features.py exactly:
forward_features -> forced-align on the CTC log-probs -> pool the wav2vec2 hidden states
over each phone's span -> trained GOPT head (or the placeholder fallback).
"""
from __future__ import annotations

import torch

from . import config
from .acoustic import Acoustic
from .g2p import g2p, to_token_ids
from .gop import align_spans, gop_to_score, score_to_label
from .scorer import Scorer


def score_word(
    wav: torch.Tensor,
    word: str,
    acoustic: Acoustic,
    exact: dict[str, int],
    norm: dict[str, int],
    scorer: Scorer | None,
) -> tuple[list[str], list[dict], int]:
    """Returns (canonical phones, per-phone dicts, overall score).

    Raises ValueError on unmappable phones or audio too short — callers map it to HTTP 422.
    """
    phones = g2p(word)
    token_ids, unmapped = to_token_ids(phones, exact, norm)
    if not token_ids:
        raise ValueError(f"no phones produced for '{word}'")
    if unmapped:
        raise ValueError(f"unmapped phones for '{word}': {unmapped}")

    log_probs, hidden = acoustic.forward_features(wav)
    if len(token_ids) > log_probs.shape[0]:
        raise ValueError("audio too short for the number of target phones")

    # One span per phone; merge_tokens may collapse adjacent identical tokens, so we
    # derive everything (count, ids, labels) from the spans actually returned.
    spans = align_spans(log_probs, token_ids, acoustic.blank_id)

    if scorer is not None:
        # Pool wav2vec2 hidden states over each span -> [L, H], identical to training.
        features = torch.stack(
            [hidden[start:end].mean(0) for _, start, end, _ in spans]
        ).to(torch.float32)
        aligned_ids = [tok for tok, _, _, _ in spans]
        scores = scorer.score(features, aligned_ids)
    else:
        # Fallback: the placeholder GOP->score mapping (USE_TRAINED_HEAD=false).
        scores = [gop_to_score(float(lpp[tok] - lpp.max())) for tok, _, _, lpp in spans]

    phonemes = [
        {
            "phone": acoustic.id2phone[tok],
            "score": sc,
            "label": score_to_label(sc),
            "start_sec": round(start * config.FRAME_SEC, 3),
            "end_sec": round(end * config.FRAME_SEC, 3),
        }
        for (tok, start, end, _), sc in zip(spans, scores)
    ]
    overall = int(round(sum(p["score"] for p in phonemes) / max(len(phonemes), 1)))
    return phones, phonemes, overall
