"""Stage 2 — forced alignment + GOP feature extraction (aligned GOP)."""
from __future__ import annotations

import math

import torch
from torchaudio.functional import forced_align, merge_tokens

from . import config


def align_spans(
    log_probs: torch.Tensor, token_ids: list[int], blank_id: int
) -> list[tuple[int, int, int, torch.Tensor]]:
    """Force-align canonical phones to frames.

    Returns one tuple per canonical phone: (token_id, start_frame, end_frame_exclusive,
    lpp) where lpp is the mean log-posterior over the span for every phone (the [V] feature
    vector). Both Stage-2 scoring and Stage-3 feature extraction build on this.
    """
    if len(token_ids) > log_probs.shape[0]:
        raise ValueError("audio too short for the number of target phones")

    targets = torch.tensor([token_ids], dtype=torch.int32)
    emission = log_probs.unsqueeze(0).contiguous()  # [1, T, V]
    aligned, scores = forced_align(emission, targets, blank=blank_id)
    spans = merge_tokens(aligned[0], scores[0])  # one span per target phone

    out: list[tuple[int, int, int, torch.Tensor]] = []
    for span in spans:
        seg = log_probs[span.start : span.end]  # [span_len, V]; span.end is exclusive
        out.append((int(span.token), span.start, span.end, seg.mean(dim=0)))
    return out


def align_and_score(
    log_probs: torch.Tensor,
    token_ids: list[int],
    id2phone: dict[int, str],
    blank_id: int,
    frame_sec: float = config.FRAME_SEC,
) -> list[dict]:
    """Force-align canonical phones to frames and compute per-phone GOP features.

    Returns one dict per canonical phone with: phone, time span, scalar gop (ratio form,
    <= 0), and the LPP feature vector the Stage 3 GOPT head will consume.
    """
    results: list[dict] = []
    for token_id, start, end, lpp in align_spans(log_probs, token_ids, blank_id):
        gop = float(lpp[token_id] - lpp.max())  # <= 0; 0 means the canonical phone dominated
        results.append(
            {
                "phone": id2phone[token_id],
                "start_sec": round(start * frame_sec, 3),
                "end_sec": round(end * frame_sec, 3),
                "gop": gop,
                "lpp": lpp.tolist(),  # feature vector reserved for Stage 3
            }
        )
    return results


def gop_to_score(gop: float) -> int:
    """PLACEHOLDER calibration — replaced by the trained GOPT head in Stage 3."""
    s = 1.0 / (1.0 + math.exp(-(config.GOP_SCALE_A * gop + config.GOP_SCALE_B)))
    return int(round(100 * s))


def score_to_label(score: int) -> str:
    if score >= config.LABEL_GOOD:
        return "good"
    if score >= config.LABEL_PRACTICE:
        return "practice"
    return "wrong"
