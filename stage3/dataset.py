"""Torch Dataset + padding collate over the cached Stage-3 features.

Features are per-phone pooled wav2vec2 hidden-state vectors (one [H] vector per phone).
"""
from __future__ import annotations

import json

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class GOPDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ):
        with open(manifest_path) as f:
            self.items = json.load(f)
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        rec = torch.load(self.items[i]["path"])
        feats = rec["features"]
        if self.mean is not None:
            feats = (feats - self.mean) / self.std
        return feats, rec["phone_ids"], rec["labels"]


def compute_feature_stats(manifest_path: str, eps: float = 1e-5):
    """Per-dim mean/std over all phones in a split (for input standardization)."""
    ds = GOPDataset(manifest_path)
    total = sq = None
    count = 0
    for feats, _, _ in ds:
        s, s2 = feats.sum(0), (feats ** 2).sum(0)
        total = s if total is None else total + s
        sq = s2 if sq is None else sq + s2
        count += feats.shape[0]
    mean = total / count
    std = (sq / count - mean ** 2).clamp_min(0).sqrt().clamp_min(eps)
    return mean, std


def collate(batch):
    """Pad a batch of variable-length phone sequences. Returns
    (features [B,L,H], phone_ids [B,L], labels [B,L], pad_mask [B,L] True=pad, lengths [B])."""
    feats, pids, labels = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in feats])
    features = pad_sequence(feats, batch_first=True)
    phone_ids = pad_sequence(pids, batch_first=True)
    labels_p = pad_sequence(labels, batch_first=True)
    max_len = features.shape[1]
    pad_mask = torch.arange(max_len)[None, :] >= lengths[:, None]
    return features, phone_ids, labels_p, pad_mask, lengths
