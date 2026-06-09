"""Persist the artifacts the inference service needs to mirror training exactly.

The GOPT head was trained on per-phone pooled wav2vec2 hidden states **standardized
with the train mean/std**, so inference must reproduce that same feature. This module
saves the two things the service can't otherwise reconstruct:

- ``feature_stats.pt`` — the train ``mean``/``std`` used for input standardization.
- ``model_config.json`` — the GOPT constructor args, so ``app/scorer.py`` rebuilds the
  model identically instead of hardcoding hyperparams in two places.

Both ``stage3/train.py`` (after a fresh run) and ``stage3/export_artifacts.py`` (no
retrain) write these via the helpers below.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

FEATURE_STATS_NAME = "feature_stats.pt"
MODEL_CONFIG_NAME = "model_config.json"
CHECKPOINT_NAME = "model.ckpt"


def save_feature_stats(out_dir: str | Path, mean: torch.Tensor, std: torch.Tensor) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / FEATURE_STATS_NAME
    torch.save({"mean": mean.cpu(), "std": std.cpu()}, path)
    return path


def save_model_config(
    out_dir: str | Path,
    *,
    input_dim: int,
    vocab_size: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dropout: float,
    dim_feedforward: int = 256,
    max_len: int = 256,
) -> dict:
    """Persist the GOPT constructor args. Keys match ``GOPT.__init__`` exactly so the
    service can call ``GOPT(**json.load(...))``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {
        "input_dim": int(input_dim),
        "vocab_size": int(vocab_size),
        "d_model": int(d_model),
        "nhead": int(nhead),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "dim_feedforward": int(dim_feedforward),
        "max_len": int(max_len),
    }
    (out / MODEL_CONFIG_NAME).write_text(json.dumps(cfg, indent=2))
    return cfg
