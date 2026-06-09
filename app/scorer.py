"""Inference scorer — serve the trained GOPT head.

Loads the GOPT checkpoint + the train normalization stats and maps a sequence of
per-phone pooled wav2vec2 features to calibrated 0-100 scores. The feature path here
**must mirror training exactly** (same pooling upstream, same ``(x - mean) / std`` here),
or the scores are garbage. See stage3/extract_features.py + stage3/dataset.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from stage3.model import GOPT

from . import config


def pred_to_score(pred: float) -> int:
    """Map a GOPT prediction (regressed toward speechocean762's [0, 2] labels) to 0-100."""
    return int(round(min(100.0, max(0.0, pred / 2.0 * 100.0))))


class Scorer:
    def __init__(self, model: GOPT, mean: torch.Tensor, std: torch.Tensor):
        self.model = model
        self.mean = mean
        self.std = std

    @torch.no_grad()
    def score(self, features: torch.Tensor, phone_ids: list[int]) -> list[int]:
        """features [L, H] pooled per phone, phone_ids [L] -> per-phone 0-100 scores."""
        feats = (features.to(torch.float32) - self.mean) / self.std  # identical to dataset.py
        pids = torch.tensor(phone_ids, dtype=torch.long)
        preds = self.model(feats.unsqueeze(0), pids.unsqueeze(0)).squeeze(0)  # [L]
        return [pred_to_score(p) for p in preds.tolist()]


def load_scorer(
    checkpoint_path: Path = config.CHECKPOINT_PATH,
    stats_path: Path = config.FEATURE_STATS_PATH,
    config_path: Path = config.MODEL_CONFIG_PATH,
) -> Scorer:
    """Rebuild GOPT from the persisted config, load weights + stats, set eval().

    Fails fast with a clear message if any artifact is missing, so a misconfigured
    deploy is caught at startup rather than on the first request.
    """
    for name, path in (
        ("model config", config_path),
        ("feature stats", stats_path),
        ("checkpoint", checkpoint_path),
    ):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"trained-head {name} not found at {path}. "
                "Run `python -m stage3.export_artifacts` (or set USE_TRAINED_HEAD=false)."
            )

    cfg = json.loads(Path(config_path).read_text())
    stats = torch.load(stats_path)
    mean, std = stats["mean"].to(torch.float32), stats["std"].to(torch.float32)
    if mean.numel() != cfg["input_dim"]:
        raise ValueError(
            f"feature_stats dim {mean.numel()} != model_config input_dim {cfg['input_dim']}"
        )

    model = GOPT(**cfg)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return Scorer(model, mean, std)
