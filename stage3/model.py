"""Stage 3 model — GOPT: a small transformer over per-phone GOP feature vectors.

Input is the sequence of per-phone LPP vectors (from Stage 2) plus the canonical phone id.
Output is one accuracy score per phone, regressed toward speechocean762's [0, 2] labels.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GOPT(nn.Module):
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 256,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.phone_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(d_model, 1)
        self.max_len = max_len

    def forward(
        self,
        features: torch.Tensor,        # [B, L, input_dim]  (LPP vectors)
        phone_ids: torch.Tensor,       # [B, L]
        padding_mask: torch.Tensor | None = None,  # [B, L], True = pad
    ) -> torch.Tensor:                 # [B, L] per-phone score
        length = features.shape[1]
        pos = torch.arange(length, device=features.device).clamp_max(self.max_len - 1)
        x = self.input_proj(features) + self.phone_emb(phone_ids) + self.pos_emb(pos)[None]
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.head(x).squeeze(-1)
