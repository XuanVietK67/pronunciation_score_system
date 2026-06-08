"""Stage 1 — frozen pretrained wav2vec2 + CTC phoneme acoustic model."""
from __future__ import annotations

import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

from . import config


class Acoustic:
    """Loads the pretrained model once and emits per-frame phoneme log-posteriors."""

    def __init__(self, model_id: str = config.MODEL_ID):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).eval()
        self.blank_id = self.processor.tokenizer.pad_token_id  # CTC blank
        self.vocab = self.processor.tokenizer.get_vocab()       # phone -> id
        self.id2phone = {idx: phone for phone, idx in self.vocab.items()}

    @torch.no_grad()
    def log_posteriors(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """wav (1-D float @ 16 kHz) -> [T_frames, V_phones] log-posteriors."""
        inputs = self.processor(
            wav_16k.numpy(), sampling_rate=config.SAMPLE_RATE, return_tensors="pt"
        )
        logits = self.model(inputs.input_values).logits[0]  # [T, V]
        return logits.log_softmax(-1)
