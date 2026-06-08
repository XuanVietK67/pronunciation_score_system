"""Audio decoding, resampling and a basic quality gate."""
from __future__ import annotations

import io
import math

import soundfile as sf
import torch
import torchaudio.functional as AF

from . import config


def load_16k_mono(file_bytes: bytes) -> torch.Tensor:
    """Decode audio bytes -> mono float32 tensor at 16 kHz.

    Uses libsndfile (wav/flac/ogg). webm/opus/mp3 from the browser must be transcoded
    upstream (or add an ffmpeg decode path) — that's a known spike limitation.
    """
    data, sr = sf.read(io.BytesIO(file_bytes), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data).mean(dim=1)  # average channels -> mono
    if sr != config.SAMPLE_RATE:
        wav = AF.resample(wav, sr, config.SAMPLE_RATE)
    return wav.contiguous()


def quality(wav: torch.Tensor) -> dict:
    """Cheap signal sanity checks. Not a substitute for real VAD."""
    duration = wav.numel() / config.SAMPLE_RATE
    peak = wav.abs()
    clipping = (peak > 0.99).float().mean().item() > config.CLIPPING_FRACTION
    rms = float(peak.pow(2).mean().sqrt())
    # Rough level proxy relative to a -60 dBFS noise floor; not a calibrated SNR.
    snr_db = round(20.0 * math.log10(rms + 1e-8) + 60.0, 1)
    return {
        "duration_sec": round(duration, 3),
        "too_short": duration < config.MIN_DURATION_SEC,
        "clipping": clipping,
        "snr_db": snr_db,
    }
