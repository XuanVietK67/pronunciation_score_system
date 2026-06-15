"""Audio decoding, resampling and a basic quality gate."""
from __future__ import annotations

import io
import math
import subprocess

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from . import config


def load_16k_mono(file_bytes: bytes) -> torch.Tensor:
    """Decode audio bytes -> mono float32 tensor at 16 kHz.

    Fast path is libsndfile (wav/flac/ogg). Browser recordings (webm/opus, mp4/m4a,
    mp3) aren't handled by libsndfile, so we fall back to ffmpeg for those.

    Raises ValueError if the bytes can't be decoded at all -> callers map it to HTTP 422.
    """
    try:
        data, sr = sf.read(io.BytesIO(file_bytes), dtype="float32", always_2d=True)
    except Exception:
        # Not a libsndfile-supported container; let ffmpeg handle it (returns 16 kHz mono).
        return _decode_with_ffmpeg(file_bytes)
    wav = torch.from_numpy(data).mean(dim=1)  # average channels -> mono
    if sr != config.SAMPLE_RATE:
        wav = AF.resample(wav, sr, config.SAMPLE_RATE)
    return wav.contiguous()


def _decode_with_ffmpeg(file_bytes: bytes) -> torch.Tensor:
    """Transcode arbitrary audio bytes -> 16 kHz mono float32 via ffmpeg (reads stdin).

    Handles the formats libsndfile can't (webm/opus, mp4/m4a, mp3, ...). Requires the
    ffmpeg binary on PATH (installed in the Docker image).
    """
    try:
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0",
             "-ar", str(config.SAMPLE_RATE), "-ac", "1", "-f", "f32le", "pipe:1"],
            input=file_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except FileNotFoundError as exc:  # ffmpeg not installed
        raise RuntimeError("ffmpeg not available to decode this audio format") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", "ignore").strip().splitlines()[-1:] or [""]
        raise ValueError(f"could not decode audio: {detail[0]}") from exc

    samples = np.frombuffer(proc.stdout, dtype=np.float32).copy()  # copy: buffer is read-only
    if samples.size == 0:
        raise ValueError("decoded audio is empty")
    return torch.from_numpy(samples).contiguous()


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
