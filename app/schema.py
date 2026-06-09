"""Response models for POST /score (mirrors design doc §2b)."""
from __future__ import annotations

from pydantic import BaseModel


class PhonemeScore(BaseModel):
    phone: str
    score: int
    label: str
    start_sec: float
    end_sec: float


class AudioQuality(BaseModel):
    duration_sec: float
    too_short: bool
    clipping: bool
    snr_db: float


class ScoreResponse(BaseModel):
    word: str
    transcript_phonemes: list[str]
    overall_score: int
    phonemes: list[PhonemeScore]
    audio_quality: AudioQuality
    model_version: str
