"""FastAPI service: POST /score (Stage 1 + 2 + trained GOPT head)."""
from __future__ import annotations

from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from . import audio as audio_io
from . import config
from .acoustic import Acoustic
from .g2p import build_maps
from .schema import ScoreResponse
from .score_pipeline import score_word
from .scorer import load_scorer

_state: dict = {}


def _warmup(acoustic, exact, norm, scorer) -> None:
    """Run one throwaway inference so the first real request doesn't pay lazy-init costs.

    The first forward pass triggers one-time work (oneDNN/MKL kernel selection, memory
    pools, the espeak backend's first call). Paying it here, during startup, keeps the
    first user request as fast as steady state. Best-effort: a warmup failure must not
    block the service from coming up.
    """
    try:
        silence = torch.zeros(int(config.SAMPLE_RATE * 0.5))  # 0.5 s, passes the gate
        score_word(silence, "warmup", acoustic, exact, norm, scorer)
        print("Warmup inference complete")
    except Exception as exc:  # noqa: BLE001 — never let warmup crash startup
        print(f"Warmup inference skipped: {exc!r}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    acoustic = Acoustic()
    exact, norm = build_maps(acoustic.vocab)
    scorer = None
    if config.USE_TRAINED_HEAD:
        scorer = load_scorer()  # fails fast with a clear message if artifacts are missing
        print(f"Loaded trained GOPT head from {config.ARTIFACTS_DIR}")
    else:
        print("USE_TRAINED_HEAD=false — using placeholder GOP->score mapping")
    _warmup(acoustic, exact, norm, scorer)
    _state.update(acoustic=acoustic, exact=exact, norm=norm, scorer=scorer)
    yield
    _state.clear()


app = FastAPI(title="Pronunciation Scoring", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": config.MODEL_ID,
        "model_version": config.MODEL_VERSION,
        "trained_head": bool(_state.get("scorer")),
        "ready": bool(_state),
    }


@app.post("/score", response_model=ScoreResponse)
async def score(audio: UploadFile = File(...), word: str = Form(...)):
    if not _state:
        raise HTTPException(503, "model not loaded yet")

    try:
        wav = audio_io.load_16k_mono(await audio.read())
    except ValueError as exc:  # undecodable / empty upload
        raise HTTPException(422, str(exc))
    q = audio_io.quality(wav)
    if q["too_short"]:
        raise HTTPException(422, "audio too short")

    try:
        # score_word is synchronous CPU work (the wav2vec2 forward pass); run it in a
        # threadpool so it doesn't block the event loop for other concurrent requests.
        phones, phonemes, overall = await run_in_threadpool(
            score_word, wav, word, _state["acoustic"], _state["exact"], _state["norm"], _state["scorer"]
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    return {
        "word": word,
        "transcript_phonemes": phones,
        "overall_score": overall,
        "phonemes": phonemes,
        "audio_quality": q,
        "model_version": config.MODEL_VERSION,
    }
