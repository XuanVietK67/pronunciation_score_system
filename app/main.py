"""FastAPI service: POST /score (Stage 1 + 2 + trained GOPT head)."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from . import audio as audio_io
from . import config
from .acoustic import Acoustic
from .g2p import build_maps
from .schema import ScoreResponse
from .score_pipeline import score_word
from .scorer import load_scorer

_state: dict = {}


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

    wav = audio_io.load_16k_mono(await audio.read())
    q = audio_io.quality(wav)
    if q["too_short"]:
        raise HTTPException(422, "audio too short")

    try:
        phones, phonemes, overall = score_word(
            wav, word, _state["acoustic"], _state["exact"], _state["norm"], _state["scorer"]
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
