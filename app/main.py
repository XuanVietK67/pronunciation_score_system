"""FastAPI service: POST /score (Stage 1 + 2 spike)."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from . import audio as audio_io
from . import config, gop
from .acoustic import Acoustic
from .g2p import build_maps, g2p, to_token_ids
from .schema import ScoreResponse

_state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    acoustic = Acoustic()
    exact, norm = build_maps(acoustic.vocab)
    _state.update(acoustic=acoustic, exact=exact, norm=norm)
    yield
    _state.clear()


app = FastAPI(title="Pronunciation Scoring (Stage 1+2 spike)", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": config.MODEL_ID, "ready": bool(_state)}


@app.post("/score", response_model=ScoreResponse)
async def score(audio: UploadFile = File(...), word: str = Form(...)):
    if not _state:
        raise HTTPException(503, "model not loaded yet")

    wav = audio_io.load_16k_mono(await audio.read())
    q = audio_io.quality(wav)
    if q["too_short"]:
        raise HTTPException(422, "audio too short")

    phones = g2p(word)
    token_ids, unmapped = to_token_ids(phones, _state["exact"], _state["norm"])
    if not token_ids:
        raise HTTPException(422, f"no phones produced for '{word}'")
    if unmapped:
        raise HTTPException(422, f"unmapped phones for '{word}': {unmapped}")

    acoustic: Acoustic = _state["acoustic"]
    log_probs = acoustic.log_posteriors(wav)
    try:
        feats = gop.align_and_score(
            log_probs, token_ids, acoustic.id2phone, acoustic.blank_id
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    phonemes = []
    for feat in feats:
        sc = gop.gop_to_score(feat["gop"])
        phonemes.append(
            {
                "phone": feat["phone"],
                "score": sc,
                "label": gop.score_to_label(sc),
                "gop": round(feat["gop"], 4),
                "start_sec": feat["start_sec"],
                "end_sec": feat["end_sec"],
            }
        )
    overall = int(round(sum(p["score"] for p in phonemes) / max(len(phonemes), 1)))

    return {
        "word": word,
        "transcript_phonemes": phones,
        "overall_score": overall,
        "phonemes": phonemes,
        "audio_quality": q,
        "model_version": config.MODEL_VERSION,
    }
