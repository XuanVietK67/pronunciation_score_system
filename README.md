# learning-vocab-pronunciation

Pronunciation-scoring microservice for the learning-vocab platform. **Stage 1 + 2 spike**:
frozen pretrained wav2vec2+CTC phoneme model → aligned GOP → rough per-phone score.

Design & plan live in the backend repo:
`docs/pronunciation_scoring_design.md` and `docs/pronunciation_stage1_2_plan.md`.

> The 0–100 score here uses a **placeholder** GOP→score map (`app/gop.py:gop_to_score`).
> Calibrated scores arrive with the Stage 3 GOPT head.

## Layout

```
app/
  config.py    tunables (model id, sample rate, thresholds)
  audio.py     decode -> mono 16 kHz + quality gate
  acoustic.py  Stage 1: model -> per-frame log-posteriors
  g2p.py       espeak G2P + phone->token-id reconciliation
  gop.py       Stage 2: forced align + GOP features
  schema.py    response models
  main.py      FastAPI: POST /score, GET /health
scripts/inspect_acoustic.py   step-1 smoke CLI
tests/         unit tests (no model download needed)
```

## Setup

Requires **espeak-ng** at the OS level (phonemizer backend) and libsndfile.

```bash
# Linux / Docker
sudo apt-get install -y espeak-ng libsndfile1
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Windows dev: install espeak-ng from its releases and, if phonemizer can't find it, set
`PHONEMIZER_ESPEAK_LIBRARY` to the `libespeak-ng.dll` path. Or develop in WSL/Docker.

## Run

```bash
# step-1 sanity: what does the model hear?
python -m scripts.inspect_acoustic path/to/word.wav

# the service
uvicorn app.main:app --reload
curl -F "audio=@word.wav" -F "word=thin" http://localhost:8000/score
```

First run downloads the model (~1 GB) from HuggingFace.

## Test

```bash
pytest -q          # phone-map + GOP unit tests run without the model
```

## Docker

```bash
docker build -t lv-pronunciation .
docker run -p 8000:8000 lv-pronunciation
```
