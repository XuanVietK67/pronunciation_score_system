# Audio Scoring Flow — `POST /score`

This document walks through everything that happens, in order, when a client sends
an HTTP request containing a recorded audio clip plus a target word to the
pronunciation-scoring service.

The flow turns **raw audio bytes → per-phone pronunciation scores (0–100)**.

---

## TL;DR pipeline

```
POST /score (audio + word)
   │
   ▼
audio.read()                    raw bytes from the multipart upload
   │
   ▼
load_16k_mono()                 decode → mono → resample to 16 kHz   [app/audio.py]
   │
   ▼
quality()                       duration / clipping / SNR gate       [app/audio.py]
   │   (reject 422 if too short)
   ▼
score_word()                                                          [app/score_pipeline.py]
   ├─ g2p(word) + to_token_ids()      word → canonical phones → ids   [app/g2p.py]
   ├─ acoustic.forward_features(wav)  wav2vec2+CTC → log_probs, hidden [app/acoustic.py]
   ├─ align_spans()                   forced alignment, 1 span/phone  [app/gop.py]
   └─ scorer.score()  OR  gop_to_score()   features → 0–100 per phone [app/scorer.py / app/gop.py]
   │
   ▼
JSON ScoreResponse              word, phones, per-phone scores, overall, quality
```

---

## 0. Startup — models are loaded once, not per request

Before any request is served, FastAPI's `lifespan` hook loads the heavy models a
single time and stashes them in the module-level `_state` dict.

[app/main.py:19-31](../app/main.py#L19-L31)

```python
@asynccontextmanager
async def lifespan(_app: FastAPI):
    acoustic = Acoustic()                     # frozen wav2vec2 + CTC (Stage 1)
    exact, norm = build_maps(acoustic.vocab)  # phone → id lookup tables
    scorer = None
    if config.USE_TRAINED_HEAD:
        scorer = load_scorer()                # trained GOPT head (Stage 3)
    _state.update(acoustic=acoustic, exact=exact, norm=norm, scorer=scorer)
    yield
    _state.clear()
```

- `acoustic` — the pretrained `facebook/wav2vec2-xlsr-53-espeak-cv-ft` acoustic model.
- `exact` / `norm` — phone→id maps built from the model vocab (exact match + a
  normalized fallback that strips stress/length/diacritics).
- `scorer` — the trained GOPT head, **or `None`** when `USE_TRAINED_HEAD=false`
  (a rollback switch — the service then uses the placeholder GOP→score mapping).

Loading at startup means a misconfigured deploy (missing artifacts) fails
**immediately**, not on the first user request.

---

## 1. Request arrives at the endpoint

[app/main.py:48-56](../app/main.py#L48-L56)

```python
@app.post("/score", response_model=ScoreResponse)
async def score(audio: UploadFile = File(...), word: str = Form(...)):
    if not _state:
        raise HTTPException(503, "model not loaded yet")

    wav = audio_io.load_16k_mono(await audio.read())
    q = audio_io.quality(wav)
    if q["too_short"]:
        raise HTTPException(422, "audio too short")
```

- The request is **multipart/form-data** with two parts:
  - `audio` — the uploaded file (`UploadFile`).
  - `word` — the target word being practiced (form field).
- `if not _state` → **503** if a request races startup before models finish loading.
- `await audio.read()` reads the **raw file bytes** into memory.

---

## 2. Decode + resample the audio → `load_16k_mono`

[app/audio.py:14-24](../app/audio.py#L14-L24)

```python
def load_16k_mono(file_bytes: bytes) -> torch.Tensor:
    data, sr = sf.read(io.BytesIO(file_bytes), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data).mean(dim=1)           # average channels → mono
    if sr != config.SAMPLE_RATE:
        wav = AF.resample(wav, sr, config.SAMPLE_RATE) # → 16 kHz
    return wav.contiguous()
```

- Decodes the bytes with **libsndfile** (`soundfile`) → float32 samples + sample rate.
- Averages all channels down to **mono**.
- Resamples to **16 kHz** (`config.SAMPLE_RATE`), the rate wav2vec2 expects.

> **Format support:** libsndfile is the fast path (`wav` / `flac` / `ogg`). Browser
> recordings (`webm`/`opus`, `mp4`/`m4a`, `mp3`) fall back to an **ffmpeg** decode
> (`_decode_with_ffmpeg`), which transcodes to 16 kHz mono. Undecodable bytes raise
> `ValueError` → **422**; a missing ffmpeg binary raises `RuntimeError` → **500**
> (server misconfiguration, not a bad upload). ffmpeg is installed in the Docker image.

---

## 3. Quality gate → `quality`

[app/audio.py:27-40](../app/audio.py#L27-L40)

```python
def quality(wav: torch.Tensor) -> dict:
    duration = wav.numel() / config.SAMPLE_RATE
    peak = wav.abs()
    clipping = (peak > 0.99).float().mean().item() > config.CLIPPING_FRACTION
    rms = float(peak.pow(2).mean().sqrt())
    snr_db = round(20.0 * math.log10(rms + 1e-8) + 60.0, 1)
    return {
        "duration_sec": round(duration, 3),
        "too_short": duration < config.MIN_DURATION_SEC,   # 0.2 s
        "clipping": clipping,
        "snr_db": snr_db,
    }
```

Cheap signal sanity checks (not a real VAD):

| Field          | Meaning |
|----------------|---------|
| `duration_sec` | Clip length in seconds. |
| `too_short`    | `True` if shorter than `MIN_DURATION_SEC` (0.2 s) → request rejected with **422**. |
| `clipping`     | `True` if too many near-full-scale samples (`> CLIPPING_FRACTION`). |
| `snr_db`       | Rough level proxy vs. a –60 dBFS noise floor (not a calibrated SNR). |

The full `quality` dict is also returned to the client as `audio_quality` so the UI
can warn about a bad recording.

---

## 4. Core pipeline → `score_word`

This is the heart of the flow. One shared implementation is used by both the HTTP
endpoint and the CLI (`scripts/score_file.py`) so the two paths never drift.

[app/score_pipeline.py:19-68](../app/score_pipeline.py#L19-L68)

```python
def score_word(wav, word, acoustic, exact, norm, scorer):
    phones = g2p(word)                              # 4a
    token_ids, unmapped = to_token_ids(phones, exact, norm)
    if not token_ids:
        raise ValueError(f"no phones produced for '{word}'")
    if unmapped:
        raise ValueError(f"unmapped phones for '{word}': {unmapped}")

    log_probs, hidden = acoustic.forward_features(wav)   # 4b
    if len(token_ids) > log_probs.shape[0]:
        raise ValueError("audio too short for the number of target phones")

    spans = align_spans(log_probs, token_ids, acoustic.blank_id)   # 4c

    if scorer is not None:                                # 4d (trained head)
        features = torch.stack(
            [hidden[start:end].mean(0) for _, start, end, _ in spans]
        ).to(torch.float32)
        aligned_ids = [tok for tok, _, _, _ in spans]
        scores = scorer.score(features, aligned_ids)
    else:                                                 # 4d (placeholder)
        scores = [gop_to_score(float(lpp[tok] - lpp.max())) for tok, _, _, lpp in spans]

    phonemes = [ ... per-phone dict ... ]                 # 4e
    overall = int(round(sum(p["score"] for p in phonemes) / max(len(phonemes), 1)))
    return phones, phonemes, overall
```

### 4a. Word → canonical phones → token ids (G2P)

[app/g2p.py:18-58](../app/g2p.py#L18-L58)

- `g2p(word)` runs the **espeak** backend via `phonemizer` to produce the canonical
  phone sequence (IPA/espeak symbols), e.g. `"water" → ["w", "ɔ", "t", "ɚ"]`.
- `to_token_ids` maps each phone to the model's vocab id:
  - exact match first,
  - then a **normalized** match (stress/length/diacritics stripped),
  - anything still unmapped is collected.
- **Unmapped or empty phones raise `ValueError`** → the endpoint converts it to a
  **422**. A phone the model can't represent is treated as an error, *not* a low score.

### 4b. Acoustic forward pass (Stage 1)

[app/acoustic.py:29-42](../app/acoustic.py#L29-L42)

```python
def forward_features(self, wav_16k):
    inputs = self.processor(wav_16k.numpy(), sampling_rate=config.SAMPLE_RATE, return_tensors="pt")
    out = self.model(inputs.input_values, output_hidden_states=True)
    log_probs = out.logits[0].log_softmax(-1)  # [T, V] phoneme log-posteriors
    hidden = out.hidden_states[-1][0]           # [T, H] encoder embeddings
    return log_probs, hidden
```

One wav2vec2+CTC forward pass yields two things per 20 ms frame:
- `log_probs [T, V]` — phoneme log-posteriors that **drive forced alignment**.
- `hidden [T, H]` — rich encoder embeddings that become the **features** for the head.

### 4c. Forced alignment → `align_spans` (Stage 2)

[app/gop.py:12-33](../app/gop.py#L12-L33)

```python
aligned, scores = forced_align(emission, targets, blank=blank_id)
spans = merge_tokens(aligned[0], scores[0])   # one span per target phone
```

Aligns the canonical phone sequence to the audio frames and returns **one span per
phone**: `(token_id, start_frame, end_frame_exclusive, lpp)` where `lpp` is the mean
log-posterior vector over that span. If there are more target phones than frames →
`ValueError` ("audio too short for the number of target phones") → **422**.

### 4d. Scoring — two paths

**Trained GOPT head (`scorer is not None`, the default):**

[app/scorer.py:31-37](../app/scorer.py#L31-L37)

```python
def score(self, features, phone_ids):
    feats = (features.to(torch.float32) - self.mean) / self.std   # standardize (== training)
    pids = torch.tensor(phone_ids, dtype=torch.long)
    preds = self.model(feats.unsqueeze(0), pids.unsqueeze(0)).squeeze(0)
    return [pred_to_score(p) for p in preds.tolist()]
```

- The wav2vec2 hidden states are **mean-pooled per phone span** → `[L, H]`.
- Standardized with the **train-time mean/std** (must mirror training exactly, or the
  scores are meaningless).
- The GOPT model regresses a value toward speechocean762's `[0, 2]` labels; `pred_to_score`
  maps that to a clamped **0–100** integer.

**Placeholder fallback (`USE_TRAINED_HEAD=false`):**

[app/gop.py:63-66](../app/gop.py#L63-L66) — `gop_to_score` applies a fixed sigmoid over the
raw GOP value (`lpp[tok] - lpp.max()`). Used for rollback without redeploying artifacts.

### 4e. Assemble per-phone results + overall

[app/score_pipeline.py:57-68](../app/score_pipeline.py#L57-L68)

Each phone becomes a dict with its score, a coarse **label** (`good` / `practice` /
`wrong`, thresholds in [app/gop.py:69-74](../app/gop.py#L69-L74)) and its time span in
seconds (`frame_index × FRAME_SEC`, 20 ms/frame). The **overall score** is the mean of
the per-phone scores.

---

## 5. Response

[app/main.py:65-72](../app/main.py#L65-L72), validated against
[app/schema.py](../app/schema.py)

```json
{
  "word": "water",
  "transcript_phonemes": ["w", "ɔ", "t", "ɚ"],
  "overall_score": 82,
  "phonemes": [
    { "phone": "w", "score": 90, "label": "good", "start_sec": 0.0, "end_sec": 0.12 },
    { "phone": "ɔ", "score": 78, "label": "good", "start_sec": 0.12, "end_sec": 0.30 }
  ],
  "audio_quality": { "duration_sec": 0.74, "too_short": false, "clipping": false, "snr_db": 41.2 },
  "model_version": "gopt-wav2vec2-espeak-v1"
}
```

---

## Failure modes (what the client sees)

| Status | When | Source |
|--------|------|--------|
| **503** | A request arrives before models finish loading. | [main.py:50-51](../app/main.py#L50-L51) |
| **422** | Audio shorter than `MIN_DURATION_SEC`. | [main.py:55-56](../app/main.py#L55-L56) |
| **422** | Word produces no phones / unmapped phones. | [score_pipeline.py:33-36](../app/score_pipeline.py#L33-L36) |
| **422** | More target phones than audio frames ("audio too short"). | [score_pipeline.py:39-40](../app/score_pipeline.py#L39-L40) |
| **200** | Success — full `ScoreResponse`. | [main.py:65-72](../app/main.py#L65-L72) |

All in-pipeline `ValueError`s are caught at [main.py:62-63](../app/main.py#L62-L63) and
converted to **422** with the message as the detail.

---

## Stage map

| Stage | Responsibility | Code |
|-------|----------------|------|
| **1 — Acoustic** | Pretrained wav2vec2 + CTC → per-frame phoneme posteriors + hidden states. | [app/acoustic.py](../app/acoustic.py) |
| **2 — Alignment / GOP** | Forced alignment, one span per canonical phone; placeholder GOP→score. | [app/gop.py](../app/gop.py) |
| **3 — GOPT head** | Trained model: pooled features → calibrated 0–100 scores. | [app/scorer.py](../app/scorer.py), `stage3/` |
| **Glue** | G2P, audio I/O, pipeline orchestration, HTTP. | [app/g2p.py](../app/g2p.py), [app/audio.py](../app/audio.py), [app/score_pipeline.py](../app/score_pipeline.py), [app/main.py](../app/main.py) |
