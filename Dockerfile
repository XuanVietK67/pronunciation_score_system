FROM python:3.11-slim

# espeak-ng: phonemizer backend; libsndfile1: soundfile decoding;
# ffmpeg: decode browser recordings (webm/opus, mp4/m4a, mp3) that libsndfile can't.
RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
# CPU-only torch/torchaudio (~200 MB) instead of the default CUDA wheels (~2.5 GB).
# This service scores single words; CPU is plenty and the build is ~10x smaller/faster.
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Bake the ~1 GB wav2vec2 acoustic model into the image so a fresh deploy / Space wake
# doesn't re-download it from HuggingFace on boot. HF_HOME points the cache at a known
# in-image path; keep MODEL_ID in sync with app/config.py:MODEL_ID.
ENV HF_HOME=/srv/hf-cache
ARG MODEL_ID=facebook/wav2vec2-xlsr-53-espeak-cv-ft
RUN python -c "from transformers import AutoProcessor, Wav2Vec2ForCTC; \
    AutoProcessor.from_pretrained('${MODEL_ID}'); \
    Wav2Vec2ForCTC.from_pretrained('${MODEL_ID}')"

# Model is baked above -> at runtime load from the cache only: no network call and no
# lock-file writes on boot (works when the Space runs the container as a non-root user).
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

COPY app ./app
# stage3/model.py defines GOPT (rebuilt at load time); stage3 is otherwise training-only.
COPY stage3 ./stage3
# Trained head bundle: model.ckpt + feature_stats.pt + model_config.json (~MBs).
# Generate with `python -m stage3.export_artifacts` before building.
COPY artifacts ./artifacts
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
