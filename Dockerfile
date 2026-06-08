FROM python:3.11-slim

# espeak-ng: phonemizer backend; libsndfile1: soundfile decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
# CPU-only torch/torchaudio (~200 MB) instead of the default CUDA wheels (~2.5 GB).
# This service scores single words; CPU is plenty and the build is ~10x smaller/faster.
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
