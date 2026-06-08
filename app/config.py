"""Service configuration and tunables (Stage 1 + 2 spike)."""

MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
MODEL_VERSION = "gop-wav2vec2-espeak-spike-v0"

SAMPLE_RATE = 16000
FRAME_SEC = 0.02  # wav2vec2 stride = 320 samples @ 16 kHz -> 20 ms / frame

# PLACEHOLDER GOP -> score mapping. Stand-in until the Stage 3 GOPT head exists.
# score = 100 * sigmoid(GOP_SCALE_A * gop + GOP_SCALE_B); gop is <= 0 (ratio form).
GOP_SCALE_A = 1.0
GOP_SCALE_B = 2.0

# Feedback label thresholds (design doc §2b).
LABEL_GOOD = 75
LABEL_PRACTICE = 45

# Audio quality gate.
MIN_DURATION_SEC = 0.2
CLIPPING_FRACTION = 0.01  # fraction of near-full-scale samples above which we flag clipping
