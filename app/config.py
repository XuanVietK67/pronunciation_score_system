"""Service configuration and tunables."""

import os
from pathlib import Path

MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
MODEL_VERSION = "gopt-wav2vec2-espeak-v1"

# Trained GOPT head artifacts (produced by stage3/export_artifacts.py or train.py).
# Override the directory in containers via ARTIFACTS_DIR; defaults to <repo>/artifacts.
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", Path(__file__).resolve().parent.parent / "artifacts"))
CHECKPOINT_PATH = ARTIFACTS_DIR / "model.ckpt"
FEATURE_STATS_PATH = ARTIFACTS_DIR / "feature_stats.pt"
MODEL_CONFIG_PATH = ARTIFACTS_DIR / "model_config.json"

# Serve the trained GOPT head. Set USE_TRAINED_HEAD=false to fall back to the
# placeholder gop_to_score mapping (rollback without a redeploy of artifacts).
USE_TRAINED_HEAD = os.environ.get("USE_TRAINED_HEAD", "true").strip().lower() not in {"0", "false", "no"}

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
