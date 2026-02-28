"""
VoiceShield v2 — Training Configuration
All hyperparameters in one place.

Paths are configurable via environment variables:
  VOICESHIELD_DATASET  → dataset root  (default: <project>/data/dataset)
  VOICESHIELD_MODELS   → model output  (default: <project>/data/models)
  VOICESHIELD_LOGS     → TB logs       (default: <project>/data/logs)
"""

import os
from pathlib import Path

# ── Project root (auto-detected from this file's location) ────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Paths (env-var overridable) ───────────────────────────────────
DATASET_ROOT = Path(os.environ.get("VOICESHIELD_DATASET",
                                    str(PROJECT_ROOT / "data" / "dataset")))
METADATA_CSV = DATASET_ROOT / "metadata_clean.csv"
MODEL_DIR = Path(os.environ.get("VOICESHIELD_MODELS",
                                 str(PROJECT_ROOT / "data" / "models")))
LOG_DIR = Path(os.environ.get("VOICESHIELD_LOGS",
                               str(PROJECT_ROOT / "data" / "logs")))

# ── Audio ─────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000
CLIP_DURATION_S = 4.0                       # seconds to crop/pad
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION_S)  # 64 000

# ── Mel Spectrogram ───────────────────────────────────────────────
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
FMIN = 20.0
FMAX = 8000.0

# ── SpecAugment ───────────────────────────────────────────────────
FREQ_MASK_PARAM = 12       # max width of frequency mask
TIME_MASK_PARAM = 40       # max width of time mask
NUM_FREQ_MASKS = 2
NUM_TIME_MASKS = 2

# ── Model ─────────────────────────────────────────────────────────
RESNET_FILTERS = [32, 64, 128]              # per residual stage
RESNET_BLOCKS_PER_STAGE = 2
DENSE_UNITS = 128
DROPOUT_RATE = 0.3

# ── Training ──────────────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS = 2
LEARNING_RATE = 1e-3
LR_PATIENCE = 5                             # ReduceLROnPlateau
EARLY_STOP_PATIENCE = 2
LABEL_SMOOTHING = 0.05

# ── Misc ──────────────────────────────────────────────────────────
SEED = 42
NUM_WORKERS = 4
