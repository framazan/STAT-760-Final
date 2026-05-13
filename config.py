"""
config.py — Centralized hyperparameters and constants for the wildfire
detection project.  Every script imports from here so there is a single
source of truth.
"""

import os

# ── paths ────────────────────────────────────────────────────────────────
DATA_ROOT = os.path.join(".", "dataset", "combined_smoke_fire_classification_256_balanced")
SAVE_DIR  = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── image / normalization ────────────────────────────────────────────────
IMAGE_SIZE   = 224
MEAN         = [0.485, 0.456, 0.406]   # ImageNet
STD          = [0.229, 0.224, 0.225]

# ── training defaults ────────────────────────────────────────────────────
SEED         = 42
EPOCHS       = 50
BATCH_SIZE   = 32
LR           = 1e-3
WEIGHT_DECAY = 1e-2
DROPOUT      = 0.3
NUM_WORKERS  = 4
LABEL_SMOOTH = 0.1
GRAD_CLIP    = 1.0
PATIENCE     = 8           # early-stopping patience


# ── mixup / cutmix (training) ───────────────────────────────────────────
ENABLE_MIXUP = True
MIXUP_ALPHA  = 0.2
CUTMIX_ALPHA = 1.0


# ── stochastic depth (FireNet) ──────────────────────────────────────────
# Maximum drop probability applied to the deepest residual blocks.
STOCHASTIC_DEPTH = 0.15


# ── transfer-learning ────────────────────────────────────────────────────
TL_LR_HEAD       = 1e-3
TL_LR_BACKBONE   = 1e-5
TL_FREEZE_EPOCHS = 5
TL_EPOCHS        = 30

# ── class names (alphabetical, matching ImageFolder) ─────────────────────
CLASSES = ("negative", "positive")
NUM_CLASSES = len(CLASSES)
