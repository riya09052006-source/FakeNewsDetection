from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]


# ============================================================
# DATA
# ============================================================

DATA_DIR = BASE_DIR / "data" / "processed" / "splits"

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

OUTPUT_DIR = (
    BASE_DIR /
    "models" /
    "transformer" /
    "distilbert_fake_news"
)


# ============================================================
# CPU-FRIENDLY SETTINGS
# ============================================================

MAX_LENGTH = 256

TRAIN_SAMPLES = 5000
VALIDATION_SAMPLES = 1000

BATCH_SIZE = 4

EPOCHS = 1

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

SEED = 42


# ============================================================
# DISPLAY
# ============================================================

print("=" * 60)
print("DISTILBERT CONFIGURATION")
print("=" * 60)

print("Model:", MODEL_NAME)
print("Max sequence length:", MAX_LENGTH)
print("Training samples:", TRAIN_SAMPLES)
print("Validation samples:", VALIDATION_SAMPLES)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Learning rate:", LEARNING_RATE)
