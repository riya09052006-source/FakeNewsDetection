from pathlib import Path

import pandas as pd

from config import (
    TRAIN_FILE,
    VALIDATION_FILE,
    TRAIN_SAMPLES,
    VALIDATION_SAMPLES,
    SEED,
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PREPARING TRANSFORMER DATASET")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "content",
    "label",
]

for column in required_columns:

    if column not in train_df.columns:
        raise ValueError(
            f"Missing column in training dataset: {column}"
        )

    if column not in validation_df.columns:
        raise ValueError(
            f"Missing column in validation dataset: {column}"
        )


# ============================================================
# CLEAN BASIC DATA
# ============================================================

train_df = train_df[
    ["content", "label"]
].dropna()

validation_df = validation_df[
    ["content", "label"]
].dropna()


# ============================================================
# SHUFFLE
# ============================================================

train_df = train_df.sample(
    frac=1,
    random_state=SEED,
).reset_index(drop=True)

validation_df = validation_df.sample(
    frac=1,
    random_state=SEED,
).reset_index(drop=True)


# ============================================================
# SAMPLE
# ============================================================

train_df = train_df.head(
    min(TRAIN_SAMPLES, len(train_df))
)

validation_df = validation_df.head(
    min(VALIDATION_SAMPLES, len(validation_df))
)


# ============================================================
# SAVE
# ============================================================

output_dir = (
    Path(TRAIN_FILE).parents[1] /
    "transformer"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


train_output = output_dir / "train_transformer.csv"

validation_output = (
    output_dir /
    "validation_transformer.csv"
)


train_df.to_csv(
    train_output,
    index=False,
)

validation_df.to_csv(
    validation_output,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print("\nTraining samples:", len(train_df))
print("Validation samples:", len(validation_df))

print("\nTraining class distribution:")

print(
    train_df["label"]
    .value_counts()
    .sort_index()
)

print("\nValidation class distribution:")

print(
    validation_df["label"]
    .value_counts()
    .sort_index()
)

print("\nSaved:")

print(train_output)
print(validation_output)

print("\nDataset preparation completed.")
