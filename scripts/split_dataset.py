import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "news_dataset.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "splits"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("DATASET SPLITTING")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("\nTotal records:")
print(len(df))


# ============================================================
# FIRST SPLIT
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"]
)


# ============================================================
# SECOND SPLIT
# ============================================================

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)


# ============================================================
# RESULTS
# ============================================================

print("\nTraining records:")
print(len(train_df))

print("\nValidation records:")
print(len(validation_df))

print("\nTesting records:")
print(len(test_df))


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 70)

print(train_df["label"].value_counts())


print("\n" + "=" * 70)
print("VALIDATION CLASS DISTRIBUTION")
print("=" * 70)

print(validation_df["label"].value_counts())


print("\n" + "=" * 70)
print("TEST CLASS DISTRIBUTION")
print("=" * 70)

print(test_df["label"].value_counts())


# ============================================================
# SAVE
# ============================================================

train_df.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False
)

validation_df.to_csv(
    OUTPUT_DIR / "validation.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False
)


# ============================================================
# FINAL CHECK
# ============================================================

total = (
    len(train_df)
    + len(validation_df)
    + len(test_df)
)

print("\n" + "=" * 70)
print("SPLIT CHECK")
print("=" * 70)

print("Original:", len(df))
print("Split total:", total)

if total == len(df):
    print("\nSplit verification: PASSED")
else:
    print("\nSplit verification: FAILED")


print("\nFiles saved to:")
print(OUTPUT_DIR)

print("\n" + "=" * 70)
print("DATASET SPLITTING COMPLETED")
print("=" * 70)
