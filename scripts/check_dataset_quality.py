import pandas as pd
from pathlib import Path


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "news_dataset.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("DATASET QUALITY CHECK")
print("=" * 70)

if not DATASET_FILE.exists():
    print("\nERROR: Processed dataset not found.")
    print(DATASET_FILE)
    raise SystemExit(1)


df = pd.read_csv(DATASET_FILE)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

print(missing)


# ============================================================
# DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATES")
print("=" * 70)

duplicates = df.duplicated(
    subset=["content"]
).sum()

print("Duplicate content:", duplicates)


# ============================================================
# LABEL CHECK
# ============================================================

print("\n" + "=" * 70)
print("LABEL CHECK")
print("=" * 70)

print(df["label"].value_counts())

print("\nUnique labels:")
print(sorted(df["label"].unique()))


# ============================================================
# EMPTY CONTENT
# ============================================================

print("\n" + "=" * 70)
print("EMPTY CONTENT CHECK")
print("=" * 70)

empty_content = (
    df["content"]
    .fillna("")
    .str.strip()
    .eq("")
    .sum()
)

print("Empty articles:", empty_content)


# ============================================================
# CONTENT LENGTH
# ============================================================

df["content_length"] = (
    df["content"]
    .fillna("")
    .str.len()
)

print("\n" + "=" * 70)
print("CONTENT LENGTH")
print("=" * 70)

print(df["content_length"].describe())


# ============================================================
# CLASS BALANCE
# ============================================================

print("\n" + "=" * 70)
print("CLASS BALANCE")
print("=" * 70)

counts = df["label"].value_counts()

fake_count = counts.get(0, 0)
real_count = counts.get(1, 0)

print("Fake:", fake_count)
print("Real:", real_count)

if fake_count > 0 and real_count > 0:

    difference = abs(fake_count - real_count)

    percentage_difference = (
        difference /
        max(fake_count, real_count)
    ) * 100

    print(
        f"Difference: {percentage_difference:.2f}%"
    )


# ============================================================
# SUBJECT CHECK
# ============================================================

print("\n" + "=" * 70)
print("SUBJECT DISTRIBUTION")
print("=" * 70)

print(
    df["subject"]
    .fillna("Unknown")
    .value_counts()
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)

if (
    missing.sum() == 0
    and duplicates == 0
    and empty_content == 0
    and set(df["label"].unique()) == {0, 1}
):
    print("DATASET QUALITY CHECK: PASSED")
else:
    print("DATASET QUALITY CHECK: REVIEW REQUIRED")

print("=" * 70)
