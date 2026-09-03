import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FAKE_FILE = BASE_DIR / "data" / "raw" / "Fake.csv"
TRUE_FILE = BASE_DIR / "data" / "raw" / "True.csv"


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 70)
print("FAKE NEWS DETECTION - DATASET INSPECTION")
print("=" * 70)

if not FAKE_FILE.exists():
    print("\nERROR: Fake.csv not found!")
    print("Expected location:")
    print(FAKE_FILE)
    raise SystemExit(1)

if not TRUE_FILE.exists():
    print("\nERROR: True.csv not found!")
    print("Expected location:")
    print(TRUE_FILE)
    raise SystemExit(1)


print("\nDataset files found successfully.")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading Fake.csv...")
fake_df = pd.read_csv(FAKE_FILE)

print("Loading True.csv...")
true_df = pd.read_csv(TRUE_FILE)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET SIZE")
print("=" * 70)

print("\nFake news articles:")
print(len(fake_df))

print("\nReal news articles:")
print(len(true_df))

print("\nTotal articles:")
print(len(fake_df) + len(true_df))


# ============================================================
# COLUMN INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("FAKE NEWS COLUMNS")
print("=" * 70)

print(fake_df.columns.tolist())


print("\n" + "=" * 70)
print("REAL NEWS COLUMNS")
print("=" * 70)

print(true_df.columns.tolist())


# ============================================================
# DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("FAKE NEWS DATA TYPES")
print("=" * 70)

print(fake_df.dtypes)


print("\n" + "=" * 70)
print("REAL NEWS DATA TYPES")
print("=" * 70)

print(true_df.dtypes)


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES - FAKE NEWS")
print("=" * 70)

print(fake_df.isnull().sum())


print("\n" + "=" * 70)
print("MISSING VALUES - REAL NEWS")
print("=" * 70)

print(true_df.isnull().sum())


# ============================================================
# DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATES")
print("=" * 70)

print("\nDuplicate fake articles:")
print(fake_df.duplicated().sum())

print("\nDuplicate real articles:")
print(true_df.duplicated().sum())


# ============================================================
# SAMPLE DATA
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE FAKE NEWS")
print("=" * 70)

print(
    fake_df[
        ["title", "text", "subject", "date"]
    ].head(3).to_string()
)


print("\n" + "=" * 70)
print("SAMPLE REAL NEWS")
print("=" * 70)

print(
    true_df[
        ["title", "text", "subject", "date"]
    ].head(3).to_string()
)


# ============================================================
# TEXT LENGTH ANALYSIS
# ============================================================

fake_df["text_length"] = fake_df["text"].fillna("").astype(str).str.len()
true_df["text_length"] = true_df["text"].fillna("").astype(str).str.len()


print("\n" + "=" * 70)
print("TEXT LENGTH STATISTICS")
print("=" * 70)

print("\nFake news:")
print(fake_df["text_length"].describe())

print("\nReal news:")
print(true_df["text_length"].describe())


# ============================================================
# SUBJECT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("FAKE NEWS SUBJECT DISTRIBUTION")
print("=" * 70)

print(fake_df["subject"].value_counts())


print("\n" + "=" * 70)
print("REAL NEWS SUBJECT DISTRIBUTION")
print("=" * 70)

print(true_df["subject"].value_counts())


# ============================================================
# DATE INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATE INFORMATION")
print("=" * 70)

print("\nFake news dates:")
print(fake_df["date"].head())

print("\nReal news dates:")
print(true_df["date"].head())


print("\n" + "=" * 70)
print("DATASET INSPECTION COMPLETED")
print("=" * 70)
