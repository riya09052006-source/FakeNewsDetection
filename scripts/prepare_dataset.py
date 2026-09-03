import pandas as pd
import re
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FAKE_FILE = BASE_DIR / "data" / "raw" / "Fake.csv"
TRUE_FILE = BASE_DIR / "data" / "raw" / "True.csv"

OUTPUT_FILE = BASE_DIR / "data" / "processed" / "news_dataset.csv"


# ============================================================
# TEXT CLEANING FUNCTION
# ============================================================

def normalize_text(text):
    """
    Normalize article text without performing aggressive
    NLP preprocessing.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Replace multiple spaces/newlines
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FAKE NEWS DETECTION - DATA PREPARATION")
print("=" * 70)

print("\nLoading datasets...")

fake_df = pd.read_csv(FAKE_FILE)
true_df = pd.read_csv(TRUE_FILE)


# ============================================================
# ADD LABELS
# ============================================================

print("\nAdding labels...")

# 0 = Fake
# 1 = Real

fake_df["label"] = 0
true_df["label"] = 1


# ============================================================
# ADD SOURCE TYPE
# ============================================================

fake_df["source_type"] = "fake"
true_df["source_type"] = "real"


# ============================================================
# COMBINE TITLE + TEXT
# ============================================================

print("Combining title and article text...")

fake_df["title"] = fake_df["title"].fillna("")
fake_df["text"] = fake_df["text"].fillna("")

true_df["title"] = true_df["title"].fillna("")
true_df["text"] = true_df["text"].fillna("")


fake_df["content"] = (
    fake_df["title"] + " " + fake_df["text"]
)

true_df["content"] = (
    true_df["title"] + " " + true_df["text"]
)


# ============================================================
# SELECT IMPORTANT COLUMNS
# ============================================================

fake_df = fake_df[
    [
        "title",
        "text",
        "subject",
        "date",
        "content",
        "label",
        "source_type"
    ]
]

true_df = true_df[
    [
        "title",
        "text",
        "subject",
        "date",
        "content",
        "label",
        "source_type"
    ]
]


# ============================================================
# COMBINE DATASETS
# ============================================================

print("Combining fake and real news...")

df = pd.concat(
    [fake_df, true_df],
    ignore_index=True
)


print("\nTotal records before cleaning:")
print(len(df))


# ============================================================
# NORMALIZE TEXT
# ============================================================

print("\nNormalizing text...")

df["title"] = df["title"].apply(normalize_text)
df["text"] = df["text"].apply(normalize_text)
df["content"] = df["content"].apply(normalize_text)


# ============================================================
# REMOVE EMPTY CONTENT
# ============================================================

print("\nRemoving empty articles...")

before = len(df)

df = df[df["content"].str.len() > 0]

after = len(df)

print("Removed:", before - after)


# ============================================================
# REMOVE VERY SHORT ARTICLES
# ============================================================

print("\nRemoving extremely short articles...")

before = len(df)

df = df[df["content"].str.len() >= 50]

after = len(df)

print("Removed:", before - after)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

print("\nRemoving duplicate articles...")

before = len(df)

df = df.drop_duplicates(
    subset=["content"],
    keep="first"
)

after = len(df)

print("Duplicates removed:", before - after)


# ============================================================
# RESET INDEX
# ============================================================

df = df.reset_index(drop=True)


# ============================================================
# SHUFFLE DATASET
# ============================================================

print("\nShuffling dataset...")

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# DATASET SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET SUMMARY")
print("=" * 70)

print("\nTotal articles:")
print(len(df))

print("\nFake articles:")
print((df["label"] == 0).sum())

print("\nReal articles:")
print((df["label"] == 1).sum())

print("\nClass distribution:")
print(df["label"].value_counts())


# ============================================================
# SAVE DATASET
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nProcessed dataset saved to:")

print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("DATA PREPARATION COMPLETED SUCCESSFULLY")
print("=" * 70)
