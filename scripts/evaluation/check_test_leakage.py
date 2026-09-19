from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_FILE = (
    BASE_DIR /
    "data" /
    "processed" /
    "splits" /
    "train.csv"
)

VALIDATION_FILE = (
    BASE_DIR /
    "data" /
    "processed" /
    "splits" /
    "validation.csv"
)

TEST_FILE = (
    BASE_DIR /
    "data" /
    "processed" /
    "splits" /
    "test.csv"
)


# ============================================================
# LOAD
# ============================================================

train = pd.read_csv(
    TRAIN_FILE
)

validation = pd.read_csv(
    VALIDATION_FILE
)

test = pd.read_csv(
    TEST_FILE
)


# ============================================================
# CONTENT SETS
# ============================================================

train_content = set(
    train["content"]
    .dropna()
)

validation_content = set(
    validation["content"]
    .dropna()
)

test_content = set(
    test["content"]
    .dropna()
)


# ============================================================
# OVERLAPS
# ============================================================

train_validation = (
    train_content &
    validation_content
)

train_test = (
    train_content &
    test_content
)

validation_test = (
    validation_content &
    test_content
)


# ============================================================
# RESULTS
# ============================================================

print("=" * 70)
print("EXACT CONTENT LEAKAGE CHECK")
print("=" * 70)

print(
    f"\nTrain ∩ Validation: "
    f"{len(train_validation)}"
)

print(
    f"Train ∩ Test: "
    f"{len(train_test)}"
)

print(
    f"Validation ∩ Test: "
    f"{len(validation_test)}"
)


if (
    len(train_validation) == 0
    and
    len(train_test) == 0
    and
    len(validation_test) == 0
):

    print(
        "\nNo exact duplicate content was found "
        "between the splits."
    )

else:

    print(
        "\nWARNING: Exact duplicate content exists "
        "between one or more splits."
    )


print("\nLeakage check completed.")
