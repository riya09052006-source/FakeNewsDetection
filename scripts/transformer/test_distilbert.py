from pathlib import Path

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from config import BASE_DIR


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = (
    BASE_DIR /
    "models" /
    "transformer" /
    "distilbert_fake_news" /
    "final"
)


# ============================================================
# DEVICE
# ============================================================

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("DISTILBERT FAKE NEWS PREDICTION")
print("=" * 60)

print("\nLoading model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(device)

model.eval()


# ============================================================
# SAMPLE ARTICLE
# ============================================================

news_text = """
Scientists have announced a new research breakthrough
after conducting several controlled experiments.
Researchers say the results require additional independent
verification before being applied commercially.
"""


# ============================================================
# TOKENIZE
# ============================================================

inputs = tokenizer(
    news_text,
    return_tensors="pt",
    truncation=True,
    max_length=256,
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    outputs = model(
        **inputs
    )

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1,
    )

    prediction = torch.argmax(
        probabilities,
        dim=-1,
    ).item()


# ============================================================
# RESULT
# ============================================================

fake_probability = (
    probabilities[0][0]
    .item()
)

real_probability = (
    probabilities[0][1]
    .item()
)


print("\nPrediction:")

if prediction == 0:

    print("FAKE")

else:

    print("REAL")


print(
    f"\nFake probability: "
    f"{fake_probability:.4f}"
)

print(
    f"Real probability: "
    f"{real_probability:.4f}"
)

print("\nDevice:", device)

print("\nNote:")
print(
    "This model prediction is not a guaranteed determination "
    "of factual truth."
)
