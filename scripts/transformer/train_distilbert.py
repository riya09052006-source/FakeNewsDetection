from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)

from config import (
    BASE_DIR,
    MODEL_NAME,
    OUTPUT_DIR,
    MAX_LENGTH,
    LEARNING_RATE,
    BATCH_SIZE,
    EPOCHS,
    WEIGHT_DECAY,
    SEED,
)


# ============================================================
# SETTINGS
# ============================================================

os.environ["TOKENIZERS_PARALLELISM"] = "false"

set_seed(SEED)

torch.set_num_threads(
    max(1, min(4, os.cpu_count() or 2))
)


# ============================================================
# PATHS
# ============================================================

TRANSFORMER_DATA_DIR = (
    BASE_DIR /
    "data" /
    "processed" /
    "transformer"
)

TRAIN_FILE = (
    TRANSFORMER_DATA_DIR /
    "train_transformer.csv"
)

VALIDATION_FILE = (
    TRANSFORMER_DATA_DIR /
    "validation_transformer.csv"
)


# ============================================================
# DISPLAY
# ============================================================

print("=" * 70)
print("DISTILBERT FAKE NEWS TRAINING")
print("=" * 70)

print("\nDevice:")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(device)

print("\nModel:", MODEL_NAME)
print("Max length:", MAX_LENGTH)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)


# ============================================================
# LOAD CSV
# ============================================================

train_df = pd.read_csv(
    TRAIN_FILE
)

validation_df = pd.read_csv(
    VALIDATION_FILE
)


# ============================================================
# DATASET
# ============================================================

train_dataset = Dataset.from_pandas(
    train_df[
        ["content", "label"]
    ],
    preserve_index=False,
)

validation_dataset = Dataset.from_pandas(
    validation_df[
        ["content", "label"]
    ],
    preserve_index=False,
)


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_function(examples):

    return tokenizer(
        examples["content"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


print("\nTokenizing training dataset...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["content"],
)

print("\nTokenizing validation dataset...")

validation_dataset = validation_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["content"],
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

model.to(device)


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_prediction):

    predictions, labels = eval_prediction

    predicted_labels = np.argmax(
        predictions,
        axis=1,
    )

    accuracy = (
        predicted_labels == labels
    ).mean()

    tp = np.sum(
        (predicted_labels == 1) &
        (labels == 1)
    )

    fp = np.sum(
        (predicted_labels == 1) &
        (labels == 0)
    )

    fn = np.sum(
        (predicted_labels == 0) &
        (labels == 1)
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    num_train_epochs=EPOCHS,

    weight_decay=WEIGHT_DECAY,

    eval_strategy="epoch",

    save_strategy="epoch",

    logging_strategy="steps",

    logging_steps=100,

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    report_to="none",

    fp16=False,

    dataloader_num_workers=0,

    seed=SEED,
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(
    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics,
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

trainer.train()


# ============================================================
# EVALUATE
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION EVALUATION")
print("=" * 70)

metrics = trainer.evaluate()

for key, value in metrics.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.4f}"
        )

    else:

        print(
            f"{key}: {value}"
        )


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving final model...")

final_output = (
    OUTPUT_DIR /
    "final"
)

final_output.mkdir(
    parents=True,
    exist_ok=True,
)

trainer.save_model(
    final_output
)

tokenizer.save_pretrained(
    final_output
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_output = (
    final_output /
    "validation_metrics.txt"
)

with open(
    metrics_output,
    "w",
    encoding="utf-8",
) as file:

    for key, value in metrics.items():

        file.write(
            f"{key}: {value}\n"
        )


print("\nModel saved to:")

print(final_output)

print("\nTraining completed successfully.")
