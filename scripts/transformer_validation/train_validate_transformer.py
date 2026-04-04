"""
train_validate_transformer.py

Purpose:
    - Fine-tune a pretrained clinical transformer (BioClinicalBERT) on the
      annotated dataset for binary classification (`is_valid`).
    - Perform validation during training to monitor performance and select
      the best model checkpoint.
    - Output a trained model ready for downstream evaluation (Phase 4).

Workflow:


Outputs:
    - models/bioclinicalbert/
        - Fine-tuned model weights
        - Tokenizer
        - Training logs
"""

import pandas as pd
import numpy as np
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# -------------------------
# 1. Config
# -------------------------

TRAIN_PATH = "data/extraction/splits/train.csv"
VAL_PATH = "data/extraction/splits/val.csv"

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
OUTPUT_DIR = "models/bioclinicalbert"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

# -------------------------
# 2. Load Data
# -------------------------

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(val_df)}")

# Convert boolean → int labels (required for model training)
train_df["label"] = train_df["is_valid"].astype(int)
val_df["label"] = val_df["is_valid"].astype(int)

# -------------------------
# 3. Create HF Datasets
# -------------------------

# Convert pandas DataFrames to Hugging Face Dataset for training with the Transformers library
train_dataset = Dataset.from_pandas(train_df[["sentence_text", "label"]])
val_dataset = Dataset.from_pandas(val_df[["sentence_text", "label"]])

# -------------------------
# 4. Tokenization
# -------------------------

# Load the tokenizer for the pretrained model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenization function applied to every example in the dataset
def tokenize(sentence):
    return tokenizer(
        sentence["sentence_text"],
        truncation=True, # Truncate sentences longer than MAX_LENGTH tokens
        padding="max_length", # Pad sentences shorter than MAX_LENGTH tokens
        max_length=MAX_LENGTH # Fix the input length to MAX_LENGTH for consistent input size to the model
    )

# Apply tokenization to the entire dataset. `batched=True` allows processing multiple examples at once for efficiency.
train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

# Convert datasets into Pytorch tensors to allow model training.
# We specify the columns we need for training: `input_ids` (token IDs), `attention_mask` (mask to ignore padding tokens), and `label` (the target variable).
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# -------------------------
# 5. Load Model
# -------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2 # Binary classification heads (valid vs invalid)
)

# -------------------------
# 6. Metrics (Validation Only)
# -------------------------

# Define a function to compute evaluation metrics during validation
def compute_metrics(eval_pred):
    # logits = raw model outputs, labels = ground truth labels
    logits, labels = eval_pred # `eval_pred` is a tuple of (logits, labels) returned by the model during evaluation
    preds = np.argmax(logits, axis=1) # Converts raw logits to predicted class labels (0 or 1) by taking the index of the highest logit value

    # Compute precision, recall, F1-score using sklearn's utility function.
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary" # `average="binary"` is used for binary classification to compute metrics for the positive class (label=1).
    )

    acc = accuracy_score(labels, preds) # Compute accuracy as the proportion of correct predictions (both true positives and true negatives) out of all predictions.

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# -------------------------
# 7. Training Config
# -------------------------

# Define training arguments for the Trainer
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch", # Validate every epoch
    save_strategy="epoch", # Save model checkpoint every epoch
    learning_rate=LEARNING_RATE, 
    per_device_train_batch_size=BATCH_SIZE, # Batch size for training (number of examples processed together in one forward/backward pass)
    per_device_eval_batch_size=BATCH_SIZE, # Batch size for evaluation
    num_train_epochs=EPOCHS,
    weight_decay=0.01, # L2 regularization to prevent overfitting by penalizing large weights
    load_best_model_at_end=True, # After training, load the model checkpoint that performed best on the validation set according to the specified metric
    metric_for_best_model="f1", # Use F1-score to determine the best model checkpoint during training (since we care about both precision and recall in this binary classification task)
    logging_dir=f"{OUTPUT_DIR}/logs", # Directory to save training logs for visualization
    logging_steps=10, # Log training metrics every 10 steps
    save_total_limit=2 # Limit the total number of saved checkpoints to 2 to save disk space (older checkpoints will be deleted)
)

# -------------------------
# 8. Trainer
# -------------------------

# Trainer is a high-level API provided by Hugging Face Transformers that abstracts away the training loop and evaluation logic.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# -------------------------
# 9. Train Model
# -------------------------

# Runs epochs, updates weights, evaluates each epoch, logs metrics
trainer.train()

# -------------------------
# 10. Save Model
# -------------------------

trainer.save_model(OUTPUT_DIR) # Saves the fine-tuned model weights and configuration to the specified output directory
tokenizer.save_pretrained(OUTPUT_DIR) # Saves the tokenizer configuration and vocabulary to the same directory so it can be loaded later for inference

print("Training complete. Model saved.")