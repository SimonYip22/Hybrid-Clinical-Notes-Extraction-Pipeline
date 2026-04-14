"""
run_evaluation.py

Purpose:
    Generate a unified evaluation dataset containing:
    - Ground truth labels
    - Rule-based predictions
    - Transformer probabilities
    - Transformer predictions

Output:
    outputs/evaluation/pipeline_predictions.csv
"""

# -------------------------
# Imports
# -------------------------
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# -------------------------
# Config
# -------------------------

TEST_PATH = "data/extraction/new_splits/test.csv"
MODEL_DIR = "models/bioclinicalbert_final"

OUTPUT_PATH = Path("outputs/evaluation/pipeline_predictions.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 512
THRESHOLD = 0.549
BATCH_SIZE = 16

# Run model on GPU if available, otherwise fallback to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load Data
# -------------------------

# Load the test dataset into a DataFrame
df = pd.read_csv(TEST_PATH)

# Rename is_valid → y_true and 0/1 for clarity in evaluation
df["y_true"] = df["is_valid"].astype(int)

# -------------------------
# Rule-Based Predictions
# -------------------------

def compute_rule_pred(row):
    """
    Function to compute rule-based predictions based on entity type and negation status.

    Args:
        row: A row from the DataFrame containing columns
            - entity_type
            - negated
    Returns:
        A binary prediction (0 or 1) based on the following rules:
            - If entity_type is "SYMPTOM":
                - If negated is False → return 1 (valid)
                - If negated is True → return 0 (invalid)
            - For all other entity types, return 1 (assumed valid)  
    """
    if row["entity_type"] == "SYMPTOM":
        return 1 if not row["negated"] else 0  # negated=False → valid, negated=True → invalid
    else:
        return 1  # all other entities assumed valid

# Apply the function row-wise to compute rule-based predictions which are baseline for evaluation
df["rule_pred"] = df.apply(compute_rule_pred, axis=1)

# -------------------------
# Load Model + Tokenizer
# -------------------------

# Load tokenizer and its vocabulary / rules
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# Load model and its trained weights
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# Move model to the appropriate device (GPU or CPU)
model.to(DEVICE)

# Set model to evaluation mode to disable dropout and other training-specific randomness to ensure deterministic inference
model.eval()

# -------------------------
# Tokenisation Function
# -------------------------

def build_text(row):
    """
    Function to build the input text for the model by concatenating relevant information from the DataFrame row.

    Args:
        row: A row from the DataFrame containing columns
            - section
            - entity_type
            - entity_text
            - concept
            - task
            - sentence_text
    Returns:
        A single string concatenating the above information in a structured format.
    """
    return (
        f"[SECTION] {row['section']} "
        f"[ENTITY TYPE] {row['entity_type']} "
        f"[ENTITY] {row['entity_text']} "
        f"[CONCEPT] {row['concept']} "
        f"[TASK] {row['task']} "
        f"[TEXT] {row['sentence_text']}"
    )

# Apply the function to each row to build the input texts for the model
# Produces a list of strings per row, which will be tokenised and fed into the model for inference
texts = df.apply(build_text, axis=1).tolist()

# -------------------------
# Inference (Batched)
# -------------------------

# Store model outputs
probs = []

# Loop over the text indices in batches to avoid memory issues
for i in tqdm(range(0, len(texts), BATCH_SIZE)):

    # Slice to get the current batch of texts based on the batch size using the defined pointer i
    batch_texts = texts[i:i+BATCH_SIZE]

    # Tokenise the batch of texts
    inputs = tokenizer(
        batch_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt" # Return PyTorch tensors for model input
    )

    # Move inputs to the same device as the model
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Forward pass through the model (no gradient calculation needed)
    with torch.no_grad():
        # Get the logits from the model output
        outputs = model(**inputs)
        logits = outputs.logits

        # Softmax → convert to probabilities, then [:, 1] extracts probability of class 1 (valid)
        batch_probs = torch.softmax(logits, dim=1)[:, 1]

    # Move probabilities back to CPU, convert to numpy, then append to list
    probs.extend(batch_probs.cpu().numpy())

# Add the probabilities (probability of being 1 = valid) to the DataFrame
df["model_prob"] = probs

# -------------------------
# Thresholding
# -------------------------

# Apply the threshold to get binary predictions
df["model_pred"] = (df["model_prob"] >= THRESHOLD).astype(int)

# -------------------------
# Save Output
# -------------------------

# Select the columns to save in the output CSV
cols_to_save = [
    "entity_type",  # Entity type for analysis by type
    "y_true",       # Ground truth
    "rule_pred",    # Rule-based prediction
    "model_prob",   # Keep for PR curve, ROC, threshold tuning flexibility
    "model_pred",   # Transformer prediction (thresholded)
]

# Save the selected columns to a CSV file without the index
df[cols_to_save].to_csv(OUTPUT_PATH, index=False)

print("\n===== EVALUATION COMPLETE =====")
print(f"Saved to: {OUTPUT_PATH}")