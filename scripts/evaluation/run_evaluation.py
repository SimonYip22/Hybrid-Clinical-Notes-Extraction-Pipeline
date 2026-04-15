"""
run_evaluation.py

Purpose:
    Generate a unified evaluation dataset for pipeline-level analysis by combining:
    - Ground truth labels (y_true)
    - Rule-based predictions (rule_pred)
    - Transformer outputs (model_prob, model_pred)
    This dataset serves as the single source for all Phase 4 evaluation:
    - Metric computation (precision, recall, F1)
    - Confusion matrix analysis
    - Rule vs transformer comparison
    - Threshold-dependent analysis (PR/ROC curves)

Workflow:
    1. Load test dataset and define ground truth labels.
    2. Compute rule-based predictions using deterministic extraction logic.
    3. Load trained transformer model and tokenizer.
    4. Reconstruct structured input text used during training.
    5. Tokenise inputs and run batched inference.
    6. Extract class probabilities (p(y = 1)).
    7. Apply fixed threshold to obtain binary predictions.
    8. Save consolidated evaluation dataset.

Output:
    outputs/evaluation/pipeline_predictions.csv

    Columns:
        - entity_type : entity category (for stratified analysis)
        - y_true      : ground truth label (0/1)
        - rule_pred   : rule-based prediction (0/1)
        - model_prob  : transformer probability for class 1
        - model_pred  : transformer prediction after thresholding

Notes:
    - Inference is deterministic (model.eval(), no gradient computation).
    - Tokenisation must match training to ensure input consistency.
    - model_prob is retained for threshold analysis and curve-based evaluation.
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
    Compute rule-based prediction for a single entity.

    Logic:
        - SYMPTOM:
            negated = False → valid (1)
            negated = True  → invalid (0)
        - All other entity types:
            assumed valid (1)

    Args:
        row (pd.Series): Contains 'entity_type' and 'negated'.

    Returns:
        int: Binary prediction (0 or 1).
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
    Construct model input text from structured fields.

    This replicates the exact input format used during training,
    ensuring consistency between training and inference.

    Args:
        row (pd.Series): Contains structured input fields:
            - section
            - entity_type
            - entity_text
            - concept
            - task
            - sentence_text

    Returns:
        str: Concatenated input string for tokenisation.
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

# Iterate over dataset in fixed-size batches for efficient inference
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

        # Convert logits → probabilities; select class 1 (valid) probability
        batch_probs = torch.softmax(logits, dim=1)[:, 1] # dim

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