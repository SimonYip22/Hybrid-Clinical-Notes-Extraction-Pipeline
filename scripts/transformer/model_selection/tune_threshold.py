"""
tune_threshold.py

Purpose:
    - Compute classification performance metrics across a range of probability
      thresholds using out-of-fold (OOF) predictions.
    - Provide a complete, unbiased view of the precision–recall trade-off for
      downstream decision-making.
    - This script does NOT select a threshold; it only generates the metric
      landscape required for informed selection.

Workflow:
    1. Load OOF predictions (y_true, y_prob).
    2. Define a threshold grid (0.00 → 1.00).
    3. For each threshold:
        - Convert probabilities → binary predictions
        - Compute precision, recall, and F1-score
    4. Aggregate all results into a metrics table.
    5. Save the full table to disk.
    6. Print summary statistics for inspection.

Outputs:
    results/threshold_tuning/
        └── threshold_metrics.csv

Columns:
    - threshold: decision threshold applied to probabilities
    - precision: TP / (TP + FP)
    - recall: TP / (TP + FN)
    - f1: harmonic mean of precision and recall

Notes:
    - All metrics are computed on OOF predictions (fully out-of-sample).
    - No threshold selection or optimisation is performed here.
    - This output is used later for:
        - Precision–recall analysis
        - Threshold selection (separate step)
        - Visualisation (PR curve, metric curves)
"""

# ------------------------
# Imports & Config
# ------------------------
import pathlib as path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

INPUT_PATH = "results/threshold_tuning/oof_predictions.csv"
OUTPUT_METRICS_PATH = path.Path("results/threshold_tuning/threshold_metrics.csv")

OUTPUT_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

# ------------------------
# Load Data
# ------------------------
df = pd.read_csv(INPUT_PATH)

y_true = df["y_true"].values
y_prob = df["y_prob"].values

# ------------------------
# Threshold Sweep
# ------------------------

# Higher resolution is useful for smooth curves
thresholds = np.linspace(0, 1, 1001)

rows = []

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    rows.append({
        "threshold": t,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

metrics_df = pd.DataFrame(rows)

# ------------------------
# Save Outputs
# ------------------------
metrics_df.to_csv(OUTPUT_METRICS_PATH, index=False)

# ------------------------
# Terminal Output (Sanity)
# ------------------------

print("\n===== THRESHOLD METRICS GENERATED =====\n")

print(f"Total thresholds evaluated: {len(metrics_df)}")

# Show metric ranges (min/max) for quick sanity check
print("\nMetric ranges:")
print(f"Precision: min={metrics_df['precision'].min():.4f}, max={metrics_df['precision'].max():.4f}")
print(f"Recall:    min={metrics_df['recall'].min():.4f}, max={metrics_df['recall'].max():.4f}")
print(f"F1:        min={metrics_df['f1'].min():.4f}, max={metrics_df['f1'].max():.4f}")

# Show top 5 thresholds by F1 (for quick sanity check)
print("\nTop 5 thresholds by F1:")
print(metrics_df.sort_values("f1", ascending=False).head(5).to_string(index=False))

print("\nSaved file:")
print(f"- {OUTPUT_METRICS_PATH}")