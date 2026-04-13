"""
select_threshold.py

Purpose:
    - Select optimal decision threshold using precomputed threshold metrics.
    - Apply precision-first decision policy with recall preservation constraint.
    - Produce a reproducible deployment-ready threshold.

Policy:
    1. Compute baseline recall at threshold = 0.5 (default operating point).
    2. Define minimum acceptable recall:
           recall_min = 0.85 × baseline_recall
    3. Select threshold that maximises precision subject to:
           recall ≥ recall_min
    4. If no threshold satisfies constraint, fall back to maximum F1.

Workflow:
    1. Load threshold_metrics.csv (precomputed OOF evaluation grid).
    2. Extract baseline recall, precision, and F1 at threshold = 0.5.
    3. Compute recall constraint (recall_min).
    4. Filter thresholds satisfying constraint.
    5. Select optimal threshold:
         - Primary: max precision under constraint
         - Secondary: max F1 (fallback)
    6. Save selected threshold and diagnostic metadata.
    7. Print selection summary for reproducibility and auditability.

Outputs:
    best_threshold.json
        - best_threshold
        - best_precision
        - best_recall
        - best_f1
        - baseline_f1
        - baseline_precision
        - baseline_recall
        - recall_min
        - method

Notes:
    - Uses out-of-fold (OOF) predictions only (no training leakage).
    - Threshold tuning is a post-hoc decision policy, not model training.
    - This script defines deployment decision behaviour for binary classification.
"""

# ------------------------
# Imports & Config
# ------------------------
import json
from pathlib import Path
import pandas as pd

INPUT_PATH = "results/threshold_tuning/threshold_metrics.csv"
OUTPUT_PATH = Path("results/threshold_tuning/best_threshold.json")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

# ------------------------
# Baseline (simple extraction)
# ------------------------

# Get recall at threshold = 0.5 (default operating point)
baseline_row = df[df["threshold"] == 0.5].iloc[0]
baseline_recall = baseline_row["recall"]
baseline_precision = baseline_row["precision"]
baseline_f1 = baseline_row["f1"]

# ------------------------
# Constraint
# ------------------------

# Define minimum acceptable recall (85% of baseline = accept 15% degradation)
recall_min = 0.85 * baseline_recall

# ------------------------
# Selection
# ------------------------

# Filter DataFrame to thresholds that meet recall constraint
valid = df[df["recall"] >= recall_min]

# Select the one with maximum precision
if len(valid) > 0:
    # If there are valid thresholds, select the row with highest precision
    best_row = valid.loc[valid["precision"].idxmax()]
    method = "precision-max under recall constraint"
else:
    # No thresholds meet the recall constraint → fallback to threshold with best F1
    best_row = df.loc[df["f1"].idxmax()]
    method = "f1 fallback"

# Extract best threshold
best_threshold = float(best_row["threshold"])

# ------------------------
# Save Outputs
# ------------------------

with open(OUTPUT_PATH, "w") as f:
    json.dump({
        "best_threshold": best_threshold,
        "best_precision": float(best_row["precision"]),
        "best_recall": float(best_row["recall"]),
        "best_f1": float(best_row["f1"]),
        "baseline_f1": float(baseline_f1),
        "baseline_precision": float(baseline_precision),
        "baseline_recall": float(baseline_recall),
        "recall_min": float(recall_min),
        "method": method
    }, f, indent=2)

# ------------------------
# Output
# ------------------------
print("\n===== THRESHOLD SELECTION =====\n")
print(f"Baseline_f1: {baseline_f1:.4f}")
print(f"Baseline_precision: {baseline_precision:.4f}")
print(f"Baseline_recall: {baseline_recall:.4f}")
print(f"recall_min: {recall_min:.4f}")
print(f"method: {method}")
print(f"best_threshold: {best_threshold:.4f}")
print("\nMetrics at selected threshold:")
print(best_row[["threshold", "precision", "recall", "f1"]].to_string())