"""
tune_threshold_plots.py

Purpose:
    - Visualise threshold-dependent model performance using precomputed metrics.
    - Support threshold selection by exposing precision–recall trade-offs.

Workflow:
    1. Load threshold_metrics.csv.
    2. Generate:
        a. Precision–Recall curve
        b. F1 vs Threshold
        c. Precision & Recall vs Threshold (combined)
    3. Save plots to disk.

Outputs:
    results/threshold_tuning/plots/
        ├── pr_curve.png
        ├── f1_vs_threshold.png
        └── precision_recall_vs_threshold.png

Notes:
    - This script does NOT perform threshold selection.
    - It is purely for analysis and interpretation.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------
# Paths
# ------------------------
INPUT_PATH = "results/threshold_tuning/threshold_metrics.csv"
OUTPUT_DIR = Path("results/threshold_tuning/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------
# Load Data
# ------------------------
df = pd.read_csv(INPUT_PATH).sort_values("threshold")

thresholds = df["threshold"].values
precision = df["precision"].values
recall = df["recall"].values
f1 = df["f1"].values

# ------------------------
# 1. Precision–Recall Curve
# ------------------------
plt.figure()

plt.plot(recall, precision)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")

plt.savefig(OUTPUT_DIR / "pr_curve.png", bbox_inches="tight")
plt.close()

# ------------------------
# 2. F1 vs Threshold
# ------------------------
plt.figure()

plt.plot(thresholds, f1, label="F1 Score")

# Highlight best F1 point
f1_idx = f1.argmax()
best_threshold = thresholds[f1_idx]
best_f1 = f1[f1_idx]

# Add vertical line at best threshold
plt.axvline(best_threshold, linestyle="--", color="red",
            label=f"Best threshold = {best_threshold:.3f}")

# Add horizontal line at best F1
plt.axhline(best_f1, linestyle=":",color="red", 
            label=f"Best F1 = {best_f1:.4f}")

# Mark the exact point
plt.scatter(best_threshold, best_f1, color="red")

plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("F1 vs Threshold")

plt.legend()

plt.savefig(OUTPUT_DIR / "f1_vs_threshold.png", bbox_inches="tight")
plt.close()

# ------------------------
# 3. Precision & Recall vs Threshold
# ------------------------
plt.figure()

plt.plot(thresholds, precision, label="Precision")
plt.plot(thresholds, recall, label="Recall")

plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision & Recall vs Threshold")
plt.legend()

plt.savefig(OUTPUT_DIR / "precision_recall_vs_threshold.png", bbox_inches="tight")
plt.close()

# ------------------------
# Done
# ------------------------
print("\n===== PLOTS GENERATED =====")
print(f"Saved to: {OUTPUT_DIR}")


