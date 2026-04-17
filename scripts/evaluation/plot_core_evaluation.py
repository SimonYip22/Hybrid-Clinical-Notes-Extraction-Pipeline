"""
plot_core_evaluation.py

Purpose:
    Generate core Layer 1 evaluation visualisations for pipeline-level comparison:
    - Confusion matrix heatmaps for:
        • Rule-based extraction (baseline)
        • Transformer validation (final system)
    - Bar plot comparing key performance metrics:
        • Precision, Recall, F1 score

    These visualisations support interpretation of:
        - Error types (via confusion matrices)
        - Performance trade-offs between systems (via metric comparison)

Workflow:
    1. Load precomputed core metrics (core_metrics.csv)
    2. Reconstruct confusion matrices from stored TN, FP, FN, TP values
    3. Generate annotated heatmaps for each system
    4. Plot side-by-side comparison of precision, recall, and F1
    5. Save all visual outputs to disk

Outputs:
    outputs/evaluation/core_plots/
        - rule_confusion_matrix.png
        - transformer_confusion_matrix.png
        - metrics_comparison.png

Notes:
    - Heatmaps include numeric annotations for exact interpretation
    - Colour intensity is not relied upon for analysis
    - Metrics comparison plot is used for direct system-level comparison
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# -------------------------
# Load Metrics
# -------------------------
df = pd.read_csv("outputs/evaluation/core_metrics.csv", index_col=0) # Set index to system names (Rule-Based, Transformer)

# Create output directory
out_dir = Path("outputs/evaluation/core_plots")
out_dir.mkdir(parents=True, exist_ok=True)

# -------------------------
# 1. Confusion Matrix Reconstruction
# -------------------------

def get_confusion_matrix(row):
    """
    Reconstruct a 2×2 confusion matrix from stored metric components.

    Args:
        row (pd.Series):
            A row from the metrics DataFrame corresponding to one system.
            Must contain:
                - true_negatives
                - false_positives
                - false_negatives
                - true_positives

    Returns:
        np.ndarray:
            2×2 confusion matrix in standard sklearn format:

                [[TN, FP],
                 [FN, TP]]

    Purpose:
        Converts flattened confusion matrix components stored in the CSV
        back into matrix form for visualisation.

    Notes:
        - Row corresponds to ground truth (True 0, True 1)
        - Column corresponds to predictions (Pred 0, Pred 1)
    """
    return np.array([
        [row["true_negatives"], row["false_positives"]], # TN, FP
        [row["false_negatives"], row["true_positives"]]  # FN, TP
    ])

# Use index to access rows for each system
rule_cm = get_confusion_matrix(df.loc["Rule-Based"])
model_cm = get_confusion_matrix(df.loc["Transformer"])

# -------------------------
# 2. Plot Heatmap Function
# -------------------------

def plot_confusion_matrix(cm, title, save_path):
    """
    Plot and save a confusion matrix heatmap with annotations.

    Args:
        cm (np.ndarray):
            2×2 confusion matrix:
                [[TN, FP],
                 [FN, TP]]

        title (str):
            Title for the plot

        save_path (Path or str):
            File path to save the image

    Behaviour:
        - Displays confusion matrix as a heatmap
        - Annotates each cell with the exact count
        - Labels axes for interpretability

    Notes:
        - Colour is used only as a visual aid
        - Numeric annotations are the primary source of interpretation
    """
    fig, ax = plt.subplots()

    im = ax.imshow(cm)

    # Labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])

    # Annotate cells
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    ax.set_title(title)

    plt.savefig(save_path)
    plt.close()

# -------------------------
# 3. Generate Heatmaps
# -------------------------

plot_confusion_matrix(
    rule_cm,
    "Rule-Based Confusion Matrix",
    out_dir / "rule_confusion_matrix.png"
)

plot_confusion_matrix(
    model_cm,
    "Transformer Confusion Matrix",
    out_dir / "transformer_confusion_matrix.png"
)

# -------------------------
# 4. Metrics Comparison Plot
# -------------------------

metrics = ["precision", "recall", "f1_score"]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots()

rule_vals = df.loc["Rule-Based", metrics].values        # Access values for Rule-Based row and selected metrics
model_vals = df.loc["Transformer", metrics].values      # Access values for Transformer row and selected metrics

ax.bar(x - width/2, rule_vals, width, label="Rule-Based")
ax.bar(x + width/2, model_vals, width, label="Transformer")

ax.set_xticks(x)
ax.set_xticklabels(["Precision", "Recall", "F1"])
ax.legend()

plt.savefig(out_dir / "metrics_comparison.png")
plt.close()