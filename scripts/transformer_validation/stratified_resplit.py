"""
stratified_resplit.py

Purpose:
    - Split the fully validated 1200 row annotated dataset into train and eval sets for 
      transformer training.
    - Ensure balanced representation across both:
        - task (symptom_presence, intervention_performed, clinical_condition_active)
        - label (is_valid: True/False)
    - Prevent data leakage and preserve statistical integrity.

Workflow:
    1. Load validated annotated dataset (1200 rows).
    2. Create stratification key combining:
        - task + is_valid
    3. Perform stratified split:
        - Train (85%) vs Eval (15%)
    4. Verify split sizes and distributions.
    5. Save splits to CSV files.

Outputs: data/extraction/splits/
    - train.csv (1020 rows)
    - eval.csv (180 rows)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# -------------------------
# 1. Config
# -------------------------

ORIGINAL_INPUT_FILE = "data/extraction/sampling/annotation_sample_labeled.csv"
NEW_INPUT_FILE = "data/extraction/sampling/additional_annotation_sample_labeled.csv"

TRAIN_OUTPUT_FILE = "data/extraction/new_splits/train.csv"
EVAL_OUTPUT_FILE = "data/extraction/new_splits/eval.csv"

RANDOM_STATE = 42

# Ensure output directory exists
Path("data/extraction/new_splits").mkdir(parents=True, exist_ok=True)

# -------------------------
# 2. Load Data
# -------------------------

first_df = pd.read_csv(ORIGINAL_INPUT_FILE)
second_df = pd.read_csv(NEW_INPUT_FILE)

# Combine datasets and reset index to ensure clean splits (no duplicate indices)
final_df = pd.concat([first_df, second_df], ignore_index=True)

print(f"Loaded first dataset with {len(first_df)} rows")
print(f"Loaded second dataset with {len(second_df)} rows")
print(f"Combined dataset has {len(final_df)} rows")

# -------------------------
# 3. Create Stratification Key
# -------------------------

# Combine task + is_valid to create a stratification key
# This ensures both task balance AND label balance are preserved
final_df["stratify_key"] = final_df["task"].astype(str) + "_" + final_df["is_valid"].astype(str)

# -------------------------
# 4. Split (Train vs Eval)
# -------------------------

train_df, eval_df = train_test_split(
    final_df,
    test_size=0.15,  # 15% eval = 180 rows
    stratify=final_df["stratify_key"],
    random_state=RANDOM_STATE
)

# -------------------------
# 5. Drop helper column
# -------------------------

for split in [train_df, eval_df]:
    split.drop(columns=["stratify_key"], inplace=True) # inplace=True modifies the DataFrame directly (no copy created)

# -------------------------
# 7. Reset indices (clean datasets)
# -------------------------

train_df = train_df.reset_index(drop=True)
eval_df = eval_df.reset_index(drop=True)

# -------------------------
# 8. Verification
# -------------------------

# Verify sizes
print("\n=== SPLIT SIZES ===")
print(f"Train: {len(train_df)}")
print(f"Eval: {len(eval_df)}")

# Verify distributions
def check_distribution(name, data):
    print(f"\n=== {name.upper()} DISTRIBUTION ===")
    print("\nTask distribution:")
    print(data["task"].value_counts())

    print("\nis_valid distribution:")
    print(data["is_valid"].value_counts())

    print("\nTask vs is_valid:")
    print(pd.crosstab(data["task"], data["is_valid"])) # cross-tab to show distribution of is_valid within each task

check_distribution("Train", train_df)
check_distribution("Eval", eval_df)

# -------------------------
# 9. Save Outputs
# -------------------------

train_df.to_csv(TRAIN_OUTPUT_FILE, index=False) 
eval_df.to_csv(EVAL_OUTPUT_FILE, index=False)

print("\nSplits saved successfully.")

