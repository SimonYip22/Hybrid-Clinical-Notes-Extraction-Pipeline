"""
validate_additional_manual_annotations.py

Purpose:
    - Validate the integrity and consistency of additional manually annotated 
      data prior to dataset splitting and model training.
    - Ensure that annotations are complete, correctly formatted, and
      meet expected structural and distributional requirements.
    - Detect errors early to prevent propagation into downstream tasks.

Workflow:
    1. Load annotated dataset from CSV.
    2. Perform structural checks:
        - Row count
        - Column presence
        - Missing values
    3. Validate `is_valid` labels:
        - No missing values
        - Only True/False allowed
    4. Verify task distribution:
        - Exactly 200 samples per task
    5. Analyse label distribution:
        - Overall balance
        - Task vs label breakdown
    6. Check critical fields:
        - No missing `task`
        - No missing `sentence_text`
    7. Output diagnostics for review before splitting.
"""

import pandas as pd

# -------------------------
# Load Data
# -------------------------
df = pd.read_csv("data/extraction/sampling/additional_annotation_sample_labeled.csv")

# -------------------------
# 1. Structural Checks
# -------------------------
print("=== BASIC INFO ===")
print(f"Total rows: {len(df)}")
print(df.columns)
print()

# Enforce expected dataset size
EXPECTED_ROWS = 600
if len(df) != EXPECTED_ROWS:
    print(f"WARNING: Expected {EXPECTED_ROWS} rows, found {len(df)}")
print()

# -------------------------
# 2. Missing Values
# -------------------------
print("=== MISSING VALUES ===")
print(df.isnull().sum())
print()

# Critical fields check
print("=== CRITICAL FIELD CHECKS ===")
missing_task = df['task'].isnull().sum()
missing_sentence = df['sentence_text'].isnull().sum()
missing_is_valid = df['is_valid'].isnull().sum()

print(f"Missing task: {missing_task}")
print(f"Missing sentence_text: {missing_sentence}")
print(f"Missing is_valid: {missing_is_valid}")
print()

# -------------------------
# 3. Label Validation
# -------------------------
print("=== UNIQUE is_valid VALUES ===")
print(df['is_valid'].unique())
print()

print("=== is_valid DISTRIBUTION ===")
print(df['is_valid'].value_counts())
print()

# Check for invalid label values
valid_labels = {True, False}
invalid_labels = df[~df['is_valid'].isin(valid_labels)]

print("=== INVALID LABEL ROWS ===")
print(f"Number of invalid label rows: {len(invalid_labels)}")

if len(invalid_labels) > 0:
    print(invalid_labels.head())
print()

# -------------------------
# 4. Task Distribution
# -------------------------
print("=== TASK DISTRIBUTION ===")
print(df['task'].value_counts())
print()

# Enforce expected task counts
EXPECTED_PER_TASK = 200
print("=== CHECK TASK SIZE (expect 200 each) ===")

task_counts = df['task'].value_counts()

for task, count in task_counts.items():
    print(f"{task}: {count}")
    if count != EXPECTED_PER_TASK:
        print(f"WARNING: {task} has {count} samples (expected {EXPECTED_PER_TASK})")

print()

# -------------------------
# 5. Task vs Label Distribution
# -------------------------
print("=== TASK vs is_valid ===")
print(pd.crosstab(df['task'], df['is_valid']))
print()

# -------------------------
# Final Status
# -------------------------
print("=== VALIDATION COMPLETE ===")
print("Review warnings above before proceeding to stratified split.")