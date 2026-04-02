


import pandas as pd

# -------------------------
# 1. Config
# -------------------------

INPUT_FILE = "data/extraction/sampling/annotation_sample_labeled.csv"

TRAIN_OUTPUT_FILE = "data/extraction/splits/train.csv"
VAL_OUTPUT_FILE = "data/extraction/splits/val.csv"
TEST_OUTPUT_FILE = "data/extraction/splits/test.csv"

TRAIN = 420
VALIDATION = 90
TEST = 90

df = pd.read_csv(INPUT_FILE)

# -------------------------
# 2. Check Dataset
# -------------------------
print("=== BASIC INFO ===")
print(f"Total rows: {len(df)}")
print(df.columns)
print()

# Check missing values
print("=== MISSING VALUES ===")
print(df.isnull().sum())
print()

# Specifically check is_valid
missing_is_valid = df['is_valid'].isnull().sum()
print(f"Missing is_valid: {missing_is_valid}")
print()

# Check unique values in is_valid
print("=== UNIQUE is_valid VALUES ===")
print(df['is_valid'].unique())
print()

# Count True / False
print("=== is_valid DISTRIBUTION ===")
print(df['is_valid'].value_counts())
print()

# Check task distribution
print("=== TASK DISTRIBUTION ===")
print(df['task'].value_counts())
print()

# Cross-tab (VERY IMPORTANT)
print("=== TASK vs is_valid ===")
print(pd.crosstab(df['task'], df['is_valid']))
print()

# Check if each task has 200 rows
print("=== CHECK TASK SIZE (expect 200 each) ===")
task_counts = df['task'].value_counts()

for task, count in task_counts.items():
    print(f"{task}: {count}")

print()

# Check for invalid label values
valid_labels = {True, False}

invalid_labels = df[~df['is_valid'].isin(valid_labels)]

print("=== INVALID LABEL ROWS ===")
print(f"Number of invalid label rows: {len(invalid_labels)}")

if len(invalid_labels) > 0:
    print(invalid_labels.head())