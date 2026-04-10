"""
sample_additional_entities.py

Purpose:
    - Generate an additional, non-overlapping annotation dataset of clinical entity extractions.
    - Extend the existing annotated dataset (e.g. 600 samples → 1200 total) without duplicating previously sampled rows.
    - Enforce balanced sampling across entity types while ensuring all new samples are unique relative to prior annotations.

Workflow:
    1. Load extraction candidates from JSONL file (1 line = 1 entity).
    2. Flatten nested JSON structure and extract relevant fields:
        - note_id, section, concept, entity_text, entity_type,
          sentence_text, negated, task, confidence
    3. Convert records into a pandas DataFrame.

    4. Load previously sampled dataset:
        - annotation_sample_raw.csv (existing 600 samples)

    5. Remove previously sampled rows from the full dataset:
        - Define deduplication columns:
            [sentence_text, entity_text, entity_type, concept, task]
        - Convert both datasets into comparable tuple representations
        - Filter out any rows in the full dataset that match existing samples
        - Result: df_filtered contains ONLY new, unseen samples

    6. Perform balanced sampling on filtered dataset:
        - Sample N_PER_CLASS (200) entities for each:
          SYMPTOM, INTERVENTION, CLINICAL_CONDITION
        - Ensures class balance is preserved in the additional dataset

    7. Concatenate sampled subsets into a single dataset.

    8. Shuffle dataset:
        - Mix entity types to reduce annotation ordering bias

    9. Add empty `is_valid` column:
        - Placeholder for manual ground truth annotation

Outputs:
    - additional_annotation_sample_raw.csv:
        - New, non-overlapping sample (e.g. next 600 entities)
        - Guaranteed not to duplicate rows from the original sample

    - additional_annotation_sample_labeled.csv:
        - Copy of the new sample for manual annotation
        - Created only if it does not already exist

Notes:
    - This script extends the original sampling process by introducing a pre-filtering step that 
      removes previously sampled rows before stratified sampling.
    - Deduplication is performed using a multi-column match:
        sentence_text + entity_text + entity_type + concept + task
    - Tuple-based comparison is used to enable row-level equality checking across multiple columns:
        - `df_tuples`: a pandas Series where each row is represented as a tuple of deduplication fields
          used to generate the boolean filtering mask
        - `existing_tuples`: a set used as a lookup structure for efficient membership testing
    - Sampling logic (class balance, random seed, shuffle) is identical to the original script,
      ensuring consistency across datasets.
"""

import json
import pandas as pd
from pathlib import Path

# -------------------------
# Paths
# -------------------------
INPUT_PATH = Path("data/interim/extraction_candidates.jsonl")
SAMPLE_INPUT_PATH = Path("data/extraction/sampling/annotation_sample_raw.csv")

SAMPLE_OUTPUT_PATH = Path("data/extraction/sampling/additional_annotation_sample_raw.csv")
ANNOTATED_OUTPUT_PATH = Path("data/extraction/sampling/additional_annotation_sample_labeled.csv")

# Ensure output directory exists
SAMPLE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
ANNOTATED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# -------------------------
# Load JSONL
# -------------------------

# List to hold all records
records = []

# Read JSONL file line by line (1 line = 1 entity)
with open(INPUT_PATH, "r") as f:
    for line in f:

        # Converts JSON string to Python dict
        row = json.loads(line)

        # Extract relevant fields and append to records list
        records.append({
            "note_id": row.get("note_id"),
            "section": row.get("section"),
            "concept": row.get("concept"),
            "entity_text": row.get("entity_text"),
            "entity_type": row.get("entity_type"),
            "sentence_text": row.get("sentence_text"),
            "negated": row.get("negated"),
            "task": row.get("validation", {}).get("task"), # If validation exists, extract task, else None
            "confidence": row.get("validation", {}).get("confidence", 0.0) # Optional: include confidence score if available, default to 0.0
        })

# Convert list of records to DataFrame
df = pd.DataFrame(records)

print(f"Loaded {len(df)} total entities")

# -------------------------
# Filter existing annotations (NEW)
# -------------------------

existing_samples = pd.read_csv(SAMPLE_INPUT_PATH)

# Define columns to check for duplicates
dedup_cols = [
    "sentence_text",
    "entity_text",
    "entity_type",
    "concept",
    "task"
]

# Create a series of tuples for all samples based on the deduplication columns (one tuple per row)
df_tuples = df[dedup_cols].apply(tuple, axis=1)

# Create a set of tuples for existing 600 samples based on the same deduplication columns
# Only existing_tuples is a set because it is used for lookup, so a set is faster, while df_tuples is a Series that we will filter against
existing_tuples = set(
    existing_samples[dedup_cols].apply(tuple, axis=1)
)

# Keep rows in df_tuples that are not (~) present in existing_tuples
df_filtered = df[~df_tuples.isin(existing_tuples)]

print("Before:", len(df))
print("After:", len(df_filtered))

# -------------------------
# Stratified sampling
# -------------------------
N_PER_CLASS = 200

# Sampled DataFrames for each class
sampled_dfs = []

# Sample for each entity type
for entity_type in ["SYMPTOM", "INTERVENTION", "CLINICAL_CONDITION"]:

    # Filter for current entity type
    subset = df_filtered[df_filtered["entity_type"] == entity_type]

    # Check if we have enough samples
    if len(subset) < N_PER_CLASS:
        raise ValueError(f"Not enough samples for {entity_type}")

    # Sample N_PER_CLASS from the subset
    sampled = subset.sample(n=N_PER_CLASS, random_state=42)
    # Append to list
    sampled_dfs.append(sampled)

# Combine sampled DataFrames and reset index
sample_df = pd.concat(sampled_dfs).reset_index(drop=True)

# Shuffle final dataset to mix entity types (optional but often desirable for annotation)
sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)

# -------------------------
# Add annotation column
# -------------------------

# Add empty column for manual annotation labels (ground truth)
sample_df["is_valid"] = None

print(f"Final sample size: {len(sample_df)}")

# -------------------------
# Save
# -------------------------

# Save raw sample (always overwrite)
sample_df.to_csv(SAMPLE_OUTPUT_PATH, index=False)

print(f"Created raw sample file for annotation: {SAMPLE_OUTPUT_PATH}")

# Only create annotated file if it doesn't already exist
if not ANNOTATED_OUTPUT_PATH.exists():
    sample_df.to_csv(ANNOTATED_OUTPUT_PATH, index=False)
    print(f"Created annotated file: {ANNOTATED_OUTPUT_PATH}")
else:
    print(f"Annotated file already exists, not overwriting.")
