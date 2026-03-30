import json
import pandas as pd
from pathlib import Path

# -------------------------
# Paths
# -------------------------
INPUT_PATH = Path("data/interim/extraction_candidates.jsonl")
SAMPLE_OUTPUT_PATH = Path("data/extraction/sampling/annotation_sample_raw.csv")
ANNOTATED_OUTPUT_PATH = Path("data/extraction/sampling/annotation_sample_labeled.csv")

# Ensure output directory exists
SAMPLE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
ANNOTATED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# -------------------------
# Load JSONL
# -------------------------
records = []

with open(INPUT_PATH, "r") as f:
    for line in f:
        row = json.loads(line)

        records.append({
            "note_id": row.get("note_id"),
            "entity_text": row.get("entity_text"),
            "entity_type": row.get("entity_type"),
            "sentence_text": row.get("sentence_text"),
            "negated": row.get("negated"),
            "task": row.get("validation", {}).get("task")
        })

df = pd.DataFrame(records)

print(f"Loaded {len(df)} total entities")

# -------------------------
# Stratified sampling
# -------------------------
N_PER_CLASS = 200

sampled_dfs = []

for entity_type in ["SYMPTOM", "INTERVENTION", "CLINICAL_CONDITION"]:
    subset = df[df["entity_type"] == entity_type]

    if len(subset) < N_PER_CLASS:
        raise ValueError(f"Not enough samples for {entity_type}")

    sampled = subset.sample(n=N_PER_CLASS, random_state=42)
    sampled_dfs.append(sampled)

# Combine
sample_df = pd.concat(sampled_dfs).reset_index(drop=True)

# Shuffle final dataset
sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)

# -------------------------
# Add annotation column
# -------------------------
sample_df["is_valid"] = ""

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
