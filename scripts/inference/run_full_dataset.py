"""
run_full_dataset.py

Purpose:
    Generate a large-scale structured dataset by applying the full pipeline
    (extraction + validation) to the ICU corpus.

Workflow:
    1. Load pretrained transformer model and tokenizer
    2. Stream ICU dataset in chunks to manage memory
    3. Apply unified pipeline to each chunk
    4. Write entity-level outputs incrementally to disk (JSONL)

Outputs:
    JSONL file containing one entity per line:
        - Structured entity fields
        - Validation results (confidence, is_valid)

Notes:
    - Uses chunked processing for scalability and memory safety
    - No filtering applied; full dataset is preserved
"""

import pandas as pd
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from pipeline.pipeline import run_pipeline

# -------------------------
# CONFIG
# -------------------------

DATA_PATH = "data/processed/icu_corpus.csv"
MODEL_DIR = "models/bioclinicalbert_final"

OUTPUT_PATH = Path("outputs/datasets/full_entities.jsonl")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 1000
THRESHOLD = 0.549
BATCH_SIZE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# LOAD MODEL
# -------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

model.to(DEVICE)

# -------------------------
# PROCESS DATA IN CHUNKS
# -------------------------

with open(OUTPUT_PATH, "w") as f_out:

    for chunk in tqdm(pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE)):
            
            entities = run_pipeline(
                df=chunk,
                model=model,
                tokenizer=tokenizer,
                device=DEVICE,
                threshold=THRESHOLD,
                batch_size=BATCH_SIZE
            )

            for entity in entities:
                f_out.write(json.dumps(entity, default=float) + "\n")

print(f"\nSaved full dataset to: {OUTPUT_PATH}")