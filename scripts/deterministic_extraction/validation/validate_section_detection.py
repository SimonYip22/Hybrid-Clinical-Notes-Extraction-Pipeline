"""
validate_section_detection.py

Purpose:
    Perform manual validation of the report section detection pipeline
    applied to ICU clinical notes. This ensures that section detection and
    extraction are functioning as intended before scaling to the full dataset.

Usage (terminal):
    export PYTHONPATH=$(pwd)/src
    python3 scripts/deterministic_extraction/validation/validate_section_detection.py

Workflow:
    1. Load the processed ICU corpus from a CSV file.
    2. Randomly sample a subset of notes for validation.

"""
# ---------------------------------------------------------------------
# IMPORTS & CONFIGURATION
# ---------------------------------------------------------------------

import pandas as pd
import random

from deterministic_extraction.section_detection import extract_sections

# Configuration
CORPUS_PATH = "data/processed/icu_corpus.csv"
TEXT_COLUMN = "TEXT"
SAMPLE_SIZE = 30

# Load corpus and extract notes
df = pd.read_csv(CORPUS_PATH)
notes = df[TEXT_COLUMN].dropna().tolist()

# ---------------------------------------------------------------------
# 1. MANUAL VALIDATION
# ---------------------------------------------------------------------

# Random sample notes for validation
sample = random.sample(notes, SAMPLE_SIZE)

# Apply extraction on sample and compare original vs extracted outputs
for i, note in enumerate(sample):

    extracted = extract_sections(note)

    print("\n" + "=" * 80)
    print(f"NOTE {i+1} — ORIGINAL")
    print("=" * 80)
    print(note)

    print("\n" + "-" * 80)
    print(f"NOTE {i+1} — CLEANED")
    print("-" * 80)
    print(extracted)

    input("\nPress Enter for next note...")