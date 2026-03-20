"""
validate_section_extraction.py

Purpose:
    Perform structured manual validation of the section extraction pipeline
    applied to ICU clinical notes. Combines qualitative inspection with
    basic quantitative diagnostics to ensure readiness for downstream analysis.

Usage (terminal):
    export PYTHONPATH=$(pwd)/src
    python3 scripts/deterministic_extraction/validation/validate_section_extraction.py

Workflow:
    1. Load ICU corpus
    2. Sample notes reproducibly
    3. Apply section extraction
    4. Display original vs extracted
    5. Compute basic validation diagnostics
"""

# ---------------------------------------------------------------------
# IMPORTS & CONFIGURATION
# ---------------------------------------------------------------------

import pandas as pd
import random
from collections import Counter

from deterministic_extraction.section_extraction import extract_sections

# Configuration
CORPUS_PATH = "data/processed/icu_corpus.csv"
TEXT_COLUMN = "TEXT"
SAMPLE_SIZE = 30
RANDOM_SEED = 42

# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

df = pd.read_csv(CORPUS_PATH)
notes = df[TEXT_COLUMN].dropna().tolist()

# Reproducibility
random.seed(RANDOM_SEED)

# Sample notes
sample = random.sample(notes, SAMPLE_SIZE)

# ---------------------------------------------------------------------
# VALIDATION TRACKERS
# ---------------------------------------------------------------------

section_counts = Counter()
empty_sections = 0
notes_with_no_sections = 0

# ---------------------------------------------------------------------
# MANUAL VALIDATION LOOP
# ---------------------------------------------------------------------

for i, note in enumerate(sample):

    extracted = extract_sections(note)

    # Track stats
    if not extracted:
        notes_with_no_sections += 1

    for k, v in extracted.items():
        section_counts[k] += 1
        if not v.strip():
            empty_sections += 1

    # Display

    print("\n" + "-" * 80)
    print(f"NOTE {i+1} — EXTRACTED SECTIONS")
    print("-" * 80)

    for section, content in extracted.items():
        print(f"\n[{section.upper()}]")
        print(content)

    input("\nPress Enter for next note...")

# ---------------------------------------------------------------------
# SUMMARY STATISTICS
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

print(f"Total notes reviewed: {SAMPLE_SIZE}")
print(f"Notes with no detected sections: {notes_with_no_sections}")
print(f"Empty sections detected: {empty_sections}")

print("\nSection frequency:")
for section, count in section_counts.items():
    print(f"{section}: {count}")