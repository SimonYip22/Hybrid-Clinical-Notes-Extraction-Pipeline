"""
validate_sentence_segmentation.py

Purpose:
    Perform structured manual validation of the sentence segmentation pipeline
    applied to ICU clinical notes.

Usage (terminal):
    export PYTHONPATH=$(pwd)/src
    python3 scripts/deterministic_extraction/validation/validate_sentence_segmentation.py

Workflow:
    1. Load a sample of clinical notes from the processed ICU corpus
    2. For each note:
        a. Extract sections using the existing section extraction module
        b. Apply the sentence segmentation function to each section
        c. Print the original section text along with the identified sentence spans and their offsets
    3. Manually review the output to verify:
        - Sentence boundaries are correctly identified
        - Character offsets accurately map back to the original text
        - No sentences are missed or incorrectly merged

"""

# ---------------------------------------------------------------------
# IMPORTS & CONFIGURATION
# ---------------------------------------------------------------------

import pandas as pd
import random
from collections import Counter

from deterministic_extraction.section_extraction import extract_sections
from deterministic_extraction.sentence_segmentation import split_into_sentences

# Configuration
CORPUS_PATH = "data/processed/icu_corpus.csv"
TEXT_COLUMN = "TEXT"
SAMPLE_SIZE = 10
RANDOM_SEED = 42

# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

df = pd.read_csv(CORPUS_PATH)
notes = df[TEXT_COLUMN].dropna().tolist()

random.seed(RANDOM_SEED)
sample = random.sample(notes, SAMPLE_SIZE)

# ---------------------------------------------------------------------
# VALIDATION LOOP
# ---------------------------------------------------------------------

for i, note in enumerate(sample):

    print("\n" + "="*80)
    print(f"NOTE {i+1} — SECTIONS")
    print("="*80)

    sections = extract_sections(note)

    # -------------------------------
    # PRINT SECTIONS
    # -------------------------------
    for section, content in sections.items():
        print(f"\n[{section.upper()}]")
        print(content[:300])

    # -------------------------------
    # SENTENCE SEGMENTATION
    # -------------------------------
    print("\n" + "-"*80)
    print(f"NOTE {i+1} — SENTENCE SPANS")
    print("-"*80)

    for section, content in sections.items():

        sentences = split_into_sentences(content)

        print(f"\n[{section.upper()}]")

        for s in sentences:
            print(f'{s["start"]}:{s["end"]} -> {s["sentence"]}')

    input("\nPress Enter for next note...")