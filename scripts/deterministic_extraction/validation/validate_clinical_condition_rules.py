"""
validate_clinical_condition_rules.py

Purpose:
    Evaluate the behaviour of the rule-based CLINICAL_CONDITION extraction pipeline
    on a sample of ICU clinical notes. This script analyses candidate generation
    characteristics including coverage, redundancy, and concept distribution.

Usage:
    export PYTHONPATH=$(pwd)/src
    python3 scripts/deterministic_extraction/validation/validate_clinical_condition_rules.py

Workflow:
    1. Load the ICU corpus
    2. Randomly sample a subset of notes
    3. Extract sections using `extract_sections`
    4. Filter to clinical_condition-relevant sections
    5. For each section:
        a. Extract clinical_condition candidates using `extract_clinical_conditions`
        b. Preserve all extracted spans (no concept-level deduplication)
    6. Track metrics:
        - Notes with any sections
        - Notes with target sections
        - Notes with no target sections
        - Notes with target sections but no clinical_conditions
        - Total clinical_condition candidates extracted
        - Concept frequency distribution
        - Occurrence of multiple matches per concept per sentence
    7. Print outputs per note:
        - Sections (truncated)
        - Extracted clinical_condition entities with concept and section
    8. Summarise:
        - Section coverage
        - Extraction performance (total, average per note)
        - Top clinical_condition concepts
        - Redundancy patterns (multiple matches per concept per sentence)

Design Focus:
    - Evaluates recall-oriented candidate generation behaviour
    - Inspects redundancy arising from minimal deduplication policy
    - Identifies over- or under-represented concepts
    - Does NOT evaluate:
        - Contextual correctness (handled by transformer)
        - Negation
        - Temporal validity

Key Interpretation:
    - High extraction volume is expected (recall-first design)
    - Multiple mentions of the same concept in a sentence are valid outputs
    - Redundancy analysis helps assess whether rule patterns are too broad
"""

import pandas as pd
import random
from collections import Counter, defaultdict

from deterministic_extraction.section_extraction import extract_sections
from deterministic_extraction.extraction_rules.clinical_condition_rules import (
    extract_clinical_conditions,
    TARGET_CLINICAL_CONDITION_SECTIONS
)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

CORPUS_PATH = "data/processed/icu_corpus.csv"
TEXT_COLUMN = "TEXT"
SAMPLE_SIZE = 200
RANDOM_SEED = 42

df = pd.read_csv(CORPUS_PATH)
notes = df.to_dict(orient="records")

random.seed(RANDOM_SEED)
sample = random.sample(notes, SAMPLE_SIZE)

# ---------------------------------------------------------------------
# TRACKERS
# ---------------------------------------------------------------------

total_notes = len(sample)

notes_with_any_sections = 0
notes_with_target_sections = 0
notes_with_no_target_sections = 0
notes_with_target_but_no_clinical_conditions = 0

total_clinical_conditions = 0

concept_counter = Counter()

# Global redundancy tracker
global_duplicate_concepts = []

# ---------------------------------------------------------------------
# VALIDATION LOOP
# ---------------------------------------------------------------------

for i, note in enumerate(sample):
    text = note[TEXT_COLUMN]

    note_id = note.get("NOTE_ID", f"note_{i}")
    subject_id = note.get("SUBJECT_ID", "")
    hadm_id = note.get("HADM_ID", "")
    icustay_id = note.get("ICUSTAY_ID", "")

    sections = extract_sections(text)

    if sections:
        notes_with_any_sections += 1

    # Filter to target sections
    target_sections = {
        k: v for k, v in sections.items()
        if k.lower() in TARGET_CLINICAL_CONDITION_SECTIONS
    }

    print("\n" + "-" * 80)
    print(f"NOTE {i+1}")
    print("-" * 80)

    if not target_sections:
        notes_with_no_target_sections += 1
        print("No target sections found.")
        continue

    notes_with_target_sections += 1

    # ----------------------------------------------------------
    # Show sections (truncated)
    # ----------------------------------------------------------

    print("\nSECTIONS:")
    for section, content in target_sections.items():
        print(f"\n[{section.upper()}]")
        print(content[:300])

    # ----------------------------------------------------------
    # Extraction
    # ----------------------------------------------------------

    extracted = []

    for section, content in target_sections.items():
        ents = extract_clinical_conditions(
            note_id,
            subject_id,
            hadm_id,
            icustay_id,
            section,
            content
        )
        extracted.extend(ents)

    if not extracted:
        notes_with_target_but_no_clinical_conditions += 1

    # ----------------------------------------------------------
    # Tracking
    # ----------------------------------------------------------

    for e in extracted:
        concept_counter.update([e["concept"]])
        total_clinical_conditions += 1

    # Redundancy analysis (per note)
    concept_per_sentence = defaultdict(list)

    for e in extracted:
        key = (e["sentence_text"], e["concept"])
        concept_per_sentence[key].append(e["entity_text"])

    duplicate_concepts = {
        k: v for k, v in concept_per_sentence.items() if len(v) > 1
    }

    if duplicate_concepts:
        global_duplicate_concepts.append(duplicate_concepts)

    # ----------------------------------------------------------
    # Print extracted entities
    # ----------------------------------------------------------

    print("\nCLINICAL CONDITIONS:")

    if extracted:
        print(f"Extracted {len(extracted)} clinical conditions\n")

        for e in extracted:
            print(
                f'- {e["entity_text"]} '
                f'| concept={e["concept"]} '
                f'| section={e["section"]}'
            )
    else:
        print("No clinical conditions extracted")

# ---------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION COVERAGE")
print("=" * 80)

print(f"Total notes: {total_notes}")
print(f"Notes with ANY sections: {notes_with_any_sections}")
print(f"Notes with TARGET sections: {notes_with_target_sections}")
print(f"Notes WITHOUT target sections: {notes_with_no_target_sections}")

print("\n" + "=" * 80)
print("EXTRACTION PERFORMANCE")
print("=" * 80)

if notes_with_target_sections > 0:
    print(
        f"Notes with NO clinical conditions (given sections): "
        f"{notes_with_target_but_no_clinical_conditions} / {notes_with_target_sections} "
        f"({notes_with_target_but_no_clinical_conditions / notes_with_target_sections * 100:.1f}%)"
    )

print(f"Total clinical conditions extracted: {total_clinical_conditions}")

if notes_with_target_sections > 0:
    print(f"Average per note: {total_clinical_conditions / notes_with_target_sections:.2f}")

print("\n" + "=" * 80)
print("TOP CONCEPTS")
print("=" * 80)

for concept, count in concept_counter.most_common(20):
    print(f"{concept}: {count}")

# ----------------------------------------------------------
# GLOBAL REDUNDANCY SUMMARY
# ----------------------------------------------------------

if global_duplicate_concepts:
    print("\n" + "=" * 80)
    print("MULTIPLE MATCHES (same concept, same sentence)")
    print("=" * 80)

    for dup_dict in global_duplicate_concepts:
        for (sent, concept), ents in dup_dict.items():
            print(f"[{concept}] → {ents}")
else:
    print("\nNo multiple-match redundancy detected.")