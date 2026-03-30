"""
run_extraction_pipeline.py

Purpose:
    Orchestrates the deterministic extraction pipeline for all 3 entity types
    (SYMPTOM, INTERVENTION, CLINICAL_CONDITION) from ICU clinical notes.
    Integrates preprocessing, section extraction, sentence segmentation, and
    regex-based candidate generation.
    Produces structured, high-recall candidate outputs for downstream transformer
    validation and classification.

Run:
    export PYTHONPATH=$(pwd)/src
    python3 scripts/deterministic_extraction/run_extraction_pipeline.py

Workflow: 
    Load 10,000 sample of clinical notes from ICU corpus CSV -> For each note, run through the following steps:
        1. Preprocess note text (cleaning, normalisation)
        2. Extract sections (structured segmentation of note text)
        3. For each section:
            - Apply deterministic extraction rules for:
                • SYMPTOM
                • INTERVENTION
                • CLINICAL_CONDITION
            - Each extractor internally:
                a. Filters relevant sections
                b. Splits text into sentences
                c. Applies regex-based pattern matching
                d. Generates span-aligned candidate entities
        4. Aggregate all candidates across sections into a flat note-level list
        5. Write each candidate entity as a JSON line to output file

Outputs:
    1. JSONL File (primary output):
        Path: data/interim/extraction_candidates.jsonl

        Format:
            - One JSON object per line (newline-delimited JSON)
            - Each line represents a single extracted entity
        Design:
            - Flat structure (no nesting by note)
            - Multiple entities per note allowed
            - Optimised for transformer ingestion (one entity = one row)

    2. Terminal Validation Summary (secondary output):
        Printed after pipeline execution, including:
            - Total notes processed
            - Section extraction coverage
            - Entity extraction coverage
            - Notes with zero extractions
            - Per-entity-type coverage (symptom/intervention/condition)
            - Total extraction counts
            - Average entities per note (non-empty)
            - Sample of 5 notes with per-entity counts

    3. Sample Output (debugging aid):
        - First 5 processed notes with:
            • note_id
            • number of extracted symptoms/interventions/conditions
            • total entities per note

Design Principles:
    - Deterministic and reproducible (fixed random seed)
    - Transformer-ready output format (flat, structured, metadata-rich)

Notes:
    - `note_id` is synthetically generated due to absence in source data
    - Assumes ICU corpus contains required columns:
        SUBJECT_ID, HADM_ID, ICUSTAY_ID, TEXT
"""
# ------------------------------------------------------------
# 1. IMPORTS
# ------------------------------------------------------------

import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import random

from deterministic_extraction.preprocessing import preprocess_note
from deterministic_extraction.section_extraction import extract_sections
from deterministic_extraction.sentence_segmentation import split_into_sentences # Present in the extraction functions
from deterministic_extraction.extraction_rules.symptom_rules import extract_symptoms
from deterministic_extraction.extraction_rules.intervention_rules import extract_interventions
from deterministic_extraction.extraction_rules.clinical_condition_rules import extract_clinical_conditions

# ------------------------------------------------------------
# 2. CONFIG
# ------------------------------------------------------------

ICU_CORPUS = "data/processed/icu_corpus.csv"
VALIDATION_OUTPUT_PATH = "data/interim/extraction_candidates.jsonl"

Path(VALIDATION_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 10000
RANDOM_SEED = 42

# ------------------------------------------------------------
# 3. LOAD & SAMPLE DATA
# ------------------------------------------------------------

full_df = pd.read_csv(ICU_CORPUS)

# Check that required columns are present in the dataframe
required_columns = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "TEXT"]
missing_cols = [col for col in required_columns if col not in full_df.columns]
assert len(missing_cols) == 0, f"Missing required columns: {missing_cols}"

random.seed(RANDOM_SEED)
sample_df = full_df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)

# ------------------------------------------------------------
# 4. INITIALISE METRICS
# ------------------------------------------------------------

total_notes = 0
notes_with_sections = 0
notes_with_any_entities = 0

symptom_notes = 0
intervention_notes = 0
condition_notes = 0

total_symptoms = 0
total_interventions = 0
total_conditions = 0

sample_outputs = []

# ------------------------------------------------------------
# 5. RUN PIPELINE
# ------------------------------------------------------------

# Open output file for writing JSON lines
with open(VALIDATION_OUTPUT_PATH, "w") as f:

    # Iterate through each note (row) in the sample (1 iteration = 1 note) with progress bar
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df)):

        total_notes += 1

        # Per-note counters
        note_symptom_count = 0
        note_intervention_count = 0
        note_condition_count = 0

        # Create unique note ID using total_notes counter (e.g. note_1, note_2, ...)
        note_id = f"note_{total_notes}"

        # Extract metadata and text for the note and ensure they are strings
        subject_id = str(row.get("SUBJECT_ID", ""))
        hadm_id = str(row.get("HADM_ID", ""))
        icustay_id = str(row.get("ICUSTAY_ID", ""))
        text = row["TEXT"]

        # 1. Preprocess the note text (cleaning, normalisation)
        preprocessed_text = preprocess_note(text)

        # 2. Section extraction (1 note -> multiple sections) - returns dict of section_name: section_text
        sections = extract_sections(preprocessed_text)

        if sections:
            notes_with_sections += 1

        note_entities = [] # List stores all extracted entities for the note

        # 3. Loop through sections (1 iteration = 1 section)
        for section_name, section_text in sections.items():

            # 4. Run all extractors -> each returns a list of candidate entities (each {} = one entity)
            symptoms = extract_symptoms(
                note_id, subject_id, hadm_id, icustay_id,
                section_name, section_text
            )

            interventions = extract_interventions(
                note_id, subject_id, hadm_id, icustay_id,
                section_name, section_text
            )

            conditions = extract_clinical_conditions(
                note_id, subject_id, hadm_id, icustay_id,
                section_name, section_text
            )

            # Count per note
            note_symptom_count += len(symptoms)
            note_intervention_count += len(interventions)
            note_condition_count += len(conditions)

            # 5. Aggregate all candidates from the section into the note-level list -> [{}, {}, ...] 
            # Flat list of all entities from all sections for the one note
            note_entities.extend(symptoms)
            note_entities.extend(interventions)
            note_entities.extend(conditions)

        # 6. Update global metrics
        total_symptoms += note_symptom_count
        total_interventions += note_intervention_count
        total_conditions += note_condition_count

        if note_symptom_count > 0:
            symptom_notes += 1

        if note_intervention_count > 0:
            intervention_notes += 1

        if note_condition_count > 0:
            condition_notes += 1

        total_note_entities = (
            note_symptom_count +
            note_intervention_count +
            note_condition_count
        )

        if total_note_entities > 0:
            notes_with_any_entities += 1

        # Store small sample (first 5 notes)
        if len(sample_outputs) < 5:
            sample_outputs.append({
                "note_id": note_id,
                "n_symptoms": note_symptom_count,
                "n_interventions": note_intervention_count,
                "n_conditions": note_condition_count,
                "total": total_note_entities
            })

        # 7. Loop through all entities (1 iteration = 1 entity) and write to output file as JSON lines
        for entity in note_entities:

            # Sanity checks to ensure required fields are present and non-empty
            assert "entity_text" in entity and entity["entity_text"] != ""
            assert "concept" in entity and entity["concept"] != ""
            assert "entity_type" in entity

            f.write(json.dumps(entity) + "\n") # Convert dict to JSON string and write as a line in the output file

# ------------------------------------------------------------
# 6. FINAL SUMMARY
# ------------------------------------------------------------

print("\n==================== PIPELINE SUMMARY ====================")

print(f"Total notes processed: {total_notes}")
print(f"Notes with sections: {notes_with_sections} ({notes_with_sections/total_notes:.1%})")
print(f"Notes with ANY entities: {notes_with_any_entities} ({notes_with_any_entities/total_notes:.1%})")

empty_notes = total_notes - notes_with_any_entities
print(f"Notes with ZERO entities: {empty_notes} ({empty_notes/total_notes:.1%})")

print("\n--- Entity Coverage ---")
print(f"Notes with symptoms: {symptom_notes} ({symptom_notes/total_notes:.1%})")
print(f"Notes with interventions: {intervention_notes} ({intervention_notes/total_notes:.1%})")
print(f"Notes with conditions: {condition_notes} ({condition_notes/total_notes:.1%})")

print("\n--- Total Extractions ---")
print(f"Symptoms: {total_symptoms}")
print(f"Interventions: {total_interventions}")
print(f"Conditions: {total_conditions}")

print("\n--- Averages per note (non-empty only) ---")

if symptom_notes > 0:
    print(f"Avg symptoms per note: {total_symptoms / symptom_notes:.2f}")

if intervention_notes > 0:
    print(f"Avg interventions per note: {total_interventions / intervention_notes:.2f}")

if condition_notes > 0:
    print(f"Avg conditions per note: {total_conditions / condition_notes:.2f}")

total_entities = total_symptoms + total_interventions + total_conditions

if notes_with_any_entities > 0:
    print(f"\nAvg total entities per note (non-empty): {total_entities / notes_with_any_entities:.2f}")

print("\n==================== SAMPLE OUTPUT (5 NOTES) ====================")

for sample in sample_outputs:
    print(sample)
    

