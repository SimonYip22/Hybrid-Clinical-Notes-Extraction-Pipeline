"""
extraction.py

Purpose:
    Deterministic rule-based extraction of clinical entities from free-text notes.

Workflow:
    1. Preprocess note text
    2. Extract clinical sections
    3. Apply rule-based extraction (symptoms, interventions, conditions)
    4. Aggregate entity-level outputs

Interface:
    - extract_entities_from_note(): single note
    - run_extraction_on_dataframe(): batch processing (primary interface)

Inputs:
    - TEXT (required)
    - SUBJECT_ID, HADM_ID, ICUSTAY_ID (optional)

Outputs:
    List[Dict[str, Any]]
        - One dictionary per extracted entity
        - Includes entity text, type, spans, context, and metadata

Notes:
    - Designed for high recall (may include false positives)
    - Validation is handled downstream (validation.py)
"""

from typing import List, Dict, Any
import pandas as pd

from deterministic_extraction.preprocessing import preprocess_note
from deterministic_extraction.section_extraction import extract_sections
from deterministic_extraction.extraction_rules.symptom_rules import extract_symptoms
from deterministic_extraction.extraction_rules.intervention_rules import extract_interventions
from deterministic_extraction.extraction_rules.clinical_condition_rules import extract_clinical_conditions

# ------------------------------------------------------------
# 2. CORE FUNCTION (SINGLE NOTE)
# ------------------------------------------------------------

def extract_entities_from_note(
    note_id: str,
    text: str,
    subject_id: str = "",
    hadm_id: str = "",
    icustay_id: str = ""
) -> List[Dict[str, Any]]:
    """
    Extract clinical entities from a single note using deterministic rules.

    Workflow:
        1. Preprocess raw text
        2. Extract structured sections
        3. Apply rule-based extractors per section
        4. Aggregate entity outputs

    Args:
        note_id (str): Unique identifier for the note
        text (str): Raw clinical note text
        subject_id (str, optional): Patient identifier
        hadm_id (str, optional): Hospital admission ID
        icustay_id (str, optional): ICU stay ID

    Returns:
        List[Dict[str, Any]]:
            Entity-level outputs including text, type, spans, context, and metadata
    """
    # 1. Preprocess the note text
    preprocessed_text = preprocess_note(text)

    # 2. Section extraction
    sections = extract_sections(preprocessed_text)

    all_entities = []

    # 3. Loop through sections and apply extraction rules
    for section_name, section_text in sections.items():

        # 3.1 Extract SYMPTOMS
        symptoms = extract_symptoms(
            note_id, subject_id, hadm_id, icustay_id,
            section_name, section_text
        )

        # 3.2 Extract INTERVENTIONS
        interventions = extract_interventions(
            note_id, subject_id, hadm_id, icustay_id,
            section_name, section_text
        )

        # 3.3 Extract CLINICAL_CONDITIONS
        conditions = extract_clinical_conditions(
            note_id, subject_id, hadm_id, icustay_id,
            section_name, section_text
        )

        # Aggregate entities
        all_entities.extend(symptoms)
        all_entities.extend(interventions)
        all_entities.extend(conditions)

    return all_entities


# ------------------------------------------------------------
# 3. BATCH FUNCTION (DATAFRAME)
# ------------------------------------------------------------

def run_extraction_on_dataframe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Apply deterministic extraction across a dataset of clinical notes.

    Workflow:
        1. Validate required columns
        2. Iterate over notes efficiently (itertuples)
        3. Generate note-level identifiers
        4. Call single-note extraction
        5. Aggregate all entity outputs

    Args:
        df (pd.DataFrame):
            Input dataset containing:
                - TEXT (required)
                - SUBJECT_ID, HADM_ID, ICUSTAY_ID (optional)

    Returns:
        List[Dict[str, Any]]:
            Aggregated entity-level outputs across all notes
    """
    if "TEXT" not in df.columns:
        raise ValueError("DataFrame must contain 'TEXT' column")

    all_entities = []

    for idx, row in enumerate(df.itertuples(index=False), start=1):

        text = row.TEXT
        subject_id = str(getattr(row, "SUBJECT_ID", ""))
        hadm_id = str(getattr(row, "HADM_ID", ""))
        icustay_id = str(getattr(row, "ICUSTAY_ID", ""))
        
        note_id = f"note_{idx}"

        entities = extract_entities_from_note(
            note_id=note_id,
            text=text,
            subject_id=subject_id,
            hadm_id=hadm_id,
            icustay_id=icustay_id
        )

        all_entities.extend(entities)

    return all_entities