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

    if "TEXT" not in df.columns:
        raise ValueError("DataFrame must contain 'TEXT' column")

    all_entities = []

    for idx, row in enumerate(df.itertuples(index=False), start=1):

        text = row.TEXT
        subject_id = str(getattr(row, "SUBJECT_ID", ""))
        hadm_id = str(getattr(row, "HADM_ID", ""))
        icustay_id = str(getattr(row, "ICUSTAY_ID", ""))
        
        note_id = f"note_{idx + 1}"

        entities = extract_entities_from_note(
            note_id=note_id,
            text=text,
            subject_id=subject_id,
            hadm_id=hadm_id,
            icustay_id=icustay_id
        )

        all_entities.extend(entities)

    return all_entities