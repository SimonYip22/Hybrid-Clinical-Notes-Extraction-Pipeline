"""
intervention_rules.py

Purpose:
    Deterministic (rule-based) extraction of INTERVENTION entities from clinical note text.
    Uses regex pattern matching to generate candidate intervention mentions without contextual interpretation.

Pipeline Role (Phase 2):
    - Recall-oriented, moderate-precision candidate generator
    - Produces span-aligned intervention candidates for downstream transformer validation

Workflow:
    1. Filter relevant sections (intervention-dense regions)
    2. Split text into sentences
    3. For each sentence:
        - Apply concept-specific regex patterns to each sentence
        - Extract all matching spans (no concept-level deduplication)
        - Remove exact duplicate spans (same start, end, concept)
        - Output structured candidate entities

Output:
    List of structured INTERVENTION entities, each containing:
        - Metadata (note_id, subject_id, etc.)
        - Extracted span (entity_text, concept)
        - Provenance (character offsets, sentence, section)
        - Signal (negated)
        - Placeholder validation fields for transformer classification

Design Principles:
    - Prioritises recall over precision at extraction stage
    - Intentionally broad regex patterns to over-generate candidates (recall-first strategy)
    - Allows multiple candidates per concept per sentence
    - Does NOT collapse semantically similar mentions
    - Maintains exact span traceability for auditability
    - Delegates all contextual interpretation to transformer layer
        - Filtering false positives
        - Determining whether intervention was actually performed
        - Resolving ambiguity in context and intent
"""

from typing import Dict
import re

# Import for sentence splitting
from deterministic_extraction.sentence_segmentation import split_into_sentences

# ------------------------------------------------------------
# 1. CONFIG
# ------------------------------------------------------------

# Target sections for intervention extraction
TARGET_INTERVENTION_SECTIONS = {
    "action",
    "assessment and plan",
    "assessment",
}

# ------------------------------------------------------------
# 2. INTERVENTION PATTERNS
# ------------------------------------------------------------

# Concept-level intervention candidates mapped to regex patterns for detection
INTERVENTION_PATTERNS = {

    "AIRWAY_MANAGEMENT": [
        r"\b(intubated|intubation|reintubated|extubated|endotracheal tube|ett|et tube|tracheostomy|trach(eostomy)?|airway secured)\b"
    ],

    "OXYGEN_THERAPY": [
        r"\b(oxygen therapy|supplemental oxygen|o2 therapy|nasal cannula|nc|non[- ]rebreather|nrb|face mask oxygen|venturi|high[- ]flow oxygen|hfno|hfnc)\b"
    ],

    "MECHANICAL_VENTILATION": [
        r"\b(mechanical ventilation|mv|ventilated|on ventilator|niv|non[- ]invasive ventilation|cpap|bipap|psv|pressure support|peep)\b"
    ],

    "FLUID_THERAPY": [
        r"\b(iv fluid(s)?|intravenous fluid(s)?|ivf|fluid bolus|bolus given|crystalloid(s)?|hartmann(')?s|ringer(')?s|normal saline|0\.9% saline|nacl|fluid resus(citation)?|resus fluid(s)?|ns)\b"
    ],

    "VASOPRESSOR_INOTROPE": [
        r"\b(vasopressor(s)?|inotrope(s)?|inotropic support|pressor(s)?|norad(renaline)?|norepinephrine|adrenaline|epinephrine|vasopressin|dopamine|dobutamine)\b"
    ],

    "ANALGESIA": [
        r"\b(analgesia|analgesic(s)?|pain relief|morphine|fentanyl|remifentanil|oxycodone|codeine|paracetamol|acetaminophen|ibuprofen)\b"
    ],

    "SEDATION": [
        r"\b(sedation|sedated|propofol|midazolam|dexmedetomidine|ketamine|sedative|lorazepam|diazepam)\b"
    ],

    "PARALYSIS": [
        r"\b(rocuronium|atracurium|vecuronium|neuromuscular blockade|nmba|nmb)\b"
    ],

    "ANTIBIOTIC_THERAPY": [
        r"\b(antibiotic(s)?|abx|piperacillin[-/ ]tazobactam|tazocin|meropenem|ceftriaxone|co[- ]amox(iclav)?|augmentin|vancomycin|vanc|gentamicin|doxycycline|metronidazole|flucloxacillin|amoxicillin)\b"
    ],

    "ANTICOAGULATION": [
        r"\b(anticoag(ulation|ulated)?|antiplatelet(s)?|heparin|lmwh|enoxaparin|warfarin|apixaban|rivaroxaban|dabigatran|aspirin|asa|clopidogrel|ticagrelor|fondaparinux)\b"
    ],

    "BLOOD_PRODUCT": [
        r"\b(transfusion|transfused|packed red cells|prbc|prc|ffp|fresh frozen plasma|platelet(s)?|plt|cryo(precipitate)?)\b"
    ],

    "RENAL_REPLACEMENT_THERAPY": [
        r"\b(rrt|renal replacement therapy|dialysis|haemodialysis|hemodialysis|cvvh|cvvhd|cvvhdf|dialysed)\b"
    ],

    "PROCEDURE_GENERAL": [
        r"\b(central line|cvc|central venous catheter|art(erial)? line|a[- ]line|chest drain|icc|intercostal catheter|picc|naso gastric tube|ng tube|ngt|foley catheter|urinary catheter)\b"
    ],

    "SURGICAL_PROCEDURE": [
        r"\b(surgical (procedure|intervention)|laparotomy|laparoscopy|laparoscopic|thoracotomy|craniotomy|resection|resected|amputation|amputated|transplant(ed)?|theatre)\b"
    ],

    "NUTRITION": [
        r"\b(enteral feed(ing)?|ng feed(ing)?|tube feed(ing)?|parenteral nutrition|tpn|peg feed(ing)?|j tube|jtube|peg)\b"
    ],

    "CARDIOVASCULAR_SUPPORT": [
        r"\b(pacing|pacemaker|cardioversion|dc cv|tvp)\b"
    ],

    "CARDIOVASCULAR_DRUGS": [
        r"\b(labetalol|metoprolol|nicardipine|diltiazem|verapamil)\b"
    ],

    "ELECTROLYTE_REPLACEMENT": [
        r"\b(electrolyte replacement|electrolyte repletion|potassium replacement|kcl|magnesium replacement|mgso4|calcium replacement|calcium gluconate)\b"
    ],

    "RESUSCITATION": [
        r"\b(cpr|cardiopulmonary resuscitation|resuscitation|resus|als|advanced life support|bla|basic life support|rosc|return of spontaneous circulation|defibrillation|defib|defibrillated|cardiac arrest|ventricular fibrillation|vf)\b"
    ]
}

# ------------------------------------------------------------
# 3. EXTRACTION FUNCTION
# ------------------------------------------------------------

def extract_interventions(note_id: str, subject_id: str, hadm_id: str, icustay_id:str,
                          section: str, text:str) -> list[Dict]:
    """
    Extract INTERVENTION entities from a section of clinical text using deterministic rules.

    Inputs:
        note_id, subject_id, hadm_id, icustay_id:
            Identifiers for traceability and downstream grouping

        section: str
            Section name of the note (used to filter relevant content)

        text: str
            Raw section text to process

    Processing Steps:
        1. Section filtering: 
            Only process sections likely to contain intervention information
        2. Sentence segmentation: 
            Split section text into sentences for more precise pattern matching
        3. Pattern matching: 
            For each sentence, apply regex patterns for each intervention concept
        4. Span extraction: 
            For each match, extract the exact text span and character offsets
        5. Deduplication:
            Only exact duplicate spans (same start, end, concept) are removed
            No concept-level or semantic deduplication is performed
        6. Result structuring: 
            Compile results into a structured format with metadata and placeholders

    Output:
        List[Dict]:
            Each dict represents one INTERVENTION entity with:
                - Span-aligned extraction (entity_text, char_start, char_end)
                - Concept label
                - Context (sentence, section)
                - Negation signal
                - Validation scaffold
    """

    # Initialise results list
    results = []

    # 1. Section filtering 
    if section.lower() not in TARGET_INTERVENTION_SECTIONS:
        return results
    
    # 2. Sentence segmentation
    sentence_spans = split_into_sentences(text)

    # Loop through each sentence
    for s in sentence_spans:

        # Extract sentence text and its starting character offset in the section
        sent_text = s["sentence"]
        sent_start = s["start"]

        lowered_sent = sent_text.lower()

        # To avoid exact duplicate extractions
        seen_spans = set()

        # 3. Pattern matching
        for concept, patterns in INTERVENTION_PATTERNS.items():

            # Apply each regex pattern associated with the current concept
            for pattern in patterns:

                # 4. Span extraction
                # Use re.finditer to find all non-overlapping matches of the pattern in the sentence
                for match in re.finditer(pattern, lowered_sent): 
                    
                    # Calculate character offsets of the match in the context of the entire section text
                    start_idx = sent_start + match.start()
                    end_idx = sent_start + match.end()

                    # 5. Deduplication
                    # Avoid duplicate spans (same start, same end, same concept) within the same sentence
                    span_key = (start_idx, end_idx, concept)
                    if span_key in seen_spans:
                        continue
                    
                    # Add new span to set toprevent duplicates
                    seen_spans.add(span_key)

                    # Extract the exact text span from the original sentence text using the character offsets
                    span_text = text[start_idx:end_idx]

                    # 6. Result structuring
                    results.append({
                        "note_id": note_id,
                        "subject_id": subject_id,
                        "hadm_id": hadm_id,
                        "icustay_id": icustay_id,

                        "entity_text": span_text,
                        "concept": concept,
                        "entity_type": "INTERVENTION",

                        "char_start": start_idx,
                        "char_end": end_idx,
                        "sentence_text": sent_text,
                        "section": section,

                        "negated": None, # Negation detection doesn't exist for this entity

                        "validation": {
                            "is_valid": None,
                            "confidence": 0.0,
                            "task": "intervention_performed"
                        }
                    })