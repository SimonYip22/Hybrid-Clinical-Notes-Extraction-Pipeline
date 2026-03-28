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

    "airway_management": [
        r"\b(intubated|intubation|reintubated|extubated|endotracheal tube(s)?|ett(s)?|et tube(s)?|tracheostomy|trach(eostomy)?|trachy|airway secured)\b"
    ],

    "oxygen_therapy": [
        r"\b(oxygen therapy|supplemental oxygen|o2 therapy|nasal cannula(s)?|nc(s)?|non[- ]rebreather(s)?|nrb(s)?|face mask oxygen|venturi(s)?|high[- ]flow oxygen|hfno|hfnc(s)?)\b"
    ],

    "mechanical_ventilation": [
        r"\b(mechanical vent(ilation)?|mv|ventilated|on ventilator|niv|non[- ]invasive vent(ilation)?|cpap|bipap|psv|pressure supp(ort)?|peep)\b"
    ],

    "fluid_therapy": [
        r"\b(iv fluid(s)?|intravenous fluid(s)?|ivf(s)?|fluid bolus(es)?|bolus given|crystalloid(s)?|hartmann(')?s|ringer(')?s|normal saline|0\.9% saline|nacl|fluid resus(citation)?|resus fluid(s)?|ns)\b"
    ],

    "vasopressor_inotrope": [
        r"\b(vasopressor(s)?|inotrope(s)?|inotropic support|pressor(s)?|norad(renaline)?|norepinephrine|adrenaline|epinephrine|(vaso)?pressin|dopamine|dobutamine|neo)\b"
    ],

    "analgesia": [
        r"\b(analgesia|analgesic(s)?|pain relief|(oro)?morph(ine)?|fent(anyl)?|remifentanil|oxycodone|codeine|paracetamol|acetaminophen|ibuprofen)\b"
    ],

    "sedation": [
        r"\b(sedation|sedated|propofol|midazolam|dexmedetomidine|ketamine|sedative|lorazepam|diazepam)\b"
    ],

    "paralysis": [
        r"\b(rocuronium|atracurium|vecuronium|neuromuscular blockade|nmba(s)?|nmb(s)?)\b"
    ],

    "antibiotic_therapy": [
        r"\b(antibiotic(s)?|abx('s|s)?|piperacillin[-/ ]tazobactam|tazocin|meropenem|ceftriaxone|co[- ]amox(iclav)?|augmentin|vanc(omycin)?|gent(amicin)?|doxy(cycline)?|metro(nidazole)?|fluclox(acillin)?|amox(icillin)?)\b"
    ],

    "anticoagulation": [
        r"\b(anticoag(ulation|ulated)?|antiplatelet(s)?|heparin|lmwh|enoxaparin|warfarin|apixaban|rivaroxaban|dabigatran|aspirin|asa|clopidogrel|ticagrelor|fondaparinux)\b"
    ],

    "blood_product": [
        r"\b(transfusion(s)?|transfused|packed red cell(s)?|prbc(s)?|prc(s)?|ffp|fresh frozen plasma|platelet(s)?|plt(s)?|cryo(precipitate)?)\b"
    ],

    "renal_replacement_therapy": [
        r"\b(rrt|renal replacement therapy|dialysis|h(a)?emodialysis|cvvh|cvvhd|cvvhdf|dialysed)\b"
    ],

    "procedure_general": [
        r"\b(central line(s)?|cvc(s)?|c(entral)?[ ]v(enous) cath(eter)?(s)?|art(erial)? line(s)?|a[- ]line(s)?|chest drain(s)?|icc(s)?|intercostal catheter(s)?|picc|naso gastric tube(s)?|ng tube(s)?|ngt(s)?|foley catheter(s)?|urinary cath(eter)?(s)?)\b"
    ],

    "surgical_procedure": [
        r"\b(surgical (procedure|intervention)|laparotomy|laparoscopy|laparoscopic|thoracotomy|craniotomy|resection(s)?|resected|amputation(s)?|amputated|transplant(ed)?)\b"
    ],

    "nutrition": [
        r"\b(enteral feed(ing)?|ng feed(ing)?|tube feed(ing)?|parenteral nutrition|tpn|peg feed(ing)?|j[- ]tube|jtube|peg)\b"
    ],

    "cardiovascular_support": [
        r"\b(pacing|pacemaker(s)?|cardioversion|dc cv|tvp)\b"
    ],

    "cardiovascular_drugs": [
        r"\b(labetalol|metoprolol|nicardipine|diltiazem|verapamil)\b"
    ],

    "electrolyte_replacement": [
        r"\b(electrolyte replacement|electrolyte repletion|potassium replacement|kcl|magnesium replacement|mgso4|calcium replacement|calcium gluconate)\b"
    ],

    "resuscitation": [
        r"\b(cpr|cardiopulmonary resuscitation|resuscitation|resus|als|advanced life support|bls|basic life support|rosc|return of spontaneous circulation|defibrillation|defib|defibrillated)\b"
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
    
    return results