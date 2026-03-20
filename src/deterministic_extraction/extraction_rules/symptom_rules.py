"""
symptom_rules.py

Purpose:
    Rule-based symptom extraction from clinical notes using regex + simple negation.

Workflow:
    1. Filter relevant sections
    2. Split into sentences
    3. For each sentence:
    - Detect symptoms via regex
    - Apply simple negation logic
    - Deduplicate per concept per sentence

Output:
    List of extracted symptom entities with negation flags
"""

from typing import List, Dict
import re
from deterministic_extraction.sentence_segmentation import split_into_sentences

from nltk.tokenize import sent_tokenize

import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# ------------------------------------------------------------
# 1. CONFIG
# ------------------------------------------------------------

TARGET_SYMPTOM_SECTIONS = {
    "chief complaint",
    "hpi",
    "review of systems",
}

NEGATION_TERMS = {"no", "denies", "denied", "without", "not", "negative"}

NEGATION_BREAKS = {"but", "however", "although"}

# ------------------------------------------------------------
# 2. SYMPTOM PATTERNS
# ------------------------------------------------------------

SYMPTOM_PATTERNS = {
    "pain": [
        r"\b(chest pain|abdominal pain|abd pain|abdo pain|back pain|neck pain|pain)\b",
    ],
    "headache": [
        r"\b(headache|head pain)\b"
    ],
    "chest_discomfort": [
        r"\b(chest tightness|chest discomfort)\b"
    ],
    "palpitations": [
        r"\b(palpitations|heart racing)\b"
    ],
    "dyspnoea": [
        r"\b(shortness of breath|short of breath|sob|dyspnea|dyspnoea|breathlessness|difficulty breathing)\b"
    ],
    "syncope": [
        r"\b(syncope|fainting|fainted|passing out|passed out|syncopal episode|loss of consciousness|loc)\b"
    ],
    "nausea_vomiting": [
        r"\b(nausea and vomiting|nausea|nauseated|vomiting|vomit|vomited|hematemesis|retching|emesis|n/v|n\+v)\b"
    ],
    "fatigue": [
        r"\b(fatigue|fatigued|tired|tiredness|lethargy|lethargic)\b"
    ],
    "dizziness": [
        r"\b(dizziness|lightheadedness|lightheaded|dizzy)\b"
    ],
    "fever": [
        r"\b(fever|febrile|pyrexia|high temperature|spiking temperature|feverish|fevered|rigors|chills)\b"
    ],
    "cough": [
        r"\b(productive cough|dry cough|wet cough|coughing|nonproductive cough|non-productive cough|cough)\b"
    ],
    "diarrhoea": [
        r"\b(diarrhea|diarrhoea|loose stools|loose stool|watery stools|frequent stools)\b"
    ],
    "confusion": [
        r"\b(confusion|confused|altered mental state|altered mental status|altered consciousness|disorientation|disoriented|ams|acute confusional state|acute confusional episode|delirium|delirious)\b"
    ],
    "bleeding": [
        r"\b(bleed|bleeding|hematochezia|melena|epistaxis|hemoptysis|haemoptysis)\b"
    ],
    "weakness": [
        r"\b(generalised weakness|weakness)\b"
    ],
    "anorexia": [
        r"\b(anorexia|loss of appetite|decreased appetite|poor appetite|poor intake)\b"
    ],
}

# ------------------------------------------------------------
# 3. NEGATION DETECTION
# ------------------------------------------------------------
def is_negated_simple(tokens, token_idx):
    """
    Negation rule:
    - Negation turns ON when encountering a negation term
    - Turns OFF when encountering a break word (e.g. 'but')
    - Applies to tokens before the symptom
    """
    negation_active = False

    for t in tokens[:token_idx]:
        if t in NEGATION_TERMS:
            negation_active = True
        if t in NEGATION_BREAKS:
            negation_active = False

    return negation_active

def map_char_to_token(sentence: str):
    """
    Maps character positions to token indices
    """
    tokens = re.findall(r"\w+", sentence.lower())
    positions = []

    idx = 0
    for token in tokens:
        pos = sentence.lower().find(token, idx)
        positions.append((token, pos, pos + len(token)))
        idx = pos + len(token)

    return tokens, positions

# ------------------------------------------------------------
# 4. MAIN EXTRACTION FUNCTION
# ------------------------------------------------------------

def extract_symptoms(note_id: str, subject_id: str, hadm_id: str, icustay_id: str,
                     section: str, text: str) -> List[Dict]:
    
    results = []
    
    if section.lower() not in TARGET_SYMPTOM_SECTIONS:
        return results

    sentence_spans = split_into_sentences(text)

    for s in sentence_spans:
        sent_text = s["sentence"]
        sent_start = s["start"]
        sent_end = s["end"]
        lowered_sent = sent_text.lower()

        # Token mapping (computed once per sentence)
        tokens, token_positions = map_char_to_token(sent_text)

        # Deduplication: one concept per sentence
        seen_concepts = set()

        for concept, patterns in SYMPTOM_PATTERNS.items():

            if concept in seen_concepts:
                continue

            for pattern in patterns:
                match = re.search(pattern, lowered_sent)

                if match:
                    seen_concepts.add(concept)

                    start_idx = sent_start + match.start()
                    end_idx = sent_start + match.end()
                    span_text = text[start_idx:end_idx]

                    # Map match → token index
                    token_idx = None
                    for i, (_, s_pos, e_pos) in enumerate(token_positions):
                        if s_pos <= match.start() < e_pos:
                            token_idx = i
                            break

                    negated = False
                    if token_idx is not None:
                        negated = is_negated_simple(tokens, token_idx)

                    results.append({
                        "note_id": note_id,
                        "subject_id": subject_id,
                        "hadm_id": hadm_id,
                        "icustay_id": icustay_id,
                        "entity_text": span_text,
                        "entity_type": "SYMPTOM",
                        "char_start": start_idx,
                        "char_end": end_idx,
                        "sentence_text": sent_text,
                        "section": section,
                        "negated": negated,
                        "validation_confidence": 0.0
                    })

                    break  # stop after first match for this concept

    return results