"""
symptom_rules.py

Purpose:
    Deterministic (rule-based) extraction of SYMPTOM entities from clinical note text.
    Uses regex pattern matching to generate candidate symptom mentions and applies
    lightweight negation detection at the sentence level.

Pipeline Role (Phase 2):
    - High-precision candidate generator
    - Produces span-aligned outputs for downstream transformer validation

Workflow:
    1. Filter to symptom-relevant sections (e.g., HPI, ROS)
    2. Split section text into sentences with character offsets
    3. For each sentence:
        - Detect symptom mentions via regex (concept-level)
        - Convert match spans to note-level character indices
        - Map spans to token indices for negation handling
        - Apply simple pre-scope negation detection
        - Deduplicate (max one instance per concept per sentence)

Output:
    List of structured SYMPTOM entities, each containing:
        - Metadata (IDs)
        - Extracted span (entity_text, concept)
        - Provenance (char offsets, sentence, section)
        - Signal (negated)
        - Validation placeholder (for transformer layer)

Design Constraints:
    - Prioritises precision over recall
    - Uses simple deterministic rules (no deep linguistic parsing)
    - Delegates contextual interpretation to transformer to validate negation
"""

from typing import List, Dict
import re

# For sentence splitting and tokenization
from deterministic_extraction.sentence_segmentation import split_into_sentences


# Ensures NLTK resources are available for sentence splitting
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

# Target sections for symptom extraction
TARGET_SYMPTOM_SECTIONS = {
    "chief complaint",
    "hpi",
    "review of systems",
}

# Negation words that indicate symptom absence
NEGATION_TERMS = {"no", "denies", "denied", "without", "not", "negative"}

# Break words that terminate negation scope
NEGATION_BREAKS = {"but", "however", "although"}

# ------------------------------------------------------------
# 2. SYMPTOM PATTERNS
# ------------------------------------------------------------

# Concept-level symptom candidates mapped to synonym regex patterns for detection
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
    "seizure": [
        r"\b(epileptic seizure|seizure|seizures|seizing|convulsion|convulsions|seizure-like activity|szr)\b"
    ],
    "anorexia": [
        r"\b(anorexia|loss of appetite|decreased appetite|poor appetite|poor intake)\b"
    ],
}

# ------------------------------------------------------------
# 3. TOKEN–CHARACTER ALIGNMENT
# ------------------------------------------------------------

def map_char_to_token(sentence: str):
    """
    Purpose:
        Convert a sentence into tokens and map each token to its character
        span in the original sentence.

    Why this is needed:
        - Regex operates on character indices
        - Negation logic operates on tokens (words)
        - This function bridges the two representations

    Output:
        tokens: List[str]
            Lowercased word tokens extracted from the sentence

        positions: List[Tuple[str, int, int]]
            Each entry contains:
                (token, start_char_index, end_char_index)

    Example:
        Input:
            "Denies chest pain"

        Output:
            tokens = ["denies", "chest", "pain"]
            positions = [
                ("denies", 0, 6),
                ("chest", 7, 12),
                ("pain", 13, 17)
            ]

    Notes:
        - Uses simple regex tokenisation (\w+)
        - Assumes tokens appear sequentially (left-to-right)
        - Not a full NLP tokenizer, but sufficient for rule-based processing
    """
    # Simple tokeniser splits sentence into words (tokens), removing punctuation
    tokens = re.findall(r"\w+", sentence.lower())

    # List to hold token + start/end character positions
    positions = []

    # Tracks the current character index in the sentence for mapping tokens
    idx = 0

    # Loop through tokens and find their positions in the original sentence
    for token in tokens:
        # Find the first occurrence of the token in the sentence starting from idx
        pos = sentence.lower().find(token, idx)
        # Record the token + start character + end character positions
        positions.append((token, pos, pos + len(token)))
        # Move idx forward to search for the next token after the current token
        idx = pos + len(token)

    return tokens, positions

# ------------------------------------------------------------
# 4. SIMPLE NEGATION DETECTION (SYMPTOM ONLY)
# ------------------------------------------------------------
def is_negated_simple(tokens, token_idx):
    """
    Purpose:
        Determine whether a symptom mention is negated based on
        preceding tokens in the sentence.

    Logic:
        - Scan tokens before the symptom
        - Activate negation when a negation term is found
        - Deactivate negation when a break term is found
        - Final state determines whether the symptom is negated

    Input:
        tokens: List[str]
            Tokenised sentence (lowercased words)

        token_idx: int
            Index of the token where the symptom begins

    Output:
        bool:
            True  → symptom is negated
            False → symptom is not negated

    Example:
        "denies chest pain"
            → True
        "no chest pain but nausea"
            → pain = True
            → nausea = False

    Limitations:
        - Does not model full negation scope (e.g. lists, conjunctions)
        - Does not capture syntactic structure or uncertainty
        - Assumes negation is local and linear
    """

    # Default to non-negated if no tokens before symptom
    negation_active = False

    # Loop over tokens before the symptom occurrence (token_idx)
    for t in tokens[:token_idx]:
        # Check for negation terms in the preceding tokens
        if t in NEGATION_TERMS:
            # Activate negation
            negation_active = True
        # Check for break terms in the preceding tokens
        if t in NEGATION_BREAKS:
            # Deactivate negation
            negation_active = False

    return negation_active

# ------------------------------------------------------------
# 5. MAIN EXTRACTION FUNCTION
# ------------------------------------------------------------

def extract_symptoms(note_id: str, subject_id: str, hadm_id: str, icustay_id: str,
                     section: str, text: str) -> List[Dict]:
    """
    Extract SYMPTOM entities from a section of clinical text using deterministic rules.

    Inputs:
        note_id, subject_id, hadm_id, icustay_id:
            Identifiers for traceability and downstream grouping

        section: str
            Section name of the note (used to filter relevant content)

        text: str
            Raw section text to process

    Processing Steps:
        1. Section filtering:
            - Only process predefined symptom-relevant sections

        2. Sentence segmentation:
            - Split text into sentences with character offsets relative to input text

        3. Per-sentence extraction:
            - Apply regex patterns to detect symptom mentions (concept-level)
            - Convert sentence-level match indices → note-level character offsets
            - Extract exact text span (entity_text)

        4. Token alignment:
            - Tokenise sentence and map tokens to character spans
            - Identify token index corresponding to match start

        5. Negation detection:
            - Apply simple rule over preceding tokens
            - Assign negated = True/False

        6. Deduplication:
            - Allow at most one instance per concept per sentence

        7. Output construction:
            - Create structured dict per entity
            - Include validation placeholder for downstream transformer

    Output:
        List[Dict]:
            Each dict represents one SYMPTOM entity with:
                - Span-aligned extraction (entity_text, char_start, char_end)
                - Concept label
                - Context (sentence, section)
                - Negation signal
                - Validation scaffold

    Notes:
        - Performs candidate generation, not final classification
        - Contextual correctness is determined by the transformer layer
        - Assumes sentence offsets are relative to the provided text
    """
    results = []
    
    # Step 1: Only process if the section is one of the 3 target sections for symptom extraction
    if section.lower() not in TARGET_SYMPTOM_SECTIONS:
        return results

    # Step 2: Split the section text into sentences and character spans relative to full note using split_into_sentences function
    sentence_spans = split_into_sentences(text)

    # Step 3: Loop through each sentence 
    for s in sentence_spans:
        # Extract sentence text and its character offsets in the original note
        sent_text = s["sentence"]
        sent_start = s["start"]
        sent_end = s["end"]
        # Lowercase sentence for regex matching (negation logic will use original tokens)
        lowered_sent = sent_text.lower()

        # Step 4: Token mapping (computed once per sentence) using map_char_to_token function
        tokens, token_positions = map_char_to_token(sent_text)

        # Deduplication: one type of concept per sentence. Resets for each new sentence.
        seen_concepts = set()

        # Step 5: Loop through symptom concepts
        for concept, patterns in SYMPTOM_PATTERNS.items():

            # Skip if this concept has already been matched in the current sentence (deduplication)
            if concept in seen_concepts:
                continue
            
            # Step 6: Loop through regex patterns for this concept and search in the sentence
            for pattern in patterns:
                # Use regex search to find the first occurrence of any pattern for this concept in the sentence
                match = re.search(pattern, lowered_sent)

                # If a match is found
                if match:
                    # Add the concept to the set of seen concepts
                    seen_concepts.add(concept)

                    # Step 7: Convert to global note-level character offsets by adding the sentence start offset to the match's start and end indices
                    start_idx = sent_start + match.start()
                    end_idx = sent_start + match.end()

                    # Step 8: Extract the exact text span from the original note using the computed character offsets
                    span_text = text[start_idx:end_idx]


                    # Token index is None until we find which token corresponds to the start of the matched symptom span
                    token_idx = None

                    # Step 9: Loop through token index positions in the sentence (token, start_char, end_char)
                    for i, (_, s_pos, e_pos) in enumerate(token_positions):
                        # Check if the start of the matched span falls within the character span of this token
                        if s_pos <= match.start() < e_pos:
                            # If it does, we have found the token index that corresponds to the symptom
                            token_idx = i
                            break

                    # Start with the assumption that the symptom is not negated
                    negated = False

                    # Step 10: Negation detection using is_negated_simple function, which checks the tokens before the symptom mention for negation cues
                    if token_idx is not None:
                        # If a valid negation word is found before the symptom index, the symptom is marked as negated
                        negated = is_negated_simple(tokens, token_idx)

                    # Step 11: Append the extracted symptom entity with all relevant information to the results list
                    results.append({
                        "note_id": note_id,
                        "subject_id": subject_id,
                        "hadm_id": hadm_id,
                        "icustay_id": icustay_id,

                        "entity_text": span_text,
                        "concept": concept,
                        "entity_type": "SYMPTOM",

                        "char_start": start_idx,
                        "char_end": end_idx,
                        "sentence_text": sent_text,
                        "section": section,

                        "negated": negated,

                        "validation": {
                            "is_valid": None,
                            "confidence": 0.0,
                            "task": "symptom_presence"
                        }
                    })

                    break  # stop after first match for this specific concept

    return results