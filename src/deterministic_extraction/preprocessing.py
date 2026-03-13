"""
preprocessing.py

Purpose:
    Provide a deterministic preprocessing pipeline for ICU clinical notes to prepare 
    text for rule-based extraction. The pipeline removes systematic artefacts 
    (de-identification tokens, inconsistent whitespace) while preserving structural 
    and numeric content for downstream deterministic extraction.

Workflow:
    1. Normalise newline characters to standardise line breaks across the corpus.
    2. Remove de-identification tokens of the form [** ... **] to protect patient privacy.
    3. Normalise excessive spaces and tabs to ensure stable token offsets without 
       altering semantic content or section structure.
    4. Remove EMR trailing artefacts (e.g., 'References' blocks) that do not contain
       clinical information and can interfere with parsing.

Output:
    Returns a cleaned version of the input text, semantically equivalent but free of 
    de-identification artefacts, with consistent line and space formatting.
"""

import re

# --------------------------------------------------
# Regex Patterns
# --------------------------------------------------

# MIMIC de-identification tokens: [** ... **]
DEID_PATTERN = re.compile(r"\[\*\*.*?\*\*\]")
# Pattern to match multiple spaces or tabs (but not newlines)
MULTISPACE_PATTERN = re.compile(r"[ \t]+")

# --------------------------------------------------
# Step 1 — Normalise Newlines
# --------------------------------------------------
def normalise_newlines(text: str) -> str:
    """
    Standardise newline characters to '\n'.
    """
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    return text

# --------------------------------------------------
# Step 2 — Remove De-identification Tokens
# --------------------------------------------------
def remove_deid_tokens(text: str) -> str:
    """
    Remove MIMIC de-identification tokens of the form [** ... **].
    """
    return DEID_PATTERN.sub("", text)

# --------------------------------------------------
# Step 3 — Normalise Whitespace
# --------------------------------------------------
def normalise_whitespace(text: str) -> str:
    """
    Collapse multiple spaces or tabs into a single space.
    Does not remove newline characters.
    """
    return MULTISPACE_PATTERN.sub(" ", text)

# --------------------------------------------------
# Step 4 — Remove EMR Trailing References
# --------------------------------------------------
def remove_emr_references(text: str) -> str:
    """
    Remove EMR trailing artefacts that start with the 'References' header
    and include all subsequent lines. Typically contains JavaScript popups
    and is located at the end of the document.
    """
    # Split by lines, find first occurrence of 'References' at line start
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.strip().lower() == "references":
            # Return text up to but not including the 'References' line
            return "\n".join(lines[:i]).rstrip()
    return text

# --------------------------------------------------
# Main Preprocessing Pipeline
# --------------------------------------------------
def preprocess_note(text: str) -> str:
    """
    Apply preprocessing pipeline to a single ICU note.

    Steps:
    1. Normalise newline characters
    2. Remove de-identification tokens
    3. Normalise excessive whitespace
    4. Remove trailing EMR References blocks
    """

    if text is None:
        return ""

    text = str(text)
    text = normalise_newlines(text)
    text = remove_deid_tokens(text)
    text = normalise_whitespace(text)
    text = remove_emr_references(text)

    return text.strip()