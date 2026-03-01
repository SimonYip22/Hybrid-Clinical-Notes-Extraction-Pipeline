"""
quant_profiling.py

Purpose:
    Quantitatively profile structural characteristics of a 500-note ICU sample
    to validate the feasibility of deterministic, rule-based information extraction.

Workflow:
    1. Load the processed ICU corpus from INPUT_PATH.
    2. Randomly sample 500 notes using a fixed random seed for reproducibility.
    3. Compute per-note structural metrics:
        - Colon-terminated header frequency
        - Uppercase block header frequency
        - Standalone numeric token count
        - Blood pressure pattern count
        - De-identification token count
        - Character length
        - Token count (whitespace split)
        - Line count
    4. Save per-note metrics to CSV.
    5. Compute summary statistics (mean, std, min, max, quartiles).
    6. Compute percentage of notes with non-zero occurrences per metric.
    7. Save aggregated summary statistics to CSV.

Outputs:
    profiling_sample_500.csv      — Sampled note subset.
    profiling_per_note.csv        — One row per note with computed metrics.
    profiling_summary.csv         — Aggregated descriptive statistics.
"""

import pandas as pd
import re
from collections import Counter

# ----------------------------
# Configuration
# ----------------------------

# Input path for the full ICU corpus and sampled subset
INPUT_PATH = "data/processed/icu_corpus.csv"
SAMPLE_PATH = "data/sample/profiling_sample_500.csv"

# Output paths for profiling results
PER_NOTE_OUTPUT = "data/sample/profiling_per_note.csv"
SUMMARY_OUTPUT = "data/sample/profiling_summary.csv"

# Sampling parameters
SAMPLE_SIZE = 500
RANDOM_STATE = 42

# ----------------------------
# Metric Functions
# ----------------------------

def count_colon_headers(text):
    """
    Counts colon-terminated headers at line start.
    Example: 'Assessment:', 'NEURO:', 'Plan:'
    """
    # Starts with uppercase, followed by letters/spaces, ending with colon
    pattern = r"^[A-Z][A-Za-z\s]+:"
    # Use MULTILINE to apply ^ to each line, not just the first line of the text
    return len(re.findall(pattern, text, flags=re.MULTILINE))


def count_uppercase_blocks(text):
    """
    Counts fully uppercase section headers.
    More restrictive than colon headers, as some notes use only uppercase.
    Example: 'NEURO:', 'CV:', 'RESP:'
    """
    # Starts with uppercase letters or spaces, at least 3 characters, ending with colon
    pattern = r"^[A-Z\s]{3,}:"
    # Use MULTILINE to apply ^ to each line
    return len(re.findall(pattern, text, flags=re.MULTILINE))


def count_numeric_tokens(text):
    """
    Counts standalone numeric tokens including decimals.
    Example: '120', '98.6', but not 'HR98' or 'BP120/80'
    """
    # Word boundary, one or more digits, optional decimal part, word boundary
    pattern = r"\b\d+(\.\d+)?\b"
    # Use re.findall to get all matches and count them
    return len(re.findall(pattern, text))


def count_bp_patterns(text):
    """
    Counts structured blood pressure style patterns.
    Example: '120/80', '130/85', '150/100'
    """
    # 2-3 digits, slash, 2-3 digits, word boundary
    pattern = r"\b\d{2,3}/\d{2,3}\b"
    # Use re.findall to get all matches and count them
    return len(re.findall(pattern, text))


def count_deid_tokens(text):
    """
    Counts MIMIC de-identification tokens.
    ? makes it non-greedy to avoid counting across multiple tokens in one match.
    Example: [** Name **]
    """
    # '[' followed by '**', then any characters (non-greedy), then '**' followed by ']' 
    pattern = r"\[\*\*.*?\*\*\]"
    # Use re.findall to get all matches and count them
    return len(re.findall(pattern, text))


# ----------------------------
# Main Profiling Logic
# ----------------------------

def main():

    # Load corpus
    df = pd.read_csv(INPUT_PATH)

    # Sample 500 notes
    sample_df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    sample_df.to_csv(SAMPLE_PATH, index=False)

    # Metrics list to hold per-note results
    metrics = []

    # Compute metrics for each note
    for _, row in sample_df.iterrows():

        # Ensure text is a string (in case of missing values or other types)
        text = str(row["TEXT"])

        # Append these 8 metrics for this note as a dictionary
        metrics.append({
            # Structural metrics
            "colon_header_count": count_colon_headers(text),
            "uppercase_header_count": count_uppercase_blocks(text),
            "numeric_token_count": count_numeric_tokens(text),
            "bp_pattern_count": count_bp_patterns(text),
            "deid_token_count": count_deid_tokens(text),

            # Additional basic metrics
            "char_length": len(text),
            "token_count": len(text.split()),
            "line_count": text.count("\n") + 1
        })

    # Convert metrics list of 500 dictionaries into 500-row DataFrame where each column is a metric and each row is a note
    profile_df = pd.DataFrame(metrics)

    # ----------------------------
    # Save profiling results
    # ----------------------------

    # Save per-note metrics
    profile_df.to_csv(PER_NOTE_OUTPUT, index=False)

    # Aggregate summary statistics
    # Describe gives count, mean, std, min, 25%, 50%, 75%, max for each metric column. 
    # Transpose flips it to have metrics as rows and stats as columns.
    summary = profile_df.describe().transpose()

    # Add proportion metrics column for each metric
    summary["percent_notes_nonzero"] = (
        # Calculate the percentage of notes that have a non-zero value for each metric
        (profile_df > 0).sum() / len(profile_df) * 100
    )

    # Save summary statistics
    summary.to_csv(SUMMARY_OUTPUT)

    print("Quantitative profiling completed. 500 note sample, per-note metrics, and summary statistics saved to CSV.")

# Run the main function
if __name__ == "__main__":
    main()
