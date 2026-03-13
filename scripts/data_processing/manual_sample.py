"""
manual_sample.py

Purpose:
    Generate a small random sample of ICU progress notes from the processed corpus
    for structured manual inspection during Phase 1.

Rationale:
    This sample is used exclusively for qualitative structural analysis prior to
    rule design and schema implementation. The objective is to:

        - Inspect note formatting and section header patterns
        - Identify structural variability across notes
        - Detect de-identification artefacts
        - Observe recurring linguistic patterns
        - Inform deterministic rule construction
        - Support JSON schema design decisions

Design Notes:
    - Sampling is performed at the note level because extraction in this project
      operates independently per progress note.
    - Full metadata is retained to preserve contextual information that may
      influence structural patterns (e.g., care unit, note category, LOS).

Reproducibility:
    A fixed random_state ensures deterministic sampling for documentation
    and repeatability.

Output:
    data/sample/manual_sample_30.csv
"""

import pandas as pd

df = pd.read_csv("data/processed/icu_corpus.csv")

sample_30 = df.sample(n=30, random_state=42)

sample_30.to_csv("data/sample/manual_sample_30.csv", index=False)

print("Manual sample of 30 notes generated and saved to data/sample/manual_sample_30.csv")