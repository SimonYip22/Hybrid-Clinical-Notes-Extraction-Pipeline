"""
build_corpus.py

Purpose
-------
Constructs a filtered, note-level ICU text corpus from MIMIC-III raw tables.
The script defines a reproducible adult ICU cohort and extracts early
clinical notes for downstream NLP or modelling tasks.

Workflow
--------
1. Load required columns from:
   - NOTEEVENTS.csv
   - ICUSTAYS.csv
   - PATIENTS.csv

2. Define ICU cohort:
   - Restrict to selected ICU care units (remove NICU)
   - Enforce minimum length of stay (>= 24 hours)
   - Merge patient demographics
   - Compute age at ICU admission
   - Exclude patients < 18 years
   - Cap implausible ages (>120) at 90 to handle MIMIC de-identification

3. Filter notes:
   - Keep physician and nursing categories only
   - Exclude notes where ISERROR == 1
   - Restrict to notes belonging to valid ICU stays (based on subject_id and hadm_id)
   - Keep notes written within first 24 hours of ICU admission

4. Output:
   - Note-level corpus saved to: data/processed/icu_corpus.csv
   - Corpus shape (162296, 10)

Output Structure
----------------
Each row represents a single ICU clinical note.
Columns:
    SUBJECT_ID
    HADM_ID
    ICUSTAY_ID
    AGE
    GENDER
    FIRST_CAREUNIT
    LOS_HOURS
    CATEGORY
    CHARTTIME
    TEXT

Design Decisions
----------------
- Cohort defined at ICU-stay level; output is note-level.
- Early prediction window fixed at 24 hours.
- Length-of-stay threshold prevents inclusion of short observational stays.
- ISERROR filtering excludes only rows explicitly marked as errors.
- Row counts are logged at each stage to ensure cohort integrity and
  prevent unintended data loss during filtering.

Intended Use
------------
This corpus is designed for downstream NLP feature extraction.

"""
# --------------------------------
# Load Data
# --------------------------------

import pandas as pd

# Load only required columns from raw datasets
notes = pd.read_csv(
    "data/raw/NOTEEVENTS.csv",
    usecols=[
        "SUBJECT_ID",
        "HADM_ID",
        "CHARTTIME",
        "CATEGORY",
        "ISERROR",
        "TEXT"
    ],
    low_memory=False
)

icustays = pd.read_csv(
    "data/raw/ICUSTAYS.csv",
    usecols=[
        "SUBJECT_ID",
        "HADM_ID",
        "ICUSTAY_ID",
        "FIRST_CAREUNIT",
        "INTIME",
        "OUTTIME"
    ]
)

patients = pd.read_csv(
    "data/raw/PATIENTS.csv",
    usecols=["SUBJECT_ID", "DOB", "GENDER"]
)

print("Initial ICUSTAYS:", len(icustays))
print("Initial NOTES:", len(notes))
print("Initial PATIENTS:", len(patients))

# --------------------------------
# Configuration (Frozen Design Decisions)
# --------------------------------

ALLOWED_CATEGORIES = {
    "physician",
    "nursing",
    "nursing/other"
}

ALLOWED_UNITS = {
    "MICU",
    "CCU",
    "SICU",
    "TSICU",
    "CSRU"
}

MIN_LOS_HOURS = 24
TIME_WINDOW_HOURS = 24

# --------------------------------
# ICU Stay Filtering (Define Cohort)
# --------------------------------

# Convert ICU time columns to datetime
icustays["INTIME"] = pd.to_datetime(icustays["INTIME"])
icustays["OUTTIME"] = pd.to_datetime(icustays["OUTTIME"])

# Restrict to adult ICU units
icustays = icustays[icustays["FIRST_CAREUNIT"].isin(ALLOWED_UNITS)]

print("After unit filter:", len(icustays))

# Calculate ICU length of stay in hours
icustays["LOS_HOURS"] = (
    (icustays["OUTTIME"] - icustays["INTIME"])
    .dt.total_seconds() / 3600
)

# Keep ICU stays meeting minimum LOS threshold
icustays = icustays[icustays["LOS_HOURS"] >= MIN_LOS_HOURS]

# --------------------------------
# Attach Patient Demographics & Filter Adults
# --------------------------------

# Convert DOB to datetime
patients["DOB"] = pd.to_datetime(patients["DOB"])

# Attach DOB and GENDER to ICU stays via SUBJECT_ID
icustays = icustays.merge(
    patients[["SUBJECT_ID", "DOB", "GENDER"]],
    on="SUBJECT_ID",
    how="inner"
)

# Filter out invalid DOBs
icustays = icustays[icustays["DOB"].notna()]

# Calculate age at ICU admission
icustays["AGE"] = (
    icustays["INTIME"].dt.year - icustays["DOB"].dt.year
)

# Cap implausible ages (MIMIC anonymisation)
# Any patient >89 has their DOB scrambled, so we treat any implausible age as a patient who was de-identified and >89
icustays.loc[icustays["AGE"] > 120, "AGE"] = 90

# Keep only adult ICU stays
icustays = icustays[icustays["AGE"] >= 18]

print("After adult age filter:", len(icustays))

# --------------------------------
# Note Filtering (Before Cohort Join)
# --------------------------------

# Normalise category values
notes["CATEGORY"] = notes["CATEGORY"].str.strip().str.lower()

# Keep only allowed note categories
notes_filtered = notes[notes["CATEGORY"].isin(ALLOWED_CATEGORIES)]

print("After category filter:", len(notes_filtered))

# Remove notes marked as errors
notes_filtered = notes_filtered[notes_filtered["ISERROR"] != 1]

print("After error filter:", len(notes_filtered))

# --------------------------------
# Enforce ICU Cohort Membership
# --------------------------------

# Inner join keeps only notes belonging to valid adult ICU stays
notes_filtered = notes_filtered.merge(
    icustays,
    on=["SUBJECT_ID", "HADM_ID"],
    how="inner"
)

print("After ICU merge:", len(notes_filtered))

# --------------------------------
# Time Window Restriction (Early ICU Notes)
# --------------------------------

# Convert CHARTTIME to datetime
notes_filtered["CHARTTIME"] = pd.to_datetime(notes_filtered["CHARTTIME"])

# Remove rows with missing timestamps to avoid comparison errors
notes_filtered = notes_filtered.dropna(subset=["CHARTTIME", "INTIME"])

# Keep notes written within first TIME_WINDOW_HOURS of ICU admission
notes_filtered = notes_filtered[
    (notes_filtered["CHARTTIME"] >= notes_filtered["INTIME"]) &
    (
        notes_filtered["CHARTTIME"]
        <= notes_filtered["INTIME"] + pd.Timedelta(hours=TIME_WINDOW_HOURS)
    )
]

print("After 24h time window:", len(notes_filtered))

# --------------------------------
# Final Corpus Projection
# --------------------------------

corpus = notes_filtered[
    [
        "SUBJECT_ID",
        "HADM_ID",
        "ICUSTAY_ID",
        "AGE",
        "GENDER",
        "FIRST_CAREUNIT",
        "LOS_HOURS",
        "CATEGORY",
        "CHARTTIME",
        "TEXT",
    ]
]

# Print the number of ICU stays, length of rows, and number of feature columns
print("Unique ICU stays:", corpus["ICUSTAY_ID"].nunique())
print("Number of reports:", (corpus.shape[0]))
print("Number of columns:", (corpus.shape[1]))

# Save output
corpus.to_csv("data/processed/icu_corpus.csv", index=False)