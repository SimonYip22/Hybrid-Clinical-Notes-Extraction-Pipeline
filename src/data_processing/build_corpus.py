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

# Calculate age at ICU admission
icustays["AGE"] = (
    (icustays["INTIME"] - icustays["DOB"])
    .dt.days / 365.25
)

# Keep only adult ICU stays
icustays = icustays[icustays["AGE"] >= 18]

# --------------------------------
# Note Filtering (Before Cohort Join)
# --------------------------------

# Normalise category values
notes["CATEGORY"] = notes["CATEGORY"].str.strip().str.lower()

# Keep only allowed note categories
notes_filtered = notes[notes["CATEGORY"].isin(ALLOWED_CATEGORIES)]

# Remove notes marked as errors
notes_filtered = notes_filtered[notes_filtered["ISERROR"] == 0]

# --------------------------------
# Enforce ICU Cohort Membership
# --------------------------------

# Inner join keeps only notes belonging to valid adult ICU stays
notes_filtered = notes_filtered.merge(
    icustays,
    on=["SUBJECT_ID", "HADM_ID"],
    how="inner"
)

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


# Final columns kept
corpus = notes_filtered[[
    "SUBJECT_ID",
    "HADM_ID",
    "ICUSTAY_ID",
    "AGE",
    "GENDER",
    "FIRST_CAREUNIT",
    "LOS_HOURS",
    "CATEGORY",
    "CHARTTIME",
    "TEXT"
]]