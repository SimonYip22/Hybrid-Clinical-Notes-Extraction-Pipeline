# Phase 1 — Corpus Profiling Methodological Decisions

## Objective

This document defines the methodological decisions governing execution of Phase 1 (Corpus Profiling).

It specifies:

- Dataset boundaries used during profiling
- Sampling limits for manual inspection
- Structural analysis constraints
- Explicit exclusions within this phase

This file formalises how Phase 1 is conducted.

It does not:

- Present empirical findings (see `corpus_profiling.ipynb`)
- Interpret structural implications (see `summary.md`)
- Define design choices for later phases


	•	Phase 1 — Sample Inspection & Profiling
	•	Small subset of notes (~200)
	•	Manual review for structure, sections, abbreviations, formatting
	•	Output: descriptive insights to guide extraction logic

---

## 1. Dataset Scope and Size Constraints

### 1.1 Working Corpus Size

**Decision:**  
The working corpus is capped at approximately 200 ICU progress notes.

**Rationale:**  
- Sufficient for structural pattern discovery  
- Sufficient for deterministic rule development  
- Sufficient for small-scale supervised validation  
- Avoids unnecessary scale inflation  
- Aligns with project scope (design demonstration, not population inference)

This project prioritises architectural clarity over dataset magnitude.

---

### 1.2 Inclusion Criteria

**Decision:**  
- Only ICU progress notes meeting predefined selection criteria are included.
- Detailed filtering logic is documented in the profiling notebook.

**Constraint:**  
- Corpus composition is frozen for the duration of Phase 1.

---

## 2. Quantitative Profiling Strategy

### 2.1 Full-Corpus Analysis

**Decision:**  
Quantitative profiling is performed on the full working corpus (n ≈ 200).

**Constraint:**  
- No subsampling is used for distributional measurements.

**Rationale:**  
- Computationally inexpensive  
- Ensures complete structural visibility  
- Eliminates sampling bias in distributional measurements  

---

## 3. Manual Inspection Strategy

### 3.1 Manual Inspection Cap

**Decision:**  
Manual inspection is capped at approximately 40–50 notes.

**Rationale:**  
- Structural pattern saturation expected within this range  
- Manual review beyond this threshold yields diminishing returns  
- Prevents disproportionate allocation of effort to qualitative review  

---

### 3.2 Sampling Method

**Decision:**  
- Manual inspection uses stratified sampling to capture variability in length and structure.
- Exact stratification parameters are determined after corpus loading.

**Rationale:**  
- Ensures coverage of short, medium, and long notes  
- Captures variation in section density and formatting  
- Reduces risk of overfitting rule design to homogeneous samples  

Specific sampling parameters will be finalised after corpus access.

---








---

## 4. Analytical Boundaries Within Phase 1

Phase 1 is restricted to structural and surface-level linguistic analysis.

The following are explicitly excluded:

- Entity extraction
- Rule drafting
- Schema design
- Negation implementation
- Model experimentation
- Annotation

These activities belong to subsequent phases.

---

## 5. Phase 1 Execution Freeze

At the conclusion of Phase 1:

- Corpus size remains fixed.
- Quantitative profiling has been performed on all notes.
- Manual inspection has been completed within defined limits.
- No rule construction or modelling has been initiated.

Any deviation from these constraints requires formal revision of this document.

1. Phase 1 — Working Corpus (n ≈ 200)
	•	Purpose: rapid iteration, debugging, and rule validation.
	•	Selection: stratified sampling across ICU types, note authors, and lengths to capture variability.
	•	Activities:
	•	Test preprocessing pipeline (JSON conversion, time window filtering, CATEGORY restrictions, ISERROR removal).
	•	Identify structural patterns for extraction rules.
	•	Validate metadata joins (PATIENTS → ICUSTAYS → NOTEEVENTS).
	•	Outcome: reliable rules and deterministic scripts ready to scale.

















# Dataset Decisions — ICU Early Note Corpus (MIMIC-III)

## 1. Purpose

This document specifies the exact cohort definition, filtering logic, and design decisions used to construct the ICU early-note corpus from MIMIC-III.

It serves as the formal reproducibility and audit specification for `build_corpus.py`.

All logic described here reflects the final, validated implementation.

---

# 2. Data Sources and Required Columns

Only minimal required columns were loaded to reduce memory usage and prevent accidental feature leakage.

## 2.1 PATIENTS

| Column       | Purpose                          |
|-------------|----------------------------------|
| SUBJECT_ID  | Patient-level join key           |
| DOB         | Age calculation                  |
| GENDER      | Demographic metadata             |

---

## 2.2 ICUSTAYS

| Column         | Purpose |
|---------------|---------|
| SUBJECT_ID     | Join to PATIENTS and NOTEEVENTS |
| HADM_ID        | Admission-level join key for notes |
| ICUSTAY_ID     | Unique ICU stay identifier |
| FIRST_CAREUNIT | ICU type filtering |
| INTIME         | ICU admission timestamp |
| OUTTIME        | ICU discharge timestamp |

ICUSTAY_ID is the cohort anchor.

Notes in MIMIC are linked at hospital admission level (HADM_ID), not ICU stay level.  
Therefore, SUBJECT_ID + HADM_ID are required to correctly link notes to ICU stays.

---

## 2.3 NOTEEVENTS

| Column       | Purpose |
|-------------|---------|
| SUBJECT_ID   | Join key |
| HADM_ID      | Join key |
| CHARTTIME    | Temporal filtering |
| CATEGORY     | Restrict to clinical progress notes |
| ISERROR      | Exclude corrupted notes |
| TEXT         | Primary NLP content |

---

# 3. Cohort Definition Logic

The ICU stay is the anchor. All filtering is defined relative to the ICU stay.

---

## 3.1 ICU Unit Restriction

Allowed ICU types:

- MICU  
- CCU  
- SICU  
- TSICU  
- CSRU  

NICU was excluded to restrict to adult critical care.

---

## 3.2 Minimum Length of Stay

LOS_HOURS = (OUTTIME − INTIME) converted to hours.

Constraint:

    LOS_HOURS ≥ 24

Reasoning:
- Very short stays often contain minimal documentation
- Reduces observational or administrative stays
- Ensures sufficient narrative content for NLP

---

## 3.3 Adult Cohort Definition

Age calculated as:

    AGE = INTIME.year − DOB.year

Constraint:

    AGE ≥ 18

Reasoning:
- Restricts to adult ICU population
- Removes paediatric/adolescent admissions

### Handling De-identified Ages

In MIMIC-III, patients older than 89 have shifted DOB values.

Rule:

    If AGE > 120 → set AGE = 90

Reasoning:
- Identifies implausible ages caused by de-identification
- Preserves inclusion of elderly patients without removing them
- Prevents artificial outliers

---

# 4. Note Filtering Logic

---

## 4.1 Allowed Note Categories

CATEGORY values were normalised (strip + lowercase).

Allowed:

- physician
- nursing
- nursing/other

Reasoning:
- These contain longitudinal progress documentation
- Excludes radiology, ECG, discharge summaries, administrative notes
- Focused on real-time clinical reasoning

---

## 4.2 Error Filtering

Rule:

    Keep rows where ISERROR != 1

Important:
Valid rows may contain ISERROR = 0 or NaN.

Initial filtering using ISERROR == 0 incorrectly removed all notes due to NaN values.

This was corrected to exclude only explicit errors.

---

## 4.3 ICU Cohort Enforcement

Notes were inner joined to valid ICU stays using:

    SUBJECT_ID + HADM_ID

Join type:

    how="inner"

Reasoning:
- Ensures notes belong to valid adult ICU stays
- Prevents inclusion of hospital notes unrelated to ICU admission

---

## 4.4 Early Time Window Restriction

Constraint:

    INTIME ≤ CHARTTIME ≤ INTIME + 24 hours

Rows with missing CHARTTIME were removed before comparison.

Reasoning:
- Restricts to early ICU documentation
- Aligns with early deterioration modelling
- Excludes pre-ICU and late-stay notes

TIME_WINDOW_HOURS = 24

---

# 5. Logging and Integrity Verification

Row counts were logged after each major filtering step.

Purpose:
- Detect unintended full-row elimination
- Verify cohort shrinkage is monotonic
- Ensure filtering behaves as expected

This step identified the ISERROR logic issue during development.

All final counts below reflect corrected logic.

---

# 6. Final Cohort Statistics

Initial:

- NOTES: 2,083,180  
- PATIENTS: 46,520  
- ICUSTAYS: 61,532  

After filtering:

- After unit filter: 53,432  
- After adult age filter: 45,278  
- After category filter: 1,187,677  
- After error filter: 1,186,960  
- After ICU merge: 888,224  
- After 24h window: 162,296  

Final corpus:

- Unique ICU stays: 32,910  
- Total notes: 162,296  
- Columns: 10  

---

# 7. Derived Cohort Properties

- ~72.7% of adult ICU stays contain ≥1 qualifying early note  
- ~18.3% of ICU-linked notes fall within first 24 hours  
- Mean notes per ICU stay ≈ 4.93  

These values are internally consistent and plausible for early ICU documentation.

---

# 8. Final Output Structure

Each row represents one ICU clinical note.

Columns:

- SUBJECT_ID  
- HADM_ID  
- ICUSTAY_ID  
- AGE  
- GENDER  
- FIRST_CAREUNIT  
- LOS_HOURS  
- CATEGORY  
- CHARTTIME  
- TEXT  

Saved to:

    data/processed/icu_corpus.csv

---

# 9. Post-Corpus Plan

The corpus is now frozen (162,296 notes across 32,910 ICU stays).  
Next phase: structured validation and schema design.

**Step 1 — Manual Exploratory Sample Analysis (~30 Notes)**  
Stratified ICU-stay–level sampling to:
- Inspect note structure and formatting
- Identify recurring sections
- Detect artefacts and de-identification tokens
- Inform JSON schema and extraction rules

This is qualitative structure discovery, not modelling.

**Step 2 — Quantitative Profiling (~300 notes)**  
Scripted analysis to measure:
- Section frequency
- Length distributions
- Formatting patterns
- Token characteristics

Used to empirically support schema design.

**Step 3 — Rule Validation (30–50 notes)**  
Deep manual comparison of:
- Raw text vs extracted features
- Section parsing correctness
- Regex precision and edge cases

Feature logic is refined here, then frozen and applied to the full corpus.

No train/test split is performed at this stage.  
This phase transitions from corpus construction to NLP feature engineering.