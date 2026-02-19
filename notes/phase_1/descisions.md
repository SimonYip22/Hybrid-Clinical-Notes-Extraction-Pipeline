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

















# Phase 1 — Minimal Required Columns and Preprocessing Decisions

---

## 1. Required Tables and Columns

### PATIENTS
- **SUBJECT_ID** → unique patient identifier (to join with notes and ICU stays)
- **DOB** → calculate age at ICU admission
- **GENDER** → optional metadata for JSON

### ICUSTAYS
- **SUBJECT_ID** → join to patient
- **HADM_ID** → technically only needed to join to NOTEEVENTS, because that table has both SUBJECT_ID and HADM_ID
- **ICUSTAY_ID** → essential; identifies the ICU stay you want to extract notes for
- **FIRST_CAREUNIT** → optional metadata for JSON (ICU type, e.g., MICU, SICU)
- **INTIME / OUTTIME** → calculate ICU length of stay (LOS) in hours

> **Note:** ICUSTAY_ID is crucial. HADM_ID ensures you correctly match notes to the hospital admission associated with the ICU stay. Notes in MIMIC are recorded per hospital admission (HADM_ID), not per ICU stay, so HADM_ID + SUBJECT_ID ensures the correct ICU’s notes are pulled.

### NOTEEVENTS
- **SUBJECT_ID + HADM_ID** → required to join to ICU stays correctly
- **CHARTDATE / CHARTTIME / STORETIME** → required to filter notes within first 24–48 hours of ICU admission
- **CATEGORY** → restrict to physician/nursing notes
- **DESCRIPTION** → sometimes used to identify “Progress Note” or “Nursing Note,” but optional
- **ISERROR** → flag for corrupted or erroneous rows (1 = invalid); filter these out
- **TEXT** → main NLP content

---

## 2. Minimal Required Columns Summary

| Table       | Required Columns                             | Purpose                                             |
|------------|----------------------------------------------|---------------------------------------------------|
| PATIENTS   | SUBJECT_ID, DOB                              | Join & calculate age                               |
| ICUSTAYS   | SUBJECT_ID, ICUSTAY_ID, HADM_ID, FIRST_CAREUNIT, INTIME, OUTTIME | Identify ICU stay, filter ICU type, calculate LOS, define note window |
| NOTEEVENTS | SUBJECT_ID, HADM_ID, CHARTTIME, CATEGORY, ISERROR, TEXT | Extract ICU notes in first X hours, exclude errors, main text |

---

## 3. Filtering / Scope Rules

### A. Adult Patients
- Age ≥ 18 at ICU admission → `(INTIME - DOB) >= 18 years`
- Excludes pediatric patients (standard cohort)

### B. ICU Filtering
- No ICU ID exists in NOTEEVENTS → use HADM_ID + ICUSTAY INTIME/OUTTIME to select notes for that ICU stay
- Ensures notes belong to the correct hospital admission
- HADM_ID is needed because patients may have multiple admissions

### C. Hospital / ICU Length of Stay
- Filter ICU stays ≥ 48 hours
- Reason: short stays often have limited notes for NLP extraction
- Optional: reduce to 24 hours for smaller scope; still sufficient for ~200 notes

### D. Notes Within First X Hours
- Include notes where:  
  `(NOTE CHARTTIME - ICUSTAY INTIME) <= 24 or 48 hours`
- 24 hours → smaller, concise, early signals  
- 48 hours → more notes, slightly more preprocessing

### E. Note Type / Author
- Include only:
  - Physician notes: `CATEGORY='Physician'` or `DESCRIPTION='Progress Note'`
  - Nursing notes: `CATEGORY='Nursing'` or `DESCRIPTION='Nursing/Other'`
- Options: physician only, nurse only, or both (richer context)
- Exclude: radiology, discharge summaries (or use only for validation)

### F. Exclude Errors
- `ISERROR=1` → remove
- Ensures clean text

---

## 4. Joining Logic for Preprocessing

1. Join `PATIENTS → ICUSTAYS` via `SUBJECT_ID` → get age, sex, ICU LOS
2. Filter `ICUSTAYS` by age ≥ 18 and LOS ≥ 24–48 hours
3. Join `ICUSTAYS → NOTEEVENTS` via `SUBJECT_ID + HADM_ID`
4. Filter `NOTEEVENTS` by:
   - Notes within first 24–48 hours of ICU admission
   - CATEGORY / DESCRIPTION (physician/nursing)
   - ISERROR=0

> **Result:** curated set of early ICU clinical notes with metadata (age, sex, ICU type) ready for NLP extraction

---

## 5. Constraint Table

| Constraint       | Recommendation        | Reason                                                         |
|-----------------|---------------------|---------------------------------------------------------------|
| Age              | ≥ 18                | Standard adult ICU cohort                                     |
| ICU LOS          | ≥ 24 hours          | Sufficient content; reduces trivial / short stays            |
| Notes time window| 24 hours            | Smaller, high-quality corpus; manageable preprocessing       |
| Note authors     | Physician + Nurse   | Richer clinical signals; can filter to one type if simpler   |
| Note category    | Progress / Nursing  | Exclude radiology, discharge, lab-only comments              |
| Exclude errors   | ISERROR=1           | Clean text only                                               |

---

## 6. Expected Outcome
- ~200–300 notes selected → matches NLP pipeline requirements  
- Notes are early, focused, manageable for a 4-week project  
- Metadata allows cohort description and basic filtering without “cheating” — all clinical content still comes from free text