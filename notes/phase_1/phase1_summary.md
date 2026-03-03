# Phase 1 — Corpus Profiling Summary

## 1. Purpose of This Document

This file provides a concise, outcome-focused summary of Phase 1 (Corpus Construction and Structural Profiling).

It synthesizes:

- Cohort construction logic and final corpus properties
- Manual structural findings
- Quantitative profiling results
- Boundary stress-testing outcomes
- Architectural implications for Phase 2

Where `decisions.md` contains full methodological detail, rationale, and stepwise justification, this file records the consolidated results and confirmed conclusions of Phase 1.

---

# 2. Phase 1 Objective

Phase 1 evaluates whether the adult ICU early-note corpus contains sufficiently consistent structural patterns to support deterministic, rule-based extraction in Phase 2.

The objective is structural feasibility validation, not extraction implementation.

Phase 1 confirms:

- Section delimiters are recurrent and predictable.
- Numeric content is dense and regex-accessible.
- De-identification and EMR artefacts are systematic.
- Structural variability is bounded.
- No boundary cases invalidate deterministic parsing assumptions.

Phase 1 does not implement:
- Section segmentation code
- JSON schema construction
- Entity extraction
- Transformer modelling

Its sole purpose is architectural validation.

---

# 3. Corpus Construction

The adult ICU early-note corpus was constructed using deterministic filtering logic with `build_corpus.py`.

## 3.1 Cohort Definition

Data Sources and Columns:

- **PATIENTS:** `SUBJECT_ID`, `DOB`, `GENDER`  
- **ICUSTAYS:** `SUBJECT_ID`, `HADM_ID`, `ICUSTAY_ID`, `FIRST_CAREUNIT`, `INTIME`, `OUTTIME`  
- **NOTEEVENTS:** `SUBJECT_ID`, `HADM_ID`, `CHARTTIME`, `CATEGORY`, `ISERROR`, `TEXT`

Adult ICU early notes were constructed using:

- ICU stay anchor: `ICUSTAY_ID`
- Adult filter: `AGE ≥ 18`
- Minimum LOS: `LOS_HOURS ≥ 24`
- ICU units: MICU, CCU, SICU, TSICU, CSRU
- Allowed note categories:
  - physician
  - nursing
  - nursing/other
- Error exclusion: `ISERROR != 1`
- Early window: `INTIME ≤ CHARTTIME ≤ INTIME + 24h`

## 3.2 Final Corpus Statistics

Final frozen corpus:

- Unique ICU stays: 32,910
- Total notes: 162,296
- Mean notes per ICU stay: ~4.93
- ~72.7% of adult ICU stays contain ≥1 early qualifying note

Output file: `data/processed/icu_corpus.csv` contains filtered notes with necessary columns for profiling and downstream extraction.

The corpus is fixed and reproducible.

---

# 4. Phase 1 Structural Validation Workflow

Phase 1 consisted of three sequential validation layers.

---

## 4.1 Manual Structural Inspection (n = 30)

Files:

- Script: `manual_sample.py`
- Output: `data/sample/manual_sample_30.csv`

### Structural Archetypes Identified

Four dominant structural families:

1. Physician ICU notes (templated, section-rich, long-form)
2. Nursing system-based notes (NEURO, CV, RESP patterns)
3. Procedural / respiratory notes (short, intervention-dense)
4. Specialty ICU notes (hemodynamic and lab-heavy)

### Header Regularity

Observed header formats:

- Colon-terminated (`Assessment:`)
- Uppercase system blocks (`NEURO:`)
- Title-style headers
- Numbered problem lists (`1. ...`)

Headers were structured and recurrent.

### Numeric and Flowsheet Density

- Labs and vitals appear in narrative, inline, and stacked formats.
- BP-style expressions are common.
- ICU notes contain dense numeric blocks consistent with flowsheet imports.

### Artefacts

Observed artefacts:

- `[** ... **]` de-identification tokens
- Protected sections
- EMR-generated reference blocks
- Occasional trailing JavaScript/link fragments

Artefacts were consistent and separable.

### Manual Conclusion

Macro-level structure is repetitive, templated, and deterministic-compatible.

---

## 4.2 Quantitative Structural Profiling (n = 500)

Files:

- Script: `quant_profiling.py`
- Output: `data/profiling/profiling_sample_500.csv`
- Output: `data/profiling/profiling_per_note.csv`
- Output: `data/profiling/profiling_summary.csv`

Sample size: 500 random notes (fixed seed = 42)

### Workflow

Metrics extracted per note:

- Colon-terminated header count
- Uppercase header count
- Numeric token count
- Blood pressure pattern count
- De-identification token count
- Character length
- Token count
- Line count

Summary statistics were calculated across the sample.

### Analysis Approach

Analysis was performed in two stages:

1. **Overall Summary Statistics**: Prevalence, median, and max values for each metric to confirm structural signals.
2. **Extreme Boundary Inspection**: Manual review of notes at metric extremes to confirm parser robustness and identify any structural failure modes.

---

### 4.2.1 Summary Statistics Analysis

#### Header Prevalence

| Metric | % Non-Zero | Median | Max |
|--------|------------|--------|-----|
| `colon_header_count` | 81.2% | 8 | 113 |
| `uppercase_header_count` | 51.0% | 1 | 35 |

Interpretation:

- Colon headers are structurally dominant.
- Uppercase headers are common but not universal.
- Distribution is right-skewed with templated ICU extremes.

Conclusion:
- Section-based deterministic segmentation is justified.


#### Numeric Density

| Metric | % Non-Zero | Median | Max |
|--------|------------|--------|-----|
| `numeric_token_count` | 93.6% | 17 | 432 |
| `bp_pattern_count` | 40.6% | 0 | 11 |

Interpretation:

- Numeric content is near-universal.
- Heavy right tail consistent with ICU flowsheet structure.
- BP patterns are conditional but frequent.

Conclusion:
- Regex-based numeric and physiologic extraction is feasible.



#### De-identification Artefacts

| Metric | % Non-Zero | Median | Max |
|--------|------------|--------|-----|
| `deid_token_count` | 78.8% | 3 | 66 |

Interpretation:

- De-identification is widespread and systematic.
- No malformed or nested redaction tokens observed.

Conclusion:
Preprocessing normalization is required but straightforward.



#### Structural Variability

| Metric | Median | Max |
|--------|--------|-----|
| `char_length` | 1505 | 17558 |
| `token_count` | 241 | 2557 |
| `line_count` | 26 | 456 |

Interpretation:

- Strong right-skew.
- Coexistence of short procedural notes and long ICU summaries.
- Variability is continuous, not chaotic.

Conclusion:
- Deterministic rules must scale across length extremes but variability does not undermine structure.


#### Key Findings Summary

- Colon headers present in ~81% of notes
- Uppercase headers present in ~51%
- Numeric tokens present in ~94%
- BP-style patterns present in ~41%
- De-identification artefacts present in ~79%
- Wide but continuous right-skewed size distribution

Structural signals were prevalent, dense, and stable.

---

### 4.2.2 Extreme Boundary Inspection (45 Unique Notes)

Notebook:`profiling_boundary_extremes.ipynb`

#### Workflow

To stress-test structural assumptions, extreme cases were manually reviewed:

- Top and bottom 5 per functional metric
- Top and bottom 2 by character length
- Deduplicated to 45 unique boundary notes

Extreme inspection:

- 42 functional extremes
- 4 length extremes
- 45 unique notes total


#### Extreme Findings

Extreme cases included:

- Highly templated ICU admission summaries (60–110+ headers)
- Numeric-saturated lab panels (150–400+ numeric tokens)
- Ultra-short micro-notes
- Extremely long layered notes (>15,000 characters)
- EMR reference blocks and trailing JavaScript/link artefacts

Across maximal and minimal cases:

- No malformed colon headers.
- No delimiter collisions.
- No numeric tokenization corruption.
- No broken BP-style expressions.
- No nested or truncated `[** ... **]` artefacts.
- EMR reference and JavaScript/link fragments are consistent and patternable.
- No hybrid cases invalidate segmentation logic.

Conclusion:

- No structural failure modes were identified at corpus boundaries.

---

## 4.3 Structural Feasibility Synthesis

Across manual inspection, quantitative profiling, and boundary stress-testing:

1. Colon-based segmentation is robust.
2. Uppercase headers act as reinforcement but are not mandatory.
3. Numeric density supports deterministic candidate generation.
4. BP-style regex extraction is feasible.
5. De-identification and EMR artefacts are systematic and removable.
6. Structural heterogeneity is bounded.
7. No brittleness observed at extremes.

Phase 1 provides positive structural evidence for deterministic rule-based extraction.

---

# 5. Scope and Limitations

Phase 1 does not implement extraction logic, JSON schema, or NLP-based entity generation. Its output provides evidence that deterministic rule-based extraction is structurally justified and informs:

- Header and section segmentation rules
- Numeric and physiologic pattern extraction feasibility
- Artefact normalisation requirements
- Scope for robust JSON schema design

Completion of Phase 1 ensures that Phase 2 (rule-based deterministic extraction) can proceed with high confidence in architectural validity and minimal risk of structural failure.

---

# 6. Deliverables

## Code

`src/data_processing/`:
- `build_corpus.py` — deterministic corpus construction
- `manual_sample.py` — qualitative structural sampling
- `quant_profiling.py` — quantitative structural profiling

`notes/phase_1/`:
- `profiling_boundary_extremes.ipynb` — extreme-case inspection

## Data Outputs:

`data/processed/`:
- `icu_corpus.csv`
`data/sample/`:
- `manual_sample_30.csv`
- `profiling_sample_500.csv`
- `profiling_per_note.csv`
- `profiling_summary.csv`

## Documentation

`notes/phase_1/`:
- `phase1_decisions.md` — full methodological and analytical record
- `phase1_summary.md` — structured Phase 1 executive summary

---

# 9. Phase 1 Final Conclusion

Phase 1 confirms that:

- The adult ICU early-note corpus is structurally regular.
- Section delimiters are prevalent and stable.
- Numeric content is dense and extractable.
- Artefacts are systematic and non-destructive.
- Boundary cases do not invalidate deterministic assumptions.

No structural evidence contradicts rule-based extraction.

Phase 2 (deterministic segmentation and extraction implementation) is methodologically justified and can proceed with high architectural confidence.

---