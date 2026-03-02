# Phase 1 — Corpus Profiling Methodological Decisions and Analysis

## Objective

This document defines the methodological decisions governing execution of Phase 1 (Corpus Profiling) as well as the analytical outcomes derived from quantitative profiling.

It specifies:

- Dataset boundaries used during profiling
- Sampling limits for manual inspection
- Structural analysis constraints
- Explicit exclusions within this phase
- Quantitative profiling metrics and their interpretation

This file formalises how Phase 1 is conducted.

---

## Adult ICU Early Report Corpus Construction

### 1. Purpose

- This document specifies the exact cohort definition, filtering logic, and design decisions used to construct the ICU early-note corpus from MIMIC-III.
- It serves as the formal reproducibility and audit specification for `build_corpus.py`.
- All logic described here reflects the final, validated implementation.

---

### 2. Data Sources and Required Columns

Only minimal required columns were loaded to reduce memory usage and prevent accidental feature leakage.

#### 2.1 PATIENTS

| Column       | Purpose                          |
|-------------|----------------------------------|
| `SUBJECT_ID`  | Patient-level join key           |
| `DOB`         | Age calculation                  |
| `GENDER`      | Demographic metadata             |

---

#### 2.2 ICUSTAYS

| Column         | Purpose |
|---------------|---------|
| `SUBJECT_ID`     | Join to `PATIENTS` and `NOTEEVENTS` |
| `HADM_ID`        | Admission-level join key for notes |
| `ICUSTAY_ID`     | Unique ICU stay identifier |
| `FIRST_CAREUNIT` | ICU type filtering |
| `INTIME`         | ICU admission timestamp |
| `OUTTIME`        | ICU discharge timestamp |

`ICUSTAY_ID` is the cohort anchor.

Notes in MIMIC are linked at hospital admission level (`HADM_ID`), not ICU stay level.  
Therefore, `SUBJECT_ID` + `HADM_ID` are required to correctly link notes to ICU stays.

---

#### 2.3 NOTEEVENTS

| Column       | Purpose |
|-------------|---------|
| `SUBJECT_ID`   | Join key |
| `HADM_ID`      | Join key |
| `CHARTTIME`    | Temporal filtering |
| `CATEGORY`     | Restrict to clinical progress notes |
| `ISERROR`      | Exclude corrupted notes |
| `TEXT`         | Primary NLP content |

---

### 3. Cohort Definition Logic

The ICU stay is the anchor. All filtering is defined relative to the ICU stay.

---

#### 3.1 ICU Unit Restriction

Allowed ICU types:

- `MICU`  
- `CCU`  
- `SICU`  
- `TSICU`  
- `CSRU`  

`NICU` was excluded to restrict to adult critical care.

---

#### 3.2 Minimum Length of Stay

`LOS_HOURS` = (`OUTTIME` − `INTIME`) converted to hours.

Constraint:

`LOS_HOURS ≥ 24`

Reasoning:
- Very short stays often contain minimal documentation
- Reduces observational or administrative stays
- Ensures sufficient narrative content for NLP

---

#### 3.3 Adult Cohort Definition

Age calculated as:

`AGE = INTIME.year − DOB.year`

Constraint:

`AGE ≥ 18`

Reasoning:
- Restricts to adult ICU population
- Removes paediatric/adolescent admissions

**Handling De-identified Ages**

In MIMIC-III, patients older than 89 have shifted DOB values.

Rule:

    If AGE > 120 → set AGE = 90

Reasoning:
- Identifies implausible ages caused by de-identification
- Preserves inclusion of elderly patients without removing them
- Prevents artificial outliers

---

### 4. Note Filtering Logic

---

#### 4.1 Allowed Note Categories

`CATEGORY` values were normalised (strip + lowercase).

Allowed:

- `physician`
- `nursing`
- `nursing/other`

Reasoning:
- These contain longitudinal progress documentation
- Excludes radiology, ECG, discharge summaries, administrative notes
- Focused on real-time clinical reasoning

---

#### 4.2 Error Filtering

Rule:

    Keep rows where ISERROR != 1

Important:

- Valid rows may contain `ISERROR = 0` or `NaN`.
- Initial filtering using `ISERROR == 0` incorrectly removed all notes due to `NaN` values.

This was corrected to exclude only explicit errors.

---

#### 4.3 ICU Cohort Enforcement

Notes were inner joined to valid ICU stays using:

`SUBJECT_ID` + `HADM_ID`

Join type:

`how="inner"`

Reasoning:
- Ensures notes belong to valid adult ICU stays
- Prevents inclusion of hospital notes unrelated to ICU admission

---

#### 4.4 Early Time Window Restriction

Constraint:

    INTIME ≤ CHARTTIME ≤ INTIME + 24 hours

Rows with missing `CHARTTIME` were removed before comparison.

Reasoning:
- Restricts to early ICU documentation
- Aligns with early deterioration modelling
- Excludes pre-ICU and late-stay notes

`TIME_WINDOW_HOURS = 24`

---

### 5. Logging and Integrity Verification

Row counts were logged after each major filtering step.

Purpose:
- Detect unintended full-row elimination
- Verify cohort shrinkage is monotonic
- Ensure filtering behaves as expected

This step identified the `ISERROR` logic issue during development.

All final counts below reflect corrected logic.

---

### 6. Final Cohort Statistics

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

### 7. Derived Cohort Properties

- ~72.7% of adult ICU stays contain ≥1 qualifying early note  
- ~18.3% of ICU-linked notes fall within first 24 hours  
- Mean notes per ICU stay ≈ 4.93  

These values are internally consistent and plausible for early ICU documentation.

---

### 8. Final Output Structure

Each row represents one ICU clinical note.

Columns:

- `SUBJECT_ID`
- `HADM_ID`  
- `ICUSTAY_ID`  
- `AGE`  
- `GENDER`  
- `FIRST_CAREUNIT`  
- `LOS_HOURS`  
- `CATEGORY`  
- `CHARTTIME`  
- `TEXT`  

Saved to:

`data/processed/icu_corpus.csv`

---

### 9. Post-Corpus Plan

The corpus is now frozen (162,296 notes across 32,910 ICU stays).  
Next phase: structured validation and JSON schema design.

**Step 1 — Manual Exploratory Analysis (~30 notes)**  
Stratified sampling at the ICU-stay level to:
- Inspect note structure and formatting
- Identify recurring section headers
- Detect artefacts and de-identification tokens
- Understand structural variability

This is qualitative structure discovery to inform schema and rule design.

**Step 2 — Quantitative Profiling (~300 notes)**  
Scripted analysis to measure:
- Section/header frequency
- Length distributions
- Formatting patterns
- Token characteristics

Used to empirically support and stress-test schema assumptions.

**Step 3 — JSON Schema Definition and Feature Implementation**  
Based on insights from Steps 1–2:
- Define structured JSON schema fields
- Implement section extraction logic
- Implement regex rules and text-cleaning logic
- Construct feature extraction pipeline

This produces structured JSON output from raw note text.

**Step 4 — Rule Validation (30–50 notes)**  
Perform deep manual comparison of:
- Raw text vs generated JSON
- Section parsing correctness
- Regex precision and edge cases
- Failure modes and systematic errors

Rules are refined here, then frozen and applied to the full corpus.

No train/test split is performed at this stage.  
This phase transitions from corpus construction to NLP feature engineering.

---

## Manual Structural Analysis

### 1. Purpose

- To determine whether deterministic, rule-based candidate extraction is feasible within the ICU note corpus prior to implementing any extraction logic.
- A random sample of 30 notes was selected because it is sufficient to identify dominant structural patterns and variability without being overwhelming for manual review.
- The script `manual_sample.py` randomly selected 30 notes from the processed ICU corpus and exported them as `manual_sample_30.csv` for qualitative structural inspection.
- No extraction or modeling was performed at this stage.

---

### 2. Structural Archetypes Identified

Manual inspection revealed that notes cluster into a small number of recurring structural families:

1. **Physician ICU Notes**
   - Long-form
   - Templated sections (e.g., Chief Complaint, HPI, Physical Exam, Assessment and Plan)
   - Embedded flowsheet blocks
   - Highly structured macro-organization

2. **Nursing Progress Notes**
   - System-based headers (e.g., Neuro, CV, Resp, GI)
   - Assessment / Action / Response / Plan patterns
   - Semi-structured, section-delimited format

3. **Respiratory Care / Procedural Notes**
   - Short
   - Intervention-dense
   - Limited narrative

4. **Cardiac Surgery / Specialty ICU Notes**
   - Hemodynamic and intervention-heavy
   - Dense numeric content
   - Structured system breakdown

No evidence of chaotic or unstructured free-text-only notes was observed.

Conclusion: The corpus demonstrates macro-level structural regularity.

---

### 3. Section Header Patterns

Section headers appear in multiple but predictable formats:

- Colon-terminated (e.g., `Assessment:`)
- Uppercase system blocks (e.g., `NEURO:`)
- Title-style headers (e.g., `Chief Complaint`)
- Inline system markers (e.g., `CV:`)

Header presence is common across note types.

Conclusion: Section segmentation via deterministic rules is feasible.

---

### 4. Flowsheet and Numeric Blocks

Physician notes frequently contain templated flowsheet sections including:

- Vital signs
- Hemodynamic summaries
- Fluid balance summaries

Vitals appear in three contexts:
1. Narrative sentences
2. Structured flowsheet blocks
3. Abbreviated nursing shorthand

Conclusion: Numeric pattern-based extraction is viable but must handle multiple formatting contexts.

---

### 5. Formatting Artefacts

The following systematic artefacts were observed:

- De-identification tokens (e.g., `[** ... **]`)
- Broken line wrapping
- Inconsistent spacing
- Embedded structured tables

These artefacts are consistent and predictable rather than random noise.

Conclusion: Preprocessing decisions will be required, but artefacts do not prevent deterministic parsing.

---

### 6. Entity Schema Feasibility

The provisional entity categories:

- `SYMPTOM`
- `INTERVENTION`
- `COMPLICATION`
- `VITAL_MENTION`

were observed frequently and naturally within the sample.

High intervention density and numeric density support the feasibility of rule-based candidate generation.

No structural incompatibility with the proposed schema was identified.

---

### Overall Conclusion of Manual Phase

The ICU corpus is:

- Section-rich
- Structurally repetitive at macro level
- Numerically dense
- Abbreviation-heavy but predictable
- Suitable for deterministic candidate extraction

- Phase 1 confirms that a hybrid architecture is appropriate for this dataset.
- Dual architecture feasible: Rule-based candidate generation → Transformer validation → Structured JSON output
- No extraction logic was implemented at this stage.

This phase reduces architectural risk and informs Phase 2 rule design.

---

## Quantitative Structural Profiling

### 1. Purpose

This stage performs quantitative structural profiling on a sampled subset of the frozen ICU corpus

- The objective is not clinical inference or statistical generalisation to a population, but engineering validation.
- This evaluates whether structural assumptions identified during manual inspection are stable at scale and not fragile artefacts of a 30-note sample.

Manual inspection identified recurring structural patterns which generated structural hypotheses about the corpus:

- Section-rich documents with templated headers
- Numeric-dense flowsheet content
- Systematic de-identification artefacts
- Clear macro-level structural archetypes (physician, nursing, procedural, specialty ICU)

Quantitative profiling tests whether those hypotheses hold at scale by validating the stability of:

- Section/header recurrence
- Numeric density stability
- Artefact prevalence
- Structural variability via note length

Profiling reduces architectural risk before deterministic rule implementation in Phase 2.

---

### 2. Phase Boundary Clarification

#### Why Profiling Occurs Before Rule Implementation

Rule-based systems fail when structural assumptions are incorrect — for example:

- Section delimiters are inconsistent  
- Numeric formatting is irregular  
- Artefacts are unpredictable  
- Target patterns are too sparse  

Quantitative profiling is a pre-implementation risk assessment step. It verifies that:

- Section segmentation will generalise  
- Numeric pattern-based extraction is justified  
- Artefacts are systematic  
- Structural variability is bounded  

This ensures Phase 2 rule implementation is informed rather than speculative.

#### Scope of This Stage

This stage does not:

- Extract entities  
- Implement extraction logic  
- Construct JSON schema  
- Train or validate transformer models  

It validates only that deterministic extraction is structurally feasible.

Quantitative profiling is the final structural checkpoint before transitioning from corpus analysis to rule engineering.

---

### 3. Continuity from Manual Inspection

The profiling metrics are directly derived from structural signals identified in the 30-note manual analysis.

Manual findings → Quantitative validation:

| Manual Observation | Profiling Measurement |
|-------------------|-----------------------|
| Notes are section-rich | Colon-terminated header count |
| Uppercase/system blocks common | Uppercase header count |
| ICU notes numerically dense | Numeric token count |
| BP and vital patterns frequent | BP-style pattern count |
| De-identification tokens systematic | De-id token count |
| Archetypes vary in size (short procedural vs long physician) | Character length, token count, line count |

Profiling does not attempt to recreate qualitative inspection.  
It measures only the structural signals that determine rule feasibility.

---

### 4. What Is Being Measured

For each of the 500 sampled notes, the profiling script computes the following structural signals:

| Category | Metric | What It Measures (Operational Definition) | Why It Matters |
|-----------|--------|--------------------------------------------|----------------|
| **Header Signals** | `colon_header_count` | Count of colon-terminated headers at line start, allowing leading whitespace and optional numeric prefixes (e.g., `Assessment:`, `NEURO:`, `    Plan:`, `15. Morphine:`). Pattern requires an uppercase initial character and alphabetic content before the colon. | Validates presence of consistent section delimiters for deterministic segmentation and rule-based splitting. |
|  | `uppercase_header_count` | Count of fully uppercase colon-terminated headers at line start, allowing indentation and multi-word uppercase headers (e.g., `NEURO:`, `CV:`, `CARDIO VASCULAR:`). Each word must contain ≥2 uppercase letters. | Measures intensity of strongly templated or system-based note structure. Acts as a stricter subset indicator of structured formatting. |
| **Numeric Density Signals** | `numeric_token_count` | Count of standalone numeric tokens (integers or decimals) bounded by word boundaries (e.g., `120`, `98.6`). Excludes embedded alphanumeric strings (e.g., `HR98`, `BP120/80`). | Quantifies numeric density, reflecting physiologic measurement burden and structured data presence. |
|  | `bp_pattern_count` | Count of structured blood pressure expressions, including optional ranges and optional whitespace around the slash (e.g., `120/80`, `120-130/90`, `170-180/109-112`, `120 / 80`). | Validates feasibility of deterministic extraction of physiologic measurements using regex-based rules. |
| **Artefact Signals** | `deid_token_count` | Count of MIMIC de-identification tokens using non-greedy matching of `[** ... **]` blocks. | Confirms systematic preprocessing artefacts that require removal or normalization prior to downstream modelling. |
| **Structural Variability Metrics** | `char_length` | Total character length of the note. | Quantifies macro-level length variability across note archetypes. |
|  | `token_count` | Total word count of the note. | Measures narrative density and verbosity differences. |
|  | `line_count` | Number of newline-separated lines. | Captures formatting granularity, section fragmentation, and structural spread. |

These metrics operationalize structural characteristics identified during qualitative inspection and provide quantitative validation that deterministic rule-based extraction and segmentation are feasible at scale.

---

### 5. Metric Scope Sufficiency

Quantitative profiling is not entity extraction and not schema implementation, it answers four architectural questions:

1. Do reliable section delimiters exist?
2. Is numeric information dense enough for pattern-based extraction?
3. Are formatting artefacts systematic rather than random?
4. Is note length variability bounded and predictable?

If these conditions hold across 500 notes, deterministic candidate generation is viable.

Measuring additional semantic features at this stage would prematurely enter Phase 2.

This stage validates feasibility, not completeness.

---

### 6. Sampling Decision

Sample size:

- 500 randomly sampled notes from the frozen corpus (n = 162,296)
- No stratification required
- Random seed fixed for reproducibility

Rationale:

- 30 notes sufficient for qualitative pattern discovery
- 500 notes sufficient for structural frequency stabilisation
- Structural signals (headers, numeric patterns) converge quickly
- Full 160k profiling provides negligible additional architectural value
- <200 risks instability

This is a pragmatic engineering validation sample size.

We are not estimating population statistics.
We are confirming robustness of deterministic parsing assumptions.

---

### 7. Profiling Procedure

The quantitative profiling stage is implemented in `quant_profiling.py`

#### 7.1 Data Loading and Sampling

- Input: `data/processed/icu_corpus.csv` with 162,296 notes
- Random sample of 500 notes
- Fixed `RANDOM_STATE = 42` for reproducibility
- Sample saved to: `data/sample/profiling_sample_500.csv`

---

#### 7.2 Per-Note Metric Computation

For each sampled note:

1. The `TEXT` field is coerced to string.
2. The following structural metrics are computed:

   - `colon_header_count` for colon-terminated headers
   - `uppercase_header_count` for uppercase system headers
   - `numeric_token_count` for numeric density
   - `bp_pattern_count` for BP-style patterns
   - `deid_token_count` for de-identification artefacts
   - `char_length` for note length
   - `token_count` for word count
   - `line_count` for formatting structure

3. Results are stored as one dictionary per note.
4. All dictionaries are converted into a 500-row DataFrame.
5. Per-note output saved to: `data/sample/profiling_per_note.csv`
   - Each row corresponds to one note.
   - Each column corresponds to one structural metric.

---

#### 7.3 Summary Statistics Computation

1. The script then using `.describe()` computes:
   - Count of non-null values
   - Mean
   - Standard deviation
   - Minimum 
   - 25th percentile 
   - Median 
   - 75th percentile
   - Maximum 
2. Compute `percent_notes_nonzero`:
   - Computed for each metric to measure prevalence across the sample.
   - Percentage of notes with non-zero metric value = (number of notes with metric > 0) / 500 * 100
3. Summary output saved to: `data/sample/profiling_summary.csv`

---

### 8. Interpretation of Profiling Results

This section interprets the aggregated summary statistics from `profiling_summary.csv` to determine whether structural assumptions hold at scale under the updated profiling logic.

---

#### 8.1 Header Signals

| Metric | Mean | Median | 75th % | Max | % Non-Zero |
|--------|------|--------|--------|-----|------------|
| `colon_header_count` | 18.46 | 8 | 20.25 | 113 | 81.2% |
| `uppercase_header_count` | 4.41 | 1 | 9 | 35 | 51.0% |

**Colon Headers**

- Median = 8  
- 81.2% of notes contain ≥1 colon-terminated header  
- Upper quartile = 20.25  
- Maximum = 113  

This represents a substantial increase compared to earlier profiling logic, reflecting the expanded header definition (indentation + optional numbering + broader matching).

Implications:

- Colon-terminated headers are clearly structurally dominant, not marginal.
- The distribution is strongly right-skewed, with a long tail of highly templated notes.
- Deterministic section-based segmentation is not merely feasible — it is structurally well-supported in the majority of the corpus.
- Only ~19% of notes lack colon headers, meaning fallback logic is needed but not primary.

Conclusion:

- Colon-based deterministic segmentation is justified as a core architectural assumption.


**Uppercase Headers**

- Median = 1  
- Mean = 4.41  
- 75th percentile = 9  
- Present in 51% of notes  

This confirms:

- Roughly half of notes contain strongly templated uppercase section blocks.
- Distribution remains right-skewed.
- Uppercase formatting is a secondary structural signal, not universal.

Conclusion:

- Uppercase headers act as a structural intensity marker.
- They should be treated as optional structural reinforcement rather than a required delimiter.

---

#### 8.2 Numeric Density Signals

| Metric | Mean | Median | 75th % | Max | % Non-Zero |
|--------|------|--------|--------|-----|------------|
| `numeric_token_count` | 49.43 | 17 | 58.5 | 432 | 93.6% |
| `bp_pattern_count` | 1.41 | 0 | 3 | 11 | 40.6% |

**Numeric Tokens**

- Present in 93.6% of notes  
- Median = 17  
- 75th percentile = 58.5  
- Maximum = 432  

Observations:

- Numeric content is near-universal.
- Heavy right tail consistent with ICU-style documentation and flowsheet imports.
- Large gap between median and max indicates mixed narrative vs data-dense archetypes.

Conclusion:

- Deterministic numeric extraction is strongly justified.
- Regex-based candidate generation is structurally appropriate.

---

**Blood Pressure Patterns**

- Present in 40.6% of notes  
- Median = 0  
- 75th percentile = 3  

Observations:

- BP expressions are common but not ubiquitous.
- When present, often appear multiple times.
- Whitespace-tolerant regex increases realistic capture coverage.

Conclusion:

- Structured physiologic extraction via regex is feasible but must tolerate absence.
- BP patterns represent a conditional but reliable structured signal.

---

#### 8.3 De-identification Artefacts

| Metric | Mean | Median | 75th % | Max | % Non-Zero |
|--------|------|--------|--------|-----|------------|
| `deid_token_count` | 7.05 | 3 | 9 | 66 | 78.8% |

- Present in 78.8% of notes  
- Median = 3  
- Maximum = 66  

Observations:

- De-identification artefacts are widespread and systematic.
- Heavy tail suggests some notes contain extensive redaction.

Conclusion:

- Preprocessing to remove or normalize `[** ... **]` tokens is mandatory prior to segmentation or extraction.
- Artefact handling is not optional.

---

#### 8.4 Structural Variability

| Metric | Mean | Median | 75th % | Max |
|--------|------|--------|--------|-----|
| `char_length` | 2694 | 1505 | 3417 | 17558 |
| `token_count` | 368 | 241 | 480 | 2557 |
| `line_count` | 65 | 26 | 86 | 456 |

Observations:

- Strong right-skew across all size metrics.
- Large separation between median and maximum.
- Wide variability in formatting density (line_count max = 456).

Interpretation:

- Multiple structural archetypes coexist (short procedural vs extensive summaries).
- Variability is continuous, not erratic.
- No evidence of structural sparsity collapse.

Conclusion:

- Deterministic rules must scale across wide note lengths.
- Size variability does not invalidate structural regularity.

---

#### 8.5 Structural Synthesis and Architectural Implications

Quantitative profiling demonstrates:

- Colon headers are highly prevalent (81.2%) and often dense.
- Uppercase templating appears in approximately half of notes.
- Numeric density is near-universal (93.6%).
- BP patterns are conditionally common (40.6%).
- De-identification artefacts are widespread (78.8%).
- Note length variability is large but structurally bounded.

Collectively, these findings confirm:

- Section-based segmentation is structurally supported in the majority of notes.
- Deterministic numeric and physiologic extraction is justified.
- Artefact normalization is a required preprocessing step.
- Structural heterogeneity exists but does not invalidate rule-based candidate generation.

No quantitative evidence suggests brittleness at corpus scale.  
Deterministic extraction in Phase 2 remains methodologically justified.

---

### 9. Inspection of Structural Extremes

#### Objective

- This stage performs targeted boundary inspection of the per-note metrics (`profiling_per_note.csv`) to validate deterministic robustness at structural extremes.
- Aggregate summary statistics establish prevalence and central tendency. However, they do not reveal whether extreme structural variants violate rule assumptions.
- Boundary inspection ensures Phase 2 rule logic remains valid across the full observed structural range.

---

#### 9.1 Rationale for Extreme-Based Inspection

Structural brittleness emerges at boundaries, not at the mean, therefore:

- For each functional metric, the 5 lowest-value and 5 highest-value notes are inspected.
- This captures both sparsity (absence of signal) and density (signal saturation).

This approach tests whether extreme variants:

1. Violate section segmentation assumptions  
2. Collapse formatting (e.g., single-block unstructured text)  
3. Exhibit pathological numeric density  
4. Contain malformed or nested artefacts  
5. Introduce edge-case formatting not visible in aggregate statistics  

Extremes provide maximal stress-testing efficiency without requiring full manual review of all 500 notes.

---

#### 9.2 Metric Selection Strategy

**Functional Metrics (Top 5 + Bottom 5)**

Applied to:

- `colon_header_count`
- `uppercase_header_count`
- `numeric_token_count`
- `bp_pattern_count`
- `deid_token_count`

Reasoning:

- These directly influence deterministic extraction logic.
- Both absence and saturation states are architecturally meaningful.
- Low values test fallback robustness.
- High values test over-segmentation and over-matching risk.

---

**Structural Metrics (Length Extremes Only)**

Applied only to:

- `char_length` (top 2 + bottom 2)

Reasoning:

- `token_count` and `line_count` are highly correlated with `char_length`.
- Inspecting all three produces redundant examples.
- Length captures macro-structural variability sufficiently.

Only 2 highest and 2 lowest are selected because:

- Structural size variation is continuous.
- Extreme tails are more informative than multiple mid-extreme samples.
- This reduces manual redundancy while preserving coverage.

---

#### 9.3 Deduplication Strategy

Extreme indices are collected into a `set`.

This ensures:

- Notes extreme across multiple metrics are inspected once.
- Manual review effort scales with structural diversity, not metric count.
- Overlapping outliers do not inflate inspection workload.

The resulting extreme set therefore represents unique structural boundary cases across all functional dimensions.

---

#### 9.4 Script Workflow

All steps are implemented in `profiling_boundary_extremes.ipynb`:

1. Load per-note metric file (`profiling_per_note.csv`).
2. Load corresponding raw note text file (`profiling_sample_500.csv`).
3. For each functional metric:
   - Sort ascending → select bottom 5 indices.
   - Sort descending → select top 5 indices.
   - Add these indices to the extreme indices set.
4. For structural size:
   - Select top 2 and bottom 2 by `char_length`.
   - Add these indices to the extreme length indices set.
5. For each unique extreme index:
   - Print all relevant metric values.
   - Print full raw note text.
6. Perform manual structural inspection.

This workflow produces a compact but comprehensive boundary review set.

#### 9.5 Extreme Note Analysis

Boundary inspection was performed on 45 structurally extreme notes selected across functional and macro-structural metrics:

- Total unique extreme notes: 42  
- Total unique length-extreme notes: 4  
- Total unique notes overall: 45  

This section summarizes recurring structural patterns rather than individual note commentary.

---

**A. High Header Density Notes (≈60–110+ colon headers)**

Observed Pattern:

- Highly templated ICU admission and transfer summaries  
- Repeated structured blocks (HPI, PMHx, ROS, Physical Examination, Labs/Radiology, Assessment and Plan, ICU Care)  
- Enumerated problem lists with numeric prefixes (e.g., `1. LLE swelling.`)  
- Embedded flowsheet tables (Vital Signs, Hemodynamics, Fluid Balance)  
- Appended attending addenda and “Protected Section” content  
- EMR-generated reference blocks and occasional JavaScript/link fragments appended at document end  

Structural Characteristics:

- Colon headers consistently function as reliable structural delimiters.  
- Uppercase and colon-based headers frequently coexist.  
- Numbered problem lists integrate cleanly with colon-based patterns.  
- Repeated section scaffolds (e.g., multiple “Assessment and Plan” layers) remain syntactically valid.  
- Addenda, protected sections, and trailing EMR reference/link artefacts are clearly demarcated and structurally separable.  
- Template repetition increases density but does not introduce delimiter ambiguity.  

Conclusion:

Extreme header density reflects EMR templating, interdisciplinary layering, and ICU documentation conventions rather than structural instability. Colon-based segmentation remains stable at maximal density, including across appended artefact blocks.

---

**B. Moderate-to-Low Header Notes**

Observed Pattern:

- Focused ICU updates or ward progress notes  
- Problem-oriented scaffolds (e.g., “Assessment / Action / Response / Plan”)  
- Semi-structured outlines with partially completed sections  
- Condition headers followed by minimal narrative expansion  

Structural Characteristics:

- Sparse but syntactically valid delimiter usage.  
- Header labels may appear without substantive body text.  
- No malformed colon constructs observed.  
- Structural sparsity reflects workflow variation rather than corruption.  

Conclusion:

Lower header counts represent stylistic and clinical workflow variation. Deterministic segmentation must tolerate partially populated or minimally structured templates.

---

**C. Zero-Header and Micro-Narrative Notes**

Observed Pattern:

- Brief respiratory therapy or nursing updates  
- Short narrative summaries  
- Administrative or transfer references  

Structural Characteristics:

- No colon or uppercase header markers.  
- Some notes contain no numeric tokens or de-identification artefacts.  
- Coherent free-text narrative despite brevity.  
- No malformed formatting or delimiter ambiguity.  

Conclusion:

Zero-header notes represent short-form documentation archetypes. These define fallback narrative cases rather than structural failure modes.

---

**D. High Numeric Density Notes (≈150–400+ numeric tokens)**

Observed Pattern:

- Vertically stacked laboratory panels and serial timestamped labs  
- Vital sign ranges with embedded MAP values (e.g., `104/70(77)`)  
- Medication dosing strings and infusion rates  
- Hematology, chemistry, coagulation, and microbiology result blocks  
- Mixed inline and column-style numeric alignment (ASCII lab tables)  

Structural Characteristics:

- Numeric clustering occurs within clinically structured contexts.  
- BP-style expressions include ranges and parentheses without ambiguity.  
- Lab panels may include ASCII separators (e.g., dashed dividers) but remain structurally bounded.  
- Numeric-heavy notes frequently coexist with high header and de-identification density.  
- No digit fragmentation or token boundary corruption observed.  

Conclusion:

Extreme numeric density reflects ICU complexity and integrated EMR lab reporting. Regex-based numeric extraction remains structurally robust under saturation and table-like alignment patterns.

---

**E. De-identification and EMR Artefacts**

Observed Pattern:

- Repeated `[** ... **]` masking for dates, institutions, clinicians, and identifiers  
- Redactions embedded within headers, numeric ranges, and narrative text  
- Masked timestamps and numeric intervals within brackets  
- Appended attending attestations and EMR-generated protected sections  
- EMR-generated reference blocks and occasional trailing JavaScript/link fragments  

Structural Characteristics:

- Uniform `[** ... **]` formatting without malformed, truncated, or nested markers.  
- No instances of nested or partially closed redaction tokens were observed.  
- Redactions do not break header syntax or numeric pattern detection.  
- No ambiguous overlap between masking syntax and colon delimiters.  
- EMR reference and JavaScript/link artefacts are consistently appended and structurally separable from core clinical narrative.  

Conclusion:

De-identification and EMR artefacts are systematic and structurally stable. Preprocessing normalization (mask removal and artefact trimming) is required but straightforward. No evidence of delimiter corruption, header misclassification, or numeric pattern interference was identified.

---

**F. Hybrid Template + Narrative Structures**

Observed Pattern:

- Mixed uppercase headers, colon-based headers, and numbered problem lists  
- Narrative paragraphs interleaved with structured blocks  
- ICU Care sections followed by attending addenda  
- Duplicate communication, code status, or disposition sections  

Structural Characteristics:

- Multiple structural conventions coexist without delimiter collision.  
- Header syntax remains internally consistent across sections.  
- Template layering increases length and density but not ambiguity.  
- Addenda remain separable from primary plan sections.  

Conclusion:

Hybrid structuring increases heterogeneity but does not undermine rule-based segmentation assumptions. Structural conventions remain separable and internally coherent.

---

**G. Boundary Length Extremes**

Observed Pattern:

- Ultra-short notes (1–3 lines; minimal tokens)  
- Header-only scaffolds without body text  
- Extremely long ICU admissions (>15,000 characters; >2,500 tokens; ~400 lines)  
- Long notes combining narrative history, lab tables, flowsheets, enumerated plans, protected addenda, and EMR reference blocks  

Structural Characteristics:

- Shortest notes show no malformed constructs despite minimal content.  
- Longest notes exhibit dense layering of templated sections and numeric panels.  
- Structural repetition scales linearly rather than chaotically.  
- No delimiter instability observed at maximal size.  

Conclusion:

Macro-level size variability is substantial but structurally bounded. Deterministic parsing scales across both minimal and maximal documentation without observed instability.

---

**Structural Integrity Assessment Across Extremes**

Across all inspected extremes:

- No malformed colon-header constructs.  
- No delimiter collisions or ambiguous header boundaries.  
- No uppercase non-header lines falsely triggering header logic during manual inspection.  
- No broken BP-style expressions.  
- No numeric tokenization anomalies at density extremes.  
- No corrupted, nested, or truncated de-identification markers.  
- No structural instability at minimal or maximal document length.  
- EMR reference and JavaScript/link artefacts remain consistent and patternable.  
- No hybrid cases invalidate rule-based segmentation assumptions.  

---

**Boundary Validation Outcome**

Manual inspection of structural extremes confirms:

- Colon-based segmentation is robust under sparse and saturated header conditions.  
- Uppercase header usage is stylistically variable but structurally consistent.  
- Numeric and physiologic regex extraction remains stable under dense lab stacking and ASCII-style table layouts.  
- Artefact normalization (de-identification, protected sections, and EMR reference/link trimming) is mandatory but structurally straightforward.  
- Zero-header and micro-narrative notes represent expected workflow variation rather than structural failure.  

No structural failure modes were identified at corpus boundaries.

Phase 2 deterministic extraction remains architecturally justified.

----
