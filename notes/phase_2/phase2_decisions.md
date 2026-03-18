# Phase 2 — Deterministic Rule-Based Extraction Decisions

## Objective

- This document outlines the key decisions made regarding entity schema and extraction rules for Phase 2 of the project, which focuses on deterministic rule-based extraction from ICU progress notes.
- The decisions here define the scope and approach for Phase 2, ensuring a focused and well scoped extraction process.

---

## Entity Schema Operationalisation

### 1. Purpose

- Define the four clinically meaningful entity types to extract from ICU notes, finalising scope for Phase 2 deterministic extraction and preventing uncontrolled expansion. 
- These entities form the foundation of structured JSON outputs for downstream use.

---

### 2. Entity Types 

Extraction in Phase 2 is strictly limited to four entity types:

- **SYMPTOM**  
- **INTERVENTION**  
- **COMPLICATION**  
- **VITAL_MENTION**

These four were chosen because they capture the core clinically relevant information commonly reported in ICU progress notes, balancing scope control and portfolio complementarity with other ongoing projects. Limiting to these avoids over-complexity, prevents scope creep, and ensures high precision in rule-based extraction.

---

#### 2.1 SYMPTOM

**Purpose:** Capture patient-reported complaints or clinician-observed manifestations.  
**Rationale:** Symptoms represent the subjective or observable clinical state that informs interventions and complications.

**Operational Decisions:**

- **Inclusions:**  
  - Patient-reported complaints (e.g., "chest pain")  
  - Observed clinical signs (e.g., "confused")  

- **Exclusions:**  
  - Lab or imaging values  
  - Procedures  
  - Baseline chronic diagnoses  

- **Negation Handling:**  
  - "no", "denies", "without", "not"  

- **Trigger/Pattern Notes:**  
  - Common clinical descriptors and synonymous terms captured  
  - Avoid overlap with other entity types  

**Positive Extraction Examples**

The following phrases should produce a SYMPTOM extraction:

- "patient reports chest pain"
- "complaining of nausea overnight"
- "severe headache this morning"
- "patient feeling dizzy"
- "persistent shortness of breath"

**Negative / Non-Extraction Examples**

The following phrases should NOT produce a SYMPTOM extraction:

- "history of migraine"
- "CT shows intracranial hemorrhage"
- "labs notable for elevated troponin"
- "scheduled for CT scan"

These represent diagnoses, tests, or history rather than symptoms.

**Boundary Resolution**

Ambiguous terms are resolved as follows:

- tachycardia → VITAL_MENTION (not SYMPTOM)
- hypotension → VITAL_MENTION
- delirium → SYMPTOM
- agitation → SYMPTOM

---

#### 2.2 INTERVENTION

**Purpose:** Capture therapeutic or procedural actions performed.  
**Rationale:** Interventions document treatments and procedures, critical for understanding patient management.

**Operational Decisions:**

- **Inclusions:**  
  - Medication initiation (e.g., "started norepinephrine")  
  - Procedures (e.g., "central line inserted", "intubated")  

- **Exclusions:**  
  - Hypothetical plans or recommendations not performed  

- **Negation Handling:**  
  - Same cues as SYMPTOM  

- **Trigger/Pattern Notes:**  
  - Focus on high-confidence deterministic patterns  
  - Avoid misclassifying narrative suggestions or future plans  

**Positive Extraction Examples**

The following phrases should produce an INTERVENTION extraction:

- "started norepinephrine"
- "intubated for airway protection"
- "central line inserted"
- "patient placed on ventilator"
- "administered antibiotics"

**Negative / Non-Extraction Examples**

The following phrases should NOT produce an INTERVENTION extraction:

- "plan to start antibiotics"
- "consider dialysis"
- "may require intubation"
- "recommend central line placement"

These represent future plans or recommendations rather than completed interventions.

**Boundary Resolution**

Ambiguous phrases are resolved as follows:

- "started antibiotics" → INTERVENTION
- "on antibiotics" → INTERVENTION
- "intubation planned" → NOT extracted

---

#### 2.3 COMPLICATION

**Purpose:** Capture adverse or pathological developments during ICU stay.  
**Rationale:** Complications indicate negative outcomes or new pathological events, essential for downstream analysis and evaluation.

**Operational Decisions:**

- **Inclusions:**  
  - Acute pathological conditions (e.g., "AKI", "sepsis", "pneumothorax")  

- **Exclusions:**  
  - Chronic baseline diagnoses without acute worsening  

- **Negation Handling:**  
  - Same as above  

- **Trigger/Pattern Notes:**  
  - Detect keywords and common abbreviations for acute events  
  - Maintain clear separation from INTERVENTION or SYMPTOM  

**Positive Extraction Examples**

The following phrases should produce a COMPLICATION extraction:

- "developed sepsis overnight"
- "new acute kidney injury"
- "patient with pneumothorax"
- "episode of ventricular arrhythmia"

**Negative / Non-Extraction Examples**

The following phrases should NOT produce a COMPLICATION extraction:

- "history of chronic kidney disease"
- "prior stroke"
- "family history of cancer"

These represent baseline or historical conditions.

**Boundary Resolution**

Ambiguous terms are resolved as follows:

- AKI → COMPLICATION
- sepsis → COMPLICATION
- infection → COMPLICATION
- chronic heart failure → NOT extracted unless acute worsening is described

---

#### 2.4 VITAL_MENTION

**Purpose:** Capture explicit physiological measurements or abnormal descriptors.  
**Rationale:** Vital signs provide objective, structured information embedded in narrative, useful for linking symptoms, interventions, and complications.

**Operational Decisions:**

- **Inclusions:**  
  - Measurement + value (e.g., "BP 85/50")  
  - Descriptors indicating abnormality (e.g., "tachycardic")  

- **Exclusions:**  
  - Lab-only results (covered elsewhere)  

- **Negation Handling:**  
  - Same as above  

- **Trigger/Pattern Notes:**  
  - Regex patterns targeting numeric values, units, and common shorthand  
  - Ensure alignment with section and sentence spans  

**Positive Extraction Examples**

The following phrases should produce a VITAL_MENTION extraction:

- "BP 85/50"
- "HR 120"
- "Temp 38.5"
- "tachycardic to 130"
- "oxygen saturation 92%"

**Negative / Non-Extraction Examples**

The following phrases should NOT produce a VITAL_MENTION extraction:

- "labs notable for creatinine 2.0"
- "WBC elevated"
- "troponin increased"

These represent laboratory values rather than vital signs.

**Boundary Resolution**

Ambiguous terms are resolved as follows:

- tachycardia → VITAL_MENTION
- hypotension → VITAL_MENTION
- hypoxia → VITAL_MENTION
- fever → SYMPTOM unless an explicit temperature value is present

---

#### 2.5 Summary

Phase 2 extraction will only target these four entities to:

- Ensure manageable rule creation and high precision  
- Prevent overlap and ambiguity between entity types  
- Provide structured, span-aligned JSON outputs ready for transformer validation in Phase 3  

This finalises the scope of entity extraction for Phase 2.

---

## Report Preprocessing

### 1. Objective

- The preprocessing layer performs minimal normalization of ICU clinical notes to stabilize the text for deterministic rule-based extraction while preserving traceability to the original source text.
- This serves as the formal reproducibility and audit specification for `preprocessing.py`.
- All logic described here reflects the final, validated implementation.

Preprocessing therefore functions solely as a stabilisation step applied to raw clinical text prior to structural segmentation and rule-based entity extraction.

---

### 2. Preprocessing Decisions

Preprocessing decisions are derived directly from the structural analysis conducted in Phase 1. Key findings include:

- Systematic de-identification tokens (`[** ... **]`) are widespread
- Notes contain inconsistent whitespace and line formatting
- Core structural signals (e.g., colon-delimited headers and numeric expressions) remain stable

Based on these findings, preprocessing is restricted to correcting artefacts that would interfere with rule matching or section segmentation. The preprocessing stage therefore:

- Removes systematic formatting artefacts identified in Phase 1
- Preserves the semantic content and structural organization of the note
- Maintains compatibility with downstream deterministic extraction

---

### 3. Implementation Details

The preprocessing layer performs only the minimal transformations required to stabilize the text for deterministic parsing.

1. **Normalise Newlines**  
   Standardises line breaks across the corpus (`\r`, `\r\n` → `\n`) to ensure consistent line boundaries for section parsing.

2. **Remove De-identification Tokens**  
   Strips `[** ... **]` tokens to protect patient privacy. While this occasionally breaks sentences, it is necessary for de-identification and does not impede downstream rule-based extraction.

3. **Normalise Whitespace**  
   Collapses multiple spaces or tabs into a single space. This stabilises token offsets and prevents misalignment during entity extraction, without altering section or sentence structure.

4. **Remove EMR Trailing References**  
   Eliminates end-of-document artefacts starting from the `References` header, which typically contain JavaScript popups or EMR metadata irrelevant to clinical content. This step preserves all clinical sections.

---

### 4. Preprocessing Manual Validation

#### 4.1 Overview

- Validation implemented via `validate_preprocessing.py`, which applies the preprocessing function to a random sample of 10 ICU notes, and compares original vs preprocessed outputs.
- A random sample of 10 ICU clinical notes was manually inspected to evaluate the effectiveness of the Phase 2 preprocessing function. 
- The validation focused on confirming that de-identification tokens were removed, structural elements were preserved, and that the resulting text remained suitable for deterministic rule-based extraction.

---

#### 4.2 Findings

- **Artefact Removal:**  
  All `[** ... **]` blocks, including names, dates, hospitals, and identifiers, were successfully removed across all notes.  

- **Structural Preservation:**  
  - Section headers (e.g., `S:`, `O:`, `Assessment:`, `Plan:`) remained intact.  
  - Numeric values, vitals, lab results, and medication dosages were unaffected.  
  - Colon-delimited headers and other section delimiters are preserved for downstream segmentation.

- **Sentence Integrity:**  
  - Removal of de-identification tokens occasionally produced broken or incomplete sentences.  
  - This is expected and acceptable, as Phase 2 extraction relies on section and span context rather than perfect sentence syntax.  

- **Additional Observations:**  
  - Minimal extra whitespace or dangling punctuation remains in some cleaned notes.  
  - These cosmetic issues do not compromise rule-based extraction.

---

#### 4.3 Conclusion

The preprocessing function achieves its objective: 

- It stabilises the raw clinical text by removing artefacts while maintaining semantic content, numeric density, and section structure. 
- Sentence breaks and minor cosmetic issues are acceptable within the deterministic extraction pipeline and do not require further preprocessing at this stage.

---

## Section Detection

### 1. Objective

- Section detection identifies structural narrative sections within clinical notes.  
- Clinical documentation typically follows semi-structured formats where major components of the note are introduced by headers
- Detecting these sections allows the pipeline to:
  1. Segment notes into semantically meaningful regions
  2. Restrict downstream extraction to relevant clinical contexts
  3. Reduce noise from structured flowsheet artifacts embedded in the notes
  4. Improve determinism of rule-based extraction by providing section-specific context for entity extraction.

This stage therefore converts a flat clinical note into a structured representation of sections and their contents.

---

### 2. Section Detection Decision

From the Phase 1 manual inspection of the dataset, several consistent formatting patterns were observed:

- Headers almost always occur at the start of a new line
- Headers frequently end with a colon
- Some headers appear as standalone capitalised or non-capitalised phrases
- Many lines contain leading whitespace or indentation

Based on these observations, the section detection component is designed to identify headers using a combination of:

- Line-based parsing to detect potential headers at the start of lines
- Pattern matching to identify common header formats (e.g., capitalised words followed by a colon)
- A predefined list of common headers derived from the dataset

---

### 3. Header Pattern Exploration

- The notebook `header_pattern_exploration.ipynb` was used to apply various broad regex patterns to the entire corpus of ICU notes to extract and count repetition of all potential headers.
- This process identified a comprehensive list of 300 candidate headers sorted by count number, which were then manually reviewed to determine which should be included in the final section detection rules.

#### 3.1 Regex Logic

To identify potential headers across ICU notes, we applied two general regex patterns designed to capture most narrative section headers without being overly specific.

**Colon Terminated Headers**

`colon_pattern = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 /()\-'&]{0,80})\s*:\s*")`

- Matches lines that start with optional whitespace.
- Captures a leading alphanumeric phrase (letters, numbers, spaces, /, (, ), -, ', &).
- Requires a colon somewhere after the phrase (allowing optional spaces before and after).

**Standalone Headers**

`standalone_pattern = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 /()\-\']{1,80})\s*:?\s*$")`

- Matches lines that start and end with optional whitespace.
- Captures phrases that may or may not have a colon at the end.
- Ensures the header appears alone on a line ($ anchor), to avoid picking up inline text.

**Implementation Notes**

- Each note is split line-by-line.
- Lines matching either pattern are counted using a Counter.
- This approach captures both standard colon-terminated narrative sections and headers that appear on their own line without a colon.
- The resulting counts are used to prioritise the most frequent header candidates for canonical mapping downstream.

---

#### 3.2 Header List Manual Validation

Inspection of the 300 most frequent header candidates revealed that the detected patterns fall into several distinct structural categories. These categories reflect how clinical notes mix narrative documentation, structured monitoring data, and administrative metadata within the same free-text document.

**Observed Header Categories**

Only narrative clinical headers (e.g., SOAP-style sections) are relevant for downstream section detection. Subsections, physiological monitoring fields, laboratory variables, and administrative metadata are captured by the regex but will be ignored in the canonical mapping.

**A. Narrative clinical sections**

These represent the core narrative structure of clinical documentation.  
They introduce sections where clinicians describe patient history, examination findings, and clinical reasoning.

Examples observed in the corpus:

- `Plan`
- `Assessment`
- `Chief Complaint`
- `HPI`

---

**B. Examination or system-based subsections**

These represent organ-system subsections, commonly appearing within a physical examination or assessment section.

Examples observed in the corpus:

- `Neurologic`
- `Cardiovascular`
- `Respiratory / Chest`
- `Abdominal`

---

**C. ICU monitoring and physiological fields**

A large proportion of detected headers correspond to monitoring variables or device parameters.  
These fields appear frequently because ICU documentation often embeds flowsheet-style monitoring data directly inside clinical notes.

Examples observed in the corpus:

- `HR`
- `BP`
- `SpO2`
- `FiO2`

These are structured measurements rather than narrative sections and therefore should not be interpreted as document section boundaries.

---

**D. Laboratory or diagnostic variables**

Another large group of matches corresponds to laboratory values or diagnostic measurements that appear as structured fields.

Examples observed in the corpus:

- `WBC`
- `Glucose`
- `Creatinine`
- `Hct`

These represent individual test values rather than narrative text blocks.

---

**E. Administrative or documentation metadata**

Some detected headers correspond to administrative or documentation-tracking fields that appear in admission templates or electronic health record exports.

Examples observed in the corpus:

- `Attending MD`
- `Admit diagnosis`
- `Transferred from`
- `Transferred to`

These fields describe metadata about the encounter rather than clinical narrative content.

---

**Empirical Insight**

The ranked frequency list of the 300 most common header candidates provided an empirical view of the header distribution across the dataset. Manual inspection revealed that the detected headers fall into several structural categories, but only a subset corresponds to true narrative clinical sections, which are relevant for section detection. Key observations include:

1. True narrative clinical sections, such as SOAP-style headers, appear very frequently, often tens of thousands of times across the corpus. Examples include:
2. Many detected headers correspond to structured data rather than document structure, including:
   - ICU monitoring fields (e.g., `HR`, `BP`, `SpO2`, `FiO2`)
   - Laboratory or diagnostic variables (e.g., `WBC`, `Glucose`, `Creatinine`, `Hct`)
   - Administrative or documentation metadata (e.g., `Attending MD`, `Admit diagnosis`, `Transferred from`)
3. The header vocabulary stabilises quickly: the most common narrative sections appear within the top few hundred candidates. Beyond this range, additional matches consist almost entirely of flowsheet variables, lab fields, abbreviations, or other non-structural artifacts.

Thus, we can be confident that mapping canonical headers using only the frequent, manually validated narrative headers is sufficient for robust section detection.

---

**Implication for Section Detection**

The frequency analysis confirms that regex-based detection alone is insufficient, because regex patterns capture both narrative headers and structured non-narrative fields. To ensure accurate section boundaries:

1. Candidate headers are detected using generalised regex patterns.
2. Only narrative clinical headers are retained for downstream mapping.
3. Headers are normalised and mapped to a curated canonical section set.

Structured monitoring fields, lab values, and administrative metadata are ignored, even if they match regex patterns, to prevent misidentification of section boundaries.

This approach ensures that section detection captures meaningful narrative blocks while excluding non-narrative or structured artifacts embedded within clinical notes.

---

#### 3.3 Final Headers

The following 13 headers were retained as top-level narrative section headers within the clinical notes:

- `Plan`
- `Assessment`
- `Action`
- `Response`
- `Assessment and Plan`
- `Chief Complaint`
- `HPI`
- `Past medical history`
- `Family history`
- `Social History`
- `Review of systems`
- `Physical Examination`
- `Disposition`

These represent narrative clinical sections in which clinicians write extended free-text descriptions, reasoning, or summaries. Such sections typically contain the main clinical narrative (e.g., history, examination findings, clinical reasoning, and management plans).

Because these sections introduce substantial blocks of free-text content, they provide reliable boundaries for segmenting clinical documents into meaningful narrative units for downstream processing.

---

### 4. Section Detection Decisions

#### 4.1 Broad Header Detection with Narrow Canonical Storage

Due to earlier header pattern exploration, clinical notes contain a wide range of header-like structures. To ensure reliable structural parsing, the detection strategy intentionally follows a broad detection, narrow storage principle:

1. Regex patterns are designed to detect a wide range of potential headers using the same general logic applied in `header_pattern_exploration.ipynb`.
2. Only the curated set of 13 canonical narrative headers is retained for downstream extraction.

This approach ensures that the algorithm can correctly identify section boundaries while preventing non-narrative fields from being incorrectly interpreted as meaningful narrative sections.

---

#### 4.2 Dual Header Pattern Strategy

The same two regex patterns used for the header exploration notebook were implemented in the section detection pipeline:

1. **Colon-terminated headers**

  Captures lines that begin with a header phrase followed by a colon, optionally containing inline text directly after the header on the same line. This format commonly appears in narrative documentation.

  Examples:
  - `Plan:`
  - `Chief Complaint: Chest pain`

2. **Standalone headers**

  Captures lines that consist solely of a header phrase, with or without a trailing colon, to account for headers that appear on their own line.

  Examples:
  - `Chief Complaint`
  - `HPI`

Using both patterns ensures that section boundaries are identified reliably regardless of whether a colon is present or whether the header appears inline or on a separate line.

---

#### 4.3 Canonical Header Normalisation

Section headers in clinical notes frequently vary in capitalisation and formatting (e.g., `Assessment`, `ASSESSMENT`, `assessment`). To ensure consistent representation across notes:

- All detected headers are normalised to lowercase before comparison.
- A predefined canonical header set (`CANONICAL_HEADER_SET`) is stored entirely in lowercase.
- Extracted sections are stored using the canonical lowercase representation.

This guarantees that the same section is represented consistently across all documents.


---

#### 4.4 Structural Boundary Rule

A key design rule is that any detected header acts as a structural boundary, even if the header is not part of the canonical header set. When a header is detected:

1. The current section (if one is active) is finalised and stored.
2. A new section is started only if the header matches a canonical header.

Non-canonical headers therefore function as section terminators but do not create new stored sections.

This rule is important because ICU notes often contain monitoring variables, laboratory measurements, and administrative fields that resemble headers but do not represent narrative sections. Treating all detected headers as boundaries prevents narrative sections from incorrectly spanning across unrelated structured content.

---

#### 4.5 Inline Header Content Handling

Clinical documentation frequently places section content on the same line as the header.

Example: `Chief Complaint: Chest pain for two days`

- If the algorithm only treated headers as standalone markers, the content following the colon would be lost.
- To prevent this, any text appearing after the first colon on the header line is captured and added as the first line of the section content.
- This ensures that narrative information embedded inline with headers is preserved during extraction.

---

### 5. Method Refinement

#### 5.1 Initial Design and Rationale

The initial section detection strategy aimed to maximise structural accuracy by leveraging broad header detection. Using generalised regex patterns, all header-like lines (both canonical and non-canonical) were identified. The design followed a broad detection, narrow storage principle:

- All detected headers acted as structural boundaries
- Only canonical headers were stored as section keys
- Non-canonical headers terminated sections but did not create new ones

This approach was motivated by the assumption that:

- Clinical notes contain diverse header formats
- Treating all headers as boundaries would prevent unrelated structured content (e.g., monitoring data, labs) from being included within narrative sections

---

#### 5.2 Observed Failure: Over-Segmentation

Manual validation revealed that this approach systematically failed due to over-segmentation.

Specifically:
- Non-canonical headers such as `HEENT`, `Cardiovascular`, and `Respiratory` were incorrectly treated as section boundaries
- These headers frequently occur within true narrative sections (e.g., `Physical Examination`, `Assessment and Plan`)

This resulted in:
- Premature termination of valid sections
- Empty or severely truncated outputs (e.g., missing `Physical Examination`)
- Fragmentation of continuous clinical narratives
- Loss of clinically relevant information

---

#### 5.3 Root Cause

The failure arises from a fundamental structural property of ICU notes, clinical notes contain multiple hierarchical levels that are not distinguishable using simple regex patterns:

1. **Top-level narrative sections**  
   e.g., `HPI`, `Physical Examination`, `Assessment and Plan`

2. **Nested subsections**  
   e.g., `Neurologic`, `Cardiovascular`, `Respiratory`

3. **Structured flowsheet and measurement data**  
   e.g., `HR`, `BP`, `SpO2`, laboratory values

These elements often share identical surface patterns (e.g., colon-terminated phrases), making it impossible to reliably differentiate them using rule-based pattern matching alone.

As a result, the assumption that any detected header represents a structural boundary is invalid in real-world clinical text.

---

#### 5.4 Final Design Decision

To address this, the section detection strategy was simplified to a canonical-only boundary approach:

- Only headers in the predefined canonical set are detected
- Only canonical headers define section boundaries
- All non-canonical header-like lines are treated as normal text
- Section content is defined as all text between consecutive canonical headers

---

#### 5.5 Justification of Final Approach

This refinement directly resolves the observed failure mode:

- Prevents subsection headers from prematurely terminating sections
- Preserves complete narrative blocks (e.g., full `Physical Examination`)
- Eliminates dependence on unreliable header-like pattern distinctions

This approach intentionally prioritises:

- **High recall**: ensuring clinically relevant content is not lost
- **Structural robustness**: avoiding fragmentation in noisy, real-world data

At the same time, it accepts a known trade-off:

- **Moderate precision**: some non-relevant content (e.g., flowsheet data) may be included within sections

This trade-off is appropriate because:
- Perfect structural parsing of clinical notes is not achievable with deterministic rules
- Downstream processing can tolerate or filter excess content
- The primary objective is accurate recovery of core narrative sections, not perfect boundary precision

---

#### 5.6 Final Position

The refined method reflects a realistic and methodologically sound approach to section detection in clinical text:

- It aligns with the inherent limitations of rule-based systems
- It is robust to the structural variability of ICU notes
- It achieves the project goal of reasonably accurate section extraction, rather than unattainable perfect parsing

This establishes a reliable foundation for downstream analysis and modelling.

---

### 6. Section Extraction Workflow

The section extraction algorithm implemented in `section_detection.py` processes each clinical note sequentially using a deterministic line-based parsing strategy with two functions `detect_header()` and `extract_sections()`. The workflow is as follows:

1. **Split the clinical note into lines**

  - The note is divided into individual lines to allow structural parsing of potential section headers.

2. **Detect potential headers**

  - Function 1: `detect_header()`
  - Each line is evaluated against the two header detection regex patterns:
    - Colon-terminated headers
    - Standalone headers

3. **Apply structural boundary logic**

  - Function 2: `extract_sections()`
  - When a header is detected, the currently active section (if any) is finalised and stored.

4. **Determine canonical eligibility**

  - The detected header is normalised to lowercase and checked against the canonical header set.
    - If the header is canonical, a new section is initiated.
    - If the header is not canonical, no new section is created.

5. **Accumulate section text**

  - Lines following a canonical header are appended to the section content until another header is encountered.
    - Leading and trailing whitespace is removed.
    - Empty lines are ignored.

6. **Capture inline header text**

  - If a header line contains text after a colon, that text is added as the first line of the section content.

7. **Finalise the last section**

  - After processing the entire note, any active section is stored.

8. **Return structured output**

  - The function returns a dictionary where:
    - **Keys** are canonical section headers (lowercase)
    - **Values** are the extracted narrative text belonging to each section

This structured representation converts an unstructured clinical note into clearly defined narrative segments that can be used for downstream clinical information extraction and analysis.

---

### 7. Section Detection Manual Validation

#### 7.1 Overview

- Validation implemented via `validate_section_detection.py`, which applies the section extraction function to a random sample of 10 ICU notes, and compares original vs outputs.
- A random sample of 10 ICU clinical notes was manually inspected to evaluate the effectiveness of the Phase 2 section extraction function. 
- The validation focused on confirming that the detected sections correspond to meaningful narrative blocks, that section boundaries are correctly identified, and that non-narrative fields are appropriately ignored.

---