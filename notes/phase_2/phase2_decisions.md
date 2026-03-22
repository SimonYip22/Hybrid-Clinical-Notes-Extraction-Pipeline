# Phase 2 — Deterministic Rule-Based Extraction Decisions

## Objective

- This document outlines the key decisions made regarding entity schema and extraction rules for Phase 2 of the project, which focuses on deterministic rule-based extraction from ICU progress notes.
- The decisions here define the scope and approach for Phase 2, ensuring a focused and well scoped extraction process.

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

## Section Detection and Extraction

### 1. Objective

- Section extraction identifies structural narrative sections within clinical notes.  
- Clinical documentation typically follows semi-structured formats where major components of the note are introduced by headers
- Detecting these sections allows the pipeline to:
  1. Segment notes into semantically meaningful regions
  2. Restrict downstream extraction to relevant clinical contexts
  3. Reduce noise from structured flowsheet artifacts embedded in the notes
  4. Improve determinism of rule-based extraction by providing section-specific context for entity extraction.

This stage therefore converts a flat clinical note into a structured representation of sections and their contents.

---

### 2. Section Extraction Decision

From the Phase 1 manual inspection of the dataset, several consistent formatting patterns were observed:

- Headers almost always occur at the start of a new line
- Headers frequently end with a colon
- Some headers appear as standalone capitalised or non-capitalised phrases
- Many lines contain leading whitespace or indentation

Based on these observations, the section extraction component is designed to identify headers using a combination of:

- Line-based parsing to detect potential headers at the start of lines
- Pattern matching to identify common header formats (e.g., capitalised words followed by a colon)
- A predefined list of common headers derived from the dataset

---

### 3. Header Pattern Exploration

- The notebook `header_pattern_exploration.ipynb` was used to apply various broad regex patterns to the entire corpus of ICU notes to extract and count repetition of all potential headers.
- This process identified a comprehensive list of 300 candidate headers sorted by count number, which were then manually reviewed to determine which should be included in the final section extraction rules.

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

Only narrative clinical headers (e.g., SOAP-style sections) are relevant for downstream section extraction. Subsections, physiological monitoring fields, laboratory variables, and administrative metadata are captured by the regex but will be ignored in the canonical mapping.

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

The ranked frequency list of the 300 most common header candidates provided an empirical view of the header distribution across the dataset. Manual inspection revealed that the detected headers fall into several structural categories, but only a subset corresponds to true narrative clinical sections, which are relevant for section extraction. Key observations include:

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

This approach ensures that section extraction captures meaningful narrative blocks while excluding non-narrative or structured artifacts embedded within clinical notes.

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

#### 4.1 Canonical-Only Header Detection

Based on empirical validation and observed failure modes, the section detection strategy adopts a canonical-only detection approach:

1. Only headers present in the predefined canonical header set are detected.
2. No broad regex-based detection of non-canonical headers is performed.
3. Section boundaries are defined exclusively by canonical headers.

This ensures that only clinically meaningful narrative sections are used to structure the document, avoiding interference from subsection headers, monitoring variables, or administrative fields.

---

#### 4.2 Simplified Header Matching Strategy

Header detection is implemented using a deterministic string-based approach rather than general regex pattern matching.

The function `match_canonical_header()` supports two formats:

1. **Colon-terminated headers with optional inline content**

   - Matches lines where a canonical header appears before a colon.
   - Any text after the first colon is treated as inline section content.

   Examples:
   - `Plan:`
   - `Chief Complaint: Chest pain`

2. **Standalone headers**

   - Matches lines that exactly correspond to a canonical header (case-insensitive).
   - No inline content is present.

   Examples:
   - `Chief Complaint`
   - `HPI`

This approach ensures precise and deterministic header recognition while avoiding false positives from non-canonical patterns.

---

#### 4.3 Canonical Header Normalisation

Section headers in clinical notes frequently vary in capitalisation and formatting (e.g., `Assessment`, `ASSESSMENT`, `assessment`). To ensure consistent representation across notes:

- All detected headers are normalised to lowercase before comparison.
- A predefined canonical header set (`CANONICAL_HEADER_SET`) is stored entirely in lowercase.
- Extracted sections are stored using the canonical lowercase representation.

This guarantees that the same section is represented consistently across all documents.

---

#### 4.4 Structural Boundary Rule (Canonical-Only)

Section boundaries are defined exclusively by canonical headers:

1. When a canonical header is detected:
   - The current section (if one is active) is finalised and stored.
   - A new section is started.

2. If a line does not match a canonical header:
   - It is treated as normal content.
   - It does not terminate the current section.

Non-canonical header-like patterns (e.g., `HR`, `Cardiovascular`, `HEENT`) are therefore treated as plain text and retained within the current section.

This prevents premature section termination and preserves full narrative continuity.

---

#### 4.5 Inline Header Content Handling

Clinical documentation frequently places section content on the same line as the header.

Example: `Chief Complaint: Chest pain for two days`

- The line is split on the first colon.
- The portion after the colon is treated as inline content.
- This content is added as the first entry in the section buffer.

This ensures that no clinically relevant information is lost during extraction.

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

The section extraction algorithm implemented in `section_extraction.py` processes each clinical note sequentially using a deterministic line-based parsing strategy with two functions: `match_canonical_header()` and `extract_sections()`.

The workflow is as follows:

1. **Split the clinical note into lines**

   - The note is divided into individual lines using newline separation.

2. **Detect canonical headers**

   - Function: `match_canonical_header()`
   - Each line is checked for an exact match against the canonical header set.
   - Supports:
     - Colon-terminated headers with optional inline content
     - Standalone headers

3. **Handle canonical header detection**

   - If a canonical header is detected:
     - The current section (if active) is finalised:
       - Buffered lines are joined into a single string (`content`)
       - If the current header already exists in the dictionary:
         - The new content is appended using a space separator
       - Otherwise:
         - A new dictionary entry is created
     - A new section is initialised
     - The buffer is reset

4. **Capture inline header content**

   - If inline text exists after the colon:
     - It is added immediately to the buffer as the first line of the new section

5. **Accumulate section content**

   - If no header is detected:
     - The line is treated as content
     - Leading/trailing whitespace is stripped
     - Empty lines are ignored
     - Non-empty lines are appended to the buffer

6. **Repeat until end of note**

   - Continue processing line-by-line, maintaining the current section context

7. **Finalise the last section**

   - After processing all lines:
     - Any remaining buffered content is finalised:
       - Joined into a single string (`content`)
       - If the current header already exists in the dictionary:
         - The content is appended using a space separator
       - Otherwise:
         - A new dictionary entry is created

8. **Return structured output**

   - Output is a dictionary:
     - **Keys**: canonical headers (lowercase)
     - **Values**: concatenated section text

This workflow reflects a strictly deterministic, canonical-boundary approach that prioritises structural robustness and preservation of narrative content.

---

### 7. Section Extraction Manual Validation

#### 7.1 Overview

Validation was performed using `validate_section_extraction.py`, which applies the section extraction function to a reproducible random sample of ICU clinical notes.

The validation combines:
- Qualitative inspection (manual review of extracted sections)
- Quantitative diagnostics (basic statistics on extraction behaviour): 

A sample of 30 ICU notes was evaluated.

---

#### 7.2 Key Findings

- **Notes with no detected sections**: 11 / 30 (≈37%)  
  - Notes without any canonical headers correctly resulted in zero extractions.  
  - This is consistent with the intended behaviour: zero extraction only occurs when no relevant headers are present.

- **Empty sections detected**: 13  
  - Certain canonical headers were retained even when no text followed them, as required by the extraction rules.

- **Extraction behaviour matches expectations**:
  - Canonical headers are correctly retained, even if empty.
  - Notes with no canonical headers yield zero extraction, which is appropriate.
  - All extracted sections align with their source content; there are no missing headers where they exist.

---

#### 7.3 Interpretation

- The observed zero-extraction rate (~37%) reflects natural variability in ICU notes and does not indicate a failure of the extraction script.
- This validation confirms that the section extraction is accurate, complete, and consistent with the defined canonical headers.

---

## Sentence Segmentation

### 1. Objective

Segment section-level clinical text into sentence-level units to enable:
  
- Precise entity span alignment for deterministic extraction
- Sentence-level context for downstream validation (Phase 3)

Bridges structural parsing (sections) and semantic extraction (entities) and preserves character offsets relative to original section text.

---

### 2. Sentence Segmentation Decisions

#### 2.1 Design Decisions

- **Post-section segmentation:** Applied after section extraction to preserve structural context
- **Deterministic & reproducible:** Sentence spans and offsets do not change across runs
- **Section-level granularity:** Enables targeted entity extraction per section

#### 2.2 Rationale for NLTK

Regex-based splitting (on periods or newlines) is unreliable in ICU notes due to:

- Abbreviations (Pt., Dr., numeric units)
- Dense numeric and procedural data
- Inconsistent punctuation or missing spaces

SpaCy offers high-accuracy segmentation but:

-	Requires heavier dependency and memory overhead
- Designed for general text and may not perform well on noisy clinical notes without custom training
- NLTK’s Punkt tokenizer provides a deterministic, rule-based approach that is sufficient for rule-based deterministic extraction
-	NLTK spans integrate seamlessly with current offset-based pipeline

---

### 3. Workflow & Implementation

1. **Section Extraction:**  
  Apply `extract_sections()` to obtain section-level text. Each section is a key-value pair (`header` → `text`).

2. **Sentence Tokenization:**  
  Use NLTK’s `sent_tokenize()` (Punkt tokenizer) to split each section independently into sentences.

3. **Offset Mapping:**  
  - Initialize a `cursor = 0` pointer at the start of the section.  
  - Loop through tokenized sentences:
    - Locate sentence in original section starting from `cursor` (`start = text.find(sent, cursor)`)
    - Compute `end = start + len(sent)`
    - Append `{ "sentence": sent, "start": start, "end": end }` to output
    - Move `cursor = end` to prevent duplicate matches

4. **Output:**  
  List of dictionaries for each sentence:
   ```json
   {
     "sentence": "string",
     "start": 0,
     "end": 0
   }

5. **Notes on implementation:**
  - Works per section to maintain context
  - Does not modify original text
  - Supports deterministic span alignment for regex-based entity extraction
  - Offsets are relative to section text, not the full note

---

### 4. Validation 

Validation was implemented via `validate_sentence_segmentation.py`, which applies the section extraction and sentence segmentation function to a random sample of 10 ICU notes and manually verifies the correctness of sentence splitting and offset alignment.

- Print sections and sentence spans with start:end -> sentence
- Verify correct sentence splitting
- Check alignment of offsets with original text
- Ensure robustness to long sentences and dense numeric/clinical data

**Observed characteristics:**

- Long sentences (1000+ characters) occur in structured sections (e.g., [PHYSICAL EXAMINATION])
- NLTK handles typical clinical sentence boundaries accurately
- Optional post-processing can split excessively long sentences if needed

**Conclusion:**

- NLTK-based sentence segmentation is deterministic, accurate, and lightweight
- Preserves both text content and offset integrity
- Provides sentence-level context needed for downstream entity extraction and validation

---

## Entity Schema Operationalisation

### 1. Objective

**Purpose**
- Define the scope of entity extraction: `SYMPTOM`, `INTERVENTION`, `COMPLICATION`
- Establish a structured schema for downstream validation and analysis
- Constrain scope to prevent uncontrolled expansion of entity types

**Key Principles**
- Deterministic extraction (rule-based candidate generation)
- Contextual validation (transformer-based filtering)
- Prioritise precision over recall
- Maintain auditability and traceability through structured outputs

---

### 2. System Overview (High-Level Architecture)

**Pipeline Structure**
1. Section-aware preprocessing and segmentation  
2. Rule-based candidate generation (deterministic extraction)  
3. Transformer-based validation (contextual classification)  

**Core Design Philosophy**
- Rules generate candidates, not final truths  
- The transformer performs contextual classification of candidate validity  
- The system is explicitly designed as a hybrid pipeline, separating extraction from interpretation  

**Entity-wise Responsibility Split**

| Entity Type   | Rule Strength | Transformer Role        |
|---------------|--------------|------------------------|
| `SYMPTOM`       | Strong       | Refinement             |
| `INTERVENTION`  | Moderate     | Filtering              |
| `COMPLICATION`  | Weak         | Primary classification |

**Key Interpretation**
- `SYMPTOM`: rules capture most valid cases; transformer corrects errors in negation  
- `INTERVENTION`: rules act as broad candidate generators; transformer determines true clinical validity by removing non-performed/planned cases  
- `COMPLICATION`: rules act as broad candidate generators; transformer determines true clinical validity by removing historical/resolved cases and confirming acuity

---

### 3. JSON Schema (Structure Only)

The extraction pipeline outputs one JSON object per entity. This ensures auditability, traceability, and compatibility with downstream validation and analysis.

Each entity follows a two-layer schema:
- **Extraction layer** (deterministic rules)
- **Validation layer** (transformer classification)

```json
{
  "note_id": "string",
  "subject_id": "string",
  "hadm_id": "string",
  "icustay_id": "string",

  "entity_text": "string",
  "concept": "string",
  "entity_type": "SYMPTOM | INTERVENTION | COMPLICATION",

  "char_start": 0,
  "char_end": 0,
  "sentence_text": "string",
  "section": "string",

  "negated": true | false | null,

  "validation": {
    "is_valid": true | false,
    "confidence": 0-1,
    "task": "symptom_presence | intervention_performed | complication_active"
  }
}
```

**Schema Layers**

| Layer        | Description |
|--------------|-------------|
| Metadata     | Identifiers linking entity to note and patient (IDs)|
| Extraction   | Raw entity output from rule-based system |
| Provenance   | Text span and contextual location within the note |
| Signal       | Lightweight rule-derived signal (negation) |
| Validation   | Transformer-based contextual classification |

**Field Groups**

| Group        | Fields |
|--------------|--------|
| Metadata     | `note_id`, `subject_id`, `hadm_id`, `icustay_id` |
| Extraction   | `entity_text`, `concept`, `entity_type` |
| Provenance   | `char_start`, `char_end`, `sentence_text`, `section` |
| Signal       | `negated` |
| Validation   | `is_valid`, `confidence`, `task` |

---

### 4. Core Design Decisions

#### 4.1 Multiple Entities per Note
- One JSON object is generated per entity
- A single note may yield multiple entities across all types
- Note-level identifiers enable grouping and reconstruction at patient or encounter level

#### 4.2 Concept vs Surface Form
- `entity_text` represents the exact span extracted from the note
- `concept` represents the normalised clinical meaning
- This separation enables standardisation while preserving original text

#### 4.3 Section Awareness
- Each entity is linked to its originating section (e.g., HPI, Assessment)
- Section context constrains extraction and supports downstream filtering and analysis

#### 4.4 Provenance and Traceability
- `char_start` and `char_end` preserve exact text alignment within the note
- `sentence_text` provides local context for validation and auditing 
- `section` records structural section origin within the document

---

### 5. Negation Strategy (Critical Design Decision)

#### 5.1 Core Principle

Negation is not uniformly informative across entity types. Its usefulness depends on whether it directly answers the underlying clinical question for that entity.

---

#### 5.2 Entity-Level Interpretation

| Entity Type   | Question Being Answered        | Role of Negation |
|---------------|------------------------------|------------------|
| SYMPTOM       | Is it present?               | Strong signal    |
| INTERVENTION  | Was it performed?            | Weak signal      |
| COMPLICATION  | Is it active?                | Weak signal      |

- Negation can be detected in all entity types  
- However, only for `SYMPTOM` does it directly correspond to the target variable  
- For `INTERVENTION` and `COMPLICATION`, negation captures only a narrow subset of cases and does not reflect the full decision space  

---

#### 5.3 Why Negation Works for SYMPTOM

Negation directly answers the clinical question:

- "no chest pain" → absent  
- "chest pain" → present  

**Conclusion**
- Negation provides a high-value, low-cost signal  
- It approximates ground truth in most cases  
- It is therefore used as a primary feature for symptom extraction and validation  

---

#### 5.4 Why Negation Fails for INTERVENTION and COMPLICATION

Negation only captures a limited subset of interpretations:

| Phrase                     | True Interpretation     | Negation Sufficient |
|---------------------------|-------------------------|---------------------|
| no intubation             | not performed           | ✔                   |
| intubation planned        | not performed           | ✘                   |
| may require intubation    | not performed           | ✘                   |
| intubated                 | performed               | ✔                   |

| Phrase                     | True Interpretation     | Negation Sufficient |
|---------------------------|-------------------------|---------------------|
| no sepsis                 | not active              | ✔                   |
| history of sepsis         | not active              | ✘                   |
| resolved sepsis           | not active              | ✘                   |
| ?sepsis                   | uncertain               | ✘                   |

**Conclusion**
- Negation does not capture temporality, intent, or uncertainty  
- It is therefore incomplete and unreliable as a primary signal  

---

#### 5.5 Final Design Decision

| Entity Type   | negated Field |
|---------------|--------------|
| `SYMPTOM`       | `true` / `false` |
| `INTERVENTION`  | `null`         |
| `COMPLICATION`  | `null`         |

**Rationale**
- Use negation only where it is semantically aligned with the task  
- Avoid introducing misleading or incomplete signals  
- Maintain a consistent schema while restricting interpretation  

---

#### 5.6 Why Not Expand Rule-Based Context (Planned / History / Resolved)?

- Requires complex and brittle rule sets  
- Fails on linguistic variability and unseen phrasing  
- Does not generalise reliably across notes  

**Key Decision**
- Do not encode extended context (e.g., planned, historical, resolved) using rules  
- Delegate contextual interpretation to the transformer, which operates on full sentence context  

---

### 6. Transformer Validation Layer

#### 6.1 Purpose
- Perform contextual validation of rule-extracted candidates
- Operates as inference-only classification (no training)
- Converts candidate spans into clinically meaningful, context-aware decisions

**Key Clarification**
- Validation ≠ ground truth  
- Validation ≠ extraction confidence  
- Validation = contextual correctness of the candidate entity

---

#### 6.2 Validation Tasks

Each entity type maps to a specific classification task:

| Entity Type   | Task                     |
|---------------|--------------------------|
| `SYMPTOM`       | `symptom_presence`         |
| `INTERVENTION`  | `intervention_performed`   |
| `COMPLICATION`  | `complication_active`      |

These tasks define what the model is actually deciding, not just whether the span exists.

---

#### 6.3 Input–Output Formulation

**Input to model**
- Sentence context (`sentence_text`)
- Candidate entity (`entity_text`)

**Output**
```json
{
  "is_valid": true,
  "confidence": 0.0,
  "task": "..."
}
```

---

#### 6.3 Output Interpretation

| Field        | Meaning |
|--------------|--------|
| `is_valid`     | Binary judgement of clinical validity in context |
| `confidence`   | Model confidence in that judgement |
| `task`         | Defines what “valid” means for the entity type |

- `is_valid` is the primary decision variable
- `confidence` supports thresholding, ranking, and error analysis
- `task` ensures interpretation is aligned with entity semantics

---

#### 6.5 Role in the Pipeline

The system is explicitly hybrid:

- Rules → generate candidates (high precision, limited understanding)
- Transformer → validate candidates (contextual understanding)

Division of responsibility

| Component   | Responsibility |
|-------------|----------------|
| Rules       | Pattern matching, span extraction, candidate generation |
| Transformer | Context understanding, disambiguation, classification |

This separation prevents overfitting rules to linguistic complexity.

---

#### 6.6 Role by Entity Type

**SYMPTOM**

- Refines already strong rule outputs
- Corrects negation and contextual misclassification
- Ensures true symptom presence

**INTERVENTION**

- Filters candidates to retain only performed actions
- Removes plans, recommendations, and hypotheticals
- Handles intent vs execution

**COMPLICATION**

- Acts as the primary classifier
- Distinguishes acute vs historical vs resolved conditions
- Performs most of the semantic interpretation

---

#### 6.7 Why Transformer Validation is Necessary

Rule-based extraction cannot reliably capture:

- Context (e.g., history vs current)
- Temporality (acute vs resolved)
- Intent (planned vs performed)
- Uncertainty (suspected vs confirmed)

Attempting to encode these with rules leads to:

- High complexity
- Poor generalisation
- Fragility to phrasing variation

Key Decision:

- Delegate contextual reasoning to the transformer, which operates on full sentence context

---

#### 6.8 System Behaviour Summary

| Entity Type   | Rule Strength | Transformer Reliance  |
|---------------|--------------|------------------------|
| `SYMPTOM`      | Strong       | Medium (refinement)   |
| `INTERVENTION`  | Moderate     | High (filtering)     |
| `COMPLICATION`  | Weak         | Very High (primary classification)  |

- The system is intentionally asymmetric across entity types  
- Transformer reliance increases as rule reliability decreases  
- This reflects the differing complexity of each extraction task  

---

### 7. Entity Type Definitions (Concise + Non-Redundant)

#### 7.1 Included Entity Types

Extraction in Phase 2 is strictly limited to three entity types that separate core components of clinical reasoning and provide a structured representation of ICU notes:

- **SYMPTOM** → patient state (what is happening to the patient)  
- **INTERVENTION** → clinical actions (what is being done)  
- **COMPLICATION** → adverse pathology (what has gone wrong)  

This constraint ensures:
- High precision and controlled rule design  
- Clear separation between entity types  
- Clinically meaningful but tractable coverage 

---

#### 7.2 SYMPTOM

**Definition**
- Patient-reported complaints or clinician-observed manifestations  

**Include**
- Subjective symptoms (e.g., pain, nausea)  
- Observable clinical states (e.g., confusion, agitation)  

**Exclude**
- Laboratory values  
- Imaging findings  
- Diagnoses or conditions  

**Rules Role**
- Strong, high-precision extraction  

**Transformer Role**
- Contextual refinement (negation + disambiguation)  

**Boundary Notes**
- delirium → `SYMPTOM`   
- agitation → `SYMPTOM`   
- hypotension / tachycardia → not `SYMPTOM` 

---

#### 7.3 INTERVENTION

**Definition**
- Therapeutic or procedural actions performed on the patient  

**Include**
- Procedures (e.g., intubation, line insertion)  
- Treatments initiated (e.g., medications started)  

**Exclude**
- Planned or hypothetical actions  
- Recommendations or considerations  
- Chronic/background treatments not initiated in ICU context  

**Rules Role**
- Moderate-precision candidate generation (pattern-restricted)  

**Transformer Role**
- Filtering: performed vs planned/hypothetical  

**Boundary Notes**
- “started X”, “placed on Y” → `INTERVENTION`  
- planned or suggested actions → not extracted 

---

#### 7.4 COMPLICATION

**Definition**
- Acute or active pathological conditions during ICU stay  

**Include**
- New or ongoing clinically significant conditions  
- Acute complications or reasons for ICU admission  

**Exclude**
- Historical conditions  
- Chronic baseline diagnoses without acute change  
- Resolved conditions  

**Rules Role**
- Broad candidate generation (low precision by design)  

**Transformer Role**
- Primary classification (acute vs historical vs resolved)  

**Boundary Notes**
- AKI, sepsis, pneumothorax → `COMPLICATION`  
- chronic conditions → not extracted unless acute worsening is indicated 

---

### 8. Excluded Entity Types (Scope Control)

#### 8.1 Medications

- Extremely large and heterogeneous category (naming, dosing, formulations)  
- Requires extensive ontology and normalization  
- High false positive risk (e.g., chronic meds, allergies, plans)  
- Cannot be reliably captured with simple deterministic rules  

**Decision**
- Excluded due to complexity and low precision under rule-based extraction  

---

#### 8.2 Vital Signs

- Frequently embedded as semi-structured or tabular text  
- Highly variable formatting and units  
- Difficult to extract consistently with rules  

**Additional Constraint**
- Already available in structured EHR data  

**Decision**
- Excluded due to redundancy and extraction unreliability  

---

#### 8.3 Laboratory Values

- Complex structure (values, units, reference ranges)  
- Requires interpretation beyond simple span extraction  
- High variability in formatting within notes  

**Additional Constraint**
- Already captured in structured EHR datasets  

**Decision**
- Excluded due to complexity, low reliability, and lack of added value  

---

#### 8.4 Summary Rationale

All excluded types share at least one of the following:
- Require complex normalization or interpretation  
- Are poorly suited to deterministic rule-based extraction  
- Are already available in structured form  

**Key Principle**
- Prioritise high-precision, clinically meaningful entities that benefit from text extraction  

---

### 9. Design Trade-offs and Justification

The system adopts a hybrid architecture (rules + transformer) to balance precision, control, and contextual understanding.

**Why hybrid**
- Rules provide high-precision, interpretable candidate extraction
- Transformer provides contextual validation and disambiguation
- Combined approach = controlled + context-aware

**Limitations of rule-only systems**
- Brittle to linguistic variation  
- Cannot capture context, temporality, or intent  
- Requires exponential rule complexity for marginal gains  

**Limitations of ML-only extraction**
- Reduced interpretability and traceability  
- No deterministic baseline for debugging  
- Harder to audit, validate, and control outputs  

**Final design position**
- Rules = deterministic candidate generator
- Transformer = context-aware validation layer 

This separation ensures:
- High precision at extraction stage  
- Robust handling of clinical language variability  
- Maintainable and auditable system design  

---

### 10. Summary

**Scope**
- Extraction limited to: `SYMPTOM`, `INTERVENTION`, `COMPLICATION`  
- Ensures high precision, clear boundaries, and manageable rule design  

**System Components**
- Rule-based extraction (candidate generation)  
- Negation handling (SYMPTOM only)  
- Transformer validation (contextual classification)  

**Conceptual Model**
- Rules → propose candidates  
- Transformer → determine clinical validity  

**Outputs**
- One structured JSON object per entity  
- Span-aligned and traceable  
- Context-aware (sentence + section)  
- Model-validated (`is_valid`, `confidence`, `task`)  

**Final Position**
- Deterministic, section-aware extraction pipeline  
- Hybrid system combining rule precision with model-based context understanding  

---

## Rule Based Extraction

### 1. Objective

The objective of rule-based extraction is to provide a controlled, deterministic, high-precision, and schema-constrained method for identifying specific entity types within clinical text.

- Following section extraction and sentence segmentation, this stage defines three sets of deterministic rules to extract the entities: **SYMPTOM, INTERVENTION, COMPLICATION**
- Rules are based on prototypical ICU language and clear lexical patterns, rather than attempting full linguistic coverage
- The component functions as a candidate generation layer within a hybrid pipeline, not a full clinical NLP system
- Outputs are structured, stable, interpretable, and span-aligned, making them suitable for downstream transformer validation

---

### 2. Rule-Based Extraction Decisions

Rule development is fully constrained by prior schema operationalisation, which defines what should be extracted, what should be excluded, and how outputs must be structured. This removes ambiguity and limits the scope of rule design to clearly defined entity boundaries. As a result:

- Rules focus only on high-confidence, clearly expressed patterns
- Edge cases, ambiguity, and complex language are intentionally not handled at this stage
- Responsibility for deeper interpretation is deferred to the transformer layer

Rules were constructed through targeted inspection of sectioned ICU notes, identifying common and generalisable phrasing patterns and translating them into regex-based rules. This process is guided by theory-guided clinical intuition and pattern recognition rather than dataset optimisation. The approach is intentionally constrained:

- No attempt at exhaustive coverage or ontology expansion, with minimally iterative rule design 
- No tuning to maximise dataset performance, not data-fitted or over-optimised to specific note characteristics
- No encoding of complex linguistic structure (e.g. full negation scope, uncertainty), stopped when additional rules would yield diminishing returns or require complex, brittle logic

This ensures the system remains:

- Deterministic and reproducible 
- Interpretable and easy to reason about 
- Robust to overfitting and dataset-specific quirks, preserving generalisability

Overall, the rule-based layer provides a limited but reliable extraction mechanism, designed to generate clean, structured candidates for downstream validation rather than solve the full extraction problem.

---

### 3. SYMPTOM Extraction

Rule-based symptom extraction identifies patient-reported clinical features using deterministic, concept-level regex patterns with lightweight negation handling.

---

#### 3.1 Extraction Decisions

**A. Section scope**

- Symptom extraction is restricted to just 3 relevant sections: **chief complaint**, **hpi**, **review of systems**
- These sections were selected because they contain the majority of subjective, patient-reported symptoms. 
- Other sections are excluded to avoid extracting non-symptom content and reduce false positives. This enforces contextual precision at the earliest stage.

---

**B. Concept-based pattern design**

Symptoms are modelled as concepts, each mapped to a set of synonymous lexical patterns (regex).

- One clinical concept → many lexical variants (e.g. “shortness of breath”, “SOB”, “dyspnoea” → dyspnoea)
- Patterns are designed to capture common ICU phrasing, not all possible expressions

A total of 17 symptom concepts were defined:

- Guided by clinical heuristics and expert judgement
- Anchored to common ICU presentations (e.g. Maryland short-stay critical care study, PMID: 28323374)

Limited deliberately to a small set of high-yield, representative features:

- This avoids ontology construction as the goal is representative coverage of common patterns, not exhaustive medical vocabulary  
- Restricting the concept set improves precision, interpretability, and stability of outputs  

---

**C. Token–character alignment**

Regex matching operates on character indices, while negation operates on tokens (words). A mapping step is therefore required.

- Sentences are tokenised into words
- Each token is mapped to its character span e.g., `[("chest", 0, 5), ("pain", 6, 10)]`
- The regex match start index is aligned to the corresponding token index (e.g., match at char index 6 → token index 1)

This allows:
- Locating the matched symptom within the sentence as a word position (token index)
- Enabling token-based negation logic

Without this step, negation detection cannot be applied correctly.

---

**D. Global span alignment**

Regex matches are found at the sentence level, but outputs must align to the original note section.

- Sentence-relative indices are converted to note-level character offsets 
- Achieved by adding the sentence start offset to match indices (e.g., match at sentence char index 6, sentence starts at section char index 100 → global char index 100 + 6 = 106)

This ensures:
- Exact span traceability to the original note text allowing inclusion in the final JSON output
- Compatibility with downstream validation and annotation  

---

**E. Negation detection (design and scope)**

A lightweight, token-based negation rule is applied:

- Scan tokens preceeding the matched symptom within the same sentence (using its token index)
- Activate negation if a term is found:`no`, `denies`, `without`, `negative`, etc.
- Deactivate if a break term appears: `but`, `however`, etc.

**What it captures well:**
- “no chest pain” → negated `pain` concept
- “denies nausea” → negated `nausea` concept
- “without fever” → negated `fever` concept

**Limitations:**
- Does not capture conjunction scope (“no chest pain or SOB”)  
- Does not model syntax (“not complaining of pain”)  
- Does not capture uncertainty (“unlikely to have pain”)  

This simplification is intentional:
- Negation is treated as local, linear, and word-triggered, capturing the most common, high-yield patterns cheaply  
- Complex contextual interpretation is delegated to the transformer  

The rule provides a fast, high-precision signal rather than full linguistic modelling.

---

**F. Deduplication (per sentence)**

A constraint is applied of maximum one instance per concept per sentence

- Implemented using a per-sentence `seen_concepts` set  
- Prevents duplicate matches from overlapping or repeated patterns  

Rationale:
- Regex cannot resolve semantic duplication or overlap  
- Prevents repeated extraction of the same concept within a sentence  

Limitation:
- Multiple mentions of the same concept in a sentence collapse to one 
- Example: “chest pain and abdominal pain” → one `pain` concept extracted  

This is accepted because:
- The system operates at concept level, not span enumeration  
- Precision and stability are prioritised over exhaustive extraction  

Concepts may still repeat across sentences, preserving document-level signal.

---

**Overall design rationale**

The symptom extraction component explicitly avoids:

- Full linguistic modelling  
- Exhaustive symptom coverage  
- Ontology or hierarchy construction  
- Dataset-specific optimisation  

The symptom extraction logic intentionally combines:

- Controlled lexical matching  
- Clear concept mapping  
- Minimal, interpretable rules  
- Stable integration within the pipeline  

The result is a high-precision, concept-level candidate generator that:

- Captures clear and common symptom expressions  
- Avoids encoding complex language rules  
- Produces structured, span-aligned outputs ready for transformer validation  

---

#### 3.2 Workflow Implementation

All code logic is implemented in `symptom_rules.py`, which defines the functions `map_char_to_token()`, `is_negated_simple()`, and `extract_symptoms()`. The main function `extract_symptoms()` applies the extraction across all sentences in the target sections of a note. 

**Workflow**

1. **Section filtering**  
  - Input text is processed only if its section belongs to the predefined symptom-relevant set (Chief Complaint, HPI, ROS).  
  - Non-target sections are skipped entirely.

2. **Sentence segmentation**  
  - Section text is split into sentences using `split_into_sentences()`.  
  - Each sentence retains start/end character offsets relative to the original text.

3. **Per-sentence initialisation**  
  - For each sentence:  
    - Original text and offsets are stored  
    - A lowercased version is created for regex matching  
    - Token–character mapping is generated using `map_char_to_token()`  
    - A per-sentence `seen_concepts` set is initialised for deduplication  

4. **Concept-level pattern matching**  
  - For each predefined symptom concept:  
    - Iterate through its associated regex patterns  
    - Apply `re.search()` to detect the first match in the sentence  
    - Skip concept if already matched in the current sentence  

5. **Span extraction and alignment**  
  - If a match is found:  
    - Convert sentence-level match indices → global character offsets  
      - `global_start = sentence_start + match.start()`  
      - `global_end = sentence_start + match.end()`  
    - Extract exact text span from the original note  

6. **Token index identification**  
  - Identify which token corresponds to the start of the matched span by comparing character ranges  
  - This provides the token index required for negation detection  

7. **Negation detection**  
  - Apply `is_negated_simple()` using tokens and the identified token index  
  - Scan preceding tokens to determine whether negation is active  
  - Assign `negated = True / False`  

8. **Deduplication enforcement**  
  - Once a concept is matched in a sentence:  
    - It is added to `seen_concepts`  
    - Further matches of the same concept in that sentence are ignored  

9. **Entity construction**  
  - A structured SYMPTOM entity is created containing:  
    - Identifiers (note, subject, admission, ICU stay)  
    - Extracted span (`entity_text`, `char_start`, `char_end`)  
    - Concept label  
    - Context (`sentence_text`, `section`)  
    - Negation flag  
    - Validation placeholder for downstream transformer  

10. **Aggregation and output**  
  - All extracted entities across sentences are collected into a list  
  - The final output is a list of structured `SYMPTOM` entities, ready for downstream validation  

---

#### 3.3 Validation Metrics and Manual Sample Analysis

Validation was performed using `validate_symptom_rules.py` on a random sample of 30 ICU notes. The objective was to verify that the rule-based SYMPTOM extraction behaves as designed under realistic conditions, focusing on section filtering, extraction behaviour, concept mapping, negation handling, and span alignment.

**Validation Logic**

1. **Sampling**
  - Random sample of ICU notes from the corpus
  - Ensures representative variation in structure and content

2. **Section Extraction**
  - Notes are parsed into sections
  - Only 3 target sections (Chief Complaint, HPI, ROS) are processed for symptom extraction

3. **Symptom Extraction**
  - Each target section is processed independently
  - Pipeline:
    - Sentence segmentation
    - Regex-based symptom detection
    - Token-level negation detection
    - Per-sentence concept deduplication

4. **Tracking and Metrics**
  - Section coverage
  - Extraction yield (per note and total)
  - Concept distribution
  - Negation behaviour

5. **Qualitative Inspection**
  - Raw outputs printed per note for manual inspection
  - Enables verification of span accuracy, concept mapping, and negation

---

**Key Findings**

**A. Section Coverage**

- Notes with any sections: 19 / 30  
- Notes with target sections: 9 / 30  
- Notes without target sections: 21 / 30 

This confirms:
- Section filtering is functioning correctly
- Extraction is appropriately restricted to clinically relevant regions
- Low coverage reflects variability in note structure rather than extraction failure

---

**B. Extraction Yield**

- Total symptoms extracted: 21  
- Average per note (with target sections): 2.33  
- Notes with sections but no symptoms: 3 / 9 (33.3%)

Extraction is conservative, consistent with a high-precision candidate generation design.  
Absence of symptoms in some notes reflects either true absence of symptom language or mismatch with predefined patterns, not uncontrolled behaviour.

---

**C. Concept Distribution**

- Dominant concept: `pain` (9)
- Secondary concepts: `syncope` (3), `cough` (2)
- Broad coverage across multiple symptom categories

This distribution reflects:
- Expected ICU presentation patterns (pain-dominant)
- Correct functioning of concept-level mapping (multiple phrases → single concept)

Example (Note 3):
- “chest pain” → pain  
- “LOC” → syncope  

---

**D. Negation Behaviour**

- Negated: 8 (38.1%)  
- Not negated: 13 (61.9%)

**Correct handling of simple negation patterns**
- Note 2: “Fever” → negated=True (from “No(t) Fever”)  
- Note 19:  
  - “n/v” → negated=True  
  - “SOB” → negated=True  
  - “cough” → negated=True  
  - “palpitations” → negated=True  

**Correct handling of positive symptoms in same note**
- Note 19:  
  - “Abdominal Pain” → negated=False  
  - “fatigue” → negated=False  

This demonstrates:
- Negation is applied locally and independently per mention  
- Mixed positive/negative symptom states are preserved correctly  

Limitations observed (expected):
- Negation is strictly pre-scope and linear, missing more complex patterns (e.g., conjunctions, syntax-based negation)
- Does not capture coordination or complex phrasing 

These limitations are consistent with the intended design and are delegated to the transformer layer.

---

### Overall Assessment

The validation confirms that the system:

- Correctly restricts processing to relevant clinical sections  
- Reliably extracts symptom mentions using deterministic rules  
- Maintains accurate character-level span alignment  
- Applies consistent and interpretable negation logic  
- Produces structured outputs aligned with downstream requirements  

The observed behaviour matches the intended design:

> High-precision, interpretable candidate generation for downstream refinement

---

### Conclusion

The rule-based symptom extraction module is functioning as intended.

- Outputs are **consistent, explainable, and structurally correct**
- Limitations (e.g. simplified negation, regex coverage) are **intentional design choices**
- System is **suitable for integration with transformer-based validation in later stages**

No changes required at this stage.