# Phase 2 - Rule-Based Extraction Summary

## 1. Purpose

This document provides a concise, outcome-focused summary of Phase 2: deterministic rule-based clinical entity extraction.

It consolidates:

- Implementation of preprocessing, section extraction, sentence segmentation, and rule-based entity extraction  
- Validation outcomes at both component and system level  
- Observed pipeline behaviour at scale  
- Confirmed readiness for downstream transformer-based validation (Phase 3)  

`phase2_decisions.md` contains full methodological detail, design rationale, and exploratory analysis.  
This document records the finalised system design, validated behaviour, and confirmed conclusions.

---

## 2. Phase 2 Objective

Phase 2 implements and validates a deterministic pipeline for extracting structured clinical entities from ICU notes.

The primary objective is to:

- Generate high-recall, structured candidate entities
- Maintain controlled precision through constrained rule design
- Produce outputs that are fully compatible with transformer-based validation

Phase 2 establishes:

- Feasibility of rule-based extraction across all target entity types  
- Correctness and stability of the implemented pipeline  
- Suitability of outputs for large-scale downstream processing  

Scope is limited to candidate generation, with no attempt to resolve:

- Contextual interpretation (e.g. temporality, certainty, intent)  
- Advanced linguistic modelling  
- Final classification or validation of extracted entities  

All higher-order reasoning is explicitly deferred to Phase 3.

---

## 3. System Architecture (High-Level)

### 3.1 Overview

The Phase 2 system is a deterministic, rule-based extraction pipeline designed to convert unstructured ICU clinical notes into structured, concept-level entity candidates.

The system operates as a multi-stage transformation pipeline, where raw clinical text is progressively refined into structured outputs through a series of modular processing steps.

Three entity types are extracted:

- `SYMPTOM`
- `INTERVENTION`
- `CLINICAL_CONDITION`

Each entity is represented as a span-aligned, context-aware candidate linked to its original source text.

---

### 3.2 Core Processing Stages

At a high level, the system follows a hierarchical processing flow:

1. **Preprocessing**  
   - Normalises raw clinical text to improve downstream consistency  

2. **Section Extraction**  
   - Segments notes into clinically meaningful sections  
   - Constrains extraction to contextually relevant regions  

3. **Sentence Segmentation**  
   - Divides sections into sentences for fine-grained analysis  
   - Enables precise span-level extraction  

4. **Entity Extraction**  
   - Applies deterministic, concept-based rules to identify candidate entities  
   - Operates independently per entity type  

5. **Candidate Aggregation**  
   - Consolidates all extracted entities into a unified, structured dataset  

---

### 3.3 Design Principles

The system is governed by the following principles:

- **Deterministic Processing**  
  - All extraction logic is rule-based and reproducible  
  - No stochastic or learned components are used in this phase  

- **Hierarchical Decomposition**  
  - Processing follows the natural structure of clinical text:  
    - Document → Sections → Sentences → Entities  

- **High-Recall Candidate Generation**  
  - Extraction prioritises capturing plausible clinical signals  
  - Ambiguity and noise are accepted at this stage  

- **Concept-Level Representation**  
  - Entities are normalised into clinically meaningful concept categories  
  - Multiple lexical forms map to a single concept  

- **Span-Level Traceability**  
  - All entities are linked to exact text spans and their context  
  - Ensures auditability and downstream interpretability  

---

### 3.4 Separation of Responsibilities

The system is designed as part of a two-stage architecture:

| Stage                     | Responsibility |
|--------------------------|---------------|
| Rule-based extraction     | Candidate generation (high recall, deterministic) |
| Transformer validation    | Contextual interpretation and precision filtering |

Key implications:

- Rule-based extraction does not attempt to resolve:
  - Negation (except simple cases in symptoms)  
  - Temporality  
  - Certainty or attribution  
- These are deferred to the transformer in Phase 3  

This separation:

- Reduces rule complexity  
- Improves generalisability  
- Enables modular development and evaluation  

---

### 3.5 Summary

The Phase 2 architecture defines a deterministic candidate generation layer that:
- Transforms unstructured clinical notes into structured entity candidates  
- Preserves full textual context and provenance  
- Prioritises recall and interpretability over completeness or precision  

The resulting outputs form a controlled, high-quality input space for downstream transformer-based validation.

---

## 4. Report Preprocessing

### 4.1 Overview

**Purpose:**  
- Stabilise ICU clinical note text prior to deterministic rule-based extraction. 
- Ensures reproducible, auditable input while preserving semantic content and section structure.

**Files:**
- `src/deterministic_extraction/preprocessing.py`
- `scripts/deterministic_extraction/validation/validate_preprocessing.py`

---

### 4.2 Processing Implementation

**Approach:**  
- Minimal text normalisation applied:
  - Standardised line breaks
  - Removal of de-identification tokens (`[** ... **]`)
  - Collapsed redundant whitespace
  - Removal of trailing EMR references and irrelevant metadata
- Maintains all clinical sections, numeric values, and core structural delimiters required for downstream section and sentence segmentation.

**Validation:**  
- Manual inspection of a random sample of 10 ICU notes confirmed:
  - Successful removal of de-identification artefacts  
  - Section headers and structural signals preserved  
  - Sentence breaks occasionally imperfect but acceptable for deterministic extraction  
  - Text ready for section extraction and rule-based entity extraction

**Conclusion:**  
- Preprocessing achieves its intended stabilisation goal, providing clean, structurally intact notes suitable for downstream deterministic extraction while preserving traceability and auditability.

---

## 5. Section Detection and Extraction

### 5.1 Overview

**Purpose:**  
- Identify and extract top-level narrative sections within ICU clinical notes to enable context-specific deterministic extraction. 
- Section boundaries provide semantic segmentation of the note, restricting downstream entity extraction to relevant narrative blocks while ignoring structured flowsheet data, laboratory fields, or administrative metadata.

**Files:**
- `notes/phase_2/header_pattern_exploration.ipynb`
- `src/deterministic_extraction/section_extraction.py`
- `scripts/deterministic_extraction/validation/validate_section_extraction.py`

---

### 5.2 Section Extraction Implementation

**Approach:**  
- Canonical-only detection:
  - Potential section headers identified and explored to manually curate a fixed set of canonical headers.
  - Only 13 pre-defined narrative headers are used to define section boundaries.  

	```python
	CANONICAL_HEADERS = [
		"plan",
		"assessment",
		"action",
		"response",
		"assessment and plan",
		"chief complaint",
		"hpi",
		"past medical history",
		"family history",
		"social history",
		"review of systems",
		"physical examination",
		"disposition"
	]
	```

  - Non-canonical headers and structured fields (e.g., `HR`, `BP`, `WBC`) are ignored to avoid over-segmentation.  
- Deterministic, line-based parsing:
  - Lines are sequentially processed to detect canonical headers.
  - Inline content after a header is captured and included in the corresponding section.
  - Sections are accumulated until the next canonical header is found.
- Headers are normalised to lowercase for consistency across notes.

**Validation:**  
- Manual inspection of 30 ICU notes confirmed:
  - Correct detection of canonical narrative headers
  - Non-canonical and structured headers do not prematurely terminate sections
  - Sections contain the expected narrative content
  - Notes without canonical headers correctly yield zero sections

**Conclusion:**  
- The canonical-boundary approach provides a robust, deterministic method for segmenting clinical notes into narrative sections. 
- It ensures high recall of clinically relevant text blocks, maintains structural integrity, and establishes a reliable foundation for sentence segmentation and rule-based entity extraction downstream.

---

## 6. Sentence Segmentation

### 6.1 Overview

**Purpose:**  
- Transform section-level clinical text into sentence-level units to support precise entity span alignment and provide sentence-level context for downstream transformer validation. 
- Maintains offsets relative to original section text to ensure deterministic extraction.

**Files:**
- `src/deterministic_extraction/sentence_segmentation.py`
- `scripts/deterministic_extraction/validation/validate_sentence_segmentation.py`

---

### 6.2 Sentence Segmentation Implementation

**Approach:**  
- Applied after section extraction to preserve structural context  
- Deterministic and reproducible segmentation  
- Section-level granularity allows targeted entity extraction per section  
- NLTK Punkt tokenizer used for robust rule-based splitting:
  - Handles abbreviations, numeric data, and inconsistent punctuation common in ICU notes  
  - Provides accurate sentence spans and integrates seamlessly with offset-based pipeline  
- Applied not as a seperate standalone step, but as part of the entity extraction process to ensure that all extracted entities are directly linked to their sentence context and character offsets.

**Validation:**  
- Manual inspection of a random sample of ICU notes confirmed:
  - Sentences are correctly split across diverse clinical sections  
  - Start and end offsets align with the original section text  
  - Robustness maintained for long or densely formatted sentences  

**Conclusion:**  
- Sentence segmentation is accurate, deterministic, and lightweight. 
- It preserves content integrity and character offsets, providing reliable sentence-level units for downstream rule-based entity extraction and Phase 3 transformer validation.

---

## 7. Entity Definitions and Scope

### 7.1 Included Entity Types

Phase 2 extraction is restricted to three entity types representing core components of clinical reasoning:

| Entity Type            | Description                              |
|-----------------------|------------------------------------------|
| **SYMPTOM**           | Patient state (observed or reported)     |
| **INTERVENTION**      | Clinical actions performed               |
| **CLINICAL_CONDITION**| Disease or pathological state            |

This constraint ensures:
- Clear separation between entity types  
- Controlled scope for deterministic rule design  
- Clinically meaningful but tractable coverage  

---

### 7.2 Entity Behaviour Summary

| Entity Type            | Extraction Strategy                  | Key Challenge Addressed by Validation |
|-----------------------|------------------------------------|--------------------------------------|
| **SYMPTOM**           | High-precision rule-based extraction | Negation and contextual presence     |
| **INTERVENTION**      | Moderate-precision, pattern-based   | Performed vs planned/hypothetical    |
| **CLINICAL_CONDITION**| High-recall candidate generation    | Acute vs historical/resolved         |

---

### 7.3 Scope Constraints

The pipeline deliberately excludes:

- Medications (naming variability and normalization complexity)  
- Vital signs (structured and inconsistently formatted in text)  
- Laboratory values (require interpretation and already structured)  

**Rationale:**  
Excluded categories either:
- Require complex interpretation beyond rule-based extraction, or  
- Are already available in structured EHR data  

---

### 7.4 Summary

The entity schema provides a minimal, high-value representation of clinical notes, focusing on:

- Patient state (`SYMPTOM`)  
- Clinical actions (`INTERVENTION`)  
- Disease processes (`CLINICAL_CONDITION`)  

This design balances clinical relevance, extraction feasibility, and downstream interpretability, while avoiding uncontrolled expansion of entity types.

---

## 8. Entity Schema and Output Format

### 8.1 Overview

The Phase 2 pipeline outputs one JSON object per extracted entity, producing a flat, transformer-ready dataset:  

```json
{
  "note_id": "string",
  "subject_id": "string",
  "hadm_id": "string",
  "icustay_id": "string",

  "entity_text": "string",
  "concept": "string",
  "entity_type": "SYMPTOM | INTERVENTION | CLINICAL_CONDITION",

  "char_start": 0,
  "char_end": 0,
  "sentence_text": "string",
  "section": "string",

  "negated": true | false | null,

  "validation": {
    "is_valid": true | false,
    "confidence": 0-1,
    "task": "symptom_presence | intervention_performed | clinical_condition_active"
  }
}
```

Each entity captures all necessary metadata, extraction details, provenance, signals, and placeholders for downstream validation:

| Category       | Fields / Description                                                                                 |
|----------------|-----------------------------------------------------------------------------------------------------|
| **Metadata**   | `note_id`, `subject_id`, `hadm_id`, `icustay_id` – link entity to patient and note                  |
| **Extraction** | `entity_text` – exact text span; `concept` – normalised clinical concept; `entity_type` – `SYMPTOM`, `INTERVENTION`, or `CLINICAL_CONDITION` |
| **Provenance** | `char_start`, `char_end` – offsets in section; `sentence_text` – sentence containing entity; `section` – originating note section |
| **Signal**     | `negated` – only populated for SYMPTOM entities; null for others                                     |
| **Validation** | `is_valid`, `confidence`, `task` – placeholder fields for Phase 3 transformer-based classification |

This schema ensures traceability, reproducibility, and compatibility with downstream transformer validation and analysis.

---

### 8.2 Key Design Implementations

- **One entity per JSON object:** Multiple entities may originate from the same note; flattening enables parallel processing and single-instance transformer input.  
- **Negation handling:**  
  - `SYMPTOM`: `true` / `false`  
  - `INTERVENTION` and `CLINICAL_CONDITION`: `null`  
  - Negation is used only where it provides a meaningful signal for the target clinical question.  
- **Concept vs surface form:** Original span is preserved (`entity_text`) while a normalised clinical concept (`concept`) standardises interpretation.  
- **Section-awareness:** Each entity is linked to the section in which it occurs, supporting contextual filtering and audit.  
- **Provenance preservation:** Exact character offsets and sentence context enable precise traceability from output back to source text.  
- **Output format:** Newline-delimited JSON (`.jsonl`), one row per entity, suitable for batch processing, validation, and transformer-based classification.

---

### 8.3 Summary

- The Phase 2 entity schema consolidates all necessary metadata, extraction spans, and context for downstream processing.  
- It balances deterministic extraction with readiness for contextual validation, while maintaining a simple, auditable, and transformer-compatible structure.

---

## 9. Rule-Based Entity Extraction

### 9.1 Overview

Rule-based extraction generates structured, span-aligned candidate entities from sentence-level clinical text.

- Deterministic and reproducible  
- Schema-constrained to three entity types  
- Designed as a candidate generation layer, not final classification  

The system prioritises:
- Controlled extraction space  
- High recall where required  
- Structured outputs for downstream transformer validation  

---

### 9.2 Entity Type-Specific Strategies

| Entity Type           | Rule Strength | Transformer Role        |
|-----------------------|--------------|------------------------|
| `SYMPTOM`             | Strong       | Refinement             |
| `INTERVENTION`        | Moderate     | Filtering              |
| `CLINICAL_CONDITION`  | Weak         | Primary classification |

- **SYMPTOM:** Rules capture most valid cases by incorporating negation; transformer corrects negation and contextual ambiguity  
- **INTERVENTION:** Rules generate broad candidates with no semantic handling; transformer filters performed actions and handles intent  
- **CLINICAL_CONDITION:** Rules are broadest and prioritise recall; transformer performs primary classification and distinguishes acute vs historical vs resolved

---

## 10. SYMPTOM Extraction

### 10.1 Overview

**Purpose:**  
- Extract patient-reported and clinically observed symptoms as structured, concept-level entities from ICU notes.

**Files:**
- `src/deterministic_extraction/extraction_rules/symptom_rules.py`
- `scripts/deterministic_extraction/validation/validate_symptom_rules.py`

---

### 10.2 Extraction Implementation 

**Scope:**  
- Restricted to: `chief complaint`, `hpi`, `review of systems`  
- These sections contain the majority of subjective symptom reporting and maximise contextual precision  

**Extraction Strategy:**  
- Concept-based rule design:
  - Each symptom concept maps multiple lexical variants to a single normalised label  
  - Focused on common ICU presentations rather than exhaustive coverage
- A fixed set of 17 high-yield symptom concepts is used:
  `pain`, `headache`, `chest_discomfort`, `palpitations`, `dyspnoea`, `syncope`, `nausea_vomiting`, `fatigue`, `dizziness`, `fever`, `cough`, `diarrhoea`, `confusion`, `bleeding`, `weakness`, `seizure`, `anorexia`  

**Key Signal:**  
- Binary negation captured at extraction stage (`present` vs `absent`)  
- Provides a strong, low-cost indicator of symptom presence  
- More complex contextual interpretation is deferred to transformer validation  

**Behaviour:**  
- High-precision, conservative extraction  
- Produces clean, interpretable symptom mentions  
- Avoids complex linguistic modelling and edge-case handling  
- Outputs are span-aligned and fully traceable to source text  

**Validation Outcome:**  
- Extraction is restricted correctly to relevant sections  
- Produces consistent, clinically plausible symptom distributions  
- Negation handling behaves as expected for common patterns  
- Overall behaviour aligns with a precision-focused candidate generation design  

**Role in Pipeline:**  
- Generates reliable symptom candidates  
- Provides structured inputs for downstream validation (`symptom_presence`)  
- Final classification and ambiguity resolution are performed in Phase 3  

---

## 11. INTERVENTION Extraction

### 11.1 Overview

**Purpose:**  
- Extract administered or referenced clinical interventions as structured, concept-level entities from ICU notes.
- Operates as a recall-oriented candidate generation layer, deferring contextual interpretation (performed vs planned vs historical) to Phase 3 transformer validation.

**Files:**
- `src/deterministic_extraction/extraction_rules/intervention_rules.py`
- `scripts/deterministic_extraction/validation/validate_intervention_rules.py`

---

### 11.2 Extraction Implementation 

**Scope:**  
- Restricted to: `action`, `assessment`, `assessment and plan`  
- These sections contain the highest density of intervention-related content while maintaining contextual relevance  

**Extraction Strategy:**  
- Recall-oriented, concept-based rule design:
  - Each intervention concept maps diverse lexical variants (abbreviations, verbs, nouns, phrases) to a single clinical category  
  - Designed to capture heterogeneous ICU documentation rather than enforce strict phrasing constraints  
- A fixed set of 19 ICU-relevant intervention concepts is used:
  `airway_management`, `oxygen_therapy`, `mechanical_ventilation`, `fluid_therapy`, `vasopressor_inotrope`, `analgesia`, `sedation`, `paralysis`, `antibiotic_therapy`, `anticoagulation`, `blood_product`, `renal_replacement_therapy`, `procedure_general`, `surgical_procedure`, `nutrition`, `cardiovascular_support`, `cardiovascular_drugs`, `electrolyte_replacement`, `resuscitation`

**Key Design Choices:**  
- No trigger-word dependency (e.g. “given”, “started”)  
- No negation or semantic filtering at rule stage  
- No distinction between performed, planned, or historical interventions  
- All contextual interpretation is deferred to transformer validation  

**Behaviour:**  
- High-recall, candidate-generation focused extraction  
- Captures multiple mentions and lexical variants per concept  
- Preserves all valid spans (no concept-level deduplication)  
- Outputs are span-aligned and fully traceable to source text  

**Validation Outcome:**  
- High coverage in notes containing relevant sections  
- Clinically realistic intervention density per note  
- Strong capture of ICU shorthand and abbreviations  
- Redundant mentions and contextual ambiguity observed as expected under recall-first design  

**Role in Pipeline:**  
- Generates broad intervention candidate set  
- Provides structured inputs for downstream validation (`intervention_performed`)  
- Final determination of intervention status (performed vs planned vs historical) is handled in Phase 3  

---

## 12. CLINICAL_CONDITION Extraction

### 12.1 Overview

**Purpose:**  
- Extract clinically relevant diagnoses, pathological states, and complications as structured, concept-level entities from ICU notes.

**Files:**
- `src/deterministic_extraction/extraction_rules/clinical_condition_rules.py`
- `scripts/deterministic_extraction/validation/validate_clinical_condition_rules.py`

---

### 12.2 Extraction Implementation 

**Scope:**  
- Restricted to: `assessment and plan`, `assessment`, `hpi`, `chief complaint`  
- These sections maximise diagnostic signal while preserving early admission context  

**Extraction Strategy:**  
- Recall-first, context-agnostic candidate generation:
  - Captures any mention of clinically relevant conditions without interpreting temporality, certainty, or negation  
  - All contextual resolution is deferred to transformer validation  
- Unified entity space:
  - Combines diagnoses, complications, and pathological states into a single entity type  
  - Avoids unreliable rule-based separation of overlapping clinical concepts  

**Concept Design:**  
- 13 high-level ICU-relevant condition categories:
  `infection`, `shock`, `respiratory`, `cardiovascular`, `arrhythmia`, `renal_failure`, `neurological`, `bleeding`, `gastrointestinal`, `metabolic`, `hepatic_failure`, `cardiac_arrest`, `vascular`
- Each concept maps multiple lexical variants (including abbreviations and shorthand) to a single category  
- Focused on common, high-yield diagnostic domains rather than exhaustive disease coverage  

**Key Behaviour:**  
- Broad lexical matching to capture heterogeneous clinical language  
- No negation, temporality, or certainty modelling at rule stage  
- All matched spans are preserved (no concept-level deduplication)  
- Outputs are span-aligned and fully traceable to source text  

**Validation Outcome:**  
- Section filtering operates correctly and captures high-yield diagnostic regions  
- Extraction recall improves significantly at scale, confirming stable behaviour  
- Concept distribution aligns with expected ICU diagnostic domains  
- Abbreviation refinement reduces systematic false positives (e.g. removal of ambiguous terms)  
- Lower extraction density in small samples reflects conservative pattern scope, not system failure  

**Role in Pipeline:**  
- Generates broad clinical condition candidates across diagnostic domains  
- Provides structured inputs for downstream validation (`clinical_condition_active`)  
- Final determination of active vs historical vs negated conditions is performed in Phase 3  

---

## 13. Full Extraction Pipeline

### 13.1 Overview

**Purpose:**  
- Orchestrate all deterministic extractors (`SYMPTOM`, `INTERVENTION`, `CLINICAL_CONDITION`) into a single end-to-end pipeline  
- Generate a high-recall, transformer-ready dataset of candidate entities  
- Preserve full provenance (span, sentence, section, metadata) for downstream validation and auditability  

**File:**  
- `scripts/deterministic_extraction/run_extraction_pipeline.py`
- `data/interim/extraction_candidates.jsonl`

---

### 13.2 Pipeline Design

**Architecture:**  
- Two-stage processing:
  - **Note-level:** preprocessing and section extraction  
  - **Sentence-level:** regex-based entity extraction  

**Output Format:**  
- Flat `.jsonl` (1 row = 1 entity)  
- Each entity is a self-contained classification unit with:
  - `entity_text`, `concept`, `entity_type`  
  - `char_start`, `char_end`  
  - `sentence_text`, `section`  
  - Metadata (`note_id`, `subject_id`, `hadm_id`, `icustay_id`)  
  - Validation scaffold (`is_valid`, `confidence`, `task`)  

**Design Rationale:**  
- Optimised for transformer input (independent instances)  
- Enables batch processing, parallel inference, and simple dataset loading  
- Preserves exact traceability to source text  

**Behaviour:**  
- High-recall aggregation of all entity types  
- No cross-entity interaction or deduplication  
- Produces a unified candidate pool for downstream validation  

---

### 13.3 Workflow

1. Load ICU notes and validate required fields  

2. Sample dataset (10,000 notes for scalable validation)  

3. For each note:
   - Generate `note_id` and extract metadata  
   - Preprocess text and extract sections  

4. For each section:
   - Apply:
     - `SYMPTOM` extraction  
     - `INTERVENTION` extraction  
     - `CLINICAL_CONDITION` extraction  

5. Aggregate all entities across sections  

6. For each entity:
   - Validate schema completeness  
   - Write as one JSON object (one line) to output file  

7. Compute and report aggregate metrics  

---

### 13.4 Validation Outcome

Full raw outputs in `extraction_candidates.jsonl` and validation metrics in `phase2_decisions.md`

**System-level behaviour:**  
- 10,000 notes processed successfully
- ~63% of notes contain valid sections (6,272)  
- ~44% of notes produce at least one entity (4,407)
- Empty outputs are expected due to strict section filtering  

**Entity distribution:**  
- `INTERVENTION` → highest coverage due to dense care-plan language (3,959 notes with interventions, 20,650 total extractions)
- `CLINICAL_CONDITION` → strong diagnostic signal (2,812 notes with conditions, 17,165 total extractions)
- `SYMPTOM` → lower yield due to narrow section scope (1,722 notes with symptoms, 9,672 total extractions)

**Extraction density:**  
- Average per entity type per note:
  - `SYMPTOM`: 5.62 entities per note 
  - `INTERVENTION`: 5.22 entities per note
  - `CLINICAL_CONDITION`: 6.10 entities per note
- ~11 entities per non-empty note (10.78) 
- Confirms high-recall candidate generation  

**Output quality:**  
- Schema is consistent across all entities  
- Span alignment is accurate and traceable  
- Entity-specific fields (e.g. negation, task labels) correctly assigned  

---

### 13.5 Role in Pipeline

- Integrates all deterministic extraction components into a single candidate dataset  
- Produces transformer-ready inputs with full contextual and positional information  
- Acts as the final stage of rule-based processing before transformer validation for:
  - Contextual disambiguation  
  - Final classification  

---

## 14. Deliverables

### 14.1 Code

`src/deterministic_extraction/`:
- `preprocessing.py` — report preprocessing implementation
- `section_extraction.py` — section detection and extraction implementation
- `sentence_segmentation.py` — sentence segmentation implementation

`src/deterministic_extraction/extraction_rules/` 
  - `symptom_rules.py` — SYMPTOM extraction rules
  - `intervention_rules.py` — INTERVENTION extraction rules
  - `clinical_condition_rules.py` — CLINICAL_CONDITION extraction rules

`script/deterministic_extraction/validation/`:
- `validate_preprocessing.py` — preprocessing validation script
- `validate_section_extraction.py` — section extraction validation script
- `validate_sentence_segmentation.py` — sentence segmentation validation script
- `validate_symptom_rules.py` — SYMPTOM extraction validation script
- `validate_intervention_rules.py` — INTERVENTION extraction validation script
- `validate_clinical_condition_rules.py` — CLINICAL_CONDITION extraction validation script

`script/deterministic_extraction/`:
- `run_extraction_pipeline.py` — end-to-end pipeline orchestration

`notes/phase_2/`:
- `header_pattern_exploration.ipynb` — exploration of potential section headers

### 14.2 File Outputs

`data/interim`:
- `extraction_candidates.jsonl` — full set of extracted entity candidates from 10,000-note sample

### 14.3 Documentation

`notes/phase_2/`:
- `phase2_decisions.md` — full methodological and analytical record
- `phase2_summary.md` — structured Phase 2 executive summary

---

## 15. Phase 2 Final Conclusion

Phase 2 successfully demonstrates:

- A hybrid architecture with clearly defined responsibilities, where deterministic rules generate high-quality candidate entities and transformer validation (Phase 3) will handle contextual disambiguation.
- Robust section-aware extraction, constraining candidates to clinically relevant contexts  
- Concept-level normalisation, enabling consistent representation of heterogeneous clinical language  
- High-recall candidate generation, particularly for interventions and clinical conditions  
- Accurate span alignment and provenance tracking, ensuring full traceability to source text  
- Transformer-compatible output design, with flat, self-contained JSONL entities  

Each entity type is deliberately scoped:

- `SYMPTOM` → precision-oriented with lightweight negation  
- `INTERVENTION` → recall-oriented within a constrained action space  
- `CLINICAL_CONDITION` → broad, recall-first diagnostic capture  

The resulting dataset provides a scalable and structurally consistent foundation for downstream semantic validation.

---

## 16. Next Steps: Transformer Validation (Phase 3)

### 16.1 Purpose

- Provide context-aware, clinically meaningful validation of Phase 2 rule-extracted candidate entities
- Refine high-recall candidates into task-specific outputs for downstream use

**Mechanics:**
- Operates on the same entity-level schema as Phase 2
- Populates the `validation` fields with model predictions:
  - `is_valid` → binary judgment of contextual correctness
  - `confidence` → model confidence score
  - `task` → defines the entity-specific classification

**Clarification:**
- Does not generate new candidates  
- Does not establish ground truth  
- Evaluates the contextual correctness of existing candidate entities

---

### 16.2 Classification Tasks

| Entity Type         | Task                        |
|--------------------|-----------------------------|
| `SYMPTOM`           | `symptom_presence`          |
| `INTERVENTION`      | `intervention_performed`    |
| `CLINICAL_CONDITION`| `clinical_condition_active` |

- Tasks define the entity-specific decision the model makes.

---

### 16.3 Role in the Pipeline

- **Rules (Phase 2):** Candidate generation, deterministic span extraction, high-recall coverage  
- **Transformer (Phase 3):** Contextual interpretation, disambiguation, classification

**Entity-specific focus:**

| Entity Type         | Rule Strength | Transformer Role           |
|--------------------|--------------|----------------------------|
| `SYMPTOM`           | High         | Refinement / negation     |
| `INTERVENTION`      | Moderate     | Filtering performed vs planned |
| `CLINICAL_CONDITION`| Low          | Primary semantic classification |

- Transformer reliance increases as rule confidence decreases, reflecting task complexity

---

### 16.4 Rationale

- Phase 2 rules cannot reliably resolve:
  - Context (historical vs current)  
  - Temporality (acute vs resolved)  
  - Intent (planned vs performed)  
  - Uncertainty (suspected vs confirmed)

- Delegating these tasks to a transformer enables:
  - Robust, context-aware validation  
  - Reduced rule complexity and brittleness  
  - Scalable, generalisable pipeline

---