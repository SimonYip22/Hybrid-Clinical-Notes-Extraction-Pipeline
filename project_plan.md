# Hybrid Clinical Information Extraction Pipeline

***Precision-First Hybrid Rule-Based and Transformer Pipeline for Structured Clinical Data Extraction***

---

## Project Summary

This project implements a precision-first clinical natural language processing (NLP) pipeline for extracting structured information from unstructured ICU clinical notes. The system combines deterministic rule-based extraction with context-aware transformer classification (ClinicalBERT) to produce clean, auditable, machine-readable outputs suitable for downstream machine learning applications.

The project is deliberately scoped to demonstrate clinical NLP fundamentals, engineering judgment, and pipeline design, rather than ontology engineering, interoperability standards, or production deployment.

Within a broader portfolio, this project complements:
- A completed ICU time-series deterioration predictor (structured physiological data → outcomes)
- An ongoing industry collaboration on precision-constrained LLM fine-tuning for radiology reporting (RadNomics)

Together, these projects form a coherent, non-redundant narrative spanning tabular ML, classical clinical NLP, and modern LLM-based generation.

What we are building is realistic engineering for clinical NLP, not maximal model complexity or benchmark performance. The architecture is a hybrid pipeline with clear component roles:
- Deterministic rule candidate generation
- Controlled transformer validation
- Span-aligned structured output
- Explicit evaluation
- Full-corpus execution

That is a standard architecture for clinical NLP systems in real-world settings, where precision, auditability, and downstream utility are paramount:
- Clinical IE systems
- Safety-critical NLP
- Regulated environments
- Low-annotation settings

---

## 1. Portfolio Context and Complementarity

### 1.1 Relationship to the ICU Deterioration Predictor

- The ICU predictor models structured, time-series physiological data to predict patient deterioration.
- This NLP project models unstructured clinical narrative to extract structured entities.
- Together, they implicitly define a complete clinical ML pipeline: Unstructured text → structured features → predictive modelling

This linkage is conceptual rather than integrated, avoiding unnecessary implementation complexity while still demonstrating systems-level reasoning.

---

### 1.2 Relationship to the Radnomics LLM Project

Although both this project and Radnomics operate on clinical text, they address distinct abstraction layers:
- This project: deterministic extraction + constrained transformer validation to reliably convert noisy ICU narrative into high-precision structured data suitable for downstream ML
- Radnomics: generative LLM fine-tuning to produce clinically precise reports under constrained error tolerance

Abstraction layers differ:
- Extraction pipeline vs generative modelling
- Narrow span classification vs full autoregressive generation

The former focuses on classical clinical NLP and extraction pipelines; the latter focuses on LLM fine-tuning, evaluation, and dataset alignment. This separation is intentional and avoids redundancy.

---

## 2. Project Objectives

The objective of this project is to design and implement a hybrid clinical NLP pipeline that:

- Processes unstructured ICU clinical notes
- Extracts clinically meaningful entities with high precision
- Combines deterministic and probabilistic methods in a principled manner
- Produces structured outputs designed explicitly for downstream ML consumption
- Documents trade-offs, limitations, and failure modes

The emphasis is on engineering judgment under realistic clinical constraints, not maximal model complexity.

### 2.1 In Scope

- ICU clinical note preprocessing
- Hybrid entity extraction:
  - Rule-based patterns for precision-critical extraction
  - ClinicalBERT for contextual validation and disambiguation
- Deterministic JSON output schema
- Lightweight, appropriate evaluation
- Clear documentation:
  - Minimal phase-based notes folder containing detailed notes (summary, decisions, issues)
  - A focused, synthesis-oriented README

---

### 2.2 Explicitly Out of Scope (By Design)

- SNOMED CT or ontology mapping
- FHIR or interoperability standards
- Integration with the ICU deterioration predictor
- Large-scale annotation campaigns
- Production deployment or optimisation

These exclusions are intentional and justified later

---

## 3. Dataset Definition

### 3.1 Data Source

MIMIC-IV full dataset (upon approval)

Scope is intentionally restricted to ICU progress notes from adult patients
- No discharge summaries
- No radiology reports
- No outpatient notes

This constrains linguistic variability and clinical domain.

---

### 3.2 Expected Dataset Size and Usage Strategy

Approximate corpus expectations (not guaranteed):

- 30,000–60,000 ICU progress notes
- Average note length: 300–1,200 tokens
- Total corpus: 15–40 million tokens

The dataset is used at three distinct levels:

**Phase 1 — Manual Inspection**
- 50–100 notes manually reviewed
- Used only for structural understanding and rule planning

**Phase 3 — Manual Annotation**
- 200 sentences manually labelled
- Used only for ClinicalBERT validation fine-tuning
- No expansion beyond this fixed gold set

**Phase 5 — Full Corpus Execution**
- Entire ICU progress note corpus processed
- Rule-based extraction applied at scale
- Transformer used only as a validation layer
- No additional manual annotation introduced

The full dataset is not manually annotated and is not used for large-scale supervised training.

---

## 4. Fixed Entity Schema (Deliberately Constrained)

To avoid overreach and ambiguity, the entity schema is limited to 4 clinically meaningful categories:

1. SYMPTOM  
2. INTERVENTION  
3. COMPLICATION  
4. VITAL_MENTION  

No additions unless:
- Mapping failure >15%
- Systematic ambiguity discovered during annotation

---

### 4.1 Operational Definitions

#### SYMPTOM
Subjective or clinician-observed clinical manifestations.

Includes:
- chest pain
- dyspnoea
- confusion
- fever

Excludes:
- vital measurements
- lab values
- procedures

Negation supported.

---

#### INTERVENTION
Therapeutic or procedural actions performed.

Includes:
- intubated
- started norepinephrine
- central line inserted
- mechanical ventilation

Excludes:
- hypothetical plans not performed

---

#### COMPLICATION
Adverse or pathological developments.

Includes:
- sepsis
- AKI
- pneumothorax
- delirium

Excludes:
- baseline chronic diagnoses without acute worsening

---

#### VITAL_MENTION
Explicit physiological measurement or abnormal descriptor.

Must include:
- measurement + value
OR
- abnormal vital descriptor

Examples:
- BP 85/50
- SpO2 88%
- tachycardic

Excludes:
- lab-only results

---

## 5. Output JSON Specification (Deterministic)

- Deterministic JSON schema
- One record per extracted entity
- Explicit linkage to source text

Outputs are designed to be model-agnostic and downstream-ready.

```json
{
  "note_id": "string",
  "entity_text": "string",
  "entity_type": "SYMPTOM | INTERVENTION | COMPLICATION | VITAL_MENTION",
  "char_start": 0,
  "char_end": 0,
  "sentence_text": "string",
  "section": "string | null",
  "timestamp_text": "string | null",
  "negated": true,
  "validation_confidence": 0.0
}
```

Rules:
- `char` spans must align with source
- `negated` determined by deterministic rule logic
- `validation_confidence` from ClinicalBERT probability
- no nested entities
- no ontology codes

This mirrors the first-stage transformation in real-world clinical NLP pipelines.

---

## 6. Architecture Rationale

### 6.1 Hybrid Design Philosophy

The pipeline combines:
- Rule-based extraction for precision-critical, auditable patterns
- ClinicalBERT classification for contextual interpretation and ambiguity resolution

This reflects real clinical NLP practice, where:
- Deterministic logic is preferred in error-intolerant settings
- Transformers are used selectively where context is essential

The architecture explicitly avoids naïve end-to-end modelling and instead demonstrates judgment-driven system design.

### 6.2 Architecture Principles

**Rule-Based Extraction**
- High precision
- Transparent
- Deterministic
- Auditable

**Transformer (ClinicalBERT)**
- Sentence/span-level classifier
- Used only for:
  - Contextual validation (true entity vs false positive)
  - Limited category disambiguation

This keeps the transformer component credible without bloating scope.

**Not Allowed**
- Token-level NER
- CRF
- Sequence tagging
- Generative prompting

---

## 7. Phase Plan (4–6 Weeks)

This phase plan is structured to move from uncertainty reduction → deterministic backbone → controlled model use → evaluation → full-scale execution.

---

### Phase 1 — Data Inspection & Corpus Profiling (Week 1)

**Purpose:**  
Understand structural properties of ICU progress notes before designing schema or rules.

**Manual Review**
- 50–100 ICU progress notes
- Focused qualitative inspection

**Deliverables**
- Note structure analysis (sections, formatting patterns)
- Section frequency breakdown
- Average token length statistics
- Abbreviation pattern inventory
- Preliminary entity distribution observations
- Identification of high-frequency pattern candidates

**Outputs**
- Corpus profiling summary
- Design implications for schema and rule construction

No modelling performed in this phase.

---

### Phase 2 — Schema Operationalisation, Preprocessing & Rule Design (Week 1–2)

**Purpose:**  
Formalise the entity schema and implement the deterministic backbone of the pipeline.

---

#### 2.1 Preprocessing Layer (Deterministic)

Implemented before rule extraction:

- Text cleaning and whitespace normalisation
- Section header detection and segmentation
- Abbreviation handling (limited deterministic mapping)
- Span-preserving transformations only

No probabilistic processing.  
Character offsets must remain valid.

---

#### 2.2 Schema Operationalisation

- Finalise operational definitions for:
  - `SYMPTOM`
  - `INTERVENTION`
  - `COMPLICATION`
  - `VITAL_MENTION`
- Explicit inclusion/exclusion boundaries
- No schema expansion beyond 4 entity types

---

#### 2.3 Rule Scope Targets

Expected:

- 15–25 regex patterns for `VITAL_MENTION`
- 20–40 patterns for `INTERVENTION`
- 15–25 patterns for `COMPLICATION`
- 15–30 patterns for `SYMPTOM`

**Total expected rule patterns:** 65–120

---

#### 2.4 Negation Detection

- Simple window-based negation rules
- Fixed token window (e.g., 5–7 tokens)
- No full NegEx implementation
- No dependency parsing

---

#### Deliverables

- Deterministic extraction module
- Section-aware extraction logic
- Negation detection function
- Unit-tested rule functions
- Span-aligned JSON output

Transformer not used in this phase.

---

### Phase 3 — Annotation & ClinicalBERT Fine-Tuning (Week 3)

**Purpose:**  
Introduce controlled probabilistic validation under low-data constraints.

---

#### 3.1 Annotation Scope

Manual gold set:

- 200 sentences total
- Balanced across 4 entity types
- Sampled from real ICU notes
- Includes both true entities and rule-generated false positives

Split (fixed, no reshuffling):

- 70% train (140)
- 15% validation (30)
- 15% test (30)

No expansion beyond 200 sentences.

---

#### 3.2 ClinicalBERT Fine-Tuning Plan

**Model**
- ClinicalBERT (pretrained)

**Task**
Binary classification:
- TRUE entity
- FALSE positive

Model applied at sentence/span level only.

---

#### Training Constraints

- 3–5 epochs
- Batch size: 8–16
- Learning rate: 2e-5
- No hyperparameter search
- No architecture modification
- No token-level tagging

---

#### Rationale for Small-Scale Fine-Tuning

- Demonstrates low-data adaptation
- Avoids artificial dataset inflation
- Reflects realistic clinical annotation limits
- Shows understanding of transformer limitations
- Preserves precision-first design

This is controlled fine-tuning, not large-scale model training.

---

### Phase 4 — Evaluation (Week 4)

**Purpose:**  
Quantify deterministic extraction performance and transformer validation impact before full scaling.

---

#### 4.1 Entity-Level Matching Rule

Correct extraction defined as:

- ≥1 character span overlap
- Correct entity type

Strict type matching enforced.

No fuzzy semantic matching.

---

#### 4.2 Metrics

For each entity type:

**Precision**  
P = TP / (TP + FP)

**Recall**  
R = TP / (TP + FN)

**F1**  
F1 = 2PR / (P + R)

**Micro-F1**  
Aggregate TP/FP/FN across all entity types.

---

#### 4.3 Component-Level Metrics

**Rule-Based Extraction**
- Precision
- Recall
- F1 per entity type

**Transformer Validation**
- Accuracy
- Precision
- Recall
- F1
- Confusion matrix

No AUC.  
No benchmarking against external models.

---

#### 4.4 Final Pipeline Metrics

- End-to-end entity-level F1
- Per-entity breakdown
- Error categorisation:
  - False positives (pattern drift)
  - False negatives (missed phrasing)
  - Negation failures
  - Section misclassification

The objective is disciplined evaluation, not leaderboard optimisation.

---

### Phase 5 — Full Corpus Extraction & Analysis (Week 5)

**Purpose:**  
Demonstrate scalability and structured output engineering.

---

#### 5.1 Full Dataset Execution

- Run full hybrid pipeline on entire ICU progress note corpus
- Expected scale: 30,000–60,000 notes
- No additional annotation introduced

---

#### 5.2 Structured Output Generation

- Deterministic JSON outputs
- Span-aligned entities
- Confidence scores from ClinicalBERT
- Negation flags preserved

---

#### 5.3 Aggregate Analysis

Compute:

- Entity frequency distribution
- Negation rate per entity type
- Section-based entity distribution
- Entity co-occurrence patterns (optional, descriptive only)

---

#### 5.4 Qualitative Error Analysis

- Review high-confidence false positives
- Review low-confidence true positives
- Identify systematic failure patterns
- Document model–rule interaction behaviour

No additional model training at this stage.

---

## 8. Documentation

**Purpose:**  
Consolidate technical work into structured documentation.

- Phase summaries written
- Decisions documented per phase
- Failure modes explicitly described
- README synthesised
- Example JSON outputs included
- Clear articulation of architectural trade-offs

Project considered complete when:

- End-to-end pipeline runs deterministically
- Evaluation metrics computed and documented
- Structured outputs validated
- No unresolved architectural ambiguity

---

## 9. Expected Scope Control

- Annotation: 200 sentences only
- Rules: <120 patterns
- Fine-tuning: <5 epochs
- No ontology
- No deployment
- No model benchmarking

---

## 10. Deliberate Scope Reductions and Rationale

### 10.1 SNOMED Mapping — Excluded

- High manual and cognitive overhead
- Ontology engineering rather than NLP signal
- Clinical knowledge already implied by background

Referenced only as a logical extension.

---

### 10.2 FHIR Output — Excluded

- Adds format complexity without improving ML signal
- More relevant to interoperability or product roles

Structured JSON preserves flexibility and clarity.

---

### 10.3 ICU Predictor Integration — Excluded

- Already demonstrated in the ICU predictor project
- Would duplicate signal
- Increases reviewer cognitive load if predictor was integrated

Referenced conceptually, not implemented.

---

## 11. Portfolio Signal and Positioning

This project maximises technical signal per unit effort through explicit scope control, architectural clarity, and clinically realistic constraints. Its value derives from engineering judgment and disciplined design rather than model scale, novelty, or benchmark optimisation.

---

### 11.1 Technical Signal

It demonstrates:

- Unstructured clinical NLP competence  
- Low-data transformer fine-tuning under realistic annotation limits  
- Hybrid deterministic + transformer system design  
- Clean clinical data engineering and schema-controlled outputs  
- Independent end-to-end pipeline construction  

The emphasis is precision-first extraction, controlled probabilistic validation, and auditable structured outputs suitable for downstream modelling.

---

### 11.2 Hybrid Architecture as a Constraint-Driven Decision

The hybrid design is deliberate and role-separated.

**Deterministic rule layer:**
- High-precision candidate generation  
- Transparent regex-based logic  
- Section-aware extraction  
- Auditable and reproducible behaviour  

**Transformer layer (ClinicalBERT):**
- Sentence-level binary validation  
- False-positive filtering  
- Limited category disambiguation  
- No sequence tagging or generative use  

The transformer does not perform primary extraction.  
It operates strictly as a controlled validation component.

This separation reflects:

- Awareness of precision requirements in clinical pipelines  
- Understanding of transformer brittleness in low-resource settings  
- Practical handling of limited annotation budgets  
- Avoidance of unnecessary sequence-tagging complexity  

The architecture mirrors production-oriented clinical NLP systems where deterministic logic constrains probabilistic components.

---

### 11.3 Explicit Scope Discipline

Constraint is intentional and documented.

The project explicitly avoids:

- Token-level NER  
- CRF or sequence tagging frameworks  
- Ontology expansion  
- Large-scale annotation  
- Hyperparameter search  
- External benchmark competition  
- Generative prompting  

These exclusions signal engineering restraint and resistance to artificial scale inflation.

Scope is defined by clinical utility and implementation realism rather than leaderboard optimisation.

---

### 11.4 Pipeline-Centric Framing

The project is pipeline-focused rather than model-centric.

It explicitly:

- Separates preprocessing from extraction logic  
- Handles section-dependent semantics  
- Implements negation handling at rule level  
- Produces schema-validated structured JSON  
- Designs outputs for downstream structured modelling  

This positions the work within clinical data engineering rather than generic NLP experimentation.

---

### 11.5 Complementarity Within the Portfolio

This project functions as the unstructured-text counterpart to:

- ICU time-series deterioration modelling  
- LLM-based clinical text generation (Radnomics)  

Together they demonstrate:

- Structured signal modelling (time-series)  
- Deterministic extraction from narrative  
- Controlled generative adaptation  

This provides modality coverage without redundancy and signals intentional portfolio construction.

---

### 11.6 Clinical Realism in an LLM-Dominated Landscape

Large language models do not replace structured extraction in clinical pipelines.

Clinical systems still require:

- Deterministic logic  
- Auditable components  
- Narrow, well-defined NLP modules  
- Precision-oriented outputs suitable for downstream modelling  

This project remains extraction-focused by design, preserving architectural clarity and avoiding overlap with generative systems already demonstrated elsewhere.

---

## 12. Summary

This project is intentionally:

- Precision-first  
- Pipeline-centric  
- Hybrid with strict component separation  
- Explicitly constrained  
- Clinically realistic  
- Engineering-led rather than benchmark-led  

Its strength lies in architectural judgment, scope discipline, and controlled system design rather than artificial scale.