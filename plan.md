# NLP Unstructured Clinical Notes Extractor

***Precision-First Hybrid Rule-Based and Transformer Pipeline for Structured Clinical Data Extraction***

---

## Project Summary

This project implements a precision-first clinical natural language processing (NLP) pipeline for extracting structured information from unstructured ICU clinical notes. The system combines deterministic rule-based extraction with context-aware transformer classification (ClinicalBERT) to produce clean, auditable, machine-readable outputs suitable for downstream machine learning applications.

The project is deliberately scoped to demonstrate clinical NLP fundamentals, engineering judgment, and pipeline design, rather than ontology engineering, interoperability standards, or production deployment.

Within a broader portfolio, this project complements:
- A completed ICU time-series deterioration predictor (structured physiological data → outcomes)
- An ongoing industry collaboration on precision-constrained LLM fine-tuning for radiology reporting (Radnomics)

Together, these projects form a coherent, non-redundant narrative spanning tabular ML, classical clinical NLP, and modern LLM-based generation.

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

- This project answers: How do we reliably convert noisy ICU narrative into high-precision structured data suitable for downstream ML?
- Radnomics answers: How do we adapt large generative models to produce clinically precise reports under constrained error tolerance?

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

---

## 3. Explicit Scope and Constraints (4 Weeks)

### 3.1 In Scope

- ICU clinical note preprocessing
- Hybrid entity extraction:
  - Rule-based patterns for precision-critical extraction
  - ClinicalBERT for contextual validation and disambiguation
- Deterministic JSON output schema
- Lightweight, appropriate evaluation
- Clear documentation:
  - Minimal phase-based `notes.md` (plan, actions, reflections, issues)
  - A focused, synthesis-oriented README

---

### 3.2 Explicitly Out of Scope (By Design)

- SNOMED CT or ontology mapping
- FHIR or interoperability standards
- Integration with the ICU deterioration predictor
- Large-scale annotation campaigns
- Production deployment or optimisation

These exclusions are intentional and justified in Section 9.

---

## 4. Problem Definition

### 4.1 Input

Unstructured ICU clinical notes, including:
- Progress notes
- Daily reviews
- Event summaries

### 4.2 Output

Structured JSON records containing:
- Extracted clinical entity
- Entity category
- Source text span
- Confidence score
- Timestamp (if explicitly present)

This mirrors the first-stage transformation in real-world clinical NLP pipelines.

---

## 5. Architecture Rationale

### 5.1 Hybrid Design Principle

The pipeline combines:
- Rule-based extraction for precision-critical, auditable patterns
- ClinicalBERT classification for contextual interpretation and ambiguity resolution

This reflects real clinical NLP practice, where:
- Deterministic logic is preferred in error-intolerant settings
- Transformers are used selectively where context is essential

The architecture explicitly avoids naïve end-to-end modelling and instead demonstrates judgment-driven system design.

---

## 6. Data Strategy

### 6.1 Data Source

- MIMIC-IV Demo or equivalent de-identified ICU note subsets

### 6.2 Annotation Strategy

- Full corpus used for rule-based extraction
- Approximately 50–100 manually labelled examples for ClinicalBERT fine-tuning

This dataset size is explicitly framed as:
- Proof-of-concept fine-tuning
- Demonstration of method and trade-offs
- Not a generalisable NER system

Annotation effort is deliberately:
- Minimal
- Focused
- Timeboxed to avoid diminishing returns

---

## 7. Pipeline Design

### 7.1 Preprocessing

- Text cleaning and normalisation
- Section detection and splitting
- Abbreviation handling

---

### 7.2 Entity Extraction

#### 7.2.1 Entity Schema (Deliberately Constrained)

To avoid overreach and ambiguity, the entity schema is limited to 3–4 clinically meaningful categories:
- Symptoms
- Interventions
- Complications
- Vitals mentions

Vague or overly broad categories are explicitly avoided.

---

#### 7.2.2 Rule-Based Extraction

Applied to:
- Vitals mentions
- Common ICU interventions
- High-frequency complications

Optimised for precision, interpretability, and auditability.

---

#### 7.2.3 ClinicalBERT Usage (Strictly Constrained)

ClinicalBERT is used as a:
- Sentence- or span-level classifier
- Contextual validator and disambiguator

It is explicitly not used for:
- Token-level NER
- Sequence tagging
- Over-engineered fine-tuning

This keeps the transformer component credible without bloating scope.

---

### 7.3 Structured Output

- Deterministic JSON schema
- One record per extracted entity
- Explicit linkage to source text

Outputs are designed to be model-agnostic and downstream-ready.

---

## 8. Evaluation Strategy

This is not a benchmarking project.

Evaluation focuses on:
- Precision, recall, and F1 on the labelled subset
- Comparative behaviour of rule-based versus transformer components
- Qualitative inspection of real ICU note excerpts
- Explicit documentation of failure modes

The goal is to demonstrate clinical realism and engineering judgment, not leaderboard performance.

---

## 9. Deliberate Scope Reductions and Rationale

### 9.1 SNOMED Mapping — Excluded

- High manual and cognitive overhead
- Ontology engineering rather than NLP signal
- Clinical knowledge already implied by background

Referenced only as a logical extension.

---

### 9.2 FHIR Output — Excluded

- Adds format complexity without improving ML signal
- More relevant to interoperability or product roles

Structured JSON preserves flexibility and clarity.

---

### 9.3 ICU Predictor Integration — Excluded

- Already demonstrated in the ICU predictor project
- Would duplicate signal
- Increases reviewer cognitive load if predictor was integrated

Referenced conceptually, not implemented.

---

## 10. Signal Strength and Portfolio Value

This project is designed to maximise technical signal per unit effort. Its value derives not from novelty of models or scale of data, but from clarity of framing, disciplined scope control, and clinically realistic design choices.

It explicitly signals:
- Unstructured clinical NLP competence
- Transformer fine-tuning experience in low-data regimes
- Hybrid deterministic + probabilistic system design
- Clean clinical data engineering
- Independent end-to-end project execution

These signals are orthogonal to those demonstrated by the ICU predictor and Radnomics projects, increasing overall portfolio coverage rather than redundancy.

---

## 11. Why This Project Is Non-Generic

### 11.1 Pipeline-Centric Framing

Generic NLP projects demonstrate models.  
This project demonstrates reliable transformation of messy clinical narrative into structured, downstream-ready signals.

It explicitly:
- Separates preprocessing from extraction
- Handles section-dependent semantics
- Combines deterministic and probabilistic components
- Designs outputs for downstream ML use

This places the work in clinical data engineering, not toy NLP.

---

### 11.2 Hybrid Design as an Engineering Signal

Rather than defaulting to a single paradigm, the hybrid architecture reflects:
- Awareness of precision requirements
- Understanding of transformer failure modes
- Practical handling of low-data constraints

This mirrors production-relevant clinical NLP design.

---

### 11.3 Intentional Portfolio Positioning

The project is explicitly positioned as:

> The unstructured-text counterpart to time-series ICU modelling and LLM-based clinical text generation.

This signals:
- Systems thinking
- Modality coverage
- Intentional portfolio construction

---

## 12. High-Signal, Low-Effort Reinforcements

Without expanding scope, signal is increased through:

- Precision-first framing throughout documentation
- Explicit focus on one non-trivial clinical NLP challenge (e.g. negation, section semantics, or temporal ambiguity)
- Transparent documentation of failure modes and trade-offs

These choices increase perceived seniority without increasing implementation burden.

---

## 13. Relevance in an LLM-Dominated Landscape

LLMs do not replace structured extraction in clinical pipelines.

Clinical systems still require:
- Deterministic logic
- Auditable extraction
- Narrow, well-defined NLP components

LLM-based generation is already covered by the Radnomics project; extending this project in that direction would introduce redundancy rather than strength.

---

## 14. Portfolio Positioning

- This project is listed under Projects (technical depth)
- Radnomics is listed under Experience (industry collaboration)

Together with the ICU predictor, they form a unified, structured, senior portfolio narrative.

---

## 15. Summary

This project is intentionally:
- Precision-first
- Pipeline-centric
- Hybrid by design
- Explicitly constrained
- Non-generic by construction

Its strength lies not in model novelty, but in judgment, restraint, and clinical realism.

---