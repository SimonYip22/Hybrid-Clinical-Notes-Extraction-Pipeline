# ICU Clinical Notes Extractor (Reduced Scope)

Structured Information Extraction from ICU Notes Using ClinicalBERT and Hybrid NLP

---

## Project Purpose

This project demonstrates the ability to:
- Process unstructured ICU clinical notes
- Extract clinically meaningful structured information
- Combine rule-based NLP with transformer-based models
- Produce clean, machine-readable outputs suitable for downstream ML

It complements:
- An ICU time-series deterioration predictor (tabular ML)
- An industry collaboration on precision-constrained LLMs for radiology (Radnomics)

The focus is **NLP fundamentals applied to real clinical text**, not standards engineering.

---

## 1. Explicit Scope (4 Weeks)

### In Scope
- ICU clinical note preprocessing
- Entity extraction using:
  - Rule-based patterns (high precision)
  - ClinicalBERT (contextual classification)
- Structured JSON output
- Lightweight evaluation
- Clear documentation and reproducibility (notes.md per phase so its easier to navigate, and less detailed to save time)

### Explicitly Out of Scope (by design)
- SNOMED ontology mapping
- FHIR compliance
- Multi-modal ICU predictor integration
- Large-scale annotation efforts
- Production deployment

(See Section 8 for rationale.)

---

## 2. Problem Definition

### Input
- Unstructured ICU clinical notes
  - Progress notes
  - Daily reviews
  - Event summaries

### Output
- Structured JSON containing:
  - Clinical entities (e.g. symptoms, interventions, complications)
  - Entity category
  - Source text span
  - Confidence score
  - Timestamp (if present)

This mirrors the *real first step* in clinical NLP pipelines.

---

## 3. Data

- **Source**: MIMIC-IV demo / de-identified ICU notes
- **Size**:
  - Full corpus for rule-based extraction
  - ~50–100 manually labelled examples for ClinicalBERT fine-tuning
- **Annotation strategy**:
  - Minimal, focused, representative
  - Emphasis on correctness, not scale

---

## 4. Pipeline Overview

### 4.1 Preprocessing
- Text cleaning
- Section splitting (where present)
- Normalisation of abbreviations

### 4.2 Entity Extraction (Hybrid)
- Rule-based extraction for:
  - Vitals mentions
  - Interventions
  - Common complications
- ClinicalBERT classifier for:
  - Entity type disambiguation
  - Contextual validation

### 4.3 Structured Output
- Deterministic JSON schema
- One record per entity
- Clear linkage to source text

---

## 5. Evaluation

This is not a benchmark project.

Evaluation focuses on:
- Precision / recall / F1 on labelled subset
- Rule-based vs transformer comparison
- Qualitative inspection on real ICU note excerpts

The goal is to demonstrate *engineering judgment*, not leaderboard performance.

---

## 6. Deliverables

- Clean, modular codebase
- Reproducible pipeline
- Clear README
- Example inputs and outputs
- Notes documenting design decisions and trade-offs

---

## 7. Timeline (4 Weeks)

| Week | Work |
|----|----|
| 1 | Data exploration, preprocessing, define entity schema |
| 2 | Rule-based extraction + baseline evaluation |
| 3 | ClinicalBERT fine-tuning + comparison |
| 4 | Documentation, cleanup, README, final outputs |

---

## 8. Deliberate Scope Reductions (and Why)

### 8.1 SNOMED Mapping — Removed
**Why removed:**
- Heavy manual effort
- Ontology engineering > NLP skill signal
- Already implied by clinical background

**How referenced:**
- Mentioned in README as logical extension
- Not required to prove NLP competence

---

### 8.2 FHIR Output — Removed
**Why removed:**
- Adds format complexity without improving ML signal
- More valuable in product teams than ML roles

**How referenced:**
- Structured JSON chosen to keep pipeline model-agnostic

---

### 8.3 ICU Predictor Integration — Removed
**Why removed:**
- Already demonstrated in previous project
- Would duplicate signal
- Increases cognitive load for reviewers

**How referenced:**
- Mentioned as future integration pathway

---

## 9. Why This Project Is Still Strong

This project clearly signals:
- Unstructured clinical NLP competence
- Transformer fine-tuning experience
- Hybrid rule-based + ML design
- Clean data engineering
- Independent project execution

When paired with:
- ICU time-series predictor (tabular ML)
- Radnomics collaboration (precision LLMs)

You now cover:
- Tabular ML
- Clinical NLP
- LLM research
- Industry collaboration

With **zero redundancy**.

---

## 10. Portfolio Positioning

This project belongs under:
- **Projects** (technical depth)

Radnomics belongs under:
- **Experience** (industry collaboration)

They should *not* compete with each other.
They should reinforce each other.