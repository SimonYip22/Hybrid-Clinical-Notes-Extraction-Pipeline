# Phase 5 - Inference Pipeline and Deployment

## Objective

The objective of Phase 5 is to transition from model development and evaluation to a fully operational, reproducible inference system that can be:

- Applied at scale to large clinical datasets (`icu_corpus.csv`)
- Used for individual or batch inference  
- Exposed as a deployable service  

This phase consolidates all prior work into a single unified pipeline that:

- Integrates rule-based extraction and transformer validation  
- Produces structured, high-quality clinical entity outputs
- Maintains consistency with the evaluated pipeline behaviour (Phase 4)  

The phase has three core goals:

1. **Pipeline Construction**
  - Build a reusable, modular inference pipeline
  - Ensure outputs follow the validated schema used in rule-based extraction  

2. **Large-Scale Dataset Generation**
  - Apply the pipeline to the full ICU corpus (~160K reports)  
  - Demonstrate scalability and real-world applicability  

3. **Deployment**
  - Expose the pipeline via a simple API (FastAPI)
  - Enable both single-report and batch inference  

---

## Overall Structure 

In scripts/phase_5_inference/

* run_full_dataset.py

In app/

* main.py

✔ Proper separation of concerns

* core logic vs execution vs API

✔ Reusable ML pipeline

* not a one-off script

✔ Deployable architecture

* same code used in:
    * batch processing
    * real-time inference

✔ Production thinking

* modular
* testable
* extensible
You are now transitioning from “analysis project” → production-style ML system

This structure is exactly what makes your project:

* credible
* scalable
* high-signal for jobs

---

## Full Pipeline Construction

### 1. Overview 

This stage defines the core system of the project: a unified inference pipeline that operationalises the full two-stage architecture:

1. High-recall deterministic rule-based extraction
2. Precision-oriented transformer validation

The pipeline is implemented as a single reusable module and serves as the single source of truth for all inference logic, supporting:

1. Large-scale dataset generation (160k ICU corpus)
2. Deployment (single and batch inference)

The goal is to construct a modular, scalable, and reproducible ML pipeline consistent with real-world system design.

---

### 2. Pipeline Design and Rationale

#### 2.1 Summary of System

The system implements a modular, production-style ML pipeline with:

- Clear separation between rule-based and learned components
- Unified inference interface
- Scalable and reusable architecture

A single pipeline is reused across:

- Dataset generation
- Deployment

This ensures consistency, maintainability, and real-world applicability.

---

#### 2.2 Modular Architecture

The pipeline is deliberately split into three components:

```text
src/
  └── pipeline/
        ├── extraction.py
        ├── validation.py
        └── pipeline.py
```

Each module has a single responsibility:

- `extraction.py` → deterministic entity generation  
- `validation.py` → transformer-based scoring and filtering  
- `pipeline.py` → orchestration of full pipeline  

Each stage can be modified or improved without affecting the other

Key architectural decision:

> A single pipeline is constructed once and reused across all downstream tasks.

This avoids:

- Duplicated logic  
- Inconsistencies between environments  
- Divergence between batch vs single inference  

---

### 3. Extraction Component

#### 3.1 Overview

The extraction stage performs high-recall candidate generation using the deterministic rules established from Phase 2:

- Preprocessing
- Section extraction
- Regex-based entity extraction
- Schema-aligned output

---

#### 3.2 Function Design

1. **Single Note Extraction (`extract_entities_from_note()`)**

Processes one clinical note:

- Preprocess text (`preprocess_note()`)
- Extract sections (`extract_sections()`)
- Apply all rule-based extractors (`extract_symptoms()`, `extract_interventions()`, `extract_clinical_conditions()`)
- Return flat list of entities

---

2. **Batch Extraction (`run_extraction_on_dataframe(df)`)**

Primary interface which processes a full dataset:

- Iterates over DataFrame rows
- Calls single-note function internally (`extract_entities_from_note()`)
- Aggregates all entities

Efficient iteration using `df.itertuples()` instead of `iterrows()`:

- Faster for large datasets → flattened tuple vs df columns
- Lower overhead
- Critical for later 160k+ scale

---

#### 3.3 Unified inference Design 

All inputs are treated as DataFrame, as batch wraps single:

- Single report → converted to DataFrame
- Batch → already DataFrame
- Full dataset → DataFrame

This ensures:

- Consistency:
  - Same code path for dataset generation and deployment
  - No branching logic
- Fewer bugs:
  - No duplicated logic
  - No divergence between single vs batch
  - Improves maintainability
- Production realism:
  - Real systems batch internally (even if batch size = 1)

---

#### 3.4 Identifier and Metadata Handling

The pipeline distinguishes between three categories of fields: synthetic identifiers, optional metadata, and derived features.

**A. Synthetic Note Identifier**

A unique `note_id` is generated during batch processing:

- Created using row enumeration (`note_1`, `note_2`, ...)
- Not present in the original dataset
- Required because:
  - One clinical note produces multiple entity-level outputs
  - The pipeline output is flattened (one entity per record)

This ensures all extracted entities can be traced back to their originating note.

---

**B. Optional Metadata Fields**

The following fields are treated as optional inputs:

- `subject_id`
- `hadm_id`
- `icustay_id`

Design behaviour:

- If present in the input DataFrame → extracted and passed through
- If absent → default to empty strings (`""`)

This enables the same pipeline to operate across:

- Structured datasets (e.g. MIMIC-IV)
- Unstructured deployment inputs (raw text only)

No conditional logic is required, ensuring a consistent interface.

---

**C. Derived Contextual Fields**

Several fields are not sourced from the input dataset but are dynamically generated during extraction:

- `section` → derived from structured section extraction
- `sentence_text` → derived from sentence segmentation
- `entity_text`, `char_start`, `char_end` → derived from regex-based span extraction

This design reflects the core purpose of the pipeline:

> transforming unstructured clinical text into structured, context-rich representations.

---

**Design Implications**

This approach ensures:

- Flexibility across different input types  
- Consistent schema regardless of input structure  
- Preservation of contextual information for downstream modelling  
- Elimination of branching logic between dataset and deployment scenarios  

---

### 4. Validation Component

#### 4.1 Overview

The validation stage applies a trained transformer model to each extracted entity in order to:

- Assign a probability score (`confidence`)
- Determine validity (`is_valid`) via thresholding

This stage performs precision filtering over the high-recall outputs generated during deterministic extraction.

---

#### 4.2 Function Design

`validate_entities()`

**Inputs:**

- List of extracted entities  
- Preloaded model, tokenizer, and device  

**High-Level Workflow:**

- Construct model input text from structured entity fields  
- Run batched transformer inference  
- Convert probabilities to boolean predictions via thresholding  
- Insert outputs back into existing entity structure  

**Outputs:**

- Same entities with populated validation fields:
  - `is_valid` (boolean)  
  - `confidence` (float)  

---

#### 4.3 Key Design Decisions

**A. External Model Injection**

The model is not loaded inside the function, and is instead passed as an argument:

`validate_entities(entities, model, tokenizer, device)`

This avoids:

- Repeated loading overhead
- Unnecessary memory usage
- Inefficiency in large-scale processing

---

**B. Default Parameters**


- `threshold = 0.549` → derived from evaluation and threshold tuning
- `batch_size = 16` → balanced for performance and memory
- `max_length = 512` → aligned with model constraints

These values are:

- Consistent with training configuration
- Sensible defaults for most use cases

All parameters remain overrideable, allowing flexibility for:

- Different hardware constraints
- Threshold experimentation
- Alternative deployment settings

---

**C. Training–Inference Consistency**

The input text format exactly matches the format used during training.

This is critical because:

- This ensures stable model behaviour and reliable probability outputs
- Transformer models are sensitive to input structure
- Any mismatch introduces distribution shift
- Performance degradation would otherwise occur

---

#### 4.4 Critical Design Insight

Validation enriches existing structure rather than creating a new one.

The extraction stage already produces a complete, structured JSON schema containing:

- Entity spans
- Metadata (section, concept, type)
- Context (sentence text)
- Validation placeholders

Validation only fills in:

```json
"validation": {
  "is_valid": true/false,
  "confidence": probability
}
```

This design results in:

- Zero schema transformation
- Minimal computational overhead
- Clean separation of responsibilities
- Full traceability (raw + validated outputs retained)

---

#### 4.5 Design Implications

The pipeline preserves both signal and uncertainty:

- `confidence` enables downstream threshold tuning and analysis
- `is_valid` enables immediate filtering when required

No information is discarded during validation:

- Filtering is deferred to downstream use cases
- Supports flexible applications:
  - ML dataset construction
  - Auditing and error analysis
  - Clinical decision support systems

The function is fully batch-compatible and scalable:

- Efficient for both:
  - Small inference batches (deployment)
  - Large corpora (160k+ dataset generation)

---

### 5. Pipeline Orchestration

#### 5.1. Overview

The final pipeline combines extraction and validation into a single entry point: `run_pipeline()`

Pipeline flow:

1. Rule-based extraction  
2. Transformer validation  
3. Return structured outputs  

This function acts as the single interface to the full system.

---

#### 5.2 Design Principles

A single unified pipeline is used across all use cases for reusability and scalability:

- Full dataset generation (160k corpus)
- Deployment for:
  - Batch inference (e.g. 10–50 reports)
  - Single report inference (by wrapping input into a one-row DataFrame)

All inputs are standardised as a DataFrame and passed through the same pipeline to ensure:

- **Consistency:** Identical logic across all environments  
- **Maintainability:** No duplicated code paths  
- **Scalability**: Seamless transition from small to large workloads  
- **Production realism:** Real-world systems follow a single inference pipeline  
- **Reproducibility:** Fixed model and threshold allow for consistent outputs across runs  

---

#### 5.3 Function Design

`run_pipeline(df, model, tokenizer, device)`

Responsibilities:

- Call extraction module (`run_extraction_on_dataframe(df)`) 
- Pass outputs to validation module (`validate_entities()`) 
- Return fully structured and validated entities  

The function does not perform:

- Model loading  
- File I/O  
- Dataset filtering  

These are handled externally to maintain separation of concerns.

---

### 6. Output Design

#### 6.1 Schema Format

The pipeline outputs a list of dictionaries (JSON-compatible):

- One record per entity  
- Flat structure (no nesting by note)  

Important to note that:

- The pipeline operates at the entity level (one record per extracted entity), not the note level. 
- A single clinical note may produce multiple entity records.

Each entity contains:

- Extracted text  
- Entity type  
- Positional metadata  
- Validation results:
  - `confidence` (probability)
  - `is_valid` (boolean)

Example:

```json
{
  "note_id": "note_1",
  "subject_id": "66907",
  "hadm_id": "152136.0",
  "icustay_id": "279344",
  "entity_text": "sedated",
  "concept": "sedation",
  "entity_type": "INTERVENTION",
  "char_start": 132,
  "char_end": 139,
  "sentence_text": "...",
  "section": "assessment",
  "negated": null,
  "validation": {
    "is_valid": true,
    "confidence": 0.92,
    "task": "intervention_performed"
  }
}
```

---

### 6.2 Dataset Strategy

The pipeline intentionally does not filter entities.

Two dataset forms are supported downstream:

1. **Full Dataset (Unfiltered)**
  - Contains all extracted entities
  - Includes validation scores
  - Useful for:
    - Analysis
    - Auditing
    - Threshold tuning
2. **Filtered Dataset**
  - `is_valid == True` only
  - Suitable for: 
    - Downstream ML modelling

Filtering is deferred to downstream steps to:

- Preserve full information
- Avoid irreversible decisions at inference time
- Enable flexible reuse of the dataset

This aligns with standard ML practice:

> Inference pipelines generate data; downstream pipelines decide how to use it.

---













# Large-Scale Dataset Generation

## 1. Overview

This stage applies the constructed pipeline to the full ICU dataset (~160K reports) to generate a **large-scale structured clinical entity dataset**.

This is not a new system, but a **direct application of the pipeline at scale**.

---

### Purpose

- Demonstrate **scalability** of the pipeline  

- Generate a **real-world dataset** suitable for:

  - Machine learning  

  - Clinical analysis  

  - Further research  

- Provide evidence of:

  - End-to-end system functionality  

  - Practical utility beyond evaluation datasets  

---

### Process

1. Load full ICU corpus  

2. Iterate over reports  

3. Apply `pipeline.py` to each report  

4. Collect outputs  

---

### Output

- Stored as **JSONL (JSON Lines)**:

  - One JSON object per report  

  - Efficient for large-scale processing  

  - Standard format in NLP pipelines  

Optional:

- CSV conversion may be performed later for specific analyses  

- Not required for core pipeline functionality  

Be explicit:

“The pipeline outputs a fully annotated dataset. A filtered high-precision subset (is_valid = True) is derived for downstream modelling.”

This shows:

* awareness of trade-offs
* correct ML pipeline design

---

### Key Characteristics

- **Identical logic** to evaluation pipeline  

- No retraining or modification  

- Only difference: **data scale**

This ensures:

> The large-scale dataset faithfully reflects the validated behaviour observed in Phase 4.

---

# Deployment Pipeline

## 1. Overview

This stage exposes the pipeline as a **usable inference service**, enabling:

- Single-report inference (interactive use)

- Batch inference (multiple reports)

The deployment reuses the same `pipeline.py` module, ensuring:

> No duplication of logic between development, evaluation, and production

---

### Deployment Approach

A lightweight API is implemented using **FastAPI**, providing:

#### Endpoint 1: Single Report

- Input: raw clinical text  

- Output: structured entity JSON  

#### Endpoint 2: Batch Inference (optional)

- Input: list of reports  

- Output: list of structured outputs  

---

### Purpose

- Demonstrate **real-world usability** of the system  

- Provide a **clean interface** for external use  

- Show evidence of:

  - Deployment capability  

  - API design  

  - Reproducible inference  

---

### Design Principles

- **Thin wrapper over pipeline**

  - No additional logic introduced  

  - Calls `pipeline.py` directly  

- **Consistency**

  - Outputs identical to batch generation and evaluation  

- **Simplicity**

  - No unnecessary front-end required  

  - Focus on functionality and reproducibility  

---

Deployment: Final Method (Fixed Scope)

1. What you are building

A stateless inference API that exposes your pipeline.

Nothing more.

⸻

2. Tech Stack (Do not deviate)

Backend

* Python
* FastAPI

Server

* Uvicorn

Containerisation

* Docker

Hosting

* Cloud run GCP

CI/CD

* GitHub Actions

⸻

3. What you will implement (strict scope)

3.1 Core API (FastAPI)

You will implement exactly 2 endpoints:

1. /predict

* Input: single clinical report (string)
* Output: structured JSON (your pipeline output)

2. /health

* Returns: simple status ({"status": "ok"})

That’s it.

❌ No batch endpoint
❌ No frontend
❌ No authentication
❌ No database

This setup proves:

Engineering

* Modular pipeline design
* Separation of concerns
* Reusable inference logic

ML

* End-to-end deployment of trained model
* Real inference system (not notebook)

DevOps

* Containerisation (Docker)
* CI/CD integration
* Cloud deployment

This is exactly what recruiters want.

User → API (/predict)
        ↓
   FastAPI app
        ↓
   pipeline.py
        ↓
Rule-based → Transformer → Threshold
        ↓
   JSON output

Client
  ↓
Cloud Run (FastAPI)
  ↓
pipeline.py
  ↓
Rule-based → Transformer → Threshold
  ↓
Structured JSON

   13. Why this is optimal

This approach is:

* Minimal → avoids burnout
* Correct → matches industry patterns
* High signal → ticks all boxes recruiters care about
* Reusable → same pipeline powers everything

⸻

Final instruction

Do exactly this:

1. Finish pipeline.py
2. Wrap it with FastAPI
3. Add Docker
4. Deploy on Railway
5. Add GitHub Actions
---

# Diagrams (Important)

You should include the following diagrams:

### 1. Full Pipeline Diagram (REQUIRED)

Yes — you need this.

It should show:

Raw Text
↓
Rule-Based Extraction (High Recall)
↓
Candidate Entities
↓
Transformer Validation (model_prob)
↓
Threshold (0.549)
↓
Final Entities (JSON Output)

This is the **most important diagram in your entire project**.

---

### 2. Transformer Architecture Diagram (REQUIRED)

- Shows model internals (already planned)

---

### 3. Training Pipeline Diagram (REQUIRED)

- Data → training → validation → threshold tuning  

---