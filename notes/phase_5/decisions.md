# Phase 5 - Inference Pipeline and Deployment

## Objective

The objective of Phase 5 is to transition from model development and evaluation to a fully operational, reproducible inference system that can be:

- Applied at scale to large clinical datasets (`icu_corpus.csv`)
- Used for individual report inference (interactive use)
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
  - Enable both single-report inference  

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
    * single-report inference
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
2. Deployment (single-report inference)

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

A unique `note_id` is required to track the relationship between source clinical notes and extracted entity-level outputs.

Design approach:

- `note_id` is not generated within the extraction module
- It is instead created externally in the orchestration layer and passed as an argument to the extraction functions
- The extraction functions expect `note_id` to be provided as part of the input DataFrame

Rationale:

- One clinical note produces multiple entity-level outputs
- The pipeline output is flattened (one entity per record)
- A stable identifier is required to link entities back to their source note

This separation ensures:

- Correctness under chunked processing (no ID resets across batches)
- Deterministic and reproducible identifiers
- Clear separation of responsibilities:
    - Orchestration layer → assigns identity
    - Extraction layer → transforms data

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
- Correct handling of large-scale datasets via external identifier management
- Clear separation between data identity and data transformation

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
  - Single report inference (deployment)
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

The pipeline outputs a list of dictionaries (JSON-compatible), with one record per extracted entity.

Important structural properties:

- The pipeline operates at the **entity level**, not the note level
- A single clinical note may generate multiple entity records
- The output is intentionally **flat (non-nested by note)** to support large-scale processing and downstream ML use cases

Each entity contains:

- Extracted text span
- Entity type and concept label
- Positional metadata within the source note
- Source context (sentence + section)
- Validation outputs:
  - `confidence` (model probability)
  - `is_valid` (binary decision)

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

#### 6.2 Dataset Strategy

The pipeline intentionally does not perform filtering at inference time. 
Instead, it produces a complete annotated dataset, from which multiple downstream views can be derived.

Two dataset forms are supported:

1. **Full Dataset (Unfiltered)**

  - Contains all extracted entities, regardless of validation outcome 
  - Includes:
    - All rule-based extractions
    - Transformer confidence scores
    - Binary validation labels
  - Use cases
    - Error analysis and model auditing
    - Threshold calibration and tuning
    - Dataset quality inspection
    - Research and exploratory analysis

2. **Filtered Dataset (Downstream Derived)**

  - Subset of entities where:
    - `is_valid == True`
  - Use cases:
    - Downstream machine learning models
    - Training data construction
    - High-precision clinical feature extraction

---

#### 6.3 Design Principles

The output design follows a strict separation between:

> Data generation (pipeline) vs Data usage (downstream systems)

Key principles:

1. **Non-destructive inference**
  - No entities are removed or altered during validation
  - All extracted candidates are preserved

2. **Deferred decision-making**
  - Filtering (`is_valid == True`) is not applied within the pipeline
  - Downstream tasks define their own selection criteria

3. **Reproducibility**
  - The same pipeline output can be reused across multiple experiments
  - No need to rerun inference when changing thresholds or filters

4. **Flexibility**
  - A single dataset supports multiple use cases:
    - analysis
    - modelling
    - auditing

5. **Model independence**
  - Validation outputs are recorded but not enforced
  - The pipeline remains agnostic to downstream objectives

These principles ensure the pipeline functions as a general-purpose data generation system rather than a task-specific modelling pipeline.

---

#### 6.4 Future Extensions

The current schema is intentionally text-centric and can be extended in future directions.

**A. Structured Metadata Enrichment**

Additional fields from the ICU corpus can be incorporated:

- `AGE`
- `GENDER`
- `LOS_HOURS`
- `FIRST_CAREUNIT`
- `CATEGORY`
- `CHARTTIME`

Purpose:

- Cohort stratification
- Subgroup analysis
- Temporal modelling
- Clinical context enrichment

---

**B. Increased Context-Aware Validation**

Structured metadata can be injected into the transformer input:

Example:

```python
[AGE] 74
[GENDER] F
[CAREUNIT] MICU
[TEXT] ...
```

Expected benefits:

- Improved classification robustness
- Reduced ambiguity in entity interpretation
- Better calibration across patient subgroups

---

**C. Multi-Modal Clinical Modelling**

The dataset can be extended into a multi-modal representation layer combining:

- Unstructured clinical text (current pipeline output)
- Structured EHR variables
- Entity-level features

Downstream applications:

- Patient risk prediction
- Outcome modelling
- Clinical decision support systems
- Similarity search over patient cohorts

---

**D. Feature Store Integration**

Rather than modifying the pipeline, metadata can be joined downstream via a feature store approach.

This enables:

- Reproducible dataset variants
- Experiment tracking across feature configurations
- Separation of NLP extraction vs predictive modelling pipelines

---

## Large-Scale Dataset Generation

### 1. Overview

This stage applies the extraction–validation pipeline to the full ICU corpus (~160K clinical notes) to generate a large-scale structured clinical entity dataset.

This is a direct deployment of the existing pipeline at scale:

- Identical extraction and validation logic  
- No retraining or modification  
- Only the data scale differs from earlier stages  

This ensures the resulting dataset directly reflects the behaviour validated during earlier evaluation phases.

The script acts purely as an execution layer:

- Orchestrates pipeline execution  
- Delegates all processing to `run_pipeline()`  
- Introduces no additional transformation logic  

This design demonstrates:

- Scalability to large clinical datasets
- End-to-end system robustness  
- Practical applicability to real-world settings 

---

### 2. Execution via Modular Pipeline

The script invokes the pipeline through a single entry point:

`run_pipeline(df=chunk, ...)`

This reflects the modular design:

- Extraction and validation logic are encapsulated within the pipeline  
- The script contains no business logic  
- No duplication of functionality occurs  

This separation ensures:

- Clean orchestration  
- Reproducibility  
- Ease of maintenance  

---

### 3. Scalability and Memory Management

#### 3.1 Chunked Processing

The ICU dataset is processed in chunks (`CHUNK_SIZE = 3000`) to manage memory usage:

- The full dataset cannot be safely loaded into memory alongside:
  - Intermediate entity lists
  - Transformer inference batches
- Chunking ensures:
  - Stable memory usage
  - Predictable runtime behaviour

Trade-off:

- Larger chunks → fewer iterations (faster I/O)
- Smaller chunks → lower memory usage

A value of 3000 was selected as a balanced configuration for:

- CPU-based inference
- 16GB memory constraints

---

#### 3.2 Selective Column Loading

For data efficiency, only the required columns for the pipeline are loaded from the CSV:

`usecols = ["TEXT", "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"]`

- Only columns required by the pipeline are loaded
- Reduces:
  - Memory footprint
  - Disk I/O (faster loading)
  - Parsing overhead

Other columns (e.g. demographics, timestamps) are excluded because:

- They are not used in extraction or validation
- Including them would increase computational cost without benefit

---

#### 3.3 Single Pipeline Call per Chunk

The pipeline is applied once per chunk:

`entities = run_pipeline(df=chunk, ...)`

This avoids:

- Redundant computation
- Exponential runtime increases
- Duplicated outputs

All iteration over rows is handled internally within the pipeline.

---

### 4. Global `note_id` Generation

#### 4.1 Implementation

`note_id` is generated at the orchestration level to ensure globally unique identifiers across the full dataset:

- A global counter (`global_idx`) is maintained across chunks  
- Each note is assigned: `note_{global_idx}`  
- IDs are attached to the DataFrame and passed into the pipeline  

This guarantees:

- No duplication of identifiers across chunks  
- Stable mapping between entities and their source notes  

---

#### 4.2 Rationale

Chunk-based processing would otherwise reset identifiers per chunk:

- Chunk 1 → note_1 … note_3000  
- Chunk 2 → note_1 … note_3000 (duplicate)

This would cause:

- Identifier collisions  
- Loss of traceability  
- Incorrect downstream aggregation  

Global indexing eliminates this failure mode.

---

#### 4.3 Design Principle

Identifier generation is intentionally excluded from core pipeline functions:

- Pipeline components remain stateless and reusable  
- Dataset-specific concerns (ordering, identity tracking) are handled externally  

This ensures consistent behaviour across different execution contexts and avoids duplicated logic.

---

### 5. Output Format

The pipeline outputs data in **JSONL (JSON Lines)** format:

- One JSON object per extracted entity (not per clinical note)  
- Flat structure to support streaming and large-scale processing  
- Optimised for downstream machine learning and analytical workflows  

JSON serialization is handled with `json.dumps(entity, default=float)` to ensure:

- Compatibility with NumPy scalar types (converted to native Python floats)
- Valid JSON serialization without manual preprocessing
- No loss of numerical precision at the schema level

---

### 6. Post-Generation Dataset Strategy

The dataset is intentionally unfiltered even as a full corpus output:

- All extracted entities are retained  
- Validation outputs (`confidence`, `is_valid`) are included
- Dataset generation is therefore non-destructive and preserves all information for:
  - Full auditability of model behaviour
  - Reproducibility of downstream filtering decisions

From the same output, multiple dataset variants can be derived:

- Full dataset: For analysis, debugging, threshold calibration, and error inspection
- Filtered dataset (`is_valid == True`): Used for downstream modelling and high-precision applications

---

### 7. Final Dataset Properties

Full metrics are available in `full_dataset_explore.ipynb`.

#### 7.1 Dataset Scale

| Metric | Value |
|------|------|
| Total reports (ICU corpus) | 162,296 |
| Reports with ≥1 entity | 71,917 |
| % reports with entities | **44.31%** |
| Total entities extracted | **780,941** |

- The pipeline extracts entities from ~44% of reports  
- This is expected for a high-recall deterministic system applied to heterogeneous clinical notes  
- A substantial dataset (~781K entities) is generated for downstream use  

---

#### 7.2 Subject Coverage

| Metric | Value |
|------|------|
| Unique subjects (ICU corpus) | 25,054 |
| Unique subjects (final dataset) | 8,621 |

- Not all patients have extractable content, only ~34% of subjects have at least one extracted entity
- The resulting dataset represents a clinically relevant subset rather than full population coverage  

---

#### 7.3 Entities per Report

| Statistic | Value |
|----------|------|
| Mean | 10.86 |
| Median | 7 |
| Std | 10.86 |
| Min | 1 |
| 25th percentile | 3 |
| 75th percentile | 16 |
| Max | 98 |

- Distribution is right-skewed (few reports with very high counts)  
- Median (7) is substantially lower than mean → confirms skew  
- Typical reports contain a moderate number of entities (11)
- High variance reflects heterogeneity in note structure and content  

---

#### 7.4 Entity Type Distribution

| Entity Type | Count | Percentage |
|------------|------|-----------|
| INTERVENTION | 334,872 | 42.88% |
| CLINICAL_CONDITION | 280,314 | 35.89% |
| SYMPTOM | 165,755 | 21.23% |

- Interventions dominate, consistent with ICU documentation focus  
- Clinical conditions form a substantial proportion  
- Symptoms are least frequent, reflecting the rule design of no duplicated symptom concepts per line of a note

---

#### 7.5 Validation Outcomes

| Metric | Value |
|------|------|
| Valid entities | 319,852 |
| % valid | **40.96%** |

- ~41% of candidates are retained after validation  
- Confirms expected behaviour of high-recall extraction followed by filtering  
- Remaining ~59% represent false positives or low-confidence candidates  

---

#### 7.6 Validation by Entity Type

| Entity Type | Valid (%) |
|------------|----------|
| INTERVENTION | **51.76%** |
| SYMPTOM | 36.65% |
| CLINICAL_CONDITION | 30.60% |

- Interventions achieve highest validation rate → more structured and easier to detect  
- Clinical conditions show lowest precision → higher ambiguity and lexical variation  
- Symptoms fall between the two → originally expected to be highest validation rate

This demonstrates class-dependent model performance.

---

#### 7.7 Confidence Score Distribution

| Statistic | Value |
|----------|------|
| Mean | 0.506 |
| Median | 0.508 |
| Std | 0.120 |
| Min | 0.171 |
| 25th percentile | 0.415 |
| 75th percentile | 0.609 |
| Max | 0.738 |

- Confidence scores are centred around ~0.5  
- Distribution aligns closely with the selected threshold (0.549)  
- Indicates:
  - Effective separation between valid and invalid candidates  
  - Reasonable calibration of the validation model  

---

#### 7.8 Summary

- The pipeline successfully scales to the full dataset without modification  
- Produces a large, structured entity dataset (~781K entities)  
- Maintains expected behaviour:
  - High-recall extraction  
  - Moderate precision after validation  
- Demonstrates:
  - Robustness across heterogeneous clinical data  
  - Consistent performance patterns across entity types  

The resulting dataset is suitable for:

- Downstream modelling (filtered subset)  
- Analysis and auditing (full dataset)  
- Further research and pipeline refinement

---

### 8. Workflow Implementation

All logic in `generate_full_dataset.py` follows these steps:

1. **Initialise environment and load dependencies**
  - Load transformer tokenizer and classification model
  - Move model to available device (CPU/GPU)
  - Set evaluation mode (`model.eval()`)

2. **Configure dataset streaming**
  - Read ICU corpus using `pandas.read_csv()` with `chunksize=3000`
  - Restrict input columns using `usecols` to:
    - `TEXT`
    - `SUBJECT_ID`
    - `HADM_ID`
    - `ICUSTAY_ID`
  - This reduces memory overhead and I/O cost

3. **Initialise global identifier tracking**
  - Maintain a global counter (`global_idx`) across all chunks
  - Ensures continuous indexing across dataset stream

4. **Iterate over dataset in chunks**
  - Process dataset incrementally to avoid memory overload
  - Each chunk is independently loaded and processed

5. **Assign globally unique `note_id`**
  - For each row in the chunk:
    - Increment global counter
    - Assign identifier in format: `note_{global_idx}`
  - Attach resulting `note_id` column to the DataFrame
  - Ensures cross-chunk uniqueness and stable traceability

6. **Execute pipeline per chunk**
  - Call `run_pipeline(df=chunk, ...)`
  - Pipeline internally performs:
    - Preprocessing
    - Section extraction
    - Entity extraction
    - Validation scoring
  - Returns flattened list of entity-level outputs

7. **Serialize and write outputs**
  - Iterate over extracted entities
  - Write each entity as a JSON line to disk
  - Use `json.dumps(..., default=float)` to ensure:
    - NumPy numeric types are converted to native Python floats
    - Full JSON compatibility is preserved

8. **Persist final dataset**
  - Output stored as JSONL format
  - One entity per line
  - Suitable for streaming, downstream ML, and large-scale analysis

---

## Deployment Pipeline

## 1. Overview

This stage exposes the clinical NLP pipeline as a **lightweight inference service**, enabling real-world usage of the system through an API interface.

The deployment reuses the existing `pipeline.py` module directly, ensuring:

> A single source of truth across development, evaluation, and production.

No additional model logic is introduced at deployment level.

This is a production level system design that reflects real-world ML deployment patterns

---

## 2. System Design

The system is a stateless inference API built using FastAPI.

### Core properties:

- Stateless (no database, no session tracking)  
- Deterministic (same input → same output)  
- Thin wrapper over existing pipeline  
- Fully reproducible outputs  

---

## 3. API Design

### 3.1 `/predict`

- Input: raw clinical text  
- Output: structured entity JSON  
- Logic: direct call to `run_pipeline()`  

---

### 3.2 `/health`

- Returns system status  
- Used for deployment monitoring and uptime checks  

---

## 4. System Architecture

Client (browser / script / curl)
        ↓ HTTP request
Cloud Run (Google Cloud)
        ↓
Uvicorn (web server)
        ↓
FastAPI (API framework)
        ↓
Your endpoint (/predict)
        ↓
pipeline.py (your logic)
        ↓
Model + rules
        ↓
Return JSON
        ↑
FastAPI formats response
        ↑
Uvicorn sends HTTP response
        ↑
Cloud Run returns to user

---
Real sequence:

1. User sends HTTP request:

POST https://your-app.run.app/predict

2. Cloud Run receives it
3. Cloud Run forwards to your container
4. Uvicorn receives request
5. FastAPI routes to /predict
6. Pydantic validates input
7. Your function runs:

run_pipeline(...)

8. Output returned as JSON
9. Response sent back to user

---

## 5. Design Principles

### 5.1 Minimal Wrapper Design

The API layer contains no ML logic:

- No preprocessing logic duplication  
- No model handling logic duplication  
- No transformation logic  

It only orchestrates inference calls.

---

### 5.2 Reusability

The same pipeline is used across:

- offline batch generation  
- evaluation pipeline  
- deployment API  

This ensures consistency across all system stages.

---

### 5.3 Stateless Inference

- No stored state between requests  
- Each request is independent  
- Enables horizontal scaling in cloud environments  

---

## 6. Infrastructure Stack

### Backend

- Python  
- FastAPI  

### Server

- Uvicorn  

### Containerisation

- Docker  

### Deployment Target

- Google Cloud Run  

### CI/CD

- GitHub Actions (minimal build + deploy pipeline)

---

## 7. Scope Constraints (Important)

The deployment is intentionally minimal.

The following are NOT included:

- ❌ Batch inference endpoint  
- ❌ Database integration  
- ❌ Authentication layer  
- ❌ Frontend interface  
- ❌ Feature store  

---

## 8. Rationale

This design is chosen to maximise:

### Engineering clarity

- Clean separation between API and ML logic  

### Deployment reliability

- Stateless, containerised service  

### Portfolio signal strength

- Reflects real production ML systems used in industry  

### Maintainability

- Single pipeline reused across all system stages  

This setup proves:

Engineering

* modular system design
* separation of concerns

ML

* real inference system (not notebook)
* end-to-end pipeline

Deployment

* containerisation
* cloud hosting
* reproducibility

That combination is what matters.

Are you doing “standard DevOps for ML”?

Yes — this is exactly standard for inference systems.

You are covering:

* API service (FastAPI)
* Serving layer (Uvicorn)
* Containerisation (Docker)
* Cloud deployment (Cloud Run)
* CI/CD (GitHub Actions)

That is complete DevOps coverage for an ML inference service.

Why this is only “partial MLOps”

MLOps includes the entire lifecycle of models, not just serving.

What you ARE doing:

* Model inference pipeline
* Deployment
* Reproducibility

What you are NOT doing:

* Automated retraining
* Data versioning
* Monitoring model performance in production
* Drift detection
* Experiment tracking

Those are MLOps components.

---

What real production systems would add (for context)

A full production ML system might include:

Monitoring

* track prediction distributions over time
* detect drift in inputs or outputs

Logging

* store all requests + outputs
* enable debugging and audit

Alerting

* notify if system degrades

Retraining pipeline

* update model when performance drops

If you wanted to extend (not required):

Minimal production upgrades

* request logging (store inputs/outputs)
* simple metrics (latency, counts)

ML-specific upgrade

* confidence distribution tracking
* threshold sensitivity analysis in production

True MLOps (bigger jump)

* data pipeline for retraining
* model versioning
* drift detection

---

## 9. Implementation Order

1. Finalise `run_pipeline()` stability  
2. Implement FastAPI wrapper (`/predict`, `/health`)  
3. Build Docker container  
4. Deploy to Google Cloud Run  
5. Add minimal CI/CD via GitHub Actions  

---






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