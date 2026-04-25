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

# Deployment Pipeline

## 1. Overview

This section describes how the clinical NLP pipeline is transformed from a locally validated system into a fully deployed, production-ready API service.

The deployment follows a staged approach:

- **Local API execution** → validate pipeline and model behaviour  
- **Containerisation (Docker)** → create a reproducible runtime  
- **Cloud deployment (Cloud Run)** → expose the API publicly  
- **CI/CD (GitHub Actions)** → automate build and deployment  

The focus is not on modifying the model itself, but on packaging, serving, and automating the pipeline in a consistent and production-aligned manner.

---

## 2. Architecture & Deployment Flow

### 2.1 Runtime Architecture

The system operates as a stateless request–response inference service. Incoming client requests are processed through a layered API stack and passed to the core NLP pipeline.

```text
Client (browser or curl HTTP request)
        ↓ 
Cloud Run (container execution)
        ↓
Uvicorn (ASGI server)
        ↓
FastAPI (API framework for routing + validation)
        ↓
/predict endpoint
        ↓
pipeline.py (core logic)
        ↓
Deterministic rules + model inference
        ↓
Structured JSON response → returned to client
```

This architecture enforces clear separation of concerns:

- Cloud Run → infrastructure, scaling, request routing
- FastAPI + Uvicorn → API serving layer
- `pipeline.py` → all ML and rule-based logic

---

### 2.2 Deployment Flow (CI/CD Pipeline)

The deployment pipeline automates the transition from code changes to a live, updated API service.

```text
GitHub (code + LFS model storage)
        ↓ push to main
GitHub Actions (CI/CD)
        ↓
Checkout repo + pull LFS files
        ↓
Cloud Build (build Docker image)
        ↓
Container Registry (gcr.io)
        ↓
Cloud Run deployment
        ↓
New revision → traffic routed → Live API
```

Step-by-step:

1. Code is pushed to the `main` branch  
2. GitHub Actions workflow is triggered  
3. Repository is checked out (including Git LFS model files)  
4. Cloud Build builds the Docker image  
5. Container image is pushed to Container Registry (`gcr.io`) for storage
6. Cloud Run deploys a new revision using the updated image  
7. Traffic is automatically routed to the latest revised deployment  

This means the system is:

- Versioned (immutable images per commit)
- Reproducible
- Externally accessible via HTTPS

---

### 2.3 Deployment Components

- **GitHub** → source code and model storage (via Git LFS)  
- **GitHub Actions** → CI/CD automation pipeline  
- **Cloud Build** → Docker container image build process  
- **Container Registry (`gcr.io`)** → image storage 
- **Cloud Run** → serverless API hosting

This setup froms a complete, fully automated deployment pipeline where each code change results in a consistent immuntable rebuild and redeployment of the service.

---

### 2.4 Deployment File Structure

```text
├── models/                   # Model files (via Git LFS)
├── src/                      # Core pipeline logic
│ 
├── app/
│   └── main.py               # FastAPI application
├── requirements-api.txt      # API dependencies
│ 
├── Dockerfile                # Container definition
├── .dockerignore             # Docker build context control
│ 
├── .gcloudignore             # Cloud Build context control
│
├── .github/workflows/
│   └── deploy.yaml           # CI/CD pipeline
├── .gitattributes            # Git LFS configuration

```

These files collectively define how the system is:

- built (Dockerfile)
- packaged (image)
- deployed (Cloud Run)
- automated (CI/CD)

---

## 3. API Layer

### 3.1 Purpose

The system exposes the clinical NLP pipeline as a stateless inference API using FastAPI. 
The API acts as a lightweight service layer that provides external access to the pipeline without modifying any underlying model or processing logic.

Responsibilities:

- Receive input text via HTTP requests  
- Validate input format  
- Pass input directly to the pipeline (`run_pipeline`)  
- Return structured entity extraction results as JSON  

The API does not handle:

- Model training or updates  
- Data persistence or storage  
- Additional business logic beyond inference orchestration  

This enforces a strict separation between:

- ML layer → pipeline and model logic  
- Serving layer → API interface  

---

### 3.2 Design Decisions

Framework & Interface:

- FastAPI selected for high-performance, asynchronous API handling with minimal overhead  
- Pydantic used to enforce strict input validation and ensure consistent request structure  

Model Execution:

- Model loaded at startup to avoid repeated initialisation and reduce per-request latency  
- Direct integration with `pipeline.py` ensures a single implementation of inference logic (no duplication)  

System Behaviour

- Stateless request–response design enables horizontal scaling in Cloud Run  
- Structured JSON output ensures compatibility with downstream systems and evaluation pipelines   

---

### 3.3 Endpoints

**`POST /predict`**

- Input: raw clinical text  
- Output: structured entity extraction JSON  
- Logic: direct call to `run_pipeline()`  

Request format:

```json
{
  "text": "HPI: Pt c/o CP and SOB. O2 started. Assessment: possible pneumonia."
}

Response format (simplified):

```json
{
  "entities": [...]
}
```

**`GET /health`**

- Returns system status
- Used for monitoring and deployment health checks

Example response:
```json
{
  "status": "ok"
}
```

---

### 3.4 Request Processing Flow

Each request follows a linear inference path through the serving stack:

```text
Client → Cloud Run → Container → Uvicorn → FastAPI → /predict → Pydantic validation → run_pipeline() → JSON response → Client
```

Detailed sequence:

1. Client sends HTTP POST request to `/predict`  
2. Cloud Run routes the request to the container instance  
3. Uvicorn receives the request and passes it to FastAPI  
4. FastAPI routes the request to the `/predict` endpoint  
5. Pydantic validates the input schema (`text` field)  
6. Input is transformed into pipeline-compatible format  
7. `run_pipeline()` executes model + rule-based extraction  
8. Results are returned as structured JSON to the client  

---

### 3.5 Local Validation

The API was validated locally prior to deployment to ensure correctness of:

- Model loading  
- Pipeline execution  
- Endpoint behaviour  

Validation checks:

- `GET /health` → confirms API is running  
- `POST /predict` → verifies end-to-end inference  

Local access:

- http://127.0.0.1:8000  
- http://127.0.0.1:8000/docs  

---

## 4. Docker Containerisation

### 4.1 Purpose

The API is containerised using Docker to create a reproducible runtime environment across:

- Local development  
- Containerised testing  
- Cloud deployment (Cloud Run)  

The container packages these components into a single deployable unit (Docker image), eliminating environment-specific issues::

- Application code  
- Trained model
- Dependencies
- Runtime configuration  

As a result, the system can be:

- Built once  
- Deployed consistently  
- Executed identically across environments  

This container serves as the foundation for deployment to Cloud Run.

---

### 4.2 Container Structure

The container is defined by the following files:

- `Dockerfile` → defines how the image is built  
- `requirements-api.txt` → Python dependencies for the API  
- `.dockerignore` → excludes unnecessary files from the build context  
- `models/` and `src/` → application code and model files are copied into the image for execution 

---

### 4.3 Image Build Logic

The Docker image is constructed in stages:

1. **Base image**

    ```dockerfile
    FROM python:3.11-slim
    ```
    - Lightweight Python runtime
    - Minimises image size

2. **Working directory**

    ```dockerfile
    WORKDIR /app
    ```
    - Establishes a single consistent application root for all operations

3. **System dependencies**

    ```dockerfile
    RUN apt-get update && apt-get install -y build-essential
    ```
    - Required for compiling certain Python packages

4. **Python dependencies**

    ```dockerfile
    COPY requirements-api.txt .
    RUN pip install --no-cache-dir -r requirements-api.txt
    ```
    - Installed in a cached layer for efficient rebuilds

5. **NLTK resources**

    ```dockerfile
    RUN python -m nltk.downloader punkt
    ```
    - Ensures required tokenizer is available at runtime
    - Built into the image for deterministic behaviour

6. **Application code**

    ```dockerfile
    COPY app/ ./app/
    COPY src/ ./src/
    ```
    - Copies API and pipeline code into the container

7. **Model inclusion**

    ```dockerfile
    COPY models/bioclinicalbert_final/ ./models/bioclinicalbert_final/
    ```
    - Embeds trained model directly into the image
    - Ensures no external dependency at runtime

8. **Environment configuration**

    ```dockerfile
    ENV PYTHONPATH=/app:/app/src
    ENV PORT=8080
    EXPOSE 8080
    ```
    - Enables module imports
    - Aligns with Cloud Run port requirements

9. **Startup command**

    ```dockerfile
    CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
    ```
    - Launches the API server
    - Compatible with both local container execution and Cloud Run

---

### 4.4 Local Container Validation

The container was validated locally to ensure it behaves identically to the deployed service.

Validation steps:

- Docker image built from the Dockerfile
- Container executed locally
- API endpoints tested inside container environment

Checks performed:

- API starts successfully inside container
- Model loads correctly within container
- /predict endpoint returns expected output

This step verifies:

- Dependency completeness
- Correct file inclusion (especially model files)
- Runtime compatibility

---

## 5. Cloud Deployment (Google Cloud Run)

### 5.1 Purpose

The containerised API is deployed to Google Cloud Run to transform the system from a local service into an externally accessible ML inference service enabling: 

- Public HTTPS access  
- Scalable, on-demand inference
- production-like execution environment  

Unlike local and containerised testing, this step introduces:

- Remote hosting (cloud-hosted API)
- Managed infrastructure (no server maintenance)
- Real-world access to the system  

This represents the transition from a local development system to a production-style deployment.

---

### 5.2 Deployment Model & Workflow

The system follows a container-based serverless deployment pipeline:

- Docker image built using **Cloud Build**  
- Image stored in **Google Container Registry (GCR)**  
- Container deployed as a service on **Cloud Run**  

Deployment flow:

1. Container image is built via Cloud Build  
2. Image is pushed to `gcr.io`  
3. Cloud Run deploys the image as a new revision  
4. A public HTTPS endpoint is generated  
5. Traffic is routed to the latest deployment  

This creates a fully managed, externally accessible inference service without requiring manual server setup.

---

### 5.3 Runtime System (Cloud Run)

At deployment, the system consists of:

- **Container image** (stored in GCR)  
- **Cloud Run service** (serving layer)  
- **Revisions** (immutable deployments)

Each deployment creates a new revision:

- `clinical-nlp-api-00001`
- `clinical-nlp-api-00002` etc.

Cloud Run automatically routes traffic to the latest revision.

Runtime properties:

- **Serverless execution**
  - Containers are instantiated on request  
  - No always-on server  
  - Scales to zero when idle  

- **Stateless inference**
  - No persistence between requests  
  - Each request is processed independently  

---

### 5.4 Deployment Configuration

Key deployment parameters:

- `--memory 2Gi` → required for model loading  
- `--timeout 300` → allows sufficient startup time  
- `--allow-unauthenticated` → enables public API access  

These ensure the container can initialise and serve requests reliably in the cloud environment.

---

### 5.5 Deployed Service

The API is accessible via: https://clinical-nlp-api-1064509144938.europe-west1.run.app

Available endpoints:

- `GET /health`
- `POST /predict`

The deployed service maintains:

- Identical request format
- Identical response structure
- Consistent behaviour with local execution

---

### 5.6 Validation

Deployment was verified using:

Health check

```bash
curl https://clinical-nlp-api-1064509144938.europe-west1.run.app/health
```

Inference request

```bash
curl -X POST "https://clinical-nlp-api-1064509144938.europe-west1.run.app/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "HPI: Pt c/o CP and SOB. Assessment: possible pneumonia."
  }'
```

This confirms:

- API accessibility
- Correct model loading
- End-to-end pipeline execution

---

## 6. CI/CD (Continuous Integration & Deployment)

### 6.1 Purpose

CI/CD automates the deployment process, transforming it from a manual, environment-dependent workflow into a version-controlled and reproducible pipeline.

Without CI/CD, deployment relies on:

- Local machine configuration  
- Manual execution of CLI commands  
- Correct sequencing of build and deploy steps  

CI/CD removes these dependencies by ensuring that deployments:

- Are triggered directly from source control (GitHub)  
- Follow a fixed and repeatable process  
- Produce consistent results across environments  

As a result:

- Deployment is fully automated  
- No manual intervention is required  
- The deployed system always reflects the latest committed code  

CI/CD does not introduce new functionality to the system—it standardises and automates how the system is built and deployed.

---

### 6.2 Role in the Deployment Pipeline

CI/CD orchestrates the existing deployment components into a single automated workflow:

- Docker defines the application runtime  
- Cloud Build builds the container image  
- Container Registry (`gcr.io`) stores the image  
- Cloud Run deploys and serves the application  

End-to-end flow:

```text
Code change → GitHub push → CI/CD pipeline → Build → Deploy → Live API
```

This ensures that every code change triggers:

1. A fresh container build
2. A new image pushed to the registry
3. Deployment of a new Cloud Run revision

Key outcomes:

- **Reproducibility:** Identical deployments from the same codebase
- **Reliability:** Eliminates manual errors in deployment steps
- **Traceability:** Deployments are linked to specific commits
- **Scalability:** Supports future updates without changing the deployment process

Although CI/CD is not strictly required for a static system, it demonstrates the ability to implement production-grade deployment workflows and operationalise machine learning systems.

---

### 6.3 Workflow Overview (GitHub Actions)

The pipeline is defined in: `.github/workflows/deploy.yaml`

Trigger: `git push` to main branch

Pipeline steps:

1. Checkout repository (with `Git LFS` enabled)
2. Pull model files via `Git LFS`
3. Verify model presence
4. Authenticate with Google Cloud (service account key via GitHub Secrets)
5. Configure project
6. Build container image using Cloud Build
7. Push image to Container Registry (`gcr.io`)
8. Deploy new revision to Cloud Run

This replicates the manual deployment process in a fully automated manner.

---

### 6.4 Model Handling (Git LFS)

Large model files are managed using `Git LFS`:

- `.gitattributes` tracks model file types (`.safetensors`, `.bin`, etc.)
- Prevents exceeding GitHub file size limits
- Ensures model files are available during CI/CD

CI/CD explicitly pulls these files:

```bash
git lfs install
git lfs pull
git lfs checkout
```

This guarantees the model is present before building the container.

---

### 6.5 Authentication & Secrets

Secure deployment requires authentication with Google Cloud.

This is handled via:

- Service account JSON key (created in GCP)
- Stored in GitHub as a secret: `GCP_SA_KEY`
- Project ID stored as: `GCP_PROJECT_ID`

These are injected into the workflow and used for:

- Cloud Build execution
- Cloud Run deployment

No credentials are stored in the codebase.

---

### 6.6 Deployment Execution

Container build:

```bash
gcloud builds submit \
  --tag gcr.io/$PROJECT_ID/clinical-nlp-api \
  --async
```

The pipeline then waits for build completion before deploying:

- Ensures image is fully available
- Prevents race conditions between build and deploy

Deployment:

```bash
gcloud run deploy clinical-nlp-api \
  --image gcr.io/$PROJECT_ID/clinical-nlp-api \
  --region europe-west1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300
```

Each run creates a new Cloud Run revision.

---

## 7. Critical Issues & Fixes

### 7.1 Build Context & Container Packaging

Problem:

- Model files were missing inside the container during Cloud Build  
- Local Docker builds succeeded, but cloud builds failed  

Root cause:

- `.gcloudignore` excluded files from the Cloud Build context  
- Cloud Build only has access to files sent in the build context  
- Local Docker builds used the full local filesystem, masking the issue  

Fix:

- Corrected `.gcloudignore` to ensure model files were included  
- Verified build context contents before container build  

Key insight:

- Build systems operate on an isolated file context.  
- Ignore files (`.gcloudignore`, `.dockerignore`) directly determine what exists inside the final container.

---

### 7.2 Large Model Handling (Git LFS)

Problem:

- Model files exceeded GitHub’s 100MB limit  
- CI/CD builds failed despite files appearing present  
- Runtime errors (e.g. JSON decode errors) occurred  

Root cause:

- Git LFS files were not pulled in CI/CD  
- Repository contained pointer files instead of actual model data  

Fix:

- Configured `.gitattributes` to track model files  
- Added explicit LFS pull step in CI/CD:
  ```bash
  git lfs pull
  git lfs checkout
  ```

Key insight:

- Model files must be included in the GitHHub repository for successful CI/CD builds.
- Without `git lfs pull`, files exist only as pointers.
- This results in silent corruption at runtime rather than build-time failure.

---

### 7.3 Cloud Run Runtime Failures

Problem:

- Deployment failed with: 
  ```bash
  Container failed to start and listen on PORT=8080
  ```
- Container crashed before server startup, causing Cloud Run to report a generic startup failure

Root causes:

- Missing model files in the container → application crashes on startup
- Insufficient memory for large model loading (default 512MB)
- Slow startup due to large model loading

Fix:

- Increased memory: `--memory 2Gi`
- Added startup timeout to prevent premature termination: `--timeout 300`
- Ensured model was correctly packaged in container

Key insights:

- Cloud Run errors can be misleading.
- Startup failures often indicate application crashes (e.g. memory exhaustion), not networking issues.
- Deployment requires correctness at two levels for successful deployment:
    1. **Build-time:** Dockerfile configuration, dependencies, file inclusion  
    2. **Runtime:** Memory allocation, latency, model loading behaviour
- Must validate both levels using logs to distinguish between build and runtime issues.

⸻

### 7.4 CI/CD vs Local Deployment Mismatch

Problem:

- Manual `gcloud run deploy` succeeded
- CI/CD pipeline failed

Root cause:

Differences between environments:

- Local machine had full access to model files
- CI/CD depended entirely on repository state
- Missing LFS pull in CI/CD
- Differences in authentication and build context

Fix:

- Added explicit LFS handling in CI/CD
- Standardised build process via pipeline
- Ensured all dependencies exist within repository context

Key Insight

- Local success does not guarantee CI/CD success.
- CI/CD environments are isolated and must explicitly reconstruct the full system.

---

### 7.5 Cloud Build “False Failure”

Problem:

- Cloud Build succeeded in GCP console
- GitHub Actions reported failure (exit code 1)

Root cause:

- `gcloud builds` submit attempted to stream logs
- Service account lacked permissions to read logs
- CLI reported failure despite successful build

Fix:

- Used asynchronous build: `--async`
- Added manual polling to wait for completion

Key insight:

- CI pipelines can fail due to tooling limitations, not infrastructure failure.
- Observed errors must be validated against actual system state.

---

### 7.6 IAM & Permissions

Problem:

- Authentication errors and failed CI/CD execution
- Cloud Build and deployment inconsistencies

Root cause:

Missing or incomplete IAM roles:

- Cloud Build permissions
- Artifact Registry / Container Registry access
- API enablement

Fix:

- Assigned required roles to service account:
  - Cloud Run Admin
  - Cloud Build Editor
  - Storage / Registry access
- Enabled required APIs in GCP project

Key Insight

- Deployment pipelines depend on correct IAM configuration.
- Permissions affect not only execution, but also visibility (e.g. log access).

---

## 8. System Capabilities & Scope

### 8.1 What This System Demonstrates

This deployment represents a production-style ML inference system that transitions the pipeline from local experimentation to a fully deployed service.

Core capabilities:

- End-to-end pipeline: data → model → API → deployment  
- Clear separation between ML logic (`pipeline.py`) and serving layer (API)  
- Containerised, reproducible runtime across environments  
- Cloud-hosted API with external HTTPS access  
- Automated, version-controlled deployment via CI/CD  

This demonstrates the ability to:

- Productionise model inference beyond notebook environments  
- Build modular, maintainable ML systems  
- Ensure consistent behaviour across local, containerised, and cloud runtimes  
- Deploy and operate within real-world infrastructure constraints  

---

### 8.2 DevOps Coverage (Inference Systems)

The system implements standard deployment patterns used in industry:

- API layer — FastAPI  
- Serving layer — Uvicorn  
- Containerisation — Docker  
- Cloud deployment — Cloud Run  
- CI/CD automation — GitHub Actions  

This constitutes complete DevOps coverage for an ML inference service.

---

### 8.3 MLOps Coverage

This system partially aligns with MLOps practices but is limited to the inference and deployment stages of the ML lifecycle:

- Model inference pipeline  
- Reproducible deployment environment  
- Automated build and deployment (CI/CD)  

Not covered:

- Continuous training or retraining pipelines  
- Data and model versioning  
- Production monitoring and drift detection  

As a result, the system represents a production-grade inference service, rather than a full end-to-end MLOps system.

---

## 9. Limitations & Future Extensions

### 9.1 Current Scope

As this system is intentionally designed as an inference-only deployment, the following components are not included:

- No batch inference endpoint  
- No database or data persistence  
- No authentication or access control  
- No frontend interface  
- No feature store  

Additionally, the system does not implement:

- Request logging  
- Monitoring or alerting  
- Model versioning  
- Automated retraining pipelines  
- Drift detection  

Therefore, it is focused on demonstrating a production-style inference API rather than a full MLOps pipeline:

- The service is suitable for real-time inference but not long-term production monitoring  
- No historical data is retained for auditing or analysis  
- Model performance cannot be tracked or updated automatically  

---

### 9.2 Future Extensions

A full production ML system would typically introduce:

- System-level additions:
  - Request/response logging for traceability  
  - Monitoring and alerting (latency, error rates)  
- ML-specific additions:
  - Input/output distribution tracking for drift detection  
  - Model versioning and rollback capability  
  - Automated retraining pipelines  

These extensions fall under full MLOps systems and are beyond the scope of this project.

---

## 10. API Usage

### 10.1 Endpoint & Input

Base URL: https://clinical-nlp-api-1064509144938.europe-west1.run.app

Endpoints: 

- `POST /predict` for running inference
- `GET /health` for service status check

Request format: 

```json
{
  "text": "HPI: Pt c/o CP and SOB. Assessment: possible pneumonia."
}
```

Requirements:

- Request must be a JSON object with a "text" field
- Value must be a clinical note or report (string)

The pipeline is designed for structured clinical text (designed based on MIMIC-IV report structure):

- Sections (e.g. HPI, Assessment, Plan) improve extraction quality
- Specific clinical words and phrases are expected and captured by rules
- Free-form text is accepted, but structured notes produce more reliable outputs

---

### 10.2 Example Usage

Example Request:

```bash
curl -X POST "https://clinical-nlp-api-1064509144938.europe-west1.run.app/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "HPI: Pt c/o CP and SOB. Assessment: possible pneumonia."
  }'
```

Example Response (simplified):

```json
{
  "entities": [
    {
      "note_id":"note_1",
      "subject_id":"",
      "hadm_id":"",
      "icustay_id":"",
      "entity_text":"SOB",
      "concept":"dyspnoea",
      "entity_type":"SYMPTOM",
      "char_start":14,
      "char_end":17,
      "sentence_text":"Pt c/o CP and SOB.",
      "section":"hpi",
      "negated":false,
      "validation": {
        "is_valid":true,
        "confidence":0.6005578637123108,
        "task":"symptom_presence"
      }
    }
  ]
}
```

---