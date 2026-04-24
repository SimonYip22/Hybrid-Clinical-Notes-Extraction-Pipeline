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

### 1. Overview

This section describes how the clinical NLP pipeline is transformed from a locally validated system into a fully deployed, production-ready API service.

The deployment process follows a staged approach, with each step addressing a specific requirement:

- **Local execution** → verify core pipeline and model behaviour  
- **Containerisation (Docker)** → ensure reproducible runtime environment  
- **Cloud deployment (Cloud Run)** → expose the API as a public service  
- **CI/CD (GitHub Actions)** → automate build and deployment on code changes  

The system is ultimately exposed as a lightweight inference API, enabling real-world usage via structured endpoints. Key design principles:

- Reuse of the existing `pipeline.py` module  
- Single source of truth across development, evaluation, and production  
- No additional model logic introduced at deployment stage  

Overall, the focus is not on modifying the model itself, but on packaging, serving, and automating the pipeline in a consistent and production-aligned manner.

---

## 2. System Design

The system is implemented as a stateless inference API that exposes the clinical NLP pipeline through a lightweight service layer. The API acts as a thin wrapper over the existing pipeline, enabling external access without modifying core model or processing logic.

### 2.1 Core Properties

- **Stateless Inference**
  - No database or session storage  
  - Each request is processed independently  
  - Enables horizontal scaling in cloud environments  

- **Deterministic**
  - Identical inputs produce identical outputs  
  - No stochastic behaviour at inference time  
  - Ensures reproducibility and reliability  

- **Thin Wrapper Architecture**
  - API layer contains no model or preprocessing logic and acts only as an interface to the pipeline.
  - Directly invokes the existing `pipeline.py` module  

- **Single Source of Truth**
  - All environments (local, container, cloud) use the same `pipeline.py` implementation.
  - Eliminates divergence between environments  

---

### 2.2 API Interface

The system is exposed via a minimal HTTP API designed for inference-only usage.

The API is responsible for:

- Receiving input text via HTTP requests  
- Passing input directly to the pipeline  
- Returning structured extraction outputs  

The API does not handle:

- Model training or updates  
- Data persistence or database interactions  
- Additional business logic beyond inference orchestration  

This ensures a strict separation between the ML pipeline (logic) and the serving layer (API).

---

### 2.3 Design Implications

This design leads to:

- **Consistency:** Identical behaviour across local, containerised, and deployed environments  
- **Scalability:** Stateless design enables horizontal scaling in Cloud Run  
- **Maintainability:** Changes to the pipeline propagate automatically to all system stages  
- **Clarity of responsibility:** Clear boundary between model logic and serving infrastructure  

---

## 3. Architecture & Deployment Flow

### 3.1 Runtime Architecture

The system operates as a stateless request–response inference service. Incoming client requests are processed through a layered API stack and passed to the core NLP pipeline.

```text
Client (browser / script / curl)
        ↓ HTTP request
Cloud Run (managed container)
        ↓
Uvicorn (ASGI server)
        ↓
FastAPI (API framework)
        ↓
/predict endpoint
        ↓
pipeline.py (core logic)
        ↓
Model + deterministic rules
        ↓
Structured JSON response → returned to client
```

This architecture enforces clear separation of concerns:

- Cloud Run → request handling, scaling, infrastructure
- Uvicorn + FastAPI → API serving layer
- pipeline.py → all ML and rule-based logic

---

### 3.2 Deployment Flow (CI/CD Pipeline)

The deployment pipeline automates the transition from code changes to a live, updated API service.

```text
GitHub (code + model via LFS)
        ↓ push to main
GitHub Actions (CI/CD)
        ↓
Checkout repo + pull LFS files
        ↓
Cloud Build
        ↓
Docker image build
        ↓
Container Registry (gcr.io)
        ↓
Cloud Run deployment
        ↓
New revision → traffic routed → Live API
```

Detailed flow:

1. Code is pushed to the `main` branch  
2. GitHub Actions workflow is triggered  
3. Repository is checked out (including Git LFS model files)  
4. Cloud Build builds the Docker image  
5. Image is pushed to Container Registry (`gcr.io`)  
6. Cloud Run deploys a new revision using the updated image  
7. Traffic is automatically routed to the latest revision  

---

### 3.3 Final Architecture Summary

The deployed system consists of the following components:

- **GitHub** → source code and model storage (via Git LFS)  
- **GitHub Actions** → CI/CD automation pipeline  
- **Cloud Build** → container image build process  
- **Container Registry (`gcr.io`)** → image storage  
- **Cloud Run** → serverless API hosting and scaling  

This setup enables a fully automated and reproducible deployment pipeline, where each code change results in a consistent rebuild and redeployment of the service.

---

## API Usage

Endpoint:

`POST /predict`

Request format: 

```json
{
  "text": "HPI: Pt c/o CP and SOB. Assessment: possible pneumonia."
}
```

- Input must be a JSON object with a "text" field
- The value should be a clinical note or report

Input Expectations

The pipeline is designed around clinical note structure (inspired by MIMIC-style data), therefore:

- Text should contain semi-structured clinical language
- Sections such as HPI, Assessment, Plan improve extraction performance
- Abbreviations (e.g. SOB, CP) are supported
- Free-form text is accepted, but structured notes produce more reliable outputs

Example Request:

```bash
curl -X POST "https://<your-cloud-run-url>/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "HPI: Pt c/o CP and SOB. Assessment: possible pneumonia."
  }'
```

Example Response (simplified)

```json
{
  "entities": [
    {
      "entity_text": "SOB",
      "concept": "dyspnoea",
      "entity_type": "SYMPTOM"
    }
  ]
}
```

---



2. Deployment Stages (core section)

This is the backbone.

Structure it like this:

⸻

2.1 Local API Validation

* FastAPI run locally
* Tested via:
    * /health
    * /predict
* Purpose: verify pipeline + model correctness

No deep detail. Just the role.

⸻

2.2 Docker Containerisation

* Dockerfile created
* Built image locally
* Ran container locally

Purpose:

* Ensure reproducible environment
* Match cloud runtime

Important insight:

“Works on my machine” problem eliminated here

⸻

2.3 Cloud Deployment (Google Cloud Run)

* Image built via Cloud Build
* Deployed to Cloud Run
* Public URL generated

Validation:

* /health
* /predict

Important insight:

Transition from local system → externally accessible service

⸻

2.4 CI/CD Automation (GitHub Actions)

* Trigger: push to main
* Steps:
    * Checkout + LFS pull
    * Build container
    * Deploy to Cloud Run

Purpose:

* Remove manual deployment
* Ensure reproducibility


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

1. Local setup (clear, reproducible)

* install deps
* run API
* test endpoint

2. API usage

* input format ({"text": ...})
* example request
* example response

3. Deployed endpoint

* live URL
* same request example

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

## Ratioanle 

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

This design demonstrates:

- End-to-end ML system deployment (not notebook-only work)  
- Reproducible environments across development and production  
- Separation of concerns between inference logic and serving layer  
- Practical use of containerisation and cloud infrastructure  

---

## 10. Limitations (Explicit)

Not implemented:

* request logging
* monitoring
* model versioning
* automated retraining
* drift detection

This is an inference-only deployment.

* Service URL: stable
* Deployment: persists until changed
* CI/CD: strategically valuable

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

## Cloud Deployment (Cloud Run)

Service URL: https://clinical-nlp-api-1064509144938.europe-west1.run.app

What actually exists now (important concept)

You do NOT have “one app”.

You have:

A versioned system

* Container image (in GCR)
* Cloud Run service
* Revisions (immutable deployments)

Each deploy creates:

* clinical-nlp-api-00001
* clinical-nlp-api-00002
* etc.

Cloud Run routes traffic to the latest.

1. Overview

The clinical NLP pipeline is deployed as a containerised inference service using Google Cloud Run.

This enables:

* external access via HTTPS
* scalable, on-demand inference
* production-like deployment conditions

⸻

2. Deployment Model

The system follows a container-based serverless deployment:

* Docker image built via Cloud Build
* Image stored in Google Container Registry
* Service deployed to Cloud Run

⸻

3. Runtime Characteristics

Serverless execution

* Containers are instantiated on request
* No always-on server
* Scales to zero when idle

Stateless design

* No persistence between requests
* Each request is independent

⸻

4. Key Deployment Constraints (Critical Learnings)

4.1 Memory limits

Default Cloud Run memory (512MB) was insufficient.

Observed failure:

* container failed to start
* misleading error: “failed to listen on PORT=8080”

Actual cause:

* model loading exceeded memory during startup

Resolution: --memory 2Gi

4.2 Startup time constraints

Large model loading caused slow initialisation.

Resolution:
--timeout 300

4.3 Misleading error messages

Cloud Run error:

container failed to start and listen on PORT=8080

This did NOT indicate a networking issue.

Actual issue:

* application crashed before server startup

⸻

5. Build vs Runtime Separation

A critical distinction emerged:

Build-time correctness

* Dockerfile
* .gcloudignore
* correct file inclusion (model present)

Runtime correctness

* memory allocation
* startup time
* model loading behaviour

Both must be satisfied independently.

⸻

6. Model Packaging Strategy

The model is bundled inside the container:

/app/models/bioclinicalbert_final/

Advantages:

* no external downloads
* deterministic execution
* no authentication issues (e.g. HuggingFace)

⸻

7. Deployment Command (Final)

gcloud run deploy clinical-nlp-api \
  --image gcr.io/clinical-nlp-api/clinical-nlp-api \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300

⸻

8. Validation

Health check

curl https://clinical-nlp-api-1064509144938.europe-west1.run.app/health

Inference request

curl -X POST "https://clinical-nlp-api-1064509144938.europe-west1.run.app/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "..."}'

9. System Properties Achieved

* Production-style inference service
* Containerised ML system
* Cloud-native deployment
* Reproducible runtime environment
* External API access

---





---

## CI/CD

Right now deployment depends on:

* your local machine
* your CLI state
* manual commands

CI/CD turns deployment into:

* version-controlled
* repeatable
* environment-independent

Signalling (this matters for your goals)

You are building a portfolio for ML/AI roles.

CI/CD signals:

* you understand deployment pipelines
* you can productionise systems
* you remove human-in-the-loop deployment

This is not superficial — it is expected in industry.

it’s:

* not needed for this project to function
* but highly valuable as evidence of engineering maturity



---











---

What “deployment succeeded” actually means

When Cloud Run gives you a URL like:

https://clinical-nlp-api-xxxx.europe-west1.run.app

That means:

You now have a live server running in Google’s infrastructure.

Concretely:

* Google is hosting your Docker container
* It is running 24/7 (serverless, auto-managed)
* It scales automatically on traffic
* It restarts if it crashes
* It exposes HTTP endpoints (/health, /predict)

So your system is now:

Client → HTTPS URL → Cloud Run container → your FastAPI app → model inference

Does it run permanently?

Yes, but with nuance:

* It runs continuously while deployed
* It may scale to zero when idle (cold start)
* It is not “running on your laptop anymore”
* You only redeploy when you change code or image

So CI/CD is just automation of redeployments.

Why CI/CD still matters (even if you “won’t update it”)

Even if you never touch it again, CI/CD is still useful because:

* proves production-grade engineering workflow
* shows reproducible deployment pipeline
* allows future fixes without manual redeploy steps
* is a strong DevOps signal on CV / portfolio

But yes:

In a static academic project, it is mostly demonstration value.


























## Critical Issues & Fixes (this is the valuable part)

This is where you capture what you learned. Organise by category, not timeline.

1. Build Context / Docker Issues

* .gcloudignore vs .dockerignore
* Why models were missing in container
* Difference between local Docker success vs Cloud Build failure

Key insight:
Cloud Build only sees files sent in the build context → ignore files directly affect runtime behaviour.

⸻

2. Large Model Handling

* GitHub 100MB limit
* Git LFS requirement
* Why CI/CD failed without LFS pull
* JSON decode error from partial model files

Key insight:
Without git lfs pull, files exist but are pointers, not actual data → causes runtime crashes.

⸻

3. Cloud Run Runtime Failures

* “Container failed to start on PORT=8080”
* Root causes:
    * missing model
    * insufficient memory
    * slow startup

Fixes:

* --memory 2Gi
* --timeout 300


Key lessons from this failure

Build vs runtime are separate problems

* Build issues: .gcloudignore, Docker context
* Runtime issues: memory, startup, model loading

You resolved both layers.

Cloud constraints matter more than local correctness

Local success does not validate deployment:

* Cloud Run default memory (512MB) was insufficient
* Model load caused silent container crash

⸻

4. CI/CD vs Manual Deployment Mismatch

This is a big one.

You need to explicitly document:

“Manual gcloud run deploy worked, but CI/CD failed due to differences in execution environment.”

Reasons:

* local machine had model files
* CI/CD depended on GitHub repo state
* authentication differences
* build context differences

⸻

5. Cloud Build “False Failure” (important)

This is subtle and very valuable.

Problem:

* Build succeeds in GCP
* GitHub Actions fails with exit code 1

Cause:

* gcloud builds submit tries to stream logs
* service account lacks permissions → CLI fails

Fix:

* --async + manual wait loop

Key insight:

CI tools can fail even when underlying infrastructure succeeds.

⸻

6. IAM & Permissions

* service account roles required
* difference between:
    * running builds
    * reading logs
* why missing permissions caused misleading errors


WHY THIS FAILED (simple explanation)

Your pipeline is:

GitHub Actions → Cloud Build → GCR → Cloud Run

You gave permissions for:

* deploying (Run Admin) ✔
* storage ✔

But forgot:

* building images ❌ (Cloud Build permission)
* writing images ❌ (Artifact Registry permission)
* API disabled ❌ (project API gate)
