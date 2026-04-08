# Phase 3 - Transformer Contextual Validation Decisions

## Objective

- This document outlines the key design decisions for Phase 3: transformer-based validation of extracted clinical entities.
- This covers model selection, dataset construction, manual annotation guidelines, split strategy, and model training approach.
- Phase 3 builds on Phase 2, where deterministic rules generate high-recall, span-aligned candidate entities.
- The transformer layer converts these candidates into contextually valid, task-specific outputs via semantic classification.
- The goal is a reliable, reproducible and clinically meaningful validation layer.

---

## Model Selection

### 1. Objective

- Select a model capable of sentence-level, context-aware classification of extracted entities
- Ensure compatibility with Phase 2 schema (`sentence_text`, `entity_text`)
- Balance:
  - Clinical performance  
  - Computational efficiency  
  - Scalability  
  - Reproducibility    

This serves as the formal reproducibility specification for Phase 3 model selection.

---

### 2. Design Context (From Phase 2)

Phase 2 produces:

- High-recall candidate entities:
  - `SYMPTOM`
  - `INTERVENTION`
  - `CLINICAL_CONDITION`
- Structured schema with:
  - Span (`entity_text`)
  - Context (`sentence_text`)
  - Validation placeholders (`is_valid`, `confidence`, `task`)

Input–Output Requirements:

- **Input:** `(sentence_text, entity_text)`  
- **Output:**
  - `is_valid` → binary classification  
  - `confidence` → probability score  
  - `task` → entity-specific interpretation  

The task is entity-level contextual classification, not sequence labelling or text generation

---

### 3. Nature of the Task (Critical Clarification)

The task requires contextual reasoning specific to each entity type:

| Entity Type         | Required Reasoning |
|--------------------|------------------|
| `SYMPTOM`           | Negation (present vs absent) |
| `INTERVENTION`      | Intent vs execution (planned vs performed) |
| `CLINICAL_CONDITION`| Temporality, certainty, diagnostic status |

Examples:
- “no chest pain” → symptom = **invalid (negated)**
- “plan to start antibiotics” → intervention = **invalid (not performed)**
- “history of MI” → condition = **invalid (not active)**

Key Requirement:
- Task is to interpret meaning of entity within full sentence context
- Be able to handle ICU-specific language:
  - Clinical shorthand
  - Multi-clause sentences
  - Ambiguity and implicit meaning
- With constraints of reproducibility, efficiency, and structured outputs

---

### 4. Approaches Considered

#### 4.1 Model Types

| Approach                     | Strengths                              | Limitations |
|-----------------------------|----------------------------------------|-------------|
| Rule-based systems          | Interpretable, deterministic           | Cannot model complex context, brittle to variation |
| Classical ML (e.g. LR, SVM) | Simple, efficient                    | Requires manual feature engineering, weak semantic understanding |
| CNNs (text classification)  | Capture local n-gram patterns          | Limited long-range context, weak for negation/temporal scope |
| RNNs / LSTMs                | Sequential modelling                   | Struggle with long-range dependencies, less efficient than transformers |
| Transformers (BERT-style)   | Full-context attention, strong semantics | Higher computational cost |

---

#### 4.2 Key Limitations of Non-Transformer Models

**1. Classical Machine Learning (e.g. Logistic Regression, SVM)**

These models rely on explicit feature engineering, typically using:
- Bag-of-words / TF-IDF representations  
- Manually engineered indicators (e.g. negation flags, keyword presence)

Limitations:
- No inherent understanding of word order or context  
- Inability to model relationships between tokens within a sentence  
- Heavy reliance on manually defined features such as:
  - Negation rules (“no”, “denies”)  
  - Temporal indicators (“history of”, “previous”)  

Implication:
- Reintroduces the same brittleness as rule-based systems  
- Poor generalisation to:
  - Unseen phrasing  
  - Clinical shorthand  
  - Complex sentence structures  

---

**2. Convolutional Neural Networks (CNNs for Text)**

CNNs operate by learning local n-gram patterns through sliding convolutional filters.

Strengths:
- Effective at capturing short phrases (e.g. “chest pain”, “respiratory failure”)  
- Computationally efficient  

Limitations:
- Restricted receptive field (limited context window)  
- Weak modelling of long-range dependencies  

Failure cases:
- Negation scope: “no evidence of chest pain”  
- Multi-clause sentences: “patient denies chest pain but reports shortness of breath”  

Implication:
- Cannot reliably determine whether an entity is valid within full sentence context  

---

**3. Recurrent Neural Networks (RNNs / LSTMs)**

RNN-based models process text sequentially, maintaining a hidden state across tokens.

Strengths:
- Designed for sequential data  
- Can, in theory, capture contextual flow  

Limitations:
- Difficulty capturing long-range dependencies:
  - Early tokens (e.g. negation) may not effectively influence later tokens  
- Gradient degradation over long sequences (even with LSTM/GRU variants)  
- Limited parallelisation → slower training and inference  

Failure cases:
- Long clinical sentences with multiple clauses  
- Context-dependent meaning spanning distant tokens  

Implication:
- Inconsistent performance for:
  - Negation handling  
  - Temporal reasoning  
  - Complex clinical phrasing 

---

#### 4.3 Conclusion

Across all non-transformer approaches:

- Context is either not modelled or only partially captured through manual features or local patterns
- Long range dependencies and complex sentence structures are not effectively handled
- Robust handling of negation, temporality, and intent is not achievable without extensive manual engineering

These limitations directly conflict with the requirements of Phase 3, where:

- Entity validity depends on full-sentence contextual interpretation
- Clinical meaning is often distributed across multiple tokens and clauses  

As a result, non-transformer models are not suitable for this task, motivating the selection of transformer-based architectures.

---

#### 4.3 Why Transformers Are Selected

Transformers address these limitations through:

- **Self-attention mechanism:** Directly model relationships between all tokens in a sentence  
- **Context-aware representations:** Meaning of a word depends on surrounding words (contextualized embeddings)  
- **Implicit feature engineering:** Negation, temporality, and intent are learned from data, no manual rule encoding

This is critical for distinguishing:

- **Negation:** “no chest pain” vs “chest pain”
- **Intent vs execution:** “planned intubation” vs “intubated”  
- **Temporality:** “history of stroke” vs “acute stroke”  

Transformer-based models are the minimal class of models capable of reliably performing the required contextual validation task.

---

### 5. Transformer Classes Considered

#### 5.1 Transformer Variants

| Model Type              | Strengths                                  | Weaknesses |
|------------------------|---------------------------------------------|------------|
| **General BERT**           | Efficient, well-supported                   | Weak clinical understanding |
| **Clinical BERT variants** | Domain-specific language understanding      | Slightly heavier |
| **Large LLMs (GPT, Gemma)**| Strong reasoning, flexible                  | Expensive, non-deterministic, poor structure control, unstable outputs |
| **Custom-trained**   | Fully tailored                             | Requires large labelled dataset |

---

#### 5.2 Encoder vs Generative Models

**Encoder Models (BERT-style):**

- Input → full sequence processed simultaneously  
- Output → fixed contextual representation → classification head  
- Designed for:
  - Classification
  - Token labelling
  - Sentence-level understanding  

Properties:
- Bidirectional context (full sentence considered at once)
- Outputs structured probabilities (e.g. logits → softmax)
- Deterministic at inference:
  - Given fixed weights and no stochastic layers (e.g. dropout disabled), the same input produces identical output
- Strong alignment with structured prediction tasks

**Generative Models (LLMs):**

- Input → sequential token generation  
- Output → generated text (token-by-token)  
- Designed for:
  - Open-ended reasoning
  - Text generation
  - Instruction following  

Properties:
- Autoregressive decoding (predict next token repeatedly)
- Output depends on:
  - Prompt phrasing  
  - Decoding strategy (temperature, top-k, top-p)  
- Inherently probabilistic:
  - Even with temperature = 0, outputs may vary due to:
    - Implementation-level nondeterminism
    - Tie-breaking between tokens
- Produces unstructured text rather than fixed schema outputs  

---

#### 5.3 Why LLMs Are Not Used

Although LLMs are powerful, they are fundamentally misaligned with the requirements of this pipeline.

**1. Task mismatch**

- Required task: Binary classification (`is_valid`) with structured outputs  
- LLM capability: Free-form text generation  
- Implication:
    - Requires prompt engineering to simulate classification  
    - Adds unnecessary abstraction and failure modes 

**2. Output instability and lack of strict determinism**

- LLM outputs are probabilistic by design:
  - Token generation involves probability distributions  
- Even with constrained decoding, outputs can vary across runs, hardware, or implementations  
- Problems:
  - Cannot guarantee strict JSON schema compliance  
  - Requires post-processing, output validation, and error handling layers  
- In contrast, encoder models produce:
  - Fixed-dimensional outputs
  - Stable probabilities
  - Fully reproducible predictions 

**3. Scalability and efficiency constraints**

- LLMs have high computational cost per inference and slow sequential decoding  
- Pipeline requirement:
  - High-throughput batch classification over thousands of entities  
- Implication:
  - LLMs are inefficient and costly at scale 

**4. Reproducibility and auditability**

- Clinical pipelines require:
  - Stable outputs  
  - Reproducible behaviour  
  - Auditability of decisions  
- LLMs are sensitive to:
  - Model updates  
  - Prompt changes  
  - Sampling behaviour  
- Encoders provide:
  - Deterministic forward pass 
  - Consistent outputs across runs  

LLMs are better suited for exploration, prototyping, and complex reasoning tasks, but not for high-throughput, structured, deterministic validation pipelines in clinical contexts.

---

### 6. Transformer Decision

#### 6.1 Final Model Class

Chosen approach:

1. Clinical-domain pretrained **encoder transformer**
2. Supervised **fine-tuning for classification**

Candidate models:
- BioClinicalBERT  
- PubMedBERT  
- ClinicalBERT 

---

#### 6.2 Why Clinical Pretrained Models

Clinical text differs significantly from general language:

- Abbreviations: “SOB”, “PRBC”, “NGT”  
- Domain-specific terminology  
- Fragmented, shorthand-heavy syntax  
- Non-standard grammar  

General-domain models:

- Trained on Wikipedia / BooksCorpus  
- Limited exposure to clinical language  
- Misinterpret:
  - Abbreviations  
  - ICU shorthand  
  - Domain-specific phrasing  

Clinical-domain models:

- Pretrained on:
  - MIMIC clinical notes (BioClinicalBERT)  
  - PubMed abstracts (PubMedBERT)  
- They learn:
  - Clinical vocabulary  
  - Abbreviation usage  
  - Real-world documentation patterns  

Outcome:

- Better contextual understanding  
- Higher classification accuracy  
- Reduced need for large fine-tuning datasets  

---

#### 6.3 Why Fine-Tuning vs Training from Scratch

**A. Model Training From Scratch**

Model training from scratch involves:

- Initialising model weights randomly  
- Training on very large corpora (millions–billions of tokens)  
- Learning:
  - Language structure  
  - Grammar  
  - Domain knowledge  
  - Semantic relationships  
- Requirements:
  - Massive labelled or unlabeled datasets  
  - Significant compute (GPUs/TPUs)  
  - Long training time  

Why training from scratch is not feasible:

- Phase 2 does not provide:
  - Large-scale labelled datasets  
  - Sufficient data diversity  
- Resource constraints:
  - Compute cost is prohibitive  
  - Time-to-iteration is too slow  
- Most importantly:
  - The task does not require learning language from scratch  
  - Only requires adapting existing language understanding  

---

**B. Fine-Tuning**

Fine-tuning involves:

- Starting from a pretrained model  
- Adding a task-specific classification head  
- Training on a smaller labelled dataset  

Process:

1. Input: `(sentence_text, entity_text)`  
2. Encode full sentence context  
3. Extract contextual representation  
4. Pass through classification head  
5. Optimise for task-specific objective (e.g. binary classification loss)  

Why fine-tuning is appropriate:

- Leverages pretrained knowledge as language understanding is already learned  
- Requires much less data and less compute  
- Adapts model to:
  - Task-specific definitions (e.g. “valid intervention”)  
  - Entity-specific semantics  

Fine-tuning transforms a general clinical language model into a task-specific clinical reasoning model without needing large-scale training.

---

#### 6.4 Candidate Model Comparison

| Model              | Pretraining Data        | Domain Focus        | Strengths | Limitations |
|------------------|------------------------|---------------------|----------|------------|
| **BioClinicalBERT** | MIMIC clinical notes    | ICU / clinical text | Best match to ICU language, strong shorthand handling | Slightly domain-specific |
| **PubMedBERT**     | PubMed abstracts        | Biomedical research | Strong biomedical knowledge, robust terminology | Less exposure to clinical note structure |
| **ClinicalBERT**   | Mixed clinical corpora  | General clinical    | Earlier clinical adaptation | Older, less optimised pretraining |

- **BioClinicalBERT:** best for real-world ICU notes (matches dataset distribution)  
- **PubMedBERT:** best for formal biomedical literature  
- **ClinicalBERT:** earlier, less specialised variant  

---

#### 6.5 Final Model Choice

Selected: **BioClinicalBERT**

**Rationale:**

1. **Data alignment**
  - Pretrained on MIMIC notes → closest match to ICU dataset  

2. **Language compatibility**
  - Strong handling of:
    - Abbreviations  
    - Clinical shorthand  
    - Irregular sentence structures  

3. **Task alignment**
  - Designed for clinical NLP tasks  
  - Proven performance on:
    - Clinical classification  
    - Entity understanding
  - Fine-tuning suitable for binary classification of entity validity  

4. **Efficiency**
  - Encoder-based → fast, parallelisable implementation  
  - Compute efficient
  - Suitable for batch inference  

**Trade-offs:**

| Aspect        | Outcome |
|--------------|--------|
| Performance  | High (domain-aligned) |
| Efficiency   | Moderate (acceptable for batch processing) |
| Determinism  | Strong (stable inference outputs) |
| Flexibility  | Lower than LLMs, but sufficient for classification |

---

### 7. Final Design Decision

Final design:
- Use BioClinicalBERT (clinical pretrained encoder)
- Apply supervised fine-tuning for entity-level classification
- Input: `(sentence_text, entity_text)`
- Output: `is_valid`, `confidence`, `task`

Key architectural decisions:
- Deterministic encoder-based inference  
- No generative modelling  
- No expansion of rule-based logic  
- No reliance on prompt-based systems 

The result is a robust, scalable, and reproducible validation layer that:
- Accurately interprets clinical context  
- Resolves ambiguity (negation, temporality, intent)  
- Integrates seamlessly with Phase 2 outputs  
- Produces structured, audit-ready predictions for downstream use  

---

## Dataset Construction and Annotation Preparation

### 1. Objective

To construct a high-quality, annotation-ready dataset for training and evaluating a transformer-based validation model. The dataset must:

- Be sufficient for fine-tuning a pretrained model
- Be efficient to manually annotate
- Maintain balanced representation across entity types
- Support reproducible downstream training, validation, and testing

---

### 2. Sampling Strategy

#### 2.1 Dataset Size

A total of 600 entities were selected for annotation. This size was chosen because:

- It is sufficient for fine-tuning a pretrained transformer (e.g., ClinicalBERT)
- It supports binary classification at sentence level
- It balances annotation effort vs performance gains

The datasset size is too small for training from scratch, but appropriate for fine-tuning pretrained models: 
 
- This is because the model already encodes language structure and clinical semantics. 
- The dataset is used to learn task-specific definitions and decision boundaries  

Increasing beyond 600:

- Produces diminishing returns
- Increases annotation cost significantly
- Is unnecessary for this project scope  

Dataset expansion (800–1000) is only justified if:

- Model performance is unstable  
- Clear underfitting is observed  

---

#### 2.2 Balanced Sampling by Entity Type

Sampling is performed equally across entity types using `entity_type`-based sampling:

- `SYMPTOM`: 200  
- `INTERVENTION`: 200  
- `CLINICAL_CONDITION`: 200  

This ensures:

- Balanced semantic coverage  
- Reduced bias toward any entity type  
- Improved generalisation across tasks  

---

### 3. Field Selection and Data Structure

The dataset is constructed from the JSONL extraction outputs (`extraction_candidates.jsonl`) and flattened into tabular csv format.

#### 3.1 Retained Fields

- `note_id` → traceability and debugging  
- `section` → contextual location in note  
- `concept` → normalized concept label  
- `entity_text` → extracted entity span  
- `entity_type` → classification category (SYMPTOM, INTERVENTION, CLINICAL_CONDITION)
- `sentence_text` → classification context (critical input)  
- `negated` → retained for rule-based comparison (not used in training)  
- `task` → defines classification objective  
- `confidence` → placeholder (0.0), used later for model outputs  

---

#### 3.2 Annotation Field

`is_valid` is the only manually annotated field:

- Binary label: `True` / `False`
- Represents ground truth for model training  

All other fields remain unchanged to preserve consistency with upstream extraction to ensure:

- Low annotation complexity  
- High consistency  
- Clear separation between:
  - Extraction (rule-based)
  - Validation (model-based)

---

#### 3.3 Confidence Score Retention

The `confidence` field is retained in the dataset but is not used during manual annotation or training input.

Its purpose is to support post-training evaluation and calibration, where it will store model-generated confidence scores. This enables:

- Threshold tuning (optimising decision boundaries)
- Prediction ranking (prioritising high-confidence outputs)
- Error analysis (identifying overconfident incorrect predictions)

The field is initialised during sampling (default `0.0`) to maintain schema consistency and allow seamless integration of model outputs later in the pipeline.

---

#### 3.4 Negation Flag Retention

The `negated` field is retained to enable comparison between rule-based extraction outputs and transformer-based validation, as its role differs by entity type. 

For `SYMPTOM`:

- The `negated` flag is populated during rule-based extraction (Phase 2)
- It directly determines rule-based validity:
  - `negated = True` → invalid (`False`)
  - `negated = False` → valid (`True`)
- This provides a strong baseline, as symptom negation is reliably captured using deterministic rules

For `INTERVENTION` and `CLINICAL_CONDITION`:

- The `negated` field is not populated (null) as no rule-based validity filtering is applied
- All extracted entities are treated as valid under the rule-based system
- This reflects an intentional high-recall design:
  - Maximise capture of candidate entities
  - Defer validity determination to the transformer model
- Validity for these entity types depends on temporality and intent, which cannot be reliably captured using deterministic rules alone

The `negated` field is therefore critical for `SYMPTOM` validation:

- Establishes a strong rule-based baseline where transformer improvements are expected to be more modest
- Enables direct comparison between deterministic negation logic and transformer predictions

For `INTERVENTION` and `CLINICAL_CONDITION` however:

- Model performance improvements are evaluated independently of `negated`, as no rule-based filtering is applied to these entity types.
- This establishes a weak rule-based baseline, allowing the transformer to provide substantial performance gains by learning complex contextual cues that determine validity.

---

### 4. Output Design

Two files are generated for reproducibility and data integrity:

`annotation_sample_raw.csv`:
- Always overwritten  
- Contains freshly sampled dataset  
- Used for traceability and reproducibility  

`annotation_sample_labeled.csv`:
- Created only if it does not exist  
- Used for manual annotation  
- Preserved to prevent accidental data loss  

This design ensures:
- Original sample is always recoverable  
- Manual annotations are never overwritten  

---

### 5. Implementation Workflow 

The dataset construction process is implemented in `sample_entities.py` and follows a deterministic, reproducible pipeline:

1. **Load extraction candidates**
  - Read JSONL file (`extraction_candidates.jsonl`)
  - Each line corresponds to a single extracted entity
  - Parse each line into a Python dictionary

2. **Field extraction and flattening**
  - Extract relevant fields:
    - `note_id`, `section`, `concept`, `entity_text`, `entity_type`, `sentence_text`, `negated`
  - Extract nested validation fields safely:
    - `task` via `row.get("validation", {}).get("task")`
    - `confidence` with default `0.0`
  - Append all records into a list

3. **DataFrame construction**
  - Convert records into a pandas DataFrame
  - This forms the full pool of candidate entities (~40k+)

4. **Per-entity-type filtering**
  - Split DataFrame into subsets by:
    - `SYMPTOM`
    - `INTERVENTION`
    - `CLINICAL_CONDITION`

5. **Balanced sampling**
  - For each entity type:
    - Verify sufficient sample size (≥200), else raise error
    - Randomly sample **200 rows** using fixed `random_state=42`
  - Ensures reproducibility and equal class representation

6. **Dataset assembly**
  - Concatenate sampled subsets into a single DataFrame (600 rows total)
  - Reset index to ensure clean ordering

7. **Global shuffling**
  - Shuffle entire dataset (`frac=1`, `random_state=42`)
  - Prevents ordering bias during manual annotation

8. **Annotation column initialization**
  - Add `is_valid` column initialised to `None`
  - Serves as the ground truth label to be manually populated

9. **Output handling**
  - Save `annotation_sample_raw.csv`:
    - Always overwritten
    - Acts as reproducible source sample
  - Save `annotation_sample_labeled.csv`:
    - Created only if it does not already exist
    - Prevents accidental overwrite of manual annotations

This pipeline ensures:
- Deterministic sampling (via fixed seed)
- Balanced representation across entity types
- Preservation of annotation work
- Full reproducibility of dataset construction

---

## Manual Annotation

### 1. Objective

To define a clear, consistent, and reproducible framework for assigning binary labels (`is_valid = True/False`) to extracted clinical entities.

The goal is to:
- Ensure high-quality ground truth labels for transformer training  
- Minimise annotation variability and drift
- Standardise interpretation of clinical language across entity types  
- Align task labels with clinically meaningful and model-relevant definitions 

Annotation is performed strictly based on sentence-level context, ensuring that labels reflect the information explicitly present in the text rather than inferred assumptions.

---

### 2. Annotation Framework and Principles

- **Sentence-level grounding**  
  - Labels are assigned using the full `sentence_text`  
  - The `entity_text` is interpreted only within this context

- **Context over metadata**  
  - `section` may assist interpretation in ambiguous cases, but is not determinative
  - `concept` can be used to verify whether the extracted entity aligns with the intended clinical meaning
  - If the entity is clearly mis-extracted (i.e. does not match the intended concept in context), label as `False`

- **Explicit evidence over inference**  
  - Labels must be based only on clearly stated information 
  - Do not infer presence, action, or activity from typical clinical patterns

- **Consistency over intuition**  
  - Apply rules systematically, even if clinically imperfect  
  - Prioritise reproducibility over subjective clinical judgement
  - This ensures learnable and reproducible patterns for the model  

- **Ambiguity handling (conservative bias)**  
  - Default to conservative interpretation  
  - Uncertain or implied mentions are typically labeled `False` unless clearly active/performed  

- **Temporal consistency**
	-	Interpret timing strictly based on wording in the sentence
	-	Do not assume recency or continuity unless explicitly stated
	-	This is critical for:
	  -	State-based tasks (current vs past)
	  -	Event-based tasks (performed vs planned)

---

### 3. State vs Event-Based Labeling

A key design decision is the use of state-based vs event-based labeling, depending on entity type which defines the task type. This dual approach is standard, defensible, and consistent with real-world clinical use cases and broad downstream applications.

| Entity Type                | Labeling Type | Interpretation |
|---------------------------|--------------|----------------|
| `symptom_presence`          | State-based  | Is it present now? |
| `intervention_performed`    | Event-based  | Has it occurred during admission? |
| `clinical_condition_active` | State-based  | Is it currently active? |

State-based:
- Reflects the patient’s current status at the time of the note
- Past, resolved, or uncertain mentions → `False`

Event-based:
- Reflects whether an action has occurred at any point during admission
- Includes current, completed, or past interventions → `True` if performed

This distinction is critical and must be applied consistently.

---

### 4. Entity-Specific Task Guidelines

#### 4.1 `symptom_presence`

**Overview:**
- Definition: Does the patient currently exhibit the symptom?
- Type: State-based

**Justification:** 
- Symptoms are inherently real-time
- Downstream tasks like monitoring, alerts, or clinical decision support (CDS) require the current status, not historical mentions.

**Rules:**
- `True` → symptom is currently present  
- `False` → negated, historical, chronic baseline, or not actively occurring  

**Examples:**
- “Patient is complaining of nausea” → `True`  
- “Patient denies pain” → `False` 

**Edge Cases:**
- Historical or chronic baseline symptoms → `False` 
- Provoked only (e.g. “asked to cough”) → `False`  
- “PRN nausea” → `False` (indication only, not current symptom)  
- Ambiguous phrasing → default `True` only if likely current  

---

#### 4.2 `intervention_performed`

**Overview:**
- Definition: Has the intervention been performed at any point during the admission up to this note?
- Type: Event-based

**Justification:**
- Interventions are often documented retrospectively. 
- Even if completed by the time of the note, they still count as having occurred. 
- This is useful for retrospective analyses, modeling treatment patterns, or cohort studies.

**Rules:**
- `True` → intervention has occurred (past or ongoing)  
- `False` → planned, conditional, hypothetical, or not confirmed  

**Examples:**
- “Received 2 units PRBCs” → `True`  
- “Plan to start heparin” → `False`  

**Edge Cases:**
- **PRN medications** → `False` unless explicitly administered  
- **Weaning/continuation** → `True` (active process)  
- **“Post antibiotics” / “received prior”** → `True` (event occurred)  
- **Prophylaxis headings** → `False` unless explicitly started (e.g. “Started heparin subq for DVT prophylaxis” → `True`)
- **Implicit presence (e.g. ETT in place)** → `True` (interventions like tubes and lines are typically documented in a way that implies they have been performed)  

---

#### 4.3 `clinical_condition_active`

**Overview:**
- Definition: Is the condition currently active and affecting the patient?
- Type: State-based

**Justification:**
- Acute conditions need to reflect current patient status. 
- This is aligned with real-time decision-making, risk assessment, and condition tracking.

**Rules:**
- `True` → explicitly active and clinically relevant  
- `False` → historical, resolved, chronic baseline, negated, or uncertain  

**Examples:**
- “Worsening ARDS” → `True`  
- “Resolved pneumonia” → `False`  

**Edge Cases:**
- **Implicit context:** “Hx diabetes admitted for sepsis” → diabetes `False`, sepsis `True`  
- **Uncertainty:** “Possible pneumonia” → `False`  
- **Causal mentions:** “Lactate elevated due to sepsis physiology” → `False` unless explicitly active  
- **Headers / lists:** ALL CAPS or diagnostic lists → `False` unless sentence confirms activity or sentence context indicates active relevance 
- **Single-term mentions:** “Sepsis” alone → `False` without context (e.g. the `section` field, or the entity itself)

---

### 3. Manual Annotation Validation 

#### 3.1 Purpose

Dataset sampling in general can be subject to errors such as incorrect filtering, mislabeling, or data corruption. Manual annotation also introduces potential risks such as incomplete labels, formatting inconsistencies, and distributional imbalances.

Validation is therefore required to:

- Ensure all 600 samples are fully annotated (`is_valid` complete)
- Confirm labels are strictly binary (`True` / `False`)
- Verify dataset structure matches expectations (no missing critical fields)
- Ensure balanced task representation (200 per task)
- Check label distribution among tasks for consistency and absence of severe imbalance
- Detect errors early before dataset splitting and model training

This ensures the dataset is:

- Structurally complete, label-consistent, and statistically balanced
- Ready for reproducible train/validation/test splitting

This prevents propagation of annotation errors into training, which would otherwise degrade model performance and invalidate evaluation results.

---

#### 3.2 Validation Workflow

The validation logic is implemented in `validate_manual_annotations.py` and follows a structured sequence of checks:

1. **Load annotated dataset**  
  - Read `annotation_sample_labeled.csv` into a DataFrame  
  - Establish dataset as the single source of truth for downstream steps  

2. **Structural integrity checks**  
  - Confirm total row count (expected: 600)  
  - Verify all expected columns are present  
  - Detect any global missing values  

3. **Critical field validation**  
  - Ensure no missing values in:
    - `is_valid` (must be fully annotated)
    - `task` (required for stratification)
    - `sentence_text` (required for model input)  

4. **Label validation (`is_valid`)**  
  - Confirm only valid binary labels are present (`True`, `False`)  
  - Identify and flag any invalid or unexpected values  
  - Compute overall label distribution to assess balance  

5. **Task distribution checks**  
  - Verify exact balance across tasks:
    - 200 `symptom_presence`
    - 200 `intervention_performed`
    - 200 `clinical_condition_active`  
  - Flag deviations from expected counts  

6. **Task–label cross-distribution analysis**  
  - Generate cross-tabulation of `task` vs `is_valid`  
  - Inspect for:
    - Severe imbalance within a task
    - Potential annotation bias or systematic errors  

7. **Diagnostic output and review**  
  - Print all validation metrics and warnings  
  - Require manual review before proceeding  
  - Only proceed to stratified splitting if no critical issues are detected  

---

#### 3.3 Validation Output Interpretation

**A. Dataset Structure**

- Total rows: **600** → matches expected sample size  
- All required columns present  
- No structural inconsistencies detected  

Dataset structure is correct and complete.

---

**B. Missing Values**

- No missing values in critical fields:
  - `task`: 0  
  - `sentence_text`: 0  
  - `is_valid`: 0  

- `negated`: 400 missing → expected and correct  
  - Only populated for `SYMPTOM` entities  
  - Not used for `INTERVENTION` or `CLINICAL_CONDITION`  

Missingness is expected and does not affect training.

---

**C. Label Integrity**

- Unique values: `[True, False]` → valid binary labels  
- Invalid label rows: **0**  

Labels are clean, consistent, and usable.

---

**D. Label Distribution**

| Label | Count |
|------|-------|
| True | 305   |
| False | 295  |

- Near-balanced distribution (~51% / 49%)
- No significant class imbalance detected

Dataset is suitable for training without label rebalancing

---

**E. Task Distribution**

| Task                        | Count |
|-----------------------------|-------|
| `symptom_presence`            | 200   |
| `clinical_condition_active`   | 200   |
| `intervention_performed`      | 200   |

- Exact balance across all tasks  

Stratified sampling was correctly implemented.

---

**F. Task vs Label Distribution**

| Task                        | `False` | `True` |
|-----------------------------|---------|--------|
| `clinical_condition_active`   | 122     | 78     |
| `intervention_performed`      | 71      | 129    |
| `symptom_presence`            | 102     | 98     |

- **symptom_presence:** Balanced (~50/50); maybe due to straightforward nature of symptom mentions
- **intervention_performed**: Skewed toward `True`; expected due to frequent documentation of completed interventions
- **clinical_condition_active** → skewed toward `False`; expected due to historical conditions, uncertain diagnoses, and header-like extractions  

Distribution patterns are clinically and methodologically consistent. No correction required.

---

**Final Assessment**

The dataset is validated and ready for stratified train/validation/test splitting.

- All structural checks passed  
- All labels valid and complete  
- Balanced dataset across tasks  
- Acceptable and explainable label distributions  

---

## Dataset Splitting

### 1. Objective

To partition the fully annotated dataset into train, validation, and test sets that:

- Preserve statistical integrity across all splits  
- Maintain balanced representation of:
  - Entity tasks (`symptom_presence`, `intervention_performed`, `clinical_condition_active`)
  - Binary labels (`is_valid`: True/False)  
- Prevent data leakage between training and evaluation  
- Enable reliable model training, tuning, and final evaluation  

This is the final preparation step before transformer fine-tuning and directly determines the validity of all downstream performance metrics.

---

### 2. Splitting Strategy

#### 2.1 Split Proportions

The dataset (600 annotated entities) is split as follows:

- **Train:** 420 (70%)  
- **Validation:** 90 (15%)  
- **Test:** 90 (15%)  

This split is appropriate for transformer fine-tuning, where:

- The model already encodes language and clinical knowledge  
- The dataset is used to learn task-specific decision boundaries  
- Additional data yields diminishing returns relative to annotation cost  

---

#### 2.2 Per-Entity Distribution

Each entity type is equally represented in the full dataset (200 each), and this balance is preserved across splits:

| Split       | Per Entity Type | Total |
|------------|----------------|-------|
| Train      | 140            | 420   |
| Validation | 30             | 90    |
| Test       | 30             | 90    |

This ensures consistent semantic coverage across all datasets.

---

#### 2.3 Reproducibility

- A fixed `random_state=42` is used during splitting  
- Ensures identical dataset partitions across runs  

This is required for:
- Reproducibility of experiments  
- Consistent debugging  
- Fair comparison of model iterations  

---

### 3. Rationale for Split Design

**Training Set (70%)**

- Large enough to learn:
  - Clinical language patterns  
  - Negation and contextual cues  
  - Task-specific decision boundaries  
- Maximises data available for fitting without compromising evaluation sets  

---

**Validation Set (15%)**

Used during training to:

- Monitor performance across epochs  
- Detect overfitting  
- Select the best model checkpoint  

Without a validation set:
- Model selection becomes unreliable  
- Overfitting cannot be properly controlled  

---

**Test Set (15%)**

Used only once after training to:

- Provide unbiased final evaluation  
- Report:
  - Accuracy  
  - Precision / Recall / F1  
  - Improvement over rule-based baseline  

Strict separation ensures no leakage from training or tuning.

---

### 4. Stratification Strategy

#### 4.1 Purpose of Stratification

Stratification is mandatory. Without it:

- Task imbalance may occur → biased learning  
- Label imbalance may occur → incorrect decision boundaries  
- Evaluation metrics become unreliable  
- Entity-level comparisons become invalid  

---

#### 4.2 Stratification Dimensions

Two independent distributions must be preserved:

- `task` (3 classes)  
- `is_valid` (binary labels)  

Single-dimension stratification is insufficient:

- Stratifying only by `task`:
  - Preserves 200/200/200  
  - Breaks True/False balance within tasks  

- Stratifying only by `is_valid`:
  - Preserves overall label balance  
  - Breaks task distribution  

Both must be preserved simultaneously.

---

#### 4.3 Combined Stratification

To preserve both simultaneously, a combined stratification key is constructed: `stratify_key = task + “_” + is_valid`

- This is a categorical variable with 6 unique values (3 tasks × 2 labels)
- This is added as a new column to the dataset `stratify_key`
- Used as the `stratify` variable during splitting 

Example:

| task                        | is_valid | stratify_key                         |
|----------------------------|----------|--------------------------------------|
| `symptom_presence`           | `False`    | symptom_presence_False               |
| `intervention_performed`     | `True`     | intervention_performed_True          |
| `clinical_condition_active`  | `False`    | clinical_condition_active_False      |

This ensures:

- Task proportions are preserved  
- Label distributions are preserved within each task  

The `stratify_key` is removed after splitting as it is not part of the dataset schema and must not be used as a model input.

---

#### 4.4 Two-Stage Stratified Splitting

A two-stage approach is used because direct three-way stratified splitting is not supported by scikit-learn's `train_test_split` function.

**Stage 1 — Train vs Temp**

- 600 → Train (420) + Temp (180)  
- Stratified using `stratify_key`  
- Ensures training set reflects full dataset distribution  

**Stage 2 — Validation vs Test**

- 180 → Validation (90) + Test (90)  
- Stratified again using `stratify_key`  

---

#### 4.5 Why Two-Stage Splitting Is Required

This approach ensures:

- Consistent distribution across all splits  
- No overlap between datasets  
- Proper statistical independence  
- Reproducible and controlled partitioning  

Without this:

- Sequential random splits introduce imbalance  
- Validation/test sets may become biased  
- Final evaluation becomes unreliable  

---

### 5. Implementation Workflow

The code and logic are implemented in `stratified_split.py` and follow a controlled, reproducible pipeline:

1. **Load validated dataset**  
  - Read `annotation_sample_labeled.csv` (600 rows)  
  - Assumes prior validation has confirmed:
    - No missing labels  
    - Correct task distribution (200 per task)  
    - Valid `True`/`False` labels  

2. **Create stratification key**  
  - Construct `stratify_key = task + "_" + is_valid`  
  - Encodes joint distribution of:
    - Task type  
    - Label (valid/invalid)  
  - Enables simultaneous preservation of both dimensions during splitting  

3. **Stage 1 split (Train vs Temp)**  
  - Apply `train_test_split` with:
    - `test_size=0.30` → 180 samples (temp set)  
    - `stratify=stratify_key`  
    - `random_state=42`  
  - Output:
    - Train: 420  
    - Temp: 180  
  - Ensures training set reflects full dataset distribution  

4. **Stage 2 split (Validation vs Test)**  
  - Split temp set using:
    - `test_size=0.50` → 90 / 90  
    - `stratify=temp_df["stratify_key"]`  
  - Output:
    - Validation: 90  
    - Test: 90  
  - Maintains distribution consistency in evaluation sets  

5. **Remove stratification key**  
  - Drop `stratify_key` from all splits  
  - Prevents:
    - Inclusion of artificial feature  
    - Potential data leakage into model training  

6. **Reset indices**  
  - Reset row indices for each split (`0 → n-1`)  
  - Ensures clean, sequential datasets for downstream processing  

7. **Verify split integrity**  
  - Print and confirm:
    - Dataset sizes (420 / 90 / 90)  
    - Task distribution in each split  
    - `is_valid` distribution  
    - Task vs label cross-tabulation  
  - Confirms stratification has been correctly applied  

8. **Save outputs**  
  - Write splits to `data/extraction/splits`:
    - `train.csv`  
    - `val.csv`  
    - `test.csv`  
  - Saved without indices to ensure clean CSV format  

---

### 6. Split Validation

#### 6.1 Output Analysis

**A. Dataset size verification**

- Total dataset: 600 rows
- Split sizes:
  - Train: 420
  - Validation: 90
  - Test: 90

These match the intended 70/15/15 split exactly, confirming correct partitioning.

**B. Task Distribution**

- Train: 140 / 140 / 140 across all three tasks → perfectly balanced  
- Validation & Test: Minor variation (29–31 per task) due to integer constraints during stratification; no systematic skew toward any task  

Interpretation:
- Task balance is preserved across all splits  
- Minor deviations are expected and statistically negligible 

**C. Label Distribution**

| Split       | True | False |
|------------|------|-------|
| Train      | 214  | 206   |
| Validation | 46   | 44    |
| Test       | 45   | 45    |

Interpretation:
- Near 50/50 distribution across all splits  
- No class imbalance introduced  
- Suitable for stable binary classification training  

**D. Task vs Label Distribution**

Across all splits:
- Each task retains a similar internal True/False ratio  
- Example (Train):
  - `clinical_condition_active`: 85 False / 55 True  
  - `intervention_performed`: 50 False / 90 True  
  - `symptom_presence`: 71 False / 69 True  
- Validation and test sets closely mirror these proportions.

Interpretation:
- Joint distribution of (`task`, `is_valid`) is preserved  
- Confirms correct use of `stratify_key`  
- Ensures:
  - Consistent learning signal in training  
  - Valid comparison during evaluation  

---

#### 6.2 Overall Assessment 

All validation checks confirm:

- Correct split sizes  
- Preserved task balance  
- Preserved label balance  
- Preserved joint distribution (task + label)  
- No evidence of bias or skew in any split  

The dataset is now:

- Statistically valid  
- Properly stratified  
- Fully reproducible  
- Free from leakage between splits  

No further preprocessing is required.

The pipeline is ready to proceed to transformer fine-tuning and evaluation.

---

## Transformer Training and Validation

### 1. Objective

The objective of this step is to train a transformer-based classifier (BioClinicalBERT) that functions as a validation layer for rule-based entity extraction.

This transforms the pipeline from a purely rule-based system (which prioritises recall but produces false positives) into a hybrid system where rule-based extraction generates candidates and the model filters them using contextual understanding.

The aim is to improve overall extraction quality by:
- Increasing precision through removal of false positives  
- Maintaining recall from the rule-based stage  
- Learning contextual patterns that cannot be captured by fixed rules  

This step introduces a data-driven, context-aware validation mechanism that improves robustness and generalisation of the extraction pipeline.

---

### 2. Learning Paradigm: Fine-Tuning for Classification

#### 2.1 Overview 

This step uses fine-tuning, a form of transfer learning, rather than training a model from scratch.

- Training refers to the optimisation process (epochs, batches, gradient updates)  
- Fine-tuning specifies how training is performed: starting from a pretrained model and adapting it to a new task  

In this context:
- The model is trained 
- That training is specifically fine-tuning of a pretrained transformer

---

#### 2.2 Workflow

The fine-tuning process consists of adapting a pretrained model (**BioClinicalBERT**) to a binary classification task:

1. Load pretrained BioClinicalBERT (already trained on clinical text)
2. Add a classification head (binary output layer)
3. Input task-specific data (entity + context)
4. Update model weights using labelled examples


| Component | Role | What Changes During Training |
|----------|------|-----------------------------|
| BioClinicalBERT encoder | Encodes clinical language into contextual embeddings | Slightly adjusted to better represent task-specific patterns |
| Classification head | Maps embeddings → binary output | Fully learned from scratch |
| Overall model | End-to-end prediction system | Learns decision boundary for valid vs invalid entities |

This results in the following architecture:

> **Input → BioClinicalBERT encoder → classification head → binary prediction (valid / invalid)**

---

#### 2.3 Rationale

The focus here is not on choosing fine-tuning (covered previously), but on why it is effective for this specific step:

- The pretrained model already captures clinical language structure and semantics
- The task is decision-based (classification) rather than language modelling
- Fine-tuning allows the model to learn task-specific decision boundaries without relearning language

This is necessary because:

- The annotated dataset is too small to train a model from scratch  
- The task is specialised, requiring adaptation to patterns of entity validity  
- Rule-based methods lack the ability to capture context-dependent correctness

Fine-tuning enables the model to:

- Leverage existing clinical knowledge
- Adapt efficiently to a small labelled dataset
- Learn contextual validation patterns beyond rule-based logic

This makes it the most appropriate training paradigm for introducing a learned validation layer into the pipeline.

---

### 3. Problem Formulation

This task is formulated as a supervised binary classification problem, where the model predicts whether an extracted entity is valid given its sentence context and associated metadata.

#### 3.1 Input format Decision 

The key design decision is how to represent input to a single transformer model. There are two possible approaches:

| Option | Description | Pros | Cons |
|-------|------------|------|------|
| A: Sentence-only | Model receives only `sentence_text` | Simple, standard, easy to implement | Model must infer entity and task implicitly |
| B: Structured input (chosen) | Model receives entity + task + context explicitly | Clear signal, better disambiguation, improved precision | Slightly more implementation effort |

Structured input is necessary because the task is not generic NLP, but entity-level validation within a sentence. Without explicit structure, the model must infer:

- Which part of the sentence is relevant  
- Which entity is being evaluated  
- What task context applies  

This introduces unnecessary ambiguity and reduces performance.

---

#### 3.2 Input (X)

Each example is converted into a single structured text string, combining all relevant fields.

| Column | Role | Included in Input |
|--------|------|------------------|
| `sentence_text` | Core context | Yes |
| `entity_text` | Target entity | Yes |
| `entity_type` | Entity category | Yes |
| `task` | Task context | Yes |
| `concept` | Mapping of entity to concept | Yes|

Excluded fields:

| Column | Reason for Exclusion |
|--------|--------------------|
| `is_valid` | Target variable (label leakage if included) |
| `negated`, `confidence` | Derived features → risk of leakage |
| `section` | Not essential for validation task |

---

#### 3.3 Output (y)

The target variable is:

- `is_valid` → binary label

Encoding:

| Label | Value |
|------|------|
| Valid | 1 |
| Invalid | 0 |

The model produces:
- Logits (raw scores for each class)
- Probabilities via softmax
- Final prediction via argmax

---

### 4. Tokenisation and Input Formatting

#### 4.1 Purpose

Transformer models cannot process raw text or structured tabular data directly.  
All inputs must be converted into a numerical sequence representation.

In this pipeline:
- Structured fields (entity, task, sentence) are first combined into a single text string
- This string is then tokenised into a format the model can process

Tokenisation therefore serves to:
- Convert text into numerical inputs (token IDs)
- Standardise input length for batch processing
- Preserve contextual information for the transformer model

---

#### 4.2 Input Formatting (Pre-Tokenisation)

Before tokenisation, all relevant fields are concatenated into a structured string:

```python
[ENTITY TYPE] {entity_type}
[ENTITY] {entity_text}
[CONCEPT] {concept}
[TASK] {task}
[TEXT] {sentence_text}
```

This design ensures:
- Explicit representation of the entity being evaluated  
- Clear encoding of task context  
- Preservation of full sentence information  

Key properties:

| Property | Purpose |
|---------|--------|
| Special tags (e.g. `[ENTITY]`) | Provide structure within a flat text sequence |
| Single concatenated string | Required for transformer input |
| Consistent format | Ensures alignment between training and inference |
| Input format must remain identical during training and inference | Prevents discrepancies that could degrade performance |
| No label-derived features should be included | Prevents data leakage and ensures model learns from valid patterns |

---

#### 4.3 Tokenisation Process

Tokenisation is performed using the BioClinicalBERT tokenizer (WordPiece-based):

1. **Token splitting**
  - Text is split into tokens at the subword level (WordPiece tokenisation)
  - Handles rare and clinical terms effectively
  - Example: “nausea” → [“nau”, “##sea”]

2. **Tokens are mapped to IDs**
  - Each token corresponds to an index in the model vocabulary
  - This creates a sequence of integers representing the input text
  - Example: [“nau”, “##sea”] → [1234, 5678]

3. **Attention masks are created**
  - Binary mask indicating which tokens are real (1) vs padding (1)
  - This distinguishes real tokens from padding
  - Example: [1234, 5678] → attention mask [1, 1, 0, 0, 0] (if `MAX_LENGTH`=5)

4. **Sequences are standardised with padding/truncation**
  - Tranformer models require fixed-length input sequences
  - Sequences are truncated if too long or padded if too short to a fixed length (`MAX_LENGTH`)
  - Example: [1234, 5678] → padded to [1234, 5678, 0, 0, 0] (if `MAX_LENGTH`=5)

---

#### 4.4 Output of Tokenisation

Each input example is transformed into:

| Component | Description |
|----------|------------|
| `input_ids` | Sequence of token IDs |
| `attention_mask` | Binary mask (1 = real token, 0 = padding) |

These are the actual inputs passed to the transformer model.

---

#### 4.5 Design Rationale

This approach is required because:

- Transformers operate on sequential token inputs, not structured columns  
- Clinical text contains complex terminology, requiring subword tokenisation  
- The task depends on relationships between entity, task, and context, which must be encoded in a single sequence  

By combining structured fields into text and then tokenising:

- The model can learn contextual dependencies across all inputs
- Ambiguity is reduced when multiple entities exist in a sentence
- The input remains compatible with pretrained transformer architecture

---

### 5. Model Architecture

#### 5.1 Overview

This step uses BioClinicalBERT as the base model, selected previously for its domain-specific pretraining on clinical text. Key properties include:

- End-to-end differentiable model  
- Sequence-level classification using `[CLS]` representation  
- Fully compatible with the Hugging Face training pipeline  

This section defines the model structure only, training strategy (hyperparameters, optimisation, validation) is handled in subsequent sections.

---

#### 5.2 Architecture Components

The model is instantiated using:

```python
AutoModelForSequenceClassification.from_pretrained(
    "emilyalsentzer/Bio_ClinicalBERT",
    num_labels=2
)
```

This class constructs a complete classification model consisting of:

| Component | Role |
|----------|------|
| BioClinicalBERT encoder | Generates contextual embeddings from input text |
| `[CLS]` token representation | Aggregates sequence-level information |
| Classification head (linear layer) | Maps embedding → binary logits |

---

#### 5.3 Forward Pass

For each input example, the model performs:

1. Tokenised input (`input_ids`, `attention_mask`) is passed to the encoder  
2. The encoder produces contextual embeddings for all tokens  
3. The `[CLS]` token embedding is extracted as a representation of the full sequence  
4. The classification head maps this embedding to logits (2 classes)  
5. Softmax converts logits into probabilities  

---

#### 5.4 Training Behaviour (Fine-Tuning)

The model is trained via fine-tuning, where pretrained representations are adapted to a task-specific classification objective.

- The BioClinicalBERT encoder is initialised with pretrained weights from clinical text corpora.  
- The classification head is newly initialised and trained from scratch for binary classification.  
- Pretraining-specific heads (e.g. masked language modelling, next sentence prediction) are not used.  

During training:

| Component | Learning Behaviour |
|----------|------------------|
| BioClinicalBERT encoder | Fine-tuned (weights updated with small task-specific adjustments) |
| Classification head | Fully trained from random initialisation |

This setup allows the model to:

- Retain general clinical language understanding  
- Learn task-specific decision boundaries for entity validation  

---

#### 5.5 Learned Function
The model learns a decision function:

> f(entity + context) → {valid, invalid}

Specifically, it captures:

- Contextual relationships between entity and sentence
- Task-dependent interpretation of entities
- Patterns distinguishing valid vs invalid extractions

---

### 6. Training Configuration and Hyperparameters

#### 6.1 Overview

This section defines the configuration and hyperparameters used to train the model.

These settings control:
- How training is executed (e.g. batching, evaluation, checkpointing)
- How model weights are updated (e.g. learning rate, epochs)
- How training stability and performance are managed

Training is implemented using the Hugging Face `Trainer` API, which standardises the training loop and integrates configuration, optimisation, and evaluation into a single framework.

---

#### 6.2 Training Framework

We use Hugging Face's `Trainer` API to simplify the training loop. 

```python
from transformers import Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
```

The `Trainer` API handles:

- Data loading and batching
- Device placement (CPU/GPU)
- Forward and backward passes  
- Loss computation (`CrossEntropyLoss` for classification)
- Gradient accumulation and clipping (`max_grad_norm`)  
- Optimizer and learning rate scheduler updates 
- Periodic evaluation at defined intervals (`eval_strategy="epoch"`)  
- Checkpointing and best model selection  
- Metric computation via `compute_metrics` 

Rationale:

-	Eliminates manual PyTorch training loop implementation
-	Reduces risk of implementation errors
-	Ensures consistent handling of training, evaluation, and logging
-	Integrates directly with Hugging Face datasets and tokenizers

---

#### 6.3 Training Configuration

Training configuration defines the execution behaviour of training, separate from hyperparameters. These settings control evaluation, checkpointing, logging, and how training progress is managed.

| Configuration Setting            | Value        | Purpose / Rationale                                                                 |
|----------------------------------|-------------|-----------------------------------------------------------------------------------|
| Evaluation strategy              | "epoch"     | Runs validation at the end of each epoch to track generalisation performance.     |
| Save strategy                    | "epoch"     | Saves a checkpoint at the end of each epoch for model versioning and recovery.    |
| Load best model at end           | True        | Ensures the final model is the best-performing checkpoint (not just last epoch).  |
| Metric for best model            | "f1"        | Selects best checkpoint based on F1-score, balancing precision and recall.        |
| Logging steps                    | 10          | Logs training metrics every 10 steps for monitoring convergence and stability.    |
| Save total limit                 | 2           | Retains only the 2 most recent checkpoints to control storage usage.              |
| Output directory                 | models/bioclinicalbert | Directory where model checkpoints and final model are stored.          |

**Explanation of workflow effects:**

- **Evaluation + checkpointing:** Work together to track performance over time and ensure the best model is preserved rather than simply the final model.
- **`load_best_model_at_end=True`:** Ensures that training does not depend on the final epoch, which may overfit. Instead, the model with the best validation F1 is used.
- **Logging (`logging_steps=10`):** Provides visibility into training dynamics (loss trends, instability). Without this, failures (e.g. divergence) are harder to detect.

---

#### 6.4 Key Hyperparameters

Hyperparameters directly control the weight update dynamics during training. These influence how quickly and accurately the model learns.  

| Hyperparameter              | Value  | Purpose / Rationale |
|-----------------------------|--------|---------------------|
| Batch size                  | 8      | Number of examples per forward/backward pass. Smaller batch improves stability on small datasets and fits memory constraints. |
| Gradient accumulation steps | 2      | Accumulates gradients across steps to simulate larger batches without increasing memory usage. |
| Epochs                      | 5      | Number of full dataset passes. Ensures sufficient learning signal from a small dataset. |
| Learning rate               | 3e-6   | Small learning rate prevents overwriting pretrained representations during fine-tuning. |
| Weight decay                | 0.05   | L2 regularisation to reduce overfitting by penalising large weights. |
| Warmup ratio                | 0.1    | Gradually increases learning rate for first 10% of steps to stabilise early training. |
| Learning rate scheduler     | linear | Decays learning rate over time to enable smooth convergence. |
| Max grad norm               | 1.0    | Clips gradients to prevent exploding updates and ensure numerical stability. |
| Max sequence length         | 512    | Standardises input length via truncation/padding for transformer compatibility. |

**Effective batch size:**

```text
effective_batch_size = batch_size × gradient_accumulation_steps = 8 × 2 = 16
```

- With `batch_size=8` and `gradient_accumulation_steps=2`, the effective batch size is 16 (8 examples processed before weights are updated). 
- This allows for more stable updates while fitting within memory constraints.

**Explanation of choices:**

-	Small batch size and low learning rate are standard for fine-tuning transformers on limited data to avoid instability.
-	Gradient accumulation improves gradient quality without exceeding hardware limits.
-	Warmup and scheduler stabilise early training and refine convergence later.
-	Weight decay and gradient clipping reduce overfitting and prevent unstable updates.
-	Multiple epochs ensure the model sufficiently learns dataset patterns, with validation controlling overfitting.

---

### 7. Training Loop

#### 7.1 High-Level Overview

Although Trainer abstracts implementation, the underlying training loop follows a standard sequence.

```text
for epoch:
    for batch:
        forward pass → logits
        compute loss
        backward pass (accumulate gradients)
        
        if accumulation step reached or last batch:
            clip gradients
            optimizer step (update weights)
            scheduler step (update LR)
            zero gradients
        
        log metrics (every logging_steps)
    
    evaluate on validation set
    compute metrics (via compute_metrics on validation predictions)
    save checkpoint
    update best model (based on F1)
```

---

#### 7.2 Batch-Level Training (Per Step)

For each batch:

1. Input tensors (`input_ids`, `attention_mask`) are loaded and moved to device.
2. **Forward pass:**
	-	Inputs passed through BioClinicalBERT → contextual embeddings
	-	[CLS] representation passed to classification head → logits (2 values)
3. **Loss computation:**
  - `CrossEntropyLoss` applied to logits and ground-truth labels
4. **Backward pass:**
  - Gradients computed via backpropagation
5. **Gradient accumulation:**
  - Gradients accumulated over `gradient_accumulation_steps`
  - Weights updated only after accumulation threshold reached
  - Delays weight updates to simulate larger batch sizes
6. **Weight update (conditional):**
  Performed when accumulation threshold is reached or at final batch:
  - **Gradient clipping:** Gradients clipped to `max_grad_norm=1.0`
  - **Optimizer step:** Model weights updated
  - **Scheduler step:** Learning rate adjusted according to linear schedule
7. **Logging:**
	-	Training metrics recorded every `logging_steps`
  - Used for monitoring convergence, not evaluation

---

#### 7.3 Epoch-Level Operations (per epoch)

After all batches in an epoch:

1. **Validation:**
  - Model evaluated on validation dataset
  - Predictions generated (logits → argmax → class labels)
  - Metrics computed: accuracy, precision, recall, F1-score
2. **Checkpointing:**
	-	Model state saved (`save_strategy="epoch"`)
3. **Best model selection:**
	-	Current model checkpoint compared using F1-score
	-	Best-performing model checkpoint retained

---

#### 7.4 Training Duration

Total steps:

```text
steps_per_epoch = ceil(training_size / batch_size) = ceil(420 / 8) ≈ 53
total_steps = steps_per_epoch × epochs = 53 × 5 ≈ 265
```

- Each step processes one batch of `batch_size` examples
- One epoch = full pass over the dataset
- Total steps define the number of forward and backward passes

Weight updates:

```text
updates_per_epoch ≈ ceil(steps_per_epoch / gradient_accumulation_steps)
                 ≈ ceil(53 / 2) ≈ 27
```

- Gradients are accumulated across steps
- Weight updates occur only after `gradient_accumulation_steps` or at the final batch
- Therefore, the number of weight updates is lower than total steps

End of training:

- `load_best_model_at_end=True` restores the checkpoint with the highest validation F1-score
-	The final model is selected based on performance, not the last epoch

---

### 8. Validation Strategy (During Training)

#### 8.1 Purpose

Validation monitors the model's ability to generalise to unseen data during training. Unlike final evaluation, validation: 

- Directly informs training decisions by detecting whether learning is proceeding correctly.
- Enables early stopping or adjustments if necessary.
- Ensures that the baseline configuration is effective, stable, and sufficient without requiring unnecessary iterations.

Specifically, validation allows you to identify:

- Healthy learning → the model is improving on unseen data  
- Overfitting → the model memorises training data and fails to generalise  
- Underfitting → the model is unable to capture meaningful patterns  

It provides a decision-making signal during training on whether the model is learning meaningful patterns or not

---

#### 8.2 Validation Process

Validation is performed automatically at the end of each epoch (`eval_strategy="epoch"`) using the `Training` framework. During validation:

1. The model performs a forward pass on the validation dataset (no gradients are computed in eval mode).  
2. Logits are generated for each example.  
3. Predictions are obtained via `argmax(logits)` → predicted class (0 or 1)  
4. Metrics are computed using `compute_metrics`:
   - **Accuracy:** proportion of correct predictions  
   - **Precision:** proportion of predicted positives that are correct  
   - **Recall:** proportion of true positives correctly predicted  
   - **F1-score:** harmonic mean of precision and recall 

Key properties:

- Validation data is never used for training
- Metrics reflect generalisation performance, not training performance

---

#### 8.3 Interpretation of Validation Behaviour

Validation is interpreted by trends across epochs, not single values.

| Pattern | Interpretation | Action |
|--------|---------------|--------|
| Training loss ↓, Validation loss ↓, metrics ↑ then plateau | Healthy learning | Continue training or stop when stable |
| Training loss ↓, Validation loss ↑ | Overfitting | Apply regularisation or stop earlier |
| Both training and validation performance low | Underfitting | Improve input representation or increase model capacity |

Additional signals:

| Signal | Interpretation |
|--------|----------------|
| Metrics ≈ random (~50%) | Model not learning meaningful patterns |
| Highly unstable metrics | Training instability or insufficient data |
| Large gap between training and validation | Poor generalisation |

Notes:

- No fixed numeric threshold defines “good” performance  
- Focus is on consistency, stability, and learning trends 

---

#### 8.4 Role in Model Selection

Validation determines which model checkpoint is ultimately used:

- `metric_for_best_model = "f1"`  
- `load_best_model_at_end = True`

Process:

1. After each epoch, validation F1-score is computed.  
2. The best-performing checkpoint is tracked.  
3. At the end of training, the model with the highest validation F1-score is reloaded.  

Implications:

- The final model is not necessarily from the last epoch  
- Selection is based on generalisation performance, not training metrics

---

### 9. Cross-Validation (Robustness Assessment)

#### 9.1 Purpose and Role

Cross-validation is used to assess the robustness and reliability of model performance with respect to data splitting. Unlike validation during training (Section 8), which guides learning: 

- Cross-validation is an evaluation procedure performed prior to final model training, used to estimate generalisation performance across different data splits.
- Answers whether performance is stable across different subsets of the data, rather than being an artifact of a specific train/validation split.
  -	Confirms reliability of results
  -	Quantifies uncertainty in performance

This is critical because:

- The dataset is relatively small (~420 training samples)
- Model performance can vary depending on which samples are seen during training
- A single validation split may give an overly optimistic or pessimistic estimate

---

#### 9.2 Role in Pipeline

The pipeline is structured into two distinct phases:

- **Phase 1:** Cross-validation (robustness assessment)
- **Phase 2:** Final model training (deployment model)

This ensures strict separation between evaluation and training.

- Performed before final training to assess robustness and generalisation stability  
  - Models trained during cross-validation are discarded after evaluation  
  - No weights or parameters are carried over into final training  
- Not used in this pipeline for training decisions (e.g., hyperparameter tuning or checkpoint selection)  
- Does not produce the final deployed model (final training uses the predefined train/validation split)

Cross-validation is used to verify that performance is stable across data splits, rather than to produce the deployed model.

---

#### 9.3 Method: Stratified 5-Fold Cross-Validation

We use Stratified K-Fold Cross-Validation using `StratifiedKFold` with the following configuration:

- `K = 5` folds as a trade-off between:
  - Reliability (more folds = better estimate)
  - Computational cost (more folds = more training runs)
- Stratification based on the binary label (`is_valid`) to ensure:
  - Each fold maintains a similar class distribution, preventing biased evaluation

Dataset used for CV:

- Entire training dataset: 420 samples
- The original 90-sample validation set is not used in CV as it is reserved for:
  - Standard validation during training
  - Ensuring seperation between training workflow and robustness assessment

Per fold split:

- Training: ~80% → ~336 samples
- Validation: ~20% → ~84 samples

| Fold | Training Samples | Validation Samples |
|------|-----------------|------------------|
| 1    | ~336            | ~84              |
| 2    | ~336            | ~84              |
| 3    | ~336            | ~84              |
| 4    | ~336            | ~84              |
| 5    | ~336            | ~84              |

Each sample appears:
- In validation exactly once (1/5)
- In training four times (4/5)

This produces 5 independent training runs, each evaluated on a different subset of the data.

---

#### 9.4 Implementation Logic

For each fold the following steps are performed:

```text
for fold in K:
    split data into train_fold (80%) and val_fold (20%) using stratification

    initialise a new model from the pretrained checkpoint (no weight sharing between folds)

    train model on train_fold
    evaluate model on val_fold

    store metrics
```

Key implementation details:

- **Model reset per fold:** A fresh model is initialised from the pretrained checkpoint for each fold to ensure independence of training runs
- **Identical training configuration:** All folds use the same hyperparameters and training setup
- **Independent runs:** Each fold is a fully independent training + validation cycle
- **Same metric computation:** Accuracy, precision, recall, and F1-score are computed using the same `compute_metrics` function

---

#### 9.5 Metrics Aggregation

After all folds are completed:

- Metrics are aggregated across folds
- For each metric, we compute:
	-	Mean → expected performance
	-	Standard deviation → variability across splits

Interpretation:

- **Mean performance** → Represents the average expected generalisation performance
- **Standard deviation** → Measures sensitivity to data splitting
	- Low std = stable model
	- High std = high variance / instability

No interpretation of results is performed in this section; analysis is deferred to Section 10.

---

### 10. Iterative Training and Empirical Analysis

#### 10.1 Overview

This section documents the iterative training process, including configuration changes, observed behaviour, and resulting performance. The objective is to:

- Identify a stable and effective training configuration  
- Maximise F1-score (primary metric)  
- Determine whether performance is limited by model design or dataset characteristics  

Each phase represents a controlled change to either input representation or training configuration, followed by empirical evaluation.

---

#### 10.2 First Training Run: Pipeline Sanity Check

**Input:**  
- `sentence_text` only  

**Purpose:**
- Verify end-to-end pipeline functionality (data → tokenisation → model → metrics)  
- Establish a baseline for debugging  

**Results:**

| Metric | Value |
|--------|-------|
| Accuracy | 0.511 |
| F1 | 0.676 |
| Precision | 0.511 |
| Recall | 1.0 |
| Loss | ~0.717 |

**Interpretation:**
- Model predicts all samples as positive (recall = 1.0)  
- No discriminative ability (precision = 0.511)
- F1 artificially inflated due to class imbalance  

**Conclusion:**  
- Pipeline is functional, but input is insufficient for learning. 
- This configuration is not valid for the task.

---

#### 10.3 Second Training Run: Full Input Representation

**Input:**  
- `task + concept + entity_type + entity_text + sentence_text`

**Rationale:**
- Provide maximum contextual and structured information  
- Expected to improve performance by reducing ambiguity and providing clear signals for classification

**Results:**

| Metric | Value |
|--------|-------|
| Accuracy | 0.511 |
| F1 | 0.676 |
| Precision | 0.511 |
| Recall | 1.0 |
| Loss | ~0.692 |

**Observed Issues:**
- Exploding gradients (`loss = NaN`)
- Training instability (loss diverging, metrics fluctuating wildly)
- Failure to converge (metrics do not improve, loss does not decrease)

**Interpretation:**
- Input complexity increased gradient instability  
- Default learning rate too high for this input complexity  

**Conclusion:**  
- Model cannot train reliably with default hyperparameters on full input without stabilisation.

---

#### 10.4 Third Training Run: Hyperparameter Stabilisation (Best Configuration)

**Changes and Rationale:**
1. **Learning rate:** `2e-5 → 5e-6`  
  - Lower LR = smaller weight updates, more stable convergence
  - Prevents overshooting minima
  - Critical for stability on small dataset
2. **Batch size:** `16 → 8`  
  - Smaller batches → less noisy updates
  - More stable gradients for small datasets
3. **Gradient clipping:** `max_grad_norm = 1.0`
  - Capping gradient magnitude prevents exploding gradients
  - Ensures numerical stability during training  

**Results:**

| Metric | Value |
|--------|-------|
| Accuracy | 0.733 |
| F1 | 0.75 |
| Precision | 0.72 |
| Recall | 0.783 |
| Loss | 0.645–0.658 |

**Interpretation:**
- Stable convergence achieved (loss decreases, metrics improve consistently)
- Balanced precision–recall trade-off  
- Meaningful pattern learning  

**Conclusion:**  
- This is the best-performing configuration and establishes the baseline.
- However, further tuning is explored as metrics in the mid 70's still leaves room for improvement.

---

#### 10.5 Fourth Training Run: Simplified Input

**Changes:**
- Removed `entity_type` and `concept`  
- Input: `task + entity_text + sentence_text`

**Rationale:**
- Reduce redundancy and noise  
- Aligns better with natural language structure
- Simpler input may lead to better generalisation

**Results:**

| Metric | Range |
|--------|-------|
| Accuracy | 0.644–0.667 |
| F1 | 0.686–0.714 |
| Precision | 0.606–0.654 |
| Recall | 0.739–0.870 |

**Interpretation:**
- Slight performance drop across all metrics
- Precision dropped → reduced discriminative ability  

**Conclusion:**  
- Structured features contribute meaningful signal which is lost in simplification. 
- Simplification reduces performance.
- Relevant features should be retained for best performance, even if they introduce some complexity.

---

#### 10.6 Fifth Training Run: Advanced Tuning and Partial Freezing

**Changes and Rationale:**
1. **Epochs:** `3 → 5` 
  - More epochs allow for more learning signal, especially with a small dataset. 
  - Prevents underfitting by giving the model more time to learn patterns.
  - Risks overfitting on a small dataset, but this is mitigated by other regularisation techniques.
2. **Learning rate:** `5e-6 → 3e-6`  
  - Even smaller learning rate allows for finer adjustments to weights, which can improve performance on a small dataset.
  - Slower but more precise convergence, potentially leading to better minima.
3. **Weight decay:** `0.01 → 0.05`  
  - Stronger L2 regularisation to combat overfitting, especially with more epochs.
  - Penalises large weights more heavily, encouraging simpler models that generalise better.
4. **Gradient accumulation:** `2` 
  - Stimulates larger effective batch size (16)
  - More stable updates without increasing memory usage. 
5. **Warmup ratio:** `0.1` with linear scheduler 
  - Gradually increases learning rate at the start of training to stabilise early updates.
  - Prevents divergence in the initial phase when weights are most sensitive. 
6. **Partial BERT freezing**  
  - Freeze lower layers of BioClinicalBERT to retain general language understanding.
  - Only fine-tune higher layers and classification head to adapt to the specific task.
  - Reduces risk of catastrophic forgetting of pretrained knowledge while still allowing task adaptation.

**Overall Rationale:**
- Improve generalisation and stability  
- Reduce overfitting risk  

**Results:**

| Metric | Range |
|--------|-------|
| Accuracy | 0.577–0.600 |
| F1 | 0.672–0.705 |
| Precision | 0.557–0.566 |
| Recall | 0.848–0.957 |

**Interpretation:**
- Recall increased significantly  
- Precision decreased → more false positives  
- Overall F1 worse than baseline  

**Conclusion:**  
- Additional complexity degraded performance → dataset too small for complex tuning to be effective.
- Freezing may have limited the model's learning capacity for this specific task, preventing it from adapting to the nuances of entity validation.
- Model became under-adaptive and over-regularised.

---

#### 10.7 Sixth Training Run: Stratified 5-Fold Cross-Validation

**Changes and Rationale:**
1. **Removed freezing:**
  - All BERT layers are trainable again
  - Allows full adaption to the task
2. **Retained advanced hyperparameters:**
  - To see if the tuning can improve performance when applied across multiple data splits 
  - Maintains consistency with the previous run for comparability
3. **Applied 5-fold stratified cross-validation:**
  - Assess performance across different train/validation splits
  - Provides a more robust estimate of generalisation performance
  - Quantifies variability and stability of results

**CV Results:**

Fold
Accuracy
F1
Precision
Recall
1
0.6429
0.7000
0.6034
0.8333
2
0.7381
0.7660
0.7059
0.8372
3
0.7262
0.7579
0.6923
0.8372
4
0.5714
0.6786
0.5507
0.8837
5
0.7262
0.7294
0.7381
0.7209


Metric
Mean
Std
Accuracy
0.6810
0.0721
F1
0.7264
0.0373
Precision
0.6581
0.0781
Recall
0.8225
0.0604


**Final Results:**

| Metric | Range |
|--------|-------|
| Accuracy |  |
| F1 |  |
| Precision |  |
| Recall |  |
| Loss |  |

**Observations:**

	•	F1 (primary metric):
	•	Mean = 0.726 → expected generalisation performance
	•	Std = 0.037 → low variability → stable model
	•	Recall > Precision pattern:
	•	Recall = 0.82, Precision = 0.66
	•	Model is biased toward sensitivity (detects positives well, but more false positives)
	•	Variance across folds:
	•	Accuracy std ≈ 0.07 → moderate variability (expected given small dataset ~420)
	•	Worst fold (Fold 4):
	•	Lower accuracy (0.57) but still reasonable F1 (0.68)
	•	Suggests data split sensitivity but not catastrophic instability

comparison to baseline (third training run):

**Interpretation:**

1. **Performance instability**  
  - Metrics vary across folds  
  - Model sensitive to training data selection 
  - Performance depends heavily on data splits, indicating overfitting 
2. **No benefit from additional tuning**  
  - Advanced configuration does not outperform simpler baseline  
  - Dataset size limits the effectiveness of tuning and regularisation
  - Leads to increased variance without improving mean performance
3. **Generalisation uncertainty**  
  - Variance indicates unreliable performance on unseen data  

---

Here is my full final run for the trained transformer. This will be the final one before we retrain on the 1200 - we will need to update our notes but first please analyse and give me insights:

Train size: 420
Validation size: 90
/Users/simonyip/Hybrid-Clinical-Notes-Extraction-Pipeline/venv/lib/python3.11/site-packages/huggingface_hub/file_download.py:949: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Map: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:00<00:00, 5861.61 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 90/90 [00:00<00:00, 7021.98 examples/s]
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at emilyalsentzer/Bio_ClinicalBERT and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

===== RUNNING CROSS-VALIDATION =====

=== Fold 1 / 5 ===
Map: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 336/336 [00:00<00:00, 7834.68 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 84/84 [00:00<00:00, 6903.26 examples/s]
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at emilyalsentzer/Bio_ClinicalBERT and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
{'loss': 0.707, 'grad_norm': 6.0354390144348145, 'learning_rate': 2.7272727272727272e-06, 'epoch': 0.48}                                                                     
{'loss': 0.6945, 'grad_norm': 4.542715072631836, 'learning_rate': 2.7127659574468088e-06, 'epoch': 0.95}                                                                     
{'eval_loss': 0.6863413453102112, 'eval_accuracy': 0.5, 'eval_f1': 0.6440677966101694, 'eval_precision': 0.5, 'eval_recall': 0.9047619047619048, 'eval_runtime': 3.7859, 'eval_samples_per_second': 22.187, 'eval_steps_per_second': 2.906, 'epoch': 1.0}                                                                                                 
{'loss': 0.6811, 'grad_norm': 4.770979881286621, 'learning_rate': 2.3936170212765957e-06, 'epoch': 1.43}                                                                     
{'loss': 0.6854, 'grad_norm': 4.267581462860107, 'learning_rate': 2.074468085106383e-06, 'epoch': 1.9}                                                                       
{'eval_loss': 0.6778864860534668, 'eval_accuracy': 0.5357142857142857, 'eval_f1': 0.6422018348623854, 'eval_precision': 0.5223880597014925, 'eval_recall': 0.8333333333333334, 'eval_runtime': 3.6022, 'eval_samples_per_second': 23.319, 'eval_steps_per_second': 3.054, 'epoch': 2.0}                                                                   
{'loss': 0.6607, 'grad_norm': 3.3178534507751465, 'learning_rate': 1.7553191489361702e-06, 'epoch': 2.38}                                                                    
{'loss': 0.6752, 'grad_norm': 7.511821746826172, 'learning_rate': 1.4361702127659576e-06, 'epoch': 2.86}                                                                     
{'eval_loss': 0.6651360988616943, 'eval_accuracy': 0.6071428571428571, 'eval_f1': 0.6796116504854369, 'eval_precision': 0.5737704918032787, 'eval_recall': 0.8333333333333334, 'eval_runtime': 3.6247, 'eval_samples_per_second': 23.174, 'eval_steps_per_second': 3.035, 'epoch': 3.0}                                                                   
{'loss': 0.6426, 'grad_norm': 5.404907703399658, 'learning_rate': 1.1170212765957447e-06, 'epoch': 3.33}                                                                     
{'loss': 0.6547, 'grad_norm': 8.693889617919922, 'learning_rate': 7.978723404255319e-07, 'epoch': 3.81}                                                                      
{'eval_loss': 0.6610027551651001, 'eval_accuracy': 0.6190476190476191, 'eval_f1': 0.6862745098039216, 'eval_precision': 0.5833333333333334, 'eval_recall': 0.8333333333333334, 'eval_runtime': 3.5619, 'eval_samples_per_second': 23.583, 'eval_steps_per_second': 3.088, 'epoch': 4.0}                                                                   
{'loss': 0.6655, 'grad_norm': 4.740090847015381, 'learning_rate': 4.787234042553192e-07, 'epoch': 4.29}                                                                      
{'loss': 0.6419, 'grad_norm': 4.609196186065674, 'learning_rate': 1.5957446808510638e-07, 'epoch': 4.76}                                                                     
{'eval_loss': 0.6593428254127502, 'eval_accuracy': 0.6428571428571429, 'eval_f1': 0.7, 'eval_precision': 0.603448275862069, 'eval_recall': 0.8333333333333334, 'eval_runtime': 3.5703, 'eval_samples_per_second': 23.528, 'eval_steps_per_second': 3.081, 'epoch': 5.0}                                                                                   
{'train_runtime': 272.9167, 'train_samples_per_second': 6.156, 'train_steps_per_second': 0.385, 'train_loss': 0.668915353502546, 'epoch': 5.0}                               
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 105/105 [04:32<00:00,  2.60s/it]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:03<00:00,  3.42it/s]
Fold 1 metrics: {'eval_loss': 0.6593428254127502, 'eval_accuracy': 0.6428571428571429, 'eval_f1': 0.7, 'eval_precision': 0.603448275862069, 'eval_recall': 0.8333333333333334, 'eval_runtime': 3.5713, 'eval_samples_per_second': 23.521, 'eval_steps_per_second': 3.08, 'epoch': 5.0}

=== Fold 2 / 5 ===
Map: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 336/336 [00:00<00:00, 4473.98 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 84/84 [00:00<00:00, 6359.25 examples/s]
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at emilyalsentzer/Bio_ClinicalBERT and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
{'loss': 0.6986, 'grad_norm': 3.763268232345581, 'learning_rate': 2.7272727272727272e-06, 'epoch': 0.48}                                                                     
{'loss': 0.6831, 'grad_norm': 2.286144495010376, 'learning_rate': 2.7127659574468088e-06, 'epoch': 0.95}                                                                     
{'eval_loss': 0.67332524061203, 'eval_accuracy': 0.5833333333333334, 'eval_f1': 0.7008547008547008, 'eval_precision': 0.5540540540540541, 'eval_recall': 0.9534883720930233, 'eval_runtime': 3.5849, 'eval_samples_per_second': 23.432, 'eval_steps_per_second': 3.068, 'epoch': 1.0}                                                                     
{'loss': 0.6699, 'grad_norm': 4.910408973693848, 'learning_rate': 2.3936170212765957e-06, 'epoch': 1.43}                                                                     
{'loss': 0.6738, 'grad_norm': 2.728898525238037, 'learning_rate': 2.074468085106383e-06, 'epoch': 1.9}                                                                       
{'eval_loss': 0.6578093767166138, 'eval_accuracy': 0.7142857142857143, 'eval_f1': 0.7551020408163265, 'eval_precision': 0.6727272727272727, 'eval_recall': 0.8604651162790697, 'eval_runtime': 3.5678, 'eval_samples_per_second': 23.544, 'eval_steps_per_second': 3.083, 'epoch': 2.0}                                                                   
{'loss': 0.6642, 'grad_norm': 4.079012393951416, 'learning_rate': 1.7553191489361702e-06, 'epoch': 2.38}                                                                     
{'loss': 0.6526, 'grad_norm': 7.0983099937438965, 'learning_rate': 1.4361702127659576e-06, 'epoch': 2.86}                                                                    
{'eval_loss': 0.6490338444709778, 'eval_accuracy': 0.7380952380952381, 'eval_f1': 0.7659574468085106, 'eval_precision': 0.7058823529411765, 'eval_recall': 0.8372093023255814, 'eval_runtime': 3.5719, 'eval_samples_per_second': 23.517, 'eval_steps_per_second': 3.08, 'epoch': 3.0}                                                                    
{'loss': 0.6487, 'grad_norm': 5.792099952697754, 'learning_rate': 1.1170212765957447e-06, 'epoch': 3.33}                                                                     
{'loss': 0.6522, 'grad_norm': 4.63058614730835, 'learning_rate': 7.978723404255319e-07, 'epoch': 3.81}                                                                       
{'eval_loss': 0.6431766152381897, 'eval_accuracy': 0.7261904761904762, 'eval_f1': 0.7578947368421053, 'eval_precision': 0.6923076923076923, 'eval_recall': 0.8372093023255814, 'eval_runtime': 3.5755, 'eval_samples_per_second': 23.493, 'eval_steps_per_second': 3.076, 'epoch': 4.0}                                                                   
{'loss': 0.6604, 'grad_norm': 3.549254894256592, 'learning_rate': 4.787234042553192e-07, 'epoch': 4.29}                                                                      
{'loss': 0.6245, 'grad_norm': 5.618312358856201, 'learning_rate': 1.5957446808510638e-07, 'epoch': 4.76}                                                                     
{'eval_loss': 0.6409371495246887, 'eval_accuracy': 0.7261904761904762, 'eval_f1': 0.7578947368421053, 'eval_precision': 0.6923076923076923, 'eval_recall': 0.8372093023255814, 'eval_runtime': 3.5741, 'eval_samples_per_second': 23.502, 'eval_steps_per_second': 3.078, 'epoch': 5.0}                                                                   
{'train_runtime': 274.1845, 'train_samples_per_second': 6.127, 'train_steps_per_second': 0.383, 'train_loss': 0.6619382199786958, 'epoch': 5.0}                              
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 105/105 [04:34<00:00,  2.61s/it]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:03<00:00,  3.42it/s]
Fold 2 metrics: {'eval_loss': 0.6490338444709778, 'eval_accuracy': 0.7380952380952381, 'eval_f1': 0.7659574468085106, 'eval_precision': 0.7058823529411765, 'eval_recall': 0.8372093023255814, 'eval_runtime': 3.5798, 'eval_samples_per_second': 23.465, 'eval_steps_per_second': 3.073, 'epoch': 5.0}

=== Fold 3 / 5 ===
Map: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 336/336 [00:00<00:00, 4101.52 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 84/84 [00:00<00:00, 6371.90 examples/s]
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at emilyalsentzer/Bio_ClinicalBERT and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
{'loss': 0.6878, 'grad_norm': 3.5712032318115234, 'learning_rate': 2.7272727272727272e-06, 'epoch': 0.48}                                                                    
{'loss': 0.6868, 'grad_norm': 2.7712812423706055, 'learning_rate': 2.7127659574468088e-06, 'epoch': 0.95}                                                                    
{'eval_loss': 0.6730868220329285, 'eval_accuracy': 0.6428571428571429, 'eval_f1': 0.7413793103448276, 'eval_precision': 0.589041095890411, 'eval_recall': 1.0, 'eval_runtime': 3.5689, 'eval_samples_per_second': 23.537, 'eval_steps_per_second': 3.082, 'epoch': 1.0}                                                                                   
{'loss': 0.6833, 'grad_norm': 2.6880381107330322, 'learning_rate': 2.3936170212765957e-06, 'epoch': 1.43}                                                                    
{'loss': 0.6749, 'grad_norm': 2.5890159606933594, 'learning_rate': 2.074468085106383e-06, 'epoch': 1.9}                                                                      
{'eval_loss': 0.6580777168273926, 'eval_accuracy': 0.7261904761904762, 'eval_f1': 0.7578947368421053, 'eval_precision': 0.6923076923076923, 'eval_recall': 0.8372093023255814, 'eval_runtime': 3.6023, 'eval_samples_per_second': 23.318, 'eval_steps_per_second': 3.054, 'epoch': 2.0}                                                                   
{'loss': 0.662, 'grad_norm': 3.3416481018066406, 'learning_rate': 1.7553191489361702e-06, 'epoch': 2.38}                                                                     
{'loss': 0.6566, 'grad_norm': 3.86834716796875, 'learning_rate': 1.4361702127659576e-06, 'epoch': 2.86}                                                                      
{'eval_loss': 0.6467102766036987, 'eval_accuracy': 0.7023809523809523, 'eval_f1': 0.7191011235955056, 'eval_precision': 0.6956521739130435, 'eval_recall': 0.7441860465116279, 'eval_runtime': 6.0126, 'eval_samples_per_second': 13.971, 'eval_steps_per_second': 1.829, 'epoch': 3.0}                                                                   
{'loss': 0.649, 'grad_norm': 2.3948111534118652, 'learning_rate': 1.1170212765957447e-06, 'epoch': 3.33}                                                                     
{'loss': 0.6589, 'grad_norm': 4.427390098571777, 'learning_rate': 7.978723404255319e-07, 'epoch': 3.81}                                                                      
{'eval_loss': 0.6397492289543152, 'eval_accuracy': 0.7023809523809523, 'eval_f1': 0.7191011235955056, 'eval_precision': 0.6956521739130435, 'eval_recall': 0.7441860465116279, 'eval_runtime': 3.6629, 'eval_samples_per_second': 22.933, 'eval_steps_per_second': 3.003, 'epoch': 4.0}                                                                   
{'loss': 0.6445, 'grad_norm': 3.4643948078155518, 'learning_rate': 4.787234042553192e-07, 'epoch': 4.29}                                                                     
{'loss': 0.6305, 'grad_norm': 2.284899950027466, 'learning_rate': 1.5957446808510638e-07, 'epoch': 4.76}                                                                     
{'eval_loss': 0.6379217505455017, 'eval_accuracy': 0.7142857142857143, 'eval_f1': 0.7272727272727273, 'eval_precision': 0.7111111111111111, 'eval_recall': 0.7441860465116279, 'eval_runtime': 3.5854, 'eval_samples_per_second': 23.428, 'eval_steps_per_second': 3.068, 'epoch': 5.0}                                                                   
{'train_runtime': 337.6098, 'train_samples_per_second': 4.976, 'train_steps_per_second': 0.311, 'train_loss': 0.6618761516752697, 'epoch': 5.0}                              
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 105/105 [05:37<00:00,  3.22s/it]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:03<00:00,  3.40it/s]
Fold 3 metrics: {'eval_loss': 0.6580777168273926, 'eval_accuracy': 0.7261904761904762, 'eval_f1': 0.7578947368421053, 'eval_precision': 0.6923076923076923, 'eval_recall': 0.8372093023255814, 'eval_runtime': 3.9117, 'eval_samples_per_second': 21.474, 'eval_steps_per_second': 2.812, 'epoch': 5.0}

=== Fold 4 / 5 ===
Map: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 336/336 [00:00<00:00, 3674.78 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 84/84 [00:00<00:00, 5316.78 examples/s]
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at emilyalsentzer/Bio_ClinicalBERT and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
{'loss': 0.6884, 'grad_norm': 3.1973319053649902, 'learning_rate': 2.7272727272727272e-06, 'epoch': 0.48}                                                                    
{'loss': 0.6917, 'grad_norm': 6.196396827697754, 'learning_rate': 2.7127659574468088e-06, 'epoch': 0.95}                                                                     
{'eval_loss': 0.6843015551567078, 'eval_accuracy': 0.5714285714285714, 'eval_f1': 0.6785714285714286, 'eval_precision': 0.5507246376811594, 'eval_recall': 0.8837209302325582, 'eval_runtime': 3.5706, 'eval_samples_per_second': 23.526, 'eval_steps_per_second': 3.081, 'epoch': 1.0}                                                                   
{'loss': 0.6632, 'grad_norm': 2.9160029888153076, 'learning_rate': 2.3936170212765957e-06, 'epoch': 1.43}                                                                    
{'loss': 0.6656, 'grad_norm': 4.457300662994385, 'learning_rate': 2.074468085106383e-06, 'epoch': 1.9}                                                                       
{'eval_loss': 0.6714984178543091, 'eval_accuracy': 0.5833333333333334, 'eval_f1': 0.6067415730337079, 'eval_precision': 0.5869565217391305, 'eval_recall': 0.627906976744186, 'eval_runtime': 3.586, 'eval_samples_per_second': 23.424, 'eval_steps_per_second': 3.067, 'epoch': 2.0}                                                                     
{'loss': 0.679, 'grad_norm': 2.7906129360198975, 'learning_rate': 1.7553191489361702e-06, 'epoch': 2.38}                                                                     
{'loss': 0.6546, 'grad_norm': 2.972210645675659, 'learning_rate': 1.4361702127659576e-06, 'epoch': 2.86}                                                                     
{'eval_loss': 0.6615092158317566, 'eval_accuracy': 0.6666666666666666, 'eval_f1': 0.6585365853658537, 'eval_precision': 0.6923076923076923, 'eval_recall': 0.627906976744186, 'eval_runtime': 3.5565, 'eval_samples_per_second': 23.619, 'eval_steps_per_second': 3.093, 'epoch': 3.0}                                                                    
{'loss': 0.6319, 'grad_norm': 3.6136271953582764, 'learning_rate': 1.1170212765957447e-06, 'epoch': 3.33}                                                                    
{'loss': 0.6493, 'grad_norm': 4.608612537384033, 'learning_rate': 7.978723404255319e-07, 'epoch': 3.81}                                                                      
{'eval_loss': 0.6559381484985352, 'eval_accuracy': 0.6785714285714286, 'eval_f1': 0.6746987951807228, 'eval_precision': 0.7, 'eval_recall': 0.6511627906976745, 'eval_runtime': 3.5552, 'eval_samples_per_second': 23.627, 'eval_steps_per_second': 3.094, 'epoch': 4.0}                                                                                  
{'loss': 0.6314, 'grad_norm': 4.12700080871582, 'learning_rate': 4.787234042553192e-07, 'epoch': 4.29}                                                                       
{'loss': 0.6251, 'grad_norm': 4.825655460357666, 'learning_rate': 1.5957446808510638e-07, 'epoch': 4.76}                                                                     
{'eval_loss': 0.6544142365455627, 'eval_accuracy': 0.6666666666666666, 'eval_f1': 0.6666666666666666, 'eval_precision': 0.6829268292682927, 'eval_recall': 0.6511627906976745, 'eval_runtime': 3.5717, 'eval_samples_per_second': 23.518, 'eval_steps_per_second': 3.08, 'epoch': 5.0}                                                                    
{'train_runtime': 272.7271, 'train_samples_per_second': 6.16, 'train_steps_per_second': 0.385, 'train_loss': 0.6568331582205637, 'epoch': 5.0}                               
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 105/105 [04:32<00:00,  2.60s/it]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:03<00:00,  3.44it/s]
Fold 4 metrics: {'eval_loss': 0.6843015551567078, 'eval_accuracy': 0.5714285714285714, 'eval_f1': 0.6785714285714286, 'eval_precision': 0.5507246376811594, 'eval_recall': 0.8837209302325582, 'eval_runtime': 3.5587, 'eval_samples_per_second': 23.604, 'eval_steps_per_second': 3.091, 'epoch': 5.0}

=== Fold 5 / 5 ===
Map: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 336/336 [00:00<00:00, 3456.40 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 84/84 [00:00<00:00, 7162.90 examples/s]
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at emilyalsentzer/Bio_ClinicalBERT and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
{'loss': 0.6983, 'grad_norm': 2.2787249088287354, 'learning_rate': 2.7272727272727272e-06, 'epoch': 0.48}                                                                    
{'loss': 0.6902, 'grad_norm': 3.2216548919677734, 'learning_rate': 2.7127659574468088e-06, 'epoch': 0.95}                                                                    
{'eval_loss': 0.6793776154518127, 'eval_accuracy': 0.5952380952380952, 'eval_f1': 0.6851851851851852, 'eval_precision': 0.5692307692307692, 'eval_recall': 0.8604651162790697, 'eval_runtime': 3.5459, 'eval_samples_per_second': 23.689, 'eval_steps_per_second': 3.102, 'epoch': 1.0}                                                                   
{'loss': 0.6735, 'grad_norm': 2.141087293624878, 'learning_rate': 2.3936170212765957e-06, 'epoch': 1.43}                                                                     
{'loss': 0.6649, 'grad_norm': 4.592581748962402, 'learning_rate': 2.074468085106383e-06, 'epoch': 1.9}                                                                       
{'eval_loss': 0.6663622260093689, 'eval_accuracy': 0.6547619047619048, 'eval_f1': 0.6947368421052632, 'eval_precision': 0.6346153846153846, 'eval_recall': 0.7674418604651163, 'eval_runtime': 3.5546, 'eval_samples_per_second': 23.632, 'eval_steps_per_second': 3.095, 'epoch': 2.0}                                                                   
{'loss': 0.651, 'grad_norm': 2.708678722381592, 'learning_rate': 1.7553191489361702e-06, 'epoch': 2.38}                                                                      
{'loss': 0.6669, 'grad_norm': 3.5693187713623047, 'learning_rate': 1.4361702127659576e-06, 'epoch': 2.86}                                                                    
{'eval_loss': 0.6586034297943115, 'eval_accuracy': 0.6547619047619048, 'eval_f1': 0.6947368421052632, 'eval_precision': 0.6346153846153846, 'eval_recall': 0.7674418604651163, 'eval_runtime': 3.5568, 'eval_samples_per_second': 23.617, 'eval_steps_per_second': 3.093, 'epoch': 3.0}                                                                   
{'loss': 0.644, 'grad_norm': 3.8706870079040527, 'learning_rate': 1.1170212765957447e-06, 'epoch': 3.33}                                                                     
{'loss': 0.6317, 'grad_norm': 2.106621265411377, 'learning_rate': 7.978723404255319e-07, 'epoch': 3.81}                                                                      
{'eval_loss': 0.6507155299186707, 'eval_accuracy': 0.7261904761904762, 'eval_f1': 0.7294117647058823, 'eval_precision': 0.7380952380952381, 'eval_recall': 0.7209302325581395, 'eval_runtime': 3.5532, 'eval_samples_per_second': 23.641, 'eval_steps_per_second': 3.096, 'epoch': 4.0}                                                                   
{'loss': 0.651, 'grad_norm': 5.014101505279541, 'learning_rate': 4.787234042553192e-07, 'epoch': 4.29}                                                                       
{'loss': 0.6431, 'grad_norm': 4.286768913269043, 'learning_rate': 1.5957446808510638e-07, 'epoch': 4.76}                                                                     
{'eval_loss': 0.6484043598175049, 'eval_accuracy': 0.7142857142857143, 'eval_f1': 0.7209302325581395, 'eval_precision': 0.7209302325581395, 'eval_recall': 0.7209302325581395, 'eval_runtime': 3.569, 'eval_samples_per_second': 23.536, 'eval_steps_per_second': 3.082, 'epoch': 5.0}                                                                    
{'train_runtime': 272.1678, 'train_samples_per_second': 6.173, 'train_steps_per_second': 0.386, 'train_loss': 0.6596243608565557, 'epoch': 5.0}                              
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 105/105 [04:32<00:00,  2.59s/it]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:03<00:00,  3.43it/s]
Fold 5 metrics: {'eval_loss': 0.6507155299186707, 'eval_accuracy': 0.7261904761904762, 'eval_f1': 0.7294117647058823, 'eval_precision': 0.7380952380952381, 'eval_recall': 0.7209302325581395, 'eval_runtime': 3.5859, 'eval_samples_per_second': 23.425, 'eval_steps_per_second': 3.068, 'epoch': 5.0}

===== CROSS-VALIDATION RESULTS =====
Mean:
 eval_loss                   0.660294
eval_accuracy               0.680952
eval_f1                     0.726367
eval_precision              0.658092
eval_recall                 0.822481
eval_runtime                3.641480
eval_samples_per_second    23.097800
eval_steps_per_second       3.024800
epoch                       5.000000
dtype: float64
Std:
 eval_loss                  0.014148
eval_accuracy              0.072120
eval_f1                    0.037250
eval_precision             0.078053
eval_recall                0.060445
eval_runtime               0.151402
eval_samples_per_second    0.910214
eval_steps_per_second      0.119272
epoch                      0.000000
dtype: float64

===== TRAINING FINAL MODEL =====
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at emilyalsentzer/Bio_ClinicalBERT and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
{'loss': 0.6885, 'grad_norm': 3.185253620147705, 'learning_rate': 2.307692307692308e-06, 'epoch': 0.38}                                                                      
{'loss': 0.6995, 'grad_norm': 3.8397178649902344, 'learning_rate': 2.8205128205128207e-06, 'epoch': 0.75}                                                                    
{'eval_loss': 0.6786680221557617, 'eval_accuracy': 0.5777777777777777, 'eval_f1': 0.6415094339622641, 'eval_precision': 0.5666666666666667, 'eval_recall': 0.7391304347826086, 'eval_runtime': 3.9421, 'eval_samples_per_second': 22.831, 'eval_steps_per_second': 3.044, 'epoch': 0.98}                                                                  
{'loss': 0.6679, 'grad_norm': 3.301168441772461, 'learning_rate': 2.564102564102564e-06, 'epoch': 1.13}                                                                      
{'loss': 0.6725, 'grad_norm': 2.8229527473449707, 'learning_rate': 2.307692307692308e-06, 'epoch': 1.51}                                                                     
{'loss': 0.6699, 'grad_norm': 4.908132553100586, 'learning_rate': 2.0512820512820513e-06, 'epoch': 1.89}                                                                     
{'eval_loss': 0.6606786847114563, 'eval_accuracy': 0.6777777777777778, 'eval_f1': 0.6947368421052632, 'eval_precision': 0.673469387755102, 'eval_recall': 0.717391304347826, 'eval_runtime': 6.3475, 'eval_samples_per_second': 14.179, 'eval_steps_per_second': 1.89, 'epoch': 2.0}                                                                      
{'loss': 0.6502, 'grad_norm': 4.5313849449157715, 'learning_rate': 1.7948717948717948e-06, 'epoch': 2.26}                                                                    
{'loss': 0.6608, 'grad_norm': 6.120581150054932, 'learning_rate': 1.5384615384615383e-06, 'epoch': 2.64}                                                                     
{'eval_loss': 0.6499717235565186, 'eval_accuracy': 0.6888888888888889, 'eval_f1': 0.7021276595744681, 'eval_precision': 0.6875, 'eval_recall': 0.717391304347826, 'eval_runtime': 7.5491, 'eval_samples_per_second': 11.922, 'eval_steps_per_second': 1.59, 'epoch': 2.98}                                                                                
{'loss': 0.6324, 'grad_norm': 6.830116271972656, 'learning_rate': 1.282051282051282e-06, 'epoch': 3.02}                                                                      
{'loss': 0.6406, 'grad_norm': 4.884839057922363, 'learning_rate': 1.0256410256410257e-06, 'epoch': 3.4}                                                                      
{'loss': 0.6375, 'grad_norm': 5.423312664031982, 'learning_rate': 7.692307692307691e-07, 'epoch': 3.77}                                                                                                                            
{'eval_loss': 0.6430943012237549, 'eval_accuracy': 0.6888888888888889, 'eval_f1': 0.6818181818181818, 'eval_precision': 0.7142857142857143, 'eval_recall': 0.6521739130434783, 'eval_runtime': 13.7214, 'eval_samples_per_second': 6.559, 'eval_steps_per_second': 0.875, 'epoch': 4.0}                                                                                                                                                                               
{'loss': 0.6152, 'grad_norm': 6.205811977386475, 'learning_rate': 5.128205128205128e-07, 'epoch': 4.15}                                                                                                                            
 87%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                | 113/130 [1:57:13<28:58, 102.28s/it]{'loss': 0.6206, 'grad_norm': 4.945920944213867, 'learning_rate': 2.564102564102564e-07, 'epoch': 4.53}                                                                  
{'loss': 0.6162, 'grad_norm': 6.278889179229736, 'learning_rate': 0.0, 'epoch': 4.91}                                                                                    
{'eval_loss': 0.6411712765693665, 'eval_accuracy': 0.6777777777777778, 'eval_f1': 0.6666666666666666, 'eval_precision': 0.7073170731707317, 'eval_recall': 0.6304347826086957, 'eval_runtime': 657.4883, 'eval_samples_per_second': 0.137, 'eval_steps_per_second': 0.018, 'epoch': 4.91}                                                         
{'train_runtime': 14673.9699, 'train_samples_per_second': 0.143, 'train_steps_per_second': 0.009, 'train_loss': 0.6516698800600492, 'epoch': 4.91}                       
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 130/130 [4:04:33<00:00, 112.88s/it]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:06<00:00,  1.97it/s]

Final validation metrics: {'eval_loss': 0.6499717235565186, 'eval_accuracy': 0.6888888888888889, 'eval_f1': 0.7021276595744681, 'eval_precision': 0.6875, 'eval_recall': 0.717391304347826, 'eval_runtime': 9.4863, 'eval_samples_per_second': 9.487, 'eval_steps_per_second': 1.265, 'epoch': 4.90566037735849}
Training complete. Final model saved.

---

#### 10.8 Global Interpretation

Across all phases:

- Best performance achieved early in the third training run (F1 = 0.75)
- Subsequent tuning did not improve results  
- Cross-validation confirms high variability and instability  

Evidence indicates:

- Model capacity is sufficient  
- Input representation is adequate  
- Training procedure is stable  
- Simpler configuration performs best

However:

- Performance does not consistently generalise  
- Results depend on specific data splits  

---

#### 10.9 Final Conclusion and Decision

All empirical evidence converges on a single constraint, the dataset size is the primary limiting factor for performance rather than model architecture or hyperparameter configuration.

Supporting evidence:

- No improvement from hyperparameter tuning  
- Increased variance in cross-validation  
- Degradation with added complexity  
- Stable but limited performance ceiling (~F1 = 0.75)  

Conclusion:

- The model is not underpowered  
- The training setup is not flawed  
- The dataset is insufficient to support further learning  

We must stop further training iterations and hyperparameter tuning:

- No observed performance gains  
- Increased risk of overfitting 
- Does not improve generalisation
- Diminishing returns from further adjustments 

---

#### 10.10 Next Steps

Increase the dataset size to enable further learning and performance improvements:

- Expand annotated dataset from ~600 → 1200 samples  
- Maintain the highest scoring configuration (third training run) for simplicity
- Retrain without introducing additional complexity (additional hyperparameters are not justified by current evidence)

For future iterations training and scaling should stop when:

- F1-score plateaus despite additional data  
- Cross-validation variance decreases  
- Precision improves without recall collapse  

---

### 11. Workflow Implementation

The final training workflow is implemented in `train_validate_transformer.py`. The end-to-end pipeline includes:

1. **Reproducibility setup**
  - Fix random seeds across `random`, `numpy`, and `torch`
  - Enable deterministic CUDA behaviour
  - Reproducible results across runs and environments

2. **Configuration initialisation**
  - Define file paths:
    - `train.csv` (420 samples)
    - `val.csv` (90 samples)
  - Set model and training hyperparameters
  - Create output directory: `models/bioclinicalbert/`

3. **Data loading and preprocessing**
  - Load CSV files into pandas DataFrames
  - Convert boolean labels (`is_valid`) → integer labels (`0/1`)

4. **Dataset construction**
  - Convert DataFrames → Hugging Face `Dataset`
  - Retain relevant fields: `sentence_text`, `entity_type`, `entity_text`, `concept`, `task`, `label`

5. **Input formatting and tokenisation**
  - Construct structured input string:
     ```
     [ENTITY TYPE] ... [ENTITY] ... [CONCEPT] ... [TASK] ... [TEXT] ...
     ```
  - Apply tokenizer with:
    - Truncation (`max_length = 512`)
    - Padding (`max_length`)
  - Convert to PyTorch tensors: `input_ids`, `attention_mask`, `label`

6. **Metric definition**
  - Compute:
    - Accuracy
    - Precision
    - Recall
    - F1-score (primary metric)

7. **Training configuration**
  - Key settings:
    - Batch size = 8
    - Gradient accumulation = 2 (effective batch = 16)
    - Epochs = 5
    - Learning rate = 3e-6
    - Warmup ratio = 0.1
    - Linear scheduler
    - Gradient clipping = 1.0
    - Weight decay = 0.05
  - Validation and checkpointing:
    - Evaluation per epoch
    - Best model selected via F1-score

8. **Cross-validation (robustness assessment)**
  - Performed before final model training using only the training dataset (420 samples)
  - Apply 5-fold stratified split to preserve class distribution
  - For each fold:
    - Create train/validation subsets (~336 / ~84)
    - Re-tokenize data
    - Initialise a fresh model from the pretrained checkpoint (no weight sharing between folds)
    - Train and evaluate independently using identical training configuration
  - Store metrics per fold
  - Aggregate:
    - Mean performance
    - Standard deviation (stability)
  - Models trained during cross-validation are discarded after evaluation

9. **Model training (final deployable model)**
   - Initialise a new model from the pretrained checkpoint
    - Attach binary classification head (`num_labels = 2`)
    - Classification head weights randomly initialised
   - Train using full `train.csv` (420 samples)
   - Validate on `val.csv` (90 samples)
   - Performed via Hugging Face `Trainer`:
    - Forward pass → logits
    - Loss computation → backpropagation
    - Optimizer + scheduler updates
   - Validate at end of each epoch
   - Track best-performing checkpoint based on F1-score

10. **Model saving**
  - Save final trained model: `models/bioclinicalbert/`
    - Model weights (`pytorch_model.bin`)  
    - Model configuration (`config.json`)  
  - Save tokenizer files (`vocab.txt`, `tokenizer_config.json`, etc.)  
    - Ensures full reproducibility and usability for inference

11. **Output reporting**
  - Print:
    - Cross-validation metrics per fold  
    - Aggregated CV statistics (mean ± standard deviation)  
    - Training progress logs for final model  
    - Epoch-level validation metrics  
    - Final validation performance  

---

## Data Expansion

### 1. Objective

The objective of data expansion is to increase the size of the annotated dataset from 600 samples to 1200 samples, manualy annotate another 600 samples, resplit the updated dataset into training and validation and evaluation sets (80/10/10), and retrain the model using the best-performing configuration identified in the previous section.

The previous section identified through training that the bottleneck was not hyeprparaemter tuning but instead the dataset size. The model was able to learn meaningful patterns and achieve a reasonable F1-score, but performance was unstable and did not generalise well due to the limited number of training samples.

This section therefore focuses on expanding the dataset to provide more learning signal for the model, which is expected to lead to improved performance and stability, and be teh final iteration of the model training process before final evaluation on the test set.





### 2. Dataset Expansion Rationale

The decision to increase the dataset size from 600 to 1200 samples is based on empirical observations from model performance and variance analysis.

Cross-validation results demonstrated:
	•	Moderate performance (F1 ≈ 0.75)
	•	High variability across folds (±0.04)

This indicates the model is operating in a data-limited regime, where performance is constrained by insufficient training data rather than model capacity or optimisation.

In such settings, generalisation error and variance scale approximately with 1/\sqrt{N}, meaning that increasing dataset size reduces variance and improves stability.

Doubling the dataset from 600 to 1200:
	•	Reduces expected variance by ~29%
	•	Provides sufficient additional signal to improve generalisation
	•	Represents the smallest increase likely to produce a measurable effect

This approach follows standard machine learning practice, where dataset size is increased incrementally and performance is reassessed after each scaling step.


---

### 3. Sampling Additional Data

#### 3.1 Objective

- Generate a new, balanced, annotation-ready dataset of extracted clinical entities for transformer validation, while avoiding overlap with the previously sampled dataset.  

---

#### 3.2 Deduplication Strategy

Sampling strategy is exactly the same as before, with the following key steps:

1. Sample by `entity_type` to ensure class balance across SYMPTOM, INTERVENTION, and CLINICAL_CONDITION
2. Concatenate the 3 entity types together to form a single dataset of 600 samples (200 per entity type)
3. Randomly shuffle the dataset with a fixed seed for reproducibility

The critical new addition when sampling is the addition of filtering logic for deduplication. To prevent duplicates between the new sample and the previous 600 annotated entities:

- We perform row-level deduplication based on the five key columns which the trasnformer uses: `sentence_text`, `entity_text`, `entity_type`, `concept`, `task`.
- Tuple-based comparison is used for this purpose:
  - The row-wise structure (`df_tuples`) is kept as a pandas Series of tuples, aligned with the DataFrame, to generate a boolean mask for filtering.
  - Only the lookup structure (`existing_tuples`) is converted to a set for
  efficient membership testing.
- This approach ensures that any entity-context combination already present in the previous sample is excluded from the new dataset, regardless of row index or repeated occurrences.

Due to the nature of this deduplication, it will deduplicate any rows which match those 5 columns, meaning that repeated entities in a sentence are treated as duplicates. this is becayse the trasnformer does not get access to the indexes anyways so essentially when training those are duplicates

Therefore it will be expected that there will be >600 rows removed from the dataset since some of the entities in the 600 sampled entities will correspond to more than 1 row in the dataset based on the match to the 5 columns as we dont include the other columns such as the indexes for thr specific entity, so repeated entities at different index position in a senetnce will be essentially treated as the same

This is fine though as it still aligns with what the trasnformer sees, and also is technically better therefore, since it means that the new sampled entities will always be completely unique and not be considered duplicates by the trasnformer.

---

#### 3.3 Sampling Workflow

The logic is implemented in sample_additional_entities.py and follows these steps:



---

#### 3.4 Sampling Results

Terminal valdiation shows:
- Loaded 47,487 total entities
- retained 46,674 entities 
- Final sample size of 600 entities

This shows that the deduplication worked, where 813 rows were removed, which is expected as there is most likelymultiple appearneces of the entities from the rpevious sample in the full extraction

---

### 4. Manual Annotation and Validation 

#### 4.1 Objective

- Manually annotate the new dataset with binary labels (`is_valid`) indicating whether each entity is a valid extraction in its context, following the same annotation guidelines as before.
- Validate the new annotated dataset to ensure label quality, class balance, and consistency with the previous dataset before retraining the model.

#### 4.1 Annotation and Validation Strategy

Annotation Strategy remains consistent with the previous process:

- Each new entity is assigned an empty `is_valid` column for subsequent manual labeling.
- The same annotation event-based and status-based guidelines per entity type are applied to ensure consistency in labeling criteria across both datasets.

Validation of the new annotated dataset will be the same as before, printing metrics to make sure sampling, annotation labels, and class balance are all correct before proceeding to model training.

---

#### 4.3 Validation Results

Exact same logic as before but with new dataset, validate_additional_manual_annotations.py

=== BASIC INFO ===
Total rows: 600
Index(['note_id', 'section', 'concept', 'entity_text', 'entity_type',
       'sentence_text', 'negated', 'task', 'confidence', 'is_valid'],
      dtype='object')


=== MISSING VALUES ===
note_id            0
section            0
concept            0
entity_text        0
entity_type        0
sentence_text      0
negated          400
task               0
confidence         0
is_valid           0
dtype: int64

=== CRITICAL FIELD CHECKS ===
Missing task: 0
Missing sentence_text: 0
Missing is_valid: 0

=== UNIQUE is_valid VALUES ===
[ True False]

=== is_valid DISTRIBUTION ===
is_valid
False    306
True     294
Name: count, dtype: int64

=== INVALID LABEL ROWS ===
Number of invalid label rows: 0

=== TASK DISTRIBUTION ===
task
symptom_presence             200
clinical_condition_active    200
intervention_performed       200
Name: count, dtype: int64

=== CHECK TASK SIZE (expect 200 each) ===
symptom_presence: 200
clinical_condition_active: 200
intervention_performed: 200

=== TASK vs is_valid ===
is_valid                   False  True 
task                                   
clinical_condition_active    117     83
intervention_performed        81    119
symptom_presence             108     92

=== VALIDATION COMPLETE ===
Review warnings above before proceeding to stratified split.

---

### 5. Stratified Resplitting

---

#### 5.1 Objective

- Resplit the combined dataset of 1200 annotated entities into training and evaluation sets with a 10000/200 split, ensuring stratification by `entity_type` to maintain class balance across all sets.
- This new resplitting is necessary to incorporate the new annotated entities into the training process while allowing for cross validation and final evaluation set for unbiased performance assessment.
- This is the final stage before we retrain the model using the best-performing configuration identified in the previous section.

#### 5.2 Dataset Combination


#### 5.2 Resplitting Workflow


---

#### 5.3 Resplitting Results

Loaded first dataset with 600 rows
Loaded second dataset with 600 rows
Combined dataset has 1200 rows

=== SPLIT SIZES ===
Train: 1020
Eval: 180

=== TRAIN DISTRIBUTION ===

Task distribution:
task
clinical_condition_active    340
symptom_presence             340
intervention_performed       340
Name: count, dtype: int64

is_valid distribution:
is_valid
True     510
False    510
Name: count, dtype: int64

Task vs is_valid:
is_valid                   False  True 
task                                   
clinical_condition_active    203    137
intervention_performed       129    211
symptom_presence             178    162

=== EVAL DISTRIBUTION ===

Task distribution:
task
symptom_presence             60
intervention_performed       60
clinical_condition_active    60
Name: count, dtype: int64

is_valid distribution:
is_valid
False    91
True     89
Name: count, dtype: int64

Task vs is_valid:
is_valid                   False  True 
task                                   
clinical_condition_active     36     24
intervention_performed        23     37
symptom_presence              32     28

----

## Model Retraining

===== RUNNING CROSS-VALIDATION =====


########## config_stable ##########

=== Fold 1 / 5 ===
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 816/816 [00:00<00:00, 6676.37 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 204/204 [00:00<00:00, 7457.52 examples/s]
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at emilyalsentzer/Bio_ClinicalBERT and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
{'loss': 0.7104, 'grad_norm': 2.9828312397003174, 'learning_rate': 4.836601307189543e-06, 'epoch': 0.1}                                                                                                            
{'loss': 0.6838, 'grad_norm': 4.797733306884766, 'learning_rate': 4.673202614379085e-06, 'epoch': 0.2}                                                                                                             
{'loss': 0.646, 'grad_norm': 7.429051399230957, 'learning_rate': 4.509803921568628e-06, 'epoch': 0.29}                                                                                                             
{'loss': 0.6696, 'grad_norm': 4.405584335327148, 'learning_rate': 4.34640522875817e-06, 'epoch': 0.39}                                                                                                             
{'loss': 0.6824, 'grad_norm': 7.632701873779297, 'learning_rate': 4.183006535947713e-06, 'epoch': 0.49}                                                                                                            
{'loss': 0.7011, 'grad_norm': 6.320907115936279, 'learning_rate': 4.019607843137255e-06, 'epoch': 0.59}                                                                                                            
{'loss': 0.6349, 'grad_norm': 7.496801376342773, 'learning_rate': 3.856209150326798e-06, 'epoch': 0.69}                                                                                                            
{'loss': 0.682, 'grad_norm': 7.07696008682251, 'learning_rate': 3.6928104575163404e-06, 'epoch': 0.78}                                                                                                             
{'loss': 0.664, 'grad_norm': 5.306292533874512, 'learning_rate': 3.529411764705883e-06, 'epoch': 0.88}                                                                                                             
{'loss': 0.6483, 'grad_norm': 5.939138889312744, 'learning_rate': 3.3660130718954253e-06, 'epoch': 0.98}                                                                                                           
{'eval_loss': 0.6264452338218689, 'eval_accuracy': 0.7058823529411765, 'eval_f1': 0.6739130434782609, 'eval_precision': 0.7560975609756098, 'eval_recall': 0.6078431372549019, 'eval_runtime': 8.4385, 'eval_samples_per_second': 24.175, 'eval_steps_per_second': 3.081, 'epoch': 1.0}                                                                                                                                               
{'loss': 0.6423, 'grad_norm': 6.189306735992432, 'learning_rate': 3.2026143790849674e-06, 'epoch': 1.08}                                                                                                           
{'loss': 0.6163, 'grad_norm': 5.869161605834961, 'learning_rate': 3.03921568627451e-06, 'epoch': 1.18}                                                                                                             
{'loss': 0.6285, 'grad_norm': 5.808520317077637, 'learning_rate': 2.8758169934640523e-06, 'epoch': 1.27}                                                                                                           
{'loss': 0.6235, 'grad_norm': 7.516668319702148, 'learning_rate': 2.7124183006535947e-06, 'epoch': 1.37}                                                                                                           
{'loss': 0.6209, 'grad_norm': 5.066751480102539, 'learning_rate': 2.549019607843137e-06, 'epoch': 1.47}                                                                                                            
{'loss': 0.6145, 'grad_norm': 5.928969860076904, 'learning_rate': 2.38562091503268e-06, 'epoch': 1.57}                                                                                                             
{'loss': 0.582, 'grad_norm': 7.156850337982178, 'learning_rate': 2.222222222222222e-06, 'epoch': 1.67}                                                                                                             
{'loss': 0.5838, 'grad_norm': 8.438311576843262, 'learning_rate': 2.058823529411765e-06, 'epoch': 1.76}                                                                                                            
{'loss': 0.5698, 'grad_norm': 4.204545497894287, 'learning_rate': 1.8954248366013072e-06, 'epoch': 1.86}                                                                                                           
{'loss': 0.6134, 'grad_norm': 5.319906711578369, 'learning_rate': 1.7320261437908499e-06, 'epoch': 1.96}                                                                                                           
{'eval_loss': 0.5850840210914612, 'eval_accuracy': 0.7303921568627451, 'eval_f1': 0.7236180904522613, 'eval_precision': 0.7422680412371134, 'eval_recall': 0.7058823529411765, 'eval_runtime': 8.1722, 'eval_samples_per_second': 24.963, 'eval_steps_per_second': 3.182, 'epoch': 2.0}                                                                                                                                               
{'loss': 0.5921, 'grad_norm': 6.457675933837891, 'learning_rate': 1.5686274509803923e-06, 'epoch': 2.06}                                                                                                           
{'loss': 0.5357, 'grad_norm': 4.740350246429443, 'learning_rate': 1.4052287581699348e-06, 'epoch': 2.16}                                                                                                           
{'loss': 0.6118, 'grad_norm': 3.581923484802246, 'learning_rate': 1.2418300653594772e-06, 'epoch': 2.25}                                                                                                           
{'loss': 0.564, 'grad_norm': 10.301691055297852, 'learning_rate': 1.0784313725490197e-06, 'epoch': 2.35}                                                                                                           
{'loss': 0.5807, 'grad_norm': 5.149842262268066, 'learning_rate': 9.150326797385621e-07, 'epoch': 2.45}                                                                                                            
{'loss': 0.5779, 'grad_norm': 6.3348212242126465, 'learning_rate': 7.516339869281046e-07, 'epoch': 2.55}                                                                                                           
{'loss': 0.5901, 'grad_norm': 6.308790683746338, 'learning_rate': 5.882352941176471e-07, 'epoch': 2.65}                                                                                                            
{'loss': 0.5259, 'grad_norm': 4.945024490356445, 'learning_rate': 4.248366013071896e-07, 'epoch': 2.75}                                                                                                            
{'loss': 0.5896, 'grad_norm': 5.726438999176025, 'learning_rate': 2.6143790849673207e-07, 'epoch': 2.84}                                                                                                           
{'loss': 0.5402, 'grad_norm': 9.189987182617188, 'learning_rate': 9.803921568627452e-08, 'epoch': 2.94}                                                                                                            
{'eval_loss': 0.5650818943977356, 'eval_accuracy': 0.7352941176470589, 'eval_f1': 0.7244897959183674, 'eval_precision': 0.7553191489361702, 'eval_recall': 0.696078431372549, 'eval_runtime': 8.1803, 'eval_samples_per_second': 24.938, 'eval_steps_per_second': 3.178, 'epoch': 3.0}                                                                                                                                                
{'train_runtime': 371.5997, 'train_samples_per_second': 6.588, 'train_steps_per_second': 0.823, 'train_loss': 0.6175991865544538, 'epoch': 3.0}                                                                    
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 306/306 [06:11<00:00,  1.21s/it]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 26/26 [00:07<00:00,  3.32it/s]
Fold 1 metrics: {'eval_loss': 0.5650818943977356, 'eval_accuracy': 0.7352941176470589, 'eval_f1': 0.7244897959183674, 'eval_precision': 0.7553191489361702, 'eval_recall': 0.696078431372549, 'eval_runtime': 8.162, 'eval_samples_per_second': 24.994, 'eval_steps_per_second': 3.186, 'epoch': 3.0}

=== Fold 2 / 5 ===
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 816/816 [00:00<00:00, 6164.87 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 204/204 [00:00<00:00, 7160.69 examples/s]
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at emilyalsentzer/Bio_ClinicalBERT and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
{'loss': 0.71, 'grad_norm': 3.4598779678344727, 'learning_rate': 4.836601307189543e-06, 'epoch': 0.1}                                                                                                              
{'loss': 0.7214, 'grad_norm': 6.133597373962402, 'learning_rate': 4.673202614379085e-06, 'epoch': 0.2}                                                                                                             
{'loss': 0.7096, 'grad_norm': 8.535781860351562, 'learning_rate': 4.509803921568628e-06, 'epoch': 0.29}                                                                                                            
{'loss': 0.6767, 'grad_norm': 5.241737365722656, 'learning_rate': 4.34640522875817e-06, 'epoch': 0.39}                                                                                                             
{'loss': 0.6493, 'grad_norm': 5.054990291595459, 'learning_rate': 4.183006535947713e-06, 'epoch': 0.49}                                                                                                            
{'loss': 0.6772, 'grad_norm': 3.2068755626678467, 'learning_rate': 4.019607843137255e-06, 'epoch': 0.59}                                                                                                           
{'loss': 0.6623, 'grad_norm': 6.402395248413086, 'learning_rate': 3.856209150326798e-06, 'epoch': 0.69}                                                                                                            
{'loss': 0.6641, 'grad_norm': 6.143208980560303, 'learning_rate': 3.6928104575163404e-06, 'epoch': 0.78}                                                                                                           
{'loss': 0.6586, 'grad_norm': 3.8530826568603516, 'learning_rate': 3.529411764705883e-06, 'epoch': 0.88}                                                                                                           
{'loss': 0.6592, 'grad_norm': 5.458342552185059, 'learning_rate': 3.3660130718954253e-06, 'epoch': 0.98}                                                                                                           
{'eval_loss': 0.658694863319397, 'eval_accuracy': 0.6372549019607843, 'eval_f1': 0.6262626262626263, 'eval_precision': 0.6458333333333334, 'eval_recall': 0.6078431372549019, 'eval_runtime': 8.1789, 'eval_samples_per_second': 24.942, 'eval_steps_per_second': 3.179, 'epoch': 1.0}                                                                                                                                                
{'loss': 0.6357, 'grad_norm': 4.724815368652344, 'learning_rate': 3.2026143790849674e-06, 'epoch': 1.08}                                                                                                           
{'loss': 0.6457, 'grad_norm': 7.950567722320557, 'learning_rate': 3.03921568627451e-06, 'epoch': 1.18}                                                                                                             
{'loss': 0.599, 'grad_norm': 7.461246013641357, 'learning_rate': 2.8758169934640523e-06, 'epoch': 1.27}                                                                                                            
{'loss': 0.6064, 'grad_norm': 4.712815761566162, 'learning_rate': 2.7124183006535947e-06, 'epoch': 1.37}                                                                                                           
{'loss': 0.6117, 'grad_norm': 6.822486400604248, 'learning_rate': 2.549019607843137e-06, 'epoch': 1.47}                                                                                                            
{'loss': 0.5957, 'grad_norm': 4.063061237335205, 'learning_rate': 2.38562091503268e-06, 'epoch': 1.57}                                                                                                             
{'loss': 0.6075, 'grad_norm': 5.335022926330566, 'learning_rate': 2.222222222222222e-06, 'epoch': 1.67}                                                                                                            
{'loss': 0.6106, 'grad_norm': 7.241730690002441, 'learning_rate': 2.058823529411765e-06, 'epoch': 1.76}                                                                                                            
{'loss': 0.6048, 'grad_norm': 3.4495575428009033, 'learning_rate': 1.8954248366013072e-06, 'epoch': 1.86}                                                                                                          
{'loss': 0.5829, 'grad_norm': 6.568054676055908, 'learning_rate': 1.7320261437908499e-06, 'epoch': 1.96}                                                                                                           
{'eval_loss': 0.6436905860900879, 'eval_accuracy': 0.6274509803921569, 'eval_f1': 0.6415094339622641, 'eval_precision': 0.6181818181818182, 'eval_recall': 0.6666666666666666, 'eval_runtime': 8.2187, 'eval_samples_per_second': 24.822, 'eval_steps_per_second': 3.164, 'epoch': 2.0}                                                                                                                                               
{'loss': 0.6029, 'grad_norm': 10.449905395507812, 'learning_rate': 1.5686274509803923e-06, 'epoch': 2.06}                                                                                                          
{'loss': 0.5647, 'grad_norm': 8.356180191040039, 'learning_rate': 1.4052287581699348e-06, 'epoch': 2.16}                                                                                                           
{'loss': 0.5941, 'grad_norm': 4.081063747406006, 'learning_rate': 1.2418300653594772e-06, 'epoch': 2.25}                                                                                                           
{'loss': 0.5604, 'grad_norm': 8.927538871765137, 'learning_rate': 1.0784313725490197e-06, 'epoch': 2.35}                                                                                                           
{'loss': 0.5924, 'grad_norm': 4.191283702850342, 'learning_rate': 9.150326797385621e-07, 'epoch': 2.45}                                                                                                            
{'loss': 0.5774, 'grad_norm': 10.436814308166504, 'learning_rate': 7.516339869281046e-07, 'epoch': 2.55}                                                                                                           
{'loss': 0.5829, 'grad_norm': 3.5657529830932617, 'learning_rate': 5.882352941176471e-07, 'epoch': 2.65}                                                                                                           
{'loss': 0.6015, 'grad_norm': 4.568387985229492, 'learning_rate': 4.248366013071896e-07, 'epoch': 2.75}                                                                                                            
{'loss': 0.5634, 'grad_norm': 9.550910949707031, 'learning_rate': 2.6143790849673207e-07, 'epoch': 2.84}                                                                                                           
{'loss': 0.5789, 'grad_norm': 5.297276020050049, 'learning_rate': 9.803921568627452e-08, 'epoch': 2.94}                                                                                                            
{'eval_loss': 0.6346308588981628, 'eval_accuracy': 0.6519607843137255, 'eval_f1': 0.6536585365853659, 'eval_precision': 0.6504854368932039, 'eval_recall': 0.6568627450980392, 'eval_runtime': 8.1908, 'eval_samples_per_second': 24.906, 'eval_steps_per_second': 3.174, 'epoch': 3.0}                                                                                                                                               
{'train_runtime': 566.9476, 'train_samples_per_second': 4.318, 'train_steps_per_second': 0.54, 'train_loss': 0.6223967550626768, 'epoch': 3.0}                                                                     
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 306/306 [09:26<00:00,  1.85s/it]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 26/26 [00:07<00:00,  3.31it/s]
Fold 2 metrics: {'eval_loss': 0.6346308588981628, 'eval_accuracy': 0.6519607843137255, 'eval_f1': 0.6536585365853659, 'eval_precision': 0.6504854368932039, 'eval_recall': 0.6568627450980392, 'eval_runtime': 8.1767, 'eval_samples_per_second': 24.949, 'eval_steps_per_second': 3.18, 'epoch': 3.0}

=== Fold 3 / 5 ===
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 816/816 [00:00<00:00, 6554.74 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 204/204 [00:00<00:00, 7943.98 examples/s]
'(ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')), '(Request ID: edd8ee63-e878-4df7-8efc-f4bdd0baaa6d)')' thrown while requesting HEAD https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT/resolve/main/config.json
Retrying in 1s [Retry 1/5].
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at emilyalsentzer/Bio_ClinicalBERT and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
{'loss': 0.7237, 'grad_norm': 7.450933933258057, 'learning_rate': 4.836601307189543e-06, 'epoch': 0.1}                                                                                                             
{'loss': 0.7031, 'grad_norm': 5.343350410461426, 'learning_rate': 4.673202614379085e-06, 'epoch': 0.2}                                                                                                             
{'loss': 0.6946, 'grad_norm': 3.964149236679077, 'learning_rate': 4.509803921568628e-06, 'epoch': 0.29}                                                                                                            
{'loss': 0.6831, 'grad_norm': 3.719081401824951, 'learning_rate': 4.34640522875817e-06, 'epoch': 0.39}                                                                                                             
{'loss': 0.6848, 'grad_norm': 4.116750717163086, 'learning_rate': 4.183006535947713e-06, 'epoch': 0.49}                                                                                                            
{'loss': 0.6524, 'grad_norm': 11.370431900024414, 'learning_rate': 4.019607843137255e-06, 'epoch': 0.59}                                                                                                           
{'loss': 0.6264, 'grad_norm': 4.444675922393799, 'learning_rate': 3.856209150326798e-06, 'epoch': 0.69}                                                                                                            
{'loss': 0.6349, 'grad_norm': 4.258211612701416, 'learning_rate': 3.6928104575163404e-06, 'epoch': 0.78}                                                                                                           
{'loss': 0.6495, 'grad_norm': 6.4983229637146, 'learning_rate': 3.529411764705883e-06, 'epoch': 0.88}                                                                                                              
{'loss': 0.6234, 'grad_norm': 5.4850172996521, 'learning_rate': 3.3660130718954253e-06, 'epoch': 0.98}                                                                                                             
{'eval_loss': 0.6126949191093445, 'eval_accuracy': 0.7058823529411765, 'eval_f1': 0.7087378640776699, 'eval_precision': 0.7019230769230769, 'eval_recall': 0.7156862745098039, 'eval_runtime': 8.4873, 'eval_samples_per_second': 24.036, 'eval_steps_per_second': 3.063, 'epoch': 1.0}                                                                                                                                               
{'loss': 0.624, 'grad_norm': 4.384472846984863, 'learning_rate': 3.2026143790849674e-06, 'epoch': 1.08}                                                                                                            
{'loss': 0.6123, 'grad_norm': 4.464570045471191, 'learning_rate': 3.03921568627451e-06, 'epoch': 1.18}                                                                                                             
{'loss': 0.5764, 'grad_norm': 7.515692234039307, 'learning_rate': 2.8758169934640523e-06, 'epoch': 1.27}                                                                                                           
{'loss': 0.6188, 'grad_norm': 5.103363513946533, 'learning_rate': 2.7124183006535947e-06, 'epoch': 1.37}                                                                                                           
{'loss': 0.6211, 'grad_norm': 5.6629509925842285, 'learning_rate': 2.549019607843137e-06, 'epoch': 1.47}                                                                                                           
{'loss': 0.6023, 'grad_norm': 6.067633628845215, 'learning_rate': 2.38562091503268e-06, 'epoch': 1.57}                                                                                                             
{'loss': 0.5696, 'grad_norm': 5.184009552001953, 'learning_rate': 2.222222222222222e-06, 'epoch': 1.67}                                                                                                            
{'loss': 0.6295, 'grad_norm': 4.344557762145996, 'learning_rate': 2.058823529411765e-06, 'epoch': 1.76}                                                                                                            
{'loss': 0.6016, 'grad_norm': 5.081483840942383, 'learning_rate': 1.8954248366013072e-06, 'epoch': 1.86}                                                                                                           
{'loss': 0.5664, 'grad_norm': 4.813072681427002, 'learning_rate': 1.7320261437908499e-06, 'epoch': 1.96}                                                                                                           
{'eval_loss': 0.5768228769302368, 'eval_accuracy': 0.7009803921568627, 'eval_f1': 0.7053140096618358, 'eval_precision': 0.6952380952380952, 'eval_recall': 0.7156862745098039, 'eval_runtime': 8.1798, 'eval_samples_per_second': 24.939, 'eval_steps_per_second': 3.179, 'epoch': 2.0}                                                                                                                                               
{'loss': 0.6048, 'grad_norm': 5.178508281707764, 'learning_rate': 1.5686274509803923e-06, 'epoch': 2.06}                                                                                                           
{'loss': 0.594, 'grad_norm': 8.598237037658691, 'learning_rate': 1.4052287581699348e-06, 'epoch': 2.16}                                                                                                            
{'loss': 0.5681, 'grad_norm': 4.172880172729492, 'learning_rate': 1.2418300653594772e-06, 'epoch': 2.25}                                                                                                           
{'loss': 0.5344, 'grad_norm': 7.3776535987854, 'learning_rate': 1.0784313725490197e-06, 'epoch': 2.35}                                                                                                             
{'loss': 0.5999, 'grad_norm': 5.274287700653076, 'learning_rate': 9.150326797385621e-07, 'epoch': 2.45}                                                                                                            
{'loss': 0.5769, 'grad_norm': 9.03189754486084, 'learning_rate': 7.516339869281046e-07, 'epoch': 2.55}                                                                                                             
{'loss': 0.5823, 'grad_norm': 12.922530174255371, 'learning_rate': 5.882352941176471e-07, 'epoch': 2.65}                                                                                                           
{'loss': 0.5805, 'grad_norm': 3.3748466968536377, 'learning_rate': 4.248366013071896e-07, 'epoch': 2.75}                                                                                                           
{'loss': 0.5826, 'grad_norm': 4.418302059173584, 'learning_rate': 2.6143790849673207e-07, 'epoch': 2.84}                                                                                                           
{'loss': 0.5887, 'grad_norm': 5.625654220581055, 'learning_rate': 9.803921568627452e-08, 'epoch': 2.94}                                                                                                            
{'eval_loss': 0.5694177746772766, 'eval_accuracy': 0.7156862745098039, 'eval_f1': 0.7156862745098039, 'eval_precision': 0.7156862745098039, 'eval_recall': 0.7156862745098039, 'eval_runtime': 8.187, 'eval_samples_per_second': 24.917, 'eval_steps_per_second': 3.176, 'epoch': 3.0}                                                                                                                                                
{'train_runtime': 372.246, 'train_samples_per_second': 6.576, 'train_steps_per_second': 0.822, 'train_loss': 0.6158494544185065, 'epoch': 3.0}                                                                     
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 306/306 [06:12<00:00,  1.22s/it]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 26/26 [00:07<00:00,  3.31it/s]
Fold 3 metrics: {'eval_loss': 0.5694177746772766, 'eval_accuracy': 0.7156862745098039, 'eval_f1': 0.7156862745098039, 'eval_precision': 0.7156862745098039, 'eval_recall': 0.7156862745098039, 'eval_runtime': 8.1709, 'eval_samples_per_second': 24.967, 'eval_steps_per_second': 3.182, 'epoch': 3.0}

=== Fold 4 / 5 ===
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 816/816 [00:00<00:00, 6267.62 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 204/204 [00:00<00:00, 7699.29 examples/s]
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at emilyalsentzer/Bio_ClinicalBERT and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
{'loss': 0.7939, 'grad_norm': 4.448887825012207, 'learning_rate': 4.836601307189543e-06, 'epoch': 0.1}                                                                                                             
{'loss': 0.6725, 'grad_norm': 5.915727138519287, 'learning_rate': 4.673202614379085e-06, 'epoch': 0.2}                                                                                                             
{'loss': 0.6991, 'grad_norm': 3.144515037536621, 'learning_rate': 4.509803921568628e-06, 'epoch': 0.29}                                                                                                            
{'loss': 0.6716, 'grad_norm': 3.372199773788452, 'learning_rate': 4.34640522875817e-06, 'epoch': 0.39}                                                                                                             
{'loss': 0.6689, 'grad_norm': 4.342011451721191, 'learning_rate': 4.183006535947713e-06, 'epoch': 0.49}                                                                                                            
{'loss': 0.6309, 'grad_norm': 4.792886257171631, 'learning_rate': 4.019607843137255e-06, 'epoch': 0.59}                                                                                                            
{'loss': 0.6468, 'grad_norm': 4.200644016265869, 'learning_rate': 3.856209150326798e-06, 'epoch': 0.69}                                                                                                            
{'loss': 0.6374, 'grad_norm': 3.846693277359009, 'learning_rate': 3.6928104575163404e-06, 'epoch': 0.78}                                                                                                           
{'loss': 0.6764, 'grad_norm': 3.4830870628356934, 'learning_rate': 3.529411764705883e-06, 'epoch': 0.88}                                                                                                           
{'loss': 0.6164, 'grad_norm': 4.598002910614014, 'learning_rate': 3.3660130718954253e-06, 'epoch': 0.98}                                                                                                           
{'eval_loss': 0.6229746341705322, 'eval_accuracy': 0.7205882352941176, 'eval_f1': 0.7046632124352331, 'eval_precision': 0.7472527472527473, 'eval_recall': 0.6666666666666666, 'eval_runtime': 8.631, 'eval_samples_per_second': 23.636, 'eval_steps_per_second': 3.012, 'epoch': 1.0}                                                                                                                                                
{'loss': 0.6327, 'grad_norm': 7.845837593078613, 'learning_rate': 3.2026143790849674e-06, 'epoch': 1.08}                                                                                                           
{'loss': 0.6264, 'grad_norm': 3.66449236869812, 'learning_rate': 3.03921568627451e-06, 'epoch': 1.18}                                                                                                              
{'loss': 0.5997, 'grad_norm': 5.8965044021606445, 'learning_rate': 2.8758169934640523e-06, 'epoch': 1.27}                                                                                                          
{'loss': 0.5903, 'grad_norm': 4.240746974945068, 'learning_rate': 2.7124183006535947e-06, 'epoch': 1.37}                                                                                                           
{'loss': 0.5991, 'grad_norm': 8.780424118041992, 'learning_rate': 2.549019607843137e-06, 'epoch': 1.47}                                                                                                            
{'loss': 0.5901, 'grad_norm': 7.313333034515381, 'learning_rate': 2.38562091503268e-06, 'epoch': 1.57}                                                                                                             
{'loss': 0.6164, 'grad_norm': 4.517190933227539, 'learning_rate': 2.222222222222222e-06, 'epoch': 1.67}                                                                                                            
{'loss': 0.6229, 'grad_norm': 4.648843765258789, 'learning_rate': 2.058823529411765e-06, 'epoch': 1.76}                                                                                                            
{'loss': 0.5456, 'grad_norm': 6.032140254974365, 'learning_rate': 1.8954248366013072e-06, 'epoch': 1.86}                                                                                                           
{'loss': 0.6176, 'grad_norm': 3.89502215385437, 'learning_rate': 1.7320261437908499e-06, 'epoch': 1.96}                                                                                                            
{'eval_loss': 0.5888649821281433, 'eval_accuracy': 0.7205882352941176, 'eval_f1': 0.7219512195121951, 'eval_precision': 0.7184466019417476, 'eval_recall': 0.7254901960784313, 'eval_runtime': 8.4308, 'eval_samples_per_second': 24.197, 'eval_steps_per_second': 3.084, 'epoch': 2.0}                                                                                                                                               
{'loss': 0.6031, 'grad_norm': 6.36028528213501, 'learning_rate': 1.5686274509803923e-06, 'epoch': 2.06}                                                                                                            
{'loss': 0.5465, 'grad_norm': 5.0244140625, 'learning_rate': 1.4052287581699348e-06, 'epoch': 2.16}                                                                                                                
{'loss': 0.5882, 'grad_norm': 6.1078972816467285, 'learning_rate': 1.2418300653594772e-06, 'epoch': 2.25}                                                                                                          
{'loss': 0.5898, 'grad_norm': 7.344321250915527, 'learning_rate': 1.0784313725490197e-06, 'epoch': 2.35}                                                                                                           
{'loss': 0.5646, 'grad_norm': 5.7647013664245605, 'learning_rate': 9.150326797385621e-07, 'epoch': 2.45}                                                                                                           
{'loss': 0.5781, 'grad_norm': 5.839637756347656, 'learning_rate': 7.516339869281046e-07, 'epoch': 2.55}                                                                                                            
{'loss': 0.5559, 'grad_norm': 6.378639221191406, 'learning_rate': 5.882352941176471e-07, 'epoch': 2.65}                                                                                                            
{'loss': 0.4977, 'grad_norm': 5.05781364440918, 'learning_rate': 4.248366013071896e-07, 'epoch': 2.75}                                                                                                             
{'loss': 0.5702, 'grad_norm': 8.530786514282227, 'learning_rate': 2.6143790849673207e-07, 'epoch': 2.84}                                                                                                           
{'loss': 0.5634, 'grad_norm': 4.469091415405273, 'learning_rate': 9.803921568627452e-08, 'epoch': 2.94}                                                                                                            
{'eval_loss': 0.5748319029808044, 'eval_accuracy': 0.7352941176470589, 'eval_f1': 0.7272727272727273, 'eval_precision': 0.75, 'eval_recall': 0.7058823529411765, 'eval_runtime': 8.5501, 'eval_samples_per_second': 23.859, 'eval_steps_per_second': 3.041, 'epoch': 3.0}                                                                                                                                                             
{'train_runtime': 383.9084, 'train_samples_per_second': 6.377, 'train_steps_per_second': 0.797, 'train_loss': 0.6129411934247984, 'epoch': 3.0}                                                                    
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 306/306 [06:23<00:00,  1.25s/it]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 26/26 [00:08<00:00,  3.22it/s]
Fold 4 metrics: {'eval_loss': 0.5748319029808044, 'eval_accuracy': 0.7352941176470589, 'eval_f1': 0.7272727272727273, 'eval_precision': 0.75, 'eval_recall': 0.7058823529411765, 'eval_runtime': 8.4061, 'eval_samples_per_second': 24.268, 'eval_steps_per_second': 3.093, 'epoch': 3.0}

=== Fold 5 / 5 ===
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 816/816 [00:00<00:00, 6301.94 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 204/204 [00:00<00:00, 7991.54 examples/s]
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at emilyalsentzer/Bio_ClinicalBERT and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
{'loss': 0.7407, 'grad_norm': 7.275440216064453, 'learning_rate': 4.836601307189543e-06, 'epoch': 0.1}                                                                                                             
{'loss': 0.6864, 'grad_norm': 3.576312780380249, 'learning_rate': 4.673202614379085e-06, 'epoch': 0.2}                                                                                                             
{'loss': 0.7212, 'grad_norm': 5.41370964050293, 'learning_rate': 4.509803921568628e-06, 'epoch': 0.29}                                                                                                             
{'loss': 0.6851, 'grad_norm': 4.418787002563477, 'learning_rate': 4.34640522875817e-06, 'epoch': 0.39}                                                                                                             
{'loss': 0.6622, 'grad_norm': 8.098418235778809, 'learning_rate': 4.183006535947713e-06, 'epoch': 0.49}                                                                                                            
{'loss': 0.6468, 'grad_norm': 4.703812599182129, 'learning_rate': 4.019607843137255e-06, 'epoch': 0.59}                                                                                                            
{'loss': 0.6385, 'grad_norm': 5.9076128005981445, 'learning_rate': 3.856209150326798e-06, 'epoch': 0.69}                                                                                                           
{'loss': 0.6181, 'grad_norm': 4.326222896575928, 'learning_rate': 3.6928104575163404e-06, 'epoch': 0.78}                                                                                                           
{'loss': 0.6564, 'grad_norm': 4.093571662902832, 'learning_rate': 3.529411764705883e-06, 'epoch': 0.88}                                                                                                            
{'loss': 0.6114, 'grad_norm': 3.5349013805389404, 'learning_rate': 3.3660130718954253e-06, 'epoch': 0.98}                                                                                                          
{'eval_loss': 0.6203426122665405, 'eval_accuracy': 0.7058823529411765, 'eval_f1': 0.6907216494845361, 'eval_precision': 0.7282608695652174, 'eval_recall': 0.6568627450980392, 'eval_runtime': 8.6296, 'eval_samples_per_second': 23.64, 'eval_steps_per_second': 3.013, 'epoch': 1.0}                                                                                                                                                
{'loss': 0.6346, 'grad_norm': 4.114821910858154, 'learning_rate': 3.2026143790849674e-06, 'epoch': 1.08}                                                                                                           
{'loss': 0.5681, 'grad_norm': 6.203597068786621, 'learning_rate': 3.03921568627451e-06, 'epoch': 1.18}                                                                                                             
{'loss': 0.609, 'grad_norm': 4.690674781799316, 'learning_rate': 2.8758169934640523e-06, 'epoch': 1.27}                                                                                                            
{'loss': 0.5858, 'grad_norm': 5.201934814453125, 'learning_rate': 2.7124183006535947e-06, 'epoch': 1.37}                                                                                                           
{'loss': 0.6179, 'grad_norm': 6.378772735595703, 'learning_rate': 2.549019607843137e-06, 'epoch': 1.47}                                                                                                            
{'loss': 0.6127, 'grad_norm': 7.908201217651367, 'learning_rate': 2.38562091503268e-06, 'epoch': 1.57}                                                                                                             
{'loss': 0.6047, 'grad_norm': 4.54854154586792, 'learning_rate': 2.222222222222222e-06, 'epoch': 1.67}                                                                                                             
{'loss': 0.59, 'grad_norm': 3.822951555252075, 'learning_rate': 2.058823529411765e-06, 'epoch': 1.76}                                                                                                              
{'loss': 0.5799, 'grad_norm': 6.052187442779541, 'learning_rate': 1.8954248366013072e-06, 'epoch': 1.86}                                                                                                           
{'loss': 0.5529, 'grad_norm': 5.477267265319824, 'learning_rate': 1.7320261437908499e-06, 'epoch': 1.96}                                                                                                           
{'eval_loss': 0.5875090956687927, 'eval_accuracy': 0.7303921568627451, 'eval_f1': 0.729064039408867, 'eval_precision': 0.7326732673267327, 'eval_recall': 0.7254901960784313, 'eval_runtime': 8.6086, 'eval_samples_per_second': 23.697, 'eval_steps_per_second': 3.02, 'epoch': 2.0}                                                                                                                                                 
{'loss': 0.5577, 'grad_norm': 6.20262336730957, 'learning_rate': 1.5686274509803923e-06, 'epoch': 2.06}                                                                                                            
{'loss': 0.5678, 'grad_norm': 6.735650539398193, 'learning_rate': 1.4052287581699348e-06, 'epoch': 2.16}                                                                                                           
{'loss': 0.5678, 'grad_norm': 3.600630760192871, 'learning_rate': 1.2418300653594772e-06, 'epoch': 2.25}                                                                                                           
{'loss': 0.5868, 'grad_norm': 7.054193019866943, 'learning_rate': 1.0784313725490197e-06, 'epoch': 2.35}                                                                                                           
{'loss': 0.5617, 'grad_norm': 4.698666572570801, 'learning_rate': 9.150326797385621e-07, 'epoch': 2.45}                                                                                                            
{'loss': 0.589, 'grad_norm': 4.330225944519043, 'learning_rate': 7.516339869281046e-07, 'epoch': 2.55}                                                                                                             
{'loss': 0.5593, 'grad_norm': 4.791311264038086, 'learning_rate': 5.882352941176471e-07, 'epoch': 2.65}                                                                                                            
{'loss': 0.5182, 'grad_norm': 4.761048793792725, 'learning_rate': 4.248366013071896e-07, 'epoch': 2.75}                                                                                                            
{'loss': 0.5567, 'grad_norm': 8.709695816040039, 'learning_rate': 2.6143790849673207e-07, 'epoch': 2.84}                                                                                                           
{'loss': 0.5293, 'grad_norm': 3.9157180786132812, 'learning_rate': 9.803921568627452e-08, 'epoch': 2.94}                                                                                                           
{'eval_loss': 0.5767661929130554, 'eval_accuracy': 0.7058823529411765, 'eval_f1': 0.6842105263157895, 'eval_precision': 0.7386363636363636, 'eval_recall': 0.6372549019607843, 'eval_runtime': 8.1957, 'eval_samples_per_second': 24.891, 'eval_steps_per_second': 3.172, 'epoch': 3.0}                                                                                                                                               
{'train_runtime': 385.7048, 'train_samples_per_second': 6.347, 'train_steps_per_second': 0.793, 'train_loss': 0.6079462291368471, 'epoch': 3.0}                                                                    
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 306/306 [06:25<00:00,  1.26s/it]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 26/26 [00:07<00:00,  3.31it/s]
Fold 5 metrics: {'eval_loss': 0.5767661929130554, 'eval_accuracy': 0.7058823529411765, 'eval_f1': 0.6842105263157895, 'eval_precision': 0.7386363636363636, 'eval_recall': 0.6372549019607843, 'eval_runtime': 8.1708, 'eval_samples_per_second': 24.967, 'eval_steps_per_second': 3.182, 'epoch': 3.0}

--- SUMMARY ---
Mean:
 eval_loss                   0.584146
eval_accuracy               0.708824
eval_f1                     0.701064
eval_precision              0.722025
eval_recall                 0.682353
eval_runtime                8.217300
eval_samples_per_second    24.829000
eval_steps_per_second       3.164600
dtype: float64
Std:
 eval_loss                  0.028593
eval_accuracy              0.034244
eval_f1                    0.031547
eval_precision             0.042793
eval_recall                0.033678
eval_runtime               0.105673
eval_samples_per_second    0.314020
eval_steps_per_second      0.040085
dtype: float64


########## config_advanced ##########

=== Fold 1 / 5 ===
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 816/816 [00:00<00:00, 6508.31 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 204/204 [00:00<00:00, 7498.36 examples/s]
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at emilyalsentzer/Bio_ClinicalBERT and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
{'loss': 0.747, 'grad_norm': 4.838779926300049, 'learning_rate': 1.153846153846154e-06, 'epoch': 0.2}                                                                                                              
{'loss': 0.7198, 'grad_norm': 3.7458441257476807, 'learning_rate': 2.307692307692308e-06, 'epoch': 0.39}                                                                                                           
{'loss': 0.7128, 'grad_norm': 6.110123157501221, 'learning_rate': 2.947598253275109e-06, 'epoch': 0.59}                                                                                                            
{'loss': 0.6959, 'grad_norm': 2.2874929904937744, 'learning_rate': 2.8165938864628822e-06, 'epoch': 0.78}                                                                                                          
{'loss': 0.7161, 'grad_norm': 4.080483913421631, 'learning_rate': 2.685589519650655e-06, 'epoch': 0.98}                                                                                                            
{'eval_loss': 0.6731498837471008, 'eval_accuracy': 0.5392156862745098, 'eval_f1': 0.6758620689655173, 'eval_precision': 0.5212765957446809, 'eval_recall': 0.9607843137254902, 'eval_runtime': 8.1829, 'eval_samples_per_second': 24.93, 'eval_steps_per_second': 3.177, 'epoch': 1.0}                                                                                                                                                
{'loss': 0.6734, 'grad_norm': 3.3672540187835693, 'learning_rate': 2.554585152838428e-06, 'epoch': 1.18}                                                                                                           
{'loss': 0.6872, 'grad_norm': 2.5804600715637207, 'learning_rate': 2.4235807860262008e-06, 'epoch': 1.37}                                                                                                          
{'loss': 0.6735, 'grad_norm': 2.9233412742614746, 'learning_rate': 2.292576419213974e-06, 'epoch': 1.57}                                                                                                           
{'loss': 0.6619, 'grad_norm': 2.583360195159912, 'learning_rate': 2.161572052401747e-06, 'epoch': 1.76}                                                                                                            
{'loss': 0.6333, 'grad_norm': 2.3330416679382324, 'learning_rate': 2.0305676855895198e-06, 'epoch': 1.96}                                                                                                          
{'eval_loss': 0.6442933082580566, 'eval_accuracy': 0.6470588235294118, 'eval_f1': 0.6756756756756757, 'eval_precision': 0.625, 'eval_recall': 0.7352941176470589, 'eval_runtime': 8.1817, 'eval_samples_per_second': 24.934, 'eval_steps_per_second': 3.178, 'epoch': 2.0}                                                                                                                                                            
{'loss': 0.6526, 'grad_norm': 3.1962227821350098, 'learning_rate': 1.8995633187772928e-06, 'epoch': 2.16}                                                                                                          
{'loss': 0.6553, 'grad_norm': 4.0445475578308105, 'learning_rate': 1.7685589519650657e-06, 'epoch': 2.35}                                                                                                          
{'loss': 0.6317, 'grad_norm': 3.8457465171813965, 'learning_rate': 1.6375545851528385e-06, 'epoch': 2.55}                                                                                                          
{'loss': 0.6518, 'grad_norm': 4.725681304931641, 'learning_rate': 1.5065502183406112e-06, 'epoch': 2.75}                                                                                                           
{'loss': 0.6258, 'grad_norm': 6.05428409576416, 'learning_rate': 1.3755458515283844e-06, 'epoch': 2.94}                                                                                                            
{'eval_loss': 0.6176280379295349, 'eval_accuracy': 0.696078431372549, 'eval_f1': 0.6990291262135923, 'eval_precision': 0.6923076923076923, 'eval_recall': 0.7058823529411765, 'eval_runtime': 8.1892, 'eval_samples_per_second': 24.911, 'eval_steps_per_second': 3.175, 'epoch': 3.0}                                                                                                                                                
{'loss': 0.617, 'grad_norm': 5.51137113571167, 'learning_rate': 1.2445414847161573e-06, 'epoch': 3.14}                                                                                                             
{'loss': 0.6261, 'grad_norm': 6.5803608894348145, 'learning_rate': 1.1135371179039301e-06, 'epoch': 3.33}                                                                                                          
{'loss': 0.6114, 'grad_norm': 3.584775686264038, 'learning_rate': 9.82532751091703e-07, 'epoch': 3.53}                                                                                                             
{'loss': 0.6126, 'grad_norm': 3.876573085784912, 'learning_rate': 8.515283842794759e-07, 'epoch': 3.73}                                                                                                            
{'loss': 0.6264, 'grad_norm': 3.409749984741211, 'learning_rate': 7.205240174672489e-07, 'epoch': 3.92}                                                                                                            
{'eval_loss': 0.6008004546165466, 'eval_accuracy': 0.7107843137254902, 'eval_f1': 0.7121951219512195, 'eval_precision': 0.7087378640776699, 'eval_recall': 0.7156862745098039, 'eval_runtime': 8.1914, 'eval_samples_per_second': 24.904, 'eval_steps_per_second': 3.174, 'epoch': 4.0}                                                                                                                                               
{'loss': 0.5941, 'grad_norm': 3.2961339950561523, 'learning_rate': 5.895196506550219e-07, 'epoch': 4.12}                                                                                                           
{'loss': 0.6246, 'grad_norm': 6.448966979980469, 'learning_rate': 4.5851528384279476e-07, 'epoch': 4.31}                                                                                                           
{'loss': 0.5856, 'grad_norm': 3.834684371948242, 'learning_rate': 3.275109170305677e-07, 'epoch': 4.51}                                                                                                            
{'loss': 0.6279, 'grad_norm': 3.501538038253784, 'learning_rate': 1.9650655021834062e-07, 'epoch': 4.71}                                                                                                           
{'loss': 0.6298, 'grad_norm': 4.000682830810547, 'learning_rate': 6.550218340611354e-08, 'epoch': 4.9}                                                                                                             
{'eval_loss': 0.5966908931732178, 'eval_accuracy': 0.7303921568627451, 'eval_f1': 0.7342995169082126, 'eval_precision': 0.7238095238095238, 'eval_recall': 0.7450980392156863, 'eval_runtime': 8.2079, 'eval_samples_per_second': 24.854, 'eval_steps_per_second': 3.168, 'epoch': 5.0}                                                                                                                                               
{'train_runtime': 609.5685, 'train_samples_per_second': 6.693, 'train_steps_per_second': 0.418, 'train_loss': 0.6502846269046559, 'epoch': 5.0}                                                                    
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 255/255 [10:09<00:00,  2.39s/it]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 26/26 [00:07<00:00,  3.31it/s]
Fold 1 metrics: {'eval_loss': 0.5966908931732178, 'eval_accuracy': 0.7303921568627451, 'eval_f1': 0.7342995169082126, 'eval_precision': 0.7238095238095238, 'eval_recall': 0.7450980392156863, 'eval_runtime': 8.1762, 'eval_samples_per_second': 24.95, 'eval_steps_per_second': 3.18, 'epoch': 5.0}

=== Fold 2 / 5 ===
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 816/816 [00:00<00:00, 6742.14 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 204/204 [00:00<00:00, 8114.93 examples/s]
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at emilyalsentzer/Bio_ClinicalBERT and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
{'loss': 0.6949, 'grad_norm': 3.054344415664673, 'learning_rate': 1.153846153846154e-06, 'epoch': 0.2}                                                                                                             
{'loss': 0.6957, 'grad_norm': 4.431373119354248, 'learning_rate': 2.307692307692308e-06, 'epoch': 0.39}                                                                                                            
{'loss': 0.6883, 'grad_norm': 2.2611374855041504, 'learning_rate': 2.947598253275109e-06, 'epoch': 0.59}                                                                                                           
{'loss': 0.6798, 'grad_norm': 4.601470947265625, 'learning_rate': 2.8165938864628822e-06, 'epoch': 0.78}                                                                                                           
{'loss': 0.6812, 'grad_norm': 4.784141540527344, 'learning_rate': 2.685589519650655e-06, 'epoch': 0.98}                                                                                                            
{'eval_loss': 0.6732297539710999, 'eval_accuracy': 0.6470588235294118, 'eval_f1': 0.6896551724137931, 'eval_precision': 0.6153846153846154, 'eval_recall': 0.7843137254901961, 'eval_runtime': 9.6811, 'eval_samples_per_second': 21.072, 'eval_steps_per_second': 2.686, 'epoch': 1.0}                                                                                                                                               
{'loss': 0.6664, 'grad_norm': 5.8142781257629395, 'learning_rate': 2.554585152838428e-06, 'epoch': 1.18}                                                                                                           
{'loss': 0.6436, 'grad_norm': 5.313536643981934, 'learning_rate': 2.4235807860262008e-06, 'epoch': 1.37}                                                                                                           
{'loss': 0.6458, 'grad_norm': 5.6132330894470215, 'learning_rate': 2.292576419213974e-06, 'epoch': 1.57}                                                                                                           
{'loss': 0.6427, 'grad_norm': 3.5187036991119385, 'learning_rate': 2.161572052401747e-06, 'epoch': 1.76}                                                                                                           
{'loss': 0.6356, 'grad_norm': 3.53291392326355, 'learning_rate': 2.0305676855895198e-06, 'epoch': 1.96}                                                                                                            
{'eval_loss': 0.6547300815582275, 'eval_accuracy': 0.6519607843137255, 'eval_f1': 0.6697674418604651, 'eval_precision': 0.6371681415929203, 'eval_recall': 0.7058823529411765, 'eval_runtime': 8.2007, 'eval_samples_per_second': 24.876, 'eval_steps_per_second': 3.17, 'epoch': 2.0}                                                                                                                                                
{'loss': 0.6179, 'grad_norm': 4.653458118438721, 'learning_rate': 1.8995633187772928e-06, 'epoch': 2.16}                                                                                                           
{'loss': 0.6248, 'grad_norm': 4.8401360511779785, 'learning_rate': 1.7685589519650657e-06, 'epoch': 2.35}                                                                                                          
{'loss': 0.6307, 'grad_norm': 5.774170875549316, 'learning_rate': 1.6375545851528385e-06, 'epoch': 2.55}                                                                                                           
{'loss': 0.6239, 'grad_norm': 2.9207377433776855, 'learning_rate': 1.5065502183406112e-06, 'epoch': 2.75}                                                                                                          
{'loss': 0.6012, 'grad_norm': 2.6670145988464355, 'learning_rate': 1.3755458515283844e-06, 'epoch': 2.94}                                                                                                          
{'eval_loss': 0.6430061459541321, 'eval_accuracy': 0.6470588235294118, 'eval_f1': 0.6666666666666666, 'eval_precision': 0.631578947368421, 'eval_recall': 0.7058823529411765, 'eval_runtime': 8.1945, 'eval_samples_per_second': 24.895, 'eval_steps_per_second': 3.173, 'epoch': 3.0}                                                                                                                                                
{'loss': 0.6076, 'grad_norm': 2.9811580181121826, 'learning_rate': 1.2445414847161573e-06, 'epoch': 3.14}                                                                                                          
{'loss': 0.5963, 'grad_norm': 4.058131694793701, 'learning_rate': 1.1135371179039301e-06, 'epoch': 3.33}                                                                                                           
{'loss': 0.5752, 'grad_norm': 6.015590190887451, 'learning_rate': 9.82532751091703e-07, 'epoch': 3.53}                                                                                                             
{'loss': 0.6132, 'grad_norm': 5.794313430786133, 'learning_rate': 8.515283842794759e-07, 'epoch': 3.73}                                                                                                            
{'loss': 0.5983, 'grad_norm': 3.777545690536499, 'learning_rate': 7.205240174672489e-07, 'epoch': 3.92}                                                                                                            
{'eval_loss': 0.6366034150123596, 'eval_accuracy': 0.6470588235294118, 'eval_f1': 0.6727272727272727, 'eval_precision': 0.6271186440677966, 'eval_recall': 0.7254901960784313, 'eval_runtime': 8.2236, 'eval_samples_per_second': 24.807, 'eval_steps_per_second': 3.162, 'epoch': 4.0}                                                                                                                                               
{'loss': 0.5748, 'grad_norm': 4.053283214569092, 'learning_rate': 5.895196506550219e-07, 'epoch': 4.12}                                                                                                            
{'loss': 0.571, 'grad_norm': 6.826657772064209, 'learning_rate': 4.5851528384279476e-07, 'epoch': 4.31}                                                                                                            
{'loss': 0.5717, 'grad_norm': 4.401938438415527, 'learning_rate': 3.275109170305677e-07, 'epoch': 4.51}                                                                                                            
{'loss': 0.5843, 'grad_norm': 4.451901435852051, 'learning_rate': 1.9650655021834062e-07, 'epoch': 4.71}                                                                                                           
{'loss': 0.6176, 'grad_norm': 3.800605535507202, 'learning_rate': 6.550218340611354e-08, 'epoch': 4.9}                                                                                                             
{'eval_loss': 0.63321453332901, 'eval_accuracy': 0.6519607843137255, 'eval_f1': 0.6757990867579908, 'eval_precision': 0.6324786324786325, 'eval_recall': 0.7254901960784313, 'eval_runtime': 8.2062, 'eval_samples_per_second': 24.859, 'eval_steps_per_second': 3.168, 'epoch': 5.0}                                                                                                                                                 
{'train_runtime': 2702.5606, 'train_samples_per_second': 1.51, 'train_steps_per_second': 0.094, 'train_loss': 0.6264516091814228, 'epoch': 5.0}                                                                    
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 255/255 [45:02<00:00, 10.60s/it]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 26/26 [00:07<00:00,  3.31it/s]
Fold 2 metrics: {'eval_loss': 0.63321453332901, 'eval_accuracy': 0.6519607843137255, 'eval_f1': 0.6757990867579908, 'eval_precision': 0.6324786324786325, 'eval_recall': 0.7254901960784313, 'eval_runtime': 8.1707, 'eval_samples_per_second': 24.967, 'eval_steps_per_second': 3.182, 'epoch': 5.0}

=== Fold 3 / 5 ===
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 816/816 [00:00<00:00, 3541.42 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 204/204 [00:00<00:00, 6854.59 examples/s]
'(ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')), '(Request ID: be241a3c-3f24-498f-846f-ed4e76a68a16)')' thrown while requesting HEAD https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT/resolve/main/config.json
Retrying in 1s [Retry 1/5].
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at emilyalsentzer/Bio_ClinicalBERT and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
{'loss': 0.689, 'grad_norm': 2.220276117324829, 'learning_rate': 1.153846153846154e-06, 'epoch': 0.2}                                                                                                              
{'loss': 0.675, 'grad_norm': 2.3806445598602295, 'learning_rate': 2.307692307692308e-06, 'epoch': 0.39}                                                                                                            
{'loss': 0.689, 'grad_norm': 3.681561231613159, 'learning_rate': 2.947598253275109e-06, 'epoch': 0.59}                                                                                                             
{'loss': 0.6725, 'grad_norm': 5.038473129272461, 'learning_rate': 2.8165938864628822e-06, 'epoch': 0.78}                                                                                                           
{'loss': 0.6741, 'grad_norm': 5.521280765533447, 'learning_rate': 2.685589519650655e-06, 'epoch': 0.98}                                                                                                            
{'eval_loss': 0.6637473106384277, 'eval_accuracy': 0.6911764705882353, 'eval_f1': 0.7224669603524229, 'eval_precision': 0.656, 'eval_recall': 0.803921568627451, 'eval_runtime': 8.1878, 'eval_samples_per_second': 24.915, 'eval_steps_per_second': 3.175, 'epoch': 1.0}                                                                                                                                                             
{'loss': 0.6691, 'grad_norm': 2.203174114227295, 'learning_rate': 2.554585152838428e-06, 'epoch': 1.18}                                                                                                            
{'loss': 0.6507, 'grad_norm': 3.7673847675323486, 'learning_rate': 2.4235807860262008e-06, 'epoch': 1.37}                                                                                                          
{'loss': 0.6576, 'grad_norm': 7.093120098114014, 'learning_rate': 2.292576419213974e-06, 'epoch': 1.57}                                                                                                            
{'loss': 0.6533, 'grad_norm': 4.131730079650879, 'learning_rate': 2.161572052401747e-06, 'epoch': 1.76}                                                                                                            
{'loss': 0.6541, 'grad_norm': 2.757835865020752, 'learning_rate': 2.0305676855895198e-06, 'epoch': 1.96}                                                                                                           
{'eval_loss': 0.6351597309112549, 'eval_accuracy': 0.7058823529411765, 'eval_f1': 0.7087378640776699, 'eval_precision': 0.7019230769230769, 'eval_recall': 0.7156862745098039, 'eval_runtime': 8.1904, 'eval_samples_per_second': 24.907, 'eval_steps_per_second': 3.174, 'epoch': 2.0}                                                                                                                                               
{'loss': 0.6349, 'grad_norm': 6.210569381713867, 'learning_rate': 1.8995633187772928e-06, 'epoch': 2.16}                                                                                                           
{'loss': 0.6471, 'grad_norm': 3.569507598876953, 'learning_rate': 1.7685589519650657e-06, 'epoch': 2.35}                                                                                                           
{'loss': 0.6633, 'grad_norm': 4.385367393493652, 'learning_rate': 1.6375545851528385e-06, 'epoch': 2.55}                                                                                                           
{'loss': 0.6313, 'grad_norm': 3.1536059379577637, 'learning_rate': 1.5065502183406112e-06, 'epoch': 2.75}                                                                                                          
{'loss': 0.6298, 'grad_norm': 4.305483818054199, 'learning_rate': 1.3755458515283844e-06, 'epoch': 2.94}                                                                                                           
{'eval_loss': 0.6153010725975037, 'eval_accuracy': 0.7058823529411765, 'eval_f1': 0.7169811320754716, 'eval_precision': 0.6909090909090909, 'eval_recall': 0.7450980392156863, 'eval_runtime': 8.2313, 'eval_samples_per_second': 24.784, 'eval_steps_per_second': 3.159, 'epoch': 3.0}                                                                                                                                               
{'loss': 0.6189, 'grad_norm': 3.1084561347961426, 'learning_rate': 1.2445414847161573e-06, 'epoch': 3.14}                                                                                                          
{'loss': 0.5751, 'grad_norm': 4.868940830230713, 'learning_rate': 1.1135371179039301e-06, 'epoch': 3.33}                                                                                                           
{'loss': 0.6481, 'grad_norm': 4.60409688949585, 'learning_rate': 9.82532751091703e-07, 'epoch': 3.53}                                                                                                              
{'loss': 0.6067, 'grad_norm': 5.453815937042236, 'learning_rate': 8.515283842794759e-07, 'epoch': 3.73}                                                                                                            
{'loss': 0.6164, 'grad_norm': 3.82454776763916, 'learning_rate': 7.205240174672489e-07, 'epoch': 3.92}                                                                                                             
{'eval_loss': 0.6031574010848999, 'eval_accuracy': 0.7156862745098039, 'eval_f1': 0.7238095238095238, 'eval_precision': 0.7037037037037037, 'eval_recall': 0.7450980392156863, 'eval_runtime': 8.193, 'eval_samples_per_second': 24.899, 'eval_steps_per_second': 3.173, 'epoch': 4.0}                                                                                                                                                
{'loss': 0.6065, 'grad_norm': 3.885420799255371, 'learning_rate': 5.895196506550219e-07, 'epoch': 4.12}                                                                                                            
{'loss': 0.5927, 'grad_norm': 2.9753994941711426, 'learning_rate': 4.5851528384279476e-07, 'epoch': 4.31}                                                                                                          
{'loss': 0.6, 'grad_norm': 3.7761242389678955, 'learning_rate': 3.275109170305677e-07, 'epoch': 4.51}                                                                                                              
{'loss': 0.6281, 'grad_norm': 3.9986321926116943, 'learning_rate': 1.9650655021834062e-07, 'epoch': 4.71}                                                                                                          
{'loss': 0.625, 'grad_norm': 3.3126251697540283, 'learning_rate': 6.550218340611354e-08, 'epoch': 4.9}                                                                                                             
{'eval_loss': 0.5988812446594238, 'eval_accuracy': 0.7156862745098039, 'eval_f1': 0.7264150943396226, 'eval_precision': 0.7, 'eval_recall': 0.7549019607843137, 'eval_runtime': 8.1939, 'eval_samples_per_second': 24.896, 'eval_steps_per_second': 3.173, 'epoch': 5.0}                                                                                                                                                              
{'train_runtime': 598.958, 'train_samples_per_second': 6.812, 'train_steps_per_second': 0.426, 'train_loss': 0.6390871141471115, 'epoch': 5.0}                                                                     
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 255/255 [09:58<00:00,  2.35s/it]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 26/26 [00:07<00:00,  3.31it/s]
Fold 3 metrics: {'eval_loss': 0.5988812446594238, 'eval_accuracy': 0.7156862745098039, 'eval_f1': 0.7264150943396226, 'eval_precision': 0.7, 'eval_recall': 0.7549019607843137, 'eval_runtime': 8.1726, 'eval_samples_per_second': 24.961, 'eval_steps_per_second': 3.181, 'epoch': 5.0}

=== Fold 4 / 5 ===
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 816/816 [00:00<00:00, 6355.42 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 204/204 [00:00<00:00, 7542.05 examples/s]
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at emilyalsentzer/Bio_ClinicalBERT and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
{'loss': 0.6888, 'grad_norm': 7.319414138793945, 'learning_rate': 1.153846153846154e-06, 'epoch': 0.2}                                                                                                             
{'loss': 0.6954, 'grad_norm': 3.2575368881225586, 'learning_rate': 2.307692307692308e-06, 'epoch': 0.39}                                                                                                           
{'loss': 0.6854, 'grad_norm': 2.779910087585449, 'learning_rate': 2.947598253275109e-06, 'epoch': 0.59}                                                                                                            
{'loss': 0.6936, 'grad_norm': 2.6282970905303955, 'learning_rate': 2.8165938864628822e-06, 'epoch': 0.78}                                                                                                          
{'loss': 0.6899, 'grad_norm': 2.6524596214294434, 'learning_rate': 2.685589519650655e-06, 'epoch': 0.98}                                                                                                           
{'eval_loss': 0.6643292903900146, 'eval_accuracy': 0.696078431372549, 'eval_f1': 0.7047619047619048, 'eval_precision': 0.6851851851851852, 'eval_recall': 0.7254901960784313, 'eval_runtime': 35.8614, 'eval_samples_per_second': 5.689, 'eval_steps_per_second': 0.725, 'epoch': 1.0}                                                                                                                                                
{'loss': 0.6752, 'grad_norm': 2.4689571857452393, 'learning_rate': 2.554585152838428e-06, 'epoch': 1.18}                                                                                                           
{'loss': 0.6539, 'grad_norm': 4.455059051513672, 'learning_rate': 2.4235807860262008e-06, 'epoch': 1.37}                                                                                                           
{'loss': 0.6367, 'grad_norm': 4.554679870605469, 'learning_rate': 2.292576419213974e-06, 'epoch': 1.57}                                                                                                            
{'loss': 0.6609, 'grad_norm': 3.5090441703796387, 'learning_rate': 2.161572052401747e-06, 'epoch': 1.76}                                                                                                           
{'loss': 0.6406, 'grad_norm': 3.4008772373199463, 'learning_rate': 2.0305676855895198e-06, 'epoch': 1.96}                                                                                                          
{'eval_loss': 0.6449699997901917, 'eval_accuracy': 0.6764705882352942, 'eval_f1': 0.6944444444444444, 'eval_precision': 0.6578947368421053, 'eval_recall': 0.7352941176470589, 'eval_runtime': 8.2032, 'eval_samples_per_second': 24.868, 'eval_steps_per_second': 3.169, 'epoch': 2.0}                                                                                                                                               
{'loss': 0.6417, 'grad_norm': 5.571811199188232, 'learning_rate': 1.8995633187772928e-06, 'epoch': 2.16}                                                                                                           
{'loss': 0.6488, 'grad_norm': 4.872065544128418, 'learning_rate': 1.7685589519650657e-06, 'epoch': 2.35}                                                                                                           
{'loss': 0.6317, 'grad_norm': 2.6064462661743164, 'learning_rate': 1.6375545851528385e-06, 'epoch': 2.55}                                                                                                          
{'loss': 0.6056, 'grad_norm': 4.167960166931152, 'learning_rate': 1.5065502183406112e-06, 'epoch': 2.75}                                                                                                           
{'loss': 0.6278, 'grad_norm': 4.146958351135254, 'learning_rate': 1.3755458515283844e-06, 'epoch': 2.94}                                                                                                           
{'eval_loss': 0.6229619979858398, 'eval_accuracy': 0.7156862745098039, 'eval_f1': 0.7070707070707071, 'eval_precision': 0.7291666666666666, 'eval_recall': 0.6862745098039216, 'eval_runtime': 8.5157, 'eval_samples_per_second': 23.956, 'eval_steps_per_second': 3.053, 'epoch': 3.0}                                                                                                                                               
{'loss': 0.6185, 'grad_norm': 2.442883253097534, 'learning_rate': 1.2445414847161573e-06, 'epoch': 3.14}                                                                                                           
{'loss': 0.6126, 'grad_norm': 3.79654860496521, 'learning_rate': 1.1135371179039301e-06, 'epoch': 3.33}                                                                                                            
{'loss': 0.6001, 'grad_norm': 3.3337395191192627, 'learning_rate': 9.82532751091703e-07, 'epoch': 3.53}                                                                                                            
{'loss': 0.6144, 'grad_norm': 4.189702033996582, 'learning_rate': 8.515283842794759e-07, 'epoch': 3.73}                                                                                                            
{'loss': 0.6135, 'grad_norm': 4.1502156257629395, 'learning_rate': 7.205240174672489e-07, 'epoch': 3.92}                                                                                                           
{'eval_loss': 0.6145448684692383, 'eval_accuracy': 0.7205882352941176, 'eval_f1': 0.7246376811594203, 'eval_precision': 0.7142857142857143, 'eval_recall': 0.7352941176470589, 'eval_runtime': 8.6728, 'eval_samples_per_second': 23.522, 'eval_steps_per_second': 2.998, 'epoch': 4.0}                                                                                                                                               
{'loss': 0.6061, 'grad_norm': 4.685867786407471, 'learning_rate': 5.895196506550219e-07, 'epoch': 4.12}                                                                                                            
{'loss': 0.5698, 'grad_norm': 4.7024712562561035, 'learning_rate': 4.5851528384279476e-07, 'epoch': 4.31}                                                                                                          
{'loss': 0.6071, 'grad_norm': 3.63592791557312, 'learning_rate': 3.275109170305677e-07, 'epoch': 4.51}                                                                                                             
{'loss': 0.5932, 'grad_norm': 3.754626750946045, 'learning_rate': 1.9650655021834062e-07, 'epoch': 4.71}                                                                                                           
{'loss': 0.5918, 'grad_norm': 3.7873005867004395, 'learning_rate': 6.550218340611354e-08, 'epoch': 4.9}                                                                                                            
{'eval_loss': 0.6104689836502075, 'eval_accuracy': 0.7205882352941176, 'eval_f1': 0.7219512195121951, 'eval_precision': 0.7184466019417476, 'eval_recall': 0.7254901960784313, 'eval_runtime': 8.6699, 'eval_samples_per_second': 23.53, 'eval_steps_per_second': 2.999, 'epoch': 5.0}                                                                                                                                                
{'train_runtime': 667.3721, 'train_samples_per_second': 6.114, 'train_steps_per_second': 0.382, 'train_loss': 0.6355254453771254, 'epoch': 5.0}                                                                    
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 255/255 [11:07<00:00,  2.62s/it]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 26/26 [00:08<00:00,  3.13it/s]
Fold 4 metrics: {'eval_loss': 0.6104689836502075, 'eval_accuracy': 0.7205882352941176, 'eval_f1': 0.7219512195121951, 'eval_precision': 0.7184466019417476, 'eval_recall': 0.7254901960784313, 'eval_runtime': 8.6477, 'eval_samples_per_second': 23.59, 'eval_steps_per_second': 3.007, 'epoch': 5.0}

=== Fold 5 / 5 ===
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 816/816 [00:00<00:00, 6153.89 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 204/204 [00:00<00:00, 7660.90 examples/s]
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at emilyalsentzer/Bio_ClinicalBERT and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
{'loss': 0.6786, 'grad_norm': 2.228025197982788, 'learning_rate': 1.153846153846154e-06, 'epoch': 0.2}                                                                                                             
{'loss': 0.7037, 'grad_norm': 2.673931121826172, 'learning_rate': 2.307692307692308e-06, 'epoch': 0.39}                                                                                                            
{'loss': 0.6869, 'grad_norm': 5.627331256866455, 'learning_rate': 2.947598253275109e-06, 'epoch': 0.59}                                                                                                            
{'loss': 0.6707, 'grad_norm': 3.6336982250213623, 'learning_rate': 2.8165938864628822e-06, 'epoch': 0.78}                                                                                                          
{'loss': 0.678, 'grad_norm': 4.08278226852417, 'learning_rate': 2.685589519650655e-06, 'epoch': 0.98}                                                                                                              
{'eval_loss': 0.6665748357772827, 'eval_accuracy': 0.6225490196078431, 'eval_f1': 0.6577777777777778, 'eval_precision': 0.6016260162601627, 'eval_recall': 0.7254901960784313, 'eval_runtime': 8.6764, 'eval_samples_per_second': 23.512, 'eval_steps_per_second': 2.997, 'epoch': 1.0}                                                                                                                                               
{'loss': 0.6472, 'grad_norm': 3.144113540649414, 'learning_rate': 2.554585152838428e-06, 'epoch': 1.18}                                                                                                            
{'loss': 0.6655, 'grad_norm': 3.563610076904297, 'learning_rate': 2.4235807860262008e-06, 'epoch': 1.37}                                                                                                           
{'loss': 0.6704, 'grad_norm': 7.48582124710083, 'learning_rate': 2.292576419213974e-06, 'epoch': 1.57}                                                                                                             
{'loss': 0.6422, 'grad_norm': 2.942247152328491, 'learning_rate': 2.161572052401747e-06, 'epoch': 1.76}                                                                                                            
{'loss': 0.6319, 'grad_norm': 3.5592875480651855, 'learning_rate': 2.0305676855895198e-06, 'epoch': 1.96}                                                                                                          
{'eval_loss': 0.6407867074012756, 'eval_accuracy': 0.6470588235294118, 'eval_f1': 0.6635514018691588, 'eval_precision': 0.6339285714285714, 'eval_recall': 0.696078431372549, 'eval_runtime': 8.6701, 'eval_samples_per_second': 23.529, 'eval_steps_per_second': 2.999, 'epoch': 2.0}                                                                                                                                                
{'loss': 0.6276, 'grad_norm': 4.227666854858398, 'learning_rate': 1.8995633187772928e-06, 'epoch': 2.16}                                                                                                           
{'loss': 0.6426, 'grad_norm': 6.199865341186523, 'learning_rate': 1.7685589519650657e-06, 'epoch': 2.35}                                                                                                           
{'loss': 0.6318, 'grad_norm': 4.546462535858154, 'learning_rate': 1.6375545851528385e-06, 'epoch': 2.55}                                                                                                           
{'loss': 0.6128, 'grad_norm': 5.190675735473633, 'learning_rate': 1.5065502183406112e-06, 'epoch': 2.75}                                                                                                           
{'loss': 0.6132, 'grad_norm': 3.1717946529388428, 'learning_rate': 1.3755458515283844e-06, 'epoch': 2.94}                                                                                                          
{'eval_loss': 0.6198798418045044, 'eval_accuracy': 0.6911764705882353, 'eval_f1': 0.6834170854271356, 'eval_precision': 0.7010309278350515, 'eval_recall': 0.6666666666666666, 'eval_runtime': 9.4975, 'eval_samples_per_second': 21.479, 'eval_steps_per_second': 2.738, 'epoch': 3.0}                                                                                                                                               
{'loss': 0.6415, 'grad_norm': 8.778083801269531, 'learning_rate': 1.2445414847161573e-06, 'epoch': 3.14}                                                                                                           
{'loss': 0.5856, 'grad_norm': 4.235951900482178, 'learning_rate': 1.1135371179039301e-06, 'epoch': 3.33}                                                                                                           
{'loss': 0.6451, 'grad_norm': 3.64151930809021, 'learning_rate': 9.82532751091703e-07, 'epoch': 3.53}                                                                                                              
{'loss': 0.6089, 'grad_norm': 3.779498815536499, 'learning_rate': 8.515283842794759e-07, 'epoch': 3.73}                                                                                                            
{'loss': 0.5947, 'grad_norm': 2.8549089431762695, 'learning_rate': 7.205240174672489e-07, 'epoch': 3.92}                                                                                                           
{'eval_loss': 0.606634795665741, 'eval_accuracy': 0.7009803921568627, 'eval_f1': 0.6871794871794872, 'eval_precision': 0.7204301075268817, 'eval_recall': 0.6568627450980392, 'eval_runtime': 8.6446, 'eval_samples_per_second': 23.599, 'eval_steps_per_second': 3.008, 'epoch': 4.0}                                                                                                                                                
{'loss': 0.6113, 'grad_norm': 4.134172439575195, 'learning_rate': 5.895196506550219e-07, 'epoch': 4.12}                                                                                                            
{'loss': 0.5786, 'grad_norm': 3.8280551433563232, 'learning_rate': 4.5851528384279476e-07, 'epoch': 4.31}                                                                                                          
{'loss': 0.5819, 'grad_norm': 4.305556774139404, 'learning_rate': 3.275109170305677e-07, 'epoch': 4.51}                                                                                                            
{'loss': 0.5884, 'grad_norm': 3.135591745376587, 'learning_rate': 1.9650655021834062e-07, 'epoch': 4.71}                                                                                                           
{'loss': 0.6164, 'grad_norm': 5.027393817901611, 'learning_rate': 6.550218340611354e-08, 'epoch': 4.9}                                                                                                             
{'eval_loss': 0.6037787795066833, 'eval_accuracy': 0.7058823529411765, 'eval_f1': 0.6938775510204082, 'eval_precision': 0.723404255319149, 'eval_recall': 0.6666666666666666, 'eval_runtime': 8.1896, 'eval_samples_per_second': 24.91, 'eval_steps_per_second': 3.175, 'epoch': 5.0}                                                                                                                                                 
{'train_runtime': 629.7279, 'train_samples_per_second': 6.479, 'train_steps_per_second': 0.405, 'train_loss': 0.6337822521434111, 'epoch': 5.0}                                                                    
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 255/255 [10:29<00:00,  2.47s/it]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 26/26 [00:07<00:00,  3.31it/s]
Fold 5 metrics: {'eval_loss': 0.6037787795066833, 'eval_accuracy': 0.7058823529411765, 'eval_f1': 0.6938775510204082, 'eval_precision': 0.723404255319149, 'eval_recall': 0.6666666666666666, 'eval_runtime': 8.169, 'eval_samples_per_second': 24.972, 'eval_steps_per_second': 3.183, 'epoch': 5.0}

--- SUMMARY ---
Mean:
 eval_loss                   0.608607
eval_accuracy               0.704902
eval_f1                     0.710468
eval_precision              0.699628
eval_recall                 0.723529
eval_runtime                8.267240
eval_samples_per_second    24.688000
eval_steps_per_second       3.146600
dtype: float64
Std:
 eval_loss                  0.014738
eval_accuracy              0.030886
eval_f1                    0.024654
eval_precision             0.038773
eval_recall                0.034244
eval_runtime               0.212700
eval_samples_per_second    0.613855
eval_steps_per_second      0.078047
dtype: float64

===== FINAL COMPARISON =====
            config  mean_accuracy  std_accuracy   mean_f1    std_f1  mean_precision  std_precision  mean_recall  std_recall
1  config_advanced       0.704902      0.030886  0.710468  0.024654        0.699628       0.038773     0.723529    0.034244
0    config_stable       0.708824      0.034244  0.701064  0.031547        0.722025       0.042793     0.682353    0.033678