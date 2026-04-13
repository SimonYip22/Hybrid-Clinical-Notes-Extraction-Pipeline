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
- Does not produce the final deployed model, final training uses the predefined train/validation split for checkpointing and model selection

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

#### 10.2 Evolution and Design Change

The model development pipeline evolved iteratively as understanding of validation strategies improved. These changes were incremental and reflect a transition from standard practice to a more statistically robust approach.

These initial approaches were:

- Dataset split into: Train / Validation / Test (3-way split)
- Validation set used to:
  - Monitor performance during training
  - Guide hyperparameter tuning
  - Select best model checkpoints  
- This allowed for controlled experimentation and comparability across runs 

However in the final iteration, a 5-fold cross-validation approach was adopted:

- Purpose:
  - Assess robustness across multiple data splits  
  - Quantify variance and stability of performance  
- However:
  - CV was applied only to the training subset (420 samples), not the combined train + validation data (510 samples)
  - A separate validation step was still retained during final model training 

A methodological limitation was identified in the final iteration:

- Cross-validation functionally replaces the need for a dedicated validation set  
- Optimal approach would have been to perform CV on the full training pool (train + validation = 510 samples)  
- As implemented:
  - Data utilisation was suboptimal  
  - Validation was effectively duplicated:
    - CV provided performance estimation and model selection  
    - Final validation step repeated similar evaluation  

However, this is not a critical error because:

- Performance trends and conclusions remain valid  
- Model behaviour (e.g. variance, recall bias, performance ceiling) is consistent  
- The primary issue is:
  - Redundancy and inefficiency  
  - Slight underuse of available labelled data  

The pipeline overall remains valid:

- The validation set guided training and hyperparameter selection  
- Cross-validation confirmed generalisation performance and variability  
- Combined effect is both local (per-run) and global (distributional) performance insight  

This structure is methodologically acceptable and commonly observed in exploratory model development, despite not being maximally efficient.

---

#### 10.3 First Training Run: Pipeline Sanity Check

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

#### 10.4 Second Training Run: Full Input Representation

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

#### 10.5 Third Training Run: Hyperparameter Stabilisation (Best Configuration)

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

#### 10.6 Fourth Training Run: Simplified Input

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

#### 10.7 Fifth Training Run: Advanced Tuning and Partial Freezing

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

#### 10.8 Sixth Training Run: Stratified 5-Fold Cross-Validation

**Changes and Rationale:**
1. **Removed freezing:**
  - All BERT layers are trainable again
  - Allows full adaption to the task
2. **Retained advanced hyperparameters:**
  - Maintain consistency with the previous run for controlled comparison
	-	Evaluate whether prior tuning generalises across multiple splits
3. **Applied 5-fold stratified cross-validation:**
  -	Reduces dependence on a single validation split
	-	Provides a more reliable estimate of generalisation performance
	-	Enables quantification of variance and model stability

**CV Results:**

| Fold | Accuracy | F1     | Precision | Recall |
|------|----------|--------|-----------|--------|
| 1    | 0.6429   | 0.7000 | 0.6034    | 0.8333 |
| 2    | 0.7381   | 0.7660 | 0.7059    | 0.8372 |
| 3    | 0.7262   | 0.7579 | 0.6923    | 0.8372 |
| 4    | 0.5714   | 0.6786 | 0.5507    | 0.8837 |
| 5    | 0.7262   | 0.7294 | 0.7381    | 0.7209 |

| Metric    | Mean  | Std   |
|-----------|------|-------|
| Accuracy  | 0.6810 | 0.0721 |
| F1        | 0.7264 | 0.0373 |
| Precision | 0.6581 | 0.0781 |
| Recall    | 0.8225 | 0.0604 |


**Final Model Results (Post-CV Training):**

| Metric    | Value |
|-----------|-------|
| Accuracy  | 0.6889 |
| F1        | 0.7021 |
| Precision | 0.6875 |
| Recall    | 0.7174 |
| Loss      | 0.6500 |

**Observations:**

1. **F1 (primary metric):**
	-	CV Mean = 0.726 → expected generalisation performance
	-	Final Model = 0.702 → slightly lower but within expected variance
	-	Std = 0.037 → relatively low → stable across folds
2. **Precision–Recall Trade-off:**
	-	During CV:
    -	Recall (0.82) > Precision (0.66)
    -	Model favours sensitivity → captures positives well but produces more false positives
	-	Final model:
    -	Precision (0.69) and Recall (0.72) more balanced
    -	Suggests averaging effect when trained on full dataset
3. **Variance Across Folds:**
	-	Accuracy std ≈ 0.07 → moderate variability
	-	Indicates sensitivity to data splits (expected given small dataset size ~420)
4. **Worst-case Fold (Fold 4):**
	-	Accuracy = 0.57, F1 = 0.68
	-	Performance drop not catastrophic → model retains baseline competence
	-	Confirms instability is present but bounded
5. **Training Dynamics:**
	-	Gradual loss reduction (~0.69 → ~0.65) indicates stable optimisation
	-	No evidence of severe overfitting (validation metrics improve alongside training)
	-	However, gains plateau early → model capacity exceeds dataset signal

**Comparison to Baseline (Third Training Run):**

- No substantial improvement in mean F1 despite increased complexity
-	Advanced tuning does not translate into consistent gains

**Interpretation:**

1. **Performance Instability**
	-	Metrics vary meaningfully across folds
	-	Model performance is dependent on data partitioning
	-	Indicates limited robustness due to small dataset size
2. **Diminishing Returns from Tuning**
	-	Advanced configuration does not outperform simpler setups
	-	Additional parameters increase variance without improving central tendency
	-	Suggests dataset is the primary bottleneck, not model capacity
3. **Generalisation Uncertainty**
	-	CV variance demonstrates that true performance is uncertain within a range (~0.69–0.76 F1)
	-	Single split evaluation would have been misleading
4. **Key Insight Driving Next Step**
	-	Model is data-limited, not architecture-limited
	-	Performance ceiling (~0.72 F1) reflects insufficient training signal
	-	Justifies transition to expanded manual annotation

---


#### 10.8 Global Interpretation

Across all phases:

- Best single-run performance observed in the third training run (F1 ≈ 0.75)  
- Cross-validation (iteration 6) shows mean F1 ≈ 0.73 with variability (±0.04)  
- Subsequent tuning did not improve mean performance  

Key findings:

- Model capacity is sufficient for the task  
- Input representation (full structured input) is appropriate  
- Training is stable after hyperparameter adjustment  
- Simpler configurations generalise as well as more complex ones 

However:

- Performance varies across data splits (CV std ≈ 0.04–0.07)  
- True generalisation performance lies within a range (~0.69–0.76 F1), not a single value  
- Single validation results (e.g. 0.75 F1) slightly overestimate performance  

---

#### 10.9 Final Conclusion and Decision

All empirical evidence indicates that **dataset size is the primary limiting factor**, rather than model architecture or training strategy.

Supporting evidence:

- Cross-validation confirms a performance ceiling (~0.72–0.73 mean F1)  
- No consistent improvement from additional tuning or complexity  
- Increased variance with more complex configurations  
- Stable but bounded performance across folds  

Conclusion:

- Model is not underpowered  
- Training pipeline is functionally correct  
- Hyperparameter space has been sufficiently explored  
- Performance is constrained by limited data 

Further tuning and iteration on the current dataset is not justified:

- No meaningful gains observed  
- Risk of overfitting increases  
- Generalisation does not improve  
 
---

#### 10.10 Next Steps

Increase the dataset size to enable further learning and performance improvements:

- Expand annotated dataset from 600 → 1200 samples  
- Cross validate the expanded dataset with the final complex configuration and the baseline configuration (third run) to compare performance and stability
- Retrain without introducing additional complexity (additional hyperparameters are not justified by current evidence)

Retrain using improved pipeline (CV-based training without redundant validation split):

- Split data into Train / Test only
- Apply cross-validation on the full training set for:
  - Model selection  
  - Performance estimation  
  - Checkpoint identification  
- Train final model afterwards on full training data  
- Single evaluation performed on held-out test set  

This revised pipeline:

- Maximises data utilisation  
- Eliminates unnecessary validation duplication  
- Produces a more reliable and efficient estimate of real-world performance  

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

The objective of this phase is to expand the annotated dataset from 600 to 1200 samples by manually annotating an additional 600 examples, followed by re-splitting the dataset into training, validation, and test sets (80/10/10), and retraining the model using the best-performing configuration identified previously.

- Prior experiments demonstrated that model performance was not limited by architecture or hyperparameter tuning, but by dataset size. 
- While the model was able to learn meaningful patterns (F1 ≈ 0.70–0.75), performance remained unstable and sensitive to data splits, indicating insufficient training signal and limited generalisation.

This phase addresses that constraint directly by increasing data volume, with the expectation that:

- Model performance will improve due to greater exposure to task-relevant patterns  
- Variance across splits will decrease, indicating improved stability  
- Generalisation to unseen data will become more reliable  

This represents the final data processing phase prior to definitive retraining and evaluation on the held-out test set.

---

### 2. Dataset Expansion Rationale

The dataset is expanded from 600 to 1200 samples based on empirical findings from prior training and cross-validation following standard practice: incrementally scale dataset size and reassess performance after each increase.

**Observed behaviour:**
- Moderate performance (F1 ≈ 0.72–0.75)  
- Variability across folds (±0.04 F1)

This indicates a data-limited regime, where performance is constrained by dataset size rather than model capacity or optimisation.

**Theoretical basis:**
- Generalisation error and variance scale approximately with 1/sqrt(N)  
- Increasing N improves stability and reduces performance variability  

**Expected impact of doubling data (600 → 1200):**
- ~29% reduction in variance  
- More robust pattern learning  
- Improved consistency across data splits  

This is the smallest practical increase expected to yield measurable improvement while remaining feasible for manual annotation.

---

### 3. Sampling Additional Data

#### 3.1 Objective

- Generate a new, balanced, annotation-ready dataset of extracted clinical entities for transformer validation  
- Ensure no overlap with the previously sampled dataset (based on defined deduplication criteria)  

---

#### 3.2 Deduplication Strategy

The sampling procedure is identical to the original approach, with the addition of a pre-filtering step to remove previously sampled entities:

1. Sample by `entity_type` to enforce class balance across:
   - SYMPTOM  
   - INTERVENTION  
   - CLINICAL_CONDITION  
2. Sample 200 entities per class (total = 600 samples)  
3. Concatenate all classes into a single dataset  
4. Shuffle using a fixed random seed for reproducibility  

To prevent overlap with the original annotated dataset, a filtering step is applied before sampling:

- Row-level deduplication is performed using the five key fields used by the transformer:
  - `sentence_text`, `entity_text`, `entity_type`, `concept`, `task`
- Tuple-based comparison is used:
  - `df_tuples`: pandas Series of row-wise tuples (aligned with the full dataset)
  - `existing_tuples`: set of tuples from the previously sampled dataset (used for efficient lookup)
- Rows in the full dataset are excluded if their tuple representation exists in the previous sample.

This ensures that:

- No entity-context combination (under these fields) is repeated across datasets  
- Deduplication is independent of row index or dataset ordering  

There are important implications:

- Deduplication is based on exact matches across the five columns only  
- Entity span positions (e.g. character indices) are not included  
- As a result:
  - Multiple rows referring to the same entity within the same sentence are treated as duplicates  
  - More than 600 rows may be removed during filtering, since a single annotated entity can correspond to multiple rows in the raw dataset  

This is intentional and aligned with the model design:

- The transformer does not use positional indices  
- Therefore, such rows are functionally identical from the model’s perspective  

---

#### 3.3 Sampling Workflow

The logic is implemented in `sample_additional_entities.py` and mirrors the original sampling pipeline, with the addition of a deduplication step prior to sampling.

**Workflow:**

1. **Load extraction candidates**
   - Read JSONL file (`1 line = 1 entity`)
   - Extract relevant fields:
     - `note_id`, `section`, `concept`, `entity_text`, `entity_type`, `sentence_text`, `negated`, `task`, `confidence`

2. **Convert to structured dataset**
   - Flatten records into a pandas DataFrame (`df`)

3. **Load existing annotated sample**
   - Load previously sampled dataset (`annotation_sample_raw.csv`, n = 600)

4. **Apply deduplication (new step)**
   - Define deduplication columns:
     - `sentence_text`, `entity_text`, `entity_type`, `concept`, `task`
   - Convert both datasets into tuple representations:
     - `df_tuples` (Series) for row-wise alignment
     - `existing_tuples` (set) for efficient lookup
   - Filter dataset:
     - Remove rows present in `existing_tuples`
     - Result: `df_filtered` contains only unseen samples

5. **Perform stratified sampling**
   - For each `entity_type`:
     - SYMPTOM  
     - INTERVENTION  
     - CLINICAL_CONDITION  
   - Sample `N_PER_CLASS = 200`
   - Enforces balanced class distribution

6. **Combine sampled data**
   - Concatenate class-specific samples into a single dataset (n = 600)

7. **Shuffle dataset**
   - Apply random shuffle (`random_state = 42`)
   - Reduces ordering bias for annotation

8. **Prepare for annotation**
   - Add empty column:
     - `is_valid` (ground truth label placeholder)

9. **Save outputs**
   - `additional_annotation_sample_raw.csv` (always overwritten)
   - `additional_annotation_sample_labeled.csv` (created only if not already present)

---

#### 3.4 Sampling Results

Terminal output confirms successful execution of the sampling pipeline:

- Loaded 47,487 total entities  
- Retained 46,674 entities after deduplication  
- Final sampled dataset size: 600 entities  

A total of 813 rows were removed during deduplication:

- This is expected, as multiple rows in the full extraction dataset can correspond to the same entity-context combination (based on the deduplication columns). 
- These are treated as duplicates and removed prior to sampling.

The final dataset therefore consists entirely of new, non-overlapping samples under the defined matching criteria.

---

### 4. Manual Annotation and Validation 

#### 4.1 Objective

- Manually annotate the new dataset with binary labels (`is_valid`) indicating whether each entity is a valid extraction in its context  
- Apply the same annotation guidelines as the initial dataset to ensure consistency  
- Validate the annotated dataset to confirm label quality, class balance, and structural integrity prior to retraining  

---

#### 4.2 Annotation and Validation Strategy

**Annotation:**
- Each entity is manually labeled using the `is_valid` field  
- The same event-based and status-based guidelines are applied across all entity types  
- Ensures consistency with the original 600-sample dataset  

**Validation:**
- The same validation logic is reused (`validate_additional_manual_annotations.py`)  
- Checks include:
  - Missing values  
  - Label validity (`True/False`)  
  - Class balance (`is_valid` distribution)  
  - Task distribution (200 per class)  
  - Task vs label breakdown  

This ensures the new dataset is structurally and statistically consistent before merging and retraining  

---

#### 4.3 Validation Results

Exact same logic as before but with new dataset, implemented via `validate_additional_manual_annotations.py`

Validation analysis confirms:

- All rows and columns present 
- No missing values apart from `negated` (400) which is expected and consistent with the entity types
- Label integrity confirmed: all `is_valid` values are `True` or `False`, no missing annotations
- Class distribution balanced: True (294) vs False (306)
- Task distirbution balanced: 200 per class (SYMPTOM, INTERVENTION, CLINICAL_CONDITION)
- Task vs label breakdown shows no significant imbalance within classes:
  - Both `True` and `False` labels well represented across all tasks. 
  - Variation across tasks is expected due to underlying entity types and annotation guidelines.

| Task                         | False | True |
|------------------------------|-------|------|
| clinical_condition_active    | 117   | 83   |
| intervention_performed       | 81    | 119  |
| symptom_presence             | 108   | 92   |

Conclusion:

- Dataset passes all validation checks  
- Labels are valid, complete, and consistent  
- Class and task distributions are appropriate  
- No data quality issues identified  

The dataset is suitable for merging with the original annotations and proceeding to model retraining  

---

### 5. Stratified Resplitting

#### 5.1 Objective

- Combine both annotated datasets (600 + 600) into a single dataset of 1200 samples  
- Perform a stratified split into Training / Test sets (85/15)  
- Ensure the test set remains fully unseen for final evaluation  
- Replace the previous validation split with cross-validation on the training set  

---

#### 5.2 Splitting Strategy

The two annotated datasets are concatenated to form a unified dataset of 1200 samples.

A stratification key is defined as: `task` + `is_valid`

This ensures:
- Class balance (`is_valid`) is preserved  
- Task distribution is preserved  
- Label balance is maintained within each task  

A single stratified split is then performed:
- 85% Training (1020 samples)  
- 15% Test (180 samples)  

This is a standard and efficient setup where:
- Cross-validation is performed on the training set  
- The test set is reserved for final, unbiased evaluation  

---

#### 5.3 Workflow Implementation

The resplitting logic is implemented in `stratified_resplit.py`:

1. Load both annotated datasets (original + additional)  
2. Concatenate into a single DataFrame (1200 rows)  
3. Create stratification key (`task + is_valid`)  
4. Perform stratified split (85/15) using `train_test_split`  
5. Drop helper column (`stratify_key`)  
6. Reset indices for clean datasets  
7. Verify distributions (task, label, task vs label)  
8. Save outputs to `train.csv` and `test.csv`  

---

#### 5.3 Resplitting Results

Dataset sizes:

- Total = 600 + 600 = 1200 samples
- Training: 1020 samples
- Test: 180 samples

Train distribution:

- Task distribution balanced: 340 per class (SYMPTOM, INTERVENTION, CLINICAL_CONDITION)
- Label distribution balanced: True (510) vs False (510)

Test distribution:

- Task distribution balanced: 60 per class (SYMPTOM, INTERVENTION, CLINICAL_CONDITION)
- Label distribution balanced: True (89) vs False (91)

Task vs label breakdown for train and test sets confirms:

- No significant imbalance within classes.
- Both `True` and `False` labels well represented across all tasks. 
- Variation across tasks is expected due to underlying entity types.

Conclusion:

- Perfect task balance is maintained across both splits  
- Label distribution is closely preserved (near 50/50)  
- Task–label relationships remain consistent between train and test 

This confirms that the stratified split is correct and suitable for downstream cross-validation and final evaluation.

----

## Model Cross-Validation and Model Selection

### 1. Objective

- Evaluate two candidate hyperparameter configurations derived from previous iterations:
  - **Run 3 (stable baseline)**
  - **Run 6 (advanced tuning)**
- Use stratified 5-fold cross-validation on the full training set (1020 samples) to:
  - Compare performance (primarily F1)
  - Assess stability (variance across folds)
- Select the best-performing configuration in order to retrain a final model on the full training set

This stage completes the model development phase by providing a robust performance estimate and comparison, before final training and evaluation on the held-out test set.

---

### 2. Updated Model Input Representation

The model input was extended to include the `section` field alongside the existing fields:

- Six fields are now used for the input representation: 
  `section`, `entity_type`, `entity_text`, `concept`, `task`, `sentence_text` 
- These components are concatenated into a single structured sequence using delimiters.

Clinical sentences are often brief and lack sufficient context in isolation. The `section` field provides higher-level document context (e.g. `chief complaint`), which helps:

- Disambiguate short or underspecified text  
- Provide clinically relevant context for entity interpretation  

This improves the model’s ability to correctly identify valid entities, particularly in context-dependent cases. This therefore:

- Adds contextual signal without architectural changes  
- Leverages transformer capacity to integrate structured inputs  
- Supported by increased dataset size, reducing risk of instability from added input complexity  

---

### 3. Cross-Validation Design Decisions

#### 3.1 Focused Comparison

Rather than exploring new hyperparameters, only two configurations from the previous iterations are evaluated. This controlled comparison isolates whether added complexity provides consistent benefit, and allows us to compare and choose the best model configuration.

**Stable configuration (Run 3):**

This configuration reflects a conservative fine-tuning strategy derived from earlier runs (Run 3), prioritising training stability and minimising overfitting risk. It uses:

- Learning rate: 5e-6
- Batch size: 8
- Epochs: 3
- Gradient clipping (max norm): 1.0

This is the simpler, lower variance setup which previously achieved best single-run performance (F1 ≈ 0.75).

**Advanced configuration (Run 6):**

- Learning rate: 3e-6
- Batch size: 8
- Epochs: 5
- Weight decay: 0.05
- Gradient accumulation: 2
- Warmup ratio: 0.1
- Gradient clipping (max norm): 1.0

This has more complex tuning (weight decay, warmup, accumulation), previously showed no clear improvement in single-run experiments and introduced additional training complexity, however it is included to confirm whether this scales with more data.

---

#### 3.2 Cross-Validation Strategy

Cross-validation is now the primary method for performance estimation and model selection:

- Cross-validation is performed only on the training set (1020 samples)
- No fixed validation split is used in this stage as cross-validation fully replaces it for model selection

Cross-validation procedure:

- Stratified 5-fold split based on label (`is_valid`)
- Each fold:
  - Trains a fresh model from pretrained weights
  - Evaluates on its validation fold
- Metrics are aggregated across folds (mean and standard deviation)

By replacing the previous validation split it provides:

- More reliable generalisation estimates  
- Quantification of performance variability  
- Reduced dependence on a single data partition  

---

### 3.3 Model Selection Criteria

Primary metric:
- **F1-score (mean across folds)** as it captures overall balance

Secondary considerations:
- Standard deviation (stability)
- Precision–recall balance

Selection logic:
- Prefer higher mean F1  
- If similar, prefer lower variance (more stable model)  
- Consider precision–recall trade-off if F1 is close (e.g. if one model has much higher recall but similar F1, it may be preferred)

Currently for model selection, the focus is on f1 and recall, however at the final model stage we will adjust decision threshold to meet precision requirements meaning that the overall pipeline still is precision-focused.

---

### 4. Cross-Validation Results and Analysis

#### 4.1 Stable Configuration (Run 3)

| Fold | Accuracy | F1 Score | Precision | Recall | Loss |
|------|----------|----------|-----------|--------|------|
| 1    | 0.7353   | 0.7245   | 0.7553    | 0.6961 | 0.5651 |
| 2    | 0.6520   | 0.6537   | 0.6505    | 0.6569 | 0.6346 |
| 3    | 0.7157   | 0.7157   | 0.7157    | 0.7157 | 0.5694 |
| 4    | 0.7353   | 0.7273   | 0.7500    | 0.7059 | 0.5748 |
| 5    | 0.7059   | 0.6842   | 0.7386    | 0.6373 | 0.5768 |

| Metric     | Mean   | Std Dev |
|------------|--------|---------|
| Accuracy   | 0.7088 | 0.0342  |
| F1 Score   | 0.7011 | 0.0315  |
| Precision  | 0.7220 | 0.0428  |
| Recall     | 0.6824 | 0.0337  |
| Loss       | 0.5841 | 0.0286  |

1. **Overall Performance**
- The model achieves a mean F1 score of ~0.70, indicating moderate classification performance.
- Accuracy (~0.71) is consistent with F1, suggesting no major imbalance-driven inflation.
- Precision (0.72) exceeds recall (0.68), indicating a slight bias toward conservative positive predictions (fewer false positives, more false negatives).

2. **Stability and Variance**
- Standard deviations across all metrics are low (≈0.03–0.04), indicating stable performance across folds.
- No evidence of catastrophic fold failure; worst-performing fold (Fold 2) remains within a reasonable range.

3. **Fold-Level Observations**
- Fold 2 underperforms (F1 ≈ 0.65), suggesting sensitivity to specific data partitions.
- Folds 1 and 4 achieve the highest performance (F1 ≈ 0.72–0.73), demonstrating the model’s upper bound under this configuration.
- Remaining folds cluster tightly around the mean, reinforcing robustness.

4. **Training Dynamics**
- Training loss decreases consistently across epochs, with validation metrics improving incrementally.
- Performance gains plateau by epoch 2–3, indicating that additional epochs are unlikely to yield significant improvements under this configuration.
- Absence of warmup and regularisation does not destabilise training, likely due to the low learning rate.

5. **Bias Characteristics**
- Higher precision than recall suggests:
  - The model is more cautious when predicting positive labels (`is_valid=True`)
  - Potential under-detection of valid entities (false negatives)
- This behaviour may be desirable depending on downstream tolerance for false positives vs false negatives.

---

#### 4.2 Advanced Configuration (Run 6)

| Fold | Accuracy | F1 Score | Precision | Recall | Loss     |   
|------|----------|----------|----------|-----------|--------|
| 1    | 0.7304   | 0.7343   | 0.7238    | 0.7451 | 0.5967 |
| 2    | 0.6520   | 0.6758   | 0.6325    | 0.7255 | 0.6332 |
| 3    | 0.7157   | 0.7264   | 0.7000    | 0.7549 | 0.5989 |
| 4    | 0.7206   | 0.7220   | 0.7184    | 0.7255 | 0.6105 |
| 5    | 0.7059   | 0.6939   | 0.7234    | 0.6667 | 0.6038 |

| Metric        | Mean     | Std Dev  |
|--------------|----------|----------|
| Accuracy      | 0.7049   | 0.0309   |
| F1 Score      | 0.7105   | 0.0247   |
| Precision     | 0.6996   | 0.0388   |
| Recall        | 0.7235   | 0.0342   |
| Loss          | 0.6086   | 0.0147   |

**1. Overall Performance**
- The model achieves a mean F1 score of ~0.71, indicating solid and slightly improved performance compared to the prior configuration.
- Accuracy (~0.70) aligns closely with F1, suggesting balanced classification without inflation from class imbalance.
- Recall (0.72) exceeds precision (0.70), indicating a mild tendency toward capturing positives more aggressively (higher sensitivity).

**2. Stability and Variance**
- Standard deviations remain low across metrics (≈0.02–0.04), confirming stable cross-validation performance.
- Loss variance is particularly small (~0.015), indicating consistent optimisation behaviour across folds.
- No fold exhibits instability or collapse; variability is within expected bounds for small datasets.

**3. Fold-Level Observations**
- Fold 1 achieves the highest performance (F1 ≈ 0.73), representing the upper bound under this configuration.
- Fold 3 and Fold 4 also perform strongly and consistently (~0.72–0.73 F1).
- Fold 2 is the weakest (F1 ≈ 0.68), primarily due to lower precision, suggesting sensitivity to data partitioning.
- Fold 5 shows a different error profile: relatively high precision but notably lower recall, indicating missed positives.
- Overall, folds cluster tightly, reinforcing robustness despite minor partition sensitivity.

**4. Training Dynamics**
- Training loss decreases steadily across all folds, with validation performance improving consistently.
- Most gains occur within the first 2–3 epochs, after which improvements are incremental, indicating early convergence.
- Later epochs still yield small but consistent improvements, suggesting the learning rate schedule (decay) is effective.
- No evidence of overfitting: validation metrics continue to improve or stabilise rather than degrade.

**5. Bias Characteristics**
- The model shows a slight recall bias overall:
  - More inclined to predict positives (`is_valid=True`)
  - Lower false negative rate at the expense of some false positives
- However, this bias is not uniform:
  - Fold 5 reverses this pattern (precision > recall), indicating dataset-dependent behaviour
- This suggests the decision boundary is moderately sensitive to data distribution, but not excessively unstable.
- Depending on downstream use:
  - This configuration is better suited where missing positives is more costly than false alarms

---

### 5. Configuration Comparison

#### 5.1 Overall Performance Comparison

| Metric        | Stable     | Advanced  | Comment    |
|--------------|----------|----------|------------|
| F1 Score      |  0.7011  |  **0.7105**  | The advanced configuration provides a modest improvement in balanced performance, indicating better overall classification quality |
| Accuracy      |  **0.7088**  |  0.7049  | Difference is negligible (<0.5%), confirming both models perform similarly in aggregate correctness |
| Precision     |  **0.7220**  |  0.6996  | Stable has higher precision (fewer false positives) |
| Recall        |  0.6824  |  **0.7235**  | Advanced has higher recall (fewer false negatives) |

---

#### 5.2 Precision–Recall Trade-off and Error Profile

Error Profile Comparison:

- Stable configuration:
  - Higher precision → fewer false positives
  - Lower recall → **more missed true positives (false negatives)**
- Advanced configuration:
  - Higher recall → **fewer missed true positives**
  - Slightly lower precision → more false positives
- Decision boundary behaviour:
  - Stable = conservative classifier  
  - Advanced = sensitive classifier 

Overall Precision vs Recall Trade-off:

- Stable configuration is a conservative model (precision > recall = fewer false positives, more false negatives)
- Advanced configuration is a recall-oriented model (recall > precision = fewer false negatives, more false positives)

---

#### 5.3 Stability and Variance

| Metric        | Stable (Std Dev) | Advanced (Std Dev) |
|---------------|------------------|--------------------|
| F1 Score      | 0.0315           | **0.0247**         |
| Accuracy      | 0.0342           | **0.0309**         |
| Precision     | 0.0428           | **0.0388**         |
| Recall        | **0.0337**       | 0.0342             |
| Loss          | 0.0286           | **0.0147**         |

- Both configurations show low variance across folds (std ≈ 0.02–0.04), indicating robust generalisation.
- Advanced configuration has:
  - Slightly lower F1 variance (0.0247 vs 0.0315)
  - More consistent loss behaviour  
- Suggests advanced configuration has more reliable optimisation and generalisation.

---

### 6. Interpretation and Final Decision

#### 6.1 Clinical Context

For a clinical notes extraction tool, error costs are asymmetric:

- **False Negatives (FN; missed entities)**:
  - Result in loss of clinically relevant information  
  - Cannot be recovered downstream if the model assigns low scores to true positives  
- **False Positives (FP; incorrect entities)**:
  - Typically recoverable  
  - Can be filtered via thresholding, rules, or human review  

Therefore, **recall (sensitivity)** is prioritised at the model selection stage.

---

#### 6.2 Implications for Model Behaviour

The advanced configuration aligns better with this objective:

- Higher recall → more true positives assigned higher scores  
- Slightly lower precision → more false positives, but acceptable  

This reflects a **recall-oriented model**, which is preferable because:

- Model outputs are continuous scores (probabilities), not fixed labels  
- Classification depends on a **decision threshold** (default = 0.5)  
- Threshold tuning operates on these scores, not on learned representations  

Key principle:

- Model training determines ranking (score separation between classes)
- Threshold tuning selects an operating point on that ranking

Implications:

- A good model assigns **higher scores to true positives than true negatives**  
- This enables threshold tuning to:
  - Increase precision (by raising threshold)  
  - Increase recall (by lowering threshold), within limits  

However, limits exist:

- If true positives are assigned **low scores (poor ranking)**:
  - Lowering the threshold recovers them **but also admits many false positives**
  - Precision degrades rapidly → unusable trade-off  

This is why FN are effectively “irreversible”:

- Recovery requires lowering the threshold into regions dominated by negatives  
- This introduces excessive FP, collapsing precision  

Therefore:

- **High recall indicates better positive-class coverage and score assignment**
- **Better ranking → more effective threshold tuning**
- **Poor recall → limited recoverability regardless of threshold**

Conclusion:

- Model selection should prioritise **F1 + recall**, as proxies for ranking quality  
- A recall-oriented model preserves usable signal for downstream optimisation  
- Precision can be adjusted post hoc; recall is constrained by the learned score distribution  

---

#### 6.3 Final Decision

The **advanced configuration** is selected for final model training.

Rationale:

- Higher F1 → improved overall balance  
- Higher recall → better coverage of true positives  
- Comparable accuracy and variance → no meaningful loss in stability  
- More suitable error profile for clinical extraction 

Interpretation:

- The advanced model assigns **higher probabilities to more true positives**  
- This implies **better separation between positive and negative classes**  
- This separation is critical for effective threshold tuning   

Operational consequences:

- At deployment:
  - Threshold can be **increased** to improve precision  
  - Recall will decrease in a controlled manner  
- This enables tailoring to downstream constraints without retraining  

By contrast:

- The stable model:
  - Has higher precision but lower recall  
  - Misses more true positives (lower recall ceiling)  
  - Indicates weaker positive-class scoring  
- Threshold tuning cannot recover these missed positives without severe precision loss  

Final justification:

- Select the model with **better ranking and recall (advanced)**  
- Then use **threshold tuning** to achieve the desired precision–recall balance  

This approach maximises retention of clinically relevant information while preserving flexibility for downstream optimisation.

---

### 7. Workflow Implementation

This logic is implemented in `cross_validation.py` as follows:

1. **Reproducibility Setup**  
  - Set fixed random seeds across Python, NumPy, and PyTorch to ensure deterministic training behaviour.

2. **Data Loading and Preparation**  
  - Load the full training dataset (`train.csv`)  
  - Convert target variable (`is_valid`) to binary label format

3. **Tokenizer Initialisation**  
  - Load pretrained BioClinicalBERT tokenizer  
  - Define structured input format combining:
    - section, entity_type, entity_text, concept, task, sentence_text  

4. **Metric Definition**  
  - Define evaluation metrics:
    - Accuracy
    - Precision
    - Recall
    - F1-score  

5. **Hyperparameter Configuration Loop**  
  - Iterate over candidate configurations:
    - Stable baseline
    - Advanced tuned configuration  

6. **Stratified K-Fold Cross-Validation**  
  - Apply 5-fold stratified split based on class labels  
  - For each fold:
    1. Split into training and validation subsets  
    2. Convert to Hugging Face `Dataset` format  
    3. Tokenize inputs  
    4. Remove raw text columns and set tensor format  
    5. Initialise a fresh model from pretrained weights  
    6. Configure training arguments dynamically per configuration  
    7. Train model on training fold  
    8. Evaluate on validation fold  

7. **Fold-Level Metric Collection**  
  - Store evaluation metrics for each fold:
    - `eval_accuracy`, `eval_precision`, `eval_recall`, `eval_f1`, `eval_loss`

8. **Aggregation of Results**  
  - Compute mean and standard deviation across folds for each metric  
  - Save per-fold metrics to:
    - `config_<name>_folds.csv`

9. **Final Configuration Comparison**  
  - Aggregate results across configurations  
  - Save summary comparison to:
    - `final_comparison.csv`  
  - Rank configurations based on mean F1-score  

10. **Output Generation**  
  - Persist evaluation artefacts to:
    - `results/cross_validation/`  
  - No models or checkpoints are saved in this stage  

---

## Threshold Tuning

### 1. Objective

- The objective of this section is to determine an optimal decision threshold for converting model output probabilities into binary predictions.
- Rather than using the default threshold of 0.5, the threshold is tuned to achieve a more appropriate balance between precision and recall for the clinical extraction task.
- Out-of-fold (OOF) predicted probabilities are used to:
  - Provide unbiased predictions for all training samples  
  - Enable threshold optimisation without introducing data leakage  
- The selected threshold aims to:
  - Maximise overall performance (e.g. F1-score)  
  - While ensuring recall remains sufficiently high to minimise missed clinically relevant entities  

This stage bridges model selection and final training by calibrating the decision boundary according to task-specific error trade-offs.

---

### 2. Out-of-Fold (OOF) Predictions: Design and Rationale

#### 2.1 Overview

Out-of-fold (OOF) predictions are generated using cross-validation such that:

- Each sample is predicted only by a model that was not trained on it
- Predictions are collected across all folds to produce a full-length prediction vector

As a result every training sample has:

- A true label (`y_true`)
- An out-of-sample predicted probability (`y_prob`)

This creates a dataset equivalent to a validation set covering the entire training distribution, without data leakage.

---

#### 2.2 Rationale

Threshold tuning requires unbiased probability estimates. Using:

- Training predictions → invalid (overfitted, optimistic)
- Single validation split → unstable, dependent on one partition

OOF predictions solve both issues:

- **No leakage** → each prediction is out-of-sample  
- **Full data coverage** → all samples contribute  
- **Stable estimates** → reduces variance from a single split  

This makes OOF predictions the correct input for threshold tuning.

- **Unbiased**: no sample is predicted by a model trained on it  
- **Aligned**: predictions correspond exactly to original dataset order  
- **Complete**: covers entire dataset (N samples)  
- **Continuous**: preserves full probability information (not thresholded)  

---

#### 2.3 OOF Generation

Using stratified K-fold cross-validation (K=5):

For each fold:

1. Split training data into:
  - Training subset (80%)
  - Validation subset (20%)

2. Train model on training subset only

3. Generate predictions on validation subset:
  - Model outputs logits (raw scores per class)
  - Apply softmax to convert logits → probabilities
  - Extract probability of positive class (`label = 1`)

4. Store predictions in the correct positions:
  - Use `val_idx` to place predictions back into the full dataset structure

After all folds:

- Every sample has exactly one prediction, made out-of-sample
- Predictions are combined into a single array aligned with ground truth

---

#### 2.4 Output Structure

This array of predictions for each sample is then converted into a DataFrame with two columns:

1. `y_true`:
  - Ground truth labels (0 or 1)
  - Required for computing metrics at different thresholds  

2. `y_prob`:
  - Predicted probability of the positive class
  - Continuous values in [0, 1]
  - Used to simulate different decision thresholds  

This structure is minimal and sufficient for:

- Threshold tuning  
- Precision–recall analysis (e.g. PR curves, threshold vs metrics plots)
- Metric computation across thresholds

No additional data is required.

---

### 3. Out-of-Fold Predictions Workflow Implementation

This logic is implemented in `generate_oof_predictions.py`:

1. Load full training dataset and initialise label column  
2. Create empty arrays:
  - `oof_probs` (size N) for predicted probabilities  
  - `oof_labels` (size N) for ground truth  
3. Initialise stratified 5-fold splitter  
4. For each fold:
  - Split indices into train/validation  
  - Convert subsets to Hugging Face Dataset format  
  - Tokenise structured inputs  
  - Train model using advanced configuration  
  - Generate predictions on validation fold:
    - Extract logits  
    - Apply softmax → probabilities  
    - Select positive class probability  
  - Insert predictions into `oof_probs` using `val_idx`  
5. After all folds:
  - Combine `y_true` and `y_prob` into DataFrame  
6. Save to:
  - `results/threshold_tuning/oof_predictions.csv`  
7. Run sanity checks:
  - Shape matches dataset size  
  - Label distribution is preserved  
  - Probability range is within [0, 1]  

---

### 4. OOF Generation Validation

**Dataset Coverage**

- Total samples: 1020  
- OOF predictions generated: 1020  
- Alignment: Each sample received exactly one out-of-sample prediction  

This confirms full coverage with no data leakage, as each prediction was produced by a model that did not see the corresponding sample during training.

**Label Distribution**

- Positive class (1): 50%  
- Negative class (0): 50%  

Stratified K-fold splitting preserved class balance across folds, ensuring stable and representative training and validation partitions.

**Probability Distribution**

- Minimum predicted probability: 0.1907  
- Maximum predicted probability: 0.7557  

Key observations:

- Predictions span a meaningful probability range (not collapsed to a narrow band)
- No extreme saturation near 0 or 1, indicating the model retains uncertainty
- Suitable for threshold tuning, as different decision thresholds will produce different precision–recall trade-offs

**Training Behaviour (Summary)**

Across folds:
- Training loss consistently decreased over epochs
- No signs of divergence or instability
- Gradient norms remained within reasonable bounds

This indicates stable optimisation under the selected configuration.

These outputs are therefore suitable for selecting an optimal decision threshold in the next stage.

---

### 5. Threshold Tuning Overview
	
#### 5.1 Conceptual Understanding

Threshold tuning is the process of converting model probability outputs into binary decisions by selecting a decision boundary (threshold).

- In binary classification, the model outputs a probability: `p(y = 1)`
- A threshold `t ∈ [0, 1]` is applied such that:
  - `ŷ = 1` if `p ≥ t`
  - `ŷ = 0` if `p < t`

Changing the threshold directly controls the trade-off between:

- **Precision**: proportion of predicted positives that are correct  
- **Recall**: proportion of true positives that are recovered  

How threshold tuning interacts with the model:

- Performed on out-of-fold (OOF) predictions from cross-validation  
- Ensures all probabilities are out-of-sample, avoiding bias  
- Does not change the model, only how outputs are converted into decisions  
- Therefore a post-training decision policy, not a training step 

Position in the pipeline:

- Occurs after rule-based extraction (high recall)
- Occurs before final model training and dataset construction
- Controls how strictly the transformer filters candidate entities

Threshold tuning therefore determines which extracted entities are accepted as valid, shaping the final high-confidence dataset used for downstream modelling.

---

#### 5.2 Purpose and Design Principles

The purpose of threshold tuning in this pipeline is to define a clinically and operationally appropriate decision boundary for entity validation.

Pipeline structure:

1. **Rule-based extraction (recall-first)**  
   - Captures as many candidate entities as possible  
   - Accepts higher false positives to minimise missed entities  

2. **Transformer validation (precision-biased filtering)**  
   - Evaluates each extracted candidate  
   - Removes false positives while retaining true entities  

Design principle:

- Threshold tuning is not used to optimise generic metrics (e.g. F1) in isolation  
- Instead, it aligns model behaviour with the role of each pipeline stage:
  - Extraction → maximise coverage  
  - Validation → enforce reliability  
- Ensures the validation stage operates as a **controlled precision filter**

Operational objectives:

- Increase precision by applying a stricter acceptance threshold  
- Preserve sufficient recall to retain upstream coverage  
- Reduce false positives in the final dataset  
- Define consistent and reproducible decision behaviour for deployment  

System-level effect:

- High-recall candidate generation  
- Followed by precision-biased validation  
- Producing a high-confidence entity dataset for downstream modelling  

---

#### 5.3 Model vs Decision Policy

The transformer model is trained using standard binary cross-entropy loss, which does not explicitly prioritise precision or recall.

- The model learns to estimate p(y = 1) as accurately as possible  
- No class weighting or precision-specific loss is applied  
- Therefore, the model itself is not inherently precision- or recall-biased  

Precision bias is introduced at the decision level, not during training:

- Threshold tuning defines how probabilities are converted into binary outcomes  
- A higher threshold increases precision by requiring stronger model confidence  
- This shifts the behaviour of the validation stage without altering the model  

The pipeline achieves precision-focused behaviour through threshold selection, not through modification of the model training objective.

---

### 6. Threshold Selection Objective

The threshold selection problem requires defining an explicit optimisation objective as different objectives produce materially different datasets, with direct impact on downstream model performance.

#### 6.1 Metrics Optimisation Strategies

Three coherent optimisation strategies exist:

1. **F1 maximisation (balanced objective)**  

- Optimises the harmonic mean of precision and recall  
- Produces a balanced trade-off between false positives and false negatives  

Advantages:
- Simple and widely used  
- No additional constraints or assumptions required  

Limitations:
- Treats precision and recall as equally important  
- Does not reflect asymmetric costs in this pipeline  
- May retain unnecessary false positives  

---

**2. Recall maximisation (coverage-first objective)**  

- Prioritises capturing as many true positives as possible  
- Accepts lower precision and increased false positives  

Advantages:
- Maximises coverage of true entities  
- Minimises missed detections  

Limitations:
- Produces a noisy dataset  
- High false positive rate degrades downstream model learning  
- Conflicts with the validation stage’s role as a filter  

---

**3. Precision maximisation (strict filtering objective)**  

- Prioritises correctness of accepted entities  
- Typically achieved by increasing the decision threshold  

Advantages:
- Produces a high-confidence dataset  
- Reduces label noise  
- Aligns with downstream modelling requirements  

Limitations:
- Reduces recall  
- May discard valid entities  
- Can lead to insufficient dataset size if applied without constraint  

---

#### 6.2 Objective Selection Logic

The appropriate objective depends on how errors affect the final dataset:

- False positives introduce label noise, which degrades downstream model performance  
- False negatives reduce coverage, limiting available training signal  

In this pipeline:

- Label quality is critical for downstream learning  
- Some loss of coverage is acceptable  
- Excessive filtering must be avoided to preserve dataset utility  

Therefore, neither balanced optimisation nor unconstrained precision maximisation is appropriate.

---

#### 6.3 Final Objective

The selected objective is: 

> Maximise precision subject to maintaining acceptable recall

Rationale:

- Prioritises reduction of false positives (primary risk)  
- Retains sufficient true positives for downstream modelling  
- Avoids the symmetry assumption of F1  
- Avoids the instability of unconstrained precision maximisation  

This defines threshold selection as a constrained optimisation problem where:

- Precision is the primary objective  
- Recall acts as a safeguard against excessive information loss  

This objective directly reflects the role of the validation stage as a precision-biased filtering mechanism, producing a high-confidence dataset while preserving sufficient coverage for downstream tasks.

---

### 7. Final Threshold Selection Method

#### 7.1 Mathematical Formulation

Given the objective of precision maximisation with recall retention, threshold selection is defined as a constrained optimisation problem:

- Select the threshold that maximises precision  
- Subject to: `recall ≥ 0.85 × baseline_recall`

Where:

- `baseline_recall` is the recall at the default threshold (`t = 0.5`)  
- This represents the model’s reference operating sensitivity  

Minimum acceptable recall is defined as:

- `recall_min = 0.85 × baseline_recall`  
- This allows controlled degradation of recall (maximum 15%) when increasing precision  

---

#### 7.2 Rationale

The thresholding strategy reflects the role of the transformer as a precision-oriented validation stage within the pipeline.

- Precision is prioritised to reduce false positives and improve dataset quality  
- However, unconstrained precision maximisation leads to severe recall collapse  
- A constraint is therefore required to preserve sufficient true positives  

A relative recall constraint is used instead of an absolute target:

- No fixed clinical recall threshold exists  
- Model performance varies across datasets  
- Anchoring to baseline ensures consistency and reproducibility  

The choice of the `0.85` factor defines an allowable degradation tolerance:

- Permits recall reduction when increasing precision, but within controlled limits  
- Reflects asymmetric error costs:
  - False positives degrade dataset quality (high cost)  
  - False negatives reduce coverage (lower but non-zero cost)  
- Prevents degenerate behaviour:
  - Avoids high-precision / near-zero recall solutions  
  - Ensures the validator remains practically useful  
- Preserves downstream utility:
  - Maintains sufficient positive samples for modelling  
  - Avoids over-filtering that weakens training signal  

This results in a precision-biased but constrained decision rule, rather than a purely precision- or recall-optimised system.

---

#### 7.3 Implementation of Threshold Selection

Threshold selection is performed by:

1. Computing metrics across all thresholds  
2. Extracting baseline recall at `t = 0.5`  
3. Computing `recall_min = 0.85 × baseline_recall`  
4. Filtering thresholds where `recall ≥ recall_min`  
5. Selecting the threshold with maximum precision within this set  

This produces a precision-biased validation stage with bounded recall loss, resulting in a high-confidence dataset suitable for downstream modelling.

---

### 8. Threshold Metrics Generation and Visualisation

#### 8.1 Metric Generation

Threshold-dependent performance is evaluated using out-of-fold (OOF) predictions by computing classification metrics across a dense grid of thresholds.

Implementation:

- Thresholds are defined over the range `[0, 1]` using 1001 evenly spaced values  
  - Step size = 0.001  
- For each threshold `t`:
  - Convert probabilities → binary predictions  
  - Compute:
    - Precision  
    - Recall  
    - F1-score  

Output:

- A complete metrics table (`threshold_metrics.csv`) containing:
  - `threshold`, `precision`, `recall`, `f1`  

Design rationale:

- High-resolution threshold grid (1001 points):
  - Produces smooth, continuous metric curves  
  - Enables fine-grained inspection of trade-offs  
  - Avoids missing optimal or near-optimal operating regions  

- Use of OOF predictions:
  - Ensures all evaluations are out-of-sample  
  - Prevents optimistic bias in threshold behaviour  

This table forms the **metric landscape** used for both visualisation and final threshold selection.

---

#### 8.2 Role of Visualisation

Visualisations are used to **validate and interpret** the threshold–metric relationships, not to directly select the threshold.

Purpose:

- Confirm expected model behaviour:
  - Precision increases with threshold  
  - Recall decreases with threshold  

- Identify the structure of the trade-off:
  - Smooth vs unstable transitions  
  - Presence of usable operating regions  

- Ensure the optimisation objective is feasible:
  - Existence of thresholds with improved precision  
  - Without catastrophic loss of recall  

Visualisation therefore acts as a **sanity check and interpretability layer**, ensuring that the selected threshold (Section 7) is supported by the underlying metric behaviour.

---

#### 8.3 Visualisation Types and Interpretation

Three plots are generated from the threshold metrics:

---

**1. Precision–Recall Curve**

- X-axis: Recall  
- Y-axis: Precision  

Purpose:

- Visualises the full precision–recall trade-off across thresholds  
- Confirms that increasing precision requires sacrificing recall  
- Identifies whether the model provides meaningful separation  

Interpretation:

- A smooth curve indicates stable model behaviour  
- Absence of sharp collapse suggests usable threshold regions  
- Confirms that precision gains are achievable without degenerate behaviour  

---

**2. F1 vs Threshold**

- X-axis: Threshold  
- Y-axis: F1-score  

Purpose:

- Identifies the threshold that maximises balanced performance  
- Provides a reference point for overall model capability  

Interpretation:

- Peak F1 represents the best precision–recall balance  
- Used as a **reference anchor**, not as the final decision rule  
- Helps contextualise how far the chosen threshold deviates from balanced operation  

---

**3. Precision & Recall vs Threshold**

- X-axis: Threshold  
- Y-axis: Score (Precision and Recall)  

Purpose:

- Makes the trade-off explicit in threshold space  
- Shows how each metric evolves as the decision boundary shifts  

Interpretation:

- Precision typically increases monotonically  
- Recall typically decreases monotonically  
- Key observations:
  - Where recall begins to decline  
  - Where precision meaningfully improves  
  - Where recall degradation accelerates  

This plot is most useful for verifying that the selected threshold lies within a **stable operating region**, rather than an extreme regime.

---

#### 8.4 Connection to Threshold Selection

The outputs of this stage support, but do not replace, the selection method defined in Section 7.

- Threshold selection is performed numerically using the metrics table  
- Visualisations are used to:
  - Validate that the constraint-based optimisation is appropriate  
  - Confirm that the selected threshold lies in a stable trade-off region  
  - Ensure no pathological behaviour (e.g. abrupt metric collapse)  

Together:

- Metrics table → enables exact threshold selection  
- Visualisations → ensure the selection is justified and reliable  

This separation maintains a clear distinction between:
- **Decision rule (formal, reproducible)**  
- **Interpretation (qualitative, diagnostic)**  

---




baseline_recall: 0.7039
recall_min: 0.5983
method: precision-max under recall constraint
best_threshold: 0.5490

Metrics at selected threshold:
threshold    0.549000
precision    0.724215
recall       0.633333
f1           0.675732

Your output:
	•	baseline recall: 0.7039
	•	final recall: 0.6333 (~10% drop)
	•	precision: 0.724

Interpretation:
	•	✔ recall preserved (within allowed range)
	•	✔ precision improved
	•	✔ validation acting as a filter

This is precisely what the validation stage is supposed to do.

the pieline will now be precision focused because  the decision threshold is stricter. The system behaviour is precision-biased, even if the model itself is not.

	•	High-recall extraction already captured most candidates
	•	Validation removes:
	•	false positives
	•	some true positives (controlled loss)

That trade-off is intentional.

High-recall candidate extraction followed by precision-biased validation to produce high-confidence entities for downstream modelling






---

Final Model Evaluation (test set stage)

Purpose: reporting performance (this must include plots)

After:
	•	final training (on 1020)
	•	threshold selected

Then evaluate on held-out test set

Required plots:

1. Precision–Recall Curve (test set)
	•	This is standard for imbalanced classification
	•	More informative than ROC in your case

⸻

2. Confusion Matrix (at chosen threshold)
Shows:
	•	False negatives (critical)
	•	False positives

⸻

Optional:

ROC curve
	•	Not essential unless required by convention


---



error analysis - post hoc analysis

ablation study - post hoc analysis