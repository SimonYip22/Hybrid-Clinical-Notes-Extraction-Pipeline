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

- Cross-validation is a post-training diagnostic tool, not part of the core training pipeline.
- Answers whether performance is stable across different subsets of the data, rather than being an artifact of a specific train/validation split.
  -	Confirms reliability of results
  -	Quantifies uncertainty in performance

This is critical because:

- The dataset is relatively small (~420 training samples)
- Model performance can vary depending on which samples are seen during training
- A single validation split may give an overly optimistic or pessimistic estimate

Role in pipeline:

- Not used for training decisions (hyperparameter tuning, checkpoint selection)
- Not used for final model selection (still based on standard validation)
- Used only for evaluation robustness to assess stability and generalisation confidence

Cross-validation is used to validate that this selection is robust, rather than to directly produce the deployed model.

---

#### 9.2 Method: Stratified 5-Fold Cross-Validation

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

#### 9.3 Implementation Logic

For each fold the following steps are performed:

```text
for fold in K:
    split data into train_fold (80%) and val_fold (20%) using stratification

    initialise new model (reset weights)

    train model on train_fold
    evaluate model on val_fold

    store metrics
```

Key implementation details:

- **Model reset per fold:** A fresh model is created using `deepcopy(model)` to avoid weight leakage across folds
- **Identical training configuration:** All folds use the same hyperparameters and training setup
- **Independent runs:** Each fold is a fully independent training + validation cycle
- **Same metric computation:** Accuracy, precision, recall, and F1-score are computed using the same `compute_metrics` function

---

#### 9.4 Metrics Aggregation

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

**Results:**

| Fold | Accuracy | F1 | Precision | Recall |
|------|----------|----|-----------|--------|
| 1 | 0.643 | 0.700 | 0.603 | 0.833 |
| 2 | 0.667 | 0.725 | 0.627 | 0.860 |
| 3 | 0.690 | 0.745 | 0.644 | 0.884 |
| 4 | 0.595 | 0.646 | 0.585 | 0.721 |
| 5 | 0.631 | 0.680 | 0.611 | 0.767 |

**Aggregated Metrics:**

| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | 0.645 | ±0.036 |
| F1 | 0.699 | ±0.039 |
| Precision | 0.614 | ±0.023 |
| Recall | 0.813 | ±0.067 |

**Observations:**

- Performance decreased compared to baseline
- High variability across folds (std = ±0.039 for F1)
- Some folds perform reasonably well (F1 ~0.745), while others are much worse

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

6. **Model initialisation**
  - Load pretrained BioClinicalBERT
  - Attach binary classification head (`num_labels = 2`)
  - Classification head weights randomly initialised

7. **Metric definition**
  - Compute:
    - Accuracy
    - Precision
    - Recall
    - F1-score (primary metric)

8. **Training configuration**
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

9. **Model training (standard validation)**
  - Train using `Trainer`
  - Perform:
    - Forward pass → logits
    - Loss computation → backpropagation
    - Optimizer + scheduler updates
  - Validate at end of each epoch
  - Track best-performing checkpoint

10. **Cross-validation (robustness assessment)**
  - Apply 5-fold stratified split on training data (420 samples)
  - For each fold:
    - Create train/validation subsets (~336 / ~84)
    - Re-tokenize data
    - Reset model weights (`deepcopy`)
    - Train and evaluate independently
  - Store metrics per fold
  - Aggregate:
    - Mean performance
    - Standard deviation (stability)

11. **Model saving**
  - Save final trained model: `models/bioclinicalbert/`
    - Model weights (`pytorch_model.bin`)  
    - Model configuration (`config.json`)  
  - Save Tokenizer files (`vocab.txt`, `tokenizer_config.json`, etc.)  
    - Ensures full reproducibility and usability of the model for inference

12. **Output reporting**
  - Print:
    - Training progress logs
    - Epoch-level training + validation metrics  
    - Fold-level cross-validation metrics  
    - Aggregated CV statistics (mean ± standard deviation)  

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

Expected outcomes for model training and evaluation:

- Clear improvement over rule-based extraction
- Performance pattern: highest to lowest due to Increasing semantic complexity across entity types

Entity Type
Expected Performance
SYMPTOM
Highest (strong baseline already)
INTERVENTION
Moderate improvement (complexity in temporality and intent)
CLINICAL_CONDITION
Largest improvement (highest complexity, weakest baseline)