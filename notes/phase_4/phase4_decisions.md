# Phase 4 – Model Evaluation Decisions

## Objective

This phase defines the evaluation framework used to assess both the trained transformer model and the overall extraction pipeline on a fully held-out test set (n = 180). This phase evaluates:

- Baseline rule-based extraction vs ground truth  
- Transformer validation vs ground truth  
- Net improvement introduced by the transformer layer  
- Precision–recall trade-offs under fixed thresholding  
- Performance variation across entity types  

The goal is to evaluate overall pipeline performance and quantify the contribution of the transformer validation layer, ensuring readiness for deployment on the full ICU corpus.

---

## Evaluation System Design

### 1. Overview

The evaluation is designed to assess pipeline-level performance, not the transformer in isolation, operating on three aligned components:

- **Ground truth** (manually annotated test set; reference)
- **Rule-based extraction** (high-recall baseline)
- **Transformer validation** (final system output)

This forms a 3-way evaluation framework, where both systems are evaluated against the same ground truth:

```text
			  Ground Truth (y_true)
				▲      	      	▲
				│        	    │
			Rule-Based  	Transformer
			(baseline)   	(final)
``` 

- Rule-based predictions vs ground truth  
- Transformer predictions vs ground truth  

There is no direct pairwise evaluation between systems, improvement is inferred from changes in metrics relative to ground truth.

This design ensures:

- A single, unbiased reference (`y_true`)
- Direct comparability of system performance
- Valid interpretation of precision–recall trade-offs  

The evaluation objective is to verify that the transformer functions as a:

> **precision-oriented validation layer applied to a high-recall extraction system**

Success is defined by:

- Precision increase (reduction in false positives)  
- Controlled reduction in recall  
- Net improvement in F1 score  
- Coherent performance across entity types  

---

### 2. System Structure

The evaluation is executed in two sequential layers:

#### 2.1 Core System Evaluation (Primary)

This layer establishes global system behaviour, each system is evaluated independently against ground truth to compute:

1. Accuracy, Precision, Recall, F1  
2. Confusion matrix components (TP, FP, TN, FN)

This produces two directly comparable metric sets:

- Rule-based performance (baseline)  
- Transformer performance (final system)  

Improvement is interpreted through metric deltas:

- Δ Precision → reduction in false positives  
- Δ Recall → impact on coverage  
- Δ F1 → overall system improvement  

This layers purpose is to:

- Validate that the transformer improves over the baseline  
- Quantify the precision–recall trade-off at a global level  
- Ensure that the pipeline behaves as intended before deeper analysis 

---

#### 2.2 Stratified & Diagnostic Analysis (Secondary)

This layer explains where and why performance changes occur, analysis includes:

- Performance stratified by entity type  
- Probability distributions (`model_prob`)  
- Calibration behaviour  
- Precision–Recall and ROC curves  
- Threshold sensitivity  

Expected trends:

| Entity Type             | Behaviour |
|------------------------|----------|
| `SYMPTOM`              | Smaller gains (already structured) |
| `INTERVENTION`         | Moderate gains |
| `CLINICAL_CONDITION`   | Larger or variable gains |

The purpose of this layer is to:

- Provide a deeper understanding of system behaviour (entity-level insights)
- Identify specific weaknesses or strengths in the baseline and transformer
- Validate expected behaviour patterns based on the model design and training data characteristics

---

### 3. Rationale for Two-Layer Design

The separation enforces a strict evaluation order:

1. **Validate global system behaviour first**
   - Confirm that the pipeline improves over baseline  
   - Ensure metric trends are coherent  

2. **Then perform deeper analysis**
   - Avoid over-interpreting local patterns  
   - Ensure stratified insights are grounded in real system behaviour  

Without this structure:

- Local improvements may be misleading  
- Noise may be mistaken for meaningful patterns  
- Overall system performance may be mischaracterised

---

## Evaluation Metrics Dataset 

### 1. Objective

Evaluation is based on constructing a single unified dataset where all prediction sources are aligned at the entity level. Each row corresponds to one candidate entity and contains:

- Ground truth label  
- Rule-based prediction  
- Transformer prediction (probability + thresholded output)  

This dataset is the foundation for all downstream evaluation, enabling for consistent metric computation and stratified analysis.

No metrics are computed during this stage; this step strictly produces the evaluation-ready dataset. This separation ensures reproducibility and flexibility in analysis.

---

### 2. Evaluation Components

#### 2.1 Ground Truth

- Source: manually annotated test set (n = 180)
- Label: `is_valid → y_true ∈ {0,1}`

This represents the only unbiased reference and is used to evaluate all systems.

---

#### 2.2 Rule-Based Predictions

The rule-based system represents the high-recall extraction baseline.

- All extracted entities are assumed valid (1)  
- Exception: **SYMPTOM entities**
  - `negated = True → 0 (invalid)`
  - `negated = False → 1 (valid)`

Rationale:

- The extraction stage prioritises recall (maximising coverage)  
- Rule-based methods are limited in precision due to lack of contextual understanding  
- Minimal precision logic is applied only where trivial and high-impact (negation)  
- This establishes a weak but consistent baseline for measuring transformer improvement  

---

#### 2.3 Transformer Predictions

The transformer acts as a validation layer applied after extraction.

For each entity:

- Outputs logits → converted to probability: `model_prob = p(y = 1)`
- Applies tuned threshold: `model_pred = (model_prob ≥ 0.549)`

Key properties:

- This is the first application of the tuned threshold, defining the final decision boundary used for deployment-level predictions  
- These predictions represent the final pipeline output and are directly compared to ground truth  
- The model is explicitly precision-oriented, so expected behaviour is:
  - Precision ↑ (fewer false positives)  
  - Recall ↓ (removal of uncertain positives)  

---

### 3. Dataset Design

Each row represents one extracted entity and aligns:

| Column        | Description |
|--------------|-------------|
| `entity_type` | Entity category (for stratified analysis) |
| `y_true`      | Ground truth label (manual annotation) |
| `rule_pred`   | Rule-based prediction |
| `model_prob`  | Transformer probability (p(y=1)) |
| `model_pred`  | Final transformer prediction (thresholded) |

Design rationale:

- **Single-table design:** Ensures all systems are evaluated against the same reference without recomputation  
- **model_prob retained:** Required for PR curves, ROC analysis, and threshold sensitivity  
- **model_pred included:** Represents the deployed decision rule  
- **entity_type included:** Enables stratified performance analysis  
- **Separation of concerns:** Dataset generation is independent from metric computation, enabling reproducibility 

---

### 4. Workflow Implementation

All script code and logic is implemented in `run_evaluation.py` with these steps:

1. **Initialise environment**
  - Select computation device:
    - GPU if available, otherwise CPU
  - Defines execution context for inference

2. **Load test dataset**
  - Read held-out test set (n = 180)
  - Define ground truth:
    - `y_true = is_valid` (binary labels)

3. **Compute rule-based predictions**
  - Apply deterministic extraction logic:
    - `SYMPTOM`: validity determined by `negated` (0 if negated, 1 if not)
    - All other entity types: assumed valid (1)
  - Output baseline:
    - `rule_pred ∈ {0,1}`

4. **Load trained model and tokenizer**
  - Load final model weights from `bioclinicalbert_final/`
  - Load corresponding tokenizer (ensures identical token mapping)
  - Move model to selected device
  - Set `model.eval()`:
    - Disables dropout
    - Ensures deterministic inference

5. **Reconstruct model inputs**
  - Concatenate structured fields into a single string:
    - `[SECTION] ... [ENTITY TYPE] ... [ENTITY] ... [CONCEPT] ... [TASK] ... [TEXT] ...`
  - Matches training-time input format exactly

6. **Tokenise inputs**
  - Convert text → token IDs + attention masks
  - Apply:
    - truncation (`max_length = 512`)
	- dynamic padding (batch-level)
  - Output: tensors aligned with model input requirements

7. **Run batched inference**
  - Iterate over dataset in fixed-size batches
  - Move inputs to same device as model
  - Forward pass only (`torch.no_grad()`): inputs → model → logits
  - Convert logits → probabilities (retain only class 1): `model_prob = softmax(logits, dim=1)[:, 1]`

8. **Apply threshold**
  - Convert probabilities to binary predictions:
    - `model_pred = 1 if model_prob ≥ 0.549 else 0`
  - Threshold fixed from prior tuning phase

9. **Construct evaluation dataset**
  - Combine all components into a single table:
    - `entity_type`
    - `y_true`
    - `rule_pred`
    - `model_prob`
    - `model_pred`

10. **Save output**
   - Write dataset to: `outputs/evaluation/pipeline_predictions.csv`
   - This file serves as the single source for all downstream evaluation

---

