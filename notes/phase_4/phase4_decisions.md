# Phase 4 – Model Evaluation Decisions

## Objective

This phase defines the evaluation framework decisions for assessing the final trained model and overall pipeline performance on a fully held-out test set, covering:

- Comparison of rule-based extraction and transformer validation against ground truth  
- Computation of core performance metrics (precision, recall, F1, accuracy)  
- Analysis of pipeline-level improvement introduced by the transformer  
- Stratified evaluation across entity types  
- Visualisations to illustrate performance and error patterns

The objective is to determine whether the transformer functions effectively as a precision-oriented validation layer, improving correctness over the rule-based baseline while maintaining acceptable recall.

This phase produces the final performance estimates used for evaluation and deployment readiness.

---








---

Next step (do NOT jump ahead)

Do not compute plots or stratification yet.

Immediate next step:

Create a metrics + confusion matrix script using this file.

⸻

Why this order matters

You structured Phase 4 into two layers:

1. Core system evaluation (FIRST)
	•	Overall performance
	•	Rule vs Transformer comparison
	•	Confusion matrices
	•	Precision / Recall / F1

2. Deeper analysis (SECOND)
	•	Entity-type breakdown
	•	Calibration / distributions
	•	PR / ROC curves

You must validate global behaviour first before deeper analysis.


---

## 1. Evaluation Setup

### 1.1 Ground Truth

- Source: **manually annotated test set (n = 90)**
- Each entity contains:
  - `is_valid` (ground truth label)

This is the **only unbiased dataset** used for final evaluation.

---

### 1.2 Predictions

For each test entity:

- **Rule-based prediction**
  - `SYMPTOM` → derived from `negated` field:
    - `negated = false` → valid
    - `negated = true` → invalid
  - `INTERVENTION` / `CLINICAL_CONDITION`:
    - All extracted entities assumed **valid**

- **Transformer prediction**
  - `is_valid` (model output)
  - `confidence` (probability score)

---

## 2. Comparisons

You are evaluating **two systems against ground truth**:

### 2.1 Rule-Based vs Ground Truth
- Measures baseline extraction quality

### 2.2 Transformer vs Ground Truth
- Measures final system performance

### 2.3 Improvement Analysis
- Compare rule vs transformer performance directly

---

## 3. Metrics

### 3.1 Core Metrics

For each comparison:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

Definitions:

- Precision → correctness of positive predictions  
- Recall → ability to detect true positives  
- F1 → balance of precision and recall  

---

## 4. Confusion Matrix Analysis

For both rule-based and transformer:

- True Positive (TP)
- False Positive (FP)
- True Negative (TN)
- False Negative (FN)

Purpose:
- Understand **types of errors**, not just overall score

---

## 5. Improvement Analysis

Explicitly compare:

- Rule vs Transformer:
  - Δ Precision
  - Δ Recall
  - Δ F1

Expected pattern:

| Entity Type         | Expected Outcome |
|--------------------|----------------|
| SYMPTOM            | Small improvement |
| INTERVENTION       | Moderate improvement |
| CLINICAL_CONDITION | Large improvement |

---

✔ REQUIRED:
	•	Scalar metrics (precision, recall, F1)
	•	Precision–Recall curve
	•	Confusion matrix (at selected threshold)
	•	Baseline vs final comparison (selected threshold vs default threshold)

✔ STRONGLY RECOMMENDED:
	•	Calibration check
    •	Histogram of predicted probabilities
    or
    •	Reliability curve (if possible)
	•	ROC curve

Pipeline-level (critical):
	•	Rule-based vs Transformer comparison
	•	3-way framing with ground truth

Outputs:
	•	Save:
	•	y_true
	•	y_prob
	•	y_pred

Pipeline performance (THIS is your real objective)

This is what you were originally aiming for.

You are not just evaluating a model, you are evaluating a system:

Ground truth
vs
Rule-based extraction
vs
Transformer validation (final output)

“we need the model to output label + confidence”

✔ Yes. You need:

For each test sample:
	•	y_true
	•	y_prob (softmax output)
	•	y_pred (after threshold = 0.549)

⸻

Why this is necessary

Because you need to compute:

1. Transformer performance
	•	compare y_pred vs y_true

2. Rule-based extraction performance
	•	compare extraction labels vs y_true

3. Pipeline improvement
	•	compare:
	•	rule-based vs ground truth
	•	transformer vs ground truth

Overall Pipeline Comparison (CORE)

You need a single evaluation table where each row represents one candidate entity, with:
  - entity_type (e.g. symptom, intervention, condition)
	•	y_true → ground truth label (0/1)
	•	rule_pred → rule-based output (0/1)
    If entity is extracted → rule_pred = 1
    If entity is not extracted → rule_pred = 0
    If negated → rule_pred = 0
    Else → rule_pred = 1
    rule_pred = is_extracted AND NOT negated
	•	model_prob → transformer probability
	•	model_pred → transformer prediction (after threshold = 0.549)

You need:

Table: Pipeline Comparison

Metrics table + FP and FN table

You compute metrics twice, using the same y_true:

rule-based performance:

precision_score(y_true, rule_pred)
recall_score(y_true, rule_pred)
f1_score(y_true, rule_pred)

Transformer performance:

precision_score(y_true, model_pred)
recall_score(y_true, model_pred)
f1_score(y_true, model_pred)


That gives your 3-way comparison

You are not comparing systems directly to each other.

You are comparing both against ground truth.

Ground truth (reference)

→ Rule-based predictions vs ground truth
→ Transformer predictions vs ground truth

What about comparing rule-based vs transformer?

You do not compute TP/FP between them.

Instead, you interpret:

Change
Meaning
Precision ↑
fewer false positives
Recall ↓
some true positives removed

That implicitly shows improvement.


Entity-Type Stratified Performance (SECONDARY)
deeper analysis


“Where does the pipeline actually help the most?”

It is useful because:
	•	Different entity types have different linguistic structure
	•	Rule-based extraction performance will vary by type
	•	Transformer gains will likely be non-uniform


Entity Type
Rule-based
Transformer effect
Structured / explicit (e.g. symptoms)
High precision already
Small improvement
Ambiguous / contextual (e.g. interventions)
Lower precision
Larger improvement
Complex / semantic (e.g. conditions)
Variable
Depends on model understanding

table:

Entity Type
Stage
Precision
Recall
F1
SYMPTOM
Rule-based
…
…
…
SYMPTOM
Transformer
…
…
…
INTERVENTION
Rule-based
…
…
…
INTERVENTION
Transformer
…
…
…
CLINICAL_CONDITION
Rule-based
…
…
…
CLINICAL_CONDITION
Transformer
…
…
…

This lets you answer:
	•	Where is rule-based weakest?
	•	Where does transformer add most value?
	•	Is improvement consistent across entity types?

That is real insight, not just metrics.


What each table tells you

Table 1:
	•	Overall system behaviour
	•	Confirms:
	•	precision ↑
	•	recall ↓ (controlled)

⸻

Table 2:
	•	Where improvements occur
	•	Tests your hypothesis:
“Does the transformer help weaker entity types more?”


	•	Correctness is already measured in Table 1
	•	Table 2 shows variation in performance across entity types

So it answers:
	•	“Where is the model strong/weak?”
	•	“Where does the pipeline add value?”