# Phase 4 – Evaluation, Metrics, and Analysis Plan

## Objective

- Evaluate the performance of:
  1. Rule-based extraction (Phase 2)
  2. Transformer validation (Phase 3)
- Quantify improvement from rules → transformer
- Identify strengths, weaknesses, and failure modes
- Produce clear, defensible results for reporting

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

### 3.2 Metric Levels

#### A. Overall Metrics (Primary)

- Computed on all 90 entities
- Most reliable and reportable

#### B. Entity-Specific Metrics (Secondary)

- Per entity type (~30 each):
  - `SYMPTOM`
  - `INTERVENTION`
  - `CLINICAL_CONDITION`

Use for:
- Directional insights only (not strong statistical claims)

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

## 6. Confidence Score Analysis

### 6.1 What is Confidence?

- Output probability from transformer (softmax)
- Represents model certainty in prediction

---

### 6.2 What to Do With It

#### A. Distribution Analysis

- Plot confidence for:
  - Correct predictions
  - Incorrect predictions

Goal:
- Check if model is **well-calibrated**

---

#### B. Threshold Analysis

Test different thresholds:

- Default: `0.5`
- Alternatives:
  - `0.6`, `0.7`, `0.8`

Evaluate impact on:
- Precision
- Recall

Insight:
- Higher threshold → higher precision, lower recall

---

#### C. Use Cases

- Ranking predictions by confidence
- Filtering low-confidence outputs
- Error analysis (low-confidence errors vs high-confidence errors)

---

## 7. Error Analysis

### 7.1 Purpose

- Identify **systematic failure patterns**
- Understand limitations of both:
  - Rules
  - Transformer

---

### 7.2 Method

Manually review:

- False Positives
- False Negatives

Categorise errors:

#### SYMPTOM
- Negation scope errors
- Ambiguous phrasing

#### INTERVENTION
- Planned vs performed confusion
- Documentation ambiguity

#### CLINICAL_CONDITION
- Historical vs active confusion
- Uncertain diagnoses

---

### 7.3 Outcome

- Clear explanation of:
  - Where rules fail
  - Where transformer improves
  - Remaining limitations

---

## 8. Visualisations

### 8.1 Required (High Value)

1. **Bar Chart – Model Comparison**
   - X-axis: Model (Rules vs Transformer)
   - Y-axis: F1 Score

2. **Metric Breakdown Chart**
   - Precision / Recall / F1 side-by-side

---

### 8.2 Optional (If Time Allows)

1. **Confidence Histogram**
   - Correct vs incorrect predictions

2. **Threshold Curve**
   - Precision–Recall tradeoff

---

## 9. Key Outputs

You should produce:

- Overall metrics table
- Entity-specific summary
- Improvement comparison (rules vs transformer)
- Error analysis summary
- (Optional) visualisations

---

## 10. Final Conclusions

Your evaluation should clearly answer:

1. Does transformer validation improve over rules?
2. Where is improvement strongest?
3. What errors still remain?
4. Is the system reliable enough for downstream use?

---

## 11. Phase Positioning

- **Phase 3:** Model development (training + inference)
- **Phase 4:** Evaluation and analysis (this document)

Separation ensures:
- Clean experimental design
- Reproducibility
- Clear reporting

---