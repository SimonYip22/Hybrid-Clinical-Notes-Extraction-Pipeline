# Pipeline Design and Downstream Alignment

## 1. Pipeline Objective and Structure

- The pipeline is designed to transform unstructured clinical text into a structured, entity-level dataset suitable for downstream analysis and modelling.
- This requires extracting clinically relevant entities and validating them to produce a dataset that is both usable and reliable.

The system follows a two-stage architecture:

> High-recall extraction → precision-oriented validation

- The rule-based stage generates a broad set of candidate entities (maximising coverage)
- The transformer stage filters these candidates to improve correctness

This design reflects a deliberate separation between coverage (recall) and quality control (precision).

---

## 2. Downstream Applications and Requirements

Structured clinical entities are used in several types of downstream tasks, each with different performance requirements:

- **Structured dataset generation (primary use case)**  
  - Entities are converted into tabular features for:
    - Machine learning models (e.g. risk prediction)
    - Cohort selection
    - Epidemiological analysis  
  - Requires high precision  
  - Incorrect entities become incorrect features and corrupt downstream outputs  
- **Clinical summarisation**  
  - Aggregation of key findings into structured summaries  
  - Requires balanced precision and recall  
- **Information retrieval**  
  - Searching or indexing clinical concepts  
  - Requires high recall (missing entities reduces retrievability)  
- **Clinical decision support**   
  - Triggering alerts or interventions
  - Requires extremely high recall (missing signals is unsafe)  
  
---

## 3. Selected Use Case and Design Consequences

This pipeline is explicitly optimised for structured dataset generation for downstream modelling and analysis. In this setting:

- Extracted entities act as model features
- Data quality directly determines model validity

Error impact is therefore asymmetric:

- **False Positives (FP)**  
  - Introduce incorrect features  
  - Corrupt models and analyses  
  - Difficult to detect downstream  
  → **High cost**

- **False Negatives (FN)**  
  - Represent missing features  
  - Reduce completeness but do not introduce noise  
  → **Lower cost (within limits)**  

This leads to the core design principle:

> Precision is prioritised over recall

The system therefore accepts controlled loss in recall to ensure that retained entities are reliable.

---

## 4. Entity-Type Behaviour and Error Tolerance

Error tolerance is not uniform across entity types:

**State-Based Entities (Symptoms, Clinical Conditions)**

- Often repeated or contextually redundant  
- Signal can be preserved even if some mentions are missed  
- False negatives are more tolerable

**Event-Based Entities (Interventions)**

- Discrete, non-redundant clinical events  
- Each instance carries specific meaning  
- False negatives represent true information loss

Implication:

- A single global threshold may disproportionately affect entity types  
- Precision-oriented filtering is likely to impact event-based entities more strongly

This is an inherent limitation of a uniform decision boundary.

---

## 5. Alignment with Pipeline Design Choices

The pipeline components directly reflect the selected objective:

- **Rule-based extraction**
  - High recall candidate generation  
  - Minimal filtering  

- **Transformer validation**
  - Precision-oriented filtering  
  - Removes incorrect or ambiguous entities  

- **Threshold tuning**
  - Optimised using out-of-fold predictions  
  - Maximises precision under a recall constraint  
  - Defines the operating point of the system  

- **Output structure**
  - Binary predictions (`model_pred`)
  - Probabilities (`model_prob`)  

This design allows for:

- Flexible threshold adjustment  
- Downstream modelling  
- Calibration and uncertainty analysis  

This corresponds to a standard and well-established pattern:

> Candidate generation → learned validation

---

## 6. Implications for Evaluation

Evaluation is designed to directly reflect the pipeline objective:

- **Precision** → primary metric (correctness of features)
- **Recall** → controlled trade-off (coverage loss)
- **F1 score** → overall balance of system behaviour
- **Confusion matrix** → explicit analysis of FP vs FN trade-offs  

The goal of evaluation is therefore not generic model assessment, but:

- To verify that precision improves over the baseline  
- To quantify the cost in recall  
- To ensure that the trade-off aligns with the intended downstream use case  

---

## 7. Summary

- The pipeline is designed for structured dataset generation from clinical text
- This use case prioritises precision over recall
- The two-stage architecture separates:
  - High-recall extraction  
  - Precision-oriented validation  
- Error tolerance varies by entity type, introducing non-uniform effects
- Evaluation is explicitly aligned with these design decisions and objectives