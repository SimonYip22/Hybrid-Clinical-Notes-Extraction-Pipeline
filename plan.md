# NLP-Based ICU Clinical Notes Extractor

***Structured Extraction and Interoperability from ICU Notes Using ClinicalBERT & Rule-Based NLP***

---

## Project Focus

This project extracts clinically relevant, structured information from ICU patient notes to generate actionable outputs in FHIR-compatible format. It demonstrates advanced NLP skills, clinical knowledge integration, and practical deployment, complementing your previous ICU Time-Series Predictor.

**Key focus areas:**
- Extract structured entities (e.g., symptoms, interventions, complications) from unstructured ICU notes
- Map entities to **SNOMED** terms for interoperability
- Output structured data in **FHIR JSON**
- Build a clinician-friendly **Streamlit demo**
- Optional: augment ICU Predictor features with extracted insights (multi-modal integration)

---

## 1. Scope & Deliverables

### 1.1 Standard Components (6–8 weeks)
- **Rule-based + scispaCy** NLP pipeline for entity extraction
- **ClinicalBERT classifier** for classifying extracted entities
- **SNOMED mapping** of entities for interoperability
- **FHIR JSON output** to standardize extracted data
- **Streamlit demo** for per-note structured output

### 1.2 High-Impact Advanced Component (Optional)
- **ICU Predictor Integration**  
  - Select 5–10 clinically meaningful features extracted from notes
  - Feed them as optional inputs into the existing ICU predictor
  - Show delta in key metrics and case-level examples
  - Adds strong multi-modal demonstration value without full retraining

---

## 2. Skills & Models Highlighted

| Skill / Component | Purpose | Technical Depth / Impression |
|------------------|---------|-----------------------------|
| Rule-based NLP | Preprocessing, regex-based entity extraction | Demonstrates hybrid engineering approach |
| scispaCy / ClinicalBERT | Entity recognition & classification | Transformer-based NLP, clinical adaptation |
| SNOMED mapping | Standardized ontology | Clinical interoperability |
| FHIR JSON output | Structured, EHR-ready outputs | Real-world healthcare application |
| Streamlit demo | Interactive front-end | Deployable, user-facing MVP |
| ICU Predictor integration | Multi-modal augmentation | Practical application of NLP features into ML pipeline |

---

## 3. Data

- **Source:** Synthetic or de-identified ICU clinical notes (MIMIC-IV demo v2.2 or equivalent)
- **Annotations:** 50–300 manually labelled examples for ClinicalBERT fine-tuning (small, representative sample)
- **Output:** Structured entities + classifications + FHIR-compatible JSON

---

## 4. Pipeline Overview

1. **Data Preprocessing**
   - Clean notes
   - Standardize units, timestamps
   - Optional section splitting

2. **Entity Extraction**
   - Rule-based extraction for high-precision entities
   - scispaCy NER for broader coverage
   - ClinicalBERT classifier for semantic context

3. **Ontology Mapping**
   - Map extracted entities to SNOMED concepts
   - Handle synonyms, abbreviations

4. **Structured Output**
   - Format into FHIR-compliant JSON
   - Include metadata: patient ID, timestamps, entity confidence

5. **UI Demo**
   - Streamlit app: paste notes → structured output + entity highlights
   - Optional integration: pass key features to ICU predictor for risk calculation

6. **Evaluation**
   - Compare rule-based vs. transformer extraction
   - Accuracy, F1-score, precision, recall for entities
   - Qualitative review by simulated clinical use cases
   - Optional: delta in ICU predictor metrics with NLP features

---

## 5. Timeline (6–8 Weeks)

| Week | Task |
|------|------|
| 1 | Collect & preprocess ICU notes, define target entities |
| 2 | Implement rule-based extraction & scispaCy NER baseline |
| 3 | Fine-tune ClinicalBERT on annotated dataset |
| 4 | Map extracted entities to SNOMED, generate FHIR JSON |
| 5 | Build Streamlit demo, integrate pipeline for per-note inference |
| 6 | Evaluate extraction accuracy, qualitative review |
| 7 | Optional: integrate selected features into ICU predictor, run ablation |
| 8 | Write README, document code, finalize demo & repository |

> Note: If ICU integration is skipped, project completes in ~6 weeks

---

## 6. File Structure


NLP_ICU_Extractor/
├─ data/
│  ├─ raw_notes/          # Original ICU notes
│  └─ annotated/          # 50–300 labelled examples
├─ src/
│  ├─ preprocessing/      # Cleaning & splitting notes
│  ├─ nlp_pipeline/       # Rule-based + scispaCy + ClinicalBERT
│  ├─ ontology_mapping/   # SNOMED mapping & FHIR JSON
│  ├─ ui_demo/            # Streamlit interface
│  └─ integration/        # Optional ICU predictor feature integration
├─ outputs/
│  ├─ structured_json/    # FHIR-compatible outputs
│  └─ metrics/            # Evaluation metrics + logs
├─ README.md
├─ requirements.txt
└─ notes.md               # Development & planning notes


---

## 7. Evaluation Plan

- **Entity-level metrics:** precision, recall, F1
- **Transformer performance:** ClinicalBERT accuracy on 50–300 labelled examples
- **Rule-based coverage vs transformer coverage**
- **Clinical relevance:** case-level validation against simulated clinical scenarios
- **Optional ICU integration:** delta in ICU predictor metrics with vs. without NLP features

---

## 8. README Outline

1. Project description & clinical motivation
2. Technical overview (pipeline + models)
3. Dataset & annotation details
4. Pipeline walkthrough: preprocessing → extraction → mapping → UI
5. Evaluation metrics & results
6. Optional: ICU predictor integration results
7. How to run (environment, pipeline, demo)
8. Repository structure
9. Requirements & dependencies
10. License & acknowledgements

---

## 9. Justification

- **Clinical relevance:** Extracting structured data from ICU notes is a critical workflow gap in healthcare ML
- **Technical complexity:** NLP on unstructured clinical text + transformer fine-tuning
- **Portfolio impact:** Demonstrates full-stack healthcare ML: unstructured → structured → deployment → optional multi-modal integration
- **Recruiter appeal:** Highlights real-world skills in NLP, transformers, clinical ontologies, interoperability standards, and UI development
- **Manageable scope:** Focuses on high-value components; optional advanced feature (ICU integration) maximizes impact without overwhelming timeline