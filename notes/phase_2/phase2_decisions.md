# Phase 2 — Deterministic Rule-Based Extraction Decisions

## Objective

This document defines the methodological decisions governing execution of Phase 2: Deterministic Rule-Based Extraction, which constructs the structured information extraction backbone of the project.

Phase 2 transforms raw ICU early-note clinical text into structured, span-aligned JSON representations using deterministic rule-based methods. The purpose of this phase is to convert unstructured narrative text into reproducible structured data that can be consumed by later modelling stages.

The extraction system operates using fully deterministic logic, ensuring that identical input text always produces identical output. No probabilistic models, machine learning systems, or transformer architectures are introduced at this stage.

The pipeline performs the following core functions:

- artefact-aware text preprocessing while preserving character offsets
- section segmentation of clinical notes
- deterministic extraction of predefined clinical entities
- negation detection for extracted entities
- structured JSON output generation

The extraction targets four predefined clinical entity categories:

1. **SYMPTOM** — patient-reported complaints or clinician-observed manifestations  
2. **INTERVENTION** — therapeutic or procedural actions performed by clinicians  
3. **COMPLICATION** — adverse events or pathological developments  
4. **VITAL_MENTION** — explicit physiological measurements or abnormal vital descriptors

These entities are extracted with span alignment, meaning the character offsets of each entity are preserved relative to the original source text. This allows downstream components to trace structured outputs back to their exact locations in the clinical note.

The goal of Phase 2 is therefore to establish a reproducible, auditable, and deterministic extraction backbone upon which all later phases (transformer validation, modelling, and deployment) depend.

No statistical modelling or predictive evaluation occurs during this phase.

---

## Phase 2 Operational Pipeline

Phase 2 is implemented as a sequential pipeline of deterministic processing stages.  
Each stage prepares the data required by the next stage.

| Step | Component | Purpose |
|-----|-----|-----|
| 1 | Schema Operationalisation | Define the exact clinical entities to extract and freeze scope to prevent uncontrolled expansion. |
| 2 | Preprocessing Layer | Normalize artefacts (e.g. headers, formatting, encoding issues) while preserving original character offsets. |
| 3 | Section Detection | Identify structural sections within clinical notes (e.g. assessment, plan, vitals) and map text spans to section labels. |
| 4 | Rule-Based Extraction Engine | Apply deterministic pattern rules to identify candidate entities belonging to the predefined schema. |
| 5 | Negation Detection | Detect negation cues (e.g. "no", "denies", "without") affecting extracted entities and attach negation flags. |
| 6 | JSON Output Construction | Convert extracted entities into a standardized schema-aligned JSON representation. |
| 7 | Deterministic Stability Testing | Verify reproducibility and structural correctness of the full pipeline before downstream modelling. |

This ordering is intentional.

Extraction rules depend on clean text and known section boundaries, while negation detection requires identified entities to operate on. JSON output construction occurs only after the extraction and negation steps are complete.

The final step ensures that the entire pipeline is deterministic, stable, and structurally valid before any downstream evaluation or modelling occurs.

---

# Entity Schema Decisions

## Objective

- Define the four clinically meaningful entity types to extract from ICU notes, finalising scope for Phase 2 deterministic extraction. 
- These entities form the foundation of structured JSON outputs for downstream use.

---

## 4.1 Entity Types Overview

Extraction in Phase 2 is strictly limited to four entity types:

- **SYMPTOM**  
- **INTERVENTION**  
- **COMPLICATION**  
- **VITAL_MENTION**

These four were chosen because they capture the core clinically relevant information commonly reported in ICU progress notes, balancing scope control and portfolio complementarity with other ongoing projects. Limiting to these avoids over-complexity, prevents scope creep, and ensures high precision in rule-based extraction.

---

## 4.2 Entity Details

### 4.2.1 SYMPTOM

**Purpose:** Capture patient-reported complaints or clinician-observed manifestations.  
**Rationale:** Symptoms represent the subjective or observable clinical state that informs interventions and complications.

**Operational Decisions:**

- **Inclusions:**  
  - Patient-reported complaints (e.g., "chest pain")  
  - Observed clinical signs (e.g., "confused")  

- **Exclusions:**  
  - Lab or imaging values  
  - Procedures  
  - Baseline chronic diagnoses  

- **Negation Handling:**  
  - "no", "denies", "without", "not"  

- **Trigger/Pattern Notes:**  
  - Common clinical descriptors and synonymous terms captured  
  - Avoid overlap with other entity types  

**Positive Extraction Examples**

The following phrases should produce a SYMPTOM extraction:

- "patient reports chest pain"
- "complaining of nausea overnight"
- "severe headache this morning"
- "patient feeling dizzy"
- "persistent shortness of breath"

**Negative / Non-Extraction Examples**

The following phrases should NOT produce a SYMPTOM extraction:

- "history of migraine"
- "CT shows intracranial hemorrhage"
- "labs notable for elevated troponin"
- "scheduled for CT scan"

These represent diagnoses, tests, or history rather than symptoms.

**Boundary Resolution**

Ambiguous terms are resolved as follows:

- tachycardia → VITAL_MENTION (not SYMPTOM)
- hypotension → VITAL_MENTION
- delirium → SYMPTOM
- agitation → SYMPTOM

---

### 4.2.2 INTERVENTION

**Purpose:** Capture therapeutic or procedural actions performed.  
**Rationale:** Interventions document treatments and procedures, critical for understanding patient management.

**Operational Decisions:**

- **Inclusions:**  
  - Medication initiation (e.g., "started norepinephrine")  
  - Procedures (e.g., "central line inserted", "intubated")  

- **Exclusions:**  
  - Hypothetical plans or recommendations not performed  

- **Negation Handling:**  
  - Same cues as SYMPTOM  

- **Trigger/Pattern Notes:**  
  - Focus on high-confidence deterministic patterns  
  - Avoid misclassifying narrative suggestions or future plans  

**Positive Extraction Examples**

The following phrases should produce an INTERVENTION extraction:

- "started norepinephrine"
- "intubated for airway protection"
- "central line inserted"
- "patient placed on ventilator"
- "administered antibiotics"

**Negative / Non-Extraction Examples**

The following phrases should NOT produce an INTERVENTION extraction:

- "plan to start antibiotics"
- "consider dialysis"
- "may require intubation"
- "recommend central line placement"

These represent future plans or recommendations rather than completed interventions.

**Boundary Resolution**

Ambiguous phrases are resolved as follows:

- "started antibiotics" → INTERVENTION
- "on antibiotics" → INTERVENTION
- "intubation planned" → NOT extracted

---

### 4.2.3 COMPLICATION

**Purpose:** Capture adverse or pathological developments during ICU stay.  
**Rationale:** Complications indicate negative outcomes or new pathological events, essential for downstream analysis and evaluation.

**Operational Decisions:**

- **Inclusions:**  
  - Acute pathological conditions (e.g., "AKI", "sepsis", "pneumothorax")  

- **Exclusions:**  
  - Chronic baseline diagnoses without acute worsening  

- **Negation Handling:**  
  - Same as above  

- **Trigger/Pattern Notes:**  
  - Detect keywords and common abbreviations for acute events  
  - Maintain clear separation from INTERVENTION or SYMPTOM  

**Positive Extraction Examples**

The following phrases should produce a COMPLICATION extraction:

- "developed sepsis overnight"
- "new acute kidney injury"
- "patient with pneumothorax"
- "episode of ventricular arrhythmia"

**Negative / Non-Extraction Examples**

The following phrases should NOT produce a COMPLICATION extraction:

- "history of chronic kidney disease"
- "prior stroke"
- "family history of cancer"

These represent baseline or historical conditions.

**Boundary Resolution**

Ambiguous terms are resolved as follows:

- AKI → COMPLICATION
- sepsis → COMPLICATION
- infection → COMPLICATION
- chronic heart failure → NOT extracted unless acute worsening is described

---

### 4.2.4 VITAL_MENTION

**Purpose:** Capture explicit physiological measurements or abnormal descriptors.  
**Rationale:** Vital signs provide objective, structured information embedded in narrative, useful for linking symptoms, interventions, and complications.

**Operational Decisions:**

- **Inclusions:**  
  - Measurement + value (e.g., "BP 85/50")  
  - Descriptors indicating abnormality (e.g., "tachycardic")  

- **Exclusions:**  
  - Lab-only results (covered elsewhere)  

- **Negation Handling:**  
  - Same as above  

- **Trigger/Pattern Notes:**  
  - Regex patterns targeting numeric values, units, and common shorthand  
  - Ensure alignment with section and sentence spans  

**Positive Extraction Examples**

The following phrases should produce a VITAL_MENTION extraction:

- "BP 85/50"
- "HR 120"
- "Temp 38.5"
- "tachycardic to 130"
- "oxygen saturation 92%"

**Negative / Non-Extraction Examples**

The following phrases should NOT produce a VITAL_MENTION extraction:

- "labs notable for creatinine 2.0"
- "WBC elevated"
- "troponin increased"

These represent laboratory values rather than vital signs.

**Boundary Resolution**

Ambiguous terms are resolved as follows:

- tachycardia → VITAL_MENTION
- hypotension → VITAL_MENTION
- hypoxia → VITAL_MENTION
- fever → SYMPTOM unless an explicit temperature value is present

---

## 4.3 Summary

Phase 2 extraction will only target these four entities to:

- Ensure manageable rule creation and high precision  
- Prevent overlap and ambiguity between entity types  
- Provide structured, span-aligned JSON outputs ready for transformer validation in Phase 3  

This finalises the scope of entity extraction for Phase 2.
