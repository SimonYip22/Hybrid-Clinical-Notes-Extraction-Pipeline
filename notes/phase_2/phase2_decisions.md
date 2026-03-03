# Phase 2 — Deterministic Rule-Based Extraction Decisions

## Objective

This document defines the methodological decisions governing execution of Phase 2 (Deterministic Rule-Based Extraction) which builds the extraction engine that all subsequent phases depend upon.

Phase 2 constructs the deterministic extraction backbone of the system.

Its purpose is to transform raw ICU early-note text into structurally segmented, artefact-normalized, and schema-aligned JSON representations using fully reproducible rule-based logic.

Deterministic, section-aware extraction of 4 predefined clinical entity types with span alignment and negation handling.

Implement a fully deterministic, span-preserving, section-aware extraction pipeline for four predefined clinical entity types:

	1.	SYMPTOM – patient-reported or clinician-observed manifestations
	2.	INTERVENTION – actions performed by clinicians
	3.	COMPLICATION – adverse events or pathological changes
	4.	VITAL_MENTION – physiological measurements

This phase establishes a reproducible, auditable extraction backbone.
No probabilistic models, transformers, or statistical evaluation are introduced at this stage.

---

Step
Why it comes here
1. Schema Operationalisation
Define exactly what you extract (the 4 entities). Freeze scope to prevent scope creep.
2. Preprocessing Layer
Clean text without breaking character offsets; this ensures everything downstream works correctly.
3. Section Detection
Map text spans to sections; improves interpretability and allows attaching entities to context.
4. Rule Extraction Engine
Apply deterministic rules to extract candidate entities for each type.
5. Negation Detection
Add flags to entities; can only do this after rules find them.
6. JSON Output Construction
Wrap everything in a reproducible format for downstream use.
7. Deterministic Stability Testing
Verify that rules + preprocessing + negation + JSON work correctly; prevents surprises in Phase 3/5.


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

---

## 4.3 Summary

Phase 2 extraction will only target these four entities to:

- Ensure manageable rule creation and high precision  
- Prevent overlap and ambiguity between entity types  
- Provide structured, span-aligned JSON outputs ready for transformer validation in Phase 3  

This finalises the scope of entity extraction for Phase 2.
