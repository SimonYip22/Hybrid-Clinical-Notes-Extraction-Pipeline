# Hybrid Clinical Notes Extraction Pipeline

***Hybrid Natural Language Processing Architecture Using ClinicalBERT Classifier & Rule-Based Entity Extraction for Structured Clinical Data***

---

## Executive Summary

**Tech stack:** ***Python, PyTorch, HuggingFace Transformers, scikit-learn***

This project implements a precision-first clinical natural language processing (NLP) pipeline for extracting structured information from unstructured ICU clinical notes. The system combines deterministic rule-based entity extraction with context-aware transformer classification (ClinicalBERT) to produce clean, auditable, machine-readable outputs suitable for downstream machine learning applications.

The dual-architecture pipeline converts ICU progress notes into structured clinical entities, with outputs following a standardised JSON schema. Rule-based extraction ensures high-precision identification of clinically relevant patterns, while the ClinicalBERT classifier provides contextual disambiguation and validation of ambiguous spans.

The project focuses on classical clinical NLP pipeline development rather than ontology mapping, interoperability standards, or production deployment. Model training was performed using credentialed access to the MIMIC-IV dataset via PhysioNet, ensuring realistic clinical data handling and scenario simulation.

---

## Table of Contents

---

## 1. Clinical Background & Motivation

3. Cross-reference projects lightly

In each README:
	•	One short “Related work” or “Context” section
	•	No deep linking or shared code

This reinforces cohesion without coupling.

---

## 2. Project Goals & Contributions

### 2.1 Primary Objectives

##
### 2.2 Key Technical Contributions

---

## 3. Phase 1

---

## 4. ML Modelling Strategy

---

## 5. Phase 2

---

## 6. Phase 3

---

## 7. Methodological Rationale and Design Reflection

---

## 8. Limitations

---

## 9. Future Work

### 9.1 Overview & Rationale

---

## 10. Potential Clinical Integration

---

## 11. Repository Structure

---

## 12. How to Run

---

## 13. Requirements & Dependencies

---

## 14. License

This project is licensed under the MIT License; see the [LICENSE](LICENSE) file for details

---

## 15. Copyright

---

## 16. Citation

---

## 20. Acknowledgments

**ChatGPT**
- Provided guidance throughout the project, including code explanations, debugging, project structure and architectural design

All other components, including Python scripts, preprocessing, model training, and visualisations, were developed by the author. No additional proprietary datasets, papers, or external tutorials were required beyond those cited above

---

**Project Status:** Core Development Complete  
**Last Updated:** 3rd February 2026  
**Author & Maintainer:** Simon Yip - simon.yip@city.ac.uk

---

