# Hybrid Clinical Notes Extraction Pipeline

***Precision-First Natural Language Processing Architecture Using ClinicalBERT Classifier & Rule-Based Entity Extraction for Structuring Clinical Notes***

---

## Executive Summary

**Tech stack:** ***Python, PyTorch, HuggingFace Transformers, scikit-learn***

This repository presents a dual-architecture natural language processing (NLP) pipeline combining deterministic rule-based entity extraction with context-aware transformer (ClinicalBERT) validation to produce structured, schema-aligned JSON outputs from 162,296 unstructured ICU progress reports over 32,910 unique ICU stays, suitable for downstream machine learning applications.

Rule-based extraction ensures high-precision identification of clinically relevant patterns, while the fine-tuned ClinicalBERT classifier provides negation handling, entity validation, and contextual disambiguation.

This focused on classical clinical NLP pipeline development rather than ontology mapping, interoperability standards, or production deployment. Model training was performed using credentialed access to the MIMIC-III dataset via PhysioNet, ensuring realistic clinical data handling.

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

