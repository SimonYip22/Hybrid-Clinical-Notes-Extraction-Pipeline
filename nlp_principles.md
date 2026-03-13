# NLP System Design Principles

*A first-principles guide to understanding how NLP systems are conceptualised, structured, and designed in practice.*

---

## 0. Purpose of This Document

This document provides a structured framework for understanding NLP system design from first principles. It is **not** a textbook summary, a list of models, or a deep mathematical derivation.  

Its goals:

- Explain what an NLP system is fundamentally.
- Define major design paradigms and their trade-offs.
- Clarify common NLP tasks and their structural implications.
- Explain evaluation methods and their influence on design.
- Describe how transformers and hybrid architectures are used in practice.
- Provide a decision-oriented mindset for designing new NLP projects.

The aim is to move from project-specific execution to **general architectural reasoning**, so you can design, justify, and critically assess NLP systems independently.

---

# 1. What Is an NLP System Fundamentally?

At its core, an NLP system converts unstructured text into structured or actionable outputs using a pipeline:

**Input → Tokenisation → Representation → Transformation → Post-processing → Output → Evaluation**

Each stage has a distinct role:

---

## 1.1 Input: Raw Text

The system receives unstructured text, for example:

- Clinical notes
- Emails or support tickets
- Product reviews
- Legal documents
- Social media posts

Key considerations:

- **Format:** plain text, HTML, PDF, scanned documents (OCR)
- **Domain:** medical, legal, e-commerce — affects vocabulary and style
- **Noise:** typos, abbreviations, inconsistent punctuation, formatting
- **Length:** short snippets vs multi-page notes

Design begins by understanding **constraints and variability** in the input.

---

## 1.2 Tokenisation

**Definition:** Tokenisation breaks text into discrete units (tokens) for model or rule processing.

Types:

- **Character-level:** each character is a token (useful for OCR/noisy text)
- **Word-level:** splits by spaces or punctuation
- **Subword-level:** breaks words into common subunits, e.g., BPE (Byte-Pair Encoding) or WordPiece (used by BERT)

**Why it matters:**

- Determines span alignment for entity extraction
- Models operate on tokens, not raw text
- Influences interpretability, precision, and recall

**Example:** `"BP 120/80 stable"`  
- Word tokens: `["BP", "120/80", "stable"]`  
- Subword tokens: `["BP", "120", "/", "80", "stable"]`

Tokenisation is non-trivial in clinical or technical domains due to abbreviations, hyphens, numeric patterns, and non-standard punctuation.

---

## 1.3 Representation

**Definition:** Representation transforms tokens into structured signals interpretable by models or rules.

- **Rule-based:** raw strings, lexicons, dictionaries, pattern lists
- **Classical ML:** numeric feature vectors derived from token statistics, part-of-speech tags, or dependency relations
- **Deep learning:** learned dense embeddings representing words or subwords
- **Transformers:** context-aware embeddings that capture meaning based on surrounding text

Representation is effectively **what the system “understands”** about the text.

---

## 1.4 Transformation (Core Logic)

**Definition:** Transformation maps representations to task-specific outputs.

Examples:

- **Regex (regular expressions):** patterns that match character sequences, e.g., detecting blood pressure with `r"\bBP\s\d{2,3}/\d{2,3}\b"`
- **Classical ML:** logistic regression, SVM, CRF (Conditional Random Fields), HMM
- **Deep learning:** BiLSTM, Transformer token classifiers
- **Generative models:** GPT-style LLMs producing text outputs

**Design choices depend on:**

- Task type (NER, classification, relation extraction)
- Data availability
- Error tolerance
- Interpretability and auditability requirements

---

## 1.5 Post-Processing

Raw outputs often require cleaning and refinement:

- Span validation (align predictions to token boundaries)
- Deduplication (remove repeated entities)
- Type enforcement (ensure entity labels are valid)
- Schema formatting (convert to JSON, CSV, or database-ready structure)
- Confidence thresholding (filter low-confidence outputs)

Post-processing often **recovers precision**, especially in regulated domains like healthcare.

---

## 1.6 Evaluation

Evaluation defines what "success" means and directly affects architecture.

Consider:

- Correctness criteria: exact match vs partial overlap
- Type matching: entity labels must align
- Precision vs recall trade-off
- Micro vs macro metrics

Example: In ICU note extraction, **precision is often prioritised** because false positives can lead to incorrect clinical decisions.

---

# 2. Major NLP Paradigms

---

## 2.1 Rule-Based Systems

**Definition:** Deterministic systems using patterns, dictionaries, and explicit rules.

**Examples:** regex, lexicons, keyword lists

**How it works:** The system searches text for predefined patterns or matches against dictionaries. Rules often include conditions and negation handling.

**Strengths:**

- High precision when rules are correct
- No labelled data required
- Fully interpretable and auditable
- Suitable for regulated or safety-critical domains

**Weaknesses:**

- Fragile to new phrasing
- High maintenance cost
- Poor generalisation beyond initial scope

**Use cases:** ICU notes, lab reports, legal compliance, finance text.

---

## 2.2 Statistical / Classical Machine Learning

**Definition:** Models learn relationships between engineered features and outputs.

**Examples:** logistic regression, SVM, CRF, HMM

**How it works:** Text is converted into numeric features (counts, TF-IDF, POS tags), which are then fed into a supervised model trained on labelled examples.

**Strengths:**

- Better generalisation than pure rules
- Reduces manual rule writing
- Models probabilistic relationships

**Weaknesses:**

- Requires labelled data
- Feature engineering can be complex
- Contextual understanding limited

**Use cases:** Sentence classification, token sequence labelling with moderate data availability.

---

## 2.3 Deep Learning (Sequence Tagging)

**Definition:** Neural networks learn representations automatically to label tokens or spans.

**Examples:** BiLSTM-CRF, Transformer token classifiers

**How it works:** Networks take sequences of embeddings and output token-level predictions. CRF layers enforce structured output consistency.

**Strengths:**

- Captures long-range dependencies
- Learns complex contextual patterns
- Handles ambiguity better than classical ML

**Weaknesses:**

- Requires large labelled datasets
- Less interpretable
- Harder to audit in regulated settings

---

## 2.4 Generative LLM Systems

**Definition:** Large language models generate text outputs conditioned on prompts or instructions.

**Examples:** GPT, instruction-tuned LLMs, RAG

**How it works:** The model generates sequences token-by-token based on input context. Few-shot or zero-shot prompts allow flexible behaviour.

**Strengths:**

- Rapid prototyping
- Handles open-ended, creative tasks
- Can generalise across multiple tasks with minimal data

**Weaknesses:**

- Output may hallucinate or be inaccurate
- Difficult to constrain for deterministic use
- Risky for regulated environments

---

## 2.5 Hybrid Systems

**Definition:** Combine rule-based and neural paradigms for controlled outcomes.

**Example architecture:**

1. **Rule-based candidate generation:** high-precision spans
2. **Transformer validation layer:** filters false positives, confirms context
3. **Schema enforcement:** deterministic formatting to output JSON

**Strengths:**

- Precision-focused
- Low annotation requirement
- Auditable and safe for regulated domains

**Use cases:** Clinical information extraction, legal entity extraction, critical QA pipelines.

---

# 3. Common NLP Task Categories

---

## 3.1 Named Entity Recognition (NER)

Detect and classify spans of text.

**Output:** span + type

**Example:** `"Patient has sepsis."` → `sepsis (COMPLICATION)`

---

## 3.2 Relation Extraction

Detect relationships between entities.

**Example:** `(sepsis) CAUSED_BY (infection)`

---

## 3.3 Text Classification

Assign a label to an entire sentence or document.

**Example:** Sentiment analysis: `"I feel sick"` → `negative`

---

## 3.4 Span Classification

Classify candidate spans as valid or invalid.  
Often used in hybrid validation pipelines.

---

## 3.5 Sequence Tagging

Token-level labelling: `B-SYMPTOM`, `I-SYMPTOM`, `O`  
Requires structured supervision and labelled sequences.

---

## 3.6 Information Extraction

Broad category of extracting structured entities, relations, and attributes from unstructured text.

**Example:** ICU notes → SYMPTOM, INTERVENTION, COMPLICATION, VITAL

---

## 3.7 Generation Tasks

- Summarisation
- Question answering
- Report generation

Focus is on generating text rather than extracting structured outputs.

---

# 4. Design Considerations in Real Systems

Questions that guide architecture:

- **Error tolerance:** low → deterministic rules; high → generative models may be acceptable
- **Annotation budget:** small → rule-based or hybrid; large → deep learning feasible
- **Regulatory risk:** high → auditability, interpretability required
- **Precision vs recall:** clinical often prioritises precision
- **Domain stability:** stable → rules; evolving → learning-based
- **Maintenance cost:** rules need manual updates; models need retraining

---

# 5. Evaluation Philosophy

- **Span matching:** exact vs partial vs token-level  
- **Micro vs macro F1:** micro favours frequent entities; macro averages across types  
- **Precision-first in clinical domains**  
- **AUC often irrelevant** for extraction tasks

---

# 6. Transformer Usage Patterns

- **Sequence tagging:** token-level predictions  
- **Span classification:** sentence + candidate span → valid/invalid  
- **Frozen embeddings:** pretrained embeddings used without fine-tuning  
- **Prompt-based extraction:** LLM prompted to output entities  
- **Zero-shot vs supervised:** trade-offs in annotation cost and reliability  
- **Validation filters:** transformers used to confirm outputs, not primary extraction

---

# 7. How to Think When Designing an NLP System

Before selecting a model:

1. What is the task?  
2. What error profile is acceptable?  
3. How much labelled data exists?  
4. How stable is the domain?  
5. Is interpretability required?  
6. Who consumes the outputs?  
7. How is success measured?

Architecture follows **constraints**, not default models.

---

# 8. From Implementation to Architectural Reasoning

Shift from following instructions to independent reasoning:

- Recognise task category
- Evaluate data constraints
- Balance precision and recall
- Plan for maintenance
- Justify trade-offs

---

# 9. Outcome

After studying this, you will be able to:

- Design NLP systems independently
- Justify architecture choices
- Decide between rule-based, classical ML, deep learning, or hybrid
- Estimate annotation requirements
- Anticipate evaluation pitfalls
- Reason in terms of **pipelines, constraints, and trade-offs**, not just models