NLP_Clinical_Notes_Extractor/
│
├─ data/
│   ├─ raw/                     # Raw ICU notes (read-only)
│   ├─ processed/               # Preprocessed text after cleaning/section splitting
│   └─ labelled/                # Small dataset (50–100 examples) for ClinicalBERT
│
├─ src/
│   ├─ preprocessing/           # Text cleaning, section splitting, abbreviation handling
│   │   └─ preprocess.py
│   ├─ rules/                   # Rule-based entity extraction logic
│   │   └─ rule_extractor.py
│   ├─ transformer/             # ClinicalBERT integration for validation/disambiguation
│   │   └─ clinicalbert_validator.py
│   ├─ utils/                   # Helper functions (logging, JSON schema enforcement)
│   │   └─ utils.py
│   └─ main_pipeline.py         # Orchestrates preprocessing → rules → transformer → JSON output
│
├─ outputs/
│   ├─ extracted_entities/      # JSON outputs from pipeline
│   └─ evaluation/              # Precision/recall/F1 metrics + qualitative inspection logs
│
├─ notes/
│   ├─ phase_0_design.md
│   ├─ phase_1_data_familiarisation.md
│   ├─ phase_2_rule_based.md
│   ├─ phase_3_transformer_validation.md
│   └─ phase_4_evaluation.md
│
├─ notebooks/                   # Optional exploratory notebooks (data inspection, rule experiments)
│
├─ requirements.txt             # All Python dependencies
├─ README.md                    # Final project synthesis and technical explanation
└─ .gitignore