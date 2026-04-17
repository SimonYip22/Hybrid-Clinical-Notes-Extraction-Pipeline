

# Phase 8 — Full Dataset Inference (160,000 Reports)

## Objective
Demonstrate real-world scalability

## Pipeline
160k reports
↓
rule-based extraction
↓
transformer validation (with threshold)
↓
final entities (~500k expected)

## Notes
- No labels used here
- This is NOT evaluation

## Output
- Final extracted dataset
- Aggregate statistics

---

# Phase 9 — Deployment

## Objective
Provide usable inference pipeline

## Minimum Viable Setup

### Option 1 — CLI
python run_pipeline.py input.txt
### Option 2 — Simple API (optional)
- FastAPI endpoint

## Hosting (optional)
- Lightweight platforms:
  - Render / Railway / HuggingFace Spaces

---

# Phase 10 — Batch vs Individual Inference

## Clarification
Same pipeline, different usage modes:

### Individual
- 1 report → output
- Used for demo/UI

### Batch
- N reports → outputs
- Used for 160k dataset

## Implementation
- Single pipeline wrapped in loop

---

# Phase 11 — CI/CD (Optional)

## Minimal Version
- GitHub repository
- README with usage instructions

## Optional
- Basic GitHub Actions for:
  - Linting
  - Script testing

## Not required for project success

---

# Final Execution Order

1. Threshold tuning (CV predictions)
2. Train final model (1020 samples)
3. Evaluate on test set (180 samples)
4. Generate visualisations

7. Run full 160k dataset
8. Build simple deployment interface

---

# Final Output Summary

## You will produce:
- Final trained model
- Fixed decision threshold
- Test set metrics (primary results)
- Visualisations
- Large-scale dataset (160k processed)
- Simple deployment pipeline

---