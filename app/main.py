"""
main.py

Purpose:
    Expose the clinical NLP pipeline as a stateless FastAPI inference service.

Run Locally:
    pip install -r requirements-api.txt
    PYTHONPATH=src uvicorn app.main:app --reload

Call API (in separate terminal):
    curl -X POST "http://127.0.0.1:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{
    "text": "HPI: Pt c/o CP and SOB. O2 started. Assessment: possible pneumonia."
    }'

Access docs:
    http://127.0.0.1:8000/docs

Endpoints:
    GET /health
        - Health check endpoint for monitoring

    POST /predict
        - Input: raw clinical text
        - Output: structured entity extraction results

Notes:
    - Thin wrapper over pipeline.run_pipeline
    - Model is loaded once at startup for efficiency
    - Stateless design (no persistence, no session tracking)
    - Input validation via Pydantic schema
"""

from fastapi import FastAPI 
from pydantic import BaseModel
import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pipeline.pipeline import run_pipeline

# -------------------------
# INITIALISE API APP
# -------------------------

app = FastAPI()

# -------------------------
# LOAD MODEL (once at startup)
# -------------------------

MODEL_DIR = "models/bioclinicalbert_final"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

DEVICE = torch.device("cpu")
model.to(DEVICE)
model.eval()

# -------------------------
# CONFIG
# -------------------------

THRESHOLD = 0.549
BATCH_SIZE = 16

# -------------------------
# PYDANTIC SCHEMA (INPUT VALIDATION)
# -------------------------

class ReportRequest(BaseModel):
    text: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "HPI: Pt c/o CP and SOB. O2 started. Assessment: possible pneumonia."
            }
        }
    }

# -------------------------
# ENDPOINTS
# -------------------------

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# Prediction endpoint
@app.post("/predict")
def predict(request: ReportRequest) -> dict:

    # Handle empty input case
    if not request.text.strip():
        return {"entities": []}
    
    # Convert input into DataFrame (pipeline requirement)
    df = pd.DataFrame([{
        "TEXT": request.text,
        "SUBJECT_ID": "",
        "HADM_ID": "",
        "ICUSTAY_ID": "",
        "note_id": "note_1"
    }])

    # Run pipeline
    entities = run_pipeline(
        df=df,
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        threshold=THRESHOLD,
        batch_size=BATCH_SIZE
    )

    return {"entities": entities}