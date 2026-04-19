"""
pipeline.py

Purpose:
    Orchestrate the full extraction + validation pipeline.

Workflow:
    1. Run deterministic extraction
    2. Apply transformer validation
    3. Return structured, validated entities

Inputs:
    - DataFrame with TEXT column (required)
    - Model, tokenizer, device

Outputs:
    List[Dict[str, Any]]
        - Fully structured and validated entity-level records

Notes:
    - Single entry point for all inference (dataset + deployment)
    - No model loading or I/O performed inside this module
"""

from typing import List, Dict, Any
import pandas as pd

from pipeline.extraction import run_extraction_on_dataframe
from pipeline.validation import validate_entities

# ------------------------------------------------------------
# PIPELINE FUNCTION
# ------------------------------------------------------------
def run_pipeline(
    df: pd.DataFrame,
    model,
    tokenizer,
    device,
    threshold: float = 0.549,
    batch_size: int = 16
) -> List[Dict[str, Any]]:
    """
    Execute full pipeline: extraction followed by validation.

    Args:
        df (pd.DataFrame): Input notes (must contain TEXT column)
        model: Preloaded transformer model
        tokenizer: Corresponding tokenizer
        device: Torch device (cpu/cuda)
        threshold (float, optional): Classification threshold
        batch_size (int, optional): Inference batch size

    Returns:
        List[Dict[str, Any]]:
            Validated entity-level outputs
    """
    # -------------------------
    # 1. Rule-based extraction
    # -------------------------
    entities = run_extraction_on_dataframe(df)

    # -------------------------
    # 2. Transformer validation
    # -------------------------
    validated_entities = validate_entities(
        entities=entities,
        model=model,
        tokenizer=tokenizer,
        device=device,
        threshold=threshold,
        batch_size=batch_size
    )

    return validated_entities