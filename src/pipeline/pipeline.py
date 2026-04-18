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