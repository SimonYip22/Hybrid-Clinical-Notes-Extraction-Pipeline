"""
validation.py

Purpose:
    Transformer-based validation of extracted clinical entities.

Workflow:
    1. Construct model input text from structured entity fields
    2. Run batched inference using pretrained transformer
    3. Convert probabilities to boolean predictions via thresholding
    4. Insert results back into entity structure

Interface:
    - validate_entities(): batch validation of extracted entities

Inputs:
    - entities: List of extracted entity dictionaries
    - model, tokenizer, device (preloaded)
    - threshold, batch_size, max_length (optional)

Outputs:
    List[Dict[str, Any]]
        - Same entities with populated validation fields:
            - confidence (float)
            - is_valid (bool)

Notes:
    - Designed for precision filtering of rule-based outputs
    - Preserves original entity structure (no schema transformation)
    - Supports efficient batch processing for large datasets
"""

from typing import List, Dict, Any
import torch
from tqdm import tqdm

# ------------------------------------------------------------
# VALIDATION FUNCTION
# ------------------------------------------------------------
def validate_entities(
    entities: List[Dict[str, Any]],
    model,
    tokenizer,
    device,
    threshold: float = 0.549,
    batch_size: int = 16,
    max_length: int = 512
) -> List[Dict[str, Any]]:
    """
    Apply transformer-based validation to extracted entities.

    Purpose:
        Assign a confidence score and binary validity label to each entity.

    Workflow:
        1. Construct model input text from structured entity fields
        2. Perform batched tokenisation and inference
        3. Convert probabilities to boolean predictions via thresholding
        4. Insert results into the entity "validation" field

    Args:
        entities: List of extracted entity dictionaries
        model: Preloaded transformer model (inference mode)
        tokenizer: Corresponding tokenizer
        device: Torch device (CPU or GPU)
        threshold: Probability cutoff for validity classification
        batch_size: Number of samples per inference batch
        max_length: Maximum token length for model input

    Returns:
        List of entities with updated validation fields:
            - validation["confidence"]: float
            - validation["is_valid"]: bool
    """
    if len(entities) == 0:
        return entities

    model.eval()

    # -------------------------
    # 1. Build input texts
    # -------------------------
    texts = [
        (
            f"[SECTION] {e['section']} "
            f"[ENTITY TYPE] {e['entity_type']} "
            f"[ENTITY] {e['entity_text']} "
            f"[CONCEPT] {e['concept']} "
            f"[TASK] {e.get('validation', {}).get('task', "")} "
            f"[TEXT] {e['sentence_text']}"
        )
        for e in entities
    ]

    probs = []

    # -------------------------
    # 2. Batched inference
    # -------------------------
    for i in tqdm(range(0, len(texts), batch_size)):

        batch_texts = texts[i:i+batch_size]

        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            batch_probs = torch.softmax(logits, dim=1)[:, 1]

        probs.extend(batch_probs.cpu().numpy())

    # -------------------------
    # 3. Write back into entities
    # -------------------------
    for entity, prob in zip(entities, probs):

        pred = bool(prob >= threshold)

        if "validation" not in entity:
            entity["validation"] = {}

        entity["validation"]["confidence"] = float(prob)
        entity["validation"]["is_valid"] = pred

    return entities