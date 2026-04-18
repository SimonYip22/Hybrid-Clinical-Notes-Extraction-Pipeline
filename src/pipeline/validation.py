from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

def validate_entities(
    entities: List[Dict[str, Any]],
    model,
    tokenizer,
    device,
    threshold: float = 0.549,
    batch_size: int = 16,
    max_length: int = 512
) -> List[Dict[str, Any]]:

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
            f"[TASK] {e.get("validation", {}).get("task", "")} "
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

        entity["validation"]["confidence"] = float(prob)
        entity["validation"]["is_valid"] = pred

    return entities