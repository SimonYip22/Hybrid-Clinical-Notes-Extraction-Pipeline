"""
sentence_segmentation.py

Purpose:
    Split section-level clinical text into sentence-level spans while preserving
    character offsets relative to the original section text.

    This enables:
    - Accurate mapping of regex matches (local sentence offsets → global text offsets)
    - Sentence-level context extraction for downstream validation (e.g. transformer models)
    - Deterministic and reproducible span alignment

Workflow:
    1. Use NLTK's sentence tokenizer to split the input text into sentences
    2. Iterate through each sentence in order
    3. Locate each sentence within the original text using a moving cursor
       to avoid incorrect matches when duplicate sentences exist
    4. Compute start and end character offsets for each sentence
    5. Store sentence text along with its offsets

Output:
    List[Dict], where each element contains:
    {
        "sentence": str,   # sentence text
        "start": int,      # start index in original text
        "end": int         # end index in original text
    }

Notes:
    - Offsets are relative to the input `text` (typically a section, not full note)
    - This function does NOT modify the text
    - Output is used internally for entity extraction and is NOT part of final JSON
    """

from nltk.tokenize import sent_tokenize

import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

def split_into_sentences(text):

    # Step 1: Sentence tokenization using NLTK (handles clinical punctuation better than regex)
    sentences = sent_tokenize(text)

    # Empty list to store sentence text, start, and end
    spans = []
    # Pointer for where to start searching from in the original text
    cursor = 0 

    # Step 2: Loop through sentences
    for sent in sentences:

        # Step 3: Find the sentence in the original text starting from the cursor
        start = text.find(sent, cursor)

        # Step 4: Compute end position based on start + sentence length
        end = start + len(sent)

        # Step 5: Store the sentence with its character spans
        spans.append({
            "sentence": sent,
            "start": start,
            "end": end
        })

        # Step 6: Move the cursor forward to avoid matching the same sentence again
        cursor = end

    return spans