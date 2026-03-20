"""
section_extraction.py

Purpose:
    Extract clinically relevant sections from unstructured clinical notes
    using a deterministic, canonical-header-based approach.

    Only predefined canonical headers are recognised. These headers define
    section boundaries. All non-canonical text, including header-like patterns 
    (e.g., subsections, vitals, labs), is treated as normal content and retained 
    within the current section.

Workflow:
    1. Split the clinical note into lines.
    2. Identify canonical headers only.
    3. Start a new section when a canonical header is encountered.
    4. Accumulate all subsequent text until the next canonical header.
    5. Include inline content after header colons where present.
    6. Return extracted sections as a dictionary.

Output:
    Dictionary mapping canonical section headers (lowercase) to text.

    Example:
    {
        "chief complaint": "...",
        "hpi": "...",
        "assessment": "...",
        "plan": "..."
    }

    All section headers are stored in lowercase to ensure consistent canonical 
    representation.
"""

# ---------------------------------------------------------------------
# CANONICAL SECTION DEFINITIONS
# ---------------------------------------------------------------------

# Canonical section headers to extract, based on empirical analysis of the dataset, lowercased for case-insensitive matching
CANONICAL_HEADERS = [
    "plan",
    "assessment",
    "action",
    "response",
    "assessment and plan",
    "chief complaint",
    "hpi",
    "past medical history",
    "family history",
    "social history",
    "review of systems",
    "physical examination",
    "disposition"
]

CANONICAL_HEADER_SET = set(CANONICAL_HEADERS)


# ---------------------------------------------------------------------
# CANONICAL HEADER DETECTION FUNCTION
# ---------------------------------------------------------------------

def match_canonical_header(line):
    """
    Check if a line matches a canonical header.

    Supports:
    - "Header:"
    - "Header"
    - "Header: content"

    Returns:
        (header_lower, inline_text) or (None, None)
    """

    stripped = line.strip()

    # Empty line case
    if not stripped:
        return None, None # No header, no content

    # Split on first colon (if exists)
    if ":" in stripped:
        # Split into header and inline content
        header_part, rest = stripped.split(":", 1) # Split only on first colon
        # Clean header for matching
        header_clean = header_part.strip().lower()

        # Check if cleaned header is in canonical set
        if header_clean in CANONICAL_HEADER_SET:
            # Return header and inline content
            return header_clean, rest.strip()

    # No colon case (standalone header)
    header_clean = stripped.lower()
    if header_clean in CANONICAL_HEADER_SET:
        return header_clean, None

    # Not a canonical header
    return None, None


# ---------------------------------------------------------------------
# SECTION EXTRACTION FUNCTION
# ---------------------------------------------------------------------
def extract_sections(report):
    """
    Extract canonical sections using canonical-only boundaries.

    Rules:
    - Only canonical headers define section boundaries
    - Non-canonical header-like text is treated as normal content
    - Inline content after header colon is included
    """

    # Empty sections dictionary
    sections = {}
    # Track current section header and content buffer
    current_header = None
    # Temporary storage for accumulating lines of the current section
    buffer = []

    # Process the report line by line
    for line in report.split("\n"):

        # Check for canonical header match
        header, inline_text = match_canonical_header(line)

        # If a canonical header is detected
        if header:
            # If we were already in a section, save it
            if current_header is not None:
                # Join accumulated lines into a single string
                content = " ".join(buffer).strip()
                # If current header already exists in section
                if current_header in sections:
                    # Append content to existing section
                    sections[current_header] += " " + content
                else:
                    # Create a new saved section entry
                    sections[current_header] = content

            # Start new section with detected header and reset buffer
            current_header = header
            buffer = []

            # Add inline text if present after the header
            if inline_text:
                buffer.append(inline_text)

            continue

        # If no header is detected it must be content
        if current_header is not None:
            # Clean the line
            clean = line.strip()
            # Only add non-empty lines to the buffer
            if clean:
                buffer.append(clean)

    # After loop ends, save the last section if it exists
    if current_header is not None:
        content = " ".join(buffer).strip()

        if current_header in sections:
            sections[current_header] += " " + content
        else:
            sections[current_header] = content

    # Return the dictionary of extracted sections
    return sections