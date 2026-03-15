"""
section_detection.py

Purpose:
    Detect and extract clinically relevant sections from unstructured clinical notes
    using deterministic rule-based header detection.

    The script identifies potential section headers using regular expressions and
    extracts text belonging to a predefined set of canonical section headers.
    Non-canonical headers are still detected so they correctly terminate previous
    sections, but their contents are not stored.

Workflow:
    1. Split the clinical note into individual lines.
    2. Detect potential headers using two regex patterns:
        - Colon-terminated headers (e.g., "Assessment:")
        - Standalone headers on their own line (e.g., "Assessment")
    3. Treat any detected header as a structural boundary that ends the previous section.
    4. Start a new section only if the detected header matches a predefined canonical header.
    5. Accumulate text belonging to canonical sections until another header is encountered.
    6. Return the extracted sections as a dictionary.

Output: 
    Dictionary mapping canonical section headers to their extracted text content.

    Example:
    {
        "chief complaint": "...",
        "hpi": "...",
        "assessment": "...",
        "plan": "..."
    }

    All section headers are stored in lowercase to ensure consistent canonical representation.
"""

import re

# ---------------------------------------------------------------------
# HEADER DETECTION PATTERNS
# ---------------------------------------------------------------------

# 1. Start of line, optional spaces, capture group of letters/numbers/special characters (up to 80 chars), optional spaces, colon, optional spaces
colon_pattern = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 /()\-'&]{0,80})\s*:\s*")
# 2. Start of line, optional spaces, capture group of letters/numbers/special characters (up to 80 chars), optional spaces, end of line
standalone_pattern = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 /()\-\']{1,80})\s*$")


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
# HEADER DETECTION FUNCTION
# ---------------------------------------------------------------------
def detect_header(line):
    """
    Detect whether a line represents a section header.

    A header is identified using two structural patterns:

    1. Colon-terminated headers
       Example:
           "Assessment:"
           "Chief Complaint: Chest pain"

    2. Standalone headers appearing on their own line
       Example:
           "Assessment"
           "Past Medical History"

    The function returns the detected header text without the trailing colon
    and with surrounding whitespace removed.

    Parameters
    ----------
    line : str
        A single line from the clinical note.

    Returns
    -------
    str or None
        The detected header text if the line matches a header pattern,
        otherwise None.
    """
    # First check for colon-terminated headers within lines or standalone lines
    match = colon_pattern.match(line)
    if match:
        # Return the captured header text, stripped of leading/trailing whitespace
        return match.group(1).strip()

    # Second check for standalone headers that may not have a colon but are on their own line
    match = standalone_pattern.match(line)
    if match:
        return match.group(1).strip()

    # No header detected
    return None


# ---------------------------------------------------------------------
# SECTION EXTRACTION LOGIC
# ---------------------------------------------------------------------
def extract_sections(report):
    """
    Extract canonical clinical sections from an unstructured clinical note.

    The function processes the note line-by-line, identifying potential section
    headers and accumulating text belonging to predefined canonical sections.

    Key rules implemented:

    - Any detected header terminates the previous section.
    - Only headers present in the canonical header list initiate a stored section.
    - Non-canonical headers are ignored but still act as structural boundaries.
    - Text appearing on the same line as a header after the colon is included
      as the first line of that section.
    - Empty lines are ignored during text accumulation.

    Parameters
    ----------
    report : str
        Full clinical note text.

    Returns
    -------
    dict
        Dictionary where:
        - Keys are canonical section headers (lowercase).
        - Values are the extracted section text.

    Example
    -------
    Input note:

        Chief Complaint: Chest pain
        HPI:
        Patient reports 2 days of chest discomfort.
        Assessment:
        Possible unstable angina.

    Output:

        {
            "chief complaint": "Chest pain",
            "hpi": "Patient reports 2 days of chest discomfort.",
            "assessment": "Possible unstable angina."
        }

    Notes
    -----
    All canonical headers are stored in lowercase to allow case-insensitive
    matching against detected headers.
    """

    # Initialise an empty dictionary to hold section headers and their corresponding text
    sections = {}
    # Initialise the current header to None, indicating that we are not currently within a section
    current_header = None
    # Initialise the buffer to hold lines of text for the current section
    buffer = []

    # Split the report into lines for processing
    lines = report.split("\n")

    # Iterate through each line in the report
    for line in lines:
        # Check if the line matches the header pattern
        header = detect_header(line)

        # If the line is a header, we need to check if its a canonical header or not
        if header:

            # Save the previous section if we are currently in one, since we have reached a new header, and need to check whether it is canonical before we can start a new section
            if current_header is not None:
                sections[current_header] = " ".join(buffer).strip()

            # Reset buffer for new section, only fill if canonical header
            buffer = []

            # Convert detected header to lowercase for comparison against the canonical header set
            header_lower = header.lower()

            # Start new section only if canonical
            if header_lower in CANONICAL_HEADER_SET:
                # Set current header to canonical header match
                current_header = header_lower
            else:
                # Set back to None if the detected header is not canonical
                current_header = None

            if ":" in line:
                # If header is within a line, capture text after header colon (first colon) and strip
                after_colon = line.split(":",1)[1].strip()
                # Only add to buffer if there is text after the header colon, since if there is text it should be part of the section content
                if after_colon and current_header:
                    buffer.append(after_colon)

            continue

        # If the line isnt a header and we are already in a section, append the line to the buffer
        if current_header is not None:
            clean = line.strip()
            # Add only non-empty lines to buffer
            if clean:
                buffer.append(clean)

    # After entire report has been checked, save final the section that is still open
    if current_header is not None:
        sections[current_header] = " ".join(buffer).strip()

    # Return the dictionary of extracted sections, where keys are canonical headers and values are the corresponding section text
    return sections