import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from .preprocessor import preprocess_for_chunking

import tiktoken


# ============================================================
# PATHS
# ============================================================

TEXT_DIR = Path(
    "data/processed/cleaned_v3"
)

METADATA_DIR = Path(
    "data/processed/metadata"
)

OUTPUT_DIR = Path(
    "data/processed/chunks"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TOKENIZER
# ============================================================

encoding = tiktoken.get_encoding(
    "cl100k_base"
)


def count_tokens(text: str) -> int:
    return len(
        encoding.encode(text)
    )


# ============================================================
# SECTION HEADINGS
# ============================================================

SECTION_HEADINGS = [
    "Frequently Asked Questions",
    "Documents Required",
    "Application Process",
    "Sources And References",
    "Supported Programmes",
    "Objectives",
    "Objective",
    "Benefits",
    "Eligibility",
    "Exclusions",
    "Introduction",
    "Details",
]

SECTION_HEADINGS = sorted(
    SECTION_HEADINGS,
    key=len,
    reverse=True
)


def detect_section(line: str) -> str | None:

    value = re.sub(
        r"\s+",
        " ",
        line.strip()
    )

    for heading in SECTION_HEADINGS:

        # Exact heading
        if value.lower() == heading.lower():
            return heading

        # Heading followed by punctuation/content
        if re.match(
            rf"^{re.escape(heading)}"
            rf"(\s|:|-|$)",
            value,
            flags=re.IGNORECASE
        ):
            return heading

    return None


# ============================================================
# SECTION SPLITTING
# ============================================================

def split_into_sections(text: str):
    """
    Preprocess first, then detect section headings.
    """

    text = preprocess_for_chunking(text)

    sections = []

    current_section = "General"
    current_lines = []

    for line in text.splitlines():

        section = detect_section(line)

        if section:

            if current_lines:

                section_text = "\n".join(
                    current_lines
                ).strip()

                if section_text:

                    sections.append(
                        (
                            current_section,
                            section_text
                        )
                    )

            current_section = section
            current_lines = []

        else:

            current_lines.append(line)

    if current_lines:

        section_text = "\n".join(
            current_lines
        ).strip()

        if section_text:

            sections.append(
                (
                    current_section,
                    section_text
                )
            )

    return sections


# ============================================================
# LOAD DOCUMENT METADATA
# ============================================================

def load_metadata(
    source_file: str
) -> Dict:

    metadata_path = (
        METADATA_DIR
        / f"{Path(source_file).stem}.json"
    )

    if not metadata_path.exists():

        raise FileNotFoundError(
            f"Metadata not found for "
            f"{source_file}: "
            f"{metadata_path}"
        )

    with metadata_path.open(
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# CREATE CHUNK RECORD
# ============================================================

def create_chunk_record(
    chunk_id: str,
    chunk_text: str,
    metadata: Dict,
    section: str,
    chunk_index: int,
    method: str
) -> Dict:

    chunk_metadata = {
        **metadata,

        "section": section,

        "chunk_index": chunk_index,

        "chunk_method": method,

        "token_count":
            count_tokens(chunk_text),
    }

    return {
        "chunk_id": chunk_id,

        "text": chunk_text,

        "metadata": chunk_metadata
    }
