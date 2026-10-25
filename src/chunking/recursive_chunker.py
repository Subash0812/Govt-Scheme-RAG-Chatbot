from typing import Dict, List

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from .common import (
    count_tokens,
    create_chunk_record,
    split_into_sections,
)


# ============================================================
# CONFIGURATION
# ============================================================

CHUNK_SIZE = 500

CHUNK_OVERLAP = 75


# ============================================================
# TOKEN LENGTH FUNCTION
# ============================================================

def token_length(
    text: str
) -> int:

    return count_tokens(
        text
    )


# ============================================================
# RECURSIVE SPLITTER
# ============================================================

splitter = RecursiveCharacterTextSplitter(

    chunk_size=CHUNK_SIZE,

    chunk_overlap=CHUNK_OVERLAP,

    length_function=token_length,

    separators=[
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        "; ",
        ", ",
        " ",
        "",
    ],
)


# ============================================================
# RECURSIVE CHUNKING
# ============================================================

def recursive_chunks(
    text: str,
    metadata: Dict
) -> List[Dict]:

    sections = split_into_sections(
        text
    )

    chunks = []

    chunk_index = 0

    for section, section_text in sections:

        # Put the section name into each chunk.
        if section == "General":
            contextual_text = section_text
        else:
            contextual_text = (
              f"Section: {section}\n\n"
              f"{section_text}"
        )

        section_chunks = (
            splitter.split_text(
                contextual_text
            )
        )

        for chunk_text in section_chunks:

            chunk_text = (
                chunk_text.strip()
            )

            if not chunk_text:
                continue

            chunk_index += 1

            source_file = metadata[
                "source_file"
            ]

            chunk_id = (
                f"{source_file}"
                f"_recursive_"
                f"{chunk_index}"
            )

            chunks.append(
                create_chunk_record(
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                    metadata=metadata,
                    section=section,
                    chunk_index=chunk_index,
                    method="recursive",
                )
            )

    return chunks