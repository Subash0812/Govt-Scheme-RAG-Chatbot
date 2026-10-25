from typing import Dict, List

from .common import (
    encoding,
    count_tokens,
    create_chunk_record,
    load_metadata,
)

from .preprocessor import (
    preprocess_for_chunking
)


# ============================================================
# CONFIGURATION
# ============================================================

CHUNK_SIZE = 500

OVERLAP = 75


# ============================================================
# FIXED-SIZE CHUNKING
# ============================================================

def fixed_size_chunks(
    text: str,
    metadata: Dict
) -> List[Dict]:

    # PREPROCESS BEFORE TOKENIZATION
    text = preprocess_for_chunking(text)

    tokens = encoding.encode(text)

    chunks = []

    start = 0
    chunk_index = 0

    while start < len(tokens):

        end = min(
            start + CHUNK_SIZE,
            len(tokens)
        )

        chunk_tokens = tokens[
            start:end
        ]

        chunk_text = encoding.decode(
            chunk_tokens
        ).strip()

        if chunk_text:

            chunk_index += 1

            source_file = metadata[
                "source_file"
            ]

            chunk_id = (
                f"{source_file}"
                f"_fixed_"
                f"{chunk_index}"
            )

            chunks.append(
                create_chunk_record(
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                    metadata=metadata,
                    section="General",
                    chunk_index=chunk_index,
                    method="fixed",
                )
            )

        if end >= len(tokens):
            break

        start = end - OVERLAP

    return chunks