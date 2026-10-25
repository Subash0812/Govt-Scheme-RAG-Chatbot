"""Build the local FAISS vector store from recursive scheme chunks.

Run from the repository root:
    python -m src.embeddings.build_faiss_index
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_DIR = PROJECT_ROOT / "data" / "processed" / "chunks" / "recursive"
MODEL_ID = "BAAI/bge-base-en-v1.5"
MODEL_DIR = PROJECT_ROOT / "data" / "models"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store" / "bge_base_en_v1_5"
BATCH_SIZE = 32


def load_chunks() -> list[dict]:
    """Load and validate every recursive chunk in stable filename order."""
    chunks: list[dict] = []
    seen_ids: set[str] = set()

    for path in sorted(CHUNKS_DIR.glob("*.json")):
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"Expected a list of chunks in {path}")

        for record in records:
            chunk_id = record.get("chunk_id")
            text = record.get("text")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError(f"Missing chunk_id in {path}")
            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk_id: {chunk_id}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty text for chunk_id: {chunk_id}")
            if not isinstance(record.get("metadata"), dict):
                raise ValueError(f"Missing metadata for chunk_id: {chunk_id}")

            seen_ids.add(chunk_id)
            chunks.append(record)

    if not chunks:
        raise FileNotFoundError(f"No chunk files found in {CHUNKS_DIR}")
    return chunks


def corpus_fingerprint(chunks: list[dict]) -> str:
    payload = "\n".join(f"{chunk['chunk_id']}\t{chunk['text']}" for chunk in chunks)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_index(chunks: list[dict]) -> tuple[faiss.Index, int]:
    """Encode chunks and construct an exact cosine-similarity index."""
    model = SentenceTransformer(MODEL_ID, cache_folder=str(MODEL_DIR))
    vectors = model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, vectors.shape[1]


def save_store(chunks: list[dict], index: faiss.Index, dimensions: int) -> None:
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(VECTOR_STORE_DIR / "index.faiss"))
    (VECTOR_STORE_DIR / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "model_id": MODEL_ID,
        "chunk_method": "recursive",
        "distance": "cosine similarity (normalized vectors + inner product)",
        "vector_dimensions": dimensions,
        "chunk_count": len(chunks),
        "corpus_sha256": corpus_fingerprint(chunks),
        "created_at": datetime.now(UTC).isoformat(),
    }
    (VECTOR_STORE_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} recursive chunks from {CHUNKS_DIR}")
    index, dimensions = build_index(chunks)
    save_store(chunks, index, dimensions)
    print(f"Saved {index.ntotal} vectors ({dimensions} dimensions) to {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    main()
