from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/workspace")

from knowledge.loaders import load_treatment_knowledge
from knowledge.chunker import KnowledgeChunker
from knowledge.retrieval_text import build_chunk_retrieval_text
from knowledge.embedder import NVIDIAEmbedder

from index.vector_index import VectorIndex


TREATMENT_PATH = Path(
    "/workspace/knowledge/sample_data/treatment_knowledge.jsonl"
)

INDEX_PATH = Path(
    "/workspace/index/sample_mvp_vector"
)


def main() -> None:

    print("=== CROPGUARD SAMPLE MVP VECTOR INDEX ===")

    # ---------------------------------------------------------
    # 1. Load SampleMVP treatment knowledge
    # ---------------------------------------------------------

    print("\nLoading SampleMVP treatment knowledge...")

    records = load_treatment_knowledge(
        TREATMENT_PATH
    )

    print(f"Knowledge records: {len(records)}")

    # ---------------------------------------------------------
    # 2. Use the validated CropGuard chunker
    # ---------------------------------------------------------

    chunker = KnowledgeChunker()

    chunks = []

    for record in records:
        record_chunks = chunker.chunk_record(record)
        chunks.extend(record_chunks)

    print(f"Knowledge chunks : {len(chunks)}")

    if not chunks:
        raise RuntimeError(
            "No knowledge chunks were produced."
        )

    # ---------------------------------------------------------
    # 3. Build retrieval text
    # ---------------------------------------------------------

    texts = [
        build_chunk_retrieval_text(chunk)
        for chunk in chunks
    ]

    if any(not text.strip() for text in texts):
        raise RuntimeError(
            "One or more chunks produced empty retrieval text."
        )

    # ---------------------------------------------------------
    # 4. Load validated NVIDIA Embed 1B v2
    # ---------------------------------------------------------

    print("\nLoading NVIDIA Embedder...")

    embedder = NVIDIAEmbedder()

    print(
        f"Embedding model : {embedder.config.model_path}"
    )
    print(
        f"Embedding dim   : {embedder.dimension}"
    )

    # ---------------------------------------------------------
    # 5. Generate passage embeddings
    # ---------------------------------------------------------

    print("\nGenerating embeddings...")

    vectors = embedder.encode_passages(
        texts,
        batch_size=8,
    ).numpy()

    print(f"Embedding shape  : {vectors.shape}")

    if vectors.shape != (
        len(chunks),
        embedder.dimension,
    ):
        raise RuntimeError(
            "Unexpected embedding shape: "
            f"{vectors.shape}"
        )

    # ---------------------------------------------------------
    # 6. Prepare persistent metadata
    # ---------------------------------------------------------

    records_for_index = []

    for chunk in chunks:

        records_for_index.append(
            {
                "chunk_id": chunk.chunk_id,
                "knowledge_id": chunk.knowledge_id,
                "content": chunk.content,
                "metadata": {
                    "source": chunk.source,
                    "source_type": chunk.source_type,
                    "crop": chunk.crop,
                    "disease": chunk.disease,
                    "evidence_type": chunk.evidence_type,
                    "recommendation_preference":
                        chunk.recommendation_preference,
                    "organic_eligible":
                        chunk.organic_eligible,
                    "ipm_eligible":
                        chunk.ipm_eligible,
                    "title": chunk.title,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                },
            }
        )

    # ---------------------------------------------------------
    # 7. Create vector index
    # ---------------------------------------------------------

    index = VectorIndex(
        dimension=embedder.dimension
    )

    index.add(
        vectors,
        records_for_index,
    )

    # ---------------------------------------------------------
    # 8. Persist index
    # ---------------------------------------------------------

    print("\nSaving vector index...")

    index.save(INDEX_PATH)

    print(f"Index path       : {INDEX_PATH}")
    print(f"Indexed vectors  : {len(index.records)}")
    print(f"Vector dimension : {index.dimension}")

    # ---------------------------------------------------------
    # 9. Verify persistence
    # ---------------------------------------------------------

    loaded = VectorIndex.load(
        INDEX_PATH
    )

    print(
        f"Reloaded vectors : {len(loaded.records)}"
    )

    if len(loaded.records) != len(chunks):
        raise RuntimeError(
            "Reloaded index record count mismatch."
        )

    if loaded.vectors.shape != vectors.shape:
        raise RuntimeError(
            "Reloaded vector shape mismatch."
        )

    print("\n=== SAMPLE VECTOR INDEX BUILD: PASS ===")


if __name__ == "__main__":
    main()
