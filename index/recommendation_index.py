from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np

from knowledge.external.chunker import RecommendationChunk
from knowledge.external.loaders import load_all_sources
from knowledge.external.cleanup import clean_and_validate
from knowledge.vl_embedder import NVIDIAVLEmbedder, VLEmbeddingConfig


JSON_PATH = Path("/workspace/knowledge/sample_data/treatment_knowledge.jsonl")
PDF_PATH = Path("/workspace/knowledge/external/pdf")
WEB_REGISTRY = Path("/workspace/knowledge/external/web/sources.yaml")

INDEX_DIR = Path("/workspace/index/recommendation")


def save_chunks(
    chunks: Sequence[RecommendationChunk],
    path: Path,
) -> None:

    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            row = {
                "chunk_id": chunk.chunk_id,
                "knowledge_id": chunk.knowledge_id,
                "source": chunk.source,
                "source_type": chunk.source_type,
                "source_uri": chunk.source_uri,
                "document_id": chunk.document_id,
                "document_title": chunk.document_title,
                "page": chunk.page,
                "section": chunk.section,
                "subsection": chunk.subsection,
                "crop": chunk.crop,
                "disease": chunk.disease,
                "evidence_type": chunk.evidence_type,
                "recommendation_preference": (
                    chunk.recommendation_preference
                ),
                "organic_eligible": chunk.organic_eligible,
                "ipm_eligible": chunk.ipm_eligible,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "metadata": chunk.metadata,
            }

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_index() -> None:

    print("=" * 80)
    print("CropGuard Unified Recommendation Index Builder")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. LOAD
    # ------------------------------------------------------------------

    print("\n[1/5] Loading JSON + PDF + Web sources...")

    records = load_all_sources(
        json_path=JSON_PATH,
        pdf_directory=PDF_PATH,
        web_registry=WEB_REGISTRY,
    )

    raw_record_count = len(records)

    print(f"Raw records loaded: {raw_record_count}")

    # ------------------------------------------------------------------
    # 2. CLEANUP + VALIDATION
    # ------------------------------------------------------------------

    print("\n[2/5] Cleaning and validating recommendation records...")

    cleanup_result = clean_and_validate(records)

    accepted_records = cleanup_result.records
    rejected_records = cleanup_result.rejected

    accepted_record_count = len(accepted_records)
    rejected_record_count = len(rejected_records)

    print(f"Accepted records: {accepted_record_count}")
    print(f"Rejected records: {rejected_record_count}")
    print(
        "Duplicate records removed: "
        f"{cleanup_result.duplicate_count}"
    )
    print(
        "Boilerplate records removed: "
        f"{cleanup_result.boilerplate_count}"
    )
    print(
        "Empty records removed: "
        f"{cleanup_result.empty_count}"
    )
    print(
        "Short records removed: "
        f"{cleanup_result.short_count}"
    )

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Rejected records are retained ONLY for audit/debugging.
    rejected_path = INDEX_DIR / "rejected_records.jsonl"

    with rejected_path.open("w", encoding="utf-8") as f:
        for row in rejected_records:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        "Rejected-record audit file: "
        f"{rejected_path}"
    )

    if not accepted_records:
        raise RuntimeError(
            "Recommendation Cleanup Gate FAILED: "
            "no valid recommendation records remain."
        )

    print("Recommendation Cleanup Gate: PASS")
    print(
        "Only accepted records will continue to "
        "chunking, embedding, and indexing."
    )

    # CRITICAL:
    # Rejected records NEVER enter chunking, embedding,
    # FAISS indexing, retrieval, reranking, or generation.
    records = accepted_records

    source_counts: dict[str, int] = {}

    for record in records:
        source_counts[record.source_type] = (
            source_counts.get(record.source_type, 0) + 1
        )

    for source_type, count in sorted(source_counts.items()):
        print(f"  {source_type}: {count}")

    # ------------------------------------------------------------------
    # 3. CHUNKING
    # ------------------------------------------------------------------

    print("\n[3/5] Creating 1000 / 150 recommendation chunks...")

    from knowledge.external.chunker import RecommendationChunker

    chunker = RecommendationChunker(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = chunker.chunk_records(records)

    if not chunks:
        raise RuntimeError(
            "No recommendation chunks were produced."
        )

    print(f"Chunks created: {len(chunks)}")

    # ------------------------------------------------------------------
    # 4. EMBEDDING
    # ------------------------------------------------------------------

    print("\n[4/5] Generating NVIDIA VL embeddings...")

    embedder = NVIDIAVLEmbedder(
        VLEmbeddingConfig(
            model_path=(
                "/home/gsh-ndhmy/.cache/huggingface/"
                "models--nvidia--llama-nemotron-embed-vl-1b-v2/"
                "snapshots/582e3bf72aee355e3c59ed89de53543c5b0657ee"
            ),
            device="cuda",
            normalize=True,
            pooling="avg",
        )
    )

    texts = [chunk.content for chunk in chunks]

    embeddings = embedder.encode_passages(
        texts,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise RuntimeError(
            f"Unexpected embedding shape: {embeddings.shape}"
        )

    dimension = embeddings.shape[1]

    print(f"Embedding shape: {embeddings.shape}")

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    print(
        "Embedding norm range: "
        f"{norms.min():.6f} - {norms.max():.6f}"
    )

    if not np.all(np.isfinite(embeddings)):
        raise RuntimeError(
            "Non-finite embedding values detected."
        )

    # ------------------------------------------------------------------
    # 5. FAISS
    # ------------------------------------------------------------------

    print("\n[5/5] Building FAISS IndexFlatIP...")

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print("FAISS vectors:", index.ntotal)
    print("FAISS dimension:", index.d)

    index_path = INDEX_DIR / "faiss.index"
    chunks_path = INDEX_DIR / "chunks.jsonl"
    meta_path = INDEX_DIR / "index_meta.json"

    faiss.write_index(
        index,
        str(index_path),
    )

    save_chunks(
        chunks,
        chunks_path,
    )

    metadata = {
        "index_type": "FAISS-IndexFlatIP",
        "embedding_model": (
            "nvidia/llama-nemotron-embed-vl-1b-v2"
        ),
        "embedding_dimension": dimension,
        "pooling": "avg",
        "normalize": True,
        "chunk_size": 1000,
        "chunk_overlap": 150,

        # Only accepted records are indexed.
        "record_count": accepted_record_count,
        "raw_record_count": raw_record_count,
        "accepted_record_count": accepted_record_count,
        "rejected_record_count": rejected_record_count,

        "chunk_count": len(chunks),

        "cleanup": {
            "accepted": accepted_record_count,
            "rejected": rejected_record_count,
            "duplicate_count": cleanup_result.duplicate_count,
            "boilerplate_count": cleanup_result.boilerplate_count,
            "empty_count": cleanup_result.empty_count,
            "short_count": cleanup_result.short_count,
            "rejected_records_file": str(rejected_path),
        },

        "source_counts": source_counts,
        "source_types": sorted(source_counts),

        "files": {
            "faiss": str(index_path),
            "chunks": str(chunks_path),
            "rejected_records": str(rejected_path),
        },
    }

    meta_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nINDEX FILES:")
    print(index_path)
    print(chunks_path)
    print(meta_path)
    print(rejected_path)

    print("\nUNIFIED RECOMMENDATION INDEX BUILD: PASS")


if __name__ == "__main__":
    build_index()
