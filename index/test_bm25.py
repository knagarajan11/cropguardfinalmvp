from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/workspace")

from knowledge.loaders import load_treatment_knowledge
from knowledge.chunker import KnowledgeChunker
from knowledge.retrieval_text import build_chunk_retrieval_text

from index.bm25_index import BM25Index


TREATMENT_PATH = Path(
    "/workspace/knowledge/sample_data/treatment_knowledge.jsonl"
)


def main():

    print("=== CROPGUARD BM25 TEST ===")

    records = load_treatment_knowledge(
        TREATMENT_PATH
    )

    chunker = KnowledgeChunker()

    chunks = []

    for record in records:
        chunks.extend(
            chunker.chunk_record(record)
        )

    texts = [
        build_chunk_retrieval_text(chunk)
        for chunk in chunks
    ]

    index_records = []

    for chunk in chunks:
        index_records.append(
            {
                "chunk_id": chunk.chunk_id,
                "knowledge_id": chunk.knowledge_id,
                "content": chunk.content,
                "metadata": {
                    "crop": chunk.crop,
                    "disease": chunk.disease,
                    "evidence_type": chunk.evidence_type,
                    "recommendation_preference":
                        chunk.recommendation_preference,
                    "organic_eligible":
                        chunk.organic_eligible,
                    "ipm_eligible":
                        chunk.ipm_eligible,
                },
            }
        )

    index = BM25Index()

    index.add(
        texts,
        index_records,
    )

    print(f"Documents indexed: {len(index.records)}")
    print(
        f"Vocabulary size  : "
        f"{len(index.document_frequency)}"
    )

    query = (
        "northern leaf blight corn"
    )

    results = index.search(
        query,
        top_k=5,
    )

    print(
        f"\nQuery: {query}"
    )

    for rank, result in enumerate(
        results,
        1,
    ):
        print(
            f"\nRank {rank}"
        )
        print(
            f"Score       : {result.score:.6f}"
        )
        print(
            f"Knowledge ID: {result.knowledge_id}"
        )
        print(
            f"Crop        : "
            f"{result.metadata.get('crop')}"
        )
        print(
            f"Disease     : "
            f"{result.metadata.get('disease')}"
        )
        print(
            f"Preference  : "
            f"{result.metadata.get('recommendation_preference')}"
        )

    assert len(results) > 0, \
        "BM25 returned no results."

    assert (
        results[0].metadata["crop"].lower()
        == "corn"
    ), \
        "Expected Corn to rank first."

    assert (
        results[0].metadata["disease"].lower()
        == "northern_leaf_blight"
    ), \
        "Expected northern leaf blight to rank first."

    # Persistence test.
    test_path = Path(
        "/workspace/index/sample_mvp_bm25"
    )

    index.save(test_path)

    reloaded = BM25Index.load(
        test_path
    )

    reload_results = reloaded.search(
        query,
        top_k=5,
    )

    assert len(reload_results) > 0

    assert (
        reload_results[0].knowledge_id
        == results[0].knowledge_id
    )

    print("\nPersistence: PASS")

    print("\n======================================")
    print("BM25 TEST: PASS")
    print("======================================")


if __name__ == "__main__":
    main()
