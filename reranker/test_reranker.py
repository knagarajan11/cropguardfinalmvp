from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math

from reranker import NVIDIAReranker


def main() -> None:
    print("=== CropGuard NVIDIA Reranker Test ===")

    reranker = NVIDIAReranker()

    print()
    print("Reranker info:")
    for key, value in reranker.info().items():
        print(f"  {key}: {value}")

    query = (
        "What are the symptoms of northern leaf blight "
        "in corn?"
    )

    passages = [
        {
            "chunk_id": "test-001",
            "knowledge_id": "test-001",
            "content": (
                "Northern leaf blight in corn produces "
                "large elongated tan or gray-green lesions "
                "on leaves."
            ),
            "metadata": {
                "crop": "Corn",
                "disease": "northern_leaf_blight",
                "evidence_type": "diagnosis",
            },
        },
        {
            "chunk_id": "test-002",
            "knowledge_id": "test-002",
            "content": (
                "Rice bacterial leaf blight can cause "
                "water-soaked lesions and drying of leaves."
            ),
            "metadata": {
                "crop": "Rice",
                "disease": "bacterial_leaf_blight",
                "evidence_type": "diagnosis",
            },
        },
        {
            "chunk_id": "test-003",
            "knowledge_id": "test-003",
            "content": (
                "Cherry powdery mildew appears as a white "
                "powdery growth on affected plant surfaces."
            ),
            "metadata": {
                "crop": "Cherry",
                "disease": "powdery_mildew",
                "evidence_type": "diagnosis",
            },
        },
    ]

    results = reranker.rerank(
        query=query,
        candidates=passages,
        top_k=3,
    )

    print()
    print("Query:")
    print(query)

    print()
    print("Reranked results:")

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. "
            f"score={result.reranker_score:.6f} "
            f"chunk_id={result.chunk_id} "
            f"knowledge_id={result.knowledge_id} "
            f"crop={result.metadata.get('crop')} "
            f"disease={result.metadata.get('disease')}"
        )

    assert len(results) == 3

    scores = [
        result.reranker_score
        for result in results
    ]

    assert all(
        math.isfinite(score)
        for score in scores
    )

    assert scores == sorted(
        scores,
        reverse=True,
    )

    assert all(
        result.chunk_id
        and result.knowledge_id
        for result in results
    )

    print()
    print("Score ordering: PASS")
    print("Finite scores: PASS")
    print("Chunk traceability: PASS")
    print("Knowledge traceability: PASS")
    print()
    print("NVIDIA RERANKER BASIC TEST: PASS")


if __name__ == "__main__":
    main()
