from __future__ import annotations

from index.recommendation_retriever import RecommendationRetriever
from reranker.nvidia_reranker import NVIDIAReranker


QUERY = "What is the management for tomato early blight?"


def main() -> None:

    print("=" * 80)
    print("CropGuard Unified Recommendation Reranker Test")
    print("=" * 80)

    retriever = RecommendationRetriever()

    candidates = retriever.search(
        QUERY,
        top_k=5,
    )

    print(f"\nFAISS candidates: {len(candidates)}")

    if not candidates:
        raise RuntimeError(
            "No FAISS recommendation candidates returned."
        )

    print("\nLoading NVIDIA Reranker 1B v2...")

    reranker = NVIDIAReranker(
        model_name_or_path=(
            "nvidia/llama-nemotron-rerank-1b-v2"
        ),
        device="cuda",
    )

    results = reranker.rerank(
        QUERY,
        candidates,
        top_k=5,
    )

    print(f"\nReranked results: {len(results)}")

    for rank, result in enumerate(results, 1):

        print(f"\n--- RERANKED RESULT {rank} ---")
        print("reranker_score:", result.reranker_score)
        print("vector_score:", result.vector_score)
        print("hybrid_score:", result.hybrid_score)
        print("chunk_id:", result.chunk_id)
        print("knowledge_id:", result.knowledge_id)
        print("source:", result.metadata.get("source"))
        print(
            "source_type:",
            result.metadata.get("source_type"),
        )
        print(
            "crop:",
            result.metadata.get("crop"),
        )
        print(
            "disease:",
            result.metadata.get("disease"),
        )
        print(
            "evidence_type:",
            result.metadata.get("evidence_type"),
        )
        print(
            "recommendation_preference:",
            result.metadata.get(
                "recommendation_preference"
            ),
        )
        print(
            "content:",
            result.content[:1000],
        )

    if not results:
        raise RuntimeError(
            "NVIDIA Reranker returned no results."
        )

    top = results[0]

    top_crop = (
        top.metadata.get("crop") or ""
    ).lower()

    top_disease = (
        top.metadata.get("disease") or ""
    ).lower()

    top_source = (
        top.metadata.get("source") or ""
    ).lower()

    if top_crop != "tomato":
        raise RuntimeError(
            f"Unexpected top crop: {top_crop}"
        )

    if top_disease != "early blight":
        raise RuntimeError(
            f"Unexpected top disease: {top_disease}"
        )

    if top_source != "tnau_tomato_diseases_001":
        raise RuntimeError(
            f"Unexpected top source: {top_source}"
        )

    scores = [
        float(result.reranker_score)
        for result in results
    ]

    if not all(
        score == score
        and abs(score) != float("inf")
        for score in scores
    ):
        raise RuntimeError(
            "Non-finite reranker score detected."
        )

    print("\nTOP RESULT VALIDATION: PASS")
    print("SCORE FINITENESS: PASS")
    print("TRACEABILITY: PASS")
    print("NVIDIA RERANKER UNIFIED TEST: PASS")


if __name__ == "__main__":
    main()
