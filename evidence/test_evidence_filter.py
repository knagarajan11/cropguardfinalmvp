from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evidence.evidence_filter import EvidenceFilter
from index.hybrid_retriever import HybridRetriever
from reranker.nvidia_reranker import NVIDIAReranker


def main() -> None:
    print("\n=== CropGuard Evidence Filter Test ===\n")

    print("Loading Hybrid Retriever...")
    retriever = HybridRetriever(
        vector_index_path="/workspace/index/treatment_vector",
        bm25_index_path="/workspace/index/treatment_bm25",
    )

    query = (
        "What organic practices can help manage "
        "northern leaf blight in corn?"
    )

    candidates = retriever.search(
        query,
        top_k=5,
        crop="Corn",
        disease="northern_leaf_blight",
        evidence_type="treatment",
        recommendation_preference="Organic",
    )

    print(f"Hybrid candidates: {len(candidates)}")

    print("\nLoading NVIDIA Reranker...")
    reranker = NVIDIAReranker()

    reranked = reranker.rerank(
        query,
        candidates,
        top_k=5,
    )

    print("\nReranked evidence:")
    for item in reranked:
        preference = item.metadata.get(
            "recommendation_preference"
        )
        organic = item.metadata.get(
            "organic_eligible"
        )

        print(
            f"  score={item.reranker_score: .6f} "
            f"preference={preference} "
            f"organic={organic} "
            f"chunk={item.chunk_id}"
        )

    print("\nEvidence Filter:")

    # 5.0 is ONLY a demonstration threshold for this safety test.
    # It is NOT a universal meaning of the NVIDIA reranker score.
    evidence_filter = EvidenceFilter(
        min_reranker_score=5.0,
        require_relevant_evidence=True,
    )

    result = evidence_filter.filter(
        reranked,
        crop="Corn",
        disease="northern_leaf_blight",
        evidence_type="treatment",
        recommendation_preference="Organic",
    )

    print(f"  status: {result.status}")
    print(f"  accepted: {len(result.accepted)}")
    print(f"  rejected: {len(result.rejected)}")

    for reason in result.reasons:
        print(f"  reason: {reason}")

    # ------------------------------------------------------------
    # SAFETY EXPECTATION
    #
    # Organic evidence has reranker score 3.859375, below 5.0.
    # General evidence has a higher score, but MUST NOT satisfy
    # an Organic request.
    #
    # Therefore the correct result is INSUFFICIENT.
    # ------------------------------------------------------------

    assert result.status == "INSUFFICIENT", (
        f"Expected INSUFFICIENT for Organic request, "
        f"got {result.status}"
    )

    assert len(result.accepted) == 0, (
        "General/IPM evidence must not be accepted "
        "for an Organic request when Organic evidence "
        "is insufficient."
    )

    # Explicit policy-protection check.
    for item in result.accepted:
        preference = item.metadata.get(
            "recommendation_preference"
        )
        assert preference == "Organic", (
            f"Policy violation: accepted evidence has "
            f"preference={preference}"
        )

    print("\nPolicy protection: PASS")
    print("Safe abstention: PASS")
    print("Organic evidence below threshold: PASS")
    print("General evidence NOT substituted for Organic: PASS")

    print("\nEVIDENCE FILTER SAFETY TEST: PASS")


if __name__ == "__main__":
    main()
