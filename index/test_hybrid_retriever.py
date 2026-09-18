from __future__ import annotations

import sys

sys.path.insert(0, "/workspace")

from index.hybrid_retriever import HybridRetriever


VECTOR_INDEX = "/workspace/index/sample_mvp_vector"
BM25_INDEX = "/workspace/index/sample_mvp_bm25"


def show_results(title, results):
    print(f"\n=== {title} ===")

    if not results:
        print("NO RESULTS")
        return

    for rank, result in enumerate(results, 1):
        print(f"\nRank {rank}")
        print(f"Knowledge ID     : {result.knowledge_id}")
        print(f"Crop             : {result.metadata.get('crop')}")
        print(f"Disease          : {result.metadata.get('disease')}")
        print(
            f"Preference       : "
            f"{result.metadata.get('recommendation_preference')}"
        )
        print(
            f"Vector rank      : {result.vector_rank}"
        )
        print(
            f"BM25 rank        : {result.bm25_rank}"
        )
        print(
            f"Vector score     : {result.vector_score:.6f}"
        )
        print(
            f"BM25 score       : {result.bm25_score:.6f}"
        )
        print(
            f"RRF score        : {result.rrf_score:.6f}"
        )
        print(
            f"Preference score : {result.preference_score:.6f}"
        )
        print(
            f"Final score      : {result.final_score:.6f}"
        )


def main():

    print("=== CROPGUARD HYBRID RETRIEVAL TEST ===")

    retriever = HybridRetriever(
        vector_index_path=VECTOR_INDEX,
        bm25_index_path=BM25_INDEX,
    )

    # ---------------------------------------------------------
    # Test 1: Organic preference
    # ---------------------------------------------------------

    organic = retriever.search(
        query=(
            "How can I manage northern leaf blight "
            "in corn using organic practices?"
        ),
        top_k=3,
        recommendation_preference="Organic",
        evidence_type="treatment",
        crop="Corn",
        disease="northern_leaf_blight",
    )

    show_results(
        "ORGANIC HYBRID RETRIEVAL",
        organic,
    )

    assert organic, \
        "Organic hybrid retrieval returned no results."

    for result in organic:
        assert (
            result.metadata["organic_eligible"]
            is True
        )

        assert (
            result.metadata["crop"].lower()
            == "corn"
        )

        assert (
            result.metadata["disease"].lower()
            == "northern_leaf_blight"
        )

        assert (
            result.metadata["evidence_type"]
            == "treatment"
        )

    # Explicit Organic guidance should receive the
    # highest preference score.
    organic_specific = [
        r for r in organic
        if r.metadata.get(
            "recommendation_preference"
        ) == "Organic"
    ]

    assert organic_specific, \
        "Organic-specific evidence was not retrieved."

    # ---------------------------------------------------------
    # Test 2: IPM preference
    # ---------------------------------------------------------

    ipm = retriever.search(
        query=(
            "What IPM practices can help manage "
            "northern leaf blight in corn?"
        ),
        top_k=3,
        recommendation_preference="IPM",
        evidence_type="treatment",
        crop="Corn",
        disease="northern_leaf_blight",
    )

    show_results(
        "IPM HYBRID RETRIEVAL",
        ipm,
    )

    assert ipm, \
        "IPM hybrid retrieval returned no results."

    for result in ipm:
        assert (
            result.metadata["ipm_eligible"]
            is True
        )

    ipm_specific = [
        r for r in ipm
        if r.metadata.get(
            "recommendation_preference"
        ) == "IPM"
    ]

    assert ipm_specific, \
        "IPM-specific evidence was not retrieved."

    # ---------------------------------------------------------
    # Test 3: General preference
    # ---------------------------------------------------------

    general = retriever.search(
        query=(
            "What treatment guidance is available "
            "for northern leaf blight in corn?"
        ),
        top_k=3,
        recommendation_preference="General",
        evidence_type="treatment",
        crop="Corn",
        disease="northern_leaf_blight",
    )

    show_results(
        "GENERAL HYBRID RETRIEVAL",
        general,
    )

    assert general, \
        "General hybrid retrieval returned no results."

    # General guidance should be preferred.
    assert (
        general[0].metadata.get(
            "recommendation_preference"
        ) == "General"
    ), \
        "General-specific evidence did not rank first."

    # ---------------------------------------------------------
    # Test 4: Cross-crop isolation
    # ---------------------------------------------------------

    rice = retriever.search(
        query=(
            "northern leaf blight management"
        ),
        top_k=3,
        recommendation_preference="General",
        evidence_type="treatment",
        crop="Rice",
    )

    show_results(
        "RICE CROSS-CROP ISOLATION",
        rice,
    )

    for result in rice:
        assert (
            result.metadata["crop"].lower()
            == "rice"
        )

    # ---------------------------------------------------------
    # Test 5: Semantic query without exact disease name
    # ---------------------------------------------------------

    semantic = retriever.search(
        query=(
            "long tan lesions on corn leaves "
            "and how to manage them"
        ),
        top_k=3,
        recommendation_preference="General",
        evidence_type="treatment",
        crop="Corn",
    )

    show_results(
        "SEMANTIC QUERY",
        semantic,
    )

    assert semantic, \
        "Semantic hybrid retrieval returned no results."

    assert any(
        result.metadata.get(
            "disease"
        ) == "northern_leaf_blight"
        for result in semantic
    ), \
        "Northern leaf blight was not found for semantic query."

    print("\n======================================")
    print("HYBRID RETRIEVAL TEST: PASS")
    print("======================================")


if __name__ == "__main__":
    main()
