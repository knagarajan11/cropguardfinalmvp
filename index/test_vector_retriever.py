from __future__ import annotations

import sys

sys.path.insert(0, "/workspace")

from index.vector_retriever import VectorRetriever


INDEX_PATH = "/workspace/index/sample_mvp_vector"


def print_results(title, results):
    print(f"\n=== {title} ===")

    if not results:
        print("NO RESULTS")
        return

    for i, result in enumerate(results, 1):
        print(f"\nRank {i}")
        print(f"Score       : {result.score:.6f}")
        print(f"Knowledge ID: {result.knowledge_id}")
        print(f"Chunk ID    : {result.chunk_id}")
        print(f"Crop        : {result.metadata.get('crop')}")
        print(f"Disease     : {result.metadata.get('disease')}")
        print(
            "Preference  : "
            f"{result.metadata.get('recommendation_preference')}"
        )
        print(
            "Organic     : "
            f"{result.metadata.get('organic_eligible')}"
        )
        print(
            "IPM         : "
            f"{result.metadata.get('ipm_eligible')}"
        )


def main():

    print("=== CROPGUARD VECTOR RETRIEVAL TEST ===")

    retriever = VectorRetriever(
        INDEX_PATH
    )

    # ---------------------------------------------------------
    # Test 1: Organic
    # ---------------------------------------------------------

    organic_results = retriever.search(
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

    print_results(
        "ORGANIC CORN NORTHERN LEAF BLIGHT",
        organic_results,
    )

    assert len(organic_results) > 0, \
        "Organic retrieval returned no results."

    for result in organic_results:
        assert result.metadata["organic_eligible"] is True
        assert result.metadata["crop"].lower() == "corn"
        assert (
            result.metadata["disease"].lower()
            == "northern_leaf_blight"
        )
        assert (
            result.metadata["evidence_type"]
            == "treatment"
        )

    # ---------------------------------------------------------
    # Test 2: IPM
    # ---------------------------------------------------------

    ipm_results = retriever.search(
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

    print_results(
        "IPM CORN NORTHERN LEAF BLIGHT",
        ipm_results,
    )

    assert len(ipm_results) > 0, \
        "IPM retrieval returned no results."

    for result in ipm_results:
        assert result.metadata["ipm_eligible"] is True
        assert result.metadata["crop"].lower() == "corn"
        assert (
            result.metadata["disease"].lower()
            == "northern_leaf_blight"
        )

    # ---------------------------------------------------------
    # Test 3: General
    # ---------------------------------------------------------

    general_results = retriever.search(
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

    print_results(
        "GENERAL CORN NORTHERN LEAF BLIGHT",
        general_results,
    )

    assert len(general_results) > 0, \
        "General retrieval returned no results."

    # ---------------------------------------------------------
    # Test 4: Disease isolation
    # ---------------------------------------------------------

    rice_results = retriever.search(
        query=(
            "How should northern leaf blight be managed?"
        ),
        top_k=3,
        recommendation_preference="General",
        evidence_type="treatment",
        crop="Rice",
    )

    print_results(
        "RICE FILTER TEST",
        rice_results,
    )

    # There should be no Corn result when crop=Rice.
    for result in rice_results:
        assert (
            result.metadata["crop"].lower()
            == "rice"
        )

    print("\n======================================")
    print("VECTOR RETRIEVAL TEST: PASS")
    print("======================================")


if __name__ == "__main__":
    main()
