from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from index.hybrid_retriever import HybridRetriever
from reranker import NVIDIAReranker


DIAGNOSIS_INDEX = PROJECT_ROOT / "index" / "diagnosis_vector"
DIAGNOSIS_BM25 = PROJECT_ROOT / "index" / "diagnosis_bm25"

TREATMENT_INDEX = PROJECT_ROOT / "index" / "treatment_vector"
TREATMENT_BM25 = PROJECT_ROOT / "index" / "treatment_bm25"


def print_results(title, results):
    print()
    print(f"--- {title} ---")

    for i, result in enumerate(results, 1):
        md = result.metadata

        print(
            f"{i}. "
            f"hybrid={result.hybrid_score:.6f} "
            f"reranker={getattr(result, 'reranker_score', None)} "
            f"chunk_id={result.chunk_id} "
            f"knowledge_id={result.knowledge_id} "
            f"crop={md.get('crop')} "
            f"disease={md.get('disease')} "
            f"source={md.get('source')} "
            f"evidence={md.get('evidence_type')} "
            f"preference={md.get('recommendation_preference')}"
        )


def validate_traceability(results):
    assert results, "No results returned"

    for result in results:
        assert result.chunk_id, "Missing chunk_id"
        assert result.knowledge_id, "Missing knowledge_id"

        md = result.metadata

        assert md.get("chunk_id") == result.chunk_id
        assert md.get("knowledge_id") == result.knowledge_id

    print("Chunk/knowledge traceability: PASS")


def validate_scores(results):
    assert results, "No results"

    scores = [float(r.reranker_score) for r in results]

    assert all(math.isfinite(s) for s in scores), scores

    assert all(
        scores[i] >= scores[i + 1]
        for i in range(len(scores) - 1)
    ), scores

    print("Reranker score ordering: PASS")
    print("Finite reranker scores: PASS")


def run_diagnosis_test(reranker):
    print()
    print("========================================")
    print("DIAGNOSIS HYBRID → NVIDIA RERANKER")
    print("========================================")

    hybrid = HybridRetriever(
        DIAGNOSIS_INDEX,
        DIAGNOSIS_BM25,
    )

    cases = [
        (
            "Corn northern leaf blight",
            "What are the symptoms of northern leaf blight in corn?",
            "Corn",
            "northern_leaf_blight",
        ),
        (
            "Rice bacterial leaf blight",
            "What are the symptoms of bacterial leaf blight in rice?",
            "Rice",
            "bacterial_leaf_blight",
        ),
        (
            "Cherry powdery mildew",
            "What are the symptoms of powdery mildew in cherry?",
            "Cherry",
            "powdery_mildew",
        ),
    ]

    for name, query, crop, disease in cases:
        print()
        print(f"CASE: {name}")
        print(f"Query: {query}")

        candidates = hybrid.search(
            query=query,
            top_k=8,
            crop=crop,
            disease=disease,
            evidence_type="diagnosis",
            candidate_k=40,
        )

        print(f"Hybrid candidates: {len(candidates)}")

        assert candidates, "Hybrid retrieval returned no candidates"

        reranked = reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=8,
        )

        assert reranked, "Reranker returned no results"

        print_results(name, reranked)

        validate_scores(reranked)
        validate_traceability(reranked)

        top = reranked[0]
        top_md = top.metadata

        assert top_md.get("crop") == crop
        assert top_md.get("disease") == disease
        assert top_md.get("evidence_type") == "diagnosis"

        print("Top-result relevance: PASS")


def run_treatment_test(reranker):
    print()
    print("========================================")
    print("TREATMENT HYBRID → NVIDIA RERANKER")
    print("========================================")

    hybrid = HybridRetriever(
        TREATMENT_INDEX,
        TREATMENT_BM25,
    )

    cases = [
        (
            "Organic",
            "What organic practices can help manage northern leaf blight in corn?",
        ),
        (
            "IPM",
            "What IPM practices can help manage northern leaf blight in corn?",
        ),
        (
            "General",
            "How can northern leaf blight in corn be managed?",
        ),
    ]

    for preference, query in cases:
        print()
        print(f"CASE: Corn NLB / {preference}")
        print(f"Query: {query}")

        candidates = hybrid.search(
            query=query,
            top_k=8,
            recommendation_preference=preference,
            crop="Corn",
            disease="northern_leaf_blight",
            evidence_type="treatment",
            candidate_k=40,
        )

        print(f"Hybrid candidates: {len(candidates)}")

        assert candidates, (
            f"No treatment candidates for preference={preference}"
        )

        reranked = reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=8,
        )

        assert reranked, "Reranker returned no results"

        print_results(
            f"Corn NLB / {preference}",
            reranked,
        )

        validate_scores(reranked)
        validate_traceability(reranked)

        for result in reranked:
            md = result.metadata

            assert md.get("crop") == "Corn"
            assert md.get("disease") == "northern_leaf_blight"
            assert md.get("evidence_type") == "treatment"

            if preference == "Organic":
                assert md.get("organic_eligible") is True

            elif preference == "IPM":
                assert md.get("ipm_eligible") is True

        print("Treatment preference filtering: PASS")


def main():
    print("=== CropGuard Hybrid Retrieval → NVIDIA Reranker Test ===")

    print()
    print("Loading NVIDIA Reranker...")
    reranker = NVIDIAReranker()

    print()
    print("Reranker:")
    for key, value in reranker.info().items():
        print(f"  {key}: {value}")

    run_diagnosis_test(reranker)
    run_treatment_test(reranker)

    print()
    print("========================================")
    print("HYBRID → NVIDIA RERANKER INTEGRATION: PASS")
    print("========================================")


if __name__ == "__main__":
    main()
