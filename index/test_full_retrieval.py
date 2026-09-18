from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from index.hybrid_retriever import HybridRetriever


ROOT = Path("/workspace/index")

diagnosis = HybridRetriever(
    vector_index_path=ROOT / "diagnosis_vector",
    bm25_index_path=ROOT / "diagnosis_bm25",
)

treatment = HybridRetriever(
    vector_index_path=ROOT / "treatment_vector",
    bm25_index_path=ROOT / "treatment_bm25",
)


def show(title, results):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    for i, result in enumerate(results, 1):
        record = result.metadata

        print(
            f"{i}. "
            f"{record.get('chunk_id')} | "
            f"{record.get('source')} | "
            f"{record.get('crop')} | "
            f"{record.get('disease')} | "
            f"{record.get('evidence_type')} | "
            f"score={result.final_score:.6f}"
        )


# ------------------------------------------------------------------
# Diagnosis tests
# ------------------------------------------------------------------

queries = [
    (
        "Corn northern leaf blight diagnosis",
        "corn northern leaf blight symptoms",
    ),
    (
        "Rice bacterial leaf blight diagnosis",
        "rice bacterial leaf blight symptoms",
    ),
    (
        "Powdery mildew diagnosis",
        "powdery mildew symptoms on cherry leaves",
    ),
]

for title, query in queries:
    results = diagnosis.search(
        query=query,
        top_k=5,
        evidence_type="diagnosis",
    )

    show(title, results)

    assert results, f"No results for: {query}"

    for result in results:
        assert (
            result.metadata.get("evidence_type")
            == "diagnosis"
        )


# ------------------------------------------------------------------
# Treatment tests
# ------------------------------------------------------------------

treatment_tests = [
    (
        "Organic Corn/NLB",
        "how to manage northern leaf blight in corn",
        "Organic",
        "Corn",
        "northern_leaf_blight",
    ),
    (
        "IPM Corn/NLB",
        "integrated pest management for northern leaf blight in corn",
        "IPM",
        "Corn",
        "northern_leaf_blight",
    ),
    (
        "General Corn/NLB",
        "management of northern leaf blight in corn",
        "General",
        "Corn",
        "northern_leaf_blight",
    ),
]

for (
    title,
    query,
    preference,
    crop,
    disease,
) in treatment_tests:

    results = treatment.search(
        query=query,
        top_k=5,
        recommendation_preference=preference,
        crop=crop,
        disease=disease,
        evidence_type="treatment",
    )

    show(title, results)

    assert results, (
        f"No treatment results for: {title}"
    )

    for result in results:
        record = result.metadata

        assert (
            record.get("evidence_type")
            == "treatment"
        )

        assert (
            str(record.get("crop")).lower()
            == crop.lower()
        )

        assert (
            str(record.get("disease")).lower()
            == disease.lower()
        )

        if preference == "Organic":
            assert record.get("organic_eligible") is True

        elif preference == "IPM":
            assert record.get("ipm_eligible") is True


# ------------------------------------------------------------------
# Cross-domain protection
# ------------------------------------------------------------------

print()
print("=" * 70)
print("CROSS-DOMAIN PROTECTION")
print("=" * 70)

diagnosis_results = diagnosis.search(
    query="how to treat northern leaf blight",
    top_k=5,
    evidence_type="diagnosis",
)

treatment_results = treatment.search(
    query="symptoms of northern leaf blight",
    top_k=5,
    evidence_type="treatment",
)

assert diagnosis_results
assert treatment_results

assert all(
    r.metadata.get("evidence_type") == "diagnosis"
    for r in diagnosis_results
)

assert all(
    r.metadata.get("evidence_type") == "treatment"
    for r in treatment_results
)

print("Diagnosis index contains only diagnosis evidence: PASS")
print("Treatment index contains only treatment evidence: PASS")


print()
print("=" * 70)
print("FULL RETRIEVAL BASELINE: PASS")
print("=" * 70)
