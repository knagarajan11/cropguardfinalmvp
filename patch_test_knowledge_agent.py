from pathlib import Path
import shutil

path = Path("/workspace/agents/test_knowledge_agent.py")
backup = path.with_suffix(".py.bak2")

shutil.copy2(path, backup)

text = path.read_text()

old = '''    # ------------------------------------------------------------
    # TEST 1: Organic request
    #
    # Current known scores:
    #   General = 14.125
    #   Organic = 3.859375
    #   IPM     = -2.9375
    #
    # General must NOT satisfy Organic.
    # Organic is below demonstration threshold.
    # Therefore the agent MUST abstain.
    # ------------------------------------------------------------

    organic = agent.retrieve(
        query=(
            "What organic practices can help manage "
            "northern leaf blight in corn?"
        ),
        crop="Corn",
        disease="northern_leaf_blight",
        evidence_type="treatment",
        recommendation_preference="Organic",
    )

    print_result(
        "TEST 1: ORGANIC SAFE ABSTENTION",
        organic,
    )

    assert organic.status == "INSUFFICIENT"
    assert len(organic.accepted_evidence) == 0

    for item in organic.accepted_evidence:
        assert (
            item.metadata.get(
                "recommendation_preference"
            )
            == "Organic"
        )

    print("\\nOrganic policy: PASS")
    print("Organic safe abstention: PASS")
    print(
        "General evidence cannot satisfy Organic: PASS"
    )
'''

new = '''    # ------------------------------------------------------------
    # TEST 1: Organic request
    #
    # Explicit Organic requests are restricted to Organic
    # recommendation evidence before reranking.
    #
    # The Evidence Filter then enforces:
    #   - crop/disease context
    #   - Organic policy
    #   - organic eligibility
    #   - evidence type
    #
    # A lower reranker score must not by itself reject an
    # explicitly policy-matched Organic record.
    # ------------------------------------------------------------

    organic = agent.retrieve(
        query=(
            "What organic practices can help manage "
            "northern leaf blight in corn?"
        ),
        crop="Corn",
        disease="northern_leaf_blight",
        evidence_type="treatment",
        recommendation_preference="Organic",
    )

    print_result(
        "TEST 1: ORGANIC EVIDENCE",
        organic,
    )

    assert organic.status == "SUFFICIENT"
    assert len(organic.accepted_evidence) >= 1

    for item in organic.accepted_evidence:
        assert (
            item.metadata.get(
                "recommendation_preference"
            )
            == "Organic"
        )
        assert (
            item.metadata.get("organic_eligible")
            is True
        )
        assert (
            item.metadata.get("crop")
            == "Corn"
        )

    print("\\nOrganic policy: PASS")
    print("Organic evidence retrieval: PASS")
    print("Organic eligibility: PASS")
'''

if old not in text:
    raise SystemExit(
        "ERROR: Expected Test 1 block was not found. "
        "No changes made."
    )

path.write_text(text.replace(old, new, 1))

print("PATCH PASS")
print(f"Updated: {path}")
print(f"Backup: {backup}")
