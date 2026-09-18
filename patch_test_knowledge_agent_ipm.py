from pathlib import Path
import shutil

path = Path("/workspace/agents/test_knowledge_agent.py")
backup = path.with_suffix(".py.bak3")

shutil.copy2(path, backup)

text = path.read_text()

marker = '''    # ------------------------------------------------------------
    # TEST 3: Accepted evidence must be traceable
    # ------------------------------------------------------------
'''

insert = '''    # ------------------------------------------------------------
    # TEST 3: IPM request
    #
    # Explicit IPM requests are restricted to IPM recommendation
    # evidence before reranking.
    # ------------------------------------------------------------

    ipm = agent.retrieve(
        query=(
            "How can northern leaf blight "
            "in corn be managed using IPM?"
        ),
        crop="Corn",
        disease="northern_leaf_blight",
        evidence_type="treatment",
        recommendation_preference="IPM",
    )

    print_result(
        "TEST 3: IPM EVIDENCE",
        ipm,
    )

    assert ipm.status == "SUFFICIENT"
    assert len(ipm.accepted_evidence) >= 1

    for item in ipm.accepted_evidence:
        assert (
            item.metadata.get(
                "recommendation_preference"
            )
            == "IPM"
        )
        assert (
            item.metadata.get("ipm_eligible")
            is True
        )
        assert (
            item.metadata.get("crop")
            == "Corn"
        )

    print("\\nIPM policy: PASS")
    print("IPM evidence retrieval: PASS")
    print("IPM eligibility: PASS")

    # ------------------------------------------------------------
    # TEST 4: True safe abstention
    #
    # Cherry powdery mildew has no IPM-specific record in the
    # current controlled knowledge set. General/Organic evidence
    # must not be substituted for an explicit IPM request.
    # ------------------------------------------------------------

    abstention = agent.retrieve(
        query=(
            "What IPM practices can help manage "
            "powdery mildew in cherry?"
        ),
        crop="Cherry",
        disease="powdery_mildew",
        evidence_type="treatment",
        recommendation_preference="IPM",
    )

    print_result(
        "TEST 4: TRUE SAFE ABSTENTION",
        abstention,
    )

    assert abstention.status == "INSUFFICIENT"
    assert len(abstention.accepted_evidence) == 0

    for item in abstention.accepted_evidence:
        assert (
            item.metadata.get(
                "recommendation_preference"
            )
            == "IPM"
        )

    print("\\nIPM safe abstention: PASS")
    print(
        "General/Organic evidence cannot satisfy IPM: PASS"
    )

'''

if marker not in text:
    raise SystemExit(
        "ERROR: Test 3 traceability marker was not found. "
        "No changes made."
    )

path.write_text(text.replace(marker, insert + marker, 1))

print("PATCH PASS")
print(f"Updated: {path}")
print(f"Backup: {backup}")
