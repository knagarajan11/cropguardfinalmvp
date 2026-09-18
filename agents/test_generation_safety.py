from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent),
)

from PIL import Image

from agents.knowledge_agent import KnowledgeAgentResult
from agents.generation_agent import GroundedGenerationAgent


class ModelMustNotBeCalled:
    """Fake model used to prove the abstention gate."""

    def diagnose(self, *args, **kwargs):
        raise AssertionError(
            "MODEL WAS CALLED DURING SAFE ABSTENTION"
        )


def main() -> None:
    print("\n=== CropGuard Grounded Generation Safety Test ===")

    knowledge_result = KnowledgeAgentResult(
        status="INSUFFICIENT",
        query=(
            "What organic practices can help manage "
            "northern leaf blight in corn?"
        ),
        crop="Corn",
        disease="northern_leaf_blight",
        evidence_type="treatment",
        recommendation_preference="Organic",
        accepted_evidence=[],
        rejected_evidence=[],
        reasons=[
            "No sufficiently relevant Organic evidence."
        ],
        hybrid_candidates=3,
        reranked_candidates=3,
    )

    generator = GroundedGenerationAgent(
        model=ModelMustNotBeCalled()
    )

    image = Image.new(
        "RGB",
        (32, 32),
        "white",
    )

    result = generator.generate(
        image=image,
        knowledge_result=knowledge_result,
        farmer_query=knowledge_result.query,
    )

    print(f"Status:       {result.status}")
    print(f"Model called: {result.model_called}")
    print(f"Evidence:     {result.evidence_count}")
    print(f"Answer:       {result.answer}")

    assert result.status == "INSUFFICIENT"
    assert result.model_called is False
    assert result.evidence_count == 0
    assert result.abstain

    print("\nSafe abstention: PASS")
    print("Model bypass protection: PASS")
    print("No evidence -> no generation: PASS")

    print(
        "\nGROUNDED GENERATION SAFETY TEST: PASS"
    )


if __name__ == "__main__":
    main()
