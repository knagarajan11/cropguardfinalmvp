from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, "/workspace")

from knowledge.embedder import NVIDIAEmbedder
from index.vector_index import VectorIndex, VectorSearchResult


class VectorRetriever:
    """
    CropGuard vector retrieval layer.

    Query encoding:
        NVIDIA Embed 1B v2

    Retrieval:
        cosine similarity over normalized vectors

    Filtering:
        recommendation preference
        evidence type
        crop
        disease
    """

    VALID_PREFERENCES = {
        "Organic",
        "IPM",
        "General",
    }

    VALID_EVIDENCE_TYPES = {
        "diagnosis",
        "treatment",
        "management",
        "prevention",
        "general",
    }

    def __init__(
        self,
        index_path: str | Path,
        embedder: Optional[NVIDIAEmbedder] = None,
    ) -> None:

        self.index_path = Path(index_path)

        self.index = VectorIndex.load(
            self.index_path
        )

        self.embedder = (
            embedder
            if embedder is not None
            else NVIDIAEmbedder()
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        recommendation_preference: str = "General",
        evidence_type: str | None = None,
        crop: str | None = None,
        disease: str | None = None,
    ) -> list[VectorSearchResult]:

        if not query or not query.strip():
            return []

        if recommendation_preference not in self.VALID_PREFERENCES:
            raise ValueError(
                "Unsupported recommendation preference: "
                f"{recommendation_preference}"
            )

        if (
            evidence_type is not None
            and evidence_type not in self.VALID_EVIDENCE_TYPES
        ):
            raise ValueError(
                f"Unsupported evidence type: {evidence_type}"
            )

        query_vector = self.embedder.encode_query(query)

        # Retrieve a larger candidate pool before filtering.
        candidate_k = min(
            max(top_k * 5, 20),
            len(self.index.records),
        )

        candidates = self.index.search(
            query_vector,
            top_k=candidate_k,
        )

        filtered = []

        for result in candidates:

            metadata = result.metadata

            # ---------------------------------------------
            # Recommendation preference
            # ---------------------------------------------

            if recommendation_preference == "Organic":
                if not metadata.get(
                    "organic_eligible",
                    False,
                ):
                    continue

            elif recommendation_preference == "IPM":
                if not metadata.get(
                    "ipm_eligible",
                    False,
                ):
                    continue

            # General:
            # no eligibility restriction.

            # ---------------------------------------------
            # Evidence type
            # ---------------------------------------------

            if evidence_type is not None:
                if metadata.get(
                    "evidence_type"
                ) != evidence_type:
                    continue

            # ---------------------------------------------
            # Crop
            # ---------------------------------------------

            if crop is not None:
                if (
                    metadata.get("crop", "").lower()
                    != crop.lower()
                ):
                    continue

            # ---------------------------------------------
            # Disease
            # ---------------------------------------------

            if disease is not None:
                if (
                    metadata.get("disease", "").lower()
                    != disease.lower()
                ):
                    continue

            filtered.append(result)

            if len(filtered) >= top_k:
                break

        return filtered
