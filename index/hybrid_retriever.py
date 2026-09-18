from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, "/workspace")

from knowledge.embedder import NVIDIAEmbedder

from index.vector_index import VectorIndex
from index.bm25_index import BM25Index


@dataclass
class HybridSearchResult:
    chunk_id: str
    knowledge_id: str
    vector_rank: int | None
    bm25_rank: int | None
    vector_score: float
    bm25_score: float
    rrf_score: float
    preference_score: float
    final_score: float
    content: str
    metadata: dict


class HybridRetriever:
    """
    CropGuard hybrid retrieval.

    Combines:
      1. NVIDIA Embed 1B v2 semantic retrieval
      2. BM25 lexical retrieval
      3. Reciprocal Rank Fusion
      4. CropGuard recommendation-preference ranking
      5. Metadata eligibility filters
    """

    VALID_PREFERENCES = {
        "Organic",
        "IPM",
        "General",
    }

    def __init__(
        self,
        vector_index_path: str | Path,
        bm25_index_path: str | Path,
        embedder: NVIDIAEmbedder | None = None,
        rrf_k: int = 60,
    ) -> None:

        self.vector_index = VectorIndex.load(
            vector_index_path
        )

        self.bm25_index = BM25Index.load(
            bm25_index_path
        )

        self.embedder = (
            embedder
            if embedder is not None
            else NVIDIAEmbedder()
        )

        self.rrf_k = rrf_k

    @staticmethod
    def _preference_score(
        metadata: dict,
        preference: str,
    ) -> float:

        record_preference = metadata.get(
            "recommendation_preference"
        )

        if preference == "Organic":

            if record_preference == "Organic":
                return 1.0

            if (
                record_preference == "General"
                and metadata.get(
                    "organic_eligible",
                    False,
                )
            ):
                return 0.5

            return 0.0

        if preference == "IPM":

            if record_preference == "IPM":
                return 1.0

            if (
                record_preference == "General"
                and metadata.get(
                    "ipm_eligible",
                    False,
                )
            ):
                return 0.5

            return 0.0

        # General preference.
        if record_preference == "General":
            return 1.0

        return 0.8

    def search(
        self,
        query: str,
        top_k: int = 8,
        recommendation_preference: str = "General",
        evidence_type: str | None = None,
        crop: str | None = None,
        disease: str | None = None,
        candidate_k: int = 40,
    ) -> list[HybridSearchResult]:

        if not query.strip():
            return []

        if recommendation_preference not in self.VALID_PREFERENCES:
            raise ValueError(
                "Unsupported recommendation preference: "
                f"{recommendation_preference}"
            )

        # -----------------------------------------------------
        # 1. Query embedding
        # -----------------------------------------------------

        query_vector = self.embedder.encode_query(
            query
        )

        # -----------------------------------------------------
        # 2. Vector retrieval
        # -----------------------------------------------------

        vector_results = self.vector_index.search(
            query_vector,
            top_k=min(
                candidate_k,
                len(self.vector_index.records),
            ),
        )

        # -----------------------------------------------------
        # 3. BM25 retrieval
        # -----------------------------------------------------

        bm25_results = self.bm25_index.search(
            query,
            top_k=min(
                candidate_k,
                len(self.bm25_index.records),
            ),
        )

        # -----------------------------------------------------
        # 4. Combine candidates
        # -----------------------------------------------------

        candidates = {}

        for rank, result in enumerate(
            vector_results,
            start=1,
        ):

            candidates[result.chunk_id] = {
                "chunk_id": result.chunk_id,
                "knowledge_id": result.knowledge_id,
                "vector_rank": rank,
                "bm25_rank": None,
                "vector_score": result.score,
                "bm25_score": 0.0,
                "content": result.content,
                "metadata": result.metadata,
            }

        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):

            if result.chunk_id not in candidates:

                candidates[result.chunk_id] = {
                    "chunk_id": result.chunk_id,
                    "knowledge_id": result.knowledge_id,
                    "vector_rank": None,
                    "bm25_rank": rank,
                    "vector_score": 0.0,
                    "bm25_score": result.score,
                    "content": result.content,
                    "metadata": result.metadata,
                }

            else:

                candidates[
                    result.chunk_id
                ]["bm25_rank"] = rank

                candidates[
                    result.chunk_id
                ]["bm25_score"] = result.score

        # -----------------------------------------------------
        # 5. Metadata filtering + RRF
        # -----------------------------------------------------

        results = []

        for candidate in candidates.values():

            metadata = candidate["metadata"]

            # Hard eligibility filters.
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

            if evidence_type is not None:
                if metadata.get(
                    "evidence_type"
                ) != evidence_type:
                    continue

            if crop is not None:
                if (
                    metadata.get("crop", "").lower()
                    != crop.lower()
                ):
                    continue

            if disease is not None:
                if (
                    metadata.get("disease", "").lower()
                    != disease.lower()
                ):
                    continue

            # RRF contribution.
            rrf = 0.0

            if candidate["vector_rank"] is not None:
                rrf += 1.0 / (
                    self.rrf_k
                    + candidate["vector_rank"]
                )

            if candidate["bm25_rank"] is not None:
                rrf += 1.0 / (
                    self.rrf_k
                    + candidate["bm25_rank"]
                )

            preference_score = self._preference_score(
                metadata,
                recommendation_preference,
            )

            # Preference is intentionally a secondary
            # ranking signal, not a replacement for relevance.
            final_score = (
                rrf
                + 0.05 * preference_score
            )

            results.append(
                HybridSearchResult(
                    chunk_id=candidate["chunk_id"],
                    knowledge_id=candidate["knowledge_id"],
                    vector_rank=candidate["vector_rank"],
                    bm25_rank=candidate["bm25_rank"],
                    vector_score=candidate["vector_score"],
                    bm25_score=candidate["bm25_score"],
                    rrf_score=rrf,
                    preference_score=preference_score,
                    final_score=final_score,
                    content=candidate["content"],
                    metadata=metadata,
                )
            )

        results.sort(
            key=lambda result: result.final_score,
            reverse=True,
        )

        return results[:top_k]
