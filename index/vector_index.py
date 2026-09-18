from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np


@dataclass
class VectorSearchResult:
    chunk_id: str
    knowledge_id: str
    score: float
    content: str
    metadata: dict


class VectorIndex:
    """
    Simple persistent cosine-similarity vector index.

    Final MVP design:
      - NVIDIA Embed 1B v2 produces 2048-D normalized vectors.
      - Vectors are stored as float32.
      - Search uses dot product because vectors are L2 normalized.
      - Metadata is persisted separately.
    """

    def __init__(self, dimension: int = 2048):
        self.dimension = dimension
        self.vectors: np.ndarray | None = None
        self.records: list[dict] = []

    def add(
        self,
        vectors: np.ndarray,
        records: list[dict],
    ) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError(
                f"Expected 2-D vectors, got shape {vectors.shape}"
            )

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected dimension {self.dimension}, "
                f"got {vectors.shape[1]}"
            )

        if len(vectors) != len(records):
            raise ValueError(
                f"Vector count {len(vectors)} != record count {len(records)}"
            )

        # Defensive normalization at index boundary.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)

        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])

        self.records.extend(records)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:

        if self.vectors is None or len(self.records) == 0:
            return []

        query_vector = np.asarray(
            query_vector,
            dtype=np.float32,
        ).reshape(-1)

        if query_vector.shape[0] != self.dimension:
            raise ValueError(
                f"Expected query dimension {self.dimension}, "
                f"got {query_vector.shape[0]}"
            )

        query_norm = np.linalg.norm(query_vector)
        query_vector = query_vector / max(query_norm, 1e-12)

        scores = self.vectors @ query_vector

        top_k = min(top_k, len(scores))

        # Efficient top-k selection.
        candidate_idx = np.argpartition(
            -scores,
            top_k - 1,
        )[:top_k]

        # Sort selected candidates by descending score.
        candidate_idx = candidate_idx[
            np.argsort(-scores[candidate_idx])
        ]

        results = []

        for idx in candidate_idx:
            record = self.records[int(idx)]

            # Retrieval metadata must expose the canonical
            # top-level fields as well as any source-specific
            # metadata. Hybrid retrieval depends on fields such
            # as crop, disease, evidence_type, and eligibility.
            metadata = dict(record.get("metadata", {}))

            for key in (
                "chunk_id",
                "knowledge_id",
                "source",
                "source_type",
                "crop",
                "disease",
                "evidence_type",
                "recommendation_preference",
                "organic_eligible",
                "ipm_eligible",
                "image_path",
                "original_label",
                "health_status",
                "title",
                "chunk_index",
                "total_chunks",
            ):
                if key in record:
                    metadata[key] = record[key]

            results.append(
                VectorSearchResult(
                    chunk_id=record["chunk_id"],
                    knowledge_id=record["knowledge_id"],
                    score=float(scores[idx]),
                    content=record["content"],
                    metadata=metadata,
                )
            )

        return results

    def save(self, directory: str | Path) -> None:
        if self.vectors is None:
            raise ValueError("Cannot save an empty index")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        np.save(
            directory / "vectors.npy",
            self.vectors,
        )

        with open(
            directory / "records.jsonl",
            "w",
            encoding="utf-8",
        ) as f:
            for record in self.records:
                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        with open(
            directory / "index_meta.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "dimension": self.dimension,
                    "count": len(self.records),
                    "metric": "cosine",
                    "storage_dtype": "float32",
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, directory: str | Path) -> "VectorIndex":
        directory = Path(directory)

        with open(
            directory / "index_meta.json",
            "r",
            encoding="utf-8",
        ) as f:
            meta = json.load(f)

        index = cls(
            dimension=int(meta["dimension"])
        )

        index.vectors = np.load(
            directory / "vectors.npy"
        ).astype(np.float32)

        with open(
            directory / "records.jsonl",
            "r",
            encoding="utf-8",
        ) as f:
            index.records = [
                json.loads(line)
                for line in f
                if line.strip()
            ]

        if len(index.vectors) != len(index.records):
            raise ValueError(
                "Vector/metadata count mismatch"
            )

        return index
