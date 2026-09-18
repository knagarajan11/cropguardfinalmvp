from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from knowledge.vl_embedder import NVIDIAVLEmbedder, VLEmbeddingConfig


INDEX_DIR = Path("/workspace/index/recommendation")


def _normalize_lookup_label(value: str | None) -> str | None:
    """
    Normalize application-level crop/disease lookup labels.

    This does not modify persisted KB records. It only makes
    semantically equivalent labels comparable during retrieval.
    """

    if value is None:
        return None

    value = value.strip()

    if not value:
        return None

    value = value.lower()

    # Common disease alias normalization.
    disease_aliases = {
        "northern corn leaf blight": "northern_leaf_blight",
        "northern leaf blight": "northern_leaf_blight",
        "nclb": "northern_leaf_blight",
        "bacterial leaf blight": "bacterial_leaf_blight",
    }

    compact = re.sub(r"\s+", " ", value).strip()

    if compact in disease_aliases:
        return disease_aliases[compact]

    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"_+", "_", value)
    value = value.strip("_")

    return value or None


class RecommendationRetriever:
    """Persisted FAISS retriever for unified JSON/PDF/Web recommendations."""

    def __init__(
        self,
        *,
        index_dir: str | Path = INDEX_DIR,
        embedder: NVIDIAVLEmbedder | None = None,
    ) -> None:

        self.index_dir = Path(index_dir)

        self.index_path = self.index_dir / "faiss.index"
        self.chunks_path = self.index_dir / "chunks.jsonl"
        self.meta_path = self.index_dir / "index_meta.json"

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {self.index_path}"
            )

        if not self.chunks_path.exists():
            raise FileNotFoundError(
                f"Chunk metadata not found: {self.chunks_path}"
            )

        self.index = faiss.read_index(
            str(self.index_path)
        )

        self.chunks: list[dict[str, Any]] = []

        with self.chunks_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                line = line.strip()

                if line:
                    self.chunks.append(
                        json.loads(line)
                    )

        if self.index.ntotal != len(self.chunks):
            raise RuntimeError(
                "FAISS vector count does not match chunk count: "
                f"{self.index.ntotal} != {len(self.chunks)}"
            )

        if embedder is None:
            embedder = NVIDIAVLEmbedder(
                VLEmbeddingConfig(
                    model_path=(
                        "/home/gsh-ndhmy/.cache/huggingface/"
                        "models--nvidia--llama-nemotron-embed-vl-1b-v2/"
                        "snapshots/"
                        "582e3bf72aee355e3c59ed89de53543c5b0657ee"
                    ),
                    device="cuda",
                    normalize=True,
                    pooling="avg",
                )
            )

        self.embedder = embedder

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        crop: str | None = None,
        disease: str | None = None,
    ) -> list[dict[str, Any]]:

        if not query or not query.strip():
            return []

        if top_k <= 0:
            return []

        query_embedding = self.embedder.encode_queries(
            [query]
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_embedding.ndim != 2:
            raise RuntimeError(
                f"Unexpected query embedding shape: "
                f"{query_embedding.shape}"
            )

        # Search the full persisted recommendation index before
        # applying application-level crop/disease filters.
        #
        # This prevents relevant evidence from being discarded
        # merely because it did not appear in the first global
        # top_k FAISS results.
        candidate_k = self.index.ntotal

        scores, indices = self.index.search(
            query_embedding,
            candidate_k,
        )

        lookup_crop = _normalize_lookup_label(crop)
        lookup_disease = _normalize_lookup_label(disease)

        results: list[dict[str, Any]] = []

        for score, index_id in zip(
            scores[0],
            indices[0],
        ):

            if index_id < 0:
                continue

            chunk = dict(
                self.chunks[index_id]
            )

            # Optional application-level context filters.
            #
            # Normalize both the incoming diagnosis label and the
            # persisted knowledge label so equivalent labels such as
            # "northern_leaf_blight" and
            # "Northern Corn Leaf Blight" can match.
            if lookup_crop:
                chunk_crop = _normalize_lookup_label(
                    chunk.get("crop")
                )

                if chunk_crop != lookup_crop:
                    continue

            if lookup_disease:
                chunk_disease = _normalize_lookup_label(
                    chunk.get("disease")
                )

                if chunk_disease != lookup_disease:
                    continue

            # Build the metadata contract expected by the
            # validated NVIDIA Reranker.
            metadata = dict(
                chunk.get("metadata") or {}
            )

            metadata.update(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "knowledge_id": chunk.get("knowledge_id"),
                    "source": chunk.get("source"),
                    "source_type": chunk.get("source_type"),
                    "source_uri": chunk.get("source_uri"),
                    "document_id": chunk.get("document_id"),
                    "document_title": chunk.get("document_title"),
                    "page": chunk.get("page"),
                    "section": chunk.get("section"),
                    "subsection": chunk.get("subsection"),
                    "crop": chunk.get("crop"),
                    "disease": chunk.get("disease"),
                    "evidence_type": chunk.get("evidence_type"),
                    "recommendation_preference": chunk.get(
                        "recommendation_preference"
                    ),
                    "organic_eligible": chunk.get(
                        "organic_eligible"
                    ),
                    "ipm_eligible": chunk.get(
                        "ipm_eligible"
                    ),
                }
            )

            chunk["metadata"] = metadata
            chunk["vector_score"] = float(score)

            results.append(chunk)

        return results

    def info(self) -> dict[str, Any]:

        info = {
            "index_type": type(self.index).__name__,
            "dimension": self.index.d,
            "vectors": self.index.ntotal,
            "chunks": len(self.chunks),
        }

        if self.meta_path.exists():
            info["metadata"] = json.loads(
                self.meta_path.read_text(
                    encoding="utf-8"
                )
            )

        return info
