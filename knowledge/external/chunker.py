from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class RecommendationChunk:
    chunk_id: str
    knowledge_id: str

    source: str
    source_type: str
    source_uri: str | None

    document_id: str
    document_title: str | None

    page: int | None
    section: str | None
    subsection: str | None

    crop: str | None
    disease: str | None
    evidence_type: str | None

    recommendation_preference: str
    organic_eligible: bool
    ipm_eligible: bool

    content: str

    chunk_index: int
    total_chunks: int

    metadata: dict[str, Any]


def _tokenize(text: str) -> list[str]:
    return text.split()


def _detokenize(tokens: Sequence[str]) -> str:
    return " ".join(tokens).strip()


class RecommendationChunker:
    """
    Unified recommendation chunker for JSON, PDF and Web sources.

    Final MVP configuration:
        target: 1000 tokens
        overlap: 150 tokens

    This chunker is separate from the existing validated internal
    treatment chunker, which remains unchanged.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ) -> None:

        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> list[str]:
        tokens = _tokenize(text)

        if not tokens:
            return []

        step = self.chunk_size - self.chunk_overlap
        chunks: list[str] = []

        start = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))

            chunk = _detokenize(tokens[start:end])

            if chunk:
                chunks.append(chunk)

            if end >= len(tokens):
                break

            start += step

        return chunks

    def chunk_record(self, record: Any) -> list[RecommendationChunk]:
        content = getattr(record, "content", "") or ""

        text_chunks = self.chunk_text(content)

        total_chunks = len(text_chunks)

        results: list[RecommendationChunk] = []

        for index, content_chunk in enumerate(text_chunks):

            chunk_id = (
                f"{record.knowledge_id}"
                f"-chunk-{index:04d}"
            )

            metadata = dict(
                getattr(record, "metadata", {}) or {}
            )

            metadata.update(
                {
                    "chunk_index": index,
                    "total_chunks": total_chunks,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                }
            )

            results.append(
                RecommendationChunk(
                    chunk_id=chunk_id,
                    knowledge_id=record.knowledge_id,
                    source=record.source,
                    source_type=record.source_type,
                    source_uri=getattr(record, "source_uri", None),
                    document_id=record.document_id,
                    document_title=getattr(
                        record,
                        "document_title",
                        None,
                    ),
                    page=getattr(record, "page", None),
                    section=getattr(record, "section", None),
                    subsection=getattr(record, "subsection", None),
                    crop=getattr(record, "crop", None),
                    disease=getattr(record, "disease", None),
                    evidence_type=getattr(
                        record,
                        "evidence_type",
                        None,
                    ),
                    recommendation_preference=(
                        getattr(
                            record,
                            "recommendation_preference",
                            "General",
                        )
                        or "General"
                    ),
                    organic_eligible=bool(
                        getattr(
                            record,
                            "organic_eligible",
                            False,
                        )
                    ),
                    ipm_eligible=bool(
                        getattr(
                            record,
                            "ipm_eligible",
                            False,
                        )
                    ),
                    content=content_chunk,
                    chunk_index=index,
                    total_chunks=total_chunks,
                    metadata=metadata,
                )
            )

        return results

    def chunk_records(
        self,
        records: Sequence[Any],
    ) -> list[RecommendationChunk]:

        chunks: list[RecommendationChunk] = []

        for record in records:
            chunks.extend(self.chunk_record(record))

        return chunks
