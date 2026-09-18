"""Structure-aware chunking for CropGuard knowledge records."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable

from knowledge.schema import KnowledgeChunk, KnowledgeRecord


class KnowledgeChunker:
    """Chunk CropGuard knowledge while preserving useful structure.

    Defaults follow the CropGuard MVP architecture:
    - target: approximately 650 tokens
    - overlap: approximately 100 tokens
    - maximum: 800 tokens
    - minimum: 120 tokens
    """

    def __init__(
        self,
        *,
        target_tokens: int = 650,
        overlap_tokens: int = 100,
        max_tokens: int = 800,
        min_tokens: int = 120,
    ) -> None:
        if not (
            0 < min_tokens <= target_tokens <= max_tokens
        ):
            raise ValueError(
                "Expected min_tokens <= target_tokens <= max_tokens."
            )

        if overlap_tokens >= target_tokens:
            raise ValueError(
                "overlap_tokens must be smaller than target_tokens."
            )

        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Approximate tokens using whitespace splitting.

        The exact NVIDIA tokenizer will be applied later for embedding/
        retrieval validation. This keeps the chunking layer dependency-light.
        """
        return text.split()

    @staticmethod
    def _detokenize(tokens: Iterable[str]) -> str:
        return " ".join(tokens).strip()

    @staticmethod
    def _split_structures(text: str) -> list[str]:
        """Split text while retaining headings, paragraphs and list blocks."""

        text = text.strip()

        if not text:
            return []

        # Preserve paragraph/list boundaries.
        blocks = re.split(r"\n\s*\n+", text)

        result: list[str] = []

        for block in blocks:
            block = block.strip()

            if not block:
                continue

            # Keep list items together where possible.
            lines = block.splitlines()

            if len(lines) > 1:
                current: list[str] = []

                for line in lines:
                    line = line.strip()

                    if not line:
                        continue

                    is_list = bool(
                        re.match(
                            r"^(?:[-*•]|\d+[.)]|[A-Za-z][.)])\s+",
                            line,
                        )
                    )

                    if is_list and current:
                        result.append("\n".join(current))
                        current = []

                    current.append(line)

                if current:
                    result.append("\n".join(current))
            else:
                result.append(block)

        return result

    def _split_large_block(self, block: str) -> list[list[str]]:
        """Split an oversized structural block into token windows."""

        tokens = self._tokenize(block)

        if len(tokens) <= self.max_tokens:
            return [tokens]

        chunks: list[list[str]] = []

        step = self.max_tokens - self.overlap_tokens
        start = 0

        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))
            chunks.append(tokens[start:end])

            if end >= len(tokens):
                break

            start += step

        return chunks

    def chunk_text(self, text: str) -> list[str]:
        """Create structure-aware text chunks."""

        blocks = self._split_structures(text)

        if not blocks:
            return []

        # First divide very large structural blocks.
        token_blocks: list[list[str]] = []

        for block in blocks:
            token_blocks.extend(self._split_large_block(block))

        chunks: list[list[str]] = []
        current: list[str] = []

        for block_tokens in token_blocks:
            if not current:
                current = list(block_tokens)
                continue

            if len(current) + len(block_tokens) <= self.target_tokens:
                current.extend(block_tokens)
                continue

            chunks.append(current)

            overlap = current[-self.overlap_tokens:]
            current = overlap + list(block_tokens)

            # Never allow an overlap to push us beyond max_tokens.
            if len(current) > self.max_tokens:
                chunks.append(current[:self.max_tokens])
                current = current[
                    self.max_tokens - self.overlap_tokens :
                ]

        if current:
            chunks.append(current)

        # Merge tiny trailing chunks where possible.
        merged: list[list[str]] = []

        for chunk in chunks:
            if (
                merged
                and len(chunk) < self.min_tokens
                and len(merged[-1]) + len(chunk) <= self.max_tokens
            ):
                merged[-1].extend(chunk)
            else:
                merged.append(chunk)

        return [
            self._detokenize(chunk)
            for chunk in merged
            if chunk
        ]

    def chunk_record(
        self,
        record: KnowledgeRecord,
    ) -> list[KnowledgeChunk]:
        """Convert one KnowledgeRecord into retrieval chunks."""

        texts = self.chunk_text(record.content)

        if not texts:
            return []

        total = len(texts)
        chunks: list[KnowledgeChunk] = []

        for index, content in enumerate(texts):
            chunk_id = f"{record.knowledge_id}::chunk-{index:04d}"

            metadata = dict(record.metadata)
            metadata.update(
                {
                    "chunk_index": index,
                    "total_chunks": total,
                    "target_tokens": self.target_tokens,
                    "overlap_tokens": self.overlap_tokens,
                    "max_tokens": self.max_tokens,
                    "min_tokens": self.min_tokens,
                }
            )

            chunks.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    knowledge_id=record.knowledge_id,
                    source=record.source,
                    source_type=record.source_type,
                    crop=record.crop,
                    disease=record.disease,
                    evidence_type=record.evidence_type,
                    content=content,
                    recommendation_preference=(
                        record.recommendation_preference
                    ),
                    organic_eligible=record.organic_eligible,
                    ipm_eligible=record.ipm_eligible,
                    title=record.title,
                    chunk_index=index,
                    total_chunks=total,
                    metadata=metadata,
                )
            )

        return chunks
