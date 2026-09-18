from __future__ import annotations

from typing import Dict

from .schema import KnowledgeRecord, KnowledgeChunk


def build_retrieval_text(record: KnowledgeRecord) -> str:
    """
    Build the canonical retrieval/embedding text for a KnowledgeRecord.

    Diagnosis evidence and treatment evidence are intentionally represented
    differently so that retrieval can preserve evidence-type separation.
    """

    if record.evidence_type == "diagnosis":
        parts = [
            f"Source: {record.source}",
            "Evidence type: diagnosis",
        ]

        if record.crop:
            parts.append(f"Crop: {record.crop}")

        if record.disease:
            parts.append(f"Disease: {record.disease}")

        if record.health_status:
            parts.append(f"Health status: {record.health_status}")

        if record.original_label:
            parts.append(f"Original label: {record.original_label}")

        return "\n".join(parts)

    if record.evidence_type == "treatment":
        parts = [
            f"Source: {record.source}",
            "Evidence type: treatment",
        ]

        if record.crop:
            parts.append(f"Crop: {record.crop}")

        if record.disease:
            parts.append(f"Disease: {record.disease}")

        if record.recommendation_preference:
            parts.append(
                f"Recommendation preference: "
                f"{record.recommendation_preference}"
            )

        parts.append(
            f"Organic eligible: "
            f"{str(bool(record.organic_eligible)).lower()}"
        )

        parts.append(
            f"IPM eligible: "
            f"{str(bool(record.ipm_eligible)).lower()}"
        )

        if record.title:
            parts.append(f"Title: {record.title}")

        if record.content:
            parts.append(f"Content: {record.content}")

        return "\n".join(parts)

    # Generic fallback for future evidence types.
    parts = [
        f"Source: {record.source}",
        f"Evidence type: {record.evidence_type}",
    ]

    if record.crop:
        parts.append(f"Crop: {record.crop}")

    if record.disease:
        parts.append(f"Disease: {record.disease}")

    if record.title:
        parts.append(f"Title: {record.title}")

    if record.content:
        parts.append(f"Content: {record.content}")

    return "\n".join(parts)


def build_chunk_retrieval_text(chunk: KnowledgeChunk) -> str:
    """
    Build retrieval text from a KnowledgeChunk.

    Chunk metadata is deliberately included because CropGuard retrieval
    needs metadata-aware matching and filtering.
    """

    if chunk.evidence_type == "diagnosis":
        parts = [
            f"Source: {chunk.source}",
            "Evidence type: diagnosis",
        ]

        if chunk.crop:
            parts.append(f"Crop: {chunk.crop}")

        if chunk.disease:
            parts.append(f"Disease: {chunk.disease}")

        parts.append(f"Chunk: {chunk.chunk_index + 1}/{chunk.total_chunks}")

        if chunk.content:
            parts.append(f"Content: {chunk.content}")

        return "\n".join(parts)

    if chunk.evidence_type == "treatment":
        parts = [
            f"Source: {chunk.source}",
            "Evidence type: treatment",
        ]

        if chunk.crop:
            parts.append(f"Crop: {chunk.crop}")

        if chunk.disease:
            parts.append(f"Disease: {chunk.disease}")

        if chunk.recommendation_preference:
            parts.append(
                f"Recommendation preference: "
                f"{chunk.recommendation_preference}"
            )

        parts.append(
            f"Organic eligible: "
            f"{str(bool(chunk.organic_eligible)).lower()}"
        )

        parts.append(
            f"IPM eligible: "
            f"{str(bool(chunk.ipm_eligible)).lower()}"
        )

        if chunk.title:
            parts.append(f"Title: {chunk.title}")

        parts.append(f"Chunk: {chunk.chunk_index + 1}/{chunk.total_chunks}")

        if chunk.content:
            parts.append(f"Content: {chunk.content}")

        return "\n".join(parts)

    return "\n".join(
        part
        for part in [
            f"Source: {chunk.source}",
            f"Evidence type: {chunk.evidence_type}",
            f"Crop: {chunk.crop}" if chunk.crop else "",
            f"Disease: {chunk.disease}" if chunk.disease else "",
            f"Title: {chunk.title}" if chunk.title else "",
            f"Content: {chunk.content}" if chunk.content else "",
        ]
        if part
    )


def retrieval_metadata(
    record: KnowledgeRecord,
) -> Dict[str, object]:
    """
    Metadata retained alongside the embedding.

    This is important because retrieval should not depend only on
    semantic similarity. Knowledge Agent filtering will use these fields.
    """

    return {
        "knowledge_id": record.knowledge_id,
        "source": record.source,
        "source_type": record.source_type,
        "crop": record.crop,
        "disease": record.disease,
        "evidence_type": record.evidence_type,
        "recommendation_preference": record.recommendation_preference,
        "organic_eligible": bool(record.organic_eligible),
        "ipm_eligible": bool(record.ipm_eligible),
        "image_path": record.image_path,
        "original_label": record.original_label,
        "health_status": record.health_status,
    }


def chunk_retrieval_metadata(
    chunk: KnowledgeChunk,
) -> Dict[str, object]:
    """
    Metadata retained alongside a chunk embedding.
    """

    metadata = dict(chunk.metadata or {})

    metadata.update(
        {
            "chunk_id": chunk.chunk_id,
            "knowledge_id": chunk.knowledge_id,
            "source": chunk.source,
            "source_type": chunk.source_type,
            "crop": chunk.crop,
            "disease": chunk.disease,
            "evidence_type": chunk.evidence_type,
            "recommendation_preference":
                chunk.recommendation_preference,
            "organic_eligible": bool(chunk.organic_eligible),
            "ipm_eligible": bool(chunk.ipm_eligible),
            "title": chunk.title,
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
        }
    )

    return metadata
