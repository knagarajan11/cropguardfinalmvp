"""Canonical knowledge schemas for CropGuard RAG."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class KnowledgeRecord:
    """Canonical source-level knowledge record."""

    knowledge_id: str
    source: str
    source_type: str

    crop: Optional[str]
    disease: Optional[str]

    evidence_type: str
    content: str
    title: Optional[str] = None

    recommendation_preference: str = "General"
    organic_eligible: bool = False
    ipm_eligible: bool = False

    image_path: Optional[str] = None
    original_label: Optional[str] = None
    health_status: Optional[str] = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary."""
        return asdict(self)


@dataclass
class KnowledgeChunk:
    """Canonical retrieval unit produced from a KnowledgeRecord."""

    chunk_id: str
    knowledge_id: str

    source: str
    source_type: str

    crop: Optional[str]
    disease: Optional[str]
    evidence_type: str

    content: str

    recommendation_preference: str = "General"
    organic_eligible: bool = False
    ipm_eligible: bool = False

    title: Optional[str] = None
    chunk_index: int = 0
    total_chunks: int = 1

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary."""
        return asdict(self)
