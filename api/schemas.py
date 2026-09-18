"""FastAPI request/response schemas for CropGuard."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DiagnosisResponse(BaseModel):
    crop: str | None = None
    disease: str | None = None
    symptoms: str | None = None
    confidence: str | None = None


class EvidenceResponse(BaseModel):
    knowledge_id: str
    chunk_id: str
    source: str
    source_type: str
    crop: str | None = None
    disease: str | None = None
    evidence_type: str | None = None
    recommendation_preference: str | None = None
    reranker_score: float
    vector_score: float | None = None
    bm25_score: float | None = None
    hybrid_score: float | None = None


class CropGuardResponse(BaseModel):
    status: str
    answer: str

    diagnosis: DiagnosisResponse

    recommendation_preference: str
    language: str

    evidence_count: int
    evidence: list[EvidenceResponse] = Field(default_factory=list)

    traceability: list[dict[str, Any]] = Field(default_factory=list)

    model_called: bool
    generated_token_count: int = 0
    truncated: bool = False
    inference_time_seconds: float = 0.0


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    diagnosis_agent: bool
    knowledge_agent: bool
    generation_agent: bool
