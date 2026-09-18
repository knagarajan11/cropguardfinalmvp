"""Knowledge source loaders for CropGuard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from knowledge.schema import KnowledgeRecord


def _read_jsonl(path: Path) -> Iterator[dict]:
    """Read non-empty JSONL records."""
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from exc


def load_normalized_dataset(
    path: str | Path,
) -> list[KnowledgeRecord]:
    """Load PlantVillage, PlantDoc, or Rice normalized metadata.

    These datasets are image/diagnosis evidence rather than treatment
    knowledge. Their content is therefore represented as structured
    diagnosis evidence.
    """

    path = Path(path)
    records: list[KnowledgeRecord] = []

    for item in _read_jsonl(path):
        sample_id = str(item["sample_id"])
        crop = item.get("crop")
        disease = item.get("disease")
        source = item.get("source", path.stem)

        content_parts = []

        if crop:
            content_parts.append(f"Crop: {crop}")

        if disease:
            content_parts.append(f"Disease or condition: {disease}")

        if item.get("health_status"):
            content_parts.append(
                f"Health status: {item['health_status']}"
            )

        if item.get("original_label"):
            content_parts.append(
                f"Original label: {item['original_label']}"
            )

        content = ". ".join(content_parts)

        records.append(
            KnowledgeRecord(
                knowledge_id=sample_id,
                source=source,
                source_type="image_diagnosis_metadata",
                crop=crop,
                disease=disease,
                evidence_type="diagnosis",
                content=content,
                title=f"{source} diagnosis evidence",
                image_path=item.get("image_path"),
                original_label=item.get("original_label"),
                health_status=item.get("health_status"),
                metadata={
                    "dataset_path": str(path),
                },
            )
        )

    return records


def load_treatment_knowledge(
    path: str | Path,
) -> list[KnowledgeRecord]:
    """Load treatment/recommendation knowledge records."""

    path = Path(path)
    records: list[KnowledgeRecord] = []

    for item in _read_jsonl(path):
        records.append(
            KnowledgeRecord(
                knowledge_id=str(item["knowledge_id"]),
                source=item["source"],
                source_type=item.get(
                    "source_type",
                    "agricultural_guidance",
                ),
                crop=item.get("crop"),
                disease=item.get("disease"),
                evidence_type=item.get(
                    "evidence_type",
                    "treatment",
                ),
                content=item["content"],
                title=item.get("title"),
                recommendation_preference=item.get(
                    "recommendation_preference",
                    "General",
                ),
                organic_eligible=bool(
                    item.get("organic_eligible", False)
                ),
                ipm_eligible=bool(
                    item.get("ipm_eligible", False)
                ),
                metadata={
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "knowledge_id",
                        "source",
                        "source_type",
                        "crop",
                        "disease",
                        "evidence_type",
                        "content",
                        "title",
                        "recommendation_preference",
                        "organic_eligible",
                        "ipm_eligible",
                    }
                },
            )
        )

    return records


def load_all_knowledge(
    *,
    plantvillage_path: str | Path,
    plantdoc_path: str | Path,
    rice_path: str | Path,
    treatment_path: str | Path,
) -> list[KnowledgeRecord]:
    """Load all current CropGuard knowledge sources."""

    records: list[KnowledgeRecord] = []

    records.extend(load_normalized_dataset(plantvillage_path))
    records.extend(load_normalized_dataset(plantdoc_path))
    records.extend(load_normalized_dataset(rice_path))
    records.extend(load_treatment_knowledge(treatment_path))

    return records
