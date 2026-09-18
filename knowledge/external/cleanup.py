from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from knowledge.external.loaders import ExternalKnowledgeRecord


@dataclass
class CleanupResult:
    records: list[ExternalKnowledgeRecord]
    rejected: list[dict[str, Any]]
    duplicate_count: int
    boilerplate_count: int
    empty_count: int
    short_count: int


_BOILERPLATE_PATTERNS = (
    r"^subscribe by email$",
    r"^receive .* news updates",
    r"^please,? insert a valid email",
    r"^spam protection",
    r"^privacy policy",
    r"^web accessibility",
    r"^log in$",
    r"^sign in$",
    r"^search$",
    r"^cookie",
    r"^accept cookies",
    r"^all rights reserved$",
    r"^source of images:",
    r"^content validator:",
    r"^more information/prepared by:",
)


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    lines = []

    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()

        if line:
            lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _normalize_encoding_artifacts(text: str) -> str:
    replacements = {
        "Â": " ",
        "â€™": "'",
        "â€œ": '"',
        "â€\x9d": '"',
        "â€“": "-",
        "â€”": "-",
        "â€¢": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def _clean_text(text: str) -> str:
    text = _normalize_encoding_artifacts(text)
    text = _normalize_whitespace(text)

    # Remove repeated blank-space artifacts.
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def _content_key(text: str) -> str:
    normalized = " ".join(text.lower().split())
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return normalized.strip()


def _is_boilerplate(record: ExternalKnowledgeRecord) -> bool:
    text = " ".join(record.content.lower().split())

    for pattern in _BOILERPLATE_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True

    boilerplate_terms = (
        "recaptcha",
        "spam protection has stopped",
        "privacy policy",
        "web accessibility",
        "valid email",
        "subscribe by email",
        "source of images:",
        "content validator:",
    )

    return any(term in text for term in boilerplate_terms)


def _is_meaningful(
    record: ExternalKnowledgeRecord,
    *,
    min_content_length: int,
) -> bool:

    text = record.content.strip()

    if len(text) < min_content_length:
        return False

    alpha_count = sum(ch.isalpha() for ch in text)

    if alpha_count < 40:
        return False

    # Reject heading-only records even when their length is marginal.
    if (
        record.source_type == "web"
        and record.section
        and len(text) < 100
        and not record.evidence_type
    ):
        return False

    return True


def _validate_record(
    record: ExternalKnowledgeRecord,
) -> list[str]:

    errors: list[str] = []

    if not record.knowledge_id:
        errors.append("missing knowledge_id")

    if not record.source:
        errors.append("missing source")

    if record.source_type not in {
        "json",
        "pdf",
        "web",
        "mock_agricultural_guidance",
    }:
        errors.append(
            f"invalid source_type: {record.source_type}"
        )

    if not record.content.strip():
        errors.append("empty content")

    if record.source_type in {"pdf", "web"} and not record.source_uri:
        errors.append("missing source_uri")

    if record.recommendation_preference not in {
        "General",
        "Organic",
        "IPM",
    }:
        errors.append(
            "invalid recommendation_preference: "
            f"{record.recommendation_preference}"
        )

    if not isinstance(record.organic_eligible, bool):
        errors.append("organic_eligible is not boolean")

    if not isinstance(record.ipm_eligible, bool):
        errors.append("ipm_eligible is not boolean")

    # Recommendation RAG requires content-derived metadata for
    # external PDF/Web evidence. Controlled JSON records already
    # carry authoritative metadata.
    if record.source_type in {"pdf", "web"}:
        if not record.crop:
            errors.append("missing crop metadata")

        if not record.disease:
            errors.append("missing disease metadata")

        if not record.evidence_type:
            errors.append("missing evidence_type metadata")

    return errors


def clean_and_validate(
    records: list[ExternalKnowledgeRecord],
    *,
    min_content_length: int = 80,
) -> CleanupResult:

    accepted: list[ExternalKnowledgeRecord] = []
    rejected: list[dict[str, Any]] = []

    seen_ids: set[str] = set()
    seen_content: set[str] = set()

    duplicate_count = 0
    boilerplate_count = 0
    empty_count = 0
    short_count = 0

    for record in records:

        record.content = _clean_text(record.content)

        record.metadata["cleanup_applied"] = True

        errors = _validate_record(record)

        if errors:
            rejected.append(
                {
                    "knowledge_id": record.knowledge_id,
                    "source": record.source,
                    "reason": "; ".join(errors),
                    "content_preview": record.content[:300],
                }
            )
            continue

        if not record.content:
            empty_count += 1

            rejected.append(
                {
                    "knowledge_id": record.knowledge_id,
                    "source": record.source,
                    "reason": "empty content after cleanup",
                }
            )

            continue

        if len(record.content) < min_content_length:
            short_count += 1

            rejected.append(
                {
                    "knowledge_id": record.knowledge_id,
                    "source": record.source,
                    "reason": (
                        f"content shorter than minimum "
                        f"{min_content_length} characters"
                    ),
                    "content_preview": record.content[:300],
                }
            )

            continue

        if _is_boilerplate(record):
            boilerplate_count += 1

            rejected.append(
                {
                    "knowledge_id": record.knowledge_id,
                    "source": record.source,
                    "reason": "boilerplate/non-knowledge content",
                    "content_preview": record.content[:300],
                }
            )

            continue

        if not _is_meaningful(
            record,
            min_content_length=min_content_length,
        ):
            rejected.append(
                {
                    "knowledge_id": record.knowledge_id,
                    "source": record.source,
                    "reason": "content not meaningful for indexing",
                    "content_preview": record.content[:300],
                }
            )

            continue

        if record.knowledge_id in seen_ids:
            duplicate_count += 1

            rejected.append(
                {
                    "knowledge_id": record.knowledge_id,
                    "source": record.source,
                    "reason": "duplicate knowledge_id",
                }
            )

            continue

        content_key = _content_key(record.content)

        if content_key in seen_content:
            duplicate_count += 1

            rejected.append(
                {
                    "knowledge_id": record.knowledge_id,
                    "source": record.source,
                    "reason": "duplicate content",
                }
            )

            continue

        seen_ids.add(record.knowledge_id)
        seen_content.add(content_key)

        accepted.append(record)

    return CleanupResult(
        records=accepted,
        rejected=rejected,
        duplicate_count=duplicate_count,
        boilerplate_count=boilerplate_count,
        empty_count=empty_count,
        short_count=short_count,
    )


def print_cleanup_report(
    result: CleanupResult,
    *,
    original_count: int,
) -> None:

    source_counts = Counter(
        r.source_type
        for r in result.records
    )

    print("=" * 64)
    print("CropGuard Recommendation Knowledge Cleanup")
    print("=" * 64)

    print("\nINPUT")
    print(f"  records loaded:        {original_count}")

    print("\nVALIDATED")
    print(f"  records accepted:     {len(result.records)}")

    for source_type, count in sorted(source_counts.items()):
        print(f"  {source_type:22s}: {count}")

    print("\nREJECTED / REMOVED")
    print(f"  duplicates:            {result.duplicate_count}")
    print(f"  boilerplate:           {result.boilerplate_count}")
    print(f"  empty:                 {result.empty_count}")
    print(f"  too short:             {result.short_count}")

    counted = (
        result.duplicate_count
        + result.boilerplate_count
        + result.empty_count
        + result.short_count
    )

    other = max(0, len(result.rejected) - counted)

    print(f"  validation/other:      {other}")
    print(f"  total rejected:        {len(result.rejected)}")

    print("\nCONTENT QUALITY")

    print(
        "  crop identified:       "
        f"{sum(r.crop is not None for r in result.records)}"
    )

    print(
        "  disease identified:    "
        f"{sum(r.disease is not None for r in result.records)}"
    )

    print(
        "  evidence type:         "
        f"{sum(r.evidence_type is not None for r in result.records)}"
    )

    print("\nSOURCE COUNTS")

    for source, count in Counter(
        r.source
        for r in result.records
    ).most_common():
        print(f"  {source:45s} {count}")

    print("\nRESULT")
    print("  CLEANUP + VALIDATION: PASS")
    print("=" * 64)
