from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
import yaml
from bs4 import BeautifulSoup
from pypdf import PdfReader


@dataclass
class ExternalKnowledgeRecord:
    """
    Common representation for recommendation knowledge from
    JSON, PDF, and HTML/Web sources.

    Metadata such as crop/disease is content-derived and is
    not inferred from a URL or filename.
    """

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
    metadata: dict[str, Any]


def _stable_id(*parts: str) -> str:
    value = "||".join(str(p) for p in parts)
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"ext-{digest}"


def _clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def _title_from_html(soup: BeautifulSoup) -> str | None:
    if soup.title and soup.title.get_text(strip=True):
        return soup.title.get_text(" ", strip=True)

    heading = soup.find(["h1", "h2"])
    if heading:
        return heading.get_text(" ", strip=True)

    return None


def _extract_html_sections(
    html: str,
    *,
    source_id: str,
    url: str,
    document_title: str | None,
) -> list[ExternalKnowledgeRecord]:

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()

    records: list[ExternalKnowledgeRecord] = []

    current_section: str | None = None
    current_subsection: str | None = None
    buffer: list[str] = []
    section_counter = 0

    def flush_buffer() -> None:
        nonlocal buffer, section_counter

        content = _clean_text("\n".join(buffer))

        if not content:
            buffer = []
            return

        section_counter += 1

        knowledge_id = _stable_id(
            source_id,
            url,
            str(section_counter),
            current_section or "",
            current_subsection or "",
            content,
        )

        records.append(
            ExternalKnowledgeRecord(
                knowledge_id=knowledge_id,
                source=source_id,
                source_type="web",
                source_uri=url,
                document_id=_stable_id(url),
                document_title=document_title,
                page=None,
                section=current_section,
                subsection=current_subsection,
                crop=None,
                disease=None,
                evidence_type=None,
                recommendation_preference="General",
                organic_eligible=False,
                ipm_eligible=False,
                content=content,
                metadata={
                    "source_id": source_id,
                    "url": url,
                    "document_title": document_title,
                    "section": current_section,
                    "subsection": current_subsection,
                },
            )
        )

        buffer = []

    for element in soup.find_all(
        ["h1", "h2", "h3", "h4", "p", "li", "table"]
    ):
        tag_name = element.name.lower()

        if tag_name in {"h1", "h2"}:
            flush_buffer()
            current_section = element.get_text(" ", strip=True)
            current_subsection = None
            continue

        if tag_name in {"h3", "h4"}:
            flush_buffer()
            current_subsection = element.get_text(" ", strip=True)
            continue

        text = element.get_text(" ", strip=True)

        if text:
            buffer.append(text)

    flush_buffer()

    return records



def _normalize_label(value: str | None) -> str | None:
    """Normalize document-derived labels without inventing metadata."""
    if not value:
        return None

    value = " ".join(value.split()).strip(" :;-")
    if not value:
        return None

    return value


def _derive_crop_disease(
    text: str,
    *,
    document_title: str | None = None,
    section: str | None = None,
    subsection: str | None = None,
) -> tuple[str | None, str | None]:
    """
    Derive crop/disease from document content and structural context.

    IMPORTANT:
    - Never derive metadata from URL or source_id.
    - Document title, section, subsection and content are actual
      document-derived evidence.
    - Keep derivation conservative.
    """

    context_parts = [
        document_title or "",
        section or "",
        subsection or "",
        text or "",
    ]

    normalized = " ".join(
        " ".join(part.split())
        for part in context_parts
        if part
    )

    crop = None
    disease = None

    # Explicit crop + disease combinations.
    pattern = (
        r"\b(Tomato|Corn|Maize|Rice|Potato|Wheat|Soybean|"
        r"Cotton|Apple|Cherry)\b"
        r"[\s\-]+"
        r"(Northern Corn Leaf Blight|Northern Leaf Blight|"
        r"Early Blight|Late Blight|Powdery Mildew|"
        r"Bacterial Leaf Blight|Bacterial Canker|Bacterial Speck|"
        r"Leaf Mold|Septoria Leaf Spot|Anthracnose|Gray Mold|"
        r"Phytophthora Blight|White Mold|"
        r"Tomato Spotted Wilt Virus)"
    )

    match = re.search(
        pattern,
        normalized,
        flags=re.IGNORECASE,
    )

    if match:
        crop = _normalize_label(match.group(1))
        disease = _normalize_label(match.group(2))

    # TNAU Tomato disease article header.
    #
    # TNAU pages use a compact article format such as:
    # "Horticultural crops :: Vegetables:: Tomato Damping off : Pythium aphanidermatum"
    # or
    # "Horticultural crops :: Vegetables:: Tomato Bacterial leaf spot : Xanthomonas ..."
    #
    # Derive the disease from the actual page content; do not use URL/source_id.
    tnau_match = re.search(
        r"Horticultural\s+cro(?:p|ps)\s*::\s*Vegetables\s*::\s*"
        r"Tomato\s+([^:]+?)\s*:\s*",
        normalized,
        flags=re.IGNORECASE,
    )

    if tnau_match:
        candidate_disease = tnau_match.group(1).strip()

        # Remove common article-header noise.
        candidate_disease = re.sub(
            r"\s+(?:Field\s+Diagnostic\s+Symptoms|Symptoms|Symptom)$",
            "",
            candidate_disease,
            flags=re.IGNORECASE,
        ).strip()

        if candidate_disease:
            crop = "Tomato"
            disease = candidate_disease

    # Cornell NCLB document terminology.
    if disease is None:
        if re.search(
            r"\bNorthern\s+Corn\s+Leaf\s+Blight\b",
            normalized,
            flags=re.IGNORECASE,
        ) or re.search(
            r"\bNCLB\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            disease = "Northern Corn Leaf Blight"

            if re.search(
                r"\b(sweet\s+corn|field\s+corn|corn)\b",
                normalized,
                flags=re.IGNORECASE,
            ):
                crop = "Corn"

    # Generic disease-only patterns.
    if disease is None:
        disease_patterns = [
            r"\bEarly Blight\b",
            r"\bLate Blight\b",
            r"\bPowdery Mildew\b",
            r"\bBacterial Leaf Blight\b",
            r"\bBacterial Canker\b",
            r"\bBacterial Speck\b",
            r"\bLeaf Mold\b",
            r"\bSeptoria Leaf Spot\b",
            r"\bAnthracnose\b",
            r"\bGray Mold\b",
            r"\bPhytophthora Blight\b",
            r"\bWhite Mold\b",
            r"\bTomato Spotted Wilt Virus\b",
        ]

        for pattern in disease_patterns:
            match = re.search(
                pattern,
                normalized,
                flags=re.IGNORECASE,
            )

            if match:
                disease = _normalize_label(match.group(0))
                break

    # Establish tomato when the actual document content says tomato.
    if crop is None:
        crop_patterns = [
            r"\btomatoes?\b",
            r"\btomato\s+plants?\b",
            r"\btomato\s+seedlings?\b",
        ]

        if any(
            re.search(p, normalized, flags=re.IGNORECASE)
            for p in crop_patterns
        ):
            crop = "Tomato"

    # Existing generic crop derivation.
    if crop is None:
        crop_match = re.search(
            r"\b(?:infected|disease|blight|mildew|rust|rot|"
            r"plants?|leaves?|hosts?)\s+"
            r"(tomato|corn|maize|rice|potato|wheat|soybean|"
            r"cotton|apple|cherry)\b",
            normalized,
            flags=re.IGNORECASE,
        )

        if crop_match:
            crop = _normalize_label(crop_match.group(1))

    # Normalize Maize to Corn for consistent retrieval metadata.
    if crop and crop.lower() == "maize":
        crop = "Corn"

    return crop, disease


def _derive_evidence_type(
    text: str,
    *,
    section: str | None = None,
    subsection: str | None = None,
) -> str | None:
    """
    Derive evidence type from actual document terminology and
    structural section context.

    Section classification takes precedence over incidental words
    appearing in the body text.
    """

    section_text = " ".join(
        part.strip()
        for part in (section or "", subsection or "")
        if part
    ).lower()

    body = (text or "").lower()
    context = f"{section_text} {body}"

    # Structural section semantics.
    if re.search(r"\bdisease\s+management\b|\bmanagement\b", section_text):
        return "treatment"

    if re.search(r"\bsymptoms?\b", section_text):
        return "symptoms"

    if re.search(
        r"\bdisease\s+development\b|\bpathogen\b|"
        r"\bcausal\s+agent\b|\bepidemiology\b",
        section_text,
    ):
        return "epidemiology"

    if re.search(r"\bhosts?\b|\bhost\s+range\b", section_text):
        return "host_range"

    if re.search(r"\bsignificance\b|\bimportance\b|\byield\s+impact\b", section_text):
        return "impact"

    if re.search(r"\bconditions?\b|\bfavourable\s+condition\b|\bfavorable\s+condition\b", section_text):
        return "conditions"

    # Body terminology when structural context is unavailable.
    if re.search(
        r"\bdisease\s+management\s*:",
        body,
    ):
        return "treatment"

    if re.search(
        r"\bmanagement\s*:\b|\btreatment\s*:\b|\bcontrol\s*:\b",
        body,
    ):
        return "treatment"

    if re.search(r"\bsymptoms?\s*:\b", body):
        return "symptoms"

    if re.search(
        r"\bspread\s+and\s+survival\b|"
        r"\bfavourable\s+condition\b|"
        r"\bfavorable\s+condition\b",
        body,
    ):
        return "epidemiology"

    # General management prose can still be treatment evidence.
    if re.search(
        r"\bmanage(?:ment)?\b|\bcontrol(?:ling)?\b|"
        r"\bprevent(?:ion)?\b|\brotate\s+crops?\b|"
        r"\bresistant\s+(?:cultivar|variet(?:y|ies)|hybrid)s?\b",
        context,
    ):
        return "treatment"

    return None



def _derive_recommendation_metadata(
    text: str,
) -> tuple[str, bool, bool]:
    """
    Conservative recommendation-policy derivation.

    Organic/IPM eligibility is TRUE only when the source explicitly
    indicates the corresponding practice or recommendation.
    """
    normalized = text.lower()

    organic = bool(
        re.search(
            r"\borganic\b|\borganically\b|\borganic farming\b",
            normalized,
        )
    )

    ipm = bool(
        re.search(
            r"\bintegrated pest management\b|\bipm\b",
            normalized,
        )
    )

    return "General", organic, ipm


def _clean_external_html_text(text: str) -> str:
    """
    Clean external HTML-derived text conservatively.

    Preserve agricultural source content while removing common
    navigation, footer, validator, and image-source noise.
    """
    text = text.replace("\xa0", " ")
    text = text.replace("Â", " ")

    # Normalize whitespace first.
    text = "\n".join(
        " ".join(line.split())
        for line in text.splitlines()
        if line.strip()
    ).strip()

    # Remove leading site navigation only.
    navigation_markers = [
        "Home | About Us | Success Stories | Farmers Association",
        "Home | About Us | Success Stories",
    ]

    for marker in navigation_markers:
        if text.startswith(marker):
            # The actual crop content starts after "Contact".
            contact_pos = text.find("Contact", len(marker))
            if contact_pos != -1:
                text = text[contact_pos + len("Contact"):].strip()
            break

    # Remove footer from the first copyright marker onward.
    copyright_markers = [
        "© TNAU.",
        "© TNAU",
    ]

    for marker in copyright_markers:
        pos = text.find(marker)
        if pos != -1:
            text = text[:pos].strip()
            break

    # Remove validator/image-source material without consuming the
    # preceding agricultural management content.
    validator_pos = text.find("Content validator:")
    if validator_pos != -1:
        text = text[:validator_pos].strip()

    image_pos = text.find("Source of Images:")
    if image_pos != -1:
        text = text[:image_pos].strip()

    # Remove duplicate paragraphs/blocks conservatively.
    blocks = [
        block.strip()
        for block in text.split("\n\n")
        if block.strip()
    ]

    unique_blocks = []
    seen = set()

    for block in blocks:
        key = " ".join(block.split()).lower()
        if key in seen:
            continue
        seen.add(key)
        unique_blocks.append(block)

    return "\n\n".join(unique_blocks).strip()

def _apply_content_metadata(record: ExternalKnowledgeRecord) -> None:
    """
    Populate metadata from actual document content plus document
    structural context. URL/source configuration is never used
    as crop/disease evidence.
    """
    crop, disease = _derive_crop_disease(
        record.content,
        document_title=record.document_title,
        section=record.section,
        subsection=record.subsection,
    )

    evidence_type = _derive_evidence_type(
        record.content,
        section=record.section,
        subsection=record.subsection,
    )

    preference, organic, ipm = _derive_recommendation_metadata(
        record.content
    )

    if crop:
        record.crop = crop

    if disease:
        record.disease = disease

    if evidence_type:
        record.evidence_type = evidence_type

    record.recommendation_preference = preference
    record.organic_eligible = organic
    record.ipm_eligible = ipm

    record.metadata["metadata_origin"] = "content_derived"



def load_web_source(
    *,
    source_id: str,
    name: str,
    url: str,
    authority: str | None = None,
    verify_ssl: bool = True,
    timeout: int = 30,
) -> list[ExternalKnowledgeRecord]:

    # TNAU currently exposes an expired TLS certificate on some pages.
    # Suppress the warning only for sources explicitly configured with
    # verify_ssl=False.
    if not verify_ssl:
        import urllib3
        from urllib3.exceptions import InsecureRequestWarning

        urllib3.disable_warnings(InsecureRequestWarning)

    response = requests.get(
        url,
        timeout=timeout,
        verify=verify_ssl,
        headers={
            "User-Agent": (
                "CropGuard-MVP/1.0 "
                "(agricultural-recommendation-knowledge-ingestion)"
            )
        },
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    document_title = _title_from_html(soup)

    records = _extract_html_sections(
        response.text,
        source_id=source_id,
        url=url,
        document_title=document_title or name,
    )

    for record in records:
        # Clean navigation/footer/duplicate HTML noise before indexing.
        record.content = _clean_external_html_text(record.content)

        # Derive recommendation metadata from the actual document content.
        _apply_content_metadata(record)

        # Preserve source-level provenance.
        record.metadata["authority"] = authority
        record.metadata["http_status"] = response.status_code
        record.metadata["verify_ssl"] = verify_ssl

    return records


def load_web_registry(
    registry_path: str | Path,
) -> list[ExternalKnowledgeRecord]:

    registry_path = Path(registry_path)

    with registry_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    external = config.get("external_knowledge", {})

    if not external.get("enabled", True):
        return []

    web_config = external.get("web", {})

    if not web_config.get("enabled", True):
        return []

    records: list[ExternalKnowledgeRecord] = []

    for source in web_config.get("sources", []):
        if not source.get("enabled", True):
            continue

        records.extend(
            load_web_source(
                source_id=source["source_id"],
                name=source.get("name", source["source_id"]),
                url=source["url"],
                authority=source.get("authority"),
                verify_ssl=source.get("verify_ssl", True),
            )
        )

    return records


def _pdf_document_id(path: Path) -> str:
    stat = path.stat()

    return _stable_id(
        str(path.resolve()),
        str(stat.st_size),
        str(stat.st_mtime_ns),
    )


def load_pdf(
    pdf_path: str | Path,
) -> list[ExternalKnowledgeRecord]:

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))

    document_id = _pdf_document_id(pdf_path)

    metadata = reader.metadata or {}

    document_title = None

    if metadata.get("/Title"):
        document_title = str(metadata["/Title"])

    if not document_title:
        document_title = pdf_path.stem

    records: list[ExternalKnowledgeRecord] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = _clean_text(text)

        if not text:
            continue

        knowledge_id = _stable_id(
            document_id,
            str(page_number),
            text,
        )

        records.append(
            ExternalKnowledgeRecord(
                knowledge_id=knowledge_id,
                source=pdf_path.name,
                source_type="pdf",
                source_uri=str(pdf_path.resolve()),
                document_id=document_id,
                document_title=document_title,
                page=page_number,
                section=None,
                subsection=None,
                crop=None,
                disease=None,
                evidence_type=None,
                recommendation_preference="General",
                organic_eligible=False,
                ipm_eligible=False,
                content=text,
                metadata={
                    "file_name": pdf_path.name,
                    "file_path": str(pdf_path.resolve()),
                    "page": page_number,
                },
            )
        )

    return records


def load_pdf_directory(
    directory: str | Path,
) -> list[ExternalKnowledgeRecord]:

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"PDF directory not found: {directory}"
        )

    records: list[ExternalKnowledgeRecord] = []

    for pdf_path in sorted(directory.glob("*.pdf")):
        records.extend(load_pdf(pdf_path))

    return records


def load_json_treatment(
    jsonl_path: str | Path,
) -> list[ExternalKnowledgeRecord]:

    jsonl_path = Path(jsonl_path)

    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Treatment JSONL not found: {jsonl_path}"
        )

    records: list[ExternalKnowledgeRecord] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            knowledge_id = item["knowledge_id"]

            records.append(
                ExternalKnowledgeRecord(
                    knowledge_id=knowledge_id,
                    source=item.get("source", "Unknown"),
                    source_type=item.get(
                        "source_type",
                        "json",
                    ),
                    source_uri=None,
                    document_id=knowledge_id,
                    document_title=item.get("title"),
                    page=None,
                    section=None,
                    subsection=None,
                    crop=item.get("crop"),
                    disease=item.get("disease"),
                    evidence_type=item.get("evidence_type"),
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
                    content=item.get("content", ""),
                    metadata={
                        "jsonl_path": str(jsonl_path),
                        "line_number": line_number,
                        "title": item.get("title"),
                    },
                )
            )

    return records


def records_to_jsonl(
    records: list[ExternalKnowledgeRecord],
    output_path: str | Path,
) -> None:

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    asdict(record),
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_all_sources(
    *,
    json_path: str | Path | None = None,
    pdf_directory: str | Path | None = None,
    web_registry: str | Path | None = None,
) -> list[ExternalKnowledgeRecord]:

    records: list[ExternalKnowledgeRecord] = []

    if json_path is not None:
        records.extend(load_json_treatment(json_path))

    if pdf_directory is not None:
        records.extend(load_pdf_directory(pdf_directory))

    if web_registry is not None:
        records.extend(load_web_registry(web_registry))

    return records
