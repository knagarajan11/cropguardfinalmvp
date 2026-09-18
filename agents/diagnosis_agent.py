"""CropGuard Diagnosis Agent.

The Diagnosis Agent is responsible only for multimodal crop diagnosis.
Treatment and recommendations must be grounded later by the Knowledge Agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from vision.nemotron_omni import (
    NemotronGeneration,
    NemotronOmniConfig,
    NemotronOmniVisionModel,
)


@dataclass
class DiagnosisResult:
    """Structured output from the CropGuard Diagnosis Agent."""

    crop: Optional[str]
    disease: Optional[str]
    symptoms: Optional[str]
    confidence: Optional[str]

    raw_text: str
    final_text: str

    generated_token_count: int
    truncated: bool
    inference_time_seconds: float

    @property
    def success(self) -> bool:
        """Whether a diagnosis was extracted from the model response."""
        return bool(self.crop or self.disease)


class DiagnosisAgent:
    """CropGuard multimodal Diagnosis Agent.

    This class deliberately does not provide treatment recommendations.
    Treatment must be supplied by the Knowledge Agent after evidence retrieval.
    """

    DEFAULT_PROMPT = """
Analyze this crop image for CropGuard.

Identify:
1. Crop or plant
2. Most likely disease or condition
3. Key visual symptoms supporting the diagnosis
4. Confidence

Base the diagnosis on the visible image.
Do not invent information that cannot be inferred from the image.
Do not provide pesticide, fungicide, dosage, or treatment recommendations.
"""

    def __init__(
        self,
        config: NemotronOmniConfig | None = None,
        *,
        model: NemotronOmniVisionModel | None = None,
    ) -> None:
        """Create the Diagnosis Agent.

        A pre-loaded model can be supplied so multiple agents share the
        same frozen Nemotron Omni + CropGuard LoRA instance.
        """

        if model is not None:
            self.model = model
        elif config is not None:
            self.model = NemotronOmniVisionModel(config)
        else:
            raise ValueError("Either config or model must be provided.")

    def load(self) -> None:
        """Load the validated Nemotron Omni + CropGuard LoRA."""
        self.model.load()

    def diagnose(
        self,
        image: Image.Image,
        farmer_query: str = "",
        *,
        max_new_tokens: int | None = None,
    ) -> DiagnosisResult:
        """Diagnose a crop image."""

        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image instance.")

        query = farmer_query.strip()

        prompt = self.DEFAULT_PROMPT

        if query:
            prompt += f"""

Farmer's question:
{query}

Answer the question only to the extent that it is supported by the image.
"""

        generation = self.model.diagnose(
            image=image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        return self._build_result(generation)

    def close(self) -> None:
        """Release model resources."""
        self.model.close()

    @staticmethod
    def _build_result(
        generation: NemotronGeneration,
    ) -> DiagnosisResult:
        text = generation.final_text.strip()

        return DiagnosisResult(
            crop=DiagnosisAgent._extract_crop(text),
            disease=DiagnosisAgent._extract_field(
                text,
                "Most likely disease or condition",
            ),
            symptoms=DiagnosisAgent._extract_field(
                text,
                "Key visual symptoms supporting the diagnosis",
            ),
            confidence=DiagnosisAgent._extract_field(
                text,
                "Confidence",
            ),
            raw_text=generation.raw_text,
            final_text=text,
            generated_token_count=generation.generated_token_count,
            truncated=generation.truncated,
            inference_time_seconds=generation.inference_time_seconds,
        )

    @staticmethod
    def _extract_crop(text: str) -> Optional[str]:
        """Extract crop name from plain or Markdown-formatted labels."""
        return DiagnosisAgent._extract_label(
            text,
            ("Crop", "Crop or plant"),
        )

    @staticmethod
    def _extract_field(
        text: str,
        field_name: str,
    ) -> Optional[str]:
        """Extract a structured field with backward-compatible aliases."""

        aliases = {
            "Most likely disease or condition": (
                "Most likely disease or condition",
                "Most likely disease",
                "Disease",
                "Disease or condition",
            ),
            "Key visual symptoms supporting the diagnosis": (
                "Key visual symptoms supporting the diagnosis",
                "Key visual symptoms",
                "Symptoms",
                "Key symptoms",
                "Visual symptoms",
            ),
            "Confidence": (
                "Confidence",
            ),
        }

        candidates = aliases.get(
            field_name,
            (field_name,),
        )

        return DiagnosisAgent._extract_label(
            text,
            candidates,
        )

    @staticmethod
    def _extract_label(
        text: str,
        candidates: tuple[str, ...],
    ) -> Optional[str]:
        """
        Extract a label from plain text or Markdown.

        Supported examples:

            Crop: Corn
            **Crop:** Corn
            * **Crop:** Corn
            *   **Crop:** Corn
            - **Confidence:** High
        """

        for candidate in candidates:
            pattern = (
                rf"(?im)^\s*"
                rf"(?:[-*+]\s+)?"
                rf"(?:\*\*)?"
                rf"{re.escape(candidate)}"
                rf"(?:\*\*)?"
                rf"\s*:\s*"
                rf"(.+?)\s*$"
            )

            match = re.search(pattern, text)

            if match:
                value = match.group(1).strip()

                # Remove Markdown emphasis left around the value.
                value = value.strip("*").strip()

                # Remove trailing punctuation.
                value = value.rstrip(" .;:").strip()

                if value:
                    return value

        return None

