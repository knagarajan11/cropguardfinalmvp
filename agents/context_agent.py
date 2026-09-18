"""CropGuard Context Agent.

The Context Agent assembles the contextual information needed by downstream
CropGuard components.

It manages:
- farmer profile
- current diagnosis case
- location
- preferred language
- recommendation preference
- soil context
- weather context

A new crop/image diagnosis starts a new current case. Previous unrelated
diagnoses must not be inherited into the new case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


SUPPORTED_LANGUAGES = (
    "English",
    "Tamil",
    "Kannada",
    "Telugu",
    "Hindi",
    "Malayalam",
    "Marathi",
    "Odia",
    "Punjabi",
    "Bengali",
)

SUPPORTED_RECOMMENDATION_PREFERENCES = (
    "Organic",
    "IPM",
    "General",
)


@dataclass
class FarmerProfile:
    """Persistent farmer-level information."""

    farmer_id: Optional[str] = None
    name: Optional[str] = None
    preferred_language: str = "English"
    location: Optional[str] = None

    def __post_init__(self) -> None:
        if self.preferred_language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: {self.preferred_language}. "
                f"Supported languages: {SUPPORTED_LANGUAGES}"
            )


@dataclass
class DiagnosisCase:
    """Current crop/image diagnosis case.

    This is deliberately separate from FarmerProfile so that a new image
    creates a fresh diagnosis context.
    """

    case_id: Optional[str] = None
    crop: Optional[str] = None
    disease: Optional[str] = None
    symptoms: Optional[str] = None
    confidence: Optional[str] = None
    image_path: Optional[str] = None
    farmer_query: Optional[str] = None


@dataclass
class FarmContext:
    """Environmental context associated with the current case."""

    location: Optional[str] = None
    soil: dict[str, Any] = field(default_factory=dict)
    weather: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextResult:
    """Complete context assembled for downstream agents."""

    farmer: FarmerProfile
    current_case: DiagnosisCase
    farm: FarmContext

    recommendation_preference: str = "General"

    @property
    def language(self) -> str:
        return self.farmer.preferred_language


class ContextAgent:
    """Assemble and manage CropGuard contextual information."""

    def __init__(
        self,
        farmer_profile: Optional[FarmerProfile] = None,
    ) -> None:
        self.farmer_profile = farmer_profile or FarmerProfile()
        self.current_case = DiagnosisCase()
        self.farm_context = FarmContext(
            location=self.farmer_profile.location
        )
        self.recommendation_preference = "General"

    def set_farmer_profile(
        self,
        profile: FarmerProfile,
    ) -> None:
        """Set or replace the persistent farmer profile."""

        self.farmer_profile = profile

        if profile.location:
            self.farm_context.location = profile.location

    def set_recommendation_preference(
        self,
        preference: str,
    ) -> None:
        """Set the farmer's recommendation preference."""

        if preference not in SUPPORTED_RECOMMENDATION_PREFERENCES:
            raise ValueError(
                f"Unsupported recommendation preference: {preference}. "
                f"Supported values: "
                f"{SUPPORTED_RECOMMENDATION_PREFERENCES}"
            )

        self.recommendation_preference = preference

    def start_new_case(
        self,
        *,
        case_id: Optional[str] = None,
        image_path: Optional[str] = None,
        farmer_query: Optional[str] = None,
    ) -> DiagnosisCase:
        """Start a completely new crop/image diagnosis case.

        Previous diagnosis information is intentionally discarded.
        Persistent farmer profile information is retained.
        """

        self.current_case = DiagnosisCase(
            case_id=case_id,
            image_path=image_path,
            farmer_query=farmer_query,
        )

        return self.current_case

    def update_diagnosis(
        self,
        *,
        crop: Optional[str] = None,
        disease: Optional[str] = None,
        symptoms: Optional[str] = None,
        confidence: Optional[str] = None,
    ) -> DiagnosisCase:
        """Attach the latest Diagnosis Agent result to the current case."""

        self.current_case.crop = crop
        self.current_case.disease = disease
        self.current_case.symptoms = symptoms
        self.current_case.confidence = confidence

        return self.current_case

    def update_farm_context(
        self,
        *,
        location: Optional[str] = None,
        soil: Optional[dict[str, Any]] = None,
        weather: Optional[dict[str, Any]] = None,
    ) -> FarmContext:
        """Update environmental context for the current case."""

        if location is not None:
            self.farm_context.location = location

        if soil is not None:
            self.farm_context.soil = dict(soil)

        if weather is not None:
            self.farm_context.weather = dict(weather)

        return self.farm_context

    def build_context(self) -> ContextResult:
        """Build the context consumed by downstream agents."""

        return ContextResult(
            farmer=self.farmer_profile,
            current_case=self.current_case,
            farm=self.farm_context,
            recommendation_preference=self.recommendation_preference,
        )

    def clear_current_case(self) -> None:
        """Clear only the current diagnosis case."""

        self.current_case = DiagnosisCase()
