"""
Model capabilities and metadata definitions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Set


class Modality(Enum):
    """Supported input/output modalities for a model."""

    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    VIDEO = "video"


class PricingTier(Enum):
    """Relative pricing tier for a model."""

    BUDGET = "budget"
    STANDARD = "standard"
    PREMIUM = "premium"


class ModelStatus(Enum):
    """Current availability status of a model."""

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    BETA = "beta"
    PREVIEW = "preview"


@dataclass(frozen=True)
class ModelCapabilities:
    """Immutable dataclass representing a model's capabilities."""
    supports_vision: bool = False
    supports_audio: bool = False
    supports_video: bool = False
    supports_function_calling: bool = False
    supports_json_mode: bool = False
    supports_streaming: bool = True
    supports_system_message: bool = True

    @property
    def modalities(self) -> Set[Modality]:
        """Return set of all supported modalities for this model."""
        result = {Modality.TEXT}
        if self.supports_vision:
            result.add(Modality.VISION)
        if self.supports_audio:
            result.add(Modality.AUDIO)
        if self.supports_video:
            result.add(Modality.VIDEO)
        return result


@dataclass(frozen=True)
class ModelMetadata:
    """Immutable dataclass containing complete metadata for a model."""
    name: str
    provider: str
    context_window: int
    max_output_tokens: Optional[int] = None
    training_cutoff: Optional[str] = None
    pricing_tier: PricingTier = PricingTier.STANDARD
    status: ModelStatus = ModelStatus.ACTIVE
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)

    @property
    def supports_vision(self) -> bool:
        """Check if the model supports vision/image inputs."""
        return self.capabilities.supports_vision

    @property
    def supports_function_calling(self) -> bool:
        """Check if the model supports function/tool calling."""
        return self.capabilities.supports_function_calling

    @property
    def supports_json_mode(self) -> bool:
        """Check if the model supports structured JSON output mode."""
        return self.capabilities.supports_json_mode

    def has_min_context(self, min_tokens: int) -> bool:
        """Check if the model's context window meets the minimum requirement."""
        return self.context_window >= min_tokens

    def is_active(self) -> bool:
        """Check if the model is currently active and available."""
        return self.status == ModelStatus.ACTIVE

    def is_deprecated(self) -> bool:
        """Check if the model has been deprecated."""
        return self.status == ModelStatus.DEPRECATED


DEFAULT_CAPABILITIES = ModelCapabilities()

VISION_CAPABILITIES = ModelCapabilities(
    supports_vision=True,
    supports_function_calling=True,
    supports_json_mode=True,
)

ADVANCED_CAPABILITIES = ModelCapabilities(
    supports_function_calling=True,
    supports_json_mode=True,
)

BASIC_CAPABILITIES = ModelCapabilities(
    supports_streaming=True,
    supports_system_message=True,
)

MULTIMODAL_CAPABILITIES = ModelCapabilities(
    supports_vision=True,
    supports_audio=True,
    supports_function_calling=True,
    supports_json_mode=True,
)
