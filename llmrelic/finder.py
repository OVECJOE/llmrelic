"""
Model finder utilities for searching models by capabilities.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set
from .capabilities import ModelMetadata, PricingTier, ModelStatus, Modality
from .metadata import MODEL_METADATA


@dataclass
class ModelQuery:
    """Query parameters for filtering models by capabilities and attributes."""

    min_context: Optional[int] = None
    max_context: Optional[int] = None
    supports_vision: Optional[bool] = None
    supports_audio: Optional[bool] = None
    supports_function_calling: Optional[bool] = None
    supports_json_mode: Optional[bool] = None
    supports_streaming: Optional[bool] = None
    pricing_tiers: Optional[Set[PricingTier]] = None
    statuses: Optional[Set[ModelStatus]] = None
    providers: Optional[Set[str]] = None
    modalities: Optional[Set[Modality]] = None

    def matches(self, metadata: ModelMetadata) -> bool:
        """Check if a model's metadata matches this query's criteria."""
        if self.min_context and metadata.context_window < self.min_context:
            return False
        if self.max_context and metadata.context_window > self.max_context:
            return False
        if self.supports_vision is not None:
            if metadata.capabilities.supports_vision != self.supports_vision:
                return False
        if self.supports_audio is not None:
            if metadata.capabilities.supports_audio != self.supports_audio:
                return False
        if self.supports_function_calling is not None:
            if metadata.capabilities.supports_function_calling != self.supports_function_calling:
                return False
        if self.supports_json_mode is not None:
            if metadata.capabilities.supports_json_mode != self.supports_json_mode:
                return False
        if self.supports_streaming is not None:
            if metadata.capabilities.supports_streaming != self.supports_streaming:
                return False
        if self.pricing_tiers and metadata.pricing_tier not in self.pricing_tiers:
            return False
        if self.statuses and metadata.status not in self.statuses:
            return False
        if self.providers and metadata.provider not in self.providers:
            return False
        if self.modalities:
            if not self.modalities.issubset(metadata.capabilities.modalities):
                return False
        return True


@dataclass
class ModelFinder:
    """Utility class for searching and filtering models by various criteria."""

    _metadata: dict = field(default_factory=lambda: MODEL_METADATA)

    def find(self, query: ModelQuery) -> List[ModelMetadata]:
        """Find all models matching the given query."""
        return [m for m in self._metadata.values() if query.matches(m)]

    def find_by_context(
        self,
        min_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> List[ModelMetadata]:
        """Find models within a context window range."""
        query = ModelQuery(min_context=min_tokens, max_context=max_tokens)
        return self.find(query)

    def find_with_vision(self) -> List[ModelMetadata]:
        """Find all models that support vision/image inputs."""
        query = ModelQuery(supports_vision=True)
        return self.find(query)

    def find_with_function_calling(self) -> List[ModelMetadata]:
        """Find all models that support function/tool calling."""
        query = ModelQuery(supports_function_calling=True)
        return self.find(query)

    def find_by_pricing(self, *tiers: PricingTier) -> List[ModelMetadata]:
        """Find models matching any of the specified pricing tiers."""
        query = ModelQuery(pricing_tiers=set(tiers))
        return self.find(query)

    def find_active(self) -> List[ModelMetadata]:
        """Find all models with active status."""
        query = ModelQuery(statuses={ModelStatus.ACTIVE})
        return self.find(query)

    def find_by_provider(self, *providers: str) -> List[ModelMetadata]:
        """Find all models from the specified providers."""
        query = ModelQuery(providers={p.lower() for p in providers})
        return self.find(query)

    def find_budget_with_large_context(
        self,
        min_context: int = 32000,
    ) -> List[ModelMetadata]:
        """Find active budget-tier models with large context windows."""
        query = ModelQuery(
            min_context=min_context,
            pricing_tiers={PricingTier.BUDGET},
            statuses={ModelStatus.ACTIVE},
        )
        return self.find(query)

    def find_multimodal(self) -> List[ModelMetadata]:
        """Find all models that support more than one modality."""
        return [
            m for m in self._metadata.values()
            if len(m.capabilities.modalities) > 1
        ]


def find_models(
    min_context: Optional[int] = None,
    max_context: Optional[int] = None,
    supports_vision: Optional[bool] = None,
    supports_function_calling: Optional[bool] = None,
    supports_json_mode: Optional[bool] = None,
    pricing_tier: Optional[PricingTier] = None,
    status: Optional[ModelStatus] = None,
    provider: Optional[str] = None,
) -> List[ModelMetadata]:
    """Search for models matching the specified criteria."""
    query = ModelQuery(
        min_context=min_context,
        max_context=max_context,
        supports_vision=supports_vision,
        supports_function_calling=supports_function_calling,
        supports_json_mode=supports_json_mode,
        pricing_tiers={pricing_tier} if pricing_tier else None,
        statuses={status} if status else None,
        providers={provider.lower()} if provider else None,
    )
    return ModelFinder().find(query)
