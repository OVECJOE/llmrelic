"""
Model registry for defining supported models in your application.
"""

from typing import List, Dict, Optional, Set, Iterator
from .models import PROVIDERS


class ModelRegistry:
    """Registry to manage and validate supported models."""

    def __init__(self):
        self._supported_models: Set[str] = set()
        self._provider_filter: Optional[Set[str]] = None

    def __contains__(self, model: str) -> bool:
        """Check if a model is supported."""
        return self.is_supported(model)

    def add_model(self, model: str) -> "ModelRegistry":
        """Add a single model to supported models."""
        self._supported_models.add(model)
        return self

    def add_models(self, models: List[str]) -> "ModelRegistry":
        """Add multiple models to supported models."""
        self._supported_models.update(models)
        return self

    def add_provider(self, provider_name: str) -> "ModelRegistry":
        """Add all models from a provider to supported models."""
        if provider_name.lower() in PROVIDERS:
            provider = PROVIDERS[provider_name.lower()]
            self._supported_models.update(provider.list_models())
        return self

    def add_providers(self, provider_names: List[str]) -> "ModelRegistry":
        """Add all models from multiple providers to supported models."""
        for provider_name in provider_names:
            self.add_provider(provider_name)
        return self

    def remove_model(self, model: str) -> "ModelRegistry":
        """Remove a model from supported models."""
        self._supported_models.discard(model)
        return self

    def is_supported(self, model: str) -> bool:
        """Check if a model is supported."""
        return model in self._supported_models

    def get_supported_models(self) -> List[str]:
        """Get list of supported models."""
        return sorted(self._supported_models)

    def get_supported_by_provider(self) -> Dict[str, List[str]]:
        """Get supported models organized by provider."""
        result = {}
        for provider_name, provider in PROVIDERS.items():
            supported = [
                model for model in provider.list_models()
                if self.is_supported(model)
            ]
            if supported:
                result[provider_name] = supported
        return result

    def clear(self) -> "ModelRegistry":
        """Clear all supported models."""
        self._supported_models.clear()
        return self

    def __iter__(self) -> Iterator[str]:
        """Iterate over supported models."""
        return iter(self.get_supported_models())


class SupportedModels:
    """Fluent interface for easily defining supported models."""

    def __init__(self):
        self.registry = ModelRegistry()

    def __contains__(self, model: str) -> bool:
        """Check if a model is supported."""
        return self.registry.is_supported(model)

    @classmethod
    def create(cls) -> "SupportedModels":
        """Create a new SupportedModels instance."""
        return cls()

    def _register(
        self,
        provider_name: str,
        models: Optional[List[str]] = None,
    ) -> "SupportedModels":
        """Add models from a provider."""
        if models is None:
            self.registry.add_provider(provider_name)
        else:
            # Validate models exist in provider
            provider = PROVIDERS[provider_name.lower()]
            valid_models = [m for m in models if m in provider.list_models()]
            self.registry.add_models(valid_models)
        return self

    def openai(self, models: Optional[List[str]] = None) -> "SupportedModels":
        """Add OpenAI models."""
        return self._register("openai", models)

    def anthropic(
        self,
        models: Optional[List[str]] = None,
    ) -> "SupportedModels":
        """Add Anthropic models."""
        return self._register("anthropic", models)

    def google(self, models: Optional[List[str]] = None) -> "SupportedModels":
        """Add Google models."""
        return self._register("google", models)

    def cohere(self, models: Optional[List[str]] = None) -> "SupportedModels":
        """Add Cohere models."""
        return self._register("cohere", models)

    def mistral(self, models: Optional[List[str]] = None) -> "SupportedModels":
        """Add Mistral models."""
        return self._register("mistral", models)

    def meta(self, models: Optional[List[str]] = None) -> "SupportedModels":
        """Add Meta models."""
        return self._register("meta", models)

    def huggingface(
        self,
        models: Optional[List[str]] = None,
    ) -> "SupportedModels":
        """Add Hugging Face models."""
        return self._register("huggingface", models)
    
    def moonshot(self, models: Optional[List[str]] = None) -> "SupportedModels":
        """Add Moonshot models."""
        return self._register("moonshot", models)

    def custom(self, models: List[str]) -> "SupportedModels":
        """Add custom model names."""
        self.registry.add_models(models)
        return self

    def build(self) -> ModelRegistry:
        """Build the model registry."""
        return self.registry

    def is_supported(self, model: str) -> bool:
        """Check if a model is supported."""
        return self.registry.is_supported(model)

    def get_models(self) -> List[str]:
        """Get list of supported models."""
        return self.registry.get_supported_models()