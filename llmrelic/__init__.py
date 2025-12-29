"""
LLM Relic - A lighweight library for LLM model names and support
definitions.
"""

__version__ = "0.2.1"
__author__ = "OVECJOE"

from .models import (
    OpenAI,
    Anthropic,
    Google,
    Cohere,
    Mistral,
    Meta,
    Huggingface,
    Moonshot,
    get_all_models,
    find_model,
)
from .registry import ModelRegistry, SupportedModels
from .capabilities import (
    ModelCapabilities,
    ModelMetadata,
    Modality,
    PricingTier,
    ModelStatus,
)
from .metadata import MODEL_METADATA, get_metadata
from .finder import ModelFinder, ModelQuery, find_models

__all__ = [
    "OpenAI",
    "Anthropic",
    "Google",
    "Cohere",
    "Mistral",
    "Meta",
    "Huggingface",
    "Moonshot",
    "ModelRegistry",
    "SupportedModels",
    "get_all_models",
    "find_model",
    "ModelCapabilities",
    "ModelMetadata",
    "Modality",
    "PricingTier",
    "ModelStatus",
    "MODEL_METADATA",
    "get_metadata",
    "ModelFinder",
    "ModelQuery",
    "find_models",
]
