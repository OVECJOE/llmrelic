"""
LLM Relic - A lighweight library for LLM model names and support
definitions.
"""

__version__ = "0.1.0"
__author__ = "OVECJOE"

from models import (
    OpenAI,
    Anthropic,
    Google,
    Cohere,
    Mistral,
    Meta,
    Huggingface,
    Moonshot
)
from registry import ModelRegistry, SupportedModels

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
]
