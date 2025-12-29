"""
Model metadata registry with capabilities for all supported models.
"""

from typing import Dict, Optional
from .capabilities import (
    ModelMetadata,
    PricingTier,
    ModelStatus,
    ADVANCED_CAPABILITIES,
    VISION_CAPABILITIES,
    BASIC_CAPABILITIES,
    MULTIMODAL_CAPABILITIES,
)


OPENAI_METADATA: Dict[str, ModelMetadata] = {
    "gpt-4": ModelMetadata(
        name="gpt-4",
        provider="openai",
        context_window=8192,
        max_output_tokens=8192,
        training_cutoff="2023-04",
        pricing_tier=PricingTier.PREMIUM,
        capabilities=ADVANCED_CAPABILITIES,
    ),
    "gpt-4-turbo": ModelMetadata(
        name="gpt-4-turbo",
        provider="openai",
        context_window=128000,
        max_output_tokens=4096,
        training_cutoff="2023-12",
        pricing_tier=PricingTier.PREMIUM,
        capabilities=VISION_CAPABILITIES,
    ),
    "gpt-4-turbo-preview": ModelMetadata(
        name="gpt-4-turbo-preview",
        provider="openai",
        context_window=128000,
        max_output_tokens=4096,
        training_cutoff="2023-12",
        pricing_tier=PricingTier.PREMIUM,
        status=ModelStatus.PREVIEW,
        capabilities=VISION_CAPABILITIES,
    ),
    "gpt-4-32k": ModelMetadata(
        name="gpt-4-32k",
        provider="openai",
        context_window=32768,
        max_output_tokens=32768,
        training_cutoff="2023-04",
        pricing_tier=PricingTier.PREMIUM,
        capabilities=ADVANCED_CAPABILITIES,
    ),
    "gpt-3.5-turbo": ModelMetadata(
        name="gpt-3.5-turbo",
        provider="openai",
        context_window=16385,
        max_output_tokens=4096,
        training_cutoff="2021-09",
        pricing_tier=PricingTier.BUDGET,
        capabilities=ADVANCED_CAPABILITIES,
    ),
    "gpt-3.5-turbo-16k": ModelMetadata(
        name="gpt-3.5-turbo-16k",
        provider="openai",
        context_window=16385,
        max_output_tokens=4096,
        training_cutoff="2021-09",
        pricing_tier=PricingTier.BUDGET,
        capabilities=ADVANCED_CAPABILITIES,
    ),
    "gpt-3.5-turbo-instruct": ModelMetadata(
        name="gpt-3.5-turbo-instruct",
        provider="openai",
        context_window=4096,
        max_output_tokens=4096,
        training_cutoff="2021-09",
        pricing_tier=PricingTier.BUDGET,
        capabilities=BASIC_CAPABILITIES,
    ),
    "davinci-002": ModelMetadata(
        name="davinci-002",
        provider="openai",
        context_window=16384,
        max_output_tokens=4096,
        pricing_tier=PricingTier.STANDARD,
        status=ModelStatus.DEPRECATED,
        capabilities=BASIC_CAPABILITIES,
    ),
    "babbage-002": ModelMetadata(
        name="babbage-002",
        provider="openai",
        context_window=16384,
        max_output_tokens=4096,
        pricing_tier=PricingTier.BUDGET,
        status=ModelStatus.DEPRECATED,
        capabilities=BASIC_CAPABILITIES,
    ),
}


ANTHROPIC_METADATA: Dict[str, ModelMetadata] = {
    "claude-3-opus-20240229": ModelMetadata(
        name="claude-3-opus-20240229",
        provider="anthropic",
        context_window=200000,
        max_output_tokens=4096,
        training_cutoff="2023-08",
        pricing_tier=PricingTier.PREMIUM,
        capabilities=VISION_CAPABILITIES,
    ),
    "claude-3-sonnet-20240229": ModelMetadata(
        name="claude-3-sonnet-20240229",
        provider="anthropic",
        context_window=200000,
        max_output_tokens=4096,
        training_cutoff="2023-08",
        pricing_tier=PricingTier.STANDARD,
        capabilities=VISION_CAPABILITIES,
    ),
    "claude-3-haiku-20240307": ModelMetadata(
        name="claude-3-haiku-20240307",
        provider="anthropic",
        context_window=200000,
        max_output_tokens=4096,
        training_cutoff="2023-08",
        pricing_tier=PricingTier.BUDGET,
        capabilities=VISION_CAPABILITIES,
    ),
    "claude-2.1": ModelMetadata(
        name="claude-2.1",
        provider="anthropic",
        context_window=200000,
        max_output_tokens=4096,
        training_cutoff="2023-01",
        pricing_tier=PricingTier.STANDARD,
        capabilities=ADVANCED_CAPABILITIES,
    ),
    "claude-2.0": ModelMetadata(
        name="claude-2.0",
        provider="anthropic",
        context_window=100000,
        max_output_tokens=4096,
        training_cutoff="2023-01",
        pricing_tier=PricingTier.STANDARD,
        status=ModelStatus.DEPRECATED,
        capabilities=BASIC_CAPABILITIES,
    ),
    "claude-instant-1.2": ModelMetadata(
        name="claude-instant-1.2",
        provider="anthropic",
        context_window=100000,
        max_output_tokens=4096,
        pricing_tier=PricingTier.BUDGET,
        status=ModelStatus.DEPRECATED,
        capabilities=BASIC_CAPABILITIES,
    ),
}


GOOGLE_METADATA: Dict[str, ModelMetadata] = {
    "gemini-pro": ModelMetadata(
        name="gemini-pro",
        provider="google",
        context_window=32760,
        max_output_tokens=8192,
        pricing_tier=PricingTier.STANDARD,
        capabilities=ADVANCED_CAPABILITIES,
    ),
    "gemini-pro-vision": ModelMetadata(
        name="gemini-pro-vision",
        provider="google",
        context_window=16384,
        max_output_tokens=2048,
        pricing_tier=PricingTier.STANDARD,
        capabilities=VISION_CAPABILITIES,
    ),
    "gemini-1.5-pro": ModelMetadata(
        name="gemini-1.5-pro",
        provider="google",
        context_window=1000000,
        max_output_tokens=8192,
        pricing_tier=PricingTier.PREMIUM,
        capabilities=MULTIMODAL_CAPABILITIES,
    ),
    "gemini-1.5-flash": ModelMetadata(
        name="gemini-1.5-flash",
        provider="google",
        context_window=1000000,
        max_output_tokens=8192,
        pricing_tier=PricingTier.BUDGET,
        capabilities=MULTIMODAL_CAPABILITIES,
    ),
    "bard": ModelMetadata(
        name="bard",
        provider="google",
        context_window=8192,
        pricing_tier=PricingTier.STANDARD,
        status=ModelStatus.DEPRECATED,
        capabilities=BASIC_CAPABILITIES,
    ),
    "palm-2": ModelMetadata(
        name="palm-2",
        provider="google",
        context_window=8192,
        pricing_tier=PricingTier.STANDARD,
        status=ModelStatus.DEPRECATED,
        capabilities=BASIC_CAPABILITIES,
    ),
    "palm-2-chat": ModelMetadata(
        name="palm-2-chat",
        provider="google",
        context_window=8192,
        pricing_tier=PricingTier.STANDARD,
        status=ModelStatus.DEPRECATED,
        capabilities=BASIC_CAPABILITIES,
    ),
}


COHERE_METADATA: Dict[str, ModelMetadata] = {
    "command": ModelMetadata(
        name="command",
        provider="cohere",
        context_window=4096,
        pricing_tier=PricingTier.STANDARD,
        capabilities=ADVANCED_CAPABILITIES,
    ),
    "command-light": ModelMetadata(
        name="command-light",
        provider="cohere",
        context_window=4096,
        pricing_tier=PricingTier.BUDGET,
        capabilities=BASIC_CAPABILITIES,
    ),
    "command-nightly": ModelMetadata(
        name="command-nightly",
        provider="cohere",
        context_window=4096,
        pricing_tier=PricingTier.STANDARD,
        status=ModelStatus.BETA,
        capabilities=ADVANCED_CAPABILITIES,
    ),
    "command-r": ModelMetadata(
        name="command-r",
        provider="cohere",
        context_window=128000,
        pricing_tier=PricingTier.STANDARD,
        capabilities=ADVANCED_CAPABILITIES,
    ),
    "command-r-plus": ModelMetadata(
        name="command-r-plus",
        provider="cohere",
        context_window=128000,
        pricing_tier=PricingTier.PREMIUM,
        capabilities=ADVANCED_CAPABILITIES,
    ),
}


MISTRAL_METADATA: Dict[str, ModelMetadata] = {
    "mistral-7b-instruct": ModelMetadata(
        name="mistral-7b-instruct",
        provider="mistral",
        context_window=32768,
        pricing_tier=PricingTier.BUDGET,
        capabilities=BASIC_CAPABILITIES,
    ),
    "mistral-8x7b-instruct": ModelMetadata(
        name="mistral-8x7b-instruct",
        provider="mistral",
        context_window=32768,
        pricing_tier=PricingTier.STANDARD,
        capabilities=BASIC_CAPABILITIES,
    ),
    "mistral-small-latest": ModelMetadata(
        name="mistral-small-latest",
        provider="mistral",
        context_window=32768,
        pricing_tier=PricingTier.BUDGET,
        capabilities=ADVANCED_CAPABILITIES,
    ),
    "mistral-medium-latest": ModelMetadata(
        name="mistral-medium-latest",
        provider="mistral",
        context_window=32768,
        pricing_tier=PricingTier.STANDARD,
        capabilities=ADVANCED_CAPABILITIES,
    ),
    "mistral-large-latest": ModelMetadata(
        name="mistral-large-latest",
        provider="mistral",
        context_window=32768,
        pricing_tier=PricingTier.PREMIUM,
        capabilities=ADVANCED_CAPABILITIES,
    ),
    "mixtral-8x7b-instruct": ModelMetadata(
        name="mixtral-8x7b-instruct",
        provider="mistral",
        context_window=32768,
        pricing_tier=PricingTier.STANDARD,
        capabilities=BASIC_CAPABILITIES,
    ),
}


META_METADATA: Dict[str, ModelMetadata] = {
    "llama-2-7b-chat": ModelMetadata(
        name="llama-2-7b-chat",
        provider="meta",
        context_window=4096,
        pricing_tier=PricingTier.BUDGET,
        capabilities=BASIC_CAPABILITIES,
    ),
    "llama-2-13b-chat": ModelMetadata(
        name="llama-2-13b-chat",
        provider="meta",
        context_window=4096,
        pricing_tier=PricingTier.BUDGET,
        capabilities=BASIC_CAPABILITIES,
    ),
    "llama-2-70b-chat": ModelMetadata(
        name="llama-2-70b-chat",
        provider="meta",
        context_window=4096,
        pricing_tier=PricingTier.STANDARD,
        capabilities=BASIC_CAPABILITIES,
    ),
    "code-llama-7b-instruct": ModelMetadata(
        name="code-llama-7b-instruct",
        provider="meta",
        context_window=16384,
        pricing_tier=PricingTier.BUDGET,
        capabilities=BASIC_CAPABILITIES,
    ),
    "code-llama-13b-instruct": ModelMetadata(
        name="code-llama-13b-instruct",
        provider="meta",
        context_window=16384,
        pricing_tier=PricingTier.BUDGET,
        capabilities=BASIC_CAPABILITIES,
    ),
    "code-llama-34b-instruct": ModelMetadata(
        name="code-llama-34b-instruct",
        provider="meta",
        context_window=16384,
        pricing_tier=PricingTier.STANDARD,
        capabilities=BASIC_CAPABILITIES,
    ),
}


HUGGINGFACE_METADATA: Dict[str, ModelMetadata] = {
    "tiiuae/falcon-7b-instruct": ModelMetadata(
        name="tiiuae/falcon-7b-instruct",
        provider="huggingface",
        context_window=2048,
        pricing_tier=PricingTier.BUDGET,
        capabilities=BASIC_CAPABILITIES,
    ),
    "tiiuae/falcon-40b-instruct": ModelMetadata(
        name="tiiuae/falcon-40b-instruct",
        provider="huggingface",
        context_window=2048,
        pricing_tier=PricingTier.STANDARD,
        capabilities=BASIC_CAPABILITIES,
    ),
    "lmsys/vicuna-7b-v1.5": ModelMetadata(
        name="lmsys/vicuna-7b-v1.5",
        provider="huggingface",
        context_window=4096,
        pricing_tier=PricingTier.BUDGET,
        capabilities=BASIC_CAPABILITIES,
    ),
    "lmsys/vicuna-13b-v1.5": ModelMetadata(
        name="lmsys/vicuna-13b-v1.5",
        provider="huggingface",
        context_window=4096,
        pricing_tier=PricingTier.BUDGET,
        capabilities=BASIC_CAPABILITIES,
    ),
    "chavinlo/alpaca-native": ModelMetadata(
        name="chavinlo/alpaca-native",
        provider="huggingface",
        context_window=2048,
        pricing_tier=PricingTier.BUDGET,
        capabilities=BASIC_CAPABILITIES,
    ),
    "google/flan-t5-small": ModelMetadata(
        name="google/flan-t5-small",
        provider="huggingface",
        context_window=512,
        pricing_tier=PricingTier.BUDGET,
        capabilities=BASIC_CAPABILITIES,
    ),
    "google/flan-t5-base": ModelMetadata(
        name="google/flan-t5-base",
        provider="huggingface",
        context_window=512,
        pricing_tier=PricingTier.BUDGET,
        capabilities=BASIC_CAPABILITIES,
    ),
    "google/flan-t5-large": ModelMetadata(
        name="google/flan-t5-large",
        provider="huggingface",
        context_window=512,
        pricing_tier=PricingTier.BUDGET,
        capabilities=BASIC_CAPABILITIES,
    ),
    "google/flan-t5-xl": ModelMetadata(
        name="google/flan-t5-xl",
        provider="huggingface",
        context_window=512,
        pricing_tier=PricingTier.BUDGET,
        capabilities=BASIC_CAPABILITIES,
    ),
    "google/flan-t5-xxl": ModelMetadata(
        name="google/flan-t5-xxl",
        provider="huggingface",
        context_window=512,
        pricing_tier=PricingTier.STANDARD,
        capabilities=BASIC_CAPABILITIES,
    ),
}


MOONSHOT_METADATA: Dict[str, ModelMetadata] = {
    "moonshot-v1-8k": ModelMetadata(
        name="moonshot-v1-8k",
        provider="moonshot",
        context_window=8192,
        pricing_tier=PricingTier.BUDGET,
        capabilities=ADVANCED_CAPABILITIES,
    ),
    "moonshot-v1-32k": ModelMetadata(
        name="moonshot-v1-32k",
        provider="moonshot",
        context_window=32768,
        pricing_tier=PricingTier.STANDARD,
        capabilities=ADVANCED_CAPABILITIES,
    ),
    "moonshot-v1-128k": ModelMetadata(
        name="moonshot-v1-128k",
        provider="moonshot",
        context_window=131072,
        pricing_tier=PricingTier.PREMIUM,
        capabilities=ADVANCED_CAPABILITIES,
    ),
    "moonshot-v1-auto": ModelMetadata(
        name="moonshot-v1-auto",
        provider="moonshot",
        context_window=131072,
        pricing_tier=PricingTier.STANDARD,
        capabilities=ADVANCED_CAPABILITIES,
    ),
    "moonshot-v1-8k-vision-preview": ModelMetadata(
        name="moonshot-v1-8k-vision-preview",
        provider="moonshot",
        context_window=8192,
        pricing_tier=PricingTier.STANDARD,
        status=ModelStatus.PREVIEW,
        capabilities=VISION_CAPABILITIES,
    ),
    "moonshot-v1-32k-vision-preview": ModelMetadata(
        name="moonshot-v1-32k-vision-preview",
        provider="moonshot",
        context_window=32768,
        pricing_tier=PricingTier.STANDARD,
        status=ModelStatus.PREVIEW,
        capabilities=VISION_CAPABILITIES,
    ),
    "moonshot-v1-128k-vision-preview": ModelMetadata(
        name="moonshot-v1-128k-vision-preview",
        provider="moonshot",
        context_window=131072,
        pricing_tier=PricingTier.PREMIUM,
        status=ModelStatus.PREVIEW,
        capabilities=VISION_CAPABILITIES,
    ),
}


MODEL_METADATA: Dict[str, ModelMetadata] = {
    **OPENAI_METADATA,
    **ANTHROPIC_METADATA,
    **GOOGLE_METADATA,
    **COHERE_METADATA,
    **MISTRAL_METADATA,
    **META_METADATA,
    **HUGGINGFACE_METADATA,
    **MOONSHOT_METADATA,
}


def get_metadata(model_name: str) -> Optional[ModelMetadata]:
    """Get metadata for a model by its name, or None if not found."""
    return MODEL_METADATA.get(model_name)
