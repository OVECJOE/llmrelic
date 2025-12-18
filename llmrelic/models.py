"""
LLM model name constants organized by provider.
"""

from typing import List, Dict, Optional
from .metadata import MODEL_METADATA, ModelMetadata


class ModelProvider:
    """Base class for model providers."""

    def __init__(self, name: str, models: Dict[str, str]):
        self.name = name
        self._models = models

    def __getattr__(self, name: str) -> str:
        """Allow accessing models as attributes."""
        if name.upper() in self._models:
            return self._models[name.upper()]
        raise AttributeError(
            f"'{self.__class__.__name__}' has no model named '{name}'"
        )

    def list_models(self) -> List[str]:
        """Return list of all model names."""
        return list(self._models.values())

    def get_model(self, key: str) -> str:
        """Get model name by key."""
        return self._models.get(key, key)

    def get_metadata(self, model_name: str) -> Optional[ModelMetadata]:
        """Get metadata for a specific model by name."""
        if model_name in self._models.values():
            return MODEL_METADATA.get(model_name)
        return None

    def list_metadata(self) -> List[ModelMetadata]:
        """Return metadata for all models from this provider."""
        return [
            MODEL_METADATA[model]
            for model in self._models.values()
            if model in MODEL_METADATA
        ]

    def __str__(self) -> str:
        """Return provider name."""
        return self.name

    def __contains__(self, model: str) -> bool:
        """Check if model exists in this provider."""
        return model in self._models.values()


# OpenAI models
class OpenAIModels(ModelProvider):
    def __init__(self):
        models = {
            "GPT_4": "gpt-4",
            "GPT_4_TURBO": "gpt-4-turbo",
            "GPT_4_TURBO_PREVIEW": "gpt-4-turbo-preview",
            "GPT_4_32K": "gpt-4-32k",
            "GPT_3_5_TURBO": "gpt-3.5-turbo",
            "GPT_3_5_TURBO_16K": "gpt-3.5-turbo-16k",
            "GPT_3_5_TURBO_INSTRUCT": "gpt-3.5-turbo-instruct",
            "DAVINCI_002": "davinci-002",
            "BABBAGE_002": "babbage-002",
        }
        super().__init__("OpenAI", models)


# Anthropic models
class AnthropicModels(ModelProvider):
    def __init__(self):
        models = {
            "CLAUDE_3_OPUS": "claude-3-opus-20240229",
            "CLAUDE_3_SONNET": "claude-3-sonnet-20240229",
            "CLAUDE_3_HAIKU": "claude-3-haiku-20240307",
            "CLAUDE_2_1": "claude-2.1",
            "CLAUDE_2": "claude-2.0",
            "CLAUDE_INSTANT_1_2": "claude-instant-1.2",
        }
        super().__init__("Anthropic", models)


# Google models
class GoogleModels(ModelProvider):
    def __init__(self):
        models = {
            "GEMINI_PRO": "gemini-pro",
            "GEMINI_PRO_VISION": "gemini-pro-vision",
            "GEMINI_1_5_PRO": "gemini-1.5-pro",
            "GEMINI_1_5_FLASH": "gemini-1.5-flash",
            "BARD": "bard",
            "PALM_2": "palm-2",
            "PALM_2_CHAT": "palm-2-chat",
        }
        super().__init__("Google", models)


# Cohere Models
class CohereModels(ModelProvider):
    def __init__(self):
        models = {
            "COMMAND": "command",
            "COMMAND_LIGHT": "command-light",
            "COMMAND_NIGHTLY": "command-nightly",
            "COMMAND_R": "command-r",
            "COMMAND_R_PLUS": "command-r-plus",
        }
        super().__init__("Cohere", models)


# Mistral Models
class MistralModels(ModelProvider):
    def __init__(self):
        models = {
            "MISTRAL_7B": "mistral-7b-instruct",
            "MISTRAL_8X7B": "mistral-8x7b-instruct",
            "MISTRAL_SMALL": "mistral-small-latest",
            "MISTRAL_MEDIUM": "mistral-medium-latest",
            "MISTRAL_LARGE": "mistral-large-latest",
            "MIXTRAL_8X7B": "mixtral-8x7b-instruct",
        }
        super().__init__("Mistral", models)


# Meta Models
class MetaModels(ModelProvider):
    def __init__(self):
        models = {
            "LLAMA_2_7B": "llama-2-7b-chat",
            "LLAMA_2_13B": "llama-2-13b-chat",
            "LLAMA_2_70B": "llama-2-70b-chat",
            "CODE_LLAMA_7B": "code-llama-7b-instruct",
            "CODE_LLAMA_13B": "code-llama-13b-instruct",
            "CODE_LLAMA_34B": "code-llama-34b-instruct",
        }
        super().__init__("Meta", models)


# Popular Hugging Face Models
class HuggingfaceModels(ModelProvider):
    def __init__(self):
        models = {
            "FALCON_7B": "tiiuae/falcon-7b-instruct",
            "FALCON_40B": "tiiuae/falcon-40b-instruct",
            "VICUNA_7B": "lmsys/vicuna-7b-v1.5",
            "VICUNA_13B": "lmsys/vicuna-13b-v1.5",
            "ALPACA_7B": "chavinlo/alpaca-native",
            "FLAN_T5_SMALL": "google/flan-t5-small",
            "FLAN_T5_BASE": "google/flan-t5-base",
            "FLAN_T5_LARGE": "google/flan-t5-large",
            "FLAN_T5_XL": "google/flan-t5-xl",
            "FLAN_T5_XXL": "google/flan-t5-xxl",
        }
        super().__init__("Huggingface", models)

# Popular Moonshot Models
class MoonshotModels(ModelProvider):
    def __init__(self):
        models = {
            "MOONSHOT_V1_8K": "moonshot-v1-8k",
            "MOONSHOT_V1_32K": "moonshot-v1-32k", 
            "MOONSHOT_V1_128K": "moonshot-v1-128k",
            "MOONSHOT_V1_AUTO": "moonshot-v1-auto",
            "MOONSHOT_V1_8K_VISION": "moonshot-v1-8k-vision-preview",
            "MOONSHOT_V1_32K_VISION": "moonshot-v1-32k-vision-preview",
            "MOONSHOT_V1_128K_VISION": "moonshot-v1-128k-vision-preview"
        }
        super().__init__("Moonshot", models)


# Initialize provider instances
OpenAI = OpenAIModels()
Anthropic = AnthropicModels()
Google = GoogleModels()
Cohere = CohereModels()
Mistral = MistralModels()
Meta = MetaModels()
Huggingface = HuggingfaceModels()
Moonshot = MoonshotModels()

# Provider registry
PROVIDERS: Dict[str, ModelProvider] = {
    "openai": OpenAI,
    "anthropic": Anthropic,
    "google": Google,
    "cohere": Cohere,
    "mistral": Mistral,
    "meta": Meta,
    "huggingface": Huggingface,
    "moonshot": Moonshot,
}


def get_all_models() -> Dict[str, List[str]]:
    """Get all models organized by provider."""
    return {
        name: provider.list_models()
        for name, provider in PROVIDERS.items()
    }


def find_model(model_name: str) -> Optional[str]:
    """Find which provider a model belongs to."""
    for provider_name, provider in PROVIDERS.items():
        if model_name in provider:
            return provider_name
    return None
