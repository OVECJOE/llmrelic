import pytest
from .models import (
    OpenAI, Anthropic, Google, Cohere, Mistral, Meta, Huggingface, Moonshot,
    PROVIDERS, get_all_models, find_model
)
from .registry import ModelRegistry, SupportedModels

def test_provider_str():
    assert str(OpenAI) == "OpenAI"
    assert str(Anthropic) == "Anthropic"

def test_list_models():
    models = OpenAI.list_models()
    assert "gpt-4" in models
    assert isinstance(models, list)
    assert len(models) > 0

def test_getattr():
    assert OpenAI.GPT_4 == "gpt-4"
    assert Anthropic.CLAUDE_3_OPUS.startswith("claude-3-opus")

def invalid_model_error_raises():
    with pytest.raises(AttributeError):
        OpenAI.INVALID_MODEL

def test_model_with_valid_key():
    assert Mistral.MISTRAL_7B == "mistral-7b-instruct"
    assert Meta.LLAMA_2_7B == "llama-2-7b-chat"

def test_model_with_invalid_key():
    assert Google.get_model("INVALID_KEY") == "INVALID_KEY"

def test_operators():
    assert "moonshot-v1-8k" in Moonshot
    assert "command" in Cohere
    assert "some_random_model" not in Huggingface

def test_get_all_models():
    all_models = get_all_models()
    assert isinstance(all_models, dict)
    assert len(all_models) > 0
    flat_models = [model for models in all_models.values() for model in models]
    assert "claude-2.1" in flat_models
    assert "lmsys/vicuna-13b-v1.5" in flat_models

def test_find_model():
    assert find_model("chavinlo/alpaca-native") == "huggingface"
    assert find_model("moonshot-v1-128k-vision-preview") == "moonshot"
    assert find_model("some_random_model") is None

def test_provider_registry():
    expected = {"openai", "anthropic", "google", "cohere", "mistral", "meta", "huggingface", "moonshot"}
    assert set(PROVIDERS.keys()) == expected


def test_add_and_check_model():
    registry = ModelRegistry()
    registry.add_model("gpt-4")
    assert "gpt-4" in registry
    assert registry.is_supported("gpt-4") is True

def test_add_multiple_models():
    registry = ModelRegistry()
    registry.add_models(["gpt-4", "gpt-3.5-turbo"])
    assert "gpt-4" in registry
    assert "gpt-3.5-turbo" in registry

def test_provider_adds_all_models():
    registry = ModelRegistry()
    registry.add_provider("openai")
    for model in OpenAI.list_models():
        assert model in registry

def test_add_providers():
    registry = ModelRegistry()
    registry.add_providers(["openai", "anthropic"])
    for provider in ["openai", "anthropic"]:
        for model in PROVIDERS[provider].list_models():
            assert model in registry

def test_remove_model():
    registry = ModelRegistry()
    registry.add_model("gpt-4")
    registry.remove_model("gpt-4")
    assert "gpt-4" not in registry

def test_get_models_sorted():
    registry = ModelRegistry()
    registry.add_models(["z-model", "a-model"])
    models = registry.get_supported_models()
    assert models == ["a-model", "z-model"]

# Returns only supported models by provider
def test_get_supported_by_provider():
    registry = ModelRegistry()
    registry.add_model("gpt-4")  # belongs to openai
    result = registry.get_supported_by_provider()
    assert "openai" in result
    assert "gpt-4" in result["openai"]
    assert all("gpt-4" not in v for k, v in result.items() if k != "openai")

def test_clear_registry():
    registry = ModelRegistry()
    registry.add_model("gpt-4")
    registry.clear()
    assert registry.get_supported_models() == []

def test_iter_returns_models():
    registry = ModelRegistry()
    registry.add_models(["a", "b"])
    models = list(iter(registry))
    assert models == ["a", "b"] or models == ["b", "a"]  # order depends on sorting

def test_create_supported_models():
    sm = SupportedModels.create()
    assert isinstance(sm, SupportedModels)

def test_openai_builder_adds_models():
    sm = SupportedModels.create().openai()
    models = sm.get_models()
    assert "gpt-4" in models

def test_openai_builder_with_subset():
    sm = SupportedModels.create().openai(["gpt-4"])
    models = sm.get_models()
    assert "gpt-4" in models
    # should not add invalid names
    sm2 = SupportedModels.create().openai(["invalid-model"])
    assert sm2.get_models() == []

def test_multiple_providers_builder():
    sm = SupportedModels.create().openai(["gpt-4"]).anthropic(["claude-2.1"])
    models = sm.get_models()
    assert "gpt-4" in models
    assert "claude-2.1" in models

def test_custom_models():
    sm = SupportedModels.create().custom(["custom-model"]).build()
    assert "custom-model" in sm

def test_build_returns_registry():
    sm = SupportedModels.create().openai()
    registry = sm.build()
    assert isinstance(registry, ModelRegistry)
    assert "gpt-4" in registry