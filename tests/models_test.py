import pytest
# from ...llmrelic.llmrelic.models import (
#     OpenAI, Anthropic, Google, Cohere, Mistral, Meta, Huggingface,
#     PROVIDERS, get_all_models, find_model
# )
from ..llmrelic import models
Anthropic = models.Anthropic
OpenAI = models.OpenAI
Google = models.Google
Cohere = models.Cohere
Mistral = models.Mistral
Meta = models.Meta
Huggingface = models.Huggingface
Moonshot = models.Moonshot
PROVIDERS = models.PROVIDERS
get_all_models = models.get_all_models
find_model = models.find_model 

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
    # assert "moonshot-v1-8k" in Moonshot
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