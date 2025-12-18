"""
Comprehensive tests for model provider classes and utility functions.
"""
import pytest
from llmrelic.models import (
    OpenAI,
    Anthropic,
    Google,
    Cohere,
    Mistral,
    Meta,
    Huggingface,
    Moonshot,
    ModelProvider,
    PROVIDERS,
    get_all_models,
    find_model,
)


class TestModelProvider:
    """Tests for the base ModelProvider class functionality."""

    def test_provider_name_property(self):
        """Provider should return its name when converted to string."""
        assert str(OpenAI) == "OpenAI"
        assert str(Anthropic) == "Anthropic"
        assert str(Google) == "Google"
        assert str(Cohere) == "Cohere"
        assert str(Mistral) == "Mistral"
        assert str(Meta) == "Meta"
        assert str(Huggingface) == "Huggingface"
        assert str(Moonshot) == "Moonshot"

    def test_list_models_returns_list(self):
        """list_models() should return a non-empty list of strings."""
        for provider in PROVIDERS.values():
            models = provider.list_models()
            assert isinstance(models, list)
            assert len(models) > 0
            assert all(isinstance(model, str) for model in models)

    def test_contains_operator(self):
        """The 'in' operator should check if a model exists in provider."""
        assert "gpt-4" in OpenAI
        assert "claude-3-opus-20240229" in Anthropic
        assert "gemini-pro" in Google
        assert "command" in Cohere
        assert "mistral-7b-instruct" in Mistral
        assert "llama-2-7b-chat" in Meta
        assert "tiiuae/falcon-7b-instruct" in Huggingface
        assert "moonshot-v1-8k" in Moonshot

        # Test negative cases
        assert "nonexistent-model" not in OpenAI
        assert "fake-model" not in Anthropic

    def test_get_model_with_valid_key(self):
        """get_model() should return the model name for valid keys."""
        assert OpenAI.get_model("GPT_4") == "gpt-4"
        assert Anthropic.get_model("CLAUDE_3_OPUS") == "claude-3-opus-20240229"

    def test_get_model_with_invalid_key_returns_key(self):
        """get_model() should return the key itself if not found."""
        assert OpenAI.get_model("INVALID_KEY") == "INVALID_KEY"
        assert Google.get_model("NONEXISTENT") == "NONEXISTENT"


class TestOpenAI:
    """Tests specific to OpenAI models."""

    def test_gpt4_models(self):
        """Test GPT-4 model names."""
        assert OpenAI.gpt_4 == "gpt-4"
        assert OpenAI.gpt_4_turbo == "gpt-4-turbo"
        assert OpenAI.gpt_4_turbo_preview == "gpt-4-turbo-preview"
        assert OpenAI.gpt_4_32k == "gpt-4-32k"

    def test_gpt35_models(self):
        """Test GPT-3.5 model names."""
        assert OpenAI.gpt_3_5_turbo == "gpt-3.5-turbo"
        assert OpenAI.gpt_3_5_turbo_16k == "gpt-3.5-turbo-16k"
        assert OpenAI.gpt_3_5_turbo_instruct == "gpt-3.5-turbo-instruct"

    def test_legacy_models(self):
        """Test legacy model names."""
        assert OpenAI.davinci_002 == "davinci-002"
        assert OpenAI.babbage_002 == "babbage-002"

    def test_case_insensitive_access(self):
        """Attributes should be case-insensitive."""
        assert OpenAI.GPT_4 == "gpt-4"
        assert OpenAI.gpt_4 == "gpt-4"

    def test_invalid_attribute_raises_error(self):
        """Accessing invalid model should raise AttributeError."""
        with pytest.raises(AttributeError, match="has no model named"):
            _ = OpenAI.invalid_model_name


class TestAnthropic:
    """Tests specific to Anthropic models."""

    def test_claude3_models(self):
        """Test Claude 3 model names."""
        assert Anthropic.claude_3_opus == "claude-3-opus-20240229"
        assert Anthropic.claude_3_sonnet == "claude-3-sonnet-20240229"
        assert Anthropic.claude_3_haiku == "claude-3-haiku-20240307"

    def test_claude2_models(self):
        """Test Claude 2 model names."""
        assert Anthropic.claude_2_1 == "claude-2.1"
        assert Anthropic.claude_2 == "claude-2.0"

    def test_claude_instant_model(self):
        """Test Claude Instant model."""
        assert Anthropic.claude_instant_1_2 == "claude-instant-1.2"


class TestGoogle:
    """Tests specific to Google models."""

    def test_gemini_models(self):
        """Test Gemini model names."""
        assert Google.gemini_pro == "gemini-pro"
        assert Google.gemini_pro_vision == "gemini-pro-vision"
        assert Google.gemini_1_5_pro == "gemini-1.5-pro"
        assert Google.gemini_1_5_flash == "gemini-1.5-flash"

    def test_legacy_google_models(self):
        """Test legacy Google models."""
        assert Google.bard == "bard"
        assert Google.palm_2 == "palm-2"
        assert Google.palm_2_chat == "palm-2-chat"


class TestCohere:
    """Tests specific to Cohere models."""

    def test_command_models(self):
        """Test Command model variants."""
        assert Cohere.command == "command"
        assert Cohere.command_light == "command-light"
        assert Cohere.command_nightly == "command-nightly"
        assert Cohere.command_r == "command-r"
        assert Cohere.command_r_plus == "command-r-plus"


class TestMistral:
    """Tests specific to Mistral models."""

    def test_mistral_models(self):
        """Test Mistral model names."""
        assert Mistral.mistral_7b == "mistral-7b-instruct"
        assert Mistral.mistral_8x7b == "mistral-8x7b-instruct"
        assert Mistral.mistral_small == "mistral-small-latest"
        assert Mistral.mistral_medium == "mistral-medium-latest"
        assert Mistral.mistral_large == "mistral-large-latest"
        assert Mistral.mixtral_8x7b == "mixtral-8x7b-instruct"


class TestMeta:
    """Tests specific to Meta models."""

    def test_llama2_models(self):
        """Test Llama 2 model names."""
        assert Meta.llama_2_7b == "llama-2-7b-chat"
        assert Meta.llama_2_13b == "llama-2-13b-chat"
        assert Meta.llama_2_70b == "llama-2-70b-chat"

    def test_code_llama_models(self):
        """Test Code Llama model names."""
        assert Meta.code_llama_7b == "code-llama-7b-instruct"
        assert Meta.code_llama_13b == "code-llama-13b-instruct"
        assert Meta.code_llama_34b == "code-llama-34b-instruct"


class TestHuggingface:
    """Tests specific to Hugging Face models."""

    def test_falcon_models(self):
        """Test Falcon model names."""
        assert Huggingface.falcon_7b == "tiiuae/falcon-7b-instruct"
        assert Huggingface.falcon_40b == "tiiuae/falcon-40b-instruct"

    def test_vicuna_models(self):
        """Test Vicuna model names."""
        assert Huggingface.vicuna_7b == "lmsys/vicuna-7b-v1.5"
        assert Huggingface.vicuna_13b == "lmsys/vicuna-13b-v1.5"

    def test_flan_t5_models(self):
        """Test FLAN-T5 model names."""
        assert Huggingface.flan_t5_small == "google/flan-t5-small"
        assert Huggingface.flan_t5_base == "google/flan-t5-base"
        assert Huggingface.flan_t5_large == "google/flan-t5-large"
        assert Huggingface.flan_t5_xl == "google/flan-t5-xl"
        assert Huggingface.flan_t5_xxl == "google/flan-t5-xxl"

    def test_alpaca_model(self):
        """Test Alpaca model name."""
        assert Huggingface.alpaca_7b == "chavinlo/alpaca-native"


class TestMoonshot:
    """Tests specific to Moonshot models."""

    def test_standard_moonshot_models(self):
        """Test standard Moonshot model names."""
        assert Moonshot.moonshot_v1_8k == "moonshot-v1-8k"
        assert Moonshot.moonshot_v1_32k == "moonshot-v1-32k"
        assert Moonshot.moonshot_v1_128k == "moonshot-v1-128k"
        assert Moonshot.moonshot_v1_auto == "moonshot-v1-auto"

    def test_moonshot_vision_models(self):
        """Test Moonshot vision model names."""
        assert Moonshot.moonshot_v1_8k_vision == "moonshot-v1-8k-vision-preview"
        assert Moonshot.moonshot_v1_32k_vision == "moonshot-v1-32k-vision-preview"
        assert Moonshot.moonshot_v1_128k_vision == "moonshot-v1-128k-vision-preview"


class TestProvidersRegistry:
    """Tests for the PROVIDERS global registry."""

    def test_providers_dict_contains_all_providers(self):
        """PROVIDERS should contain all provider instances."""
        expected_providers = {
            "openai",
            "anthropic",
            "google",
            "cohere",
            "mistral",
            "meta",
            "huggingface",
            "moonshot",
        }
        assert set(PROVIDERS.keys()) == expected_providers

    def test_all_providers_are_model_provider_instances(self):
        """All values in PROVIDERS should be ModelProvider instances."""
        for provider in PROVIDERS.values():
            assert isinstance(provider, ModelProvider)

    def test_provider_names_match_keys(self):
        """Provider names should match their registry keys (case-insensitive)."""
        for key, provider in PROVIDERS.items():
            assert provider.name.lower() == key.lower()


class TestGetAllModels:
    """Tests for the get_all_models() utility function."""

    def test_returns_dict(self):
        """get_all_models() should return a dictionary."""
        result = get_all_models()
        assert isinstance(result, dict)

    def test_contains_all_providers(self):
        """Result should contain all registered providers."""
        result = get_all_models()
        expected_providers = {
            "openai",
            "anthropic",
            "google",
            "cohere",
            "mistral",
            "meta",
            "huggingface",
            "moonshot",
        }
        assert set(result.keys()) == expected_providers

    def test_each_provider_has_model_list(self):
        """Each provider should have a list of models."""
        result = get_all_models()
        for provider_name, models in result.items():
            assert isinstance(models, list)
            assert len(models) > 0
            assert all(isinstance(model, str) for model in models)

    def test_contains_expected_models(self):
        """Result should contain known models from each provider."""
        result = get_all_models()
        
        # Flatten all models
        all_models = [model for models in result.values() for model in models]
        
        # Check for sample models from each provider
        assert "gpt-4" in all_models
        assert "claude-3-opus-20240229" in all_models
        assert "gemini-pro" in all_models
        assert "command" in all_models
        assert "mistral-7b-instruct" in all_models
        assert "llama-2-7b-chat" in all_models
        assert "tiiuae/falcon-7b-instruct" in all_models
        assert "moonshot-v1-8k" in all_models


class TestFindModel:
    """Tests for the find_model() utility function."""

    def test_find_openai_models(self):
        """Should correctly identify OpenAI models."""
        assert find_model("gpt-4") == "openai"
        assert find_model("gpt-3.5-turbo") == "openai"
        assert find_model("gpt-4-turbo") == "openai"

    def test_find_anthropic_models(self):
        """Should correctly identify Anthropic models."""
        assert find_model("claude-3-opus-20240229") == "anthropic"
        assert find_model("claude-2.1") == "anthropic"
        assert find_model("claude-instant-1.2") == "anthropic"

    def test_find_google_models(self):
        """Should correctly identify Google models."""
        assert find_model("gemini-pro") == "google"
        assert find_model("gemini-1.5-pro") == "google"
        assert find_model("palm-2") == "google"

    def test_find_cohere_models(self):
        """Should correctly identify Cohere models."""
        assert find_model("command") == "cohere"
        assert find_model("command-r-plus") == "cohere"

    def test_find_mistral_models(self):
        """Should correctly identify Mistral models."""
        assert find_model("mistral-7b-instruct") == "mistral"
        assert find_model("mixtral-8x7b-instruct") == "mistral"

    def test_find_meta_models(self):
        """Should correctly identify Meta models."""
        assert find_model("llama-2-7b-chat") == "meta"
        assert find_model("code-llama-34b-instruct") == "meta"

    def test_find_huggingface_models(self):
        """Should correctly identify Hugging Face models."""
        assert find_model("tiiuae/falcon-7b-instruct") == "huggingface"
        assert find_model("chavinlo/alpaca-native") == "huggingface"
        assert find_model("lmsys/vicuna-13b-v1.5") == "huggingface"

    def test_find_moonshot_models(self):
        """Should correctly identify Moonshot models."""
        assert find_model("moonshot-v1-8k") == "moonshot"
        assert find_model("moonshot-v1-128k-vision-preview") == "moonshot"

    def test_unknown_model_returns_none(self):
        """Should return None for unknown models."""
        assert find_model("unknown-model") is None
        assert find_model("fake-gpt-5") is None
        assert find_model("nonexistent") is None

    def test_empty_string_returns_none(self):
        """Should return None for empty string."""
        assert find_model("") is None

    def test_case_sensitive(self):
        """find_model should be case-sensitive."""
        assert find_model("gpt-4") == "openai"
        # This should fail since the actual model name is lowercase
        assert find_model("GPT-4") is None
