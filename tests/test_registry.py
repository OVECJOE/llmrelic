"""
Comprehensive tests for ModelRegistry and SupportedModels classes.
"""
from llmrelic.models import OpenAI, Anthropic
from llmrelic.registry import ModelRegistry, SupportedModels


class TestModelRegistry:
    """Tests for the ModelRegistry class."""

    def test_init_creates_empty_registry(self):
        """New registry should be empty."""
        registry = ModelRegistry()
        assert registry.get_supported_models() == []

    def test_add_model(self):
        """add_model() should add a single model."""
        registry = ModelRegistry()
        result = registry.add_model("gpt-4")
        
        assert "gpt-4" in registry
        assert registry.is_supported("gpt-4")
        assert result is registry  # Check method chaining

    def test_add_model_multiple_times_is_idempotent(self):
        """Adding the same model multiple times should not duplicate it."""
        registry = ModelRegistry()
        registry.add_model("gpt-4")
        registry.add_model("gpt-4")
        
        models = registry.get_supported_models()
        assert models.count("gpt-4") == 1

    def test_add_models(self):
        """add_models() should add multiple models."""
        registry = ModelRegistry()
        models_to_add = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus-20240229"]
        result = registry.add_models(models_to_add)
        
        for model in models_to_add:
            assert model in registry
        assert result is registry  # Check method chaining

    def test_add_models_with_empty_list(self):
        """add_models() with empty list should not fail."""
        registry = ModelRegistry()
        registry.add_models([])
        assert registry.get_supported_models() == []

    def test_add_provider(self):
        """add_provider() should add all models from a provider."""
        registry = ModelRegistry()
        result = registry.add_provider("openai")
        
        openai_models = OpenAI.list_models()
        for model in openai_models:
            assert model in registry
        assert result is registry  # Check method chaining

    def test_add_provider_case_insensitive(self):
        """add_provider() should be case-insensitive."""
        registry1 = ModelRegistry()
        registry2 = ModelRegistry()
        
        registry1.add_provider("openai")
        registry2.add_provider("OpenAI")
        
        assert registry1.get_supported_models() == registry2.get_supported_models()

    def test_add_provider_with_invalid_name(self):
        """add_provider() with invalid name should not crash."""
        registry = ModelRegistry()
        registry.add_provider("invalid_provider")
        assert registry.get_supported_models() == []

    def test_add_providers(self):
        """add_providers() should add models from multiple providers."""
        registry = ModelRegistry()
        result = registry.add_providers(["openai", "anthropic"])
        
        for model in OpenAI.list_models():
            assert model in registry
        for model in Anthropic.list_models():
            assert model in registry
        assert result is registry  # Check method chaining

    def test_remove_model(self):
        """remove_model() should remove a model."""
        registry = ModelRegistry()
        registry.add_model("gpt-4")
        result = registry.remove_model("gpt-4")
        
        assert "gpt-4" not in registry
        assert not registry.is_supported("gpt-4")
        assert result is registry  # Check method chaining

    def test_remove_nonexistent_model(self):
        """remove_model() on non-existent model should not fail."""
        registry = ModelRegistry()
        registry.remove_model("nonexistent-model")
        assert registry.get_supported_models() == []

    def test_is_supported(self):
        """is_supported() should correctly check model support."""
        registry = ModelRegistry()
        
        assert not registry.is_supported("gpt-4")
        
        registry.add_model("gpt-4")
        assert registry.is_supported("gpt-4")
        
        registry.remove_model("gpt-4")
        assert not registry.is_supported("gpt-4")

    def test_contains_operator(self):
        """The 'in' operator should work with registry."""
        registry = ModelRegistry()
        registry.add_model("gpt-4")
        
        assert "gpt-4" in registry
        assert "nonexistent" not in registry

    def test_get_supported_models_returns_sorted_list(self):
        """get_supported_models() should return alphabetically sorted list."""
        registry = ModelRegistry()
        registry.add_models(["zebra-model", "alpha-model", "beta-model"])
        
        models = registry.get_supported_models()
        assert models == ["alpha-model", "beta-model", "zebra-model"]

    def test_get_supported_by_provider(self):
        """get_supported_by_provider() should organize models by provider."""
        registry = ModelRegistry()
        registry.add_model("gpt-4")
        registry.add_model("claude-3-opus-20240229")
        
        result = registry.get_supported_by_provider()
        
        assert "openai" in result
        assert "gpt-4" in result["openai"]
        assert "anthropic" in result
        assert "claude-3-opus-20240229" in result["anthropic"]

    def test_get_supported_by_provider_excludes_empty_providers(self):
        """get_supported_by_provider() should not include providers with no supported models."""
        registry = ModelRegistry()
        registry.add_model("gpt-4")
        
        result = registry.get_supported_by_provider()
        
        assert "openai" in result
        # Other providers shouldn't be in the result
        assert "cohere" not in result
        assert "mistral" not in result

    def test_clear(self):
        """clear() should remove all models."""
        registry = ModelRegistry()
        registry.add_models(["model1", "model2", "model3"])
        result = registry.clear()
        
        assert registry.get_supported_models() == []
        assert result is registry  # Check method chaining

    def test_iter(self):
        """Registry should be iterable."""
        registry = ModelRegistry()
        models = ["alpha", "beta", "gamma"]
        registry.add_models(models)
        
        iterated_models = list(registry)
        assert sorted(iterated_models) == sorted(models)

    def test_method_chaining(self):
        """All methods should support chaining."""
        registry = (ModelRegistry()
                   .add_model("model1")
                   .add_models(["model2", "model3"])
                   .add_provider("openai")
                   .remove_model("model1"))
        
        assert "model2" in registry
        assert "model3" in registry
        assert "gpt-4" in registry
        assert "model1" not in registry


class TestSupportedModels:
    """Tests for the SupportedModels fluent interface."""

    def test_create_returns_instance(self):
        """create() should return a SupportedModels instance."""
        sm = SupportedModels.create()
        assert isinstance(sm, SupportedModels)

    def test_openai_adds_all_models(self):
        """openai() without arguments should add all OpenAI models."""
        sm = SupportedModels.create().openai()
        
        for model in OpenAI.list_models():
            assert sm.is_supported(model)

    def test_openai_adds_specific_models(self):
        """openai() with list should add only specified models."""
        sm = SupportedModels.create().openai(["gpt-4", "gpt-3.5-turbo"])
        
        assert sm.is_supported("gpt-4")
        assert sm.is_supported("gpt-3.5-turbo")
        # Other OpenAI models should not be added
        assert not sm.is_supported("davinci-002")

    def test_openai_filters_invalid_models(self):
        """openai() should filter out non-OpenAI models."""
        sm = SupportedModels.create().openai(["gpt-4", "invalid-model", "fake-gpt"])
        
        models = sm.get_models()
        assert "gpt-4" in models
        assert "invalid-model" not in models
        assert "fake-gpt" not in models

    def test_anthropic_adds_all_models(self):
        """anthropic() without arguments should add all Anthropic models."""
        sm = SupportedModels.create().anthropic()
        
        for model in Anthropic.list_models():
            assert sm.is_supported(model)

    def test_anthropic_adds_specific_models(self):
        """anthropic() with list should add only specified models."""
        sm = SupportedModels.create().anthropic(["claude-3-opus-20240229"])
        
        assert sm.is_supported("claude-3-opus-20240229")
        assert not sm.is_supported("claude-2.1")

    def test_google_adds_models(self):
        """google() should add Google models."""
        sm = SupportedModels.create().google()
        
        assert sm.is_supported("gemini-pro")
        assert sm.is_supported("gemini-1.5-pro")

    def test_cohere_adds_models(self):
        """cohere() should add Cohere models."""
        sm = SupportedModels.create().cohere()
        
        assert sm.is_supported("command")
        assert sm.is_supported("command-r")

    def test_mistral_adds_models(self):
        """mistral() should add Mistral models."""
        sm = SupportedModels.create().mistral()
        
        assert sm.is_supported("mistral-7b-instruct")
        assert sm.is_supported("mixtral-8x7b-instruct")

    def test_meta_adds_models(self):
        """meta() should add Meta models."""
        sm = SupportedModels.create().meta()
        
        assert sm.is_supported("llama-2-7b-chat")
        assert sm.is_supported("code-llama-7b-instruct")

    def test_huggingface_adds_models(self):
        """huggingface() should add Hugging Face models."""
        sm = SupportedModels.create().huggingface()
        
        assert sm.is_supported("tiiuae/falcon-7b-instruct")
        assert sm.is_supported("lmsys/vicuna-13b-v1.5")

    def test_moonshot_adds_models(self):
        """moonshot() should add Moonshot models."""
        sm = SupportedModels.create().moonshot()
        
        assert sm.is_supported("moonshot-v1-8k")
        assert sm.is_supported("moonshot-v1-128k")

    def test_custom_adds_arbitrary_models(self):
        """custom() should add any model names provided."""
        sm = SupportedModels.create().custom(["my-custom-model", "another-model"])
        
        assert sm.is_supported("my-custom-model")
        assert sm.is_supported("another-model")

    def test_build_returns_model_registry(self):
        """build() should return a ModelRegistry instance."""
        sm = SupportedModels.create().openai()
        registry = sm.build()
        
        assert isinstance(registry, ModelRegistry)
        assert "gpt-4" in registry

    def test_is_supported_without_build(self):
        """is_supported() should work before calling build()."""
        sm = SupportedModels.create().openai(["gpt-4"])
        
        assert sm.is_supported("gpt-4")
        assert not sm.is_supported("claude-3-opus-20240229")

    def test_get_models_returns_list(self):
        """get_models() should return list of supported models."""
        sm = SupportedModels.create().openai(["gpt-4"]).anthropic(["claude-2.1"])
        models = sm.get_models()
        
        assert isinstance(models, list)
        assert "gpt-4" in models
        assert "claude-2.1" in models

    def test_contains_operator(self):
        """The 'in' operator should work with SupportedModels."""
        sm = SupportedModels.create().openai(["gpt-4"])
        
        assert "gpt-4" in sm
        assert "nonexistent" not in sm

    def test_method_chaining(self):
        """All methods should support fluent chaining."""
        sm = (SupportedModels.create()
              .openai(["gpt-4"])
              .anthropic(["claude-3-opus-20240229"])
              .google()
              .custom(["my-model"]))
        
        assert sm.is_supported("gpt-4")
        assert sm.is_supported("claude-3-opus-20240229")
        assert sm.is_supported("gemini-pro")
        assert sm.is_supported("my-model")

    def test_chaining_ending_with_build(self):
        """Should be able to chain and end with build()."""
        registry = (SupportedModels.create()
                   .openai(["gpt-4"])
                   .anthropic(["claude-2.1"])
                   .build())
        
        assert isinstance(registry, ModelRegistry)
        assert "gpt-4" in registry
        assert "claude-2.1" in registry


class TestIntegrationScenarios:
    """Integration tests for real-world usage scenarios."""

    def test_restrict_to_specific_models_from_multiple_providers(self):
        """Scenario: Only allow specific models from different providers."""
        registry = (SupportedModels.create()
                   .openai(["gpt-4", "gpt-3.5-turbo"])
                   .anthropic(["claude-3-opus-20240229"])
                   .build())
        
        # Allowed models
        assert "gpt-4" in registry
        assert "gpt-3.5-turbo" in registry
        assert "claude-3-opus-20240229" in registry
        
        # Disallowed models from same providers
        assert "gpt-4-turbo" not in registry
        assert "claude-2.1" not in registry

    def test_allow_all_from_one_provider_specific_from_another(self):
        """Scenario: All models from one provider, specific from another."""
        registry = (SupportedModels.create()
                   .openai()  # All OpenAI models
                   .anthropic(["claude-3-opus-20240229"])  # Only one Anthropic model
                   .build())
        
        # All OpenAI models should be present
        for model in OpenAI.list_models():
            assert model in registry
        
        # Only specified Anthropic model
        assert "claude-3-opus-20240229" in registry
        assert "claude-2.1" not in registry

    def test_combine_provider_models_and_custom_models(self):
        """Scenario: Mix provider models with custom models."""
        registry = (SupportedModels.create()
                   .openai(["gpt-4"])
                   .custom(["my-internal-model", "another-custom-model"])
                   .build())
        
        assert "gpt-4" in registry
        assert "my-internal-model" in registry
        assert "another-custom-model" in registry

    def test_validate_user_input_against_registry(self):
        """Scenario: Validate user's model choice."""
        supported = (SupportedModels.create()
                    .openai(["gpt-4", "gpt-3.5-turbo"])
                    .build())
        
        # Simulate user input
        valid_input = "gpt-4"
        invalid_input = "gpt-5"
        
        assert supported.is_supported(valid_input)
        assert not supported.is_supported(invalid_input)

    def test_get_available_models_by_provider(self):
        """Scenario: Show user which models are available per provider."""
        registry = (ModelRegistry()
                   .add_provider("openai")
                   .add_provider("anthropic"))
        
        by_provider = registry.get_supported_by_provider()
        
        assert "openai" in by_provider
        assert "anthropic" in by_provider
        assert len(by_provider["openai"]) == len(OpenAI.list_models())
        assert len(by_provider["anthropic"]) == len(Anthropic.list_models())

    def test_dynamically_add_and_remove_models(self):
        """Scenario: Runtime modification of supported models."""
        registry = ModelRegistry()
        
        # Start with some models
        registry.add_provider("openai")
        initial_count = len(registry.get_supported_models())
        
        # Add custom model
        registry.add_model("custom-model")
        assert len(registry.get_supported_models()) == initial_count + 1
        
        # Remove a model
        registry.remove_model("gpt-4")
        assert "gpt-4" not in registry
        
        # Clear all
        registry.clear()
        assert len(registry.get_supported_models()) == 0
