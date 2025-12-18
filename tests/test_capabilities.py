"""
Tests for model capabilities, metadata, and finder functionality.
"""
import pytest
from llmrelic.capabilities import (
    ModelCapabilities,
    ModelMetadata,
    Modality,
    PricingTier,
    ModelStatus,
    ADVANCED_CAPABILITIES,
    VISION_CAPABILITIES,
    BASIC_CAPABILITIES,
    MULTIMODAL_CAPABILITIES,
)
from llmrelic.metadata import (
    MODEL_METADATA,
    get_metadata,
    OPENAI_METADATA,
    ANTHROPIC_METADATA,
    GOOGLE_METADATA,
)
from llmrelic.finder import ModelFinder, ModelQuery, find_models
from llmrelic.models import OpenAI, Anthropic, Google


class TestModality:

    def test_modality_values(self):
        assert Modality.TEXT.value == "text"
        assert Modality.VISION.value == "vision"
        assert Modality.AUDIO.value == "audio"
        assert Modality.VIDEO.value == "video"


class TestPricingTier:

    def test_pricing_tier_values(self):
        assert PricingTier.BUDGET.value == "budget"
        assert PricingTier.STANDARD.value == "standard"
        assert PricingTier.PREMIUM.value == "premium"


class TestModelStatus:

    def test_status_values(self):
        assert ModelStatus.ACTIVE.value == "active"
        assert ModelStatus.DEPRECATED.value == "deprecated"
        assert ModelStatus.BETA.value == "beta"
        assert ModelStatus.PREVIEW.value == "preview"


class TestModelCapabilities:

    def test_default_capabilities(self):
        caps = ModelCapabilities()
        assert caps.supports_vision is False
        assert caps.supports_audio is False
        assert caps.supports_video is False
        assert caps.supports_function_calling is False
        assert caps.supports_json_mode is False
        assert caps.supports_streaming is True
        assert caps.supports_system_message is True

    def test_vision_capabilities(self):
        caps = ModelCapabilities(supports_vision=True)
        assert caps.supports_vision is True
        assert Modality.VISION in caps.modalities

    def test_modalities_property(self):
        caps = ModelCapabilities()
        assert caps.modalities == {Modality.TEXT}

        caps = ModelCapabilities(supports_vision=True, supports_audio=True)
        assert caps.modalities == {Modality.TEXT, Modality.VISION, Modality.AUDIO}

    def test_predefined_capabilities(self):
        assert BASIC_CAPABILITIES.supports_streaming is True
        assert ADVANCED_CAPABILITIES.supports_function_calling is True
        assert VISION_CAPABILITIES.supports_vision is True
        assert MULTIMODAL_CAPABILITIES.supports_audio is True


class TestModelMetadata:

    def test_metadata_creation(self):
        metadata = ModelMetadata(
            name="test-model",
            provider="test",
            context_window=4096,
        )
        assert metadata.name == "test-model"
        assert metadata.provider == "test"
        assert metadata.context_window == 4096
        assert metadata.pricing_tier == PricingTier.STANDARD
        assert metadata.status == ModelStatus.ACTIVE

    def test_metadata_with_capabilities(self):
        caps = ModelCapabilities(supports_vision=True)
        metadata = ModelMetadata(
            name="vision-model",
            provider="test",
            context_window=8192,
            capabilities=caps,
        )
        assert metadata.supports_vision is True
        assert metadata.capabilities.supports_vision is True

    def test_has_min_context(self):
        metadata = ModelMetadata(
            name="test",
            provider="test",
            context_window=32000,
        )
        assert metadata.has_min_context(16000) is True
        assert metadata.has_min_context(32000) is True
        assert metadata.has_min_context(64000) is False

    def test_is_active(self):
        active = ModelMetadata(
            name="active",
            provider="test",
            context_window=4096,
            status=ModelStatus.ACTIVE,
        )
        deprecated = ModelMetadata(
            name="deprecated",
            provider="test",
            context_window=4096,
            status=ModelStatus.DEPRECATED,
        )
        assert active.is_active() is True
        assert deprecated.is_active() is False

    def test_is_deprecated(self):
        deprecated = ModelMetadata(
            name="deprecated",
            provider="test",
            context_window=4096,
            status=ModelStatus.DEPRECATED,
        )
        assert deprecated.is_deprecated() is True


class TestMetadataRegistry:

    def test_model_metadata_contains_all_providers(self):
        providers = {m.provider for m in MODEL_METADATA.values()}
        expected = {
            "openai",
            "anthropic",
            "google",
            "cohere",
            "mistral",
            "meta",
            "huggingface",
            "moonshot"
        }
        assert providers == expected

    def test_get_metadata_existing_model(self):
        metadata = get_metadata("gpt-4")
        assert metadata is not None
        assert metadata.name == "gpt-4"
        assert metadata.provider == "openai"

    def test_get_metadata_nonexistent_model(self):
        metadata = get_metadata("nonexistent-model")
        assert metadata is None

    def test_openai_metadata(self):
        assert "gpt-4" in OPENAI_METADATA
        assert "gpt-3.5-turbo" in OPENAI_METADATA
        gpt4 = OPENAI_METADATA["gpt-4"]
        assert gpt4.context_window == 8192
        assert gpt4.pricing_tier == PricingTier.PREMIUM

    def test_anthropic_metadata(self):
        assert "claude-3-opus-20240229" in ANTHROPIC_METADATA
        opus = ANTHROPIC_METADATA["claude-3-opus-20240229"]
        assert opus.context_window == 200000
        assert opus.supports_vision is True

    def test_google_metadata(self):
        assert "gemini-1.5-pro" in GOOGLE_METADATA
        gemini = GOOGLE_METADATA["gemini-1.5-pro"]
        assert gemini.context_window == 1000000
        assert gemini.capabilities.supports_audio is True


class TestModelQuery:

    def test_empty_query_matches_all(self):
        query = ModelQuery()
        metadata = ModelMetadata(
            name="test",
            provider="test",
            context_window=4096,
        )
        assert query.matches(metadata) is True

    def test_min_context_filter(self):
        query = ModelQuery(min_context=32000)
        small = ModelMetadata(name="small", provider="test", context_window=8192)
        large = ModelMetadata(name="large", provider="test", context_window=128000)
        assert query.matches(small) is False
        assert query.matches(large) is True

    def test_vision_filter(self):
        query = ModelQuery(supports_vision=True)
        vision_caps = ModelCapabilities(supports_vision=True)
        with_vision = ModelMetadata(
            name="vision",
            provider="test",
            context_window=4096,
            capabilities=vision_caps,
        )
        without_vision = ModelMetadata(
            name="text",
            provider="test",
            context_window=4096,
        )
        assert query.matches(with_vision) is True
        assert query.matches(without_vision) is False

    def test_pricing_tier_filter(self):
        query = ModelQuery(pricing_tiers={PricingTier.BUDGET})
        budget = ModelMetadata(
            name="budget",
            provider="test",
            context_window=4096,
            pricing_tier=PricingTier.BUDGET,
        )
        premium = ModelMetadata(
            name="premium",
            provider="test",
            context_window=4096,
            pricing_tier=PricingTier.PREMIUM,
        )
        assert query.matches(budget) is True
        assert query.matches(premium) is False

    def test_provider_filter(self):
        query = ModelQuery(providers={"openai"})
        openai_model = ModelMetadata(name="gpt", provider="openai", context_window=4096)
        anthropic_model = ModelMetadata(name="claude", provider="anthropic", context_window=4096)
        assert query.matches(openai_model) is True
        assert query.matches(anthropic_model) is False

    def test_combined_filters(self):
        query = ModelQuery(
            min_context=32000,
            supports_vision=True,
            pricing_tiers={PricingTier.BUDGET, PricingTier.STANDARD},
        )
        vision_caps = ModelCapabilities(supports_vision=True)
        matching = ModelMetadata(
            name="match",
            provider="test",
            context_window=128000,
            pricing_tier=PricingTier.STANDARD,
            capabilities=vision_caps,
        )
        small_context = ModelMetadata(
            name="small",
            provider="test",
            context_window=8192,
            pricing_tier=PricingTier.STANDARD,
            capabilities=vision_caps,
        )
        assert query.matches(matching) is True
        assert query.matches(small_context) is False


class TestModelFinder:

    def test_find_with_empty_query(self):
        finder = ModelFinder()
        results = finder.find(ModelQuery())
        assert len(results) == len(MODEL_METADATA)

    def test_find_by_context(self):
        finder = ModelFinder()
        results = finder.find_by_context(min_tokens=100000)
        assert all(m.context_window >= 100000 for m in results)
        assert len(results) > 0

    def test_find_with_vision(self):
        finder = ModelFinder()
        results = finder.find_with_vision()
        assert all(m.supports_vision for m in results)
        assert len(results) > 0

    def test_find_with_function_calling(self):
        finder = ModelFinder()
        results = finder.find_with_function_calling()
        assert all(m.supports_function_calling for m in results)

    def test_find_by_pricing(self):
        finder = ModelFinder()
        results = finder.find_by_pricing(PricingTier.BUDGET)
        assert all(m.pricing_tier == PricingTier.BUDGET for m in results)
        assert len(results) > 0

    def test_find_active(self):
        finder = ModelFinder()
        results = finder.find_active()
        assert all(m.status == ModelStatus.ACTIVE for m in results)

    def test_find_by_provider(self):
        finder = ModelFinder()
        results = finder.find_by_provider("openai")
        assert all(m.provider == "openai" for m in results)
        assert len(results) == len(OPENAI_METADATA)

    def test_find_by_multiple_providers(self):
        finder = ModelFinder()
        results = finder.find_by_provider("openai", "anthropic")
        providers = {m.provider for m in results}
        assert providers == {"openai", "anthropic"}

    def test_find_budget_with_large_context(self):
        finder = ModelFinder()
        results = finder.find_budget_with_large_context(min_context=32000)
        for m in results:
            assert m.context_window >= 32000
            assert m.pricing_tier == PricingTier.BUDGET
            assert m.status == ModelStatus.ACTIVE

    def test_find_multimodal(self):
        finder = ModelFinder()
        results = finder.find_multimodal()
        assert all(len(m.capabilities.modalities) > 1 for m in results)
        assert len(results) > 0


class TestFindModelsFunction:

    def test_find_models_basic(self):
        results = find_models()
        assert len(results) == len(MODEL_METADATA)

    def test_find_models_with_context(self):
        results = find_models(min_context=100000)
        assert all(m.context_window >= 100000 for m in results)

    def test_find_models_with_vision(self):
        results = find_models(supports_vision=True)
        assert all(m.supports_vision for m in results)

    def test_find_models_by_provider(self):
        results = find_models(provider="anthropic")
        assert all(m.provider == "anthropic" for m in results)

    def test_find_models_combined(self):
        results = find_models(
            min_context=32000,
            supports_function_calling=True,
            pricing_tier=PricingTier.BUDGET,
        )
        for m in results:
            assert m.context_window >= 32000
            assert m.supports_function_calling
            assert m.pricing_tier == PricingTier.BUDGET


class TestModelProviderMetadataIntegration:

    def test_openai_get_metadata(self):
        metadata = OpenAI.get_metadata("gpt-4")
        assert metadata is not None
        assert metadata.name == "gpt-4"
        assert metadata.provider == "openai"

    def test_anthropic_get_metadata(self):
        metadata = Anthropic.get_metadata("claude-3-opus-20240229")
        assert metadata is not None
        assert metadata.supports_vision is True

    def test_google_get_metadata(self):
        metadata = Google.get_metadata("gemini-1.5-pro")
        assert metadata is not None
        assert metadata.context_window == 1000000

    def test_get_metadata_invalid_model(self):
        metadata = OpenAI.get_metadata("invalid-model")
        assert metadata is None

    def test_list_metadata(self):
        metadata_list = OpenAI.list_metadata()
        assert len(metadata_list) == len(OpenAI.list_models())
        assert all(isinstance(m, ModelMetadata) for m in metadata_list)

    def test_list_metadata_all_openai(self):
        metadata_list = OpenAI.list_metadata()
        assert all(m.provider == "openai" for m in metadata_list)


class TestRealWorldScenarios:

    def test_find_cheapest_model_with_vision(self):
        results = find_models(
            supports_vision=True,
            pricing_tier=PricingTier.BUDGET,
            status=ModelStatus.ACTIVE,
        )
        assert len(results) > 0
        for m in results:
            assert m.supports_vision
            assert m.pricing_tier == PricingTier.BUDGET

    def test_find_model_for_long_document(self):
        results = find_models(min_context=100000, status=ModelStatus.ACTIVE)
        model_names = {m.name for m in results}
        assert "claude-3-opus-20240229" in model_names
        assert "gemini-1.5-pro" in model_names

    def test_find_openai_models_with_json_mode(self):
        results = find_models(
            provider="openai",
            supports_json_mode=True,
        )
        assert all(m.provider == "openai" for m in results)
        assert all(m.supports_json_mode for m in results)

    def test_validate_model_before_use(self):
        metadata = get_metadata("gpt-4-turbo")
        assert metadata is not None
        assert metadata.supports_vision
        assert metadata.supports_function_calling
        assert not metadata.is_deprecated()
