# LLM Relic

A lightweight Python library that provides easy access to popular LLM model names and allows you to define which models your application supports.

## Why LLM Relic?

- **No more hardcoded model names**: Access standardized model names from major providers
- **Easy support definition**: Fluent interface to define which models your app supports
- **Model metadata**: Context windows, capabilities, pricing tiers, and status for all models
- **Smart model search**: Find models by capabilities, context size, or pricing tier
- **Validation**: Built-in validation to ensure only supported models are used
- **Zero dependencies**: Lightweight library with no external dependencies
- **Type hints**: Full type hint support for better IDE experience

## Installation

```bash
pip install llmrelic
```

## Quick Start

### Access Model Names

```python
from llmrelic import OpenAI, Anthropic, Google

# Access model names directly
print(OpenAI.gpt_4)  # "gpt-4"
print(Anthropic.claude_3_opus)  # "claude-3-opus-20240229"
print(Google.gemini_pro)  # "gemini-pro"

# List all models from a provider
print(OpenAI.list_models())
```

### Define Supported Models

```python
from llmrelic import SupportedModels

# Define which models your app supports
supported = (SupportedModels.create()
             .openai()  # All OpenAI models
             .anthropic(["claude-3-opus-20240229", "claude-3-sonnet-20240229"])  # Specific models
             .google()  # All Google models
             .custom(["my-custom-model"])  # Your custom models
             .build())

# Validate model support
if supported.is_supported("gpt-4"):
    print("GPT-4 is supported!")

# Get all supported models
print(supported.get_supported_models())
```

### Use in Your Application

```python
from llmrelic import OpenAI, SupportedModels

class MyLLMApp:
    def __init__(self):
        # Define what models your app supports
        self.supported_models = (SupportedModels.create()
                                .openai(["gpt-4", "gpt-3.5-turbo"])
                                .anthropic()
                                .build())
    
    def chat(self, model_name: str, message: str):
        if not self.supported_models.is_supported(model_name):
            available = ", ".join(self.supported_models.get_supported_models())
            raise ValueError(f"Model {model_name} not supported. Available: {available}")
        
        # Your chat logic here
        return f"Response from {model_name}"

# Usage
app = MyLLMApp()
app.chat(OpenAI.gpt_4, "Hello!")  # Works
app.chat("gpt-4", "Hello!")  # Works
# app.chat("unsupported-model", "Hello!")  # Raises ValueError
```

## Supported Providers

- **OpenAI**: GPT-4, GPT-3.5-turbo, and more
- **Anthropic**: Claude 3 Opus, Sonnet, Haiku, and more
- **Google**: Gemini Pro, Bard, PaLM-2, and more
- **Cohere**: Command, Command-Light, Command-R, and more
- **Mistral**: Mistral 7B, Mixtral 8x7B, and more
- **Meta**: Llama 2, Code Llama, and more
- **Hugging Face**: Popular open-source models
- **Moonshot**: moonshot-v1-8k, moonshot-v1-32k, moonshot-v1-128k, moonshot-v1-auto, and vision preview variants

## API Reference

### Model Providers

Each provider exposes models as attributes:

```python
from llmrelic import OpenAI, Anthropic, Google, Cohere, Mistral, Meta, Huggingface

# Access models
OpenAI.gpt_4  # "gpt-4"
Anthropic.claude_3_opus  # "claude-3-opus-20240229"
Google.gemini_pro  # "gemini-pro"

# List all models
OpenAI.list_models()

# Check if model exists
"gpt-4" in OpenAI  # True
```

### SupportedModels (Fluent Interface)

```python
from llmrelic import SupportedModels

supported = (SupportedModels.create()
             .openai()  # All OpenAI models
             .openai(["gpt-4", "gpt-3.5-turbo"])  # Specific OpenAI models
             .anthropic()  # All Anthropic models
             .google(["gemini-pro"])  # Specific Google models
             .custom(["my-model"])  # Custom models
             .build())

# Check support
supported.is_supported("gpt-4")  # True

# Get models
supported.get_models()  # List of all supported models
```

### ModelRegistry (Direct Interface)

```python
from llmrelic import ModelRegistry

registry = ModelRegistry()
registry.add_provider("openai")
registry.add_models(["custom-model-1", "custom-model-2"])
registry.add_model("another-model")

# Check support
registry.is_supported("gpt-4")  # True
"gpt-4" in registry  # True

# Get models
registry.get_supported_models()
registry.get_supported_by_provider()

# Iterate
for model in registry:
    print(model)
```

## Utility Functions

```python
from llmrelic import get_all_models, find_model

# Get all available models by provider
all_models = get_all_models()

# Find which provider a model belongs to
provider = find_model("gpt-4")  # "openai"
```

## Model Metadata

Access detailed information about any model:

```python
from llmrelic import get_metadata, OpenAI

# Get metadata for a specific model
metadata = get_metadata("gpt-4-turbo")
print(metadata.context_window)  # 128000
print(metadata.supports_vision)  # True
print(metadata.pricing_tier)  # PricingTier.PREMIUM
print(metadata.status)  # ModelStatus.ACTIVE

# Access via provider
metadata = OpenAI.get_metadata("gpt-4")
all_openai_metadata = OpenAI.list_metadata()

# Check capabilities
if metadata.supports_function_calling:
    print("Function calling supported!")

if metadata.has_min_context(100000):
    print("Large context window available!")

if not metadata.is_deprecated():
    print("Model is still active!")
```

### ModelMetadata Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Model name |
| `provider` | `str` | Provider name |
| `context_window` | `int` | Maximum context size in tokens |
| `max_output_tokens` | `int \| None` | Maximum output tokens |
| `training_cutoff` | `str \| None` | Training data cutoff date |
| `pricing_tier` | `PricingTier` | BUDGET, STANDARD, or PREMIUM |
| `status` | `ModelStatus` | ACTIVE, DEPRECATED, BETA, or PREVIEW |
| `capabilities` | `ModelCapabilities` | Capability flags |

### ModelCapabilities Flags

| Flag | Description |
|------|-------------|
| `supports_vision` | Accepts image inputs |
| `supports_audio` | Accepts audio inputs |
| `supports_video` | Accepts video inputs |
| `supports_function_calling` | Supports function/tool calling |
| `supports_json_mode` | Supports structured JSON output |
| `supports_streaming` | Supports streaming responses |
| `supports_system_message` | Supports system messages |

## Finding Models

Search for models by capabilities and requirements:

```python
from llmrelic import find_models, ModelFinder, PricingTier, ModelStatus

# Simple function-based search
models = find_models(
    min_context=100000,
    supports_vision=True,
    pricing_tier=PricingTier.BUDGET
)

for model in models:
    print(f"{model.name}: {model.context_window} tokens")

# Using ModelFinder for more options
finder = ModelFinder()

# Find all vision-capable models
vision_models = finder.find_with_vision()

# Find budget models with large context
budget_large = finder.find_budget_with_large_context(min_context=32000)

# Find models by provider
openai_models = finder.find_by_provider("openai", "anthropic")

# Find all active models
active_models = finder.find_active()

# Find multimodal models (vision, audio, etc.)
multimodal = finder.find_multimodal()

# Find by pricing tier
cheap_models = finder.find_by_pricing(PricingTier.BUDGET)
```

### Real-World Examples

```python
from llmrelic import find_models, get_metadata, PricingTier, ModelStatus

# Find the cheapest model that supports vision
budget_vision = find_models(
    supports_vision=True,
    pricing_tier=PricingTier.BUDGET,
    status=ModelStatus.ACTIVE
)

# Find models suitable for processing long documents
long_context = find_models(min_context=100000, status=ModelStatus.ACTIVE)

# Validate a model before using it
def validate_model(model_name: str, needs_vision: bool = False):
    metadata = get_metadata(model_name)
    if metadata is None:
        raise ValueError(f"Unknown model: {model_name}")
    if metadata.is_deprecated():
        raise ValueError(f"Model {model_name} is deprecated")
    if needs_vision and not metadata.supports_vision:
        raise ValueError(f"Model {model_name} doesn't support vision")
    return metadata
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Run tests: `pytest`
5. Submit a pull request

## License

MIT License
