# LLM Relic

A lightweight Python library that provides easy access to popular LLM model names and allows you to define which models your application supports.

## Why LLM Relic?

- **No more hardcoded model names**: Access standardized model names from major providers
- **Easy support definition**: Fluent interface to define which models your app supports
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
- **Moonshot**: moonshot-v1-8k (Kimi K1.5), moonshot-v1-32k (Kimi K2), moonshot-vl-32k (Kimi‑VL)

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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Run tests: `pytest`
5. Submit a pull request

## License

MIT License
