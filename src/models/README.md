# LLM Model Management System

A clean, modular factory-based system for managing both OpenAI API and open-weight models with a unified interface.

## Architecture

```
src/models/
├── __init__.py              # Package exports
├── base.py                  # BaseAssistant abstract class
├── openai_assistant.py      # OpenAI API implementation
├── openweight_assistant.py  # Open-weight models (Qwen, Phi, SaulLM, etc.)
├── factory.py               # AssistantFactory
└── examples.py              # Usage examples
```

## Key Features

- **Unified Interface**: All assistants implement the same `generate(messages)` method
- **Standard Message Format**: Uses OpenAI chat format (role/content dicts)
- **Factory Pattern**: Easy model switching via `AssistantFactory`
- **Modular Design**: Add new model types by extending `BaseAssistant`
- **Configuration-Driven**: Load models from config dictionaries
- **Auto-Detection**: Automatically detect model type from name

## Supported Models

### OpenAI
- gpt-4, gpt-4-turbo, gpt-4o
- gpt-3.5-turbo
- Any OpenAI model name

### Open-Weight
- **Qwen**: Qwen2.5-8B, Qwen2.5-32B, etc.
- **Phi**: phi-3.5-mini-instruct, phi-2, etc.
- **SaulLM**: 7B, 54B variants
- **Mistral, Llama**: Any causal language model on Hugging Face

## Quick Start

### 1. Basic Usage with OpenAI

```python
from src.models import AssistantFactory

# Create assistant
assistant = AssistantFactory.create(
    assistant_type="openai",
    model_name="gpt-4",
    temperature=0.3
)

# Prepare messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"}
]

# Generate response
response = assistant.generate(messages)
print(response)  # Only the generated content, no prompt
```

### 2. Using Open-Weight Models

```python
# Create assistant for Qwen2.5-8B
assistant = AssistantFactory.create(
    assistant_type="openweight",
    model_name="Qwen/Qwen2.5-8B",
    temperature=0.5,
    max_tokens=512,
    device="cuda"
)

# Same message format, same generate() method
response = assistant.generate(messages)
```

### 3. Configuration-Based Creation (Recommended)

```python
config = {
    "type": "openweight",
    "model_name": "Qwen/Qwen2.5-8B",
    "temperature": 0.7,
    "max_tokens": 1024,
    "device": "cuda"
}

assistant = AssistantFactory.create_from_config(config)
response = assistant.generate(messages)
```

### 4. Auto-Detection

```python
# Automatically detects model type from name
assistant = AssistantFactory.auto_detect("Qwen/Qwen2.5-8B")
response = assistant.generate(messages)
```

## Message Format

All assistants expect messages in the standard OpenAI format:

```python
messages = [
    {
        "role": "system",
        "content": "You are a helpful legal assistant."
    },
    {
        "role": "user",
        "content": "What is a contract?"
    },
    {
        "role": "assistant",
        "content": "A contract is an agreement..."  # Optional, for context
    }
]
```

The `generate()` method returns **only the generated content** (assistant's response), not the full conversation.

## Configuration Examples

### Qwen Models
```python
{
    "type": "openweight",
    "model_name": "Qwen/Qwen2.5-8B",
    "temperature": 0.5,
    "max_tokens": 512,
    "device": "cuda"
}

{
    "type": "openweight",
    "model_name": "Qwen/Qwen2.5-32B",
    "temperature": 0.3,
    "max_tokens": 1024,
    "device": "cuda",
    "dtype": "torch.float16"  # For efficient memory usage
}
```

### Phi Models
```python
{
    "type": "openweight",
    "model_name": "microsoft/phi-3.5-mini-instruct",
    "temperature": 0.4,
    "max_tokens": 512,
    "device": "cuda"
}
```

### SaulLM Models
```python
{
    "type": "openweight",
    "model_name": "saul-lm/saulm-7b",  # or saulm-54b
    "temperature": 0.6,
    "max_tokens": 1024,
    "device": "cuda"
}
```

### OpenAI
```python
{
    "type": "openai",
    "model_name": "gpt-4",
    "temperature": 0.3
}
```

## Adding New Model Types

To add support for a new model type (e.g., Anthropic Claude, HuggingFace endpoints):

1. Create a new file: `src/models/new_model_assistant.py`
2. Extend `BaseAssistant`:
   ```python
   from src.models.base import BaseAssistant
   
   class NewModelAssistant(BaseAssistant):
       def generate(self, messages: List[Dict[str, str]]) -> str:
           # Your implementation
           pass
   ```
3. Add to factory in `src/models/factory.py`

## Advanced Options

### For Open-Weight Models

```python
{
    "type": "openweight",
    "model_name": "Qwen/Qwen2.5-32B",
    "temperature": 0.5,
    "max_tokens": 2048,
    "device": "cuda",
    "dtype": "torch.float16",           # float32 or float16
    "attn_implementation": "flash_attention_2"  # Faster attention
}
```

### For OpenAI

```python
{
    "type": "openai",
    "model_name": "gpt-4",
    "temperature": 0.3,
    "api_key": "sk-...",  # Optional, uses env var if not provided
    "timeout": 30.0
}
```

## Environment Variables

For OpenAI:
```bash
export OPENAI_API_KEY="sk-..."
```

## Performance Tips

1. **Use `float16` for open-weight models on GPU** to reduce memory usage
2. **Use `flash_attention_2`** if your GPU supports it (significantly faster)
3. **Batch multiple requests** when possible
4. **Use smaller models (8B) for lower latency**, larger models (32B+) for quality
5. **Consider quantization** (e.g., 8-bit) for very large models on limited memory

## Troubleshooting

### Model Loading Issues
- Ensure Hugging Face model exists: `huggingface-hub` must be installed
- Check disk space for model weights
- Use `device="cpu"` if CUDA memory issues occur

### CUDA Out of Memory
- Reduce `max_tokens`
- Use `dtype="torch.float16"`
- Use a smaller model
- Enable quantization

### Chat Template Warnings
- Some models may not have chat templates; the system falls back to simple formatting
- This is usually fine but responses may be less structured

## Files in This Directory

| File | Purpose |
|------|---------|
| `base.py` | Abstract base class defining the assistant interface |
| `openai_assistant.py` | OpenAI API implementation |
| `openweight_assistant.py` | Open-weight model implementation |
| `factory.py` | Factory for creating assistants |
| `__init__.py` | Package exports |
| `examples.py` | Usage examples and workflows |
| `README.md` | This file |

## See Also

- [examples.py](examples.py) - Complete working examples
- [base.py](base.py) - Interface documentation
- [factory.py](factory.py) - Factory usage and model detection
