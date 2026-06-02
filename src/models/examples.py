"""Usage examples for the model factory system."""

from src.models import AssistantFactory

# Example messages in the standard format
example_messages = [
    {"role": "system", "content": "You are a helpful legal assistant."},
    {"role": "user", "content": "What are the key aspects of contract law?"},
]

# ============================================================================
# Example 1: Using OpenAI (API-based)
# ============================================================================
print("=" * 70)
print("Example 1: OpenAI Assistant")
print("=" * 70)

openai_assistant = AssistantFactory.create(
    assistant_type="openai",
    model_name="gpt-4",
    temperature=0.3
)

response = openai_assistant.generate(example_messages)
print(f"Response: {response}\n")


# ============================================================================
# Example 2: Using Open-Weight Models (Local)
# ============================================================================
print("=" * 70)
print("Example 2: Open-Weight Assistant (Qwen)")
print("=" * 70)

# For Qwen2.5-8B (requires model to be downloaded first)
qwen_assistant = AssistantFactory.create(
    assistant_type="openweight",
    model_name="Qwen/Qwen2.5-8B",
    temperature=0.5,
    max_tokens=512,
    device="cuda"  # or "cpu"
)

response = qwen_assistant.generate(example_messages)
print(f"Response: {response}\n")


# ============================================================================
# Example 3: Using Phi Model
# ============================================================================
print("=" * 70)
print("Example 3: Open-Weight Assistant (Phi)")
print("=" * 70)

phi_assistant = AssistantFactory.create(
    assistant_type="openweight",
    model_name="microsoft/phi-3.5-mini-instruct",
    temperature=0.3,
    max_tokens=512,
    device="cuda"
)

response = phi_assistant.generate(example_messages)
print(f"Response: {response}\n")


# ============================================================================
# Example 4: Using Config Dictionary (Recommended for Production)
# ============================================================================
print("=" * 70)
print("Example 4: Creating from Config Dictionary")
print("=" * 70)

configs = {
    "qwen_8b": {
        "type": "openweight",
        "model_name": "Qwen/Qwen2.5-8B",
        "temperature": 0.5,
        "max_tokens": 512,
        "device": "cuda"
    },
    "qwen_32b": {
        "type": "openweight",
        "model_name": "Qwen/Qwen2.5-32B",
        "temperature": 0.3,
        "max_tokens": 1024,
        "device": "cuda"
    },
    "phi": {
        "type": "openweight",
        "model_name": "microsoft/phi-3.5-mini-instruct",
        "temperature": 0.4,
        "max_tokens": 512,
        "device": "cuda"
    },
    "openai_gpt4": {
        "type": "openai",
        "model_name": "gpt-4",
        "temperature": 0.3,
    }
}

# Create assistants from config
for name, config in configs.items():
    try:
        assistant = AssistantFactory.create_from_config(config)
        print(f"✓ Created {name}")
    except Exception as e:
        print(f"✗ Failed to create {name}: {e}")

print()


# ============================================================================
# Example 5: Auto-Detection
# ============================================================================
print("=" * 70)
print("Example 5: Auto-Detection of Model Type")
print("=" * 70)

# Auto-detect will choose based on model name
models_to_test = [
    "gpt-4",  # Will be detected as OpenAI
    "Qwen/Qwen2.5-8B",  # Will be detected as open-weight
    "microsoft/phi-3.5-mini-instruct",  # Will be detected as open-weight
]

for model in models_to_test:
    try:
        assistant = AssistantFactory.auto_detect(model, temperature=0.5)
        print(f"✓ {model} -> {type(assistant).__name__}")
    except Exception as e:
        print(f"✗ {model} failed: {e}")

print()


# ============================================================================
# Example 6: Complete Workflow
# ============================================================================
print("=" * 70)
print("Example 6: Complete Workflow")
print("=" * 70)

# Create config for the model you want to use
model_config = {
    "type": "openweight",
    "model_name": "Qwen/Qwen2.5-8B",
    "temperature": 0.7,
    "max_tokens": 256,
    "device": "cuda"
}

# Create assistant
assistant = AssistantFactory.create_from_config(model_config)

# Prepare messages
messages = [
    {
        "role": "system",
        "content": "You are an expert in Canadian law. Provide clear, concise answers."
    },
    {
        "role": "user",
        "content": "Explain the concept of mens rea in criminal law."
    }
]

# Generate response
try:
    response = assistant.generate(messages)
    print(f"Generated response:\n{response}")
except Exception as e:
    print(f"Error during generation: {e}")

print()
