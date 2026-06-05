# src/models/registry.py
import os

MODELS_DIR = "/home/zagar/scratch"  # your scratch dir

MODEL_REGISTRY = {
    "Qwen2.5-7B-Instruct": {
        "path": os.path.join(MODELS_DIR, "Qwen2.5-7B-Instruct"),
        "type": "openweight",
        "has_system_role": True,
        "trust_remote_code": True,
        "thinking": False,
        "default_quantization": "fp16",
    },
    "Qwen2.5-32B-Instruct": {
        "path": os.path.join(MODELS_DIR, "Qwen2.5-32B-Instruct"),
        "type": "openweight",
        "has_system_role": True,
        "trust_remote_code": True,
        "thinking": False,
        "default_quantization": "4bit",  # 32B needs quantization
    },
    "Qwen3.5-9B": {
        "path": os.path.join(MODELS_DIR, "Qwen3.5-9B"),
        "type": "openweight",
        "has_system_role": True,
        "trust_remote_code": True,
        "thinking": True,
        "default_quantization": "fp16",
    },
    "SaulLM-7B-Instruct": {
        "path": os.path.join(MODELS_DIR, "SaulLM-7B-Instruct"),
        "type": "openweight",
        "has_system_role": True,
        "trust_remote_code": False,
        "thinking": False,
        "default_quantization": "fp16",
    },
    "SaulLM-54B-Instruct": {
        "path": os.path.join(MODELS_DIR, "SaulLM-54B-Instruct"),
        "type": "openweight",
        "has_system_role": True,
        "trust_remote_code": False,
        "thinking": False,
        "default_quantization": "4bit",
    },
    "phi-4": {
        "path": os.path.join(MODELS_DIR, "phi-4"),
        "type": "openweight",
        "has_system_role": True,
        "trust_remote_code": True,
        "thinking": False,
        "default_quantization": "fp16",
    },
    "Mistral-7B-Instruct-v0.1": {
        "path": os.path.join(MODELS_DIR, "Mistral-7B-Instruct-v0.1"),
        "type": "openweight",
        "has_system_role": False,  # older Mistral has no system role
        "trust_remote_code": False,
        "thinking": False,
        "default_quantization": "fp16",
    },
}