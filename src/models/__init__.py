"""LLM Model Management - Factory pattern for unified assistant interface."""

from src.models.base import BaseAssistant, get_messages
from src.models.openai_assistant import OpenAIAssistant
from src.models.openweight_assistant import OpenWeightAssistant
from src.models.muse_glimmer_assistant import MuseGlimmerAssistant
from src.models.qwen35_assistant import HybridQwenAssistant
from src.models.factory import AssistantFactory

__all__ = [
    "BaseAssistant",
    "OpenAIAssistant",
    "OpenWeightAssistant",
    "MuseGlimmerAssistant",
    "HybridQwenAssistant",
    "AssistantFactory",
]
