"""OpenAI API-based assistant implementation."""
import os
from typing import List, Dict
from openai import OpenAI
from src.models.base import BaseAssistant


class OpenAIAssistant(BaseAssistant):
    """Assistant using OpenAI's API (GPT models)."""
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.3, **kwargs):
        """Initialize the OpenAI assistant.
        
        Args:
            model_name: OpenAI model name (e.g., "gpt-4", "gpt-3.5-turbo")
            temperature: Sampling temperature
            **kwargs: Additional parameters (api_key, timeout, etc.)
        """
        super().__init__(model_name, temperature, **kwargs)
        self.api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=self.api_key)
    
    def generate(self, message: List[Dict[str, str]]) -> str:
        """Generate response using OpenAI API.
        
        Args:
            message: List of message dictionaries
        
        Returns:
            str: Generated content
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message,
            temperature=self.temperature
        )
        return response.choices[0].message.content
