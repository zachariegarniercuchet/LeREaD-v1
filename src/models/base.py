"""Base assistant class defining the interface for all LLM assistants."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseAssistant(ABC):
    """Abstract base class for all LLM assistants.
    
    All concrete implementations should inherit from this class and implement
    the generate method to handle model-specific logic.
    """
    
    def __init__(self, model_name: str, temperature: float = 0.3, has_system_role: bool = True, **kwargs):
        """Initialize the assistant.
        
        Args:
            model_name: Name or path of the model
            temperature: Sampling temperature (controls randomness)
            has_system_role: Whether the assistant's chat template has a system role
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.temperature = temperature
        self.has_system_role = has_system_role
        self.config = kwargs
    
    @abstractmethod
    def generate(self, message: List[Dict[str, str]]) -> str:
        """Generate a response based on the provided messages.
        
        Args:
            message: List of message dictionaries with "role" and "content" keys.
                     Format: [
                         {"role": "system", "content": "You are a helpful assistant."},
                         {"role": "user", "content": "What's the weather?"},
                         {"role": "assistant", "content": "It's sunny."}
                     ]
        
        Returns:
            str: The generated content (assistant's response only)
        """
        pass
    
    def _extract_user_content(self, message: List[Dict[str, str]]) -> str:
        """Extract the last user message content.
        
        Helper method to get the user's latest message from the conversation.
        """
        for msg in reversed(message):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""
    
    def _extract_system_prompt(self, message: List[Dict[str, str]]) -> str:
        """Extract the system prompt if present.
        
        Helper method to get the system message from the conversation.
        """
        for msg in message:
            if msg.get("role") == "system":
                return msg.get("content", "")
        return ""



def get_message(system_prompt, user_input, fewshot_examples=None, has_system_role=True):

    if has_system_role:
        message = [{"role": "system", "content": system_prompt}]
        if fewshot_examples is not None:
            message.append({"role": "user", "content": f"{fewshot_examples[0][0]}"})
    else:
        message = [{"role": "user", "content": f"{system_prompt} \n\n {fewshot_examples[0][0] if fewshot_examples is not None else user_input}"}]

    
    if fewshot_examples is not None:
        for i, example in enumerate(fewshot_examples):

            if i != 0:
                message.append({"role": "user", "content": f"{example[0]}"})

            message.append({"role": "assistant", "content": f"{example[1]}"})

        # User input 
        message.append({"role": "user", "content": f"{user_input}"})
    
    return message