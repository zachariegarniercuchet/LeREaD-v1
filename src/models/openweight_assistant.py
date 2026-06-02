"""Open-weight model assistant implementation using Hugging Face transformers."""
from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.models.base import BaseAssistant


class OpenWeightAssistant(BaseAssistant):
    """Assistant using open-weight models from Hugging Face.
    
    Supports models like Qwen, Phi, SaulLM, and other causal language models.
    """
    
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        """Initialize the open-weight assistant.
        
        Args:
            model_name: Hugging Face model name or local path
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            device: Device to run on ("cuda" or "cpu")
            **kwargs: Additional parameters (dtype, attn_implementation, etc.)
        """
        super().__init__(model_name, temperature, **kwargs)
        self.max_tokens = max_tokens
        self.device = device
        
        # Model-specific settings
        self.dtype = kwargs.get("dtype", torch.float16 if device == "cuda" else torch.float32)
        self.attn_implementation = kwargs.get("attn_implementation", "flash_attention_2")
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load the tokenizer and model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with device_map for efficient loading
            model_kwargs = {
                "torch_dtype": self.dtype,
                "device_map": "auto" if self.device == "cuda" else None,
            }
            
            # Try to use flash_attention_2 if available (faster)
            if self.device == "cuda":
                try:
                    model_kwargs["attn_implementation"] = self.attn_implementation
                except Exception:
                    # Fall back if flash_attention_2 not available
                    pass
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {str(e)}")
    
    def _format_messages_for_generation(self, message: List[Dict[str, str]]) -> str:
        """Format messages into a prompt string.
        
        This uses the model's chat template if available, otherwise creates a simple format.
        """
        # Try to use chat template if available
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            try:
                prompt = self.tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return prompt
            except Exception as e:
                print(f"Warning: Failed to apply chat template: {e}")
        
        # Fallback: manually format messages
        prompt_parts = []
        for msg in message:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        
        # Add a prompt for the assistant to respond
        prompt_parts.append("assistant:")
        return "\n".join(prompt_parts)
    
    def generate(self, message: List[Dict[str, str]]) -> str:
        """Generate response using the open-weight model.
        
        Args:
            message: List of message dictionaries
        
        Returns:
            str: Generated content (assistant's response only)
        """
        # Format messages into prompt
        prompt = self._format_messages_for_generation(message)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode and extract only the new tokens (not the input prompt)
        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text.strip()
