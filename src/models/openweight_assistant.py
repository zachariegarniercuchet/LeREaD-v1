# src/models/openweight_assistant.py
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Optional
from src.models.base import BaseAssistant


class OpenWeightAssistant(BaseAssistant):

    def __init__(
        self,
        model_name: str,
        model_path: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
        quantization: str = "fp16",       # "fp16", "8bit", "4bit"
        trust_remote_code: bool = False,
        has_system_role: bool = True,
        thinking: bool = False,
        strip_thinking: bool = True,
        **kwargs,
    ):
        super().__init__(model_name, temperature, has_system_role=has_system_role)
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.quantization = quantization
        self.trust_remote_code = trust_remote_code
        self.thinking = thinking
        self.strip_thinking = strip_thinking
        self._load_model()

    def _load_model(self):
        print(f"Loading {self.model_name} from {self.model_path} [{self.quantization}]")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=self.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        common_kwargs = dict(
            trust_remote_code=self.trust_remote_code,
            device_map="cuda",
            low_cpu_mem_usage=True,
        )

        if self.quantization == "4bit":
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, quantization_config=bnb_cfg, **common_kwargs
            )
        elif self.quantization == "8bit":
            bnb_cfg = BitsAndBytesConfig(
                load_in_8bit=True, 
                llm_int8_enable_fp32_cpu_offload=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, quantization_config=bnb_cfg,**common_kwargs
            )
        else:  # fp16
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=torch.float16, **common_kwargs
            )

        self.model.config.use_cache = False
        self.model.eval()

    @staticmethod
    def _strip_think_block(text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def generate(
        self,
        messages,  # Full conversation history
        max_new_tokens: int = 512,
    ) -> str:
        """
        Generate a response for Qwen2.5-7B-Instruct.
        
        Args:
            messages: List of dicts with "role" and "content" keys.
                    Example: [
                        {"role": "system", "content": "You are..."},
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi!"},
                        {"role": "user", "content": "Another question"}
                    ]
            max_new_tokens: Maximum tokens to generate.
        """
        extra = {}
        if self.tokenizer.chat_template and "enable_thinking" in self.tokenizer.chat_template:
            extra["enable_thinking"] = self.thinking

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **extra,
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                repetition_penalty=1.05,
            )

        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if self.strip_thinking:
            response = self._strip_think_block(response)

        return response