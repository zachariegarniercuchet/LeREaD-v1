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
            device_map="auto",
        )

        if self.quantization == "4bit":
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, quantization_config=bnb_cfg, **common_kwargs
            )
        elif self.quantization == "8bit":
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, quantization_config=bnb_cfg, **common_kwargs
            )
        else:  # fp16
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=torch.float16, **common_kwargs
            )

        self.model.config.use_cache = False
        self.model.eval()

    def _format_prompt(self, messages: List[Dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            try:
                # enable_thinking only for models that support it
                extra = {"enable_thinking": self.thinking} if self.thinking else {}
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **extra
                )
            except Exception as e:
                print(f"Warning: chat_template failed ({e}), falling back to manual format")
        # Fallback for models without a chat template
        parts = [f"{m['role']}: {m['content']}" for m in messages]
        parts.append("assistant:")
        return "\n".join(parts)

    @staticmethod
    def _strip_think_block(text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def generate(self, message: List[Dict[str, str]]) -> str:
        prompt = self._format_prompt(message)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.model.device)

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

        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if self.thinking and self.strip_thinking:
            text = self._strip_think_block(text)
        return text