# src/models/gemma4_assistant.py
"""Assistant for Google's Gemma 4 family (gemma4 arch, e.g. 31B/26B/12B):
dense/MoE transformer, native multimodal (text/image/video, +audio on E2B/E4B/12B).

Kept separate from MuseGlimmerAssistant and HybridQwenAssistant because Gemma 4:
  - is multimodal like Muse Glimmer (needs AutoProcessor, not AutoTokenizer)
  - BUT controls thinking via an `enable_thinking` bool kwarg on
    apply_chat_template, like HybridQwen -- NOT a system-prompt string
  - does NOT use <think> tags OR Muse Glimmer's to=user<|message|> channel
    syntax -- it uses its own <|channel>thought ... <|channel> / <|think|>
    markers, and ships a processor.parse_response() helper that already
    knows how to split them -- prefer that over hand-rolled regex
  - the loader class name has moved between transformers releases
    (AutoModelForImageTextToText vs AutoModelForMultimodalLM depending on
    version) -- import defensively
"""
import re
import torch
from transformers import AutoProcessor, BitsAndBytesConfig
try:
    from transformers import AutoModelForImageTextToText as AutoModelForGemma4
except ImportError:
    from transformers import AutoModelForMultimodalLM as AutoModelForGemma4
from typing import List, Dict
from src.models.base import BaseAssistant


class Gemma4Assistant(BaseAssistant):

    # Fallback only -- prefer processor.parse_response() when available.
    _THOUGHT_RE = re.compile(r"<\|channel>thought\n(.*?)<\|channel>", re.DOTALL)

    def __init__(
        self,
        model_name: str,
        model_path: str,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        quantization: str = "bf16",
        trust_remote_code: bool = True,
        has_system_role: bool = True,
        thinking: bool = False,           # bool, like HybridQwen -- not a string
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
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=self.trust_remote_code
        )

        common_kwargs = dict(
            trust_remote_code=self.trust_remote_code,
            device_map="cuda",
            low_cpu_mem_usage=True,
        )

        if self.quantization == "4bit":
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True,
            )
            self.model = AutoModelForGemma4.from_pretrained(
                self.model_path, quantization_config=bnb_cfg, **common_kwargs
            )
        elif self.quantization == "8bit":
            bnb_cfg = BitsAndBytesConfig(
                load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True
            )
            self.model = AutoModelForGemma4.from_pretrained(
                self.model_path, quantization_config=bnb_cfg, **common_kwargs
            )
        else:  # bf16, native release precision
            self.model = AutoModelForGemma4.from_pretrained(
                self.model_path, torch_dtype=torch.bfloat16, **common_kwargs
            )

        self.model.config.use_cache = False
        self.model.eval()

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int = None) -> str:
        max_new_tokens = max_new_tokens or self.max_tokens

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=self.thinking,   # kwarg, not a system-prompt injection
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        # skip_special_tokens=False: parse_response / channel regex both need
        # the markers intact.
        raw = self.processor.decode(generated_ids, skip_special_tokens=False)

        if self.strip_thinking:
            if hasattr(self.processor, "parse_response"):
                parsed = self.processor.parse_response(raw)
                return parsed.get("content", parsed) if isinstance(parsed, dict) else parsed
            return self._THOUGHT_RE.sub("", raw).strip()

        return raw