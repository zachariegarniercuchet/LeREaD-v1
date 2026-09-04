# src/models/muse_glimmer_assistant.py
"""Assistant for Meta's Muse Glimmer family (muse_glimmer arch):
dense causal transformer + ViT-G/14 perception encoder, agentic/tool-use tuned.

Kept separate from OpenWeightAssistant/HybridQwenAssistant because this model:
  - is multimodal: needs AutoProcessor (not AutoTokenizer) and
    AutoModelForMultimodalLM (not AutoModelForCausalLM)
  - does NOT wrap reasoning in <think> tags -- it emits channel-scoped
    output (`to=self<|message|>...<|eom|>` for CoT, `to=user<|message|>...<|eot|>`
    for the final answer), so stripping requires channel parsing, not a
    <think> regex
  - controls reasoning effort via a "Reasoning strength: <value>" line in the
    system prompt, not a chat-template boolean kwarg
  - officially recommends temperature=1.0 / top_p=0.95 / top_k=64 and warns
    against greedy decoding (non-reproducible even at temperature=0)
"""
import re
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig
from typing import List, Dict
from src.models.base import BaseAssistant


class MuseGlimmerAssistant(BaseAssistant):

    _USER_CHANNEL_RE = re.compile(r"to=user<\|message\|>(.*?)(?:<\|eot\|>|$)", re.DOTALL)

    def __init__(
        self,
        model_name: str,
        model_path: str,
        temperature: float = 1.0,          # Meta's published default, not 0.3
        top_p: float = 0.95,
        top_k: int = 64,
        max_tokens: int = 1024,            # channel overhead eats budget fast --
                                            # a tight cap can truncate before the
                                            # user channel closes
        quantization: str = "bf16",
        trust_remote_code: bool = True,
        has_system_role: bool = True,
        thinking: str = "high",            # low / medium / high / xhigh
        strip_thinking: bool = True,
        **kwargs,
    ):
        super().__init__(model_name, temperature, has_system_role=has_system_role)
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
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
            self.model = AutoModelForMultimodalLM.from_pretrained(
                self.model_path, quantization_config=bnb_cfg, **common_kwargs
            )
        elif self.quantization == "8bit":
            bnb_cfg = BitsAndBytesConfig(
                load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True
            )
            self.model = AutoModelForMultimodalLM.from_pretrained(
                self.model_path, quantization_config=bnb_cfg, **common_kwargs
            )
        else:  # bf16 -- the released precision, safest choice for the vision tower
            self.model = AutoModelForMultimodalLM.from_pretrained(
                self.model_path, torch_dtype=torch.bfloat16, **common_kwargs
            )

        self.model.config.use_cache = False
        self.model.eval()

    def _inject_reasoning_strength(self, messages: List[Dict]) -> List[Dict]:
        """Muse Glimmer reads effort off a line in the system prompt, not a
        template flag. Append/insert it rather than mutating caller's dicts."""
        if not self.thinking:
            return messages
        line = f"Reasoning strength: {self.thinking}"
        messages = [dict(m) for m in messages]  # shallow copy, don't mutate caller state
        for m in messages:
            if m.get("role") == "system":
                m["content"] = f"{m['content']}\n\n{line}"
                return messages
        return [{"role": "system", "content": line}] + messages

    @classmethod
    def _extract_final_channel(cls, text: str) -> str:
        """Pull the last to=user<|message|>...<|eot|> segment out of the raw
        (special-tokens-included) decode. Falls back to the raw text if no
        channel markers are present, e.g. if the model was run with
        skip_special_tokens stripping them already."""
        matches = cls._USER_CHANNEL_RE.findall(text)
        if matches:
            return matches[-1].strip()
        return text.strip()

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int = None) -> str:
        max_new_tokens = max_new_tokens or self.max_tokens
        messages = self._inject_reasoning_strength(messages)

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                do_sample=True,  # never run greedy -- Meta's own docs note it's
                                  # non-reproducible even at temperature=0
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]

        if self.strip_thinking:
            # Need the channel markers to find the user-facing segment, so
            # decode with special tokens intact and parse them out ourselves.
            raw = self.processor.decode(generated_ids, skip_special_tokens=False)
            return self._extract_final_channel(raw)

        return self.processor.decode(generated_ids, skip_special_tokens=True)