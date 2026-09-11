# src/models/saul_assistant.py
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from src.models.base import BaseAssistant


# SaulLM-54B/141B (Equall) are continued-pretrained from Mixtral and do NOT
# ship a chat_template in tokenizer_config.json. As of transformers>=4.44,
# apply_chat_template() no longer falls back to a generic default, so calling
# it on these checkpoints either raises or silently mis-renders the prompt.
# Equall (Pierre Colombo) confirmed on the model's HF discussion page that
# the correct template is simply the base Mixtral-Instruct one:
# https://huggingface.co/Equall/SaulLM-54B-Instruct/discussions/1
# Note it only supports "user" and "assistant" roles (no "system"), which is
# why has_system_role should stay False for this family.
MIXTRAL_INSTRUCT_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] + eos_token }}"
    "{% else %}"
    "{{ raise_exception('Only user and assistant roles are supported!') }}"
    "{% endif %}"
    "{% endfor %}"
)


class SaulAssistant(BaseAssistant):
    """Assistant wrapper for the SaulLM family (Equall), e.g.
    SaulLM-54B-Instruct / SaulLM-141B-Instruct.

    Distinct from OpenWeightAssistant because:
      1. It force-injects the Mixtral-Instruct chat template, since the
         released checkpoints don't carry one and transformers>=4.44 will
         no longer default one for you.
      2. It defaults device_map="auto" instead of a hardcoded "cuda", since
         these models are large enough (54B/141B) that fp16/8bit runs
         typically need to be sharded across multiple GPUs.
      3. use_cache stays True by default (KV caching). OpenWeightAssistant
         disables it; for a model this size that makes generation far
         slower and, combined with any upstream time limits, can look like
         "the model doesn't work" when it's actually just timing out.
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
        quantization: str = "fp16",       # "fp16", "8bit", "4bit"
        trust_remote_code: bool = False,
        has_system_role: bool = False,    # Mixtral template: user/assistant only
        thinking: bool = False,           # SaulLM has no thinking mode
        strip_thinking: bool = False,
        device_map: str = "auto",
        use_cache: bool = True,
        **kwargs,
    ):
        super().__init__(model_name, temperature, has_system_role=has_system_role)
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.quantization = quantization
        self.trust_remote_code = trust_remote_code
        self.thinking = thinking
        self.strip_thinking = strip_thinking
        self.device_map = device_map
        self.use_cache = use_cache
        self._load_model()

    def _load_model(self):
        print(f"Loading {self.model_name} from {self.model_path} [{self.quantization}]")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=self.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if not getattr(self.tokenizer, "chat_template", None):
            print(
                f"[WARN] {self.model_name}: no chat_template found on tokenizer, "
                "injecting Mixtral-Instruct template."
            )
            self.tokenizer.chat_template = MIXTRAL_INSTRUCT_TEMPLATE

        common_kwargs = dict(
            trust_remote_code=self.trust_remote_code,
            device_map=self.device_map,
            low_cpu_mem_usage=True,
        )

        if self.quantization == "4bit":
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, quantization_config=bnb_cfg, **common_kwargs
            )
        elif self.quantization == "8bit":
            # MatMul8bitLt only supports fp16 compute; if we let the model load
            # in its default bfloat16 (Mixtral/SaulLM's config dtype), bnb has to
            # cast every activation from bf16 -> fp16 on every forward pass and
            # warns about it. Load in fp16 directly so there's nothing to cast.
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_cfg,
                torch_dtype=torch.float16,
                **common_kwargs,
            )
        else:  # fp16 / bf16
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=torch.bfloat16, **common_kwargs
            )

        self.model.config.use_cache = self.use_cache
        self.model.eval()

    @staticmethod
    def _strip_think_block(text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def generate(
        self,
        messages,  # Full conversation history: list of {"role", "content"} dicts
        max_new_tokens: int = 512,
    ) -> str:
        """
        Generate a response from a SaulLM model.

        Note: SaulLM's chat template only supports 'user' and 'assistant'
        roles. If you have upstream system-prompt logic, make sure it's
        merged into the first user turn before calling this (has_system_role
        is False for this reason).
        """
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Debug aid: uncomment when validating a new checkpoint/template
        # print(repr(text))

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